#!/usr/bin/env python3
"""Benchmark script to compare prediction methods."""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator
from src.core.metrics import calculate_aggregate_metrics
from examples.run_statistical_benchmark import resolve_model_path, set_seed

def run_benchmark(scenario_path: str, steps: int = None, output_path: str = "benchmark_results",
                  seed: int = 0):
    """Run benchmark for all prediction methods."""

    methods = ['cv', 'lstm', 'sgan']
    results_list = []

    scenario_name = Path(scenario_path).stem
    output_dir = Path(output_path) / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting benchmark on scenario: {scenario_path}")
    logger.info(f"Results will be saved to: {output_dir} (seed={seed})")

    for method in methods:
        logger.info(f"Testing method: {method.upper()}")

        # Re-seed per method: the methods run sequentially in one process, and
        # without this the RNG state (and thus the pedestrian ground truth)
        # would differ between methods, breaking the comparison.
        set_seed(seed)

        # Reload config for fresh start
        config = load_config(scenario_path)
        config.prediction_method = method
        config.output_path = str(output_dir / method)
        config.visualization_enabled = False  # Disable viz for speed

        # Init simulator
        try:
            # Inside the try: a missing model directory becomes an Error row
            # for this method instead of aborting the whole benchmark and
            # discarding the already-completed methods.
            resolve_model_path(config, method)
            simulator = IntegratedSimulator(config)
            
            # Run simulation
            history = simulator.run(n_steps=steps)
            
            # Calculate metrics
            metrics = calculate_aggregate_metrics(
                history,
                config.dt,
                prediction_dt=simulator.observer.sgan_dt,
                prediction_steps=config.pred_len,
            )
            
            # Add additional efficiency metrics
            total_time = history[-1].time
            avg_speed = np.mean([r.ego_state.v for r in history])
            
            row = {
                "Method": method.upper(),
                "Termination": simulator.termination_reason,
                "Goal Reached": simulator.goal_reached,
                "Min Dist (m)": metrics['min_dist'],
                "Collisions": metrics['collision_count'],
                "Min TTC (s)": metrics['min_ttc'],
                "Max Jerk": metrics['max_jerk'],
                "Standard ADE (m)": metrics['ade'],
                "Standard FDE (m)": metrics['fde'],
                "Planning ADE (m)": metrics['planning_ade'],
                "Planning FDE (m)": metrics['planning_fde'],
                "Total Time (s)": total_time,
                "Avg Speed (m/s)": avg_speed,
                "Steps": len(history)
            }
            results_list.append(row)
            
            logger.success(f"Finished {method.upper()}")
            
        except Exception as e:
            logger.error(f"Failed method {method}: {e}")
            results_list.append({
                "Method": method.upper(),
                "Error": str(e)
            })

    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Drop Error column if empty (fillna to handle mixed success/fail)
    if "Error" in df.columns and df["Error"].isna().all():
        df = df.drop(columns=["Error"])
    
    # Formatting
    float_cols = [c for c in df.columns if c not in ["Method", "Collisions", "Steps", "Error"]]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)

    # Print table
    print("\n" + "="*80)
    print(f"BENCHMARK RESULTS: {Path(scenario_path).name}")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
    print("="*80 + "\n")
    
    # Save markdown
    report_path = output_dir / "benchmark_report.md"
    with open(report_path, 'w') as f:
        f.write(f"# Benchmark Results: {Path(scenario_path).name}\n\n")
        f.write(tabulate(df, headers='keys', tablefmt='github', showindex=False))
        f.write("\n\n## Analysis Notes\n")
        f.write("- **CV**: Constant Velocity (Baseline)\n")
        f.write("- **LSTM**: SGAN with pooling disabled (Interaction-unaware proxy)\n")
        f.write("- **SGAN**: Full Social-GAN (Interaction-aware)\n")
    
    logger.info(f"Report saved to {report_path}")

def main():
    # Configure logging to reduce noise
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Suppress verbose debug logs from libraries using standard logging (e.g., Numba)
    import logging
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description='Run prediction benchmark')
    parser.add_argument('--scenario', type=str, default='scenarios/scenario_01.yaml',
                      help='Path to scenario config')
    parser.add_argument('--steps', type=int, default=None,
                      help='Number of steps')
    parser.add_argument('--output', type=str, default='output/benchmark',
                      help='Output directory')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed applied before each method')

    args = parser.parse_args()

    run_benchmark(args.scenario, args.steps, args.output, seed=args.seed)

if __name__ == "__main__":
    main()
