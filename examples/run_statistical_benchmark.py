#!/usr/bin/env python3
"""Statistical benchmark: run simulations multiple times and report mean +/- std."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.core.metrics import calculate_aggregate_metrics
from src.simulation.integrated_simulator import IntegratedSimulator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_path(config, method: str):
    """Switch model directory based on prediction method."""
    if not config.sgan_model_path or method == "cv":
        return
    original_path = Path(config.sgan_model_path)
    model_name = original_path.name
    new_dir = "models/sgan-models" if method == "lstm" else "models/sgan-p-models"
    new_path = Path(new_dir) / model_name
    if new_path.exists():
        config.sgan_model_path = str(new_path)
    else:
        logger.warning(f"Model {new_path} not found, using {original_path}")


def run_single(scenario_path: str, method: str, seed: int,
               v0_randomization: bool = False) -> dict | None:
    """Run one simulation and return metrics dict."""
    set_seed(seed)
    config = load_config(scenario_path)
    config.prediction_method = method
    config.visualization_enabled = False
    if v0_randomization:
        config.sfm_v0_randomization = True
    resolve_model_path(config, method)

    try:
        simulator = IntegratedSimulator(config)
        history = simulator.run()
        metrics = calculate_aggregate_metrics(
            history,
            config.dt,
            prediction_dt=simulator.observer.sgan_dt,
            prediction_steps=config.pred_len,
        )
        total_time = history[-1].time
        avg_speed = float(np.mean([r.ego_state.v for r in history]))
        return {
            "method": method.upper(),
            "seed": seed,
            "time_s": round(total_time, 2),
            "speed_ms": round(avg_speed, 3),
            "min_dist_m": round(metrics["min_dist"], 4),
            "min_ttc_s": round(metrics["min_ttc"], 4),
            "collision_count": metrics["collision_count"],
            "ade": round(metrics["ade"], 4),
            "fde": round(metrics["fde"], 4),
            "mean_accel": round(metrics["mean_accel"], 4),
            "rms_jerk": round(metrics["rms_jerk"], 4),
            "planning_ade": round(metrics["planning_ade"], 4),
            "planning_fde": round(metrics["planning_fde"], 4),
            "nll": round(metrics["nll"], 4),
        }
    except Exception as e:
        logger.error(f"{method} seed={seed} failed: {e}")
        return None


def generate_latex_table(summary: pd.DataFrame) -> str:
    """Generate LaTeX table source from summary stats.

    ADE is the SGAN-style best-of-N metric (single forecast for CV); P-ADE
    scores the single trajectory actually consumed by the planner (rolling,
    planner resolution) and neutralizes the best-of-N pathology (A-1). NLL is
    the KDE negative log-likelihood of the ground truth under the prediction
    samples (undefined for the single-forecast CV model); the column appears
    only when the runs carry it (older cached CSVs do not).
    """
    has_nll = "nll_mean" in summary.columns and summary["nll_mean"].notna().any()
    nll_caption = (
        " NLL: KDE negative log-likelihood of the ground truth under the"
        " prediction samples (n/a for the single-forecast CV model)."
        if has_nll else ""
    )
    nll_header = " & NLL (nats)" if has_nll else ""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Benchmark results (mean $\pm$ std over 20 runs for LSTM/SGAN; CV is deterministic). Bold values indicate the best mean in each column. ADE: best-of-$N$ displacement error; P-ADE: error of the single predicted trajectory consumed by the planner." + nll_caption + r"}",
        r"  \label{tab:benchmark}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{lccccc" + ("cc" if has_nll else "c") + r"}",
        r"    \hline",
        r"    Method & Time (s) & Speed (m/s) & Min Dist (m) & Min TTC (s) & ADE (m) & P-ADE (m)" + nll_header + r" \\",
        r"    \hline",
    ]

    # Determine best (bold) values per column
    # Lower is better for time and the error metrics; higher is better for
    # speed and the safety margins.
    means = {}
    for _, row in summary.iterrows():
        means[row["method"]] = {
            "time_s": row["time_s_mean"],
            "speed_ms": row["speed_ms_mean"],
            "min_dist_m": row["min_dist_m_mean"],
            "min_ttc_s": row["min_ttc_s_mean"],
            "ade": row["ade_mean"],
            "planning_ade": row["planning_ade_mean"],
        }

    best_time = min(means.values(), key=lambda x: x["time_s"])["time_s"]
    best_speed = max(means.values(), key=lambda x: x["speed_ms"])["speed_ms"]
    best_dist = max(means.values(), key=lambda x: x["min_dist_m"])["min_dist_m"]
    best_ttc = max(means.values(), key=lambda x: x["min_ttc_s"])["min_ttc_s"]
    best_ade = min(means.values(), key=lambda x: x["ade"])["ade"]
    best_pade = min(means.values(), key=lambda x: x["planning_ade"])["planning_ade"]
    best_nll = float("nan")
    if has_nll:
        nll_means = summary["nll_mean"].dropna()
        if not nll_means.empty:
            best_nll = float(nll_means.min())

    for _, row in summary.iterrows():
        method = row["method"]
        is_deterministic = row["time_s_std"] == 0 or pd.isna(row["time_s_std"])

        def fmt(mean_val, std_val, best_val, prec=2):
            bold = abs(mean_val - best_val) < 1e-9
            if is_deterministic:
                s = f"{mean_val:.{prec}f}"
            else:
                s = f"{mean_val:.{prec}f}$\\pm${std_val:.{prec}f}"
            return f"\\textbf{{{s}}}" if bold else s

        time_str = fmt(row["time_s_mean"], row.get("time_s_std", 0), best_time, 1)
        speed_str = fmt(row["speed_ms_mean"], row.get("speed_ms_std", 0), best_speed, 2)
        dist_str = fmt(row["min_dist_m_mean"], row.get("min_dist_m_std", 0), best_dist, 2)
        ttc_str = fmt(row["min_ttc_s_mean"], row.get("min_ttc_s_std", 0), best_ttc, 2)
        ade_str = fmt(row["ade_mean"], row.get("ade_std", 0), best_ade, 2)
        pade_str = fmt(row["planning_ade_mean"], row.get("planning_ade_std", 0), best_pade, 2)

        nll_cell = ""
        if has_nll:
            if pd.isna(row.get("nll_mean")):
                nll_cell = " & --"
            else:
                nll_cell = " & " + fmt(row["nll_mean"], row.get("nll_std", 0), best_nll, 2)

        lines.append(f"    {method} & {time_str} & {speed_str} & {dist_str} & {ttc_str} & {ade_str} & {pade_str}{nll_cell} \\\\")

    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    import logging
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Statistical benchmark")
    parser.add_argument("--scenario", type=str, default="scenarios/scenario_01.yaml")
    parser.add_argument("--n-runs", type=int, default=20, help="Runs per stochastic method")
    parser.add_argument("--output", type=str, default="output/statistical_benchmark")
    parser.add_argument("--table-only", action="store_true",
                        help="Rebuild summary_stats.csv and latex_table.txt from "
                             "the existing all_runs.csv without simulating")
    parser.add_argument("--v0-randomization", action="store_true",
                        help="Per-agent desired-speed randomization "
                             "(sfm_v0_randomization=true; use a separate --output, "
                             "cached all_runs.csv does not key on this)")
    args = parser.parse_args()

    if args.v0_randomization and args.output == "output/statistical_benchmark":
        parser.error("--v0-randomization changes the ground truth; "
                     "use a non-default --output")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.table_only:
        csv_path = output_dir / "all_runs.csv"
        if not csv_path.exists():
            sys.exit(f"--table-only requires {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        all_rows = []

        # CV: deterministic — 1 run (n_runs when the ground truth is randomized)
        cv_runs = args.n_runs if args.v0_randomization else 1
        logger.info(f"Running CV ({cv_runs} run(s))")
        for i in range(cv_runs):
            row = run_single(args.scenario, "cv", seed=i,
                             v0_randomization=args.v0_randomization)
            if row:
                all_rows.append(row)

        # LSTM & SGAN: stochastic — n_runs each
        for method in ["lstm", "sgan"]:
            logger.info(f"Running {method.upper()} ({args.n_runs} runs)")
            for i in range(args.n_runs):
                logger.info(f"  {method.upper()} run {i+1}/{args.n_runs} (seed={i})")
                row = run_single(args.scenario, method, seed=i,
                                 v0_randomization=args.v0_randomization)
                if row:
                    all_rows.append(row)

        # Save raw data
        df = pd.DataFrame(all_rows)
        df.to_csv(output_dir / "all_runs.csv", index=False)
        logger.info(f"Raw data saved to {output_dir / 'all_runs.csv'}")

    # Compute summary stats
    metrics_cols = [
        "time_s",
        "speed_ms",
        "min_dist_m",
        "min_ttc_s",
        "collision_count",
        "ade",
        "fde",
        "mean_accel",
        "rms_jerk",
        "planning_ade",
        "planning_fde",
    ]
    if "nll" in df.columns:
        metrics_cols.append("nll")
    summary_rows = []
    for method in ["CV", "LSTM", "SGAN"]:
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue
        row_summary = {"method": method, "n_runs": len(method_df)}
        for col in metrics_cols:
            row_summary[f"{col}_mean"] = method_df[col].mean()
            row_summary[f"{col}_std"] = method_df[col].std(ddof=1) if len(method_df) > 1 else 0.0
            row_summary[f"{col}_min"] = method_df[col].min()
            row_summary[f"{col}_max"] = method_df[col].max()
        summary_rows.append(row_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_stats.csv", index=False)
    logger.info(f"Summary saved to {output_dir / 'summary_stats.csv'}")

    # Print summary table
    print("\n" + "=" * 90)
    print("STATISTICAL BENCHMARK RESULTS")
    print("=" * 90)
    for _, row in summary_df.iterrows():
        n = int(row["n_runs"])
        print(f"\n{row['method']} (n={n}):")
        for col in metrics_cols:
            mean = row[f"{col}_mean"]
            std = row[f"{col}_std"]
            mn = row[f"{col}_min"]
            mx = row[f"{col}_max"]
            print(f"  {col:20s}: {mean:.4f} +/- {std:.4f}  [min={mn:.4f}, max={mx:.4f}]")
    print("=" * 90)

    # Generate LaTeX
    latex = generate_latex_table(summary_df)
    latex_path = output_dir / "latex_table.txt"
    latex_path.write_text(latex)
    logger.info(f"LaTeX table saved to {latex_path}")
    print(f"\nLaTeX table:\n{latex}")


if __name__ == "__main__":
    main()
