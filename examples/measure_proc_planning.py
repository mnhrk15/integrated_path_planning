#!/usr/bin/env python3
"""Per-step planning-time measurement (A-4 issue 5(ii) / C-3).

Runs the same scenario/seed under a grid of {footprint} x {collision-check
mode} and reports per-step proc_planning statistics, answering whether the
5x check points (multi-circle cover) and the 20-sample distribution check
stay within the planner real-time budget (dt = 0.1 s).

Timing is wall-clock per step (time.perf_counter around planner.plan inside
IntegratedSimulator), so run this on an otherwise idle machine, sequentially
(never in parallel with other campaigns).

Usage:
    python examples/measure_proc_planning.py [--scenario scenarios/scenario_02.yaml]
        [--method sgan] [--seed 0] [--outdir output/exp_proc_planning]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator
from examples.run_statistical_benchmark import set_seed, resolve_model_path

CONDITIONS = [
    # (label, ego_footprint, n_circles, distribution_aware)
    ("circle_single", "circle", None, False),
    ("mc5_single", "multi_circle", 5, False),
    ("circle_robust20", "circle", None, True),
    ("mc5_robust20", "multi_circle", 5, True),
]


def run_one(scenario, method, seed, footprint, n_circles, distribution_aware):
    set_seed(seed)
    config = load_config(scenario)
    config.prediction_method = method
    config.visualization_enabled = False
    config.ego_footprint = footprint
    if n_circles is not None:
        config.ego_footprint_n_circles = n_circles
    config.distribution_aware_planning = distribution_aware
    config.chance_epsilon = 0.0
    resolve_model_path(config, method)

    sim = IntegratedSimulator(config)
    history = sim.run()
    plan_ms = np.array([r.processing_times["planning"] for r in history]) * 1e3
    pred_ms = np.array([r.processing_times["prediction"] for r in history]) * 1e3
    return {
        "steps": len(history),
        "plan_ms_mean": float(plan_ms.mean()),
        "plan_ms_p50": float(np.percentile(plan_ms, 50)),
        "plan_ms_p95": float(np.percentile(plan_ms, 95)),
        "plan_ms_max": float(plan_ms.max()),
        "plan_share_over_dt": float((plan_ms > 100.0).mean()),
        "pred_ms_mean": float(pred_ms.mean()),
        "pred_ms_max": float(pred_ms.max()),
    }


def main():
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/scenario_02.yaml")
    ap.add_argument("--method", default="sgan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="output/exp_proc_planning")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, footprint, n_circles, da in CONDITIONS:
        print(f"== {label} ==", flush=True)
        r = run_one(args.scenario, args.method, args.seed, footprint, n_circles, da)
        results[label] = r
        print(json.dumps(r, indent=1), flush=True)

    base = results["circle_single"]["plan_ms_mean"]
    lines = [
        "# proc_planning measurement (A-4 issue 5(ii) / C-3)",
        "",
        f"- scenario={args.scenario} method={args.method} seed={args.seed}, "
        "sequential runs on an idle machine; planner real-time budget dt = 100 ms",
        "",
        "| condition | steps | plan mean [ms] | p50 | p95 | max | share >100ms | xbase | pred mean [ms] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, *_ in CONDITIONS:
        r = results[label]
        lines.append(
            f"| {label} | {r['steps']} | {r['plan_ms_mean']:.1f} | "
            f"{r['plan_ms_p50']:.1f} | {r['plan_ms_p95']:.1f} | {r['plan_ms_max']:.1f} | "
            f"{r['plan_share_over_dt']:.1%} | {r['plan_ms_mean'] / base:.2f}x | "
            f"{r['pred_ms_mean']:.1f} |")
    report = "\n".join(lines) + "\n"
    (outdir / "REPORT.md").write_text(report)
    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\n" + report)


if __name__ == "__main__":
    main()
