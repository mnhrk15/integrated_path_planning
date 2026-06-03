#!/usr/bin/env python3
"""PoC: distribution-aware (chance-constrained) planning on Scenario 2 (SGAN).

Compares single-sample planning (baseline) against chance-constrained planning
over the full 20-sample SGAN distribution for several epsilon values, so we can
test whether consuming the distribution converts SGAN's interaction-aware
samples into a larger planner safety margin.

Usage:
    python examples/run_da_poc.py [--seeds N] [--scenario PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.core.metrics import calculate_aggregate_metrics
from src.simulation.integrated_simulator import IntegratedSimulator
from examples.run_statistical_benchmark import set_seed, resolve_model_path

METHOD = "sgan"

CONDITIONS = [
    # (label, distribution_aware, epsilon)
    ("baseline_single", False, 0.0),
    ("da_eps0.0", True, 0.0),
    ("da_eps0.1", True, 0.1),
]


def run_one(scenario, distribution_aware, epsilon, seed):
    set_seed(seed)
    config = load_config(scenario)
    config.prediction_method = METHOD
    config.visualization_enabled = False
    config.distribution_aware_planning = distribution_aware
    config.chance_epsilon = epsilon
    resolve_model_path(config, METHOD)

    sim = IntegratedSimulator(config)
    history = sim.run()
    m = calculate_aggregate_metrics(
        history, config.dt,
        prediction_dt=sim.observer.sgan_dt,
        prediction_steps=config.pred_len,
    )
    return {
        "time_s": float(history[-1].time),
        "speed_ms": float(np.mean([r.ego_state.v for r in history])),
        "min_dist_m": float(m["min_dist"]),
        "min_ttc_s": float(m["min_ttc"]),
        "collision_count": int(m["collision_count"]),
        "ade": float(m["ade"]),
        "fde": float(m["fde"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--scenario", default="scenarios/scenario_02.yaml")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    rows = []
    for name, da, eps in CONDITIONS:
        for seed in seeds:
            r = run_one(args.scenario, da, eps, seed)
            r.update({"condition": name, "seed": seed})
            rows.append(r)
            print(f"{name:16s} seed={seed:2d}: "
                  f"dist={r['min_dist_m']:.3f} ttc={r['min_ttc_s']:.3f} "
                  f"t={r['time_s']:.1f} coll={r['collision_count']} ade={r['ade']:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    outdir = Path("output/poc_da_s2")
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "all_runs.csv", index=False)

    def fmt(g, col, p=3):
        return f"{g[col].mean():.{p}f}±{g[col].std(ddof=1):.{p}f}"

    summary_rows = []
    for name, _, _ in CONDITIONS:
        g = df[df.condition == name]
        summary_rows.append({
            "condition": name,
            "time_s": fmt(g, "time_s", 2),
            "speed_ms": fmt(g, "speed_ms", 2),
            "min_dist_m": fmt(g, "min_dist_m"),
            "min_ttc_s": fmt(g, "min_ttc_s"),
            "collisions": int(g.collision_count.sum()),
            "ade": fmt(g, "ade"),
        })
    summary = pd.DataFrame(summary_rows)
    print("\n=== Summary (Scenario 2, SGAN, n={} per condition) ===".format(len(seeds)))
    print(summary.to_string(index=False))
    summary.to_csv(outdir / "summary.csv", index=False)

    # Welch t-tests: each DA condition vs single-sample baseline
    base = df[df.condition == "baseline_single"]
    print("\n=== Welch t-test vs baseline_single ===")
    stat_rows = []
    for name, _, _ in CONDITIONS:
        if name == "baseline_single":
            continue
        g = df[df.condition == name]
        for col in ["min_dist_m", "min_ttc_s", "time_s"]:
            t, p = stats.ttest_ind(g[col], base[col], equal_var=False)
            d_mean = g[col].mean() - base[col].mean()
            print(f"{name:12s} {col:11s} delta={d_mean:+.3f}  p={p:.3e}")
            stat_rows.append({"condition": name, "metric": col,
                              "delta_vs_base": d_mean, "p": p})
    pd.DataFrame(stat_rows).to_csv(outdir / "welch_vs_baseline.csv", index=False)


if __name__ == "__main__":
    main()
