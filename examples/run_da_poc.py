#!/usr/bin/env python3
"""Margin-control experiment: distribution information vs. mere conservatism.

Extends the original distribution-aware planning PoC (single-sample baseline
vs. chance-constrained planning over the full 20-sample SGAN distribution)
with two controls that separate "value of the predicted distribution" from
"mere conservatism" (RESEARCH_ISSUES_AND_SOLUTIONS.md §C-1):

  Experiment A: single-sample planning with an inflated collision margin
      (collision_margin_inflation in {1.0, 1.1, 1.2, 1.35, 1.5}) compared
      against the robust (eps=0) planner on the MinDist-vs-Time trade-off.
  Experiment B: robust (eps=0) planning over the LSTM (no pooling)
      20-sample distribution, to test whether the robust gain is specific
      to the interaction-aware (pooling) distribution.

Runs are cached per (scenario, condition, seed) under <outdir>/runs/, so an
interrupted campaign resumes where it left off. all_runs.csv is rebuilt from
the cache on every invocation.

Usage:
    python examples/run_da_poc.py [--seeds N] [--scenarios A.yaml,B.yaml]
                                  [--conditions label1,label2] [--outdir DIR]
                                  [--ego-footprint circle|multi_circle]
                                  [--ego-footprint-n-circles N] [--total-time T]

--ego-footprint/--total-time re-verify the campaign under a different collision
geometry / time cap (A-4 issue 5(i)); they require a non-default --outdir
because cached runs do not key on them.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.core.metrics import calculate_aggregate_metrics
from src.simulation.integrated_simulator import IntegratedSimulator
from examples.run_statistical_benchmark import set_seed, resolve_model_path

DEFAULT_SCENARIOS = [
    "scenarios/scenario_01.yaml",
    "scenarios/scenario_02.yaml",
    "scenarios/scenario_03.yaml",
]

CONDITIONS = [
    # (label, method, distribution_aware, epsilon, inflation)
    ("sgan_single_inf1.00", "sgan", False, 0.0, 1.00),  # behavior-preservation anchor (= old baseline_single)
    ("sgan_single_inf1.10", "sgan", False, 0.0, 1.10),
    ("sgan_single_inf1.20", "sgan", False, 0.0, 1.20),
    ("sgan_single_inf1.35", "sgan", False, 0.0, 1.35),
    ("sgan_single_inf1.50", "sgan", False, 0.0, 1.50),
    ("sgan_robust_eps0.0",  "sgan", True,  0.0, 1.00),  # = old da_eps0.0
    ("lstm_single",         "lstm", False, 0.0, 1.00),  # Experiment B
    ("lstm_robust_eps0.0",  "lstm", True,  0.0, 1.00),  # Experiment B
]

BASELINE_LABEL = "sgan_single_inf1.00"


def run_one(scenario, method, distribution_aware, epsilon, inflation, seed,
            ego_footprint=None, n_circles=None, total_time=None,
            v0_randomization=False):
    set_seed(seed)
    config = load_config(scenario)
    config.prediction_method = method
    config.visualization_enabled = False
    config.distribution_aware_planning = distribution_aware
    config.chance_epsilon = epsilon
    config.collision_margin_inflation = inflation
    if v0_randomization:
        config.sfm_v0_randomization = True
    if ego_footprint is not None:
        config.ego_footprint = ego_footprint
    if n_circles is not None:
        config.ego_footprint_n_circles = n_circles
    if total_time is not None:
        config.total_time = total_time
    resolve_model_path(config, method)

    sim = IntegratedSimulator(config)
    history = sim.run()
    m = calculate_aggregate_metrics(
        history, config.dt,
        prediction_dt=sim.observer.sgan_dt,
        prediction_steps=config.pred_len,
    )
    end_time = float(history[-1].time)
    return {
        "ego_footprint": config.ego_footprint,
        "n_circles": int(config.ego_footprint_n_circles),
        "time_cap": float(config.total_time),
        "termination": sim.termination_reason,
        "goal_reached": sim.goal_reached,
        "time_s": end_time,
        "speed_ms": float(np.mean([r.ego_state.v for r in history])),
        "min_dist_m": float(m["min_dist"]),
        "min_ttc_s": float(m["min_ttc"]),
        "collision_count": int(m["collision_count"]),
        "ade": float(m["ade"]),
        "fde": float(m["fde"]),
    }


def cache_path(outdir: Path, scenario: str, label: str, seed: int) -> Path:
    return outdir / "runs" / Path(scenario).stem / label / f"seed_{seed:02d}.json"


def write_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def collect_rows(outdir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((outdir / "runs").glob("*/*/seed_*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS),
                    help="Comma-separated scenario YAML paths")
    ap.add_argument("--conditions", default="",
                    help="Comma-separated condition labels (default: all)")
    ap.add_argument("--outdir", default="output/exp_margin_control")
    ap.add_argument("--ego-footprint", choices=["circle", "multi_circle"], default=None,
                    help="Override ego_footprint for all runs (use a separate "
                         "--outdir: cached runs do not key on this)")
    ap.add_argument("--ego-footprint-n-circles", type=int, default=None,
                    help="Override ego_footprint_n_circles (multi_circle only)")
    ap.add_argument("--total-time", type=float, default=None,
                    help="Override total_time [s] for all scenarios (use a "
                         "separate --outdir: cached runs do not key on this)")
    ap.add_argument("--v0-randomization", action="store_true",
                    help="Per-agent desired-speed randomization "
                         "(sfm_v0_randomization=true, as in the rand benchmark; "
                         "use a separate --outdir: cached runs do not key on this)")
    args = ap.parse_args()

    if ((args.ego_footprint or args.total_time or args.v0_randomization) and
            args.outdir == "output/exp_margin_control"):
        ap.error("--ego-footprint/--total-time/--v0-randomization change run "
                 "semantics but are not part of the cache key; use a separate "
                 "--outdir")

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
    conditions = [c for c in CONDITIONS if not wanted or c[0] in wanted]
    unknown = wanted - {c[0] for c in CONDITIONS}
    if unknown:
        ap.error(f"Unknown condition labels: {sorted(unknown)}")
    seeds = list(range(args.seeds))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(scenarios) * len(conditions) * len(seeds)
    done = 0
    failed = 0
    for scenario in scenarios:
        for label, method, da, eps, inflation in conditions:
            for seed in seeds:
                done += 1
                cpath = cache_path(outdir, scenario, label, seed)
                if cpath.exists():
                    continue
                try:
                    r = run_one(scenario, method, da, eps, inflation, seed,
                                ego_footprint=args.ego_footprint,
                                n_circles=args.ego_footprint_n_circles,
                                total_time=args.total_time,
                                v0_randomization=args.v0_randomization)
                except Exception:
                    failed += 1
                    print(f"[{done}/{total}] FAILED {Path(scenario).stem} "
                          f"{label} seed={seed}", file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)
                    continue
                r.update({
                    "scenario": Path(scenario).stem,
                    "condition": label,
                    "method": method,
                    "distribution_aware": da,
                    "epsilon": eps,
                    "inflation": inflation,
                    "seed": seed,
                })
                write_atomic(cpath, r)
                print(f"[{done}/{total}] {Path(scenario).stem} {label:20s} seed={seed:2d}: "
                      f"dist={r['min_dist_m']:.3f} ttc={r['min_ttc_s']:.3f} "
                      f"t={r['time_s']:.1f} coll={r['collision_count']} ade={r['ade']:.3f}",
                      flush=True)

    df = collect_rows(outdir)
    if df.empty:
        print("No cached runs found; nothing to aggregate.", file=sys.stderr)
        sys.exit(1)
    column_order = ["scenario", "condition", "method", "distribution_aware",
                    "epsilon", "inflation", "seed", "time_s", "speed_ms",
                    "min_dist_m", "min_ttc_s", "collision_count", "ade", "fde"]
    # Footprint/cap fields only exist in caches written by newer code
    column_order += [c for c in ["ego_footprint", "n_circles", "time_cap",
                                 "termination", "goal_reached"] if c in df.columns]
    df = df[column_order].sort_values(["scenario", "condition", "seed"])
    df.to_csv(outdir / "all_runs.csv", index=False)

    # inf = "no TTC event in the run"; keep it out of means and t-tests.
    df["min_ttc_s"] = df["min_ttc_s"].replace([np.inf, -np.inf], np.nan)

    def fmt(g, col, p=3):
        return f"{g[col].mean():.{p}f}±{g[col].std(ddof=1):.{p}f}"

    summary_rows = []
    for scenario, sdf in df.groupby("scenario"):
        for label, *_ in CONDITIONS:
            g = sdf[sdf.condition == label]
            if g.empty:
                continue
            # Collision runs end early, so their time/speed would read as
            # "fast"; report those means over collision-free runs only.
            g_nc = g[g.collision_count == 0]
            summary_rows.append({
                "scenario": scenario,
                "condition": label,
                "n": len(g),
                "n_collision_runs": int((g.collision_count > 0).sum()),
                "time_s": fmt(g_nc, "time_s", 2) if not g_nc.empty else "n/a",
                "speed_ms": fmt(g_nc, "speed_ms", 2) if not g_nc.empty else "n/a",
                "min_dist_m": fmt(g, "min_dist_m"),
                "min_ttc_s": fmt(g, "min_ttc_s"),
                "collisions": int(g.collision_count.sum()),
                "ade": fmt(g, "ade"),
            })
    summary = pd.DataFrame(summary_rows)
    print(f"\n=== Summary (n={len(seeds)} seeds per condition) ===")
    print(summary.to_string(index=False))
    summary.to_csv(outdir / "summary.csv", index=False)

    # Welch t-tests: each condition vs the single-sample SGAN baseline, per scenario
    stat_rows = []
    print(f"\n=== Welch t-test vs {BASELINE_LABEL} ===")
    for scenario, sdf in df.groupby("scenario"):
        base = sdf[sdf.condition == BASELINE_LABEL]
        if base.empty:
            continue
        for label, *_ in CONDITIONS:
            if label == BASELINE_LABEL:
                continue
            g = sdf[sdf.condition == label]
            if g.empty:
                continue
            for col in ["min_dist_m", "min_ttc_s", "time_s"]:
                if col == "time_s":
                    # Collision runs end early; compare completion times only.
                    a = g[g.collision_count == 0][col]
                    b = base[base.collision_count == 0][col]
                else:
                    a, b = g[col], base[col]
                if len(a) < 2 or len(b) < 2:
                    continue
                t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                d_mean = a.mean() - b.mean()
                print(f"{scenario} {label:20s} {col:11s} delta={d_mean:+.3f}  p={p:.3e}")
                stat_rows.append({"scenario": scenario, "condition": label,
                                  "metric": col, "delta_vs_base": d_mean, "p": p})
    pd.DataFrame(stat_rows).to_csv(outdir / "welch_vs_baseline.csv", index=False)

    if failed:
        print(f"\nWARNING: {failed} run(s) failed and were not cached "
              f"(they will be retried on the next invocation).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
