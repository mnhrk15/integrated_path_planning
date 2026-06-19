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


def apply_sfm_and_cruise_overrides(config, ego_repulsion_sigma=None,
                                   ego_repulsion_v0=None, ego_target_speed=None):
    """Merge RQ1b GT / cruise overrides into a loaded config, in place.

    sigma/v0 are merged into ``social_force_params`` so scenario-level keys
    (e.g. agent_radius, and the other of sigma/v0) survive instead of being
    dropped by a wholesale assignment. A cruise override also clamps the initial
    speed so the ego does not start above the new target. Returns ``config``.
    """
    if ego_repulsion_sigma is not None or ego_repulsion_v0 is not None:
        sfp = dict(getattr(config, "social_force_params", None) or {})
        if ego_repulsion_sigma is not None:
            sfp["ego_repulsion.sigma"] = float(ego_repulsion_sigma)
        if ego_repulsion_v0 is not None:
            sfp["ego_repulsion.v0"] = float(ego_repulsion_v0)
        config.social_force_params = sfp
    if ego_target_speed is not None:
        config.ego_target_speed = float(ego_target_speed)
        st = list(config.ego_initial_state)
        if len(st) > 3:
            st[3] = min(st[3], float(ego_target_speed))
        config.ego_initial_state = st
    return config


def run_one(scenario, method, distribution_aware, epsilon, inflation, seed,
            ego_footprint=None, n_circles=None, total_time=None,
            v0_randomization=False, ego_repulsion_sigma=None,
            ego_repulsion_v0=None, ego_target_speed=None):
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
    # RQ1b: inject calibrated pedestrian SFM ego-repulsion and restrict cruise
    # to the RQ2 calibration-valid speed domain (see helper).
    apply_sfm_and_cruise_overrides(config, ego_repulsion_sigma,
                                   ego_repulsion_v0, ego_target_speed)
    resolve_model_path(config, method)

    sim = IntegratedSimulator(config)
    history = sim.run()
    m = calculate_aggregate_metrics(
        history, config.dt,
        prediction_dt=sim.observer.sgan_dt,
        prediction_steps=config.pred_len,
    )
    end_time = float(history[-1].time)
    # Provenance: the SFM ego-repulsion actually used. Read it off the
    # pedestrian simulator (authoritative after normalization); fall back to
    # the config dict when a scenario has no pedestrians (pedestrian_sim None).
    ped_sim = getattr(sim, "pedestrian_sim", None)
    if ped_sim is not None:
        eff_sigma = float(ped_sim.ego_repulsion_sigma)
        eff_v0 = float(ped_sim.ego_repulsion_v0)
    else:
        sfp = getattr(config, "social_force_params", None) or {}
        eff_sigma = float(sfp.get("ego_repulsion.sigma", float("nan")))
        eff_v0 = float(sfp.get("ego_repulsion.v0", float("nan")))
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
        "rms_jerk": float(m["rms_jerk"]),
        "mean_accel": float(m["mean_accel"]),
        "ego_repulsion_sigma": eff_sigma,
        "ego_repulsion_v0": eff_v0,
        "ego_target_speed": float(config.ego_target_speed),
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


def run_campaign(scenarios, conditions, seeds, outdir, overrides=None):
    """Run every (scenario, condition, seed) cell into ``outdir/runs/``.

    Runs are cached per cell (resumable); returns ``(df, n_failed)`` where df is
    rebuilt from the cache on every call. ``overrides`` is forwarded verbatim as
    keyword arguments to :func:`run_one` (ego_footprint, n_circles, total_time,
    v0_randomization, ego_repulsion_sigma, ego_repulsion_v0, ego_target_speed).
    """
    overrides = overrides or {}
    outdir = Path(outdir)
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
                                **overrides)
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

    return collect_rows(outdir), failed


def aggregate_and_write(df, outdir, conditions, baseline_label=BASELINE_LABEL,
                        n_seeds=None):
    """Write all_runs.csv / summary.csv / welch_vs_baseline.csv for one arm.

    ``conditions`` is the ordered condition tuple list (only its labels are
    used). Returns ``(summary_df, stat_df)``.
    """
    outdir = Path(outdir)
    cond_labels = [c[0] for c in conditions]

    column_order = ["scenario", "condition", "method", "distribution_aware",
                    "epsilon", "inflation", "seed", "time_s", "speed_ms",
                    "min_dist_m", "min_ttc_s", "collision_count", "ade", "fde"]
    # These fields only exist in caches written by newer code.
    column_order += [c for c in ["rms_jerk", "mean_accel", "ego_footprint",
                                 "n_circles", "time_cap", "termination",
                                 "goal_reached", "ego_repulsion_sigma",
                                 "ego_repulsion_v0", "ego_target_speed"]
                     if c in df.columns]
    df = df[column_order].sort_values(["scenario", "condition", "seed"])
    df.to_csv(outdir / "all_runs.csv", index=False)

    # inf = "no TTC event in the run"; keep it out of means and t-tests.
    df = df.copy()
    df["min_ttc_s"] = df["min_ttc_s"].replace([np.inf, -np.inf], np.nan)

    def fmt(g, col, p=3):
        return f"{g[col].mean():.{p}f}±{g[col].std(ddof=1):.{p}f}"

    summary_rows = []
    for scenario, sdf in df.groupby("scenario"):
        for label in cond_labels:
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
    hdr = f" (n={n_seeds} seeds per condition)" if n_seeds else ""
    print(f"\n=== Summary{hdr} ===")
    print(summary.to_string(index=False))
    summary.to_csv(outdir / "summary.csv", index=False)

    # Welch t-tests: each condition vs the single-sample baseline, per scenario
    stat_rows = []
    print(f"\n=== Welch t-test vs {baseline_label} ===")
    for scenario, sdf in df.groupby("scenario"):
        base = sdf[sdf.condition == baseline_label]
        if base.empty:
            continue
        for label in cond_labels:
            if label == baseline_label:
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
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(outdir / "welch_vs_baseline.csv", index=False)
    return summary, stat_df


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
    ap.add_argument("--ego-repulsion-sigma", type=float, default=None,
                    help="Override pedestrian SFM ego-repulsion sigma (RQ1b; "
                         "use a separate --outdir: not part of the cache key)")
    ap.add_argument("--ego-repulsion-v0", type=float, default=None,
                    help="Override pedestrian SFM ego-repulsion v0 (RQ1b; "
                         "use a separate --outdir: not part of the cache key)")
    ap.add_argument("--ego-target-speed", type=float, default=None,
                    help="Override ego cruise target speed [m/s] for all "
                         "scenarios (RQ1b calibration-domain speed; use a "
                         "separate --outdir: not part of the cache key)")
    args = ap.parse_args()

    non_cache_overrides = (args.ego_footprint or
                           args.ego_footprint_n_circles is not None or
                           args.total_time or
                           args.v0_randomization or
                           args.ego_repulsion_sigma is not None or
                           args.ego_repulsion_v0 is not None or
                           args.ego_target_speed is not None)
    if non_cache_overrides and args.outdir == "output/exp_margin_control":
        ap.error("--ego-footprint/--ego-footprint-n-circles/--total-time/"
                 "--v0-randomization/--ego-repulsion-sigma/--ego-repulsion-v0/"
                 "--ego-target-speed change run semantics but are not part of "
                 "the cache key; use a separate --outdir")

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
    conditions = [c for c in CONDITIONS if not wanted or c[0] in wanted]
    unknown = wanted - {c[0] for c in CONDITIONS}
    if unknown:
        ap.error(f"Unknown condition labels: {sorted(unknown)}")
    seeds = list(range(args.seeds))

    overrides = {
        "ego_footprint": args.ego_footprint,
        "n_circles": args.ego_footprint_n_circles,
        "total_time": args.total_time,
        "v0_randomization": args.v0_randomization,
        "ego_repulsion_sigma": args.ego_repulsion_sigma,
        "ego_repulsion_v0": args.ego_repulsion_v0,
        "ego_target_speed": args.ego_target_speed,
    }
    df, failed = run_campaign(scenarios, conditions, seeds, args.outdir, overrides)
    if df.empty:
        print("No cached runs found; nothing to aggregate.", file=sys.stderr)
        sys.exit(1)
    aggregate_and_write(df, args.outdir, conditions, BASELINE_LABEL,
                        n_seeds=len(seeds))

    if failed:
        print(f"\nWARNING: {failed} run(s) failed and were not cached "
              f"(they will be retried on the next invocation).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
