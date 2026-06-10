#!/usr/bin/env python3
"""Footprint re-verification benchmark (A-4, step 3 of 3).

Replays the paper's 123-run campaign (CV seed 0; LSTM/SGAN seeds 0..19; three
scenarios) under two conditions:

  circle       : the paper configuration (single-circle ego footprint in both
                 the planner and the metrics) — behavior anchor, expected to
                 reproduce output/statistical_benchmark*/all_runs.csv per seed
  multi_circle : ego_footprint=multi_circle (3-circle cover of the 4.5 x 2.0 m
                 rectangle) in the planner, the metrics, and the state machine

For every run, observational footprint metrics are additionally computed from
the simulation history with geometry FIXED ACROSS CONDITIONS (legacy centre
distance, 3-circle clearance, exact-rectangle clearance), so the two
conditions are comparable regardless of what the in-loop metric used.

Questions answered:
  Q1: do the paper-configuration trajectories violate the vehicle-shaped
      footprint (rect_violation_steps > 0 in the circle condition)?
  Q2: does planning with the multi-circle footprint remove those violations,
      and at what travel-time / clearance cost?

Runs are cached per (scenario, condition, method, seed) under <outdir>/runs/,
so an interrupted campaign resumes where it left off.

Usage:
    python examples/run_footprint_benchmark.py [--seeds N]
        [--scenarios A.yaml,B.yaml] [--conditions circle,multi_circle]
        [--outdir DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.core.footprint import (
    EgoFootprint,
    rectangle_surface_distance,
    world_to_vehicle_frame,
)
from src.core.metrics import calculate_aggregate_metrics
from src.simulation.integrated_simulator import IntegratedSimulator
from examples.run_statistical_benchmark import set_seed, resolve_model_path

DEFAULT_SCENARIOS = [
    "scenarios/scenario_01.yaml",
    "scenarios/scenario_02.yaml",
    "scenarios/scenario_03.yaml",
]

CONDITIONS = ["circle", "multi_circle"]

# Methods exactly as in the paper campaign (123 runs total)
METHOD_SEEDS = [("cv", [0]), ("lstm", None), ("sgan", None)]  # None = 0..N-1

# Fixed observational geometry (identical across conditions)
VEHICLE_LENGTH = 4.5
VEHICLE_WIDTH = 2.0
OBS_FOOTPRINT = EgoFootprint.multi_circle(VEHICLE_LENGTH, VEHICLE_WIDTH, 3)
PED_RADIUS = 0.2

# Scenario-default total_time caps (for censoring-aware anchor comparison)
SCENARIO_DEFAULT_CAP = {
    "scenario_01": 20.0,
    "scenario_02": 30.0,
    "scenario_03": 30.0,
}

# Paper campaign CSVs for the per-seed behavior-preservation check.
# comfort_s{1,2,3} are the campaigns the paper tables were produced from
# (verified: regenerating with current code reproduces them bit-exactly);
# the older statistical_benchmark{,_s2,_s3} dirs are stale pre-715d7b3
# precursors and must not be used as anchors.
PAPER_CSVS = {
    "scenario_01": "output/comfort_s1/all_runs.csv",
    "scenario_02": "output/comfort_s2/all_runs.csv",
    "scenario_03": "output/comfort_s3/all_runs.csv",
}


def observational_footprint_metrics(history) -> dict:
    """Centre/multi-circle/rectangle clearances from history, config-agnostic."""
    legacy_min = np.inf
    multi_min = np.inf
    rect_min = np.inf
    multi_steps = 0
    rect_steps = 0
    for r in history:
        peds = np.asarray(r.ped_state.positions, dtype=float)
        if peds.size == 0:
            continue
        peds = peds.reshape(-1, 2)
        e = r.ego_state

        legacy_min = min(legacy_min, float(np.min(
            np.linalg.norm(peds - np.array([e.x, e.y]), axis=1))))

        centers = OBS_FOOTPRINT.circle_centers(e.x, e.y, e.yaw)
        d_multi = float(np.min(np.linalg.norm(
            peds[None, :, :] - centers[:, None, :], axis=2)))
        multi_min = min(multi_min, d_multi)
        if d_multi < OBS_FOOTPRINT.radius + PED_RADIUS:
            multi_steps += 1

        ped_local = world_to_vehicle_frame(peds, e.x, e.y, e.yaw)
        d_rect = float(np.min(rectangle_surface_distance(
            ped_local, VEHICLE_LENGTH, VEHICLE_WIDTH)))
        rect_min = min(rect_min, d_rect)
        if d_rect < PED_RADIUS:
            rect_steps += 1

    return {
        "legacy_min_dist": legacy_min,
        "multi_clearance": multi_min - (OBS_FOOTPRINT.radius + PED_RADIUS),
        "multi_violation_steps": multi_steps,
        "rect_clearance": rect_min - PED_RADIUS,
        "rect_violation_steps": rect_steps,
    }


def run_one(scenario: str, condition: str, method: str, seed: int,
            total_time: float | None = None) -> dict:
    set_seed(seed)
    config = load_config(scenario)
    config.prediction_method = method
    config.visualization_enabled = False
    config.ego_footprint = "circle" if condition == "circle" else "multi_circle"
    if total_time is not None:
        # Extending the cap only affects runs that would have been
        # right-censored: uncensored runs end at the goal as before.
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
    row = {
        "scenario": Path(scenario).stem,
        "condition": condition,
        "method": method,
        "seed": seed,
        "time_cap": float(config.total_time),
        "goal_reached": bool(end_time < config.total_time - 1.5 * config.dt),
        "time_s": end_time,
        "speed_ms": float(np.mean([r.ego_state.v for r in history])),
        "min_dist_m": float(m["min_dist"]),
        "min_ttc_s": float(m["min_ttc"]),
        "collision_count": int(m["collision_count"]),
        "ade": float(m["ade"]),
        "fde": float(m["fde"]),
        "rms_jerk": float(m["rms_jerk"]),
    }
    row.update(observational_footprint_metrics(history))
    return row


def cache_path(outdir: Path, scenario: str, condition: str, method: str, seed: int) -> Path:
    return (outdir / "runs" / Path(scenario).stem / condition /
            f"{method}_seed_{seed:02d}.json")


def write_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def collect_rows(outdir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((outdir / "runs").glob("*/*/*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def behavior_check(df: pd.DataFrame, repo: Path) -> list:
    """Verify the circle condition preserves current-code behavior per seed.

    Primary anchor: the margin-control campaign caches (same code state,
    exact identity expected). Secondary: the paper campaign CSVs
    (comfort_s*), which agree to CSV rounding precision.

    When the campaign was run with an extended total_time, anchor runs that
    were right-censored at the scenario-default cap legitimately diverge
    (the new run continues past the old cap), so those seeds are excluded.
    """
    lines = ["### Primary: margin-control campaign caches (exact identity expected)", ""]
    anchors = {"sgan": "sgan_single_inf1.00", "lstm": "lstm_single"}
    keys = ["time_s", "speed_ms", "min_dist_m", "min_ttc_s", "ade", "fde"]
    for scen in sorted(df["scenario"].unique()):
        default_cap = SCENARIO_DEFAULT_CAP.get(scen)
        for method, anchor_label in anchors.items():
            sub = df[(df["scenario"] == scen) & (df["condition"] == "circle") &
                     (df["method"] == method)]
            run_dir = repo / "output" / "exp_margin_control" / "runs" / scen / anchor_label
            if sub.empty or not run_dir.exists():
                lines.append(f"- {scen}/{method}: SKIPPED (missing runs or anchor cache)")
                continue
            cap_extended = ("time_cap" in sub.columns and
                            (sub["time_cap"] > default_cap + 1e-9).any())
            max_diff, n, n_censored = 0.0, 0, 0
            for _, row in sub.iterrows():
                cache = run_dir / f"seed_{int(row['seed']):02d}.json"
                if not cache.exists():
                    continue
                with open(cache) as f:
                    ref = json.load(f)
                if cap_extended and ref["time_s"] >= default_cap - 0.15:
                    n_censored += 1
                    continue
                max_diff = max(max_diff, max(abs(row[k] - ref[k]) for k in keys))
                n += 1
            status = "PASS" if max_diff == 0.0 else "FAIL"
            note = f", {n_censored} censored-anchor seeds excluded" if n_censored else ""
            lines.append(f"- {scen}/{method}: {status} over {n} seeds "
                         f"(max|Δ|={max_diff:.2e}{note})")

    lines += ["", "### Paper campaign CSVs (comfort_s*; rounding-level "
              "agreement expected: CSVs round time to 2dp, others to 4dp)", ""]
    for scen, csv_rel in PAPER_CSVS.items():
        csv_path = repo / csv_rel
        sub = df[(df["scenario"] == scen) & (df["condition"] == "circle")]
        if not csv_path.exists() or sub.empty:
            lines.append(f"- {scen}: SKIPPED (missing paper CSV or runs)")
            continue
        paper = pd.read_csv(csv_path)
        merged = sub.assign(method=sub["method"].str.upper()).merge(
            paper, on=["method", "seed"], suffixes=("_new", "_paper"))
        if merged.empty:
            lines.append(f"- {scen}: SKIPPED (no overlapping method/seed)")
            continue
        default_cap = SCENARIO_DEFAULT_CAP.get(scen)
        if ("time_cap" in merged.columns and
                (merged["time_cap"] > default_cap + 1e-9).any()):
            n_before = len(merged)
            merged = merged[merged["time_s_paper"] < default_cap - 0.15]
            if len(merged) < n_before:
                lines.append(f"  ({scen}: {n_before - len(merged)} censored-anchor "
                             "seeds excluded from the comparison below)")
        if merged.empty:
            lines.append(f"- {scen}: SKIPPED (all anchor seeds censored)")
            continue
        diffs = {
            col: float(np.max(np.abs(merged[f"{col}_new"] - merged[f"{col}_paper"])))
            for col in ["time_s", "min_dist_m", "ade"]
        }
        lines.append(
            f"- {scen}: over {len(merged)} runs "
            f"max|Δ| time={diffs['time_s']:.2e}, "
            f"min_dist={diffs['min_dist_m']:.2e}, ade={diffs['ade']:.2e}")
    return lines


def welch(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def build_report(df: pd.DataFrame, outdir: Path, repo: Path):
    lines = [
        "# Footprint re-verification benchmark (A-4, step 3)",
        "",
        f"- Runs: {len(df)} "
        "(conditions: circle = paper config anchor, multi_circle = 3-circle "
        "planner+metrics footprint; observational geometry fixed across conditions)",
        f"- Rectangle: {VEHICLE_LENGTH} x {VEHICLE_WIDTH} m; 3 circles r={OBS_FOOTPRINT.radius:.2f} m"
        f" at offsets {np.round(OBS_FOOTPRINT.offsets, 2).tolist()} m; ped radius {PED_RADIUS} m",
        "",
        "## Behavior preservation (circle condition vs paper campaign CSVs)",
        "",
    ]
    lines += behavior_check(df, repo)

    lines += [
        "",
        "## Q1: rectangle-footprint violations under the paper configuration",
        "",
        "| scenario | method | runs | runs w/ rect violation | worst rect clearance [m] | runs w/ multi-circle violation |",
        "|---|---|---|---|---|---|",
    ]
    circ = df[df["condition"] == "circle"]
    for (scen, method), g in circ.groupby(["scenario", "method"]):
        lines.append(
            f"| {scen} | {method} | {len(g)} | "
            f"{int((g['rect_violation_steps'] > 0).sum())} | "
            f"{g['rect_clearance'].min():+.3f} | "
            f"{int((g['multi_violation_steps'] > 0).sum())} |")

    has_goal = "goal_reached" in df.columns
    goal_hdr = " goal reached (c→m) |" if has_goal else ""
    goal_sep = "---|" if has_goal else ""
    lines += [
        "",
        "## Q2: circle vs multi_circle (paired by scenario/method, Welch p for n=20)",
        "",
        "| scenario | method | Δtime [s] | p(time) | Δrect clearance [m] | p(clear) "
        f"| rect viol (c→m) | legacy MinDist (c→m) |{goal_hdr}",
        f"|---|---|---|---|---|---|---|---|{goal_sep}",
    ]
    for (scen, method), g in df.groupby(["scenario", "method"]):
        c = g[g["condition"] == "circle"]
        mc = g[g["condition"] == "multi_circle"]
        if c.empty or mc.empty:
            continue
        d_time = mc["time_s"].mean() - c["time_s"].mean()
        d_clear = mc["rect_clearance"].mean() - c["rect_clearance"].mean()
        goal_cell = ""
        if has_goal:
            goal_cell = (f" {int(c['goal_reached'].sum())}/{len(c)}→"
                         f"{int(mc['goal_reached'].sum())}/{len(mc)} |")
        lines.append(
            f"| {scen} | {method} | {d_time:+.2f} | "
            f"{welch(mc['time_s'].values, c['time_s'].values):.2g} | "
            f"{d_clear:+.3f} | "
            f"{welch(mc['rect_clearance'].values, c['rect_clearance'].values):.2g} | "
            f"{int((c['rect_violation_steps'] > 0).sum())}→"
            f"{int((mc['rect_violation_steps'] > 0).sum())} | "
            f"{c['legacy_min_dist'].mean():.2f}→{mc['legacy_min_dist'].mean():.2f} |"
            f"{goal_cell}")
    if has_goal:
        n_unreached = int((~df["goal_reached"]).sum())
        cap = df["time_cap"].max()
        lines += ["",
                  f"Note: time_cap={cap:.0f} s. Runs not reaching the goal by the cap "
                  f"({n_unreached} total) indicate persistent stalls (mission failure), "
                  "not slow completion; their time_s equals the cap and right-censors "
                  "the Δtime estimate."]

    n_q1 = int((circ["rect_violation_steps"] > 0).sum())
    mc_all = df[df["condition"] == "multi_circle"]
    n_q2 = int((mc_all["rect_violation_steps"] > 0).sum())
    lines += [
        "",
        "## Verdict",
        "",
        f"- Q1: {n_q1} / {len(circ)} paper-configuration runs violate the vehicle rectangle.",
        f"- Q2: {n_q2} / {len(mc_all)} multi-circle-planned runs violate the vehicle rectangle.",
        f"- Legacy-metric collisions (1.2 m centre threshold): "
        f"{int(df['collision_count'].sum())} across all runs.",
    ]

    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    import logging
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--outdir", default="output/exp_footprint")
    ap.add_argument("--total-time", type=float, default=None,
                    help="Override total_time [s] for all scenarios (use a "
                         "separate --outdir: cached runs do not key on this)")
    ap.add_argument("--report-only", action="store_true",
                    help="Rebuild the report from cached runs without simulating")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        ap.error(f"Unknown conditions: {sorted(unknown)}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).parent.parent

    if not args.report_only:
        jobs = [
            (scenario, condition, method, seed)
            for scenario in scenarios
            for condition in conditions
            for method, seeds in METHOD_SEEDS
            for seed in (seeds if seeds is not None else range(args.seeds))
        ]
        for i, (scenario, condition, method, seed) in enumerate(jobs, 1):
            cpath = cache_path(outdir, scenario, condition, method, seed)
            if cpath.exists():
                continue
            print(f"[{i}/{len(jobs)}] {Path(scenario).stem} {condition} {method} seed={seed}",
                  flush=True)
            try:
                row = run_one(scenario, condition, method, seed, args.total_time)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                continue
            write_atomic(cpath, row)

    df = collect_rows(outdir)
    if df.empty:
        sys.exit("No cached runs found")
    df.to_csv(outdir / "all_runs.csv", index=False)
    build_report(df, outdir, repo)
    print(f"\nWrote {outdir / 'all_runs.csv'} and {outdir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
