#!/usr/bin/env python3
"""RQ3 campaign: real-data-grounded closed loop on VCI-CITR encounters.

Planner-driven ego on the 26 recorded CITR single-vehicle encounter
geometries, pedestrians swapped along the reactivity axis:

    ped arm   x  predictor      x  plan mode
    -------      -------------     -----------------------------
    replay       cv / lstm / sgan  single (true draw, review F4)
    calib                          robust (chance-constrained, eps=0)
    avec                           medoid (reference; sgan x replay/calib only)
    norep
    calib13x

* replay: recorded pedestrians interpolated to the sim dt -- non-reactive
  (the "what the real pedestrians actually did" reference point; RQ2
  instrument audit REPORT section 5).
* calib (1.168, 1.712) / avec (0.7, 3.5) / norep (1.0, 0.0): SFM arms in the
  calibration-consistent median-cruise regime.
* calib13x: the calibrated point deployed in the closed-loop 1.3x speed
  regime -- the regime-transfer sensitivity arm (review F2).

Seeds: the SFM/replay dynamics and the planner are deterministic; torch is
the only RNG consumer, so cv runs once (seed 0) and lstm/sgan run --seeds
times. Runs are cached per (encounter, arm-label, seed) under
<outdir>/runs/ (resumable); all_runs.csv is rebuilt from the cache on every
invocation. Cache keys do NOT include arm parameters -- changing the arm
table requires a fresh --outdir.

Parallelism: one process per ped arm (--ped-arms replay etc.); the cache
subtrees are disjoint, so concurrent arms never collide.

Usage:
    OMP_NUM_THREADS=1 python examples/run_rq3_realloop.py --ped-arms replay
    python examples/run_rq3_realloop.py --report-only     # aggregate cache
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from src.core.metrics import calculate_aggregate_metrics  # noqa: E402
from src.datasets.vci_encounter import Encounter  # noqa: E402
from src.simulation.realloop import (  # noqa: E402
    build_realloop_simulator,
    dedupe_waypoints,
    encounter_eligibility,
    recorded_ego_deviation,
)
from examples.inspect_rq3_encounters import enumerate_encounters  # noqa: E402
from examples.run_da_poc import collect_rows, write_atomic  # noqa: E402
from examples.run_statistical_benchmark import set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Arm tables (single source of truth; sigma/v0 are also recorded per-run in
# the JSON provenance, so a cached campaign is self-describing).
# ---------------------------------------------------------------------------
PED_ARMS: Dict[str, Dict] = {
    # RQ2 canonical LOCO calibration point (r=0.30), AVEC hand-tuned point,
    # and the no-repulsion null -- all in the calibration median-cruise
    # regime. calib13x deploys the calibrated point in the closed-loop 1.3x
    # regime (sensitivity arm, review F2).
    "replay":   dict(ped_kind="replay", sigma=None, v0=None,
                     speed_regime="median_cruise"),
    "calib":    dict(ped_kind="sfm", sigma=1.168, v0=1.712,
                     speed_regime="median_cruise"),
    "avec":     dict(ped_kind="sfm", sigma=0.7, v0=3.5,
                     speed_regime="median_cruise"),
    "norep":    dict(ped_kind="sfm", sigma=1.0, v0=0.0,
                     speed_regime="median_cruise"),
    "calib13x": dict(ped_kind="sfm", sigma=1.168, v0=1.712,
                     speed_regime="initial_13x"),
}

PREDICTORS = ["cv", "lstm", "sgan"]
PLAN_MODES = ["single", "robust"]

# medoid reference cells (user-approved scope): sgan x {replay, calib} only.
MEDOID_CELLS = {("replay", "sgan"), ("calib", "sgan")}

# Checkpoint kept identical to the S1-S3 instrument configuration
# (scenario_01.yaml -> zara1_12_model.pt); resolve_model_path convention.
MODEL_NAME = "zara1_12_model.pt"

SUMMARY_METRICS = ["min_dist_m", "min_ttc_s", "collision_count", "time_s",
                   "progress", "ego_dev_mean_m", "rms_jerk", "mean_accel"]


def model_path_for(pred: str) -> Optional[str]:
    if pred == "cv":
        return None
    d = "models/sgan-models" if pred == "lstm" else "models/sgan-p-models"
    p = Path(d) / MODEL_NAME
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing (run scripts/download_sgan_models.py --pooling)")
    return str(p)


def arm_label(ped_arm: str, pred: str, plan: str) -> str:
    return f"{ped_arm}__{pred}__{plan}"


def seeds_for(pred: str, n_seeds: int) -> List[int]:
    """cv is deterministic end-to-end (no torch draw affects the loop)."""
    return [0] if pred == "cv" else list(range(n_seeds))


def planned_cells(ped_arms: List[str], preds: List[str], plans: List[str],
                  include_medoid: bool) -> List[Tuple[str, str, str]]:
    """Cartesian cells plus the medoid reference cells.

    Medoid cells run only on a full-matrix invocation (both canonical plan
    modes requested) or when 'medoid' is asked for explicitly via --plans;
    a restricted --plans single/robust run must not silently grow extra cells.
    """
    regular = [p for p in plans if p != "medoid"]
    cells = [(a, pr, pl) for a in ped_arms for pr in preds for pl in regular]
    medoid_wanted = include_medoid and (
        "medoid" in plans or set(PLAN_MODES) <= set(plans))
    if medoid_wanted:
        cells += [(a, pr, "medoid") for (a, pr) in sorted(MEDOID_CELLS)
                  if a in ped_arms and pr in preds]
    return cells


def run_one_rq3(enc: Encounter, enc_row: Dict, ped_arm: str, pred: str,
                plan: str, seed: int) -> Dict:
    """One closed-loop run; returns the flat row cached as JSON."""
    spec = PED_ARMS[ped_arm]
    set_seed(seed)
    sim, prov = build_realloop_simulator(
        enc, spec["ped_kind"], pred, plan,
        sigma=spec["sigma"], v0=spec["v0"],
        speed_regime=spec["speed_regime"],
        sgan_model_path=model_path_for(pred),
    )
    history = sim.run()
    m = calculate_aggregate_metrics(
        history, sim.config.dt,
        prediction_dt=sim.observer.sgan_dt,
        prediction_steps=sim.config.pred_len,
    )
    dev = recorded_ego_deviation(history, enc, dt=sim.config.dt)
    row = {
        **enc_row,
        "ped_arm": ped_arm,
        "pred": pred,
        "plan": plan,
        "seed": int(seed),
        "termination": sim.termination_reason,
        "goal_reached": bool(sim.goal_reached),
        "censored": sim.termination_reason == "timeout",
        "n_steps": len(history),
        "time_s": float(history[-1].time) if history else float("nan"),
        "min_dist_m": float(m["min_dist"]),
        "min_ttc_s": float(m["min_ttc"]),
        "collision_count": int(m["collision_count"]),
        "ade": float(m["ade"]),
        "fde": float(m["fde"]),
        "rms_jerk": float(m["rms_jerk"]),
        "mean_accel": float(m["mean_accel"]),
        **dev,
    }
    row.update(prov)
    return row


def encounter_row(enc_id: str, scenario: str, enc: Encounter) -> Dict:
    """Per-encounter provenance shared by every run of that encounter."""
    xs, ys = dedupe_waypoints(enc.ego_xy)
    d = np.diff(np.column_stack([xs, ys]), axis=0)
    return {
        "enc_id": enc_id,
        "scenario": scenario,
        "clip": enc.clip,
        "n_frames": int(len(enc.times)),
        "window_s": float(enc.times[-1] - enc.times[0]),
        "n_peds": int(enc.ped_xy.shape[1]),
        "recorded_min_sep_m": float(enc.min_separation),
        "ego_path_len_m": float(np.hypot(d[:, 0], d[:, 1]).sum()),
    }


def cache_path(outdir: Path, enc_id: str, label: str, seed: int) -> Path:
    return outdir / "runs" / enc_id / label / f"seed_{seed:02d}.json"


def _verify_cached_arm(path: Path, ped_arm: str) -> None:
    """Fail fast when a cached run was produced under a DIFFERENT arm table.

    The cache key is (enc_id, label, seed) only, so editing PED_ARMS and
    re-running into the same --outdir would silently mix old- and new-
    parameter runs (the RQ1b DEFAULT_SCENARIOS incident, same shape). The
    per-run JSON provenance records sigma/v0/speed_regime, so a cheap read
    catches the mismatch at the first cell.
    """
    spec = PED_ARMS[ped_arm]
    with open(path) as f:
        row = json.load(f)
    expected_sigma = spec["sigma"] if spec["sigma"] is not None else float("nan")
    expected_v0 = spec["v0"] if spec["v0"] is not None else float("nan")
    expected_regime = (spec["speed_regime"] if spec["ped_kind"] == "sfm"
                       else "replay")
    sigma_ok = (np.isnan(expected_sigma) and np.isnan(float(row["sigma"]))) \
        or float(row["sigma"]) == float(expected_sigma)
    v0_ok = (np.isnan(expected_v0) and np.isnan(float(row["v0"]))) \
        or float(row["v0"]) == float(expected_v0)
    if not (sigma_ok and v0_ok and row["speed_regime"] == expected_regime):
        raise SystemExit(
            f"cached run {path} was produced under a different arm table "
            f"(cached sigma={row['sigma']}, v0={row['v0']}, "
            f"regime={row['speed_regime']}; current PED_ARMS[{ped_arm!r}]="
            f"{spec}). Use a fresh --outdir for a changed arm table.")


def run_campaign(encounters, cells, n_seeds: int, outdir: Path):
    """Every (encounter, cell, seed) into the cache; resumable."""
    n_done = n_run = n_failed = n_skipped = 0
    verified_arms = set()
    for enc_id, scenario, enc in encounters:
        ok, reason = encounter_eligibility(enc)
        if not ok:
            print(f"SKIP {enc_id}: {reason} (censored, see encounters.csv)")
            n_skipped += 1
            continue
        enc_row = encounter_row(enc_id, scenario, enc)
        for ped_arm, pred, plan in cells:
            label = arm_label(ped_arm, pred, plan)
            for seed in seeds_for(pred, n_seeds):
                path = cache_path(outdir, enc_id, label, seed)
                if path.exists():
                    if ped_arm not in verified_arms:
                        _verify_cached_arm(path, ped_arm)
                        verified_arms.add(ped_arm)
                    n_done += 1
                    continue
                t0 = time.perf_counter()
                try:
                    row = run_one_rq3(enc, enc_row, ped_arm, pred, plan, seed)
                except Exception:
                    n_failed += 1
                    print(f"FAILED {enc_id}/{label}/seed{seed}:\n"
                          f"{traceback.format_exc()}")
                    continue
                write_atomic(path, row)
                n_run += 1
                print(f"[{n_run}] {enc_id}/{label}/seed{seed} "
                      f"({time.perf_counter() - t0:.1f}s, "
                      f"min_dist={row['min_dist_m']:.2f}, "
                      f"coll={row['collision_count']}, "
                      f"term={row['termination']})", flush=True)
    print(f"\ncampaign: ran={n_run} cached={n_done} failed={n_failed} "
          f"skipped_encounters={n_skipped}")
    return n_failed


def aggregate(outdir: Path) -> pd.DataFrame:
    """Rebuild all_runs.csv (untracked) + summary.csv (tracked) from cache."""
    df = collect_rows(outdir)
    if df.empty:
        print("no cached runs found")
        return df
    df = df.sort_values(["ped_arm", "pred", "plan", "enc_id", "seed"],
                        kind="mergesort").reset_index(drop=True)
    _to_csv_atomic(df, outdir / "all_runs.csv")

    rows = []
    for (arm, pred, plan), g in df.groupby(["ped_arm", "pred", "plan"],
                                           sort=True):
        row = {"ped_arm": arm, "pred": pred, "plan": plan,
               "n_runs": len(g),
               "n_encounters": g["enc_id"].nunique(),
               "n_collision_runs": int((g["collision_count"] > 0).sum()),
               "n_goal_reached": int(g["goal_reached"].sum()),
               "n_censored": int(g["censored"].sum())}
        for c in SUMMARY_METRICS:
            vals = pd.to_numeric(g[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
            row[f"{c}_mean"] = round(float(vals.mean()), 4)
            row[f"{c}_std"] = round(float(vals.std(ddof=1)), 4) \
                if vals.notna().sum() > 1 else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    _to_csv_atomic(summary, outdir / "summary.csv")
    print(f"aggregated {len(df)} runs -> all_runs.csv, summary.csv")
    return df


def _to_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """tmp + os.replace so concurrent per-arm processes never leave a torn
    file (each finishing arm calls aggregate() on the shared outdir)."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="datasets/vci_citr/data")
    p.add_argument("--fps", type=float, default=29.97)
    p.add_argument("--min-sep", type=float, default=8.0)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--seeds", type=int, default=5,
                   help="seeds per stochastic predictor (cv always runs seed 0)")
    p.add_argument("--ped-arms", default=",".join(PED_ARMS),
                   help="comma list; one process per arm for manual parallelism")
    p.add_argument("--preds", default=",".join(PREDICTORS))
    p.add_argument("--plans", default=",".join(PLAN_MODES))
    p.add_argument("--no-medoid", action="store_true",
                   help="skip the medoid reference cells")
    p.add_argument("--outdir", default="outputs/rq3_realloop")
    p.add_argument("--expect-n", type=int, default=26,
                   help="fail fast when the encounter census changes")
    p.add_argument("--report-only", action="store_true",
                   help="aggregate the cache without running anything")
    return p.parse_args()


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        aggregate(outdir)
        return

    ped_arms = [a for a in args.ped_arms.split(",") if a]
    unknown = set(ped_arms) - set(PED_ARMS)
    if unknown:
        raise SystemExit(f"unknown ped arms {sorted(unknown)}")
    preds = [p_ for p_ in args.preds.split(",") if p_]
    unknown = set(preds) - set(PREDICTORS)
    if unknown:
        raise SystemExit(f"unknown predictors {sorted(unknown)}")
    plans = [p_ for p_ in args.plans.split(",") if p_]
    unknown = set(plans) - set(PLAN_MODES) - {"medoid"}
    if unknown:
        raise SystemExit(f"unknown plan modes {sorted(unknown)}")

    encounters = enumerate_encounters(args.root, args.fps,
                                      args.min_sep, args.min_len)
    if len(encounters) != args.expect_n:
        raise SystemExit(
            f"expected {args.expect_n} encounters, got {len(encounters)} "
            "(census changed; re-run inspect_rq3_encounters.py and review)")

    cells = planned_cells(ped_arms, preds, plans,
                          include_medoid=not args.no_medoid)
    total = sum(len(seeds_for(pr, args.seeds)) for (_, pr, _) in cells) \
        * len(encounters)
    print(f"encounters={len(encounters)}  cells={len(cells)}  "
          f"planned_runs<={total}")

    n_failed = run_campaign(encounters, cells, args.seeds, outdir)
    aggregate(outdir)
    if n_failed:
        raise SystemExit(f"{n_failed} runs FAILED (not cached; re-run to retry)")


if __name__ == "__main__":
    main()
