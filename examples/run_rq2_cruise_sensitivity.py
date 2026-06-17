#!/usr/bin/env python3
"""How the cruise-speed estimator moves the calibrated (sigma, v0) (RQ2 diagnostic).

The pooled CITR calibration lands at v0~=1.6, far below the AVEC paper's hand-tuned
v0=3.5. One suspected confound: the harness sets each ped's DESIRED speed from the
median of its recorded speed (``_cruise_speeds``), but the recorded speed already
dips while the ped slows to avoid the vehicle. That biases the desired speed down,
weakens the goal-driving force, and lets a weaker (lower-v0) repulsion explain the
same avoidance. This script re-calibrates under several cruise estimators and
reports how far v0 moves back toward 3.5 -- quantifying the bias instead of just
asserting it.

Estimators (all swapped in via the harness ``cruise_fn`` hook):
* baseline_median   -- the current default (whole-window median).
* freewalk_thr{6,8,10}_q50 -- median over frames where the ped is farther than
  {6,8,10} m from the ego (not yet reacting).
* upper_q85         -- 85th-percentile speed over all frames (cheap dip-robust).

Usage:
    .venv/bin/python examples/run_rq2_cruise_sensitivity.py --scenario all
"""
import argparse
import functools
import logging
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from loguru import logger  # noqa: E402

from src.calibration import calibrate  # noqa: E402
from src.datasets.vci_loader import load_vci_clips  # noqa: E402
from src.datasets.vci_encounter import Encounter, encounters_from_clips  # noqa: E402
from src.simulation.calibration_harness import (  # noqa: E402
    CruiseEstimator,
    _cruise_speeds,
    cruise_freewalk,
    cruise_upper_quantile,
    objective_rollout_ade,
)

AVEC_V0 = 3.5  # paper's hand-tuned v0 (the reference the bias is measured against)
VEHICLE_SCENARIOS = ["vci_front", "vci_back", "vci_lat_bi", "vci_lat_uni"]


def _baseline(enc: Encounter) -> np.ndarray:
    """Default estimator (bit-identical to cruise_fn=None)."""
    return _cruise_speeds(enc.ped_vel)


def estimators() -> List[Tuple[str, CruiseEstimator]]:
    return [
        ("baseline_median", _baseline),
        ("freewalk_thr6_q50", functools.partial(cruise_freewalk, ego_distance_threshold=6.0)),
        ("freewalk_thr8_q50", functools.partial(cruise_freewalk, ego_distance_threshold=8.0)),
        ("freewalk_thr10_q50", functools.partial(cruise_freewalk, ego_distance_threshold=10.0)),
        ("upper_q85", functools.partial(cruise_upper_quantile, quantile=0.85)),
    ]


def mean_cruise(encs: List[Encounter], fn: CruiseEstimator) -> float:
    """Mean per-ped desired speed [m/s] pooled over all peds in all encounters."""
    vals = np.concatenate([np.asarray(fn(e), dtype=float).ravel() for e in encs])
    return float(np.mean(vals)) if vals.size else float("nan")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="all", choices=VEHICLE_SCENARIOS + ["all"])
    p.add_argument("--root", default="datasets/vci_citr/data")
    p.add_argument("--fps", type=float, default=29.97)
    p.add_argument("--min-sep", type=float, default=8.0)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--sigma-grid", type=float, nargs="*",
                   default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    p.add_argument("--v0-grid", type=float, nargs="*",
                   default=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0])
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--out", default="outputs/rq2_cruise_sensitivity")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    clips = load_vci_clips(args.root, "citr", fps=args.fps)
    wanted = VEHICLE_SCENARIOS if args.scenario == "all" else [args.scenario]
    clips = [c for c in clips if c.scenario in wanted]
    encs = encounters_from_clips(clips, args.min_sep, args.min_len)
    if not encs:
        raise SystemExit("no encounters extracted (loosen --min-sep/--min-len)")

    print(f"\nscenario={args.scenario}  clips={len(clips)}  encounters={len(encs)}")
    print("=== cruise-estimator sensitivity (re-calibrated per estimator) ===")

    rows: List[dict] = []
    baseline_v0 = None
    for name, fn in estimators():
        def obj(s, v, _fn=fn):
            return objective_rollout_ade(encs, s, v, cruise_fn=_fn)
        result = calibrate(obj, args.sigma_grid, args.v0_grid, refine=not args.no_refine)
        mc = mean_cruise(encs, fn)
        if name == "baseline_median":
            baseline_v0 = result.v0
        rows.append({
            "estimator": name, "sigma": result.sigma, "v0": result.v0,
            "loss_ade": result.loss, "refined": result.refined, "mean_cruise": mc,
        })
        print(f"  {name:<20} sigma={result.sigma:.3f} v0={result.v0:.3f} "
              f"ADE={result.loss:.4f} mean_cruise={mc:.2f} m/s")

    print("\n=== v0 bias verdict (does a free-walking cruise move v0 toward AVEC 3.5?) ===")
    for r in rows:
        shift = r["v0"] - baseline_v0
        closed = ((r["v0"] - baseline_v0) / (AVEC_V0 - baseline_v0) * 100
                  if AVEC_V0 != baseline_v0 else float("nan"))
        print(f"  {r['estimator']:<20} v0={r['v0']:.3f} "
              f"(shift {shift:+.3f} from baseline {baseline_v0:.3f}; "
              f"{closed:.0f}% of the way to AVEC {AVEC_V0})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cruise_sensitivity.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nsaved cruise sensitivity to {csv_path}")


if __name__ == "__main__":
    main()
