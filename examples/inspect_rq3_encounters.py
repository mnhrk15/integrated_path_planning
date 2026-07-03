#!/usr/bin/env python3
"""RQ3 Phase-0 geometry spike: per-encounter feasibility audit (read-only).

Before wiring the planner-driven ego onto the 26 CITR single-vehicle
encounters (RQ3 real-data-grounded closed loop), audit each encounter's
geometry for the failure modes that would degenerate the closed loop:

* near-duplicate spline waypoints (recorded ego standing still -> ds=0 breaks
  ``CubicSpline2D``'s arc-length parameterization),
* reference paths shorter than the goal-termination radius (2 m) plus margin
  (instant "goal" at t=0),
* near-stationary ego windows (``ego_target_speed`` ~ 0 starves the planner),
* strongly folded paths (net displacement much shorter than arc length; a
  Frenet corridor around such a path is ill-conditioned).

Encounters failing a criterion are NOT dropped silently: every encounter gets
a row with an ``eligible`` flag and a ``reason`` column, so the campaign can
skip them with a disclosed census (censoring, not exclusion).

Enumeration mirrors run_rq2_evaluation.py exactly (load_vci_clips ->
VEHICLE_SCENARIOS filter -> (scenario, stem) order -> encounters_from_clips
per clip), so ``enc_id`` is stable across RQ3 tooling.

Usage:
    python examples/inspect_rq3_encounters.py [--root datasets/vci_citr/data]
        [--out outputs/rq3_realloop/encounters.csv]
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.vci_loader import load_vci_clips  # noqa: E402
from src.datasets.vci_encounter import Encounter, encounters_from_clips  # noqa: E402
from src.simulation.realloop import (  # noqa: E402
    dedupe_waypoints,
    encounter_eligibility,
)
from examples.run_rq2_evaluation import VEHICLE_SCENARIOS  # noqa: E402

CSV_COLUMNS = [
    "enc_id", "scenario", "clip", "n_frames", "window_s", "n_peds",
    "recorded_min_sep_m", "ego_path_len_m", "ego_path_len_deduped_m",
    "n_waypoints_raw", "n_waypoints_deduped", "ego_net_disp_m", "straightness",
    "ego_speed_initial_ms", "ego_speed_median_ms", "ego_speed_p95_ms",
    "ped_speed_median_ms", "eligible", "reason",
]


def _path_length(xy: np.ndarray) -> float:
    d = np.diff(np.asarray(xy, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def audit_encounter(enc: Encounter, enc_id: str, scenario: str) -> Dict:
    """One census row per encounter; never raises on degenerate geometry."""
    ego_xy = np.asarray(enc.ego_xy, dtype=float)
    n_frames = len(enc.times)
    window_s = float(enc.times[-1] - enc.times[0])
    path_len = _path_length(ego_xy)
    net_disp = float(np.hypot(*(ego_xy[-1] - ego_xy[0])))
    straightness = net_disp / path_len if path_len > 1e-9 else 0.0

    try:
        xs, ys = dedupe_waypoints(ego_xy)
        n_dedup = len(xs)
        path_len_dedup = _path_length(np.column_stack([xs, ys]))
    except ValueError:
        n_dedup = 1
        path_len_dedup = 0.0

    med_speed = float(np.nanmedian(enc.ego_vel))
    ped_speed = float(np.nanmedian(np.hypot(enc.ped_vel[..., 0],
                                            enc.ped_vel[..., 1])))

    # Single source of truth: the campaign skips on exactly this verdict
    # (src/simulation/realloop.py), so the census can never drift from it.
    eligible, reason = encounter_eligibility(enc)

    return {
        "enc_id": enc_id,
        "scenario": scenario,
        "clip": enc.clip,
        "n_frames": n_frames,
        "window_s": round(window_s, 3),
        "n_peds": int(enc.ped_xy.shape[1]),
        "recorded_min_sep_m": round(float(enc.min_separation), 3),
        "ego_path_len_m": round(path_len, 3),
        "ego_path_len_deduped_m": round(path_len_dedup, 3),
        "n_waypoints_raw": len(ego_xy),
        "n_waypoints_deduped": n_dedup,
        "ego_net_disp_m": round(net_disp, 3),
        "straightness": round(straightness, 3),
        "ego_speed_initial_ms": round(float(enc.ego_vel[0]), 3),
        "ego_speed_median_ms": round(med_speed, 3),
        "ego_speed_p95_ms": round(float(np.nanpercentile(enc.ego_vel, 95)), 3),
        "ped_speed_median_ms": round(ped_speed, 3),
        "eligible": eligible,
        "reason": reason,
    }


def enumerate_encounters(root: str, fps: float, min_sep: float,
                         min_len: int) -> List[Tuple[str, str, Encounter]]:
    """(enc_id, scenario, Encounter) in run_rq2_evaluation's deterministic order."""
    clips = load_vci_clips(root, "citr", fps=fps)
    clips = [c for c in clips if c.scenario in VEHICLE_SCENARIOS]
    if not clips:
        raise SystemExit(f"no vehicle clips under {root}")
    ordered = sorted(clips, key=lambda c: (c.scenario or "", c.clip))
    out: List[Tuple[str, str, Encounter]] = []
    for c in ordered:
        for k, enc in enumerate(encounters_from_clips([c], min_sep, min_len)):
            out.append((f"{c.scenario}__{c.clip}__e{k:02d}", c.scenario, enc))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="datasets/vci_citr/data")
    p.add_argument("--fps", type=float, default=29.97)
    p.add_argument("--min-sep", type=float, default=8.0)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--out", default="outputs/rq3_realloop/encounters.csv")
    return p.parse_args()


def main():
    args = parse_args()
    encs = enumerate_encounters(args.root, args.fps, args.min_sep, args.min_len)
    rows = [audit_encounter(enc, enc_id, scenario)
            for enc_id, scenario, enc in encs]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_ok = int(df["eligible"].sum())
    print(f"encounters={len(df)}  eligible={n_ok}  ineligible={len(df) - n_ok}")
    if n_ok < len(df):
        print("\nineligible census:")
        print(df.loc[~df["eligible"],
                     ["enc_id", "reason", "ego_path_len_deduped_m",
                      "ego_speed_median_ms", "straightness"]].to_string(index=False))
    print(f"\nwindow_s: min={df.window_s.min():.1f} med={df.window_s.median():.1f} "
          f"max={df.window_s.max():.1f}")
    print(f"n_peds: min={df.n_peds.min()} med={int(df.n_peds.median())} "
          f"max={df.n_peds.max()}")
    print(f"ego median speed [m/s]: min={df.ego_speed_median_ms.min():.2f} "
          f"med={df.ego_speed_median_ms.median():.2f} "
          f"max={df.ego_speed_median_ms.max():.2f}")
    print(f"recorded_min_sep [m]: min={df.recorded_min_sep_m.min():.2f} "
          f"med={df.recorded_min_sep_m.median():.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
