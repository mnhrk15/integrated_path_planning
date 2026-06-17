#!/usr/bin/env python3
"""Out-of-domain validation of the CITR-calibrated ego repulsion on VCI-DUT (RQ2).

RQ2 identifies (sigma, v0) on the controlled CITR vehicle-crowd clips. This
script asks the complementary question: does that CITR-fit repulsion still
reproduce avoidance on DUT -- a natural campus shared space with crowd-crowd
forces and unobserved goals mixed in? DUT is NOT re-calibrated (those confounds
make it unsuitable for identification); the CITR (sigma, v0) is simply applied
and its fidelity compared against the AVEC default and the no-repulsion null on
the SAME DUT encounters. If the calibrated ADE collapses toward no-repulsion, or
the closest-approach distribution diverges, that is the domain gap.

DUT clips are mostly multi-vehicle, which the CITR encounter extractor skips:

* default (single-vehicle subset): ``encounters_from_clips`` keeps only the DUT
  clips that happen to carry exactly one vehicle -- zero source change, a quick
  first read, but a thin sample.
* ``--multivehicle``: ``encounters_from_clips_multivehicle`` projects every clip
  into one virtual single-vehicle view per vehicle (each vehicle as ego, all peds
  reacting). Larger sample; pedestrians reacting to several vehicles are counted
  once per vehicle (noted in the report).

The DUT root MUST be ``datasets/vci_dut/data`` (not the parent): a duplicate copy
under ``datasets/vci_dut/legacy/`` would make the loader's duplicate-clip guard
raise.

Usage:
    .venv/bin/python examples/run_rq2_dut_validation.py --sigma 1.20 --v0 1.62
    .venv/bin/python examples/run_rq2_dut_validation.py --multivehicle --sigma 1.20 --v0 1.62
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from loguru import logger  # noqa: E402

from src.datasets.vci_loader import load_vci_clips, vehicle_speed_samples  # noqa: E402
from src.datasets.vci_encounter import (  # noqa: E402
    encounters_from_clips,
    encounters_from_clips_multivehicle,
)
from src.simulation.calibration_harness import fidelity_report  # noqa: E402

AVEC_DEFAULT = (0.7, 3.5)
NO_REPULSION = (1.0, 0.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="datasets/vci_dut/data",
                   help="VCI-DUT data root (the 'data' dir, NOT the parent: legacy/ "
                        "holds a duplicate that trips the loader's duplicate guard)")
    p.add_argument("--fps", type=float, default=23.98, help="DUT frame rate (drone)")
    p.add_argument("--sigma", type=float, default=1.20,
                   help="CITR-calibrated sigma to validate (from run_rq2_evaluation)")
    p.add_argument("--v0", type=float, default=1.62, help="CITR-calibrated v0 to validate")
    p.add_argument("--citr-ref-ade", type=float, default=None,
                   help="CITR-domain calibrated ADE for the degradation ratio "
                        "(e.g. the pooled value from run_rq2_evaluation)")
    p.add_argument("--multivehicle", action="store_true",
                   help="expand every clip per-vehicle (else single-vehicle subset only)")
    p.add_argument("--min-sep", type=float, default=8.0,
                   help="max closest-approach for a span to count as an encounter [m]")
    p.add_argument("--min-len", type=int, default=5, help="min encounter length [frames]")
    p.add_argument("--out", default="outputs/rq2_dut_validation",
                   help="output directory for the fidelity CSV")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def vehicle_speed_samples_pooled(clips) -> np.ndarray:
    """All finite vehicle speed samples [m/s] pooled across clips."""
    parts = [vehicle_speed_samples(c.veh) for c in clips if c.veh is not None]
    return np.concatenate(parts) if parts else np.array([])


def main():
    args = parse_args()
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    clips = load_vci_clips(args.root, "dut", fps=args.fps)
    extractor = (encounters_from_clips_multivehicle if args.multivehicle
                 else encounters_from_clips)
    encs = extractor(clips, args.min_sep, args.min_len)
    mode = "multi-vehicle (per-vehicle expansion)" if args.multivehicle \
        else "single-vehicle subset"
    if not encs:
        raise SystemExit(f"no DUT encounters extracted in {mode} mode "
                         "(loosen --min-sep/--min-len, or try --multivehicle)")

    n_clips_with_veh = sum(1 for c in clips
                           if c.veh is not None and c.veh.positions.shape[1] >= 1)
    print(f"\nDUT validation  mode={mode}")
    print(f"clips={len(clips)} (with vehicle={n_clips_with_veh})  encounters={len(encs)}")
    if args.multivehicle:
        print("note: peds reacting to K vehicles are counted once per vehicle")

    groups = [
        ("calibrated", (args.sigma, args.v0)),
        ("AVEC default", AVEC_DEFAULT),
        ("no repulsion", NO_REPULSION),
    ]
    rows: List[dict] = []
    print(f"\n=== fidelity on DUT encounters ({mode}) ===")
    for name, (s, v) in groups:
        r = fidelity_report(encs, s, v)
        rows.append({
            "group": name, "sigma": s, "v0": v, "multivehicle": args.multivehicle,
            "n_encounters": r["n_encounters"], "ade": r["rollout_ade"],
            "ks_closest": r["ks_closest"], "p_closest": r["p_closest"],
            "ks_onset": r["ks_onset"], "mean_closest_sim": r["mean_closest_sim"],
            "mean_closest_real": r["mean_closest_real"],
        })
        print(f"  {name:<14} sigma={s:.3f} v0={v:.3f} | ADE={r['rollout_ade']:.3f} "
              f"closest sim/real={r['mean_closest_sim']:.2f}/{r['mean_closest_real']:.2f} "
              f"KS_closest={r['ks_closest']:.3f}")

    # Degradation verdict: calibrated DUT ADE vs CITR reference, and vs the
    # no-repulsion null on DUT (does the calibrated repulsion still help here?).
    cal_ade = rows[0]["ade"]
    norep_ade = rows[2]["ade"]
    print("\n=== domain-gap verdict ===")
    if args.citr_ref_ade is not None:
        print(f"  calibrated ADE: DUT={cal_ade:.3f} vs CITR={args.citr_ref_ade:.3f} "
              f"=> x{cal_ade / args.citr_ref_ade:.2f} on DUT")
    # NaN-safe: a degenerate no-repulsion ADE (NaN, which is truthy) must NOT slip
    # through as a real margin, and the collapse check must not read NaN<0.02 as False.
    margin = ((norep_ade - cal_ade) / norep_ade
              if np.isfinite(norep_ade) and norep_ade else float("nan"))
    collapsed = np.isfinite(margin) and margin < 0.02
    print(f"  calibrated vs no-repulsion on DUT: ADE {cal_ade:.3f} vs {norep_ade:.3f} "
          f"=> repulsion still cuts ADE by {100 * margin:.1f}%"
          + ("  (collapsed: no out-of-domain benefit)" if collapsed else ""))

    speed = vehicle_speed_samples_pooled(clips)
    if speed.size:
        pct = np.percentile(speed, [5, 50, 95])
        print(f"  DUT vehicle speed: p5={pct[0]:.2f} p50={pct[1]:.2f} p95={pct[2]:.2f} m/s "
              f"(n={speed.size})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "multivehicle" if args.multivehicle else "single"
    csv_path = out_dir / f"dut_fidelity_{suffix}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nsaved DUT fidelity to {csv_path}")


if __name__ == "__main__":
    main()
