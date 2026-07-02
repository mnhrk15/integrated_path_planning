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

CALIBRATION-POINT NOTE (review 1.2-5, the M6-style cross-reference): three
(sigma, v0) points coexist in this repo and must not be conflated:

* (1.2005, 1.6219) -- whole-pool single fit under the default cruise estimator
  (``baseline_median`` row of outputs/rq2_cruise_sensitivity/
  cruise_sensitivity.csv). The CLI defaults below (1.20/1.62) are this point,
  rounded.
* (1.156, 1.681)  -- radius=0.35 LOCO fold mean: the RQ1b GT ``calib`` arm AND
  the point the committed DUT fidelity CSVs were actually produced with
  (their sigma/v0 columns record it).
* (1.168, 1.712)  -- radius=0.30 LOCO fold mean = the CURRENT CANONICAL
  calibration (outputs/rq2_evaluation/summary_loco.txt). Across the three
  points sigma differs by up to ~4% and v0 by up to ~5.6%; all lie inside the
  RQ1b +/-1SD sensitivity box, so no committed conclusion depends on the
  choice -- but thesis text should cite the canonical point and name the
  others explicitly.

Usage:
    .venv/bin/python examples/run_rq2_dut_validation.py --sigma 1.20 --v0 1.62
    .venv/bin/python examples/run_rq2_dut_validation.py --multivehicle --sigma 1.20 --v0 1.62
"""
import argparse
import json
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
                   help="CITR-calibrated sigma to validate. Default 1.20 = the "
                        "pooled single-fit (cruise baseline), rounded; the "
                        "committed CSVs used 1.156 (radius-0.35 LOCO mean) and "
                        "the current canonical is 1.168 (radius-0.30 LOCO mean, "
                        "summary_loco.txt) -- see the module docstring")
    p.add_argument("--v0", type=float, default=1.62,
                   help="CITR-calibrated v0 to validate. Default 1.62 = pooled "
                        "single-fit, rounded; committed CSVs used 1.681, current "
                        "canonical 1.712 -- see the module docstring")
    p.add_argument("--citr-ref-ade", type=float, default=None,
                   help="CITR-domain calibrated ADE for the degradation ratio "
                        "(e.g. the pooled value from run_rq2_evaluation)")
    p.add_argument("--multivehicle", action="store_true",
                   help="expand every clip per-vehicle (else single-vehicle subset only)")
    p.add_argument("--sidecar-from-csv", action="store_true",
                   help="only regenerate headline_tests_dut_*.json from the existing "
                        "fidelity CSVs in --out (no dataset access, no re-run)")
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


def _slug(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def dut_headline_tests(rows: List[dict], multivehicle: bool) -> List[dict]:
    """DUT fidelity KS records as AUXILIARY multiplicity-ledger entries (1.2-3).

    The DUT KS p-values (notably multivehicle p=0.013/0.024/0.0019) previously
    lived only in the fidelity CSVs, bypassing the ledger. They are NOT
    canonical hypotheses: DUT is an out-of-domain generalization check, and the
    multivehicle per-vehicle expansion counts a pedestrian once per vehicle
    (pseudo-replication => anti-conservative p). Filing them with
    ``auxiliary: true`` discloses them in the ledger's auxiliary appendix
    (within-family BH/Holm) without inflating the canonical study-wide family.
    """
    suffix = "multivehicle" if multivehicle else "single"
    caveat = ("out-of-domain generalization check, not a headline family"
              + ("; multivehicle per-vehicle expansion counts a ped once per "
                 "vehicle (pseudo-replication, p anti-conservative)"
                 if multivehicle else "")
              + "; run point (sigma, v0) recorded per row -- committed CSVs "
                "used (1.156, 1.681), see the calibration-point note in the "
                "module docstring")
    tests = []
    for r in rows:
        p = r.get("p_closest")
        if p is None or not isinstance(p, (int, float, np.floating)) \
                or not np.isfinite(p):
            continue
        tests.append({
            "test_id": f"rq2.dut.{suffix}.closest_ks.{_slug(r['group'])}",
            "description": (f"DUT closest-approach KS: {r['group']} sim vs real "
                            f"({suffix})"),
            "family": f"rq2_dut_fidelity_ks_{suffix}",
            "auxiliary": True,
            "p_value": float(p),
            "statistic": float(r["ks_closest"]),
            "sidedness": "two-sided",
            "n_encounters": int(r["n_encounters"]),
            "sigma": float(r["sigma"]),
            "v0": float(r["v0"]),
            "headline": False,
            "caveat": caveat,
        })
    return tests


def write_dut_sidecar(out_dir: Path, rows: List[dict], multivehicle: bool) -> Path:
    """Write the headline_tests sidecar next to the fidelity CSV (deterministic)."""
    suffix = "multivehicle" if multivehicle else "single"
    path = out_dir / f"headline_tests_dut_{suffix}.json"
    path.write_text(json.dumps({
        "source": f"RQ2-DUT-{suffix}",
        "generated_by": "run_rq2_dut_validation.py",
        "tests": dut_headline_tests(rows, multivehicle),
    }, indent=2) + "\n")
    return path


def sidecars_from_csv(out_dir: Path) -> List[Path]:
    """Regenerate sidecars from the committed fidelity CSVs (no dataset needed).

    The committed CSVs were produced at (1.156, 1.681); re-running the full
    validation with today's CLI defaults (1.20/1.62) would CHANGE them. This
    path converts the existing CSVs verbatim so the ledger entries match the
    committed artifacts byte-for-byte.
    """
    wrote = []
    for multivehicle in (False, True):
        suffix = "multivehicle" if multivehicle else "single"
        csv_path = out_dir / f"dut_fidelity_{suffix}.csv"
        if not csv_path.exists():
            continue
        rows = pd.read_csv(csv_path).to_dict("records")
        wrote.append(write_dut_sidecar(out_dir, rows, multivehicle))
    return wrote


def main():
    args = parse_args()
    if args.sidecar_from_csv:
        wrote = sidecars_from_csv(Path(args.out))
        if not wrote:
            raise SystemExit(f"no dut_fidelity_*.csv found under {args.out}")
        for path in wrote:
            print(f"wrote {path}")
        return
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
    sidecar = write_dut_sidecar(out_dir, rows, args.multivehicle)
    print(f"saved auxiliary ledger sidecar to {sidecar}")


if __name__ == "__main__":
    main()
