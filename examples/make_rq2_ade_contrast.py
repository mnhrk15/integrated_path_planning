#!/usr/bin/env python3
"""Paired per-fold ADE arm contrast for the RQ2 ledger (thesis review M3).

The RQ2 evaluation reports that removing the ego repulsion increases the
held-out rollout ADE by ~5% (0.640 -> 0.672 under LOCO), but that claim only
had per-arm point estimates: each arm was tested against the REAL data
(``rq2_fidelity_paired_loco``), never against the other arm. Individual
significance does not imply a significant DIFFERENCE (thesis cross-cut review
M3), so this script adds the missing arm contrast: the paired per-fold
difference d_i = no_repulsion ADE - calibrated ADE over the 26 LOCO folds,
tested with the same sign + Wilcoxon convention as every other paired family
(``run_rq2_evaluation._paired_stats``: two-sided, zero differences dropped).

Unlike ``run_rq2_evaluation.py`` (which needs the full Nelder-Mead fold
calibration to regenerate its sidecar), this script depends ONLY on the
committed ``outputs/rq2_evaluation/folds_loco.csv``, so it re-runs on a fresh
clone. It writes a separate sidecar (``headline_tests_ade_contrast.json``)
that ``make_multiplicity_ledger.py`` picks up via its ``headline_tests*.json``
glob; the frozen ``headline_tests_loco.json`` is not touched.

The family is AUXILIARY by design: the canonical RQ2 family stays the
fidelity question (real vs sim standoff); this contrast supports the ch06
"repulsion presence matters for held-out error" sentence. Note the LOCO folds
share ~96% of their training clips (LOCO refits overlap), so the fold pairs
are not independent and the p is to be read as anti-conservative -- the same
caveat already disclosed for the LOCO (sigma, v0) spread.

Usage:
    .venv/bin/python examples/make_rq2_ade_contrast.py
    .venv/bin/python examples/make_rq2_ade_contrast.py --folds <csv> --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.run_rq2_evaluation import _paired_stats  # noqa: E402

CALIB_COL = "test_ade"                      # calibrated arm, held-out ADE
NOREP_COL = "base_norepulsion_test_ade"     # no-repulsion arm, same folds


def _json_float(x):
    """NaN-safe float for strict-JSON sidecars (mirrors make_rq3_report)."""
    if x is None:
        return None
    x = float(x)
    return None if x != x else x


def ade_contrast_tests(folds: pd.DataFrame) -> List[Dict]:
    """Sign + Wilcoxon over the per-fold (no_repulsion - calibrated) ADE."""
    missing = [c for c in ("protocol", CALIB_COL, NOREP_COL)
               if c not in folds.columns]
    if missing:
        raise SystemExit(f"folds CSV is missing required column(s): {missing}")
    protocols = folds["protocol"].unique().tolist()
    if len(protocols) != 1:
        raise SystemExit(f"expected a single-protocol folds CSV, got {protocols}")
    protocol = protocols[0]

    # _paired_stats computes d = real - sim; sim=calibrated, real=no_repulsion
    # makes d_i = norep_i - calib_i, so n_real_gt_sim counts folds where
    # removing the repulsion WORSENS the held-out ADE.
    pools = {"calibrated_ade": folds[CALIB_COL].tolist(),
             "norepulsion_ade": folds[NOREP_COL].tolist()}
    s = _paired_stats(pools, "calibrated_ade", "norepulsion_ade")
    if s is None:
        raise SystemExit("folds CSV yielded no valid ADE pairs")

    fam = f"rq2_ade_contrast_{protocol}"
    note_shared = (
        "arm contrast for the ch06 held-out-ADE claim (thesis cross-cut "
        "review M3): per-arm-vs-real significance does not imply a "
        "significant ARM DIFFERENCE, so the difference is tested directly. "
        "LOCO folds share ~96% of training clips (refit overlap), so the "
        "fold pairs are not independent and the p reads anti-conservative.")
    return [
        {
            "test_id": f"rq2.{protocol}.ade_sign.no_repulsion",
            "description": (
                f"Paired per-fold sign test: no_repulsion vs calibrated "
                f"held-out rollout ADE ({protocol}, n={s['n_pairs']} folds)"),
            "family": fam,
            "protocol": protocol,
            "auxiliary": True,
            "p_value": _json_float(s["sign_p"]),
            "statistic": float(s["n_real_gt_sim"]),
            "sidedness": "two-sided",
            "n_pairs": s["n_pairs"],
            "n_norep_gt_calib": s["n_real_gt_sim"],
            "n_norep_lt_calib": s["n_real_lt_sim"],
            "mean_gap_ade": s["mean_gap"],
            "headline": False,
            "note": note_shared,
        },
        {
            "test_id": f"rq2.{protocol}.ade_wilcoxon.no_repulsion",
            "description": (
                f"Paired per-fold Wilcoxon signed-rank: no_repulsion vs "
                f"calibrated held-out rollout ADE ({protocol}, "
                f"n={s['n_pairs']} folds)"),
            "family": fam,
            "protocol": protocol,
            "auxiliary": True,
            "p_value": _json_float(s["wilcoxon_p"]),
            "statistic": _json_float(s["wilcoxon_stat"]),
            "sidedness": "two-sided",
            "n_pairs": s["n_pairs"],
            "mean_gap_ade": s["mean_gap"],
            "headline": False,
            "note": ("magnitude-aware companion over the SAME per-fold "
                     "differences (family convention of the paired RQ2 "
                     "families); " + note_shared),
        },
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="outputs/rq2_evaluation/folds_loco.csv")
    ap.add_argument("--out", default="outputs/rq2_evaluation")
    args = ap.parse_args()

    folds = pd.read_csv(args.folds)
    tests = ade_contrast_tests(folds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / "headline_tests_ade_contrast.json"
    sidecar.write_text(json.dumps({
        "source": f"RQ2-{tests[0]['protocol']}-ade-contrast",
        "generated_by": "make_rq2_ade_contrast.py",
        "tests": tests,
    }, indent=2) + "\n")

    s = tests[0]
    fmt = lambda p: "undefined" if p is None else f"{p:.4g}"  # noqa: E731
    print(f"norep worse in {s['n_norep_gt_calib']}/{s['n_pairs']} folds  "
          f"sign p={fmt(s['p_value'])}  Wilcoxon p={fmt(tests[1]['p_value'])}  "
          f"mean ADE gap {s['mean_gap_ade']:+.4f}")
    print(f"saved headline-test sidecar to {sidecar}")


if __name__ == "__main__":
    main()
