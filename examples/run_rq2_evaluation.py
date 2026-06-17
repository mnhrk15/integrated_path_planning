#!/usr/bin/env python3
"""Cross-validated train/test evaluation of the RQ2 ego-repulsion calibration.

The spike (``run_rq2_calibration.py``) fit (sigma, v0) once on the whole CITR
pool and reported fidelity on the same encounters. That answers "can a single
(sigma, v0) reproduce the observed avoidance" but not "does the identified
(sigma, v0) GENERALISE to unseen clips, and is it stable". This script supplies
that evidence by re-fitting on a training split and reporting fidelity on a
held-out split, across folds:

* ``--protocol loco`` (Leave-One-Clip-Out, the MAIN evidence): each clip is the
  held-out test set once; the other clips train. ~26 folds give a (sigma, v0)
  mean +/- std over many samples -- the thesis's parameter-stability claim.
* ``--protocol loso`` (Leave-One-Scenario-Out, AUXILIARY): each interaction
  geometry (front/back/lateral) is held out once -- a harder "unseen geometry"
  stress test with only 4 folds, reported as a robustness check, not the
  stability claim.

Splitting is ALWAYS by clip (never by encounter): a clip yields several
encounters, and letting a clip's encounters straddle train and test would leak.

Each fold calibrates with the rollout-ADE fitter (``objective_rollout_ade``)
and validates with ``fidelity_report`` (ADE + closest-approach / avoidance-onset
KS) on the held-out encounters, with the AVEC default (0.7, 3.5) and the
no-repulsion null (1.0, 0.0) evaluated on the SAME held-out encounters as
controls. The CITR vehicle speed band [p5, p95] is reported too: (sigma, v0) is
identified only in that low-speed domain (the velocity-extrapolation limitation,
RQ2 plan risk #2).

Usage:
    .venv/bin/python examples/run_rq2_evaluation.py --protocol loco --scenario all
    .venv/bin/python examples/run_rq2_evaluation.py --protocol loso --scenario all
    # fast smoke (one scenario, no Nelder-Mead):
    .venv/bin/python examples/run_rq2_evaluation.py --protocol loco --scenario vci_front --no-refine
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# numba (via pysocialforce) and matplotlib emit copious DEBUG records on stderr;
# silence them as the benchmark CLIs do (run_statistical_benchmark.py).
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from loguru import logger  # noqa: E402

from src.calibration import calibrate  # noqa: E402
from src.datasets.vci_loader import (  # noqa: E402
    ClipTracks,
    load_vci_clips,
    vehicle_speed_samples,
)
from src.datasets.vci_encounter import Encounter, encounters_from_clips  # noqa: E402
from src.simulation.calibration_harness import (  # noqa: E402
    fidelity_report,
    objective_rollout_ade,
)

# AVEC paper's hand-tuned values and the no-repulsion null (control group),
# mirrored from run_rq2_calibration.py so the two CLIs stay in lockstep.
AVEC_DEFAULT = (0.7, 3.5)
NO_REPULSION = (1.0, 0.0)  # sigma irrelevant when v0=0

# CITR scenarios that carry a vehicle (the only ones usable for ego calibration).
VEHICLE_SCENARIOS = ["vci_front", "vci_back", "vci_lat_bi", "vci_lat_uni"]

# CSV schema: every fold row carries exactly these keys (missing values -> NaN),
# so the DataFrame is rectangular regardless of empty-test / degenerate folds.
COLUMNS = [
    "fold", "protocol", "n_train_clips", "n_test_clips", "n_train_encs", "n_test_encs",
    "sigma", "v0", "refined", "train_ade", "test_ade",
    "test_ks_closest", "test_p_closest", "test_ks_onset", "test_p_onset",
    "test_mean_closest_sim", "test_mean_closest_real",
    "base_default_test_ade", "base_default_test_ks_closest",
    "base_norepulsion_test_ade", "base_norepulsion_test_ks_closest",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", choices=["loco", "loso"], default="loco",
                   help="loco=leave-one-clip-out (main), loso=leave-one-scenario-out (aux)")
    p.add_argument("--scenario", default="all",
                   choices=VEHICLE_SCENARIOS + ["all"],
                   help="restrict to one CITR scenario, or 'all' to pool every vehicle scenario")
    p.add_argument("--root", default="datasets/vci_citr/data",
                   help="VCI-CITR data root (the 'data' dir, not legacy/)")
    p.add_argument("--fps", type=float, default=29.97, help="CITR frame rate (NTSC=29.97)")
    p.add_argument("--min-sep", type=float, default=8.0,
                   help="max closest-approach for a span to count as an encounter [m]")
    p.add_argument("--min-len", type=int, default=5, help="min encounter length [frames]")
    p.add_argument("--interaction-distance", type=float, default=None,
                   help="restrict fitter ADE to peds approaching within this distance [m]")
    p.add_argument("--sigma-grid", type=float, nargs="*",
                   default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    p.add_argument("--v0-grid", type=float, nargs="*",
                   default=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0])
    p.add_argument("--no-refine", action="store_true", help="skip Nelder-Mead refinement")
    p.add_argument("--out", default="outputs/rq2_evaluation",
                   help="output directory for the per-fold CSV and summary")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def make_folds(
    clips: List[ClipTracks], protocol: str
) -> List[Tuple[str, List[ClipTracks], List[ClipTracks]]]:
    """Partition clips into (fold_name, train_clips, test_clips) folds.

    loco: each clip is the held-out test once (>=1 train clip required). loso:
    each scenario is held out once. Folds operate on clip OBJECTS (not stems), so
    a stem reused across scenarios never merges two clips. Deterministic order
    ((scenario, stem) for loco, scenario for loso).
    """
    ordered = sorted(clips, key=lambda c: (c.scenario or "", c.clip))
    folds: List[Tuple[str, List[ClipTracks], List[ClipTracks]]] = []
    if protocol == "loco":
        for c in ordered:
            train = [d for d in ordered if d is not c]
            if not train:
                raise SystemExit("loco needs >=2 clips; got 1 (loosen --scenario)")
            folds.append((c.clip, train, [c]))
    elif protocol == "loso":
        scenarios = sorted({c.scenario for c in ordered})
        if len(scenarios) < 2:
            raise SystemExit(
                f"loso needs >=2 scenarios; got {scenarios} "
                "(use --scenario all, or --protocol loco)"
            )
        for s in scenarios:
            test = [c for c in ordered if c.scenario == s]
            train = [c for c in ordered if c.scenario != s]
            folds.append((s or "<none>", train, test))
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"unknown protocol {protocol!r}")
    return folds


def _nan_row(fold_name: str, protocol: str, train_clips, test_clips,
             n_train_encs: int, n_test_encs: int) -> Dict[str, float]:
    """A fully-populated row with all metric columns NaN (counts kept)."""
    row = {c: float("nan") for c in COLUMNS}
    row.update({
        "fold": fold_name, "protocol": protocol,
        "n_train_clips": len(train_clips), "n_test_clips": len(test_clips),
        "n_train_encs": n_train_encs, "n_test_encs": n_test_encs,
        "refined": False,
    })
    return row


def evaluate_fold(
    fold_name: str,
    protocol: str,
    train_clips: List[ClipTracks],
    test_clips: List[ClipTracks],
    train_encs: List[Encounter],
    test_encs: List[Encounter],
    args,
) -> Dict[str, float]:
    """Calibrate on train, validate on held-out test; one CSV row.

    train_ade uses ``fidelity_report``'s rollout_ade (unfiltered, every ped) so
    it shares a scale with test_ade -- NOT the fitter loss, which the
    --interaction-distance filter would put on a different (non-comparable) scale.
    """
    row = _nan_row(fold_name, protocol, train_clips, test_clips,
                   len(train_encs), len(test_encs))
    if not train_encs:
        logger.warning(f"fold {fold_name!r}: no training encounters, skipping calibration")
        return row

    def obj(s, v):
        return objective_rollout_ade(train_encs, s, v,
                                     interaction_distance=args.interaction_distance)

    try:
        result = calibrate(obj, args.sigma_grid, args.v0_grid, refine=not args.no_refine)
    except ValueError:
        logger.warning(f"fold {fold_name!r}: objective non-finite on the entire grid "
                       "(--interaction-distance too tight?); NaN row")
        return row

    row["sigma"] = result.sigma
    row["v0"] = result.v0
    row["refined"] = result.refined
    row["train_ade"] = fidelity_report(train_encs, result.sigma, result.v0)["rollout_ade"]

    if test_encs:
        te = fidelity_report(test_encs, result.sigma, result.v0)
        bd = fidelity_report(test_encs, *AVEC_DEFAULT)
        bn = fidelity_report(test_encs, *NO_REPULSION)
        row.update({
            "test_ade": te["rollout_ade"],
            "test_ks_closest": te["ks_closest"],
            "test_p_closest": te["p_closest"],
            "test_ks_onset": te["ks_onset"],
            "test_p_onset": te["p_onset"],
            "test_mean_closest_sim": te["mean_closest_sim"],
            "test_mean_closest_real": te["mean_closest_real"],
            "base_default_test_ade": bd["rollout_ade"],
            "base_default_test_ks_closest": bd["ks_closest"],
            "base_norepulsion_test_ade": bn["rollout_ade"],
            "base_norepulsion_test_ks_closest": bn["ks_closest"],
        })
    return row


def _meanstd(series: pd.Series) -> str:
    """'mean +/- std (n)' over finite entries, or 'n/a' when none are finite."""
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "n/a"
    return f"{vals.mean():.3f} +/- {vals.std():.3f} (n={vals.size})"


def speed_domain(clips: List[ClipTracks]) -> Dict[str, float]:
    """CITR vehicle speed percentiles [m/s] pooled over clips (RQ2 limitation #2)."""
    samples = [vehicle_speed_samples(c.veh) for c in clips if c.veh is not None]
    pooled = np.concatenate(samples) if samples else np.array([])
    if pooled.size == 0:
        return {}
    pct = np.percentile(pooled, [5, 50, 90, 95])
    return {"p5": pct[0], "p50": pct[1], "p90": pct[2], "p95": pct[3],
            "max": float(pooled.max()), "n": int(pooled.size)}


def write_summary(path: Path, df: pd.DataFrame, protocol: str, speed: Dict[str, float]) -> str:
    """Human-readable summary; returns the text (also printed to stdout)."""
    lines = [
        f"RQ2 cross-validated evaluation  protocol={protocol}  folds={len(df)}",
        "=" * 72,
        "",
        "Calibrated (sigma, v0) stability across folds:",
        f"  sigma     : {_meanstd(df['sigma'])}",
        f"  v0        : {_meanstd(df['v0'])}",
        f"  AVEC default for reference: sigma={AVEC_DEFAULT[0]}, v0={AVEC_DEFAULT[1]}",
        "",
        "Held-out test metrics (mean +/- std over folds), 3 groups:",
        f"  calibrated     ADE : {_meanstd(df['test_ade'])}",
        f"  AVEC default   ADE : {_meanstd(df['base_default_test_ade'])}",
        f"  no-repulsion   ADE : {_meanstd(df['base_norepulsion_test_ade'])}",
        "",
        f"  calibrated     KS_closest : {_meanstd(df['test_ks_closest'])}",
        f"  AVEC default   KS_closest : {_meanstd(df['base_default_test_ks_closest'])}",
        f"  no-repulsion   KS_closest : {_meanstd(df['base_norepulsion_test_ks_closest'])}",
        f"  calibrated     KS_onset   : {_meanstd(df['test_ks_onset'])}",
        "",
        f"  closest-approach sim/real (calibrated): "
        f"{_meanstd(df['test_mean_closest_sim'])} / {_meanstd(df['test_mean_closest_real'])}",
        "",
    ]
    if speed:
        lines += [
            "CITR vehicle speed domain (calibration is valid only here; "
            "AVEC ego runs at 5-6 m/s = extrapolation, RQ2 limitation #2):",
            f"  p5={speed['p5']:.2f}  p50={speed['p50']:.2f}  p90={speed['p90']:.2f}  "
            f"p95={speed['p95']:.2f}  max={speed['max']:.2f} m/s  (n={speed['n']} samples)",
            f"  => identified (sigma, v0) holds for ego speeds in ~[{speed['p5']:.1f}, "
            f"{speed['p95']:.1f}] m/s; 5-6 m/s is unverified by this data.",
            "",
        ]
    text = "\n".join(lines)
    path.write_text(text)
    return text


def main():
    args = parse_args()
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    clips = load_vci_clips(args.root, "citr", fps=args.fps)
    wanted = VEHICLE_SCENARIOS if args.scenario == "all" else [args.scenario]
    clips = [c for c in clips if c.scenario in wanted]
    if not clips:
        raise SystemExit(f"no clips for scenario {args.scenario!r} under {args.root}")

    # Pre-extract encounters once per clip (keyed by object identity) so the
    # alignment+slicing is not redone for every fold the clip appears in.
    clip_encs: Dict[int, List[Encounter]] = {
        id(c): encounters_from_clips([c], args.min_sep, args.min_len) for c in clips
    }
    total_encs = sum(len(v) for v in clip_encs.values())
    if total_encs == 0:
        raise SystemExit("no encounters extracted (loosen --min-sep/--min-len)")

    folds = make_folds(clips, args.protocol)
    print(f"\nprotocol={args.protocol}  scenario={args.scenario}  "
          f"clips={len(clips)}  encounters={total_encs}  folds={len(folds)}")

    rows: List[Dict[str, float]] = []
    for fold_name, train_clips, test_clips in folds:
        train_encs = [e for c in train_clips for e in clip_encs[id(c)]]
        test_encs = [e for c in test_clips for e in clip_encs[id(c)]]
        row = evaluate_fold(fold_name, args.protocol, train_clips, test_clips,
                            train_encs, test_encs, args)
        rows.append(row)
        print(f"  fold {fold_name:<22} train_encs={row['n_train_encs']:>3} "
              f"test_encs={row['n_test_encs']:>3} | sigma={row['sigma']:.3f} "
              f"v0={row['v0']:.3f} test_ade={row['test_ade']:.3f} "
              f"KS_closest={row['test_ks_closest']:.3f}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"folds_{args.protocol}.csv"
    df.to_csv(csv_path, index=False)

    speed = speed_domain(clips)
    summary_path = out_dir / f"summary_{args.protocol}.txt"
    text = write_summary(summary_path, df, args.protocol, speed)
    print("\n" + text)
    print(f"saved per-fold CSV to {csv_path}")
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
