#!/usr/bin/env python3
"""RQ2 distribution-matching calibration (novelty direction (A)-2).

The canonical fitter minimises rollout ADE alone, which leaves (sigma, v0)
weakly identified (flat v0 valley, review C2) and under-reproduces the real
standoff (+0.68 m, 24/26). This CLI re-calibrates with a MULTI-OBJECTIVE:

    loss = w_ade * rollout_ADE + w_dist * D(closest_sim, closest_real)

where D is a distribution distance over the per-encounter closest-approach
scalars (calibration_harness.objective_multi). D defaults to the EMD
(Wasserstein-1): with n=26 encounters the KS statistic is a step function with
1/26 granularity -- a plateau landscape Nelder-Mead cannot descend -- while the
EMD is continuous and measured in metres, the same unit as the ADE term, so the
weight w_dist has a direct "metres of shape error per metre of ADE" reading.
The energy distance (units m^0.5) is available as a cross-check. The
per-encounter onset term stays OFF by default (the SFM often triggers no onset
in simulation, module docstring of the harness).

Configs are compact strings: ``w1`` = (w_ade=1, w_dist=1), ``w0.5`` =
(1, 0.5), ``pure`` = (0, 1) = distribution-only, with an optional
``_id8`` suffix = --interaction-distance 8.0 m for the ADE term (the
non-diluted arm; 8.0 matches cruise_freewalk's not-yet-reacting threshold).
``w0`` is bit-for-bit the canonical ADE fitter (tested).

Modes:
  pooled  one fit per config on all encounters (the weight-sweep table used to
          pick the LOCO configs).
  loco    leave-one-clip-out folds for the --configs list (held-out evidence:
          (a) identifiability is read from the surfaces CLI, (b) standoff
          reproduction from the paired sign/Wilcoxon tests here, (c) the
          held-out ADE sacrifice from the fold CSV). Sidecars are
          auxiliary-only (rq2dm.* namespace; never the canonical rq2.*).

The cap policy is inherited from the cap-sensitivity verdict (--cap-policy);
outputs go to a NEW directory (default outputs/rq2_instrument_audit/distmatch).

Usage:
    .venv/bin/python examples/run_rq2_distmatch.py --mode pooled \\
        --weights 0 0.25 0.5 1 2 4 --pure-emd
    .venv/bin/python examples/run_rq2_distmatch.py --mode pooled --weights 1 \\
        --interaction-distance 8.0
    .venv/bin/python examples/run_rq2_distmatch.py --mode loco \\
        --configs w1 w1_id8 pure --cap-policy median
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from loguru import logger  # noqa: E402

from examples.run_rq2_cap_sensitivity import (  # noqa: E402
    aux_paired_sidecar_tests,
    cap_kwargs,
)
from examples.run_rq2_evaluation import (  # noqa: E402
    AVEC_DEFAULT,
    COLUMNS,
    NO_REPULSION,
    RAW_KEYS,
    VEHICLE_SCENARIOS,
    _empty_raw,
    _meanstd,
    _paired_line,
    _paired_stats,
    _standoff_gap,
    evaluate_fold,
    make_folds,
)
from src.calibration import calibrate  # noqa: E402
from src.core.metrics import compare_distributions_emd  # noqa: E402
from src.datasets.vci_encounter import encounters_from_clips  # noqa: E402
from src.datasets.vci_loader import load_vci_clips  # noqa: E402
from src.simulation.calibration_harness import (  # noqa: E402
    DIST_METRICS,
    fidelity_report,
    objective_multi,
)

POOLED_COLUMNS = [
    "config", "dist_metric", "w_ade", "w_dist", "interaction_distance",
    "cap_policy", "cap_multiplier", "sigma", "v0", "fit_loss", "refined",
    "ade", "emd_closest", "gap", "n_pairs", "n_real_gt_sim", "sign_p",
    "wilcoxon_p",
]
LOCO_COLUMNS = COLUMNS + ["config", "dist_metric", "w_ade", "w_dist",
                          "obj_interaction_distance", "cap_policy",
                          "cap_multiplier"]


def parse_config(spec: str) -> Dict:
    """'w1' / 'w0.5_id8' / 'pure' / 'pure_id8' -> objective parameters."""
    parts = spec.split("_")
    head = parts[0]
    if head == "pure":
        w_ade, w_dist = 0.0, 1.0
    elif head.startswith("w"):
        try:
            w_ade, w_dist = 1.0, float(head[1:])
        except ValueError:
            raise ValueError(f"bad config {spec!r} (expected 'wX' or 'pure')")
    else:
        raise ValueError(f"bad config {spec!r} (expected 'wX' or 'pure')")
    interaction: Optional[float] = None
    for p in parts[1:]:
        if p.startswith("id"):
            try:
                interaction = float(p[2:])
            except ValueError:
                raise ValueError(f"bad config suffix {p!r} in {spec!r}")
        else:
            raise ValueError(f"bad config suffix {p!r} in {spec!r}")
    return {"config": spec, "w_ade": w_ade, "w_dist": w_dist,
            "interaction_distance": interaction}


def config_name(w_ade: float, w_dist: float,
                interaction: Optional[float]) -> str:
    head = "pure" if w_ade == 0.0 else f"w{w_dist:g}"
    return head + (f"_id{interaction:g}" if interaction is not None else "")


def make_dm_objective(args, cfg: Dict):
    """Fold objective: the multi-objective under this config + cap policy."""
    def obj(encs, s, v):
        return objective_multi(
            encs, s, v, w_ade=cfg["w_ade"], w_dist=cfg["w_dist"],
            dist_metric=args.dist_metric,
            interaction_distance=cfg["interaction_distance"],
            **cap_kwargs(args.cap_policy, args.cap_multiplier))
    return obj


def pooled_row(args, encs, cfg: Dict) -> Dict:
    """One weight-sweep row: fit + fidelity + paired stats + closest EMD."""
    obj = make_dm_objective(args, cfg)
    result = calibrate(lambda s, v: obj(encs, s, v),
                       args.sigma_grid, args.v0_grid, refine=not args.no_refine)
    kw = cap_kwargs(args.cap_policy, args.cap_multiplier)
    rep = fidelity_report(encs, result.sigma, result.v0, **kw)
    pools = {"sim": rep["closest_sim_raw"], "real": rep["closest_real_raw"]}
    p = _paired_stats(pools, "sim", "real")
    return {
        "config": cfg["config"], "dist_metric": args.dist_metric,
        "w_ade": cfg["w_ade"], "w_dist": cfg["w_dist"],
        "interaction_distance": cfg["interaction_distance"],
        "cap_policy": args.cap_policy, "cap_multiplier": args.cap_multiplier,
        "sigma": result.sigma, "v0": result.v0, "fit_loss": result.loss,
        "refined": result.refined,
        "ade": rep["rollout_ade"],
        "emd_closest": compare_distributions_emd(rep["closest_sim_raw"],
                                                 rep["closest_real_raw"]),
        "gap": p["mean_gap"] if p else float("nan"),
        "n_pairs": p["n_pairs"] if p else float("nan"),
        "n_real_gt_sim": p["n_real_gt_sim"] if p else float("nan"),
        "sign_p": p["sign_p"] if p else float("nan"),
        "wilcoxon_p": p["wilcoxon_p"] if p else float("nan"),
    }


def write_loco_summary(path: Path, df: pd.DataFrame, pools: Dict[str, list],
                       cfg: Dict, args) -> str:
    lines = [
        f"RQ2 distribution-matching calibration  config={cfg['config']}  "
        f"dist_metric={args.dist_metric}  cap_policy={args.cap_policy}  "
        f"protocol=loco  folds={len(df)}",
        "=" * 72,
        "",
        f"objective: {cfg['w_ade']:g}*ADE + {cfg['w_dist']:g}*"
        f"{args.dist_metric.upper()}(closest_sim, closest_real)"
        + (f", ADE term restricted to peds within "
           f"{cfg['interaction_distance']:g} m of the ego (non-diluted arm)"
           if cfg["interaction_distance"] is not None else ""),
        "",
        f"  sigma     : {_meanstd(df['sigma'])}",
        f"  v0        : {_meanstd(df['v0'])}",
        "",
        "Held-out test metrics (mean +/- std over folds; the ADE sacrifice of",
        "the distribution term is read against the canonical w0/median row):",
        f"  calibrated     ADE : {_meanstd(df['test_ade'])}",
        f"  AVEC default   ADE : {_meanstd(df['base_default_test_ade'])}",
        f"  no-repulsion   ADE : {_meanstd(df['base_norepulsion_test_ade'])}",
        "",
        "Paired per-encounter fidelity tests (the valid RQ2 unit, review F5):",
        f"  calibrated     paired : {_paired_line(pools, 'calibrated_closest', 'real_closest')}",
        f"  AVEC default   paired : {_paired_line(pools, 'default_closest', 'real_closest')}",
        f"  no-repulsion   paired : {_paired_line(pools, 'norepulsion_closest', 'real_closest')}",
        "",
        "Honest standoff gap (pooled closest-approach, descriptive):",
        f"  {_standoff_gap(pools)}",
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text)
    return text


def run_loco(args, clips, cfg: Dict, out_dir: Path):
    clip_encs = {id(c): encounters_from_clips([c], args.min_sep, args.min_len)
                 for c in clips}
    folds = make_folds(clips, "loco")
    objective_fn = make_dm_objective(args, cfg)
    fidelity_kwargs = cap_kwargs(args.cap_policy, args.cap_multiplier)

    rows: List[Dict] = []
    pools = _empty_raw()
    for fold_name, train_clips, test_clips in folds:
        train_encs = [e for c in train_clips for e in clip_encs[id(c)]]
        test_encs = [e for c in test_clips for e in clip_encs[id(c)]]
        row, raw = evaluate_fold(fold_name, "loco", train_clips, test_clips,
                                 train_encs, test_encs, args,
                                 objective_fn=objective_fn,
                                 fidelity_kwargs=fidelity_kwargs)
        row.update({"config": cfg["config"], "dist_metric": args.dist_metric,
                    "w_ade": cfg["w_ade"], "w_dist": cfg["w_dist"],
                    "obj_interaction_distance": cfg["interaction_distance"],
                    "cap_policy": args.cap_policy,
                    "cap_multiplier": args.cap_multiplier})
        rows.append(row)
        for k in RAW_KEYS:
            pools[k].extend(raw[k])
        print(f"  fold {fold_name:<22} sigma={row['sigma']:.3f} "
              f"v0={row['v0']:.3f} test_ade={row['test_ade']:.3f}")

    df = pd.DataFrame(rows, columns=LOCO_COLUMNS)
    csv_path = out_dir / f"folds_dm_{cfg['config']}_loco.csv"
    df.to_csv(csv_path, index=False)
    summary_path = out_dir / f"summary_dm_{cfg['config']}_loco.txt"
    text = write_loco_summary(summary_path, df, pools, cfg, args)
    print("\n" + text)

    sidecar = out_dir / f"headline_tests_dm_{cfg['config']}_loco.json"
    sidecar.write_text(json.dumps({
        "source": f"RQ2-distmatch-{cfg['config']}-loco",
        "generated_by": "run_rq2_distmatch.py",
        "tests": aux_paired_sidecar_tests(
            pools, "loco",
            family="rq2_distmatch_loco",
            prefix=f"rq2dm.loco.{cfg['config']}",
            extra={"config": cfg["config"], "dist_metric": args.dist_metric,
                   "w_ade": cfg["w_ade"], "w_dist": cfg["w_dist"],
                   "cap_policy": args.cap_policy},
            note=("distribution-matching calibration diagnostic, not a "
                  "confirmatory claim; auxiliary => excluded from the "
                  "canonical study-wide correction")),
    }, indent=2) + "\n")
    print(f"saved per-fold CSV to {csv_path}")
    print(f"saved summary to {summary_path}")
    print(f"saved auxiliary sidecar to {sidecar}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["pooled", "loco"], default="pooled")
    p.add_argument("--weights", type=float, nargs="*",
                   default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
                   help="w_dist sweep for pooled mode (w_ade fixed at 1)")
    p.add_argument("--pure-emd", action="store_true",
                   help="add the distribution-only arm (w_ade=0, w_dist=1)")
    p.add_argument("--dist-metric", choices=list(DIST_METRICS), default="emd")
    p.add_argument("--configs", nargs="*", default=["w1"],
                   help="loco configs, e.g. w1 w1_id8 pure (see module docstring)")
    p.add_argument("--cap-policy", default="median",
                   help="cap policy from the cap-sensitivity verdict")
    p.add_argument("--cap-multiplier", type=float, default=1.3)
    p.add_argument("--scenario", default="all", choices=VEHICLE_SCENARIOS + ["all"])
    p.add_argument("--root", default="datasets/vci_citr/data")
    p.add_argument("--fps", type=float, default=29.97)
    p.add_argument("--min-sep", type=float, default=8.0)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--interaction-distance", type=float, default=None,
                   help="pooled mode: ADE-term filter applied to every swept "
                        "config (the non-diluted arm); loco mode ignores this "
                        "-- use the _idX config suffix instead")
    p.add_argument("--sigma-grid", type=float, nargs="*",
                   default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    p.add_argument("--v0-grid", type=float, nargs="*",
                   default=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0])
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--out", default="outputs/rq2_instrument_audit/distmatch")
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
    if not clips:
        raise SystemExit(f"no clips for scenario {args.scenario!r} under {args.root}")
    encs = encounters_from_clips(clips, args.min_sep, args.min_len)
    if not encs:
        raise SystemExit("no encounters extracted (loosen --min-sep/--min-len)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nmode={args.mode}  dist_metric={args.dist_metric}  "
          f"cap_policy={args.cap_policy}  clips={len(clips)}  encounters={len(encs)}")

    if args.mode == "loco":
        for spec in args.configs:
            cfg = parse_config(spec)
            print(f"\n=== LOCO config {spec} ===")
            run_loco(args, clips, cfg, out_dir)
        return

    # pooled weight sweep
    cfgs = [{"config": config_name(1.0, w, args.interaction_distance),
             "w_ade": 1.0, "w_dist": float(w),
             "interaction_distance": args.interaction_distance}
            for w in args.weights]
    if args.pure_emd:
        cfgs.append({"config": config_name(0.0, 1.0, args.interaction_distance),
                     "w_ade": 0.0, "w_dist": 1.0,
                     "interaction_distance": args.interaction_distance})

    print("=== pooled weight sweep ===")
    rows = []
    for cfg in cfgs:
        row = pooled_row(args, encs, cfg)
        rows.append(row)
        print(f"  {cfg['config']:<10} sigma={row['sigma']:.3f} v0={row['v0']:.3f} "
              f"ade={row['ade']:.4f} emd={row['emd_closest']:.3f} "
              f"gap={row['gap']:+.3f} m sign {row['n_real_gt_sim']:.0f}/{row['n_pairs']:.0f}")

    df = pd.DataFrame(rows, columns=POOLED_COLUMNS)
    tag = args.cap_policy + (f"_id{args.interaction_distance:g}"
                             if args.interaction_distance is not None else "")
    if args.dist_metric != "emd":
        tag += f"_{args.dist_metric}"
    csv_path = out_dir / f"dm_pooled_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nsaved pooled weight-sweep table to {csv_path}")


if __name__ == "__main__":
    main()
