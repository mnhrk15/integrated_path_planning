#!/usr/bin/env python3
"""RQ2 speed-cap policy sensitivity (review F2): is the standoff gap a harness artifact?

The calibration harness pins pysocialforce's ``max_speeds`` -- which is BOTH the
DesiredForce target speed and the hard velocity cap -- to each ped's recorded
median speed, so a simulated ped can never ACCELERATE above its cruise speed to
evade the ego (it can only re-direct). Review F2 hypothesises that this
structural inability explains (a) the flat v0 valley (weak identifiability, C2)
and (b) the standoff under-reproduction (+0.68 m, real>sim 24/26 LOCO
encounters). This CLI re-calibrates (sigma, v0) under alternative cap policies
and measures both effects; the closed loop the parameters are deployed into
runs yet another regime (max_speeds = 1.3 x initial_speeds), so the policies
also quantify that regime mismatch.

Policies (see calibration_harness._apply_cap_policy for the mechanics):
  median      target=cap=cruise            (default harness; historical numbers)
  closedloop  target=cap=1.3*cruise        (the closed-loop deployment regime;
                                            NOTE peds walk ~30% too fast, so its
                                            ADE mixes the cap effect with a
                                            cruise-speed error -- regime probe,
                                            not a fit-quality comparison)
  uncapped    target=cruise, cap=10 m/s    (decoupled; evasion acceleration
                                            allowed, walking speed unchanged)
  capfit      target=cruise, cap=m*cruise  (headroom m swept in pooled mode; the
                                            LOCO run takes --cap-multiplier=m*)

Modes:
  pooled  one fit on all encounters per policy. ``--policy all`` also sweeps the
          capfit headroom over --m-grid and picks m* by the pooled fit loss
          (capfit_m_profile.csv); the VERDICT must instead be read from the
          held-out LOCO evidence -- picking and judging on the same pooled ADE
          would double-dip.
  loco    leave-one-clip-out folds for ONE policy (reuses evaluate_fold, so the
          fold loop / NaN-row / raw-pool conventions match the canonical RQ2
          evaluation). Pooled held-out closest-approach scalars feed the PAIRED
          sign/Wilcoxon tests (the valid RQ2 unit, review F5) and an
          auxiliary-only ledger sidecar (namespace rq2cap.*, never colliding
          with the canonical rq2.* records).

The AVEC-default / no-repulsion arms are evaluated WITHIN each policy
(within-regime controls): arm-vs-arm and policy-vs-policy comparisons stay
separable. All outputs go to a NEW directory (default
outputs/rq2_instrument_audit/cap); the committed outputs/rq2_evaluation/* are
never touched. The data-driven verdict lives in
examples/make_rq2_instrument_report.py.

Usage:
    .venv/bin/python examples/run_rq2_cap_sensitivity.py --mode pooled --policy all
    .venv/bin/python examples/run_rq2_cap_sensitivity.py --mode loco --policy uncapped
    .venv/bin/python examples/run_rq2_cap_sensitivity.py --mode loco --policy capfit \\
        --cap-multiplier 1.5
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
from src.datasets.vci_encounter import Encounter  # noqa: E402
from src.datasets.vci_loader import load_vci_clips  # noqa: E402
from src.simulation.calibration_harness import (  # noqa: E402
    CAP_POLICIES,
    UNCAPPED_SPEED,
    fidelity_report,
    objective_rollout_ade,
)
from src.datasets.vci_encounter import encounters_from_clips  # noqa: E402

# CSV schema for the pooled per-policy table (make_rq2_instrument_report.py
# consumes these names; keep them in sync). NOTE the cap_multiplier column
# records the --cap-multiplier CLI value as provenance; the harness only USES
# it under the capfit policy -- read the effective regime off cap_headroom
# (median/closedloop: 1.0 over their own target, uncapped: inf).
POOLED_COLUMNS = [
    "policy", "cap_multiplier", "cap_headroom", "sigma", "v0", "fit_loss",
    "refined", "ade_calibrated", "ade_avec", "ade_norep",
    "mean_closest_sim", "mean_closest_real",
    "gap_calibrated", "n_pairs", "n_real_gt_sim_calibrated", "sign_p_calibrated",
    "wilcoxon_p_calibrated",
    "gap_avec", "n_real_gt_sim_avec", "sign_p_avec",
    "gap_norep", "n_real_gt_sim_norep", "sign_p_norep",
    "obj_interaction_distance", "scenario",
]
M_PROFILE_COLUMNS = [
    "m", "sigma", "v0", "fit_loss", "refined",
    "ade", "gap", "n_pairs", "n_real_gt_sim", "sign_p",
    "obj_interaction_distance", "scenario",
]
# Fit-affecting run parameters (--interaction-distance, --scenario) are
# recorded IN the rows/sidecars: without them a non-default run would be
# indistinguishable from the canonical one and could silently be picked up as
# the w0 reference by make_rq2_instrument_report (review finding).
LOCO_COLUMNS = COLUMNS + ["cap_policy", "cap_multiplier",
                          "obj_interaction_distance", "scenario"]


def effective_headroom(policy: str, cap_multiplier: float) -> float:
    """Cap headroom over the cruise target (cap / DesiredForce target).

    median: cap == target (1.0). closedloop: target AND cap are both
    1.3 x cruise, so the headroom over the (raised) target is still 1.0 -- the
    regime differs by the TARGET level, not by headroom. uncapped: effectively
    unbounded (inf). capfit: the swept multiplier m.
    """
    return {"median": 1.0, "closedloop": 1.0, "uncapped": float("inf"),
            "capfit": float(cap_multiplier)}[policy]


def cap_kwargs(policy: str, cap_multiplier: float) -> Dict:
    """kwargs threading one policy through the harness entry points."""
    return {"cap_policy": policy, "cap_multiplier": float(cap_multiplier)}


def make_objective(args, policy: str, cap_multiplier: float):
    """Fold objective: rollout ADE under the given cap policy."""
    def obj(encs: List[Encounter], s: float, v: float) -> float:
        return objective_rollout_ade(
            encs, s, v, interaction_distance=args.interaction_distance,
            **cap_kwargs(policy, cap_multiplier))
    return obj


def _paired_from_report(rep: Dict) -> Optional[Dict[str, float]]:
    """Paired sign/Wilcoxon stats from one fidelity_report's raw scalars."""
    pools = {"sim": rep["closest_sim_raw"], "real": rep["closest_real_raw"]}
    return _paired_stats(pools, "sim", "real")


def fit_policy(args, encs: List[Encounter], policy: str, cap_multiplier: float):
    """Calibrate (sigma, v0) on all encounters under one cap policy."""
    def obj(s, v):
        return objective_rollout_ade(
            encs, s, v, interaction_distance=args.interaction_distance,
            **cap_kwargs(policy, cap_multiplier))
    return calibrate(obj, args.sigma_grid, args.v0_grid, refine=not args.no_refine)


def pooled_row(args, encs: List[Encounter], policy: str,
               cap_multiplier: float) -> Dict:
    """One cap_pooled.csv row: fit + within-policy 3-arm fidelity + paired stats."""
    result = fit_policy(args, encs, policy, cap_multiplier)
    kw = cap_kwargs(policy, cap_multiplier)
    rep_cal = fidelity_report(encs, result.sigma, result.v0, **kw)
    rep_avec = fidelity_report(encs, *AVEC_DEFAULT, **kw)
    rep_norep = fidelity_report(encs, *NO_REPULSION, **kw)
    p_cal = _paired_from_report(rep_cal)
    p_avec = _paired_from_report(rep_avec)
    p_norep = _paired_from_report(rep_norep)
    row = {
        "policy": policy, "cap_multiplier": float(cap_multiplier),
        "cap_headroom": effective_headroom(policy, cap_multiplier),
        "sigma": result.sigma, "v0": result.v0, "fit_loss": result.loss,
        "refined": result.refined,
        "ade_calibrated": rep_cal["rollout_ade"],
        "ade_avec": rep_avec["rollout_ade"],
        "ade_norep": rep_norep["rollout_ade"],
        "mean_closest_sim": rep_cal["mean_closest_sim"],
        "mean_closest_real": rep_cal["mean_closest_real"],
    }
    for tag, p in (("calibrated", p_cal), ("avec", p_avec), ("norep", p_norep)):
        row[f"gap_{tag}"] = p["mean_gap"] if p else float("nan")
        row[f"n_real_gt_sim_{tag}"] = p["n_real_gt_sim"] if p else float("nan")
        row[f"sign_p_{tag}"] = p["sign_p"] if p else float("nan")
    row["n_pairs"] = p_cal["n_pairs"] if p_cal else float("nan")
    row["wilcoxon_p_calibrated"] = p_cal["wilcoxon_p"] if p_cal else float("nan")
    row["obj_interaction_distance"] = args.interaction_distance
    row["scenario"] = args.scenario
    return row


def sweep_capfit_m(args, encs: List[Encounter]) -> pd.DataFrame:
    """capfit headroom sweep (pooled): one fit + fidelity per m in --m-grid.

    m* is picked by the pooled FIT loss; the held-out verdict comes from the
    LOCO run at m* (never from this table -- double-dipping guard). m=1.0 is
    always worth including: it reproduces the median policy bit-for-bit BY
    CONSTRUCTION (m=1 is aliased to the median path inside _apply_cap_policy,
    because the shim's 1.0*cruise is not bit-identical to the state setter's
    multiplier*(cruise/multiplier) round trip; regression anchor in
    tests/test_cap_policies.py).
    """
    rows = []
    for m in args.m_grid:
        result = fit_policy(args, encs, "capfit", m)
        rep = fidelity_report(encs, result.sigma, result.v0,
                              **cap_kwargs("capfit", m))
        p = _paired_from_report(rep)
        rows.append({
            "m": float(m), "sigma": result.sigma, "v0": result.v0,
            "fit_loss": result.loss, "refined": result.refined,
            "ade": rep["rollout_ade"],
            "gap": p["mean_gap"] if p else float("nan"),
            "n_pairs": p["n_pairs"] if p else float("nan"),
            "n_real_gt_sim": p["n_real_gt_sim"] if p else float("nan"),
            "sign_p": p["sign_p"] if p else float("nan"),
            "obj_interaction_distance": args.interaction_distance,
            "scenario": args.scenario,
        })
        print(f"  capfit m={m:<5} sigma={result.sigma:.3f} v0={result.v0:.3f} "
              f"fit_loss={result.loss:.4f} gap={rows[-1]['gap']:+.3f} m")
    return pd.DataFrame(rows, columns=M_PROFILE_COLUMNS)


def _json_float(x) -> Optional[float]:
    """None for non-finite floats: json.dumps would emit a non-standard NaN
    token that strict parsers reject; null round-trips everywhere."""
    x = float(x)
    return x if np.isfinite(x) else None


def aux_paired_sidecar_tests(pools: Dict[str, list], protocol: str, *,
                             family: str, prefix: str, extra: Dict,
                             note: str,
                             control_note: Optional[str] = None) -> List[Dict]:
    """Auxiliary-only ledger records for one arm-triple of paired held-out tests.

    Deliberately NOT run_rq2_evaluation.headline_tests: reusing it would emit
    the canonical test_ids (rq2.*) / families (rq2_fidelity_paired_*) a second
    time and inflate the committed families' sizes in the ledger. Records built
    here live in their own namespace (``prefix`` = rq2cap.* / rq2dm.*) and are
    ALL auxiliary=True: they diagnose the measurement instrument, they are not
    confirmatory research claims, so they never join the canonical study-wide
    correction (ledger policy: no p-value bypasses the ledger; auxiliary is the
    honest tier for diagnostics). ``extra`` fields (e.g. cap_policy, weights)
    are copied into every record for provenance. ``control_note`` (optional) is
    appended to the CONTROL arms only -- e.g. the distmatch sidecars' controls
    are bitwise identical across configs (cap_policy fixed), so their family
    row count overstates the distinct hypotheses and must say so.
    """
    tests: List[Dict] = []
    for arm, key in (("calibrated", "calibrated_closest"),
                     ("avec_default", "default_closest"),
                     ("no_repulsion", "norepulsion_closest")):
        s = _paired_stats(pools, key, "real_closest")
        if s is None:
            continue
        arm_note = note
        if control_note and arm != "calibrated":
            arm_note = f"{note}; {control_note}"
        tests.append({
            "test_id": f"{prefix}.closest_sign.{arm}",
            "description": (f"(diagnostic) paired per-encounter sign test: real "
                            f"vs {arm} sim closest-approach ({protocol})"),
            "family": family,
            "protocol": protocol,
            "auxiliary": True,
            "p_value": _json_float(s["sign_p"]),
            "statistic": float(s["n_real_gt_sim"]),
            "sidedness": "two-sided",
            "n_pairs": s["n_pairs"],
            "n_real_gt_sim": s["n_real_gt_sim"],
            "mean_gap_m": _json_float(s["mean_gap"]),
            "wilcoxon_p": _json_float(s["wilcoxon_p"]),
            "headline": False,
            "note": arm_note,
            **extra,
        })
    return tests


def cap_sidecar_tests(pools: Dict[str, list], protocol: str, policy: str,
                      cap_multiplier: float,
                      interaction_distance: Optional[float] = None,
                      scenario: str = "all") -> List[Dict]:
    """Auxiliary sidecar records for one cap policy (rq2cap.* namespace)."""
    tests = aux_paired_sidecar_tests(
        pools, protocol,
        family=f"rq2_cap_sensitivity_{protocol}",
        prefix=f"rq2cap.{protocol}.{policy}",
        extra={"cap_policy": policy, "cap_multiplier": float(cap_multiplier),
               "obj_interaction_distance": interaction_distance,
               "scenario": scenario},
        note=("instrument diagnosis (review F2 speed-cap regime), not a "
              "confirmatory claim; auxiliary => excluded from the canonical "
              "study-wide correction"),
        control_note=("the median policy's arms reproduce the canonical "
                      "rq2.loco values by design (full-scale preservation "
                      "proof), so they duplicate, not add, hypotheses"
                      if policy == "median" else None),
    )
    return tests


def write_loco_summary(path: Path, df: pd.DataFrame, pools: Dict[str, list],
                       policy: str, cap_multiplier: float) -> str:
    """Human-readable per-policy LOCO summary (paired stats are the evidence)."""
    lines = [
        f"RQ2 cap-policy sensitivity  policy={policy}  "
        f"cap_multiplier={cap_multiplier}  protocol=loco  folds={len(df)}",
        "=" * 72,
        "",
        "Regime (review F2): pysocialforce max_speeds is BOTH the DesiredForce",
        "target and the hard cap; see run_rq2_cap_sensitivity.py --help for the",
        "policy table. closedloop peds walk ~30% faster than recorded, so read",
        "its ADE as a regime probe, not a fit quality.",
        "",
        f"  sigma     : {_meanstd(df['sigma'])}",
        f"  v0        : {_meanstd(df['v0'])}",
        "",
        "Held-out test metrics (mean +/- std over folds), within-policy arms:",
        f"  calibrated     ADE : {_meanstd(df['test_ade'])}",
        f"  AVEC default   ADE : {_meanstd(df['base_default_test_ade'])}",
        f"  no-repulsion   ADE : {_meanstd(df['base_norepulsion_test_ade'])}",
        "",
        "Paired per-encounter fidelity tests (pooled held-out encounters;",
        "the valid RQ2 unit, review F5):",
        f"  calibrated     paired : {_paired_line(pools, 'calibrated_closest', 'real_closest')}",
        f"  AVEC default   paired : {_paired_line(pools, 'default_closest', 'real_closest')}",
        f"  no-repulsion   paired : {_paired_line(pools, 'norepulsion_closest', 'real_closest')}",
        "",
        "Honest standoff gap (pooled closest-approach, descriptive):",
        f"  {_standoff_gap(pools)}",
        "",
        "Verdict logic: examples/make_rq2_instrument_report.py compares this",
        "policy's gap / sign direction against the median policy's.",
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text)
    return text


def run_loco(args, clips, policy: str, cap_multiplier: float, out_dir: Path):
    """LOCO folds for one policy; CSV + summary + auxiliary sidecar.

    Output names are keyed by policy only, so re-running capfit with a
    different --cap-multiplier OVERWRITES the previous capfit files -- the m
    actually used is recorded inside the CSV/sidecar (cap_multiplier field);
    archive the directory before sweeping m at the LOCO level.
    """
    clip_encs = {id(c): encounters_from_clips([c], args.min_sep, args.min_len)
                 for c in clips}
    folds = make_folds(clips, "loco")
    objective_fn = make_objective(args, policy, cap_multiplier)
    fidelity_kwargs = cap_kwargs(policy, cap_multiplier)

    rows: List[Dict] = []
    pools = _empty_raw()
    for fold_name, train_clips, test_clips in folds:
        train_encs = [e for c in train_clips for e in clip_encs[id(c)]]
        test_encs = [e for c in test_clips for e in clip_encs[id(c)]]
        row, raw = evaluate_fold(fold_name, "loco", train_clips, test_clips,
                                 train_encs, test_encs, args,
                                 objective_fn=objective_fn,
                                 fidelity_kwargs=fidelity_kwargs)
        row["cap_policy"] = policy
        row["cap_multiplier"] = float(cap_multiplier)
        row["obj_interaction_distance"] = args.interaction_distance
        row["scenario"] = args.scenario
        rows.append(row)
        for k in RAW_KEYS:
            pools[k].extend(raw[k])
        print(f"  fold {fold_name:<22} sigma={row['sigma']:.3f} "
              f"v0={row['v0']:.3f} test_ade={row['test_ade']:.3f}")

    df = pd.DataFrame(rows, columns=LOCO_COLUMNS)
    csv_path = out_dir / f"folds_cap_{policy}_loco.csv"
    df.to_csv(csv_path, index=False)
    summary_path = out_dir / f"summary_cap_{policy}_loco.txt"
    text = write_loco_summary(summary_path, df, pools, policy, cap_multiplier)
    print("\n" + text)

    sidecar = out_dir / f"headline_tests_cap_{policy}_loco.json"
    sidecar.write_text(json.dumps({
        "source": f"RQ2-cap-{policy}-loco",
        "generated_by": "run_rq2_cap_sensitivity.py",
        "tests": cap_sidecar_tests(pools, "loco", policy, cap_multiplier,
                                   args.interaction_distance, args.scenario),
    }, indent=2) + "\n")
    print(f"saved per-fold CSV to {csv_path}")
    print(f"saved summary to {summary_path}")
    print(f"saved auxiliary sidecar to {sidecar}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="all",
                   choices=list(CAP_POLICIES) + ["all"],
                   help="cap policy to run ('all' only valid with --mode pooled)")
    p.add_argument("--mode", choices=["pooled", "loco"], default="pooled")
    p.add_argument("--m-grid", type=float, nargs="*",
                   default=[1.0, 1.15, 1.3, 1.5, 2.0],
                   help="capfit headroom sweep (pooled mode)")
    p.add_argument("--cap-multiplier", type=float, default=1.3,
                   help="capfit headroom for --mode loco (pick m* from the "
                        "pooled capfit_m_profile.csv first)")
    p.add_argument("--scenario", default="all", choices=VEHICLE_SCENARIOS + ["all"])
    p.add_argument("--root", default="datasets/vci_citr/data")
    p.add_argument("--fps", type=float, default=29.97)
    p.add_argument("--min-sep", type=float, default=8.0)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--interaction-distance", type=float, default=None,
                   help="restrict fitter ADE to peds approaching within this "
                        "distance [m] (None = canonical, all peds)")
    p.add_argument("--sigma-grid", type=float, nargs="*",
                   default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    p.add_argument("--v0-grid", type=float, nargs="*",
                   default=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0])
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--out", default="outputs/rq2_instrument_audit/cap")
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
    print(f"\nmode={args.mode}  policy={args.policy}  clips={len(clips)}  "
          f"encounters={len(encs)}  (UNCAPPED_SPEED={UNCAPPED_SPEED} m/s)")

    if args.mode == "loco":
        if args.policy == "all":
            raise SystemExit("--mode loco needs a single --policy "
                             "(run one process per policy)")
        run_loco(args, clips, args.policy, args.cap_multiplier, out_dir)
        return

    # pooled mode
    rows: List[Dict] = []
    if args.policy in ("capfit", "all"):
        print("=== capfit headroom sweep (pooled; m* by FIT loss only) ===")
        profile = sweep_capfit_m(args, encs)
        profile_path = out_dir / "capfit_m_profile.csv"
        profile.to_csv(profile_path, index=False)
        m_star = float(profile.loc[profile["fit_loss"].idxmin(), "m"])
        print(f"saved m profile to {profile_path}  (m*={m_star} by pooled fit "
              "loss; held-out verdict comes from the LOCO run at m*)")
    else:
        m_star = args.cap_multiplier

    policies = list(CAP_POLICIES) if args.policy == "all" else [args.policy]
    print("=== pooled per-policy calibration (within-policy 3-arm fidelity) ===")
    for policy in policies:
        m = m_star if policy == "capfit" else args.cap_multiplier
        row = pooled_row(args, encs, policy, m)
        rows.append(row)
        print(f"  {policy:<11} sigma={row['sigma']:.3f} v0={row['v0']:.3f} "
              f"ade={row['ade_calibrated']:.4f} gap={row['gap_calibrated']:+.3f} m "
              f"sign {row['n_real_gt_sim_calibrated']:.0f}/{row['n_pairs']:.0f}")

    df = pd.DataFrame(rows, columns=POOLED_COLUMNS)
    csv_name = "cap_pooled.csv" if args.policy == "all" else f"cap_pooled_{args.policy}.csv"
    csv_path = out_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nsaved pooled per-policy table to {csv_path}")


if __name__ == "__main__":
    main()
