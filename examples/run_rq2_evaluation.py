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
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))

# numba (via pysocialforce) and matplotlib emit copious DEBUG records on stderr;
# silence them as the benchmark CLIs do (run_statistical_benchmark.py).
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from loguru import logger  # noqa: E402

from src.calibration import calibrate  # noqa: E402
from src.core.metrics import compare_distributions_ks, ks_sample_imbalance  # noqa: E402
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


# The keys of the per-fold raw-scalar bundle pooled across folds for the
# well-powered KS (review C1). Held flat so main() can extend pools by key.
RAW_KEYS = ("real_closest", "calibrated_closest", "default_closest",
            "norepulsion_closest", "real_onset", "calibrated_onset",
            # per-encounter (independent) onset scalars for the VALID onset KS
            # alongside the per-ped diagnostic (review m3/point5)
            "real_onset_per_enc", "calibrated_onset_per_enc")


def _empty_raw() -> Dict[str, list]:
    return {k: [] for k in RAW_KEYS}


def evaluate_fold(
    fold_name: str,
    protocol: str,
    train_clips: List[ClipTracks],
    test_clips: List[ClipTracks],
    train_encs: List[Encounter],
    test_encs: List[Encounter],
    args,
    objective_fn=None,
    fidelity_kwargs=None,
) -> Tuple[Dict[str, float], Dict[str, list]]:
    """Calibrate on train, validate on held-out test; (CSV row, raw-scalar pool).

    train_ade uses ``fidelity_report``'s rollout_ade (unfiltered, every ped) so
    it shares a scale with test_ade -- NOT the fitter loss, which the
    --interaction-distance filter would put on a different (non-comparable) scale.

    The second return value carries this fold's held-out raw closest-approach /
    onset scalars (calibrated, AVEC-default, no-repulsion, and the shared real
    values) so main() can pool them across folds into a single KS (C1): the
    per-fold ``test_ks_closest`` is a degenerate n=1 statistic and must not be
    averaged as if it measured fidelity.

    ``objective_fn`` (optional) replaces the fitter objective: a callable
    ``(train_encs, sigma, v0) -> float``. ``fidelity_kwargs`` (optional) is
    passed through to every ``fidelity_report`` call -- the calibrated arm AND
    both baseline arms, so a cap-policy / cruise-fn override keeps the three
    arms within the same regime (within-regime controls). The defaults (None)
    reproduce the historical behaviour exactly; both hooks exist for the
    cap-policy / distribution-matching audits (run_rq2_cap_sensitivity.py,
    run_rq2_distmatch.py), which reuse this fold loop instead of forking it.
    """
    row = _nan_row(fold_name, protocol, train_clips, test_clips,
                   len(train_encs), len(test_encs))
    raw = _empty_raw()
    fidelity_kwargs = fidelity_kwargs or {}
    if not train_encs:
        logger.warning(f"fold {fold_name!r}: no training encounters, skipping calibration")
        return row, raw

    if objective_fn is None:
        def obj(s, v):
            return objective_rollout_ade(train_encs, s, v,
                                         interaction_distance=args.interaction_distance)
    else:
        def obj(s, v):
            return objective_fn(train_encs, s, v)

    try:
        result = calibrate(obj, args.sigma_grid, args.v0_grid, refine=not args.no_refine)
    except ValueError:
        logger.warning(f"fold {fold_name!r}: objective non-finite on the entire grid "
                       "(--interaction-distance too tight?); NaN row")
        return row, raw

    row["sigma"] = result.sigma
    row["v0"] = result.v0
    row["refined"] = result.refined
    row["train_ade"] = fidelity_report(
        train_encs, result.sigma, result.v0, **fidelity_kwargs)["rollout_ade"]

    if test_encs:
        te = fidelity_report(test_encs, result.sigma, result.v0, **fidelity_kwargs)
        bd = fidelity_report(test_encs, *AVEC_DEFAULT, **fidelity_kwargs)
        bn = fidelity_report(test_encs, *NO_REPULSION, **fidelity_kwargs)
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
        # real_* is parameter-independent (same recorded closest/onset for all
        # three arms), so take it once from the calibrated report.
        raw["real_closest"] = te["closest_real_raw"]
        raw["real_onset"] = te["onset_real_raw"]
        raw["calibrated_closest"] = te["closest_sim_raw"]
        raw["calibrated_onset"] = te["onset_sim_raw"]
        raw["default_closest"] = bd["closest_sim_raw"]
        raw["norepulsion_closest"] = bn["closest_sim_raw"]
        raw["real_onset_per_enc"] = te["onset_per_enc_real_raw"]
        raw["calibrated_onset_per_enc"] = te["onset_per_enc_sim_raw"]
    return row, raw


def _meanstd(series: pd.Series) -> str:
    """'mean +/- std [min, max] (n)' over finite entries; 'n/a' when none finite.

    Uses the SAMPLE std (ddof=1), matching every other report in the repo
    (run_da_poc, make_margin_report, run_statistical_benchmark); the previous
    ddof=0 understated the spread by ~13-15% at the small LOSO n. The bracketed
    [min, max] fold range is added because (review M3) the LOCO folds share ~96%
    of their training data, so the ddof=1 std is still a DESCRIPTIVE fold-to-fold
    spread, NOT a standard error -- the raw range makes the heterogeneity
    visible rather than hiding it behind a deceptively tight +/- number.
    """
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "n/a"
    if vals.size == 1:
        return f"{vals[0]:.3f} (n=1)"
    return (f"{vals.mean():.3f} +/- {vals.std(ddof=1):.3f} "
            f"[{vals.min():.3f}, {vals.max():.3f}] (n={vals.size})")


def speed_domain(clips: List[ClipTracks]) -> Dict[str, float]:
    """CITR vehicle speed percentiles [m/s] pooled over clips (RQ2 limitation #2)."""
    samples = [vehicle_speed_samples(c.veh) for c in clips if c.veh is not None]
    pooled = np.concatenate(samples) if samples else np.array([])
    if pooled.size == 0:
        return {}
    pct = np.percentile(pooled, [5, 50, 90, 95])
    return {"p5": pct[0], "p50": pct[1], "p90": pct[2], "p95": pct[3],
            "max": float(pooled.max()), "n": int(pooled.size)}


def _pooled_ks_stat(pools: Dict[str, list], sim_key: str,
                    real_key: str) -> Dict[str, float]:
    """One KS over ALL folds' pooled raw scalars; None if either side is empty.

    Returns {ks, p, n_sim, n_real}. Shared by the human summary (_pooled_ks) and
    the machine-readable headline-test sidecar so both report the same numbers.
    """
    sim = np.asarray(pools.get(sim_key, []), dtype=float)
    real = np.asarray(pools.get(real_key, []), dtype=float)
    sim = sim[np.isfinite(sim)]
    real = real[np.isfinite(real)]
    if sim.size == 0 or real.size == 0:
        return None
    ks, p = compare_distributions_ks(sim, real)
    return {"ks": float(ks), "p": float(p),
            "n_sim": int(sim.size), "n_real": int(real.size)}


def _pooled_ks(pools: Dict[str, list], sim_key: str, real_key: str) -> str:
    """One KS over ALL folds' pooled raw scalars, or 'n/a' if a side is empty."""
    s = _pooled_ks_stat(pools, sim_key, real_key)
    if s is None:
        return "n/a (empty pool)"
    imbalance = ks_sample_imbalance(s["n_sim"], s["n_real"])
    flag = f"  [{imbalance}]" if imbalance else ""
    return (f"{s['ks']:.3f} (p={s['p']:.3f}, "
            f"n_sim={s['n_sim']}, n_real={s['n_real']}){flag}")


def _paired_stats(pools: Dict[str, list], sim_key: str,
                  real_key: str) -> Dict[str, float]:
    """Per-encounter PAIRED tests over the pooled held-out scalars (review F5).

    ``pools[sim_key][i]`` and ``pools[real_key][i]`` come from the SAME
    encounter i (evaluate_fold copies both sides of one encounter list, main()
    extends the pools in fold order), so the valid unit is the per-encounter
    difference d_i = real_i - sim_i -- NOT two independent samples. Returns the
    sign test (direction only, distribution-free) and the Wilcoxon signed-rank
    test (magnitude-aware) over the paired differences, plus the mean gap.

    Non-finite values are dropped PAIRWISE (a one-sided isfinite filter would
    silently break the pairing). A length mismatch between the pools means the
    pairing was already broken upstream; refuse to fake a paired test (None).
    Zero differences carry no direction: the sign test drops them (n shrinks),
    and an all-zero d skips Wilcoxon (its zero_method cannot handle that case).
    """
    sim = np.asarray(pools.get(sim_key, []), dtype=float)
    real = np.asarray(pools.get(real_key, []), dtype=float)
    if sim.size == 0 or real.size == 0 or sim.size != real.size:
        return None
    keep = np.isfinite(sim) & np.isfinite(real)
    sim, real = sim[keep], real[keep]
    if sim.size == 0:
        return None
    d = real - sim
    n_gt = int(np.sum(d > 0))
    n_lt = int(np.sum(d < 0))
    n_eff = n_gt + n_lt
    sign_p = float(binomtest(n_gt, n_eff, 0.5).pvalue) if n_eff else float("nan")
    if np.any(d != 0.0):
        # Two-sided; exact for n<=50 without rank ties (deterministic either way).
        w = wilcoxon(d)
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    else:
        w_stat, w_p = float("nan"), float("nan")
    return {"n_pairs": int(d.size), "n_real_gt_sim": n_gt,
            "n_real_lt_sim": n_lt, "sign_p": sign_p,
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "mean_gap": float(d.mean())}


def _paired_line(pools: Dict[str, list], sim_key: str, real_key: str) -> str:
    """Human-readable one-liner for the paired per-encounter tests."""
    s = _paired_stats(pools, sim_key, real_key)
    if s is None:
        return "n/a (empty or unpaired pool)"
    return (f"real>sim {s['n_real_gt_sim']}/{s['n_pairs']}  "
            f"sign p={s['sign_p']:.2e}  Wilcoxon p={s['wilcoxon_p']:.2e}  "
            f"mean gap {s['mean_gap']:+.3f} m")


def headline_tests(pools: Dict[str, list], protocol: str) -> List[Dict]:
    """RQ2 fidelity records for the multiplicity ledger (closest-approach).

    The ONE headline fidelity hypothesis is "the calibrated sim does not
    reproduce the real held-out closest-approach". Review F5: sim_i and real_i
    come from the SAME encounter i (shared geometry, strongly dependent), so an
    independent two-sample KS is MISSPECIFIED here -- the valid unit is the
    per-encounter paired difference d_i = real_i - sim_i. The headline is
    therefore the paired SIGN test (24/26 real>sim under LOCO), with the
    Wilcoxon signed-rank over the same differences filed as a second family
    member (family ``rq2_fidelity_paired_{protocol}`` -- two readings of one
    dataset counted honestly, both survive trivially at p<<0.001). A small
    p means the ~0.68 m standoff gap is statistically real, not Monte-Carlo
    noise; the ledger applies BH/Holm like for every other RQ test.

    The AVEC-default and no-repulsion arms are usually CONTROLS, not separate
    hypotheses. Saturation is judged on the HEADLINE sign statistic: a control
    whose (n_real_gt_sim, sign_p) is numerically identical to the calibrated
    arm's adds no distinct directional evidence, so it is recorded in
    ``controls`` instead (its Wilcoxon/mean-gap values are kept there for
    transparency -- they can differ, the demotion is deliberately based on the
    headline statistic alone, mirroring the KS headline's saturation logic).
    A DE-saturated control (different sign statistic or p) is a genuinely
    distinct fidelity hypothesis and IS emitted as a family member. The family
    size is therefore DATA-DEPENDENT (2 + the de-saturated controls; e.g. 3 on
    the current pools, where no_repulsion de-saturates at 25/26) -- read m from
    the ledger's family_size, not from this docstring.

    The former pooled-KS headline is retained as an AUXILIARY diagnostic family
    (``rq2_fidelity_ks_{protocol}_diagnostic``, excluded from the canonical
    study-wide correction): its statistic is still a useful shape summary and
    keeps the record comparable with the earlier C1 headline, but its p is not
    to be read. (onset KS is autocorrelated/diagnostic and is handled separately
    in the summary, not here.)
    """
    cal = _paired_stats(pools, "calibrated_closest", "real_closest")
    if cal is None:
        return []
    fam = f"rq2_fidelity_paired_{protocol}"
    saturated = {}      # control arms numerically identical to calibrated
    extra_family = []   # de-saturated arms = distinct hypotheses
    for name, key in (("avec_default", "default_closest"),
                      ("no_repulsion", "norepulsion_closest")):
        s = _paired_stats(pools, key, "real_closest")
        if s is None:
            continue
        if (s["n_real_gt_sim"] == cal["n_real_gt_sim"]
                and abs(s["sign_p"] - cal["sign_p"]) <= 1e-12):
            saturated[name] = {"n_real_gt_sim": s["n_real_gt_sim"],
                               "n_pairs": s["n_pairs"], "sign_p": s["sign_p"],
                               "wilcoxon_p": s["wilcoxon_p"],
                               "mean_gap_m": s["mean_gap"]}
        else:
            extra_family.append({
                "test_id": f"rq2.{protocol}.closest_sign.{name}",
                "description": (f"Paired per-encounter sign test: real vs {name} "
                                f"sim closest-approach ({protocol})"),
                "family": fam, "protocol": protocol,
                "p_value": s["sign_p"],
                "statistic": float(s["n_real_gt_sim"]),
                "sidedness": "two-sided",
                "n_pairs": s["n_pairs"],
                "n_real_gt_sim": s["n_real_gt_sim"],
                "mean_gap_m": s["mean_gap"],
                "headline": False,
                "note": ("de-saturated (distinct from calibrated) => a separate "
                         "fidelity hypothesis, counted in the family"),
            })
    sign_rec = {
        "test_id": f"rq2.{protocol}.closest_sign.calibrated",
        "description": (f"Paired per-encounter sign test: real vs calibrated sim "
                        f"closest-approach ({protocol})"),
        "family": fam,
        "protocol": protocol,
        "p_value": cal["sign_p"],
        "statistic": float(cal["n_real_gt_sim"]),
        "sidedness": "two-sided",
        "n_pairs": cal["n_pairs"],
        "n_real_gt_sim": cal["n_real_gt_sim"],
        "n_real_lt_sim": cal["n_real_lt_sim"],
        "mean_gap_m": cal["mean_gap"],
        "headline": True,
        "note": ("small p => real standoff exceeds the calibrated sim's in "
                 "nearly every encounter (the ~0.68 m fidelity gap is "
                 "statistically real). Paired replacement of the misspecified "
                 "independent-sample pooled KS (review F5)."),
        "controls": saturated,
        "controls_note": ("listed control arms SATURATE at the same paired "
                          "statistic as calibrated despite different sim arrays "
                          "=> the per-encounter direction does not discriminate "
                          "repulsion strength (review C2 echo: weak "
                          "identifiability); these are controls, excluded from "
                          "the multiplicity family. A de-saturated arm would "
                          "instead appear as its own family test."),
    }
    wilcoxon_rec = {
        "test_id": f"rq2.{protocol}.closest_wilcoxon.calibrated",
        "description": (f"Paired per-encounter Wilcoxon signed-rank: real vs "
                        f"calibrated sim closest-approach ({protocol})"),
        "family": fam,
        "protocol": protocol,
        "p_value": cal["wilcoxon_p"],
        "statistic": cal["wilcoxon_stat"],
        "sidedness": "two-sided",
        "n_pairs": cal["n_pairs"],
        "mean_gap_m": cal["mean_gap"],
        "headline": False,
        "note": ("magnitude-aware robustness companion to the sign test over "
                 "the SAME paired differences; counted in the family as honest "
                 "accounting of two readings of one dataset (family size is "
                 "data-dependent: de-saturated control arms join the family -- "
                 "read m from the ledger's family_size)"),
    }
    ks_diags = []
    for name, key in (("calibrated", "calibrated_closest"),
                      ("avec_default", "default_closest"),
                      ("no_repulsion", "norepulsion_closest")):
        s = _pooled_ks_stat(pools, key, "real_closest")
        if s is None:
            continue
        ks_diags.append({
            "test_id": f"rq2.{protocol}.closest_ks.{name}",
            "description": (f"(diagnostic) pooled closest-approach KS: {name} "
                            f"sim vs real ({protocol})"),
            "family": f"rq2_fidelity_ks_{protocol}_diagnostic",
            "protocol": protocol,
            "auxiliary": True,
            "p_value": s["p"], "statistic": s["ks"], "sidedness": "two-sided",
            "n_sim": s["n_sim"], "n_real": s["n_real"], "headline": False,
            "note": ("DIAGNOSTIC ONLY (review F5): sim_i and real_i share "
                     "encounter i, so an independent two-sample KS is "
                     "misspecified on these paired pools -- read the statistic "
                     "as a shape summary, not the p. The valid headline is the "
                     "paired sign/Wilcoxon family."),
        })
    return [sign_rec, wilcoxon_rec] + extra_family + ks_diags


def _standoff_gap(pools: Dict[str, list]) -> str:
    """Honest pooled standoff gap (real - sim mean closest-approach), ADE-blind."""
    sim = np.asarray(pools.get("calibrated_closest", []), dtype=float)
    real = np.asarray(pools.get("real_closest", []), dtype=float)
    sim = sim[np.isfinite(sim)]
    real = real[np.isfinite(real)]
    if sim.size == 0 or real.size == 0:
        return "n/a"
    return (f"real {real.mean():.3f} m vs calibrated sim {sim.mean():.3f} m  "
            f"=> gap {real.mean() - sim.mean():+.3f} m (sim under-reproduces standoff)")


def write_summary(path: Path, df: pd.DataFrame, protocol: str,
                  speed: Dict[str, float], pools: Dict[str, list] = None) -> str:
    """Human-readable summary; returns the text (also printed to stdout)."""
    pools = pools or _empty_raw()
    # Honest framing of the stability claim (review M3): LOCO folds re-fit on
    # 25 of 26 clips, so their fitted (sigma, v0) are near-replicates (~96% shared
    # training) -- the +/- is a descriptive fold-to-fold spread, NOT a standard
    # error, and the proper "unseen geometry" stability stress test is LOSO.
    if protocol == "loco":
        stability_note = (
            "Calibrated (sigma, v0) fold-to-fold spread (LOCO):\n"
            "  NOTE: each fold re-fits on 25/26 clips (~96% shared training), so\n"
            "  the +/- is a DESCRIPTIVE spread, not a standard error. The proper\n"
            "  unseen-geometry stability check is LOSO (run --protocol loso).")
    else:
        stability_note = (
            "Calibrated (sigma, v0) across left-one-scenario-out folds (LOSO):\n"
            "  geometry-holdout stress test (only ~4 folds); the [min, max] range\n"
            "  shows how far (sigma, v0) moves when a whole interaction geometry is\n"
            "  withheld -- the honest stability headline.")
    lines = [
        f"RQ2 cross-validated evaluation  protocol={protocol}  folds={len(df)}",
        "=" * 72,
        "",
        stability_note,
        f"  sigma     : {_meanstd(df['sigma'])}",
        f"  v0        : {_meanstd(df['v0'])}",
        f"  AVEC default for reference: sigma={AVEC_DEFAULT[0]}, v0={AVEC_DEFAULT[1]}",
        "",
        "Held-out test metrics (mean +/- std over folds), 3 groups:",
        f"  calibrated     ADE : {_meanstd(df['test_ade'])}",
        f"  AVEC default   ADE : {_meanstd(df['base_default_test_ade'])}",
        f"  no-repulsion   ADE : {_meanstd(df['base_norepulsion_test_ade'])}",
        "",
        "Paired per-encounter fidelity test (review F5: sim_i and real_i come from",
        "the SAME held-out encounter i, so the valid unit is the per-encounter",
        "difference d_i = real_i - sim_i -- sign test + Wilcoxon signed-rank over",
        "all folds' pooled held-out encounters. This replaces the former pooled",
        "independent-sample KS headline, which is misspecified on paired pools.",
        "NOTE the asymmetry: the 'calibrated' pool mixes each fold's OWN",
        "(sigma, v0) -- a cross-validated mixture, not one fixed model -- while the",
        "AVEC-default / no-repulsion pools use a single fixed parameter set:",
        f"  calibrated     paired : {_paired_line(pools, 'calibrated_closest', 'real_closest')}",
        f"  AVEC default   paired : {_paired_line(pools, 'default_closest', 'real_closest')}",
        f"  no-repulsion   paired : {_paired_line(pools, 'norepulsion_closest', 'real_closest')}",
        "",
        "(diagnostic only) pooled independent-two-sample KS over the same pools --",
        "kept for comparability with the earlier C1 headline; its p is NOT valid",
        "on paired data (read the statistic as a shape summary only):",
        f"  calibrated     KS_closest : {_pooled_ks(pools, 'calibrated_closest', 'real_closest')}",
        f"  AVEC default   KS_closest : {_pooled_ks(pools, 'default_closest', 'real_closest')}",
        f"  no-repulsion   KS_closest : {_pooled_ks(pools, 'norepulsion_closest', 'real_closest')}",
        "  (diagnostic only) calibrated KS_onset pools per-PED onset distances,",
        "  which are autocorrelated within an encounter (shared ego trajectory),",
        "  so its p-value is anti-conservative -- read the statistic, not the p:",
        f"  calibrated     KS_onset   : {_pooled_ks(pools, 'calibrated_onset', 'real_onset')}",
        "  (valid) per-encounter onset KS uses ONE median onset per encounter =",
        "  an independent unit (clip-independent across folds), so its p IS a",
        "  usable two-sample p (review m3/point5) -- prefer this over the per-PED:",
        f"  calibrated     KS_onset_per_enc : "
        f"{_pooled_ks(pools, 'calibrated_onset_per_enc', 'real_onset_per_enc')}",
        "  NOTE: at this small n a KS near 1.0 means the two pools simply do not "
        "overlap",
        "  (complete separation), NOT infinitely-precise distinguishability -- read "
        "with n.",
        "",
        "Honest standoff gap (pooled closest-approach; the metric ADE is blind to;",
        "descriptive mean difference, no CI / significance test):",
        f"  {_standoff_gap(pools)}",
        "",
        "(diagnostic) degenerate per-fold KS mean -- ~1.000, not fidelity:",
        f"  calibrated per-fold KS_closest : {_meanstd(df['test_ks_closest'])}",
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
    pools = _empty_raw()
    for fold_name, train_clips, test_clips in folds:
        train_encs = [e for c in train_clips for e in clip_encs[id(c)]]
        test_encs = [e for c in test_clips for e in clip_encs[id(c)]]
        row, raw = evaluate_fold(fold_name, args.protocol, train_clips, test_clips,
                                 train_encs, test_encs, args)
        rows.append(row)
        for k in RAW_KEYS:  # pool held-out scalars across folds for the C1 KS
            pools[k].extend(raw[k])
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
    text = write_summary(summary_path, df, args.protocol, speed, pools)
    print("\n" + text)
    print(f"saved per-fold CSV to {csv_path}")
    print(f"saved summary to {summary_path}")

    # Machine-readable headline-test sidecar for the cross-RQ multiplicity ledger
    # (make_multiplicity_ledger.py). Deterministic: the p-values come from the
    # paired sign/Wilcoxon tests (and the diagnostic pooled KS) over fixed
    # held-out scalars, so re-running this script overwrites the file
    # byte-for-byte.
    sidecar = out_dir / f"headline_tests_{args.protocol}.json"
    sidecar.write_text(json.dumps({
        "source": f"RQ2-{args.protocol}",
        "generated_by": "run_rq2_evaluation.py",
        "tests": headline_tests(pools, args.protocol),
    }, indent=2) + "\n")
    print(f"saved headline-test sidecar to {sidecar}")


if __name__ == "__main__":
    main()
