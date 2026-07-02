#!/usr/bin/env python3
"""Assemble the RQ2 instrument-audit REPORT (cap policies x distribution matching).

Reads the CSVs / sidecars produced by run_rq2_cap_sensitivity.py,
run_rq2_distmatch.py and run_rq2_surfaces.py under outputs/rq2_instrument_audit/
and writes REPORT.md with DATA-DRIVEN verdicts (every judgement is a pure
function over the tables; the prose is generated from the same numbers it
quotes, mirroring the RQ1b report discipline):

* cap_verdict     -- is the standoff under-reproduction (+0.68 m, 24/26) a
                     harness artifact of the speed-cap regime (review F2) or a
                     structural SFM limit? Judged on the DECOUPLED policies
                     (uncapped / capfit) only: the closedloop arm carries a
                     documented ~30% walking-speed confound and is reported as
                     a regime probe, never as verdict evidence.
* distmatch_verdict -- does adding the closest-approach EMD term to the fitter
                     (a) restore identifiability (narrower 2%-band), (b) improve
                     standoff reproduction, and at (c) what held-out ADE cost?
* f1_verdict      -- the overall question (review F1): does ANY audited
                     configuration make the calibrated parameters beat the
                     AVEC hand-tuned ones on held-out evidence?
* rq1b_domain_check -- are new candidate calibration points inside the RQ1b
                     sensitivity box / LOSO envelope? (Points outside mean the
                     committed RQ1b sweep does not cover them; the report flags
                     this for a research decision, it never re-runs RQ1b.)

The report is byte-stable on re-run (no timestamps; sorted globs).

Input contract: the pooled cap table is read from ``cap/cap_pooled.csv`` (the
``--policy all`` output) only -- single-policy pooled files
(``cap_pooled_{policy}.csv``) are not aggregated. Verdict-relevant rows are
guarded by recorded provenance: non-default fit flags, non-median distmatch
regimes and unknown capfit multipliers are excluded WITH a printed warning /
report note, never silently mixed in.

Usage:
    .venv/bin/python examples/make_rq2_instrument_report.py
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# thresholds (all quoted verbatim in the REPORT so the verdicts are auditable)
# ---------------------------------------------------------------------------
SHRINK_THRESHOLD = 0.25   # |gap| must shrink >=25% vs median to count
DOMINANCE_ALPHA = 0.05    # sign-test p above this = direction dominance broken
ADE_BEAT_MARGIN = 0.02    # calibrated must beat AVEC ADE by >=2% (F1)
GAP_BEAT_MARGIN = 0.10    # ... and |gap| by >=10% (F1)
BAND_RESTORE_FACTOR = 0.5  # 2%-band width must at least halve vs the ADE fitter

# RQ1b sweep domain (run_rq1b_sensitivity.py GT arms; see review F3/M6).
RQ1B_BOX = {"sigma": (1.040, 1.272), "v0": (1.542, 1.820)}
RQ1B_LOSO_PTS = [(1.085, 2.617), (0.743, 1.849)]
DOMAIN_TOL = 0.05  # relative closeness to a LOSO envelope point


# ---------------------------------------------------------------------------
# pure verdict functions (unit-tested on synthetic frames)
# ---------------------------------------------------------------------------
def rq1b_domain_check(sigma: float, v0: float,
                      box: Dict = RQ1B_BOX,
                      loso_pts: List = RQ1B_LOSO_PTS,
                      tol: float = DOMAIN_TOL) -> Dict:
    """Classify one (sigma, v0) against the RQ1b sweep domain.

    'inside_box' -> covered by the committed +/-1SD sweep; 'near_loso_envelope'
    -> within ``tol`` (relative) of a LOSO envelope arm; 'outside' -> the
    committed RQ1b sweep does not cover it (research decision required before
    citing RQ1b robustness for this point -- the report only flags it).
    """
    out = {"sigma": float(sigma), "v0": float(v0)}
    if not (np.isfinite(sigma) and np.isfinite(v0)):
        out["status"] = "undefined"
        return out
    if box["sigma"][0] <= sigma <= box["sigma"][1] \
            and box["v0"][0] <= v0 <= box["v0"][1]:
        out["status"] = "inside_box"
        return out
    for s, v in loso_pts:
        if abs(sigma - s) <= tol * abs(s) and abs(v0 - v) <= tol * abs(v):
            out["status"] = "near_loso_envelope"
            return out
    out["status"] = "outside"
    return out


def _dominance_broken(sign_p: float, wilcoxon_p: float,
                      alpha: float = DOMINANCE_ALPHA) -> bool:
    """True when NEITHER paired test still detects the real>sim direction.

    The RQ2 canonical family reads the sign test AND the Wilcoxon signed-rank
    as a pair; declaring "dominance broke" from the sign test's null acceptance
    alone (n=26, low power) while the magnitude-aware Wilcoxon on the same
    differences is still significant would be a one-winged claim (review
    finding: closedloop's AVEC arm is exactly that profile, sign p=0.076 /
    Wilcoxon p=0.002). A missing/NaN Wilcoxon degrades gracefully to the
    sign-only reading.
    """
    sign_broken = bool((not np.isfinite(sign_p)) or sign_p > alpha)
    wilcoxon_broken = bool((not np.isfinite(wilcoxon_p)) or wilcoxon_p > alpha)
    return sign_broken and wilcoxon_broken


def _sign_flipped(gap: float, ref_gap: float) -> bool:
    """True when the gap changed SIDE vs the reference (under- -> over-
    reproduction or vice versa). A flipped gap passing the |gap| shrink test
    means the systematic error was REPLACED by an opposite one, not explained
    away, so it must never feed an artifact/improvement claim on its own
    (review finding); it is flagged for human judgement instead.
    """
    return bool(np.isfinite(gap) and np.isfinite(ref_gap)
                and gap * ref_gap < 0.0)


def cap_verdict(df: pd.DataFrame,
                shrink_threshold: float = SHRINK_THRESHOLD,
                alpha: float = DOMINANCE_ALPHA,
                verdict_policies=("uncapped", "capfit")) -> Dict:
    """Harness-artifact vs structural-limit judgement over the LOCO cap table.

    ``df`` columns: policy, mean_gap, sign_p, n_real_gt_sim, n_pairs (the
    CALIBRATED arm of each policy's pooled held-out paired test; optional
    wilcoxon_p and cap_multiplier columns refine the judgement). The standoff
    gap counts as explained-by-the-cap only if, under a DECOUPLED policy, (a)
    |mean_gap| shrinks >= ``shrink_threshold`` vs the median policy WITHOUT
    flipping sign, AND (b) the real>sim direction dominance breaks on BOTH
    paired tests (:func:`_dominance_broken`). closedloop is evaluated too but
    EXCLUDED from the verdict: its target speed is 1.3x the recorded cruise,
    so a gap change there mixes the cap effect with a walking-speed error. A
    capfit row is ineligible when cap_multiplier == 1.0 (median by aliasing)
    or when it is missing/NaN (unknown provenance -- fail closed).

    Eligible rows that shrink the gap but keep direction dominance land in
    ``partially_explained``: the binary verdict stays structural_limit, but
    the report surfaces that the cap accounts for part of the magnitude.

    Known scope limit (disclosed in the report): the judgement reads each
    policy's RE-FITTED calibrated arm, and the ADE fitter is nearly v0-blind
    (C2), so "the gap does not shrink" could in principle reflect the fitter
    not chasing the gap rather than the model being unable to. The within-
    regime AVEC control (fixed strong repulsion, no re-fit) is reported next
    to it as the cross-check: if the gap persisted merely because the fitter
    stayed weak, the strong-repulsion control would close it.
    """
    med = df[df["policy"] == "median"]
    if med.empty:
        return {"verdict": "undetermined", "reason": "median LOCO row missing",
                "per_policy": {}, "artifact_hits": [], "confounded_hits": [],
                "partially_explained": []}
    med_gap = float(med["mean_gap"].iloc[0])
    per_policy: Dict[str, Dict] = {}
    artifact_hits: List[str] = []
    confounded_hits: List[str] = []
    partially_explained: List[str] = []
    measured_decoupled = False
    for _, r in df.iterrows():
        if r["policy"] == "median":
            continue
        gap = float(r["mean_gap"])
        sign_p = float(r["sign_p"])
        wilcoxon_p = float(r.get("wilcoxon_p", float("nan")))
        flipped = _sign_flipped(gap, med_gap)
        gap_shrunk = bool(abs(gap) <= (1.0 - shrink_threshold) * abs(med_gap)
                          and not flipped)
        dominance_broken = _dominance_broken(sign_p, wilcoxon_p, alpha)
        m = float(r.get("cap_multiplier", float("nan")))
        if r["policy"] not in verdict_policies:
            eligible, why = False, "confounded regime probe (speed confound)"
        elif r["policy"] == "capfit" and not np.isfinite(m):
            eligible, why = False, "capfit with unknown cap_multiplier (fail closed)"
        elif r["policy"] == "capfit" and m == 1.0:
            eligible, why = False, "capfit m=1.0 is the median path by aliasing"
        else:
            eligible, why = True, ""
        per_policy[str(r["policy"])] = {
            "mean_gap": gap, "median_gap": med_gap,
            "gap_change_pct": (100.0 * (abs(gap) - abs(med_gap)) / abs(med_gap)
                               if med_gap != 0 else float("nan")),
            "sign_p": sign_p,
            "wilcoxon_p": wilcoxon_p,
            "n_real_gt_sim": int(r["n_real_gt_sim"]),
            "n_pairs": int(r["n_pairs"]),
            "gap_shrunk": gap_shrunk,
            "sign_flipped": flipped,
            "dominance_broken": dominance_broken,
            "verdict_eligible": eligible,
            "ineligible_reason": why,
        }
        if eligible:
            measured_decoupled = True
            if gap_shrunk and dominance_broken:
                artifact_hits.append(str(r["policy"]))
            elif gap_shrunk:
                partially_explained.append(str(r["policy"]))
        elif gap_shrunk and dominance_broken:
            confounded_hits.append(str(r["policy"]))
    if not measured_decoupled:
        verdict = "undetermined"
        reason = "no decoupled policy (uncapped/capfit m!=1) measured under LOCO"
    elif artifact_hits:
        verdict = "harness_artifact"
        reason = (f"decoupled polic{'ies' if len(artifact_hits) > 1 else 'y'} "
                  f"{', '.join(artifact_hits)}: gap shrank >= "
                  f"{shrink_threshold:.0%} (same sign) AND both paired tests "
                  "lost the real>sim direction")
    else:
        verdict = "structural_limit"
        reason = ("no decoupled policy shrinks the gap "
                  f">= {shrink_threshold:.0%} with broken direction dominance")
    return {"verdict": verdict, "reason": reason, "per_policy": per_policy,
            "artifact_hits": artifact_hits, "confounded_hits": confounded_hits,
            "partially_explained": partially_explained, "median_gap": med_gap}


def distmatch_verdict(df: pd.DataFrame,
                      shrink_threshold: float = SHRINK_THRESHOLD,
                      alpha: float = DOMINANCE_ALPHA) -> Dict:
    """Standoff-vs-ADE trade-off judgement over the LOCO distmatch table.

    ``df`` columns: config, mean_gap, sign_p, n_real_gt_sim, n_pairs, test_ade
    (fold-mean held-out ADE of the calibrated arm), plus the reference row
    config='w0' (bit-identical to the canonical ADE fitter). Judgements per
    config: gap change vs w0, ADE sacrifice vs w0, dominance broken.
    ``standoff_improved`` = any non-w0 config with |gap| shrunk >=
    ``shrink_threshold`` WITHOUT flipping sign (:func:`_sign_flipped`);
    identifiability is judged separately from the surfaces table
    (identifiability_summary). With only the w0 reference present (e.g. the
    fallback synthesised from the cap median run) there is nothing audited yet,
    so the verdict is undetermined -- NOT "no improvement" (review finding: a
    partial run must not read as a completed negative result).
    """
    ref = df[df["config"] == "w0"]
    if ref.empty:
        return {"verdict": "undetermined", "reason": "w0 reference row missing",
                "per_config": {}}
    if df[df["config"] != "w0"].empty:
        return {"verdict": "undetermined",
                "reason": "no non-w0 distmatch configs measured under LOCO",
                "per_config": {}}
    ref_gap = float(ref["mean_gap"].iloc[0])
    ref_ade = (float(ref["test_ade"].iloc[0])
               if "test_ade" in ref.columns else float("nan"))
    per_config: Dict[str, Dict] = {}
    improved: List[str] = []
    for _, r in df.iterrows():
        if r["config"] == "w0":
            continue
        gap = float(r["mean_gap"])
        ade = float(r.get("test_ade", float("nan")))
        flipped = _sign_flipped(gap, ref_gap)
        gap_shrunk = bool(abs(gap) <= (1.0 - shrink_threshold) * abs(ref_gap)
                          and not flipped)
        wilcoxon_p = float(r.get("wilcoxon_p", float("nan")))
        per_config[str(r["config"])] = {
            "mean_gap": gap, "ref_gap": ref_gap,
            "gap_change_pct": (100.0 * (abs(gap) - abs(ref_gap)) / abs(ref_gap)
                               if ref_gap != 0 else float("nan")),
            "test_ade": ade, "ref_ade": ref_ade,
            "ade_sacrifice_pct": (100.0 * (ade - ref_ade) / ref_ade
                                  if ref_ade != 0 else float("nan")),
            "sign_p": float(r["sign_p"]),
            "wilcoxon_p": wilcoxon_p,
            "n_real_gt_sim": int(r["n_real_gt_sim"]),
            "n_pairs": int(r["n_pairs"]),
            "gap_shrunk": gap_shrunk,
            "sign_flipped": flipped,
            "dominance_broken": _dominance_broken(float(r["sign_p"]),
                                                  wilcoxon_p, alpha),
        }
        if gap_shrunk:
            improved.append(str(r["config"]))
    return {
        "verdict": "standoff_improved" if improved else "no_standoff_improvement",
        "improved_configs": improved,
        "per_config": per_config,
        "ref_gap": ref_gap, "ref_ade": ref_ade,
    }


def identifiability_summary(df: pd.DataFrame,
                            restore_factor: float = BAND_RESTORE_FACTOR) -> Dict:
    """Band-width comparison per (policy, axis): distribution objectives vs ADE.

    ``df`` = identifiability.csv. For each policy and axis, the ADE fitter's
    2%-band width is the reference; a distribution objective 'restores'
    identifiability on that axis only if ALL of:

    * its band width is <= ``restore_factor`` x the ADE width,
    * its own band is not censored (a censored band hugs the grid edge: the
      profile minimum lies outside the evaluated grid — a degenerate direction
      such as the EMD term's v0 -> inf preference, not identification),
    * its own band is contiguous (a multi-modal profile's hull straddles an
      out-of-band ridge, so its width is not a flat region),
    * its own fitted value lies inside its own band (a "sharp" band that does
      not contain the refined optimum does not describe the optimum),
    * the OTHER axis' fitted optimum is not on/over the grid edge for the same
      (objective, policy) surface — profile_band slices each axis at the grid
      node nearest the fitted value of the other axis; when that fitted value
      ran off the grid (e.g. fitted v0 clamped to the v0 = 8 node), a "sharp"
      band on this axis describes a conditional slice FAR from the actual
      optimum, so quoting it as restored identifiability would be false. A
      MISSING other-axis row means the clamp cannot be checked; fail closed.
    * the ADE REFERENCE row passes the same slice-quality bar (not censored,
      other axis not on edge): a degraded reference makes the <= factor
      comparison meaningless, so the whole entry is flagged
      ``reference_degraded`` and cannot grant restoration.

    Per-objective flags exported for the report: ``censored``,
    ``other_axis_edge``, ``fitted_in_band``, ``noncontiguous`` and
    ``single_node`` (band spans one grid node: its width reads 0 but only
    means "below grid spacing").
    """
    out: Dict[str, Dict] = {}
    restored_any = False
    other = {"sigma": "v0", "v0": "sigma"}

    def _other_edge(objective: str, policy: str, axis: str) -> bool:
        rows = df[(df["policy"] == policy) & (df["objective"] == objective)
                  & (df["axis"] == other[axis])]
        if rows.empty:  # cannot verify the slice anchor -> fail closed
            return True
        return bool(rows["fitted_on_grid_edge"].iloc[0])

    def _noncontiguous(row) -> bool:
        if "band_contiguous" not in row:
            return False  # legacy CSV without the column: no evidence either way
        return not bool(row["band_contiguous"])

    for (policy, axis), g in df.groupby(["policy", "axis"], sort=True):
        ref = g[g["objective"] == "ade"]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        ref_w = float(ref_row["band_width"])
        ref_cens = bool(ref_row["censored_lo"] or ref_row["censored_hi"])
        reference_degraded = bool(ref_cens or _noncontiguous(ref_row)
                                  or _other_edge("ade", policy, axis))
        entry = {"ade_band_width": ref_w, "ade_censored": ref_cens,
                 "reference_degraded": reference_degraded, "objectives": {}}
        for _, r in g[g["objective"] != "ade"].iterrows():
            w = float(r["band_width"])
            cens = bool(r["censored_lo"] or r["censored_hi"])
            other_edge = _other_edge(str(r["objective"]), policy, axis)
            fitted_in_band = bool(np.isfinite(r["band_lo"])
                                  and r["band_lo"] <= r["fitted"] <= r["band_hi"])
            n_nodes = float(r.get("n_nodes_in_band", float("nan")))
            single_node = bool(np.isfinite(n_nodes) and int(n_nodes) == 1)
            noncontig = _noncontiguous(r)
            restored = bool(np.isfinite(w) and np.isfinite(ref_w) and ref_w > 0
                            and w <= restore_factor * ref_w
                            and not cens and not noncontig
                            and fitted_in_band
                            and not other_edge
                            and not reference_degraded)
            entry["objectives"][str(r["objective"])] = {
                "band_width": w, "censored": cens,
                "other_axis_edge": other_edge,
                "fitted_in_band": fitted_in_band,
                "single_node": single_node,
                "noncontiguous": noncontig,
                "restored": restored,
            }
            restored_any = restored_any or restored
        out[f"{policy}/{axis}"] = entry
    return {"per_policy_axis": out, "restored_any": restored_any,
            "restore_factor": restore_factor}


def f1_verdict(rows: List[Dict],
               ade_margin: float = ADE_BEAT_MARGIN,
               gap_margin: float = GAP_BEAT_MARGIN) -> Dict:
    """Does ANY audited configuration beat the AVEC hand-tuned parameters?

    ``rows``: one dict per audited configuration with keys
    {label, ade_calibrated, ade_avec, gap_calibrated, gap_avec} -- all held-out
    (LOCO) within the SAME regime (within-policy controls). 'Beats' = held-out
    ADE better by >= ``ade_margin`` (relative) AND |standoff gap| better by >=
    ``gap_margin`` (relative). One-sided improvements are listed separately
    (partial wins do not resolve F1: the F1 finding was a TIE, so a tie again
    means the negative result stands). With NO rows at all the verdict is
    undetermined — "f1_stands" is a claim about audited configurations, and an
    empty audit must not read as a completed negative result (review finding).
    Both references must be positive finite: an all-zero degenerate row would
    otherwise satisfy both <= comparisons and fabricate the audit's strongest
    positive claim (review finding).
    """
    if not rows:
        return {"verdict": "undetermined", "beats": [], "partial": [],
                "per_label": {}, "ade_margin": ade_margin,
                "gap_margin": gap_margin}
    beats: List[str] = []
    partial: List[str] = []
    per_label: Dict[str, Dict] = {}
    for r in rows:
        ade_c, ade_a = float(r["ade_calibrated"]), float(r["ade_avec"])
        gap_c, gap_a = float(r["gap_calibrated"]), float(r["gap_avec"])
        ade_better = bool(np.isfinite(ade_c) and np.isfinite(ade_a)
                          and ade_a > 0.0
                          and ade_c <= (1.0 - ade_margin) * ade_a)
        gap_better = bool(np.isfinite(gap_c) and np.isfinite(gap_a)
                          and abs(gap_a) > 0.0
                          and abs(gap_c) <= (1.0 - gap_margin) * abs(gap_a))
        per_label[str(r["label"])] = {
            "ade_calibrated": ade_c, "ade_avec": ade_a, "ade_better": ade_better,
            "gap_calibrated": gap_c, "gap_avec": gap_a, "gap_better": gap_better,
        }
        if ade_better and gap_better:
            beats.append(str(r["label"]))
        elif ade_better or gap_better:
            partial.append(str(r["label"]))
    return {
        "verdict": "calibration_beats_hand_tuning" if beats else "f1_stands",
        "beats": beats, "partial": partial, "per_label": per_label,
        "ade_margin": ade_margin, "gap_margin": gap_margin,
    }


# ---------------------------------------------------------------------------
# IO assembly (tolerant to missing pieces; reports what is absent)
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _sidecar_calibrated_stats(path: Path) -> Optional[Dict]:
    """Extract the calibrated + avec paired records from one sidecar."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    out = {}
    for t in data.get("tests", []):
        tid = t.get("test_id", "")
        if tid.endswith(".closest_sign.calibrated"):
            out["calibrated"] = t
        elif tid.endswith(".closest_sign.avec_default"):
            out["avec_default"] = t
    return out or None


def _f(x) -> float:
    """None/missing-safe float (sidecars store non-finite values as null)."""
    return float("nan") if x is None else float(x)


def _nondefault_provenance(rec: Dict) -> bool:
    """Fit-affecting flags departing from the canonical run's defaults.

    Legacy sidecars without the provenance fields read as default (they were
    all produced by default runs); a recorded non-None interaction distance or
    a scenario subset marks the run as NOT comparable to the canonical
    reference (review finding: such runs used to be indistinguishable).
    """
    return bool(rec.get("obj_interaction_distance") is not None
                or rec.get("scenario", "all") not in (None, "all"))


def load_cap_loco(cap_dir: Path) -> pd.DataFrame:
    """One row per policy with LOCO paired stats + fold-mean ADEs."""
    rows = []
    for sidecar in sorted(cap_dir.glob("headline_tests_cap_*_loco.json")):
        policy = sidecar.name[len("headline_tests_cap_"):-len("_loco.json")]
        stats = _sidecar_calibrated_stats(sidecar)
        folds = _read_csv(cap_dir / f"folds_cap_{policy}_loco.csv")
        if stats is None or "calibrated" not in stats:
            print(f"WARNING: {sidecar} has no calibrated record; "
                  f"policy {policy!r} row DROPPED from the cap table",
                  file=sys.stderr)
            continue
        cal = stats["calibrated"]
        avec = stats.get("avec_default", {})
        row = {
            "policy": policy,
            "mean_gap": _f(cal["mean_gap_m"]), "sign_p": _f(cal["p_value"]),
            "wilcoxon_p": _f(cal.get("wilcoxon_p")),
            "n_real_gt_sim": cal["n_real_gt_sim"], "n_pairs": cal["n_pairs"],
            "gap_avec": _f(avec.get("mean_gap_m")),
            # capfit eligibility guard in cap_verdict (m=1.0 = median alias)
            "cap_multiplier": _f(cal.get("cap_multiplier")),
            "nondefault_provenance": _nondefault_provenance(cal),
        }
        if folds is not None:
            row["sigma_mean"] = float(folds["sigma"].mean())
            row["v0_mean"] = float(folds["v0"].mean())
            row["v0_min"] = float(folds["v0"].min())
            row["v0_max"] = float(folds["v0"].max())
            row["test_ade"] = float(folds["test_ade"].mean())
            row["ade_avec"] = float(folds["base_default_test_ade"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def load_dm_loco(dm_dir: Path, w0_fallback: Optional[Dict] = None) -> pd.DataFrame:
    """One row per distmatch config; w0 reference appended from the cap median
    run when no explicit w0 LOCO was run (they are bit-identical by design).

    The sidecar-recorded cap_policy / dist_metric ride along: the w0 fallback
    is a MEDIAN-regime reference, so a config run under another cap policy is
    not comparable to it and build_report excludes it from the verdict with an
    explicit note (review finding: the mix used to be silent).
    """
    rows = []
    for sidecar in sorted(dm_dir.glob("headline_tests_dm_*_loco.json")):
        config = sidecar.name[len("headline_tests_dm_"):-len("_loco.json")]
        stats = _sidecar_calibrated_stats(sidecar)
        folds = _read_csv(dm_dir / f"folds_dm_{config}_loco.csv")
        if stats is None or "calibrated" not in stats:
            print(f"WARNING: {sidecar} has no calibrated record; "
                  f"config {config!r} row DROPPED from the distmatch table",
                  file=sys.stderr)
            continue
        cal = stats["calibrated"]
        avec = stats.get("avec_default", {})
        row = {"config": config,
               "mean_gap": _f(cal["mean_gap_m"]), "sign_p": _f(cal["p_value"]),
               "wilcoxon_p": _f(cal.get("wilcoxon_p")),
               "n_real_gt_sim": cal["n_real_gt_sim"], "n_pairs": cal["n_pairs"],
               "gap_avec": _f(avec.get("mean_gap_m")),
               "cap_policy": cal.get("cap_policy", "median"),
               "dist_metric": cal.get("dist_metric", "emd"),
               # dm's interaction distance is VISIBLE in the config name
               # (the _idX suffix), so only a scenario subset counts as
               # hidden fit-affecting provenance here.
               "nondefault_provenance": bool(
                   cal.get("scenario", "all") not in (None, "all"))}
        if folds is not None:
            row["sigma_mean"] = float(folds["sigma"].mean())
            row["v0_mean"] = float(folds["v0"].mean())
            row["test_ade"] = float(folds["test_ade"].mean())
            row["ade_avec"] = float(folds["base_default_test_ade"].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    if w0_fallback is not None and (df.empty or "w0" not in set(df.get("config", []))):
        w0 = dict(w0_fallback)
        w0["config"] = "w0"
        w0.setdefault("cap_policy", "median")
        w0.setdefault("dist_metric", "emd")
        df = pd.concat([df, pd.DataFrame([w0])], ignore_index=True)
    return df


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal deterministic markdown table (no tabulate dependency)."""
    if df is None or df.empty:
        return "_(no data)_"
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "nan"
            if np.isinf(v):
                return "inf" if v > 0 else "-inf"
            if v != 0.0 and abs(v) < 1e-3:  # p-values etc: don't round to 0.000
                return f"{v:.3g}"
            return floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _bool_ja(b: bool) -> str:
    return "yes" if b else "no"


def build_report(root: Path) -> str:
    cap_dir = root / "cap"
    dm_dir = root / "distmatch"
    surf_dir = root / "surfaces"

    cap_pooled = _read_csv(cap_dir / "cap_pooled.csv")
    m_profile = _read_csv(cap_dir / "capfit_m_profile.csv")
    cap_loco = load_cap_loco(cap_dir)
    ident = _read_csv(surf_dir / "identifiability.csv")

    # w0 (canonical fitter) LOCO reference = the cap median policy run
    # (bit-identical objective; tested in tests/test_distmatch_objective.py).
    w0_ref = None
    if not cap_loco.empty and "median" in set(cap_loco["policy"]):
        m = cap_loco[cap_loco["policy"] == "median"].iloc[0]
        w0_ref = {k: m[k] for k in ("mean_gap", "sign_p", "n_real_gt_sim",
                                    "n_pairs", "test_ade", "ade_avec",
                                    "gap_avec", "sigma_mean", "v0_mean")
                  if k in m}
    dm_pooled_paths = sorted(dm_dir.glob("dm_pooled_*.csv"))
    dm_pooled = (pd.concat([pd.read_csv(p) for p in dm_pooled_paths],
                           ignore_index=True) if dm_pooled_paths else None)
    # The w0 reference synthesised from the cap median run is only valid when
    # that run used default fit-affecting flags (recorded provenance).
    if w0_ref is not None and not cap_loco.empty:
        med_row = cap_loco[cap_loco["policy"] == "median"].iloc[0]
        if bool(med_row.get("nondefault_provenance", False)):
            print("WARNING: cap median LOCO run has non-default provenance "
                  "(interaction distance / scenario subset); refusing to use "
                  "it as the canonical w0 reference", file=sys.stderr)
            w0_ref = None
    dm_loco = load_dm_loco(dm_dir, w0_fallback=w0_ref)

    # Regime guard (review finding): a dm config run under a non-median cap
    # policy is not comparable to the median-regime w0 reference; exclude it
    # from the verdict and say so, instead of silently mixing regimes.
    regime_excluded: List[str] = []
    dm_verdict_input = dm_loco
    if not dm_loco.empty and "cap_policy" in dm_loco.columns:
        ok = (dm_loco["cap_policy"] == "median") | (dm_loco["config"] == "w0")
        regime_excluded = sorted(dm_loco.loc[~ok, "config"].astype(str))
        dm_verdict_input = dm_loco[ok]
    prov_excluded: List[str] = []
    if not dm_verdict_input.empty and "nondefault_provenance" in dm_verdict_input.columns:
        ok = ~dm_verdict_input["nondefault_provenance"].astype(bool)
        prov_excluded = sorted(
            dm_verdict_input.loc[~ok, "config"].astype(str))
        dm_verdict_input = dm_verdict_input[ok]

    cv = cap_verdict(cap_loco) if not cap_loco.empty else {
        "verdict": "undetermined", "reason": "no cap LOCO outputs found",
        "per_policy": {}, "artifact_hits": [], "confounded_hits": [],
        "partially_explained": []}
    dv = (distmatch_verdict(dm_verdict_input) if not dm_verdict_input.empty
          else {"verdict": "undetermined", "reason": "no distmatch LOCO outputs",
                "per_config": {}})
    iv = (identifiability_summary(ident) if ident is not None
          else {"per_policy_axis": {}, "restored_any": False,
                "restore_factor": BAND_RESTORE_FACTOR})

    # F1 rows: every LOCO configuration with within-regime AVEC controls.
    # closedloop is EXCLUDED: both arms run 1.3x too fast there, so a relative
    # win inside that broken regime (calibrated 1.60 vs AVEC 1.77 -- both ~2.5x
    # the canonical ADE) says nothing about beating hand-tuning as deployed.
    f1_rows = []
    for _, r in cap_loco.iterrows():
        if r["policy"] == "closedloop":
            continue
        if bool(r.get("nondefault_provenance", False)):
            continue  # non-default fit flags: not comparable to canonical AVEC
        if {"test_ade", "ade_avec"} <= set(cap_loco.columns):
            f1_rows.append({"label": f"cap:{r['policy']}",
                            "ade_calibrated": r.get("test_ade", float("nan")),
                            "ade_avec": r.get("ade_avec", float("nan")),
                            "gap_calibrated": r["mean_gap"],
                            "gap_avec": r.get("gap_avec", float("nan"))})
    for _, r in dm_verdict_input.iterrows():
        if r["config"] == "w0":
            continue
        f1_rows.append({"label": f"dm:{r['config']}",
                        "ade_calibrated": r.get("test_ade", float("nan")),
                        "ade_avec": r.get("ade_avec", float("nan")),
                        "gap_calibrated": r["mean_gap"],
                        "gap_avec": r.get("gap_avec", float("nan"))})
    fv = f1_verdict(f1_rows)

    # RQ1b domain check for every candidate LOCO-mean point.
    domain_rows = []
    for _, r in cap_loco.iterrows():
        if "sigma_mean" in cap_loco.columns and np.isfinite(r.get("sigma_mean", float("nan"))):
            chk = rq1b_domain_check(r["sigma_mean"], r["v0_mean"])
            domain_rows.append({"point": f"cap:{r['policy']}",
                                "sigma": chk["sigma"], "v0": chk["v0"],
                                "status": chk["status"]})
    for _, r in dm_loco.iterrows():
        if "sigma_mean" in dm_loco.columns and np.isfinite(r.get("sigma_mean", float("nan"))):
            chk = rq1b_domain_check(r["sigma_mean"], r["v0_mean"])
            domain_rows.append({"point": f"dm:{r['config']}",
                                "sigma": chk["sigma"], "v0": chk["v0"],
                                "status": chk["status"]})
    domain_df = pd.DataFrame(domain_rows)

    # ------------------------------------------------------------------ text
    L: List[str] = []
    L += ["# RQ2 計装監査 REPORT: 速度キャップ切り分け × 分布マッチング較正",
          "",
          "生成: `examples/make_rq2_instrument_report.py`（全 verdict は純関数、"
          "prose は表から機械生成）。入力: `outputs/rq2_instrument_audit/` の "
          "cap / distmatch / surfaces 出力。",
          "",
          f"判定しきい値: gap 縮小 >= {SHRINK_THRESHOLD:.0%}・sign 優位崩壊 p > "
          f"{DOMINANCE_ALPHA}・F1 は ADE >= {ADE_BEAT_MARGIN:.0%} かつ |gap| >= "
          f"{GAP_BEAT_MARGIN:.0%} の同時改善・識別性回復はバンド幅 <= "
          f"{BAND_RESTORE_FACTOR:g}x(ADE 基準)。",
          ""]

    L += ["## 1. 速度キャップ方策の切り分け（review F2）", "",
          "pooled 表（1.1/1.2/2.1）は in-sample（全26遭遇で fit→同一遭遇で評価）"
          "の記述値であり、p 値は掲載しない（in-sample の検定 p は確証的に読めず、"
          "ledger 登載対象は held-out LOCO の対応検定のみ）。判定は全て LOCO "
          "held-out（1.3/2.2）に基づく。", ""]
    if cap_pooled is not None:
        L += ["### 1.1 pooled 較正（全26遭遇・方策内3アーム対照）", "",
              _md_table(cap_pooled[["policy", "cap_headroom", "sigma", "v0",
                                    "fit_loss", "ade_calibrated", "gap_calibrated",
                                    "n_real_gt_sim_calibrated", "n_pairs"]]),
              ""]
    if m_profile is not None:
        m_star = float(m_profile.loc[m_profile["fit_loss"].idxmin(), "m"])
        L += ["### 1.2 capfit ヘッドルーム掃引（pooled）", "",
              _md_table(m_profile[["m", "sigma", "v0", "fit_loss", "refined",
                                   "ade", "gap", "n_pairs", "n_real_gt_sim"]]),
              "",
              f"m*（pooled fit loss 最小）= **{m_star:g}**。"
              + ("m=1.0 は median 経路へのエイリアス（構成的にビット同一・"
                 "回帰テストで固定）のため、capfit の LOCO 再走は median の複製に"
                 "なるだけであり実行しない（uncapped が分離キャップの held-out "
                 "証拠を担う）。"
                 if m_star == 1.0 else
                 f"LOCO は m*={m_star:g} で実行。"),
              ""]
    if not cap_loco.empty:
        show = [c for c in ("policy", "sigma_mean", "v0_mean", "test_ade",
                            "ade_avec", "mean_gap", "gap_avec",
                            "n_real_gt_sim", "n_pairs", "sign_p")
                if c in cap_loco.columns]
        L += ["### 1.3 LOCO held-out（判定の根拠）", "",
              _md_table(cap_loco[show], floatfmt="{:.4g}"), ""]
        cl = cap_loco[cap_loco["policy"] == "closedloop"]
        if not cl.empty and float(cl.get("v0_mean", pd.Series([0.0])).iloc[0]) > 10:
            rng = ("" if "v0_min" not in cl.columns else
                   f"（fold 範囲 [{float(cl['v0_min'].iloc[0]):.3g}, "
                   f"{float(cl['v0_max'].iloc[0]):.3g}]）")
            L += ["注記: closedloop 行の v0_mean は fold ごとの縮退フィット"
                  f"{rng}の平均であり「較正値」ではない"
                  "（fold 詳細は folds_cap_closedloop_loco.csv）。", ""]
        if "nondefault_provenance" in cap_loco.columns \
                and cap_loco["nondefault_provenance"].any():
            flagged = sorted(cap_loco.loc[cap_loco["nondefault_provenance"],
                                          "policy"].astype(str))
            L += [f"**警告**: 方策 {flagged} は非デフォルトの fit 影響フラグ"
                  "（--interaction-distance / --scenario）で実行されており、"
                  "正準参照・F1 比較から除外している。", ""]
    L += ["### 1.4 verdict（standoff 過小再現の帰属）", "",
          f"**{cv['verdict']}** — {cv.get('reason', '')}", ""]
    for pol, d in cv.get("per_policy", {}).items():
        L += [f"- `{pol}`: gap {d['mean_gap']:+.3f} m（median {d['median_gap']:+.3f} m、"
              f"変化 {d['gap_change_pct']:+.1f}%）・sign {d['n_real_gt_sim']}/"
              f"{d['n_pairs']} (p={d['sign_p']:.3g})・gap縮小={_bool_ja(d['gap_shrunk'])}"
              + ("・**符号反転**（過小→過大再現＝別種の系統誤差）"
                 if d.get("sign_flipped") else "")
              + f"・優位崩壊={_bool_ja(d['dominance_broken'])}"
              + ("" if d["verdict_eligible"]
                 else f"（**verdict 対象外**: {d.get('ineligible_reason', '')}）")]
    if cv.get("partially_explained"):
        L += ["", f"部分説明: {cv['partially_explained']} は gap を "
              f"{SHRINK_THRESHOLD:.0%} 以上縮めたが方向優位は崩れていない＝"
              "キャップは gap の大きさの一部を説明するが、系統的過小再現"
              "そのものは説明しない（binary verdict は structural のまま）。"]
    if cv.get("confounded_hits"):
        L += ["", f"注記: verdict 対象外アーム {cv['confounded_hits']} は縮小基準を"
              "満たすが、対象外理由（上記 bullet 参照）により verdict 証拠に"
              "用いない（closedloop はキャップ効果と歩速誤差の混合、F2 開示参照）。"]
    # Fitter-confound cross-check (review: the re-fitted arm alone could hide a
    # v0-blind fitter; the fixed strong-repulsion AVEC control is the counter).
    if not cap_loco.empty and "gap_avec" in cap_loco.columns:
        unc = cap_loco[cap_loco["policy"] == "uncapped"]
        if not unc.empty and np.isfinite(float(unc["gap_avec"].iloc[0])):
            L += ["", "クロスチェック（fitter 交絡対策）: 判定は再フィット較正アーム"
                  "に基づくが、ADE fitter は v0 にほぼ不感（C2）なので「fitter が"
                  "縮めに行かないだけ」の可能性は方策内 AVEC 対照（固定の強斥力・"
                  "再フィットなし）で棄却する — uncapped レジームの AVEC 対照でも "
                  f"gap {float(unc['gap_avec'].iloc[0]):+.3f} m と正の standoff "
                  "過小再現が残存＝強斥力を固定しても gap は閉じない。"]
    L += [""]

    L += ["## 2. 分布マッチング較正（(A)-2）", ""]
    if dm_pooled is not None:
        L += ["### 2.1 pooled 重み掃引", "",
              _md_table(dm_pooled[[c for c in ("config", "dist_metric", "w_dist",
                                               "interaction_distance", "sigma", "v0",
                                               "fit_loss", "ade", "emd_closest",
                                               "gap", "n_real_gt_sim", "n_pairs")
                                   if c in dm_pooled.columns]]),
              "",
              "注: `fit_loss` は目的関数値そのもの（w 依存）で**行間比較不可**"
              "（pure の 0.376 は EMD 単独値であり「fit が良い」の意味ではない）。"
              "行間で比較可能な共通尺度は `ade` と `emd_closest`/`gap` のみ。", ""]
    if not dm_loco.empty:
        show = [c for c in ("config", "cap_policy", "dist_metric", "sigma_mean",
                            "v0_mean", "test_ade", "ade_avec", "mean_gap",
                            "gap_avec", "n_real_gt_sim", "n_pairs", "sign_p")
                if c in dm_loco.columns]
        L += ["### 2.2 LOCO held-out", "",
              _md_table(dm_loco[show], floatfmt="{:.4g}"), ""]
        if regime_excluded:
            L += [f"**警告**: config {regime_excluded} は非 median キャップ方策で"
                  "実行されており、median レジームの w0 参照と比較できないため "
                  "verdict/F1 から除外している（表には掲載）。", ""]
        if prov_excluded:
            L += [f"**警告**: config {prov_excluded} は非デフォルトの "
                  "--scenario で実行されており、verdict/F1 から除外している。", ""]
    L += ["### 2.3 verdict（standoff 改善 × ADE 犠牲）", "",
          f"**{dv['verdict']}**"
          + (f" — {dv['reason']}" if dv.get("reason") else ""), ""]
    for cfg, d in dv.get("per_config", {}).items():
        edge = abs(abs(d["mean_gap"]) / abs(d["ref_gap"])
                   - (1.0 - SHRINK_THRESHOLD)) < 0.01 if d["ref_gap"] else False
        L += [f"- `{cfg}`: gap {d['mean_gap']:+.3f} m（w0 {d['ref_gap']:+.3f} m、"
              f"変化 {d['gap_change_pct']:+.1f}%）・held-out ADE {d['test_ade']:.3f}"
              f"（w0 {d['ref_ade']:.3f}、犠牲 {d['ade_sacrifice_pct']:+.1f}%）・"
              f"sign {d['n_real_gt_sim']}/{d['n_pairs']} (p={d['sign_p']:.3g})"
              + ("・**符号反転**" if d.get("sign_flipped") else "")
              + ("（gap縮小判定は 25% 閾値ぎわ＝境界事例）" if edge else "")]
    # The dominance sentence is DERIVED, not asserted (review finding: a
    # hardcoded "never breaks (pure 20/26)" survived any future data).
    pc = dv.get("per_config", {})
    if pc:
        broken = sorted(c for c, d in pc.items() if d["dominance_broken"])
        if broken:
            L += ["", f"方向優位（real>sim）が崩れた構成: {broken}（sign と "
                  "Wilcoxon の両方が p > "
                  f"{DOMINANCE_ALPHA} — 個別数値は上記 bullet）。"]
        else:
            worst = min(pc.values(), key=lambda d: d["n_real_gt_sim"])
            L += ["", "方向優位（real>sim）はどの監査構成でも崩れない（最弱でも "
                  f"{worst['n_real_gt_sim']}/{worst['n_pairs']}, "
                  f"sign p={worst['sign_p']:.3g}）＝分布項は gap を部分的に縮める"
                  "だけで、standoff の系統的過小再現そのものは解消しない。"]
    L += [""]

    L += ["## 3. 識別性監査（σ軸・v0軸の 2% バンド幅）", ""]
    if ident is not None:
        show = [c for c in ("objective", "policy", "axis", "band_lo", "band_hi",
                            "band_width", "band_contiguous", "censored_lo",
                            "censored_hi", "fitted", "fitted_on_grid_edge")
                if c in ident.columns]
        L += [_md_table(ident[show]), ""]
    L += [f"識別性の回復（バンド幅 <= {BAND_RESTORE_FACTOR:g}x ADE 基準・"
          "自軸非打切り・連結バンド・fitted が自バンド内・他軸の fitted が"
          "グリッド内・ADE 基準行も同品質、の全条件）: "
          f"**{_bool_ja(iv['restored_any'])}**", ""]
    if ident is not None and not ident.empty:
        # Derived, not asserted: the range the distribution objectives' fitted
        # v0 actually spans in THIS surfaces table (review: the old hardcoded
        # "21..874" was a pooled-table number, contradicting the rows above).
        nonade = ident[ident["objective"] != "ade"]
        vfit = (nonade["fit_v0"] if "fit_v0" in nonade.columns
                else pd.Series(dtype=float))
        vfit = vfit[np.isfinite(vfit)]
        if not vfit.empty:
            L += ["判定規則の理由: profile_band は各軸を「他軸の fitted に最近傍の"
                  "グリッドノード」で切る。本表の分布目的の fitted v0 は "
                  f"{float(vfit.min()):.3g}〜{float(vfit.max()):.3g} とグリッド外へ"
                  "発散するため（最上ノードへクランプ）、そのスライス上の鋭い σ "
                  "バンドは**最適点から遠い条件付き断面**の性質であり、識別性の"
                  "回復とは読めない（(他軸端) 注記）。幅 0 は「グリッド刻み未満」"
                  "（(1ノード) 注記）、(非連結) はバンドの凸包が out-of-band の"
                  "尾根を跨ぐ多峰プロファイル。なお 2% バンドは各目的関数自身の"
                  "最小値に対する相対幅であり、目的関数間の絶対許容は同一でない"
                  "（バンド幅比較はこの前提つきで読む）。", ""]
    for key, e in iv.get("per_policy_axis", {}).items():
        objs = ", ".join(
            f"{o}: {d['band_width']:.3g}"
            + ("(打切り)" if d["censored"] else "")
            + ("(他軸端)" if d.get("other_axis_edge") else "")
            + ("(1ノード)" if d.get("single_node") else "")
            + ("(非連結)" if d.get("noncontiguous") else "")
            + ("(fit外)" if not d.get("fitted_in_band", True) else "")
            + (" RESTORED" if d["restored"] else "")
            for o, d in e["objectives"].items())
        L += [f"- `{key}`: ADE 基準 {e['ade_band_width']:.3g}"
              + ("(打切り)" if e["ade_censored"] else "")
              + ("(基準劣化)" if e.get("reference_degraded") else "")
              + f" → {objs}"]
    L += [""]

    audited = sorted(fv.get("per_label", {}))
    L += ["## 4. 総合 verdict（review F1: 較正は手調整に勝てるか）", "",
          f"**{fv['verdict']}**", "",
          f"対象（実際に判定した LOCO 構成）: {audited if audited else 'なし'}"
          "（方策内 AVEC 対照との比較）。closedloop は両アームとも歩速 ~30% 過大の"
          "壊れたレジーム内比較になるため F1 の証拠から除外（§1.4 の交絡注記と"
          "同一の理由）。非デフォルト provenance / 非 median レジームの構成も除外"
          "（§1.3/§2.2 の警告参照）。", ""]
    if fv["verdict"] == "undetermined":
        L += ["F1 を判定できる LOCO 構成がまだ無い（cap/distmatch の LOCO 実行後に"
              "本レポートを再生成すること）。", ""]
    elif fv["beats"]:
        L += [f"較正が AVEC 手調整を held-out で明確に上回る構成: {fv['beats']}",
              "（→「較正が手調整に勝つ」初の証拠＝F1 解消）", ""]
    else:
        L += [f"監査した全 {len(audited)} 構成で、較正は AVEC 手調整 (0.7, 3.5) を "
              "held-out ADE と standoff の両方で同時に上回れなかった"
              "（F1 の否定的所見は維持・強化）。",
              f"片側のみ改善した構成: {fv['partial'] if fv['partial'] else 'なし'}", ""]
    if domain_rows:
        L += ["### 4.1 新較正点の RQ1b 掃引域チェック", "",
              _md_table(domain_df), "",
              "`outside` の点は committed RQ1b 掃引（±1SD 箱＋LOSO 包絡）が"
              "カバーしない。これらは診断用の器具設定であり正準較正点"
              " (1.168, 1.712) を置換するものではないが、いずれかを研究上"
              "採用する場合は RQ1b の追加 arm が必要（研究判断・本レポートは"
              "再走しない）。", ""]

    L += ["## 5. §3(B)（実データ接地閉ループ）への設計含意", ""]
    if cv["verdict"] == "structural_limit" and not fv["beats"] \
            and fv["verdict"] != "undetermined":
        # The ADE-tie statement is quoted from the canonical (median-regime)
        # w0 reference, not asserted globally (review: uncapped shows a
        # partial ADE win, so "indistinguishable on ADE" only holds for the
        # canonical regime and must say so).
        tie = ""
        if w0_ref is not None \
                and np.isfinite(w0_ref.get("test_ade", float("nan"))) \
                and np.isfinite(w0_ref.get("ade_avec", float("nan"))):
            tie = ("正準（median レジーム）では ADE でも手調整と識別不能"
                   f"（{w0_ref['test_ade']:.3f} vs {w0_ref['ade_avec']:.3f}）の"
                   "まま残り、片側のみの改善（§4 partial）は ADE×standoff の"
                   "同時改善に届かなかった。")
        L += ["キャップ方策を解放しても分布目的を足しても、SFM 斥力は実データの "
              "standoff 分布を held-out で再現できなかった。" + tie
              + "これは (B) の replay 対照設計を直接支持する: 較正 SFM "
              "を「実歩行者の代理」として信頼する根拠は現状存在しないため、閉ループ"
              "評価の反応性軸には replay（記録実歩行者）アームが不可欠であり、"
              "SFM 系アーム（較正/手調整/斥力なし）は「反応モデル仮定の感度幅」を"
              "張る器具として位置づけるべきである。較正の限界そのものが測定妥当性"
              "研究の証拠（ベンチマーク結論の誤差棒）になる。"]
    elif cv["verdict"] == "harness_artifact" or fv["beats"]:
        L += ["キャップ規約の是正（および/または分布目的）で較正の忠実度が実際に"
              "改善した。(B) の閉ループ比較では、較正 SFM アームを本監査の最良"
              "構成（verdict 参照）で駆動し、閉ループ側 max_speeds レジームとの"
              "整合（1.3x vs 中央値固定）を必ず揃えること。改善した較正は replay "
              "対照との乖離測定において「最良の反応モデル」として位置づけられる。"]
    else:
        L += ["cap/distmatch の LOCO 出力が不足しており設計含意は未確定（実行後に"
              "本レポートを再生成すること）。"]
    L += ["",
          "### 実行上の注記",
          "- closedloop アームの歩速 ~30% 過大は F2 開示（calibration_harness "
          "module docstring）参照。閉ループ徹底整合には desired 速度も 1.3x する"
          "本アームの挙動が「閉ループが録画歩行と不整合」という所見そのもの。",
          "- 本レポートに掲載する p 値は held-out LOCO の対応検定のみで、全て "
          "auxiliary sidecar（rq2cap.*/rq2dm.*）として multiplicity ledger に"
          "登載済み（canonical family には不算入）。pooled CSV に含まれる "
          "in-sample の sign_p 列は記述用の生データであり、本レポートには掲載せず"
          "確証的に読まないこと（fit と検定が同一26遭遇＝in-sample）。",
          ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/rq2_instrument_audit")
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    text = build_report(root)
    out = root / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
