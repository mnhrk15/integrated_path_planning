#!/usr/bin/env python3
"""RQ1b sensitivity analysis: re-bench robust/inflation/single planning under
the RQ2-calibrated pedestrian reaction model.

All runs are performed within the RQ2 calibration-valid speed domain (~3 m/s),
avoiding the 5-6 m/s extrapolation flagged as RQ2 limitation #2. The AVEC-speed
(6/5/5 m/s) results are therefore NOT directly comparable; instead an `avec`
arm at the same ~3 m/s is the within-domain re-baseline.

Two campaigns mirror the two AVEC sim claims:
  margin : robust(eps=0) vs margin inflation on the MinDist-vs-Time trade-off
           (deterministic GT)  -> does the robust gain survive calibration?
  rand   : distributionless prediction (CV, LSTM-single) vs robust under
           randomized GT       -> does the CV-danger finding survive calibration?

The GT reaction model is swept over the RQ2 calibration: the AVEC per-scenario
default, the LOCO mean, and the LOCO +/-1SD corners (sensitivity of the planning
conclusions to calibration uncertainty). Each (campaign, GT) arm caches into its
own outdir under --root and is independently resumable.

Usage:
    python examples/run_rq1b_sensitivity.py [--gt core|all] [--include-offdiag]
        [--campaigns margin,rand] [--scenarios A.yaml,B.yaml]
        [--seeds-main 20] [--seeds-corner 10] [--cruise 3.0]
        [--total-time 60] [--root outputs/rq1b] [--report-only]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.run_da_poc import (CONDITIONS, aggregate_and_write, collect_rows,
                                  run_campaign)

# The RQ1b experiment runs on the dedicated within-domain variants under
# scenarios/rq1b/ (S1 path shortened to remove dead-running, S3 crossing
# retimed for a within-domain yield, S2 = frozen base copy). The AVEC base
# scenarios are NOT the RQ1b geometry; defaulting to them silently runs the
# wrong experiment, so the default must point at the variants. Override with
# --scenarios only to re-baseline against a different geometry.
DEFAULT_SCENARIOS = [
    "scenarios/rq1b/scenario_01.yaml",
    "scenarios/rq1b/scenario_02.yaml",
    "scenarios/rq1b/scenario_03.yaml",
]

# GT ego-repulsion settings. sigma/v0 = None -> no override, i.e. use each
# scenario's YAML value. The AVEC default is per-scenario (NOT uniform):
# S1/S3 sigma=0.7/v0=3.5, S2 sigma=0.3/v0=2.1. The calibrated arms apply the
# single RQ2 (sigma, v0) uniformly across scenarios.
#
# M6 corner check (radius-consistency fix): these (sigma, v0) are the radius=0.35
# LOCO calibration. After fixing DEFAULT_AGENT_RADIUS to 0.30 (matching these
# scenarios), the radius=0.30 recalibration gives LOCO mean (1.168, 1.712) -- a
# ~1-2% shift that lies INSIDE the +/-1SD box already swept here
# ([1.040,1.272] x [1.542,1.820]), so the sensitivity conclusions (robust gain
# holds across the box; claim-2 reaction-model dependent) already cover the
# corrected point. The 1980-run campaign is therefore NOT re-run, and these
# values are kept so the cached runs' provenance (ego_repulsion_sigma/v0) stays
# consistent with the GT labels.
GT_CORE = [
    {"label": "avec",     "sigma": None,  "v0": None,
     "meaning": "AVEC per-scenario default (re-baseline)"},
    {"label": "calib",    "sigma": 1.156, "v0": 1.681,
     "meaning": "LOCO mean (canonical calibration)"},
    {"label": "calib_lo", "sigma": 1.040, "v0": 1.542,
     "meaning": "-1SD (weakest avoidance)"},
    {"label": "calib_hi", "sigma": 1.272, "v0": 1.820,
     "meaning": "+1SD (strongest avoidance)"},
]
GT_OFFDIAG = [
    {"label": "calib_s-v+", "sigma": 1.040, "v0": 1.820,
     "meaning": "off-diagonal corner (wide field, strong)"},
    {"label": "calib_s+v-", "sigma": 1.272, "v0": 1.542,
     "meaning": "off-diagonal corner (narrow field, weak)"},
]

# Conditions per campaign (label, method, distribution_aware, epsilon, inflation).
_BY_LABEL = {c[0]: c for c in CONDITIONS}
INFLATION_LABELS = ["sgan_single_inf1.00", "sgan_single_inf1.10",
                    "sgan_single_inf1.20", "sgan_single_inf1.35",
                    "sgan_single_inf1.50"]
ROBUST_LABEL = "sgan_robust_eps0.0"
BASELINE_LABEL = "sgan_single_inf1.00"

MARGIN_CONDITIONS = [_BY_LABEL[lbl] for lbl in INFLATION_LABELS] + [_BY_LABEL[ROBUST_LABEL]]

# cv_single is NOT in run_da_poc.CONDITIONS (kept out of the AVEC default set);
# define it here so the rand campaign can include the distributionless CV arm.
CV_SINGLE = ("cv_single", "cv", False, 0.0, 1.00)
RAND_CONDITIONS = [
    CV_SINGLE,
    _BY_LABEL["lstm_single"],
    _BY_LABEL["lstm_robust_eps0.0"],
    _BY_LABEL["sgan_single_inf1.00"],
    _BY_LABEL["sgan_robust_eps0.0"],
]


# --------------------------------------------------------------------------- #
# Verdict logic (pure functions over a per-arm DataFrame; unit-tested)
# --------------------------------------------------------------------------- #
def _cond_mean(df, scenario, condition, col, collision_free=False):
    g = df[(df.scenario == scenario) & (df.condition == condition)]
    if collision_free:
        g = g[g.collision_count == 0]
    if g.empty:
        return float("nan")
    return float(g[col].mean())


def _cond_collisions(df, scenario, condition):
    g = df[(df.scenario == scenario) & (df.condition == condition)]
    return int(g.collision_count.sum())


# Significance gating for the rand (claim-2) verdicts (review finding M8). The
# raw collision COUNT comparison (cv_single > sgan_robust) is fragile: the
# aggregate sums single-digit counts over scenarios with unequal seed budgets,
# so a 1-vs-0 "flip" is indistinguishable from Monte-Carlo noise. We compare
# at the run level (a run "collided" iff collision_count > 0, the same Bernoulli
# unit run_da_poc already uses for n_collision_runs) and gate the verdict on a
# one-sided Fisher exact test, so a danger claim must be both directional AND
# statistically distinguishable from the robust arm.
DANGER_ALPHA = 0.05


def _run_collided(df, conditions):
    """(collided_runs, n_runs) pooled over ``conditions`` (a run = one DataFrame row).

    ``conditions`` may be a single label or an iterable of labels (e.g. all the
    single-planner conditions pooled together for a per-scenario test).
    """
    if isinstance(conditions, str):
        g = df[df.condition == conditions]
    else:
        g = df[df.condition.isin(list(conditions))]
    return int((g.collision_count > 0).sum()), int(len(g))


def _fisher_greater(a_collided, a_n, b_collided, b_n):
    """One-sided Fisher exact p that group A collides MORE often than group B.

    Run-level 2x2 (collided / clean). Returns NaN when either arm is empty so a
    missing cell never reads as significant.
    """
    if a_n == 0 or b_n == 0:
        return float("nan")
    table = [[a_collided, a_n - a_collided], [b_collided, b_n - b_collided]]
    try:
        _, p = fisher_exact(table, alternative="greater")
    except ValueError:  # pragma: no cover - guarded by the empty-arm check above
        return float("nan")
    return float(p)


def margin_verdict(df):
    """Claim (1) robust gain: does robust(eps=0) dominate margin inflation?

    AVEC criterion (FINAL_BENCHMARK_REPORT.md): the robust gain holds when NO
    single inflation level simultaneously achieves MinDist >= robust AND
    Time <= robust in EVERY scenario. NaN means/means over collision-only sets
    compare False, i.e. a missing/colliding cell never counts as dominating.

    Safety guard: a robust planner that itself collides cannot carry the claim,
    even though a colliding-robust scenario yields r_t=NaN that vacuously blocks
    every inflation from "dominating". So robust_gain_holds additionally
    requires robust to be collision-free; robust_total_collisions records the
    count regardless.
    """
    scenarios = sorted(df.scenario.unique())
    detail = []
    dominating = []
    for inf in INFLATION_LABELS:
        dom_all = bool(scenarios)
        for sc in scenarios:
            r_d = _cond_mean(df, sc, ROBUST_LABEL, "min_dist_m")
            r_t = _cond_mean(df, sc, ROBUST_LABEL, "time_s", collision_free=True)
            i_d = _cond_mean(df, sc, inf, "min_dist_m")
            i_t = _cond_mean(df, sc, inf, "time_s", collision_free=True)
            dom = (i_d >= r_d) and (i_t <= r_t)
            detail.append({"scenario": sc, "inflation": inf,
                           "robust_mindist": r_d, "inf_mindist": i_d,
                           "d_mindist_vs_robust": i_d - r_d,
                           "robust_time": r_t, "inf_time": i_t,
                           "d_time_vs_robust": i_t - r_t,
                           "dominates_robust": bool(dom)})
            if not dom:
                dom_all = False
        if dom_all:
            dominating.append(inf)
    robust_coll = sum(_cond_collisions(df, sc, ROBUST_LABEL) for sc in scenarios)
    return {
        "robust_gain_holds": len(dominating) == 0 and robust_coll == 0,
        "dominating_inflations": dominating,
        "robust_total_collisions": robust_coll,
        "detail": detail,
    }


def rand_verdict(df, alpha=DANGER_ALPHA):
    """Claim (2) CV danger: distributionless prediction collides more than
    distribution-aware (robust), per predictor (apples-to-apples).

    Significance-gated (M8): a danger claim requires BOTH a directional
    difference in collided runs AND a one-sided Fisher p < ``alpha``. A
    directional-but-non-significant difference (e.g. a single-digit 1-vs-0
    aggregate flip) is reported as ``*_undetermined`` rather than counted as the
    claim holding. The aggregate still sums over scenarios -- which the review
    flags as contaminated by GT-artifact scenarios -- so the honest evidence for
    claim (2) is the per-scenario table (rand_scenario_rows), not this number.
    """
    scenarios = sorted(df.scenario.unique())

    def tot(cond):
        return sum(_cond_collisions(df, sc, cond) for sc in scenarios)

    coll = {c: tot(c) for c in
            ["cv_single", "lstm_single", "lstm_robust_eps0.0",
             "sgan_single_inf1.00", "sgan_robust_eps0.0"]}

    cv_c, cv_n = _run_collided(df, "cv_single")
    sr_c, sr_n = _run_collided(df, "sgan_robust_eps0.0")
    ls_c, ls_n = _run_collided(df, "lstm_single")
    lr_c, lr_n = _run_collided(df, "lstm_robust_eps0.0")
    cv_p = _fisher_greater(cv_c, cv_n, sr_c, sr_n)
    lstm_p = _fisher_greater(ls_c, ls_n, lr_c, lr_n)
    cv_dir = cv_c > sr_c
    lstm_dir = ls_c > lr_c
    cv_sig = bool(np.isfinite(cv_p) and cv_p < alpha)
    lstm_sig = bool(np.isfinite(lstm_p) and lstm_p < alpha)
    return {
        # CV (no distribution at all) vs the distribution-aware SGAN robust.
        "cv_danger_holds": bool(cv_dir and cv_sig),
        "cv_danger_direction": bool(cv_dir),
        "cv_danger_undetermined": bool(cv_dir and not cv_sig),
        "cv_fisher_p": cv_p,
        # Same predictor, single vs robust over its own distribution.
        "lstm_danger_holds": bool(lstm_dir and lstm_sig),
        "lstm_danger_direction": bool(lstm_dir),
        "lstm_danger_undetermined": bool(lstm_dir and not lstm_sig),
        "lstm_fisher_p": lstm_p,
        "collisions_by_condition": coll,
        "collided_runs_by_condition": {
            "cv_single": cv_c, "sgan_robust_eps0.0": sr_c,
            "lstm_single": ls_c, "lstm_robust_eps0.0": lr_c},
        # Per-condition run denominators (NOT a single scalar: seed budgets can
        # differ across conditions, so one number would mislabel the others).
        "n_runs_by_condition": {
            "cv_single": cv_n, "sgan_robust_eps0.0": sr_n,
            "lstm_single": ls_n, "lstm_robust_eps0.0": lr_n},
    }


SINGLE_CONDS = ["cv_single", "lstm_single", "sgan_single_inf1.00"]
ROBUST_CONDS = ["lstm_robust_eps0.0", "sgan_robust_eps0.0"]


def rand_scenario_rows(master):
    """Per (gt, scenario) rand collision table + a claim-(2) classification.

    The aggregate verdict (build_verdicts) sums collisions over scenarios, so a
    scenario where the calibrated GT makes ALL planners collide (a GT artifact,
    not planner discrimination) contaminates it. This per-scenario view
    classifies each cell:
      no-conflict   : no planner collides
      single-danger : a distributionless single planner collides while BOTH
                      robust planners stay clean -> genuine claim-(2) signal
      mixed         : single collisions > robust collisions > 0 -> claim-(2)
                      direction holds (single worse) but robust is not perfectly
                      safe (e.g. a hard geometry under weak calibrated repulsion)
      GT-artifact   : robust collisions >= single collisions (>0) -> no
                      discrimination; the GT itself is conflicting in that scenario
    """
    r = master[master.campaign == "rand"]
    rows = []
    for (gt, sc), g in r.groupby(["gt_label", "scenario"]):
        coll = {c: int(g[g.condition == c].collision_count.sum())
                for c in SINGLE_CONDS + ROBUST_CONDS}
        single_tot = sum(coll[c] for c in SINGLE_CONDS)
        robust_tot = sum(coll[c] for c in ROBUST_CONDS)
        if single_tot == 0 and robust_tot == 0:
            klass = "no-conflict"
        elif robust_tot == 0:
            klass = "single-danger"
        elif single_tot > robust_tot:
            klass = "mixed"
        else:
            klass = "GT-artifact"
        # Run-level significance (M8): pool the single planners vs the robust
        # planners and test whether single collides MORE at the run level. This
        # is the per-scenario claim-(2) discriminator the review asks to front
        # (e.g. S2 avec: single 9/60 vs robust 0/40 -> Fisher p~=0.008).
        # CAVEAT (pseudo-replication): the 3 single planners on one seed share
        # the scenario geometry and RNG init, so the pooled "runs" are not
        # independent -- the run-level n is ~3x inflated and this Fisher p is
        # anti-conservative (a lower bound on the true p). Read it as suggestive,
        # not a calibrated significance level; the REPORT states this too.
        s_c, s_n = _run_collided(g, SINGLE_CONDS)
        rb_c, rb_n = _run_collided(g, ROBUST_CONDS)
        fisher_p = _fisher_greater(s_c, s_n, rb_c, rb_n)
        rows.append({"gt_label": gt, "scenario": sc, **coll,
                     "single_total": single_tot, "robust_total": robust_tot,
                     "single_collided_runs": s_c, "single_n": s_n,
                     "robust_collided_runs": rb_c, "robust_n": rb_n,
                     "fisher_p": (round(fisher_p, 4)
                                  if np.isfinite(fisher_p) else float("nan")),
                     "class": klass})
    return pd.DataFrame(rows)


def rq1b_headline_tests(srows):
    """Claim-(2) per-scenario single-vs-robust Fisher tests for the multiplicity
    ledger (make_multiplicity_ledger.py).

    These per-scenario cells -- NOT the aggregate cv/lstm danger (declared noise,
    M8) -- are the honest claim-(2) evidence unit. Every (gt, scenario) cell with
    an evaluable (finite) Fisher p is one hypothesis in the family. That family is
    exactly what the review (point 8) asks to multiplicity-correct: the headline
    'S2/avec is significant (p~=0.008)' must survive BH/Holm over ALL the cells we
    scanned, not be cherry-picked from them.

    Each record carries the pseudo-replication caveat verbatim: the 3 single
    planners on one seed share geometry + RNG init, so the run-level n is ~3x
    inflated and the Fisher p is anti-conservative (a lower bound on the true p).
    A finding that survives correction even at this optimistic p is the honest
    floor; the ledger states the caveat so the survival is not over-read.
    """
    if srows.empty:
        return []
    tests = []
    for _, r in srows.iterrows():
        p = r.get("fisher_p", float("nan"))
        if not (isinstance(p, (int, float, np.floating)) and np.isfinite(p)):
            continue
        gt, sc = str(r["gt_label"]), str(r["scenario"])
        # power_tier: avec/calib carry the full seed budget (seeds_main); the
        # +/-1SD corners (calib_lo/hi) run at half budget (seeds_corner) as a
        # robustness check, NOT a headline number. The ledger uses this to show a
        # family-definition sensitivity (the headline S2/avec signal survives BH
        # within the avec-only and headline-GT families but not over the full
        # 12-cell scan that includes the underpowered corners).
        tier = "headline" if gt in ("avec", "calib") else "corner"
        tests.append({
            "test_id": f"rq1b.rand.fisher.{gt}.{sc}",
            "description": (f"Per-scenario single-vs-robust collision Fisher "
                            f"(GT={gt}, {sc}, class={r['class']})"),
            "family": "rq1b_claim2_fisher",
            "gt": gt,
            "scenario": sc,
            "power_tier": tier,
            "p_value": float(p),
            "sidedness": "one-sided",
            "single_collided": int(r["single_collided_runs"]),
            "single_n": int(r["single_n"]),
            "robust_collided": int(r["robust_collided_runs"]),
            "robust_n": int(r["robust_n"]),
            "klass": str(r["class"]),
            "headline": bool(r["class"] in ("single-danger", "mixed")),
            "caveat": ("pseudo-replication: 3 single planners share seed/geometry, "
                       "run-level n ~3x inflated, Fisher p anti-conservative "
                       "(lower bound on true p)"),
        })
    return tests


def build_verdicts(master, gt_labels):
    rows = []
    for gt in gt_labels:
        mdf = master[(master.campaign == "margin") & (master.gt_label == gt)]
        rdf = master[(master.campaign == "rand") & (master.gt_label == gt)]
        mv = margin_verdict(mdf) if not mdf.empty else None
        rv = rand_verdict(rdf) if not rdf.empty else None
        def _p(rv_key):
            if not rv:
                return None
            val = rv[rv_key]
            return round(val, 4) if np.isfinite(val) else None
        rows.append({
            "gt_label": gt,
            "robust_gain_holds": (mv["robust_gain_holds"] if mv else None),
            "dominating_inflations": (",".join(mv["dominating_inflations"])
                                      if mv else ""),
            "robust_collisions": (mv["robust_total_collisions"] if mv else None),
            "cv_danger_holds": (rv["cv_danger_holds"] if rv else None),
            "cv_danger_undetermined": (rv["cv_danger_undetermined"] if rv else None),
            "cv_fisher_p": _p("cv_fisher_p"),
            "lstm_danger_holds": (rv["lstm_danger_holds"] if rv else None),
            "lstm_danger_undetermined": (rv["lstm_danger_undetermined"] if rv else None),
            "lstm_fisher_p": _p("lstm_fisher_p"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Campaign execution
# --------------------------------------------------------------------------- #
def _seeds_for(gt_label, seeds_main, seeds_corner):
    # Headline arms (default + canonical calibration) get the full seed budget;
    # the +/-1SD corners get fewer (a robustness check, not a headline number).
    return range(seeds_main if gt_label in ("avec", "calib") else seeds_corner)


def _run_arm(campaign, gt, scenarios, seeds, cruise, total_time, root):
    conditions = MARGIN_CONDITIONS if campaign == "margin" else RAND_CONDITIONS
    overrides = {
        "ego_repulsion_sigma": gt["sigma"],
        "ego_repulsion_v0": gt["v0"],
        "ego_target_speed": cruise,
        "total_time": total_time,
        "v0_randomization": (campaign == "rand"),
    }
    outdir = Path(root) / campaign / gt["label"]
    print(f"\n##### RQ1b arm: campaign={campaign} gt={gt['label']} "
          f"sigma={gt['sigma']} v0={gt['v0']} cruise={cruise} "
          f"seeds={len(seeds)} #####", flush=True)
    df, failed = run_campaign(scenarios, conditions, list(seeds), outdir, overrides)
    if not df.empty:
        aggregate_and_write(df, outdir, conditions, baseline_label=BASELINE_LABEL,
                            n_seeds=len(seeds))
        df = df.copy()
        df["campaign"] = campaign
        df["gt_label"] = gt["label"]
    return df, failed


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def _means_table(master):
    """Per (campaign, gt_label, scenario, condition) aggregate for the appendix."""
    rows = []
    for (camp, gt, sc, cond), g in master.groupby(
            ["campaign", "gt_label", "scenario", "condition"]):
        g_nc = g[g.collision_count == 0]
        rows.append({
            "campaign": camp, "gt_label": gt, "scenario": sc, "condition": cond,
            "n": len(g),
            "collisions": int(g.collision_count.sum()),
            "min_dist_mean": round(float(g.min_dist_m.mean()), 3),
            "time_mean_cf": (round(float(g_nc.time_s.mean()), 2)
                             if not g_nc.empty else float("nan")),
            "rms_jerk_mean": (round(float(g.rms_jerk.mean()), 3)
                              if "rms_jerk" in g else float("nan")),
            "mean_accel_mean": (round(float(g.mean_accel.mean()), 3)
                                if "mean_accel" in g else float("nan")),
        })
    return pd.DataFrame(rows)


_GT_ORDER = ["avec", "calib", "calib_lo", "calib_hi", "calib_s-v+", "calib_s+v-"]


def _scenario_narrative(srows):
    """Per-scenario claim-(2) reading generated FROM the data (review M9).

    The previous reading was hand-written against an implicit 2-GT (avec/calib)
    story and could contradict its own 4-GT table. This generates the reading
    from ``srows`` for every GT actually present, so the prose can never disagree
    with the table. A ``*`` marks cells where the per-scenario single-vs-robust
    Fisher test is significant (p<0.05) -- those are the honest claim-(2) signal.
    """
    if srows.empty:
        return ["（per-scenario データなし）"]
    # Cover EVERY GT actually in the table, not just the ones hard-coded in
    # _GT_ORDER: known GTs first (in canonical order), then any extra labels
    # (e.g. a new GT_CORE/CORNER entry, or a custom --report-only CSV) sorted
    # after. Otherwise the table keeps an unknown GT (it sorts last via fillna)
    # while the narrative silently drops it -- the exact table/prose divergence
    # this function (review M9) exists to prevent.
    present = set(srows.gt_label)
    gts_present = ([g for g in _GT_ORDER if g in present]
                   + sorted(present - set(_GT_ORDER)))

    def cell(sc, gt):
        c = srows[(srows.scenario == sc) & (srows.gt_label == gt)]
        return c.iloc[0] if not c.empty else None

    def claim2_scenarios(gt):
        sub = srows[srows.gt_label == gt]
        return sorted(sub[sub["class"].isin(["single-danger", "mixed"])]
                      .scenario.tolist())

    L = ["**読み筋（per-scenario・全 GT をデータから自動生成）**:"]
    for sc in sorted(srows.scenario.unique()):
        parts = []
        for gt in gts_present:
            row = cell(sc, gt)
            if row is None:
                continue
            tag = row["class"]
            p = row.get("fisher_p", float("nan"))
            if (isinstance(p, (int, float)) and p == p and p < DANGER_ALPHA
                    and row["class"] in ("single-danger", "mixed")):
                tag += f"*(p={p:.3f})"
            parts.append(f"{gt}={tag}")
        L.append(f"- **{sc}**: " + " / ".join(parts))
    L.append("")
    for gt in gts_present:
        cs = claim2_scenarios(gt)
        L.append(f"- **{gt}** で主張②（single-danger/mixed）が立つシナリオ: "
                 f"{cs if cs else 'なし'}。")
    L.append("")
    L.append("**結論（データ駆動）**: 主張②（分布なし計画は危険）の成否は GT 反応モデルに"
             "依存する（上表が一次情報・`*` は per-scenario の single-vs-robust run-level "
             "Fisher が有意なセル）。集計 `cv_danger_holds` は単桁・不均等シードでノイズ"
             "grade なので、主張②の主証拠はこの per-scenario 有意セル。robust 利得"
             "（主張①）は別途 `robust_gain_holds` 参照（全 GT で頑健）。")
    L.append("")
    L.append("> **循環性 caveat（M7）**: RQ1b は較正済み反応モデル下での *感度分析* で"
             "あり外的検証ではない。衝突相手の『GT 歩行者』は実歩行者ではなく較正済み "
             "SFM（RQ2 で実 standoff を ~0.7m 過小再現）が生成する。よって主張②の所見は"
             "『SFM family 内のパラメータ感度』であって、実歩行者下での安全結論ではない。"
             "独立な実データ閉ループ検証は本研究の範囲外。")
    return L


def _sensitivity_status(verdicts, col, undet_col=None):
    """Does verdict ``col`` flip across GT settings? Returns a status string.

    A ``None`` entry means the verdict was not computed for that GT (missing/
    empty arm); report that as undetermined rather than silently calling an
    uncomputed verdict "robust".

    The danger columns are now significance-gated (holds = direction AND Fisher
    p<0.05), so a ``False`` can mean EITHER a real direction reversal OR merely a
    loss of significance at a low-seed corner GT (same direction,
    undetermined=True). Only the former is calibration sensitivity; the latter is
    a detection-power artifact and must NOT read as a reversal. Pass the
    ``undet_col`` companion to make that distinction.
    """
    raw = verdicts[col].tolist()
    if not raw:  # empty verdicts frame: 0 GTs is NOT "invariant/robust"
        return "GT なし（判定不能）"
    vals = [v for v in raw if v is not None]
    if len(vals) < len(raw):
        return f"一部 GT 未計算（{len(vals)}/{len(raw)} GT のみ・判定不能）"
    if len(set(vals)) <= 1:
        return "全 GT で不変（頑健）"
    if undet_col is not None and undet_col in verdicts:
        undet = verdicts[undet_col].tolist()
        # Coerce to plain bool before the boolean logic: None is already filtered
        # by the early-return above, and a verdict is None-or-bool together with
        # its undetermined companion, so undet has no None here either. Using
        # bool() (not `is True`) is robust if a future edit ever stores a
        # numpy.bool_ in a verdict column -- `np.True_ is True` is False, which
        # would silently misread the sensitivity verdict.
        rawb = [bool(v) for v in raw]
        undetb = [bool(u) for u in undet]
        has_pos = any(rawb)
        # A genuine NEGATIVE GT: danger does not hold AND is not merely
        # undetermined (i.e. the direction itself failed).
        has_neg = any((not h) and (not u) for h, u in zip(rawb, undetb))
        if has_pos and not has_neg:
            return ("方向は全 GT で不変だが一部 GT で有意性が落ちる"
                    "（少 seed corner の検出力差＝真の方向反転ではない）")
    return "反転あり（較正に感度あり）"


def write_report(root, master, verdicts, gts, cruise):
    root = Path(root)
    means = _means_table(master)
    means.to_csv(root / "means.csv", index=False)

    L = []
    L.append("# RQ1b 感度分析レポート")
    L.append("")
    L.append("較正済み反応モデル下での robust/inflation/single 計画の再ベンチ"
             "（感度分析、外的検証ではない）。")
    L.append(f"全ラン cruise={cruise} m/s（RQ2 較正有効域 ~[0.4, 4.0] m/s 内"
             "＝5-6 m/s 外挿を回避。RQ2 limitation #2）。AVEC の 6/5/5 m/s 結果"
             "とは直接非比較で、同一 ~3 m/s の `avec` アームが域内再ベースライン。")
    L.append("")
    L.append("> M6 整合注記: GT の calib/±1SD は radius=0.35 較正値。"
             "DEFAULT_AGENT_RADIUS を 0.30 に整合した再較正は LOCO 平均 (1.168, 1.712) "
             "で、ここでスイープ済みの ±1SD box [1.040,1.272]×[1.542,1.820] 内に収まる"
             "（~1-2% シフト）。よって本キャンペーン（1980 ラン）は再実行せず、結論は"
             "補正後の点も感度範囲としてカバーする。")
    L.append("")
    L.append("## GT 反応モデル設定（σ/v0）")
    L.append("")
    gtbl = pd.DataFrame([{
        "gt_label": g["label"],
        "sigma": "per-scenario" if g["sigma"] is None else g["sigma"],
        "v0": "per-scenario" if g["v0"] is None else g["v0"],
        "meaning": g.get("meaning", ""),
    } for g in gts])
    L.append(_md_table(gtbl))
    L.append("")
    L.append("（avec の σ/v0 は各シナリオ YAML 値: S1/S3 σ0.7/v0 3.5, S2 σ0.3/v0 2.1。"
             "実効値は means.csv の min_dist 等とともに all_runs.csv の "
             "ego_repulsion_sigma/v0 列に記録。）")
    L.append("")
    L.append("## 判定サマリ")
    L.append("")
    L.append("- robust_gain_holds: 全シナリオ同時に robust を支配する inflation が"
             "無い（＝主張①保持）")
    L.append("- cv_danger_holds: CV single の collided-run > SGAN robust **かつ** "
             "run-level 片側 Fisher p<0.05（有意な主張②保持）。方向はあるが非有意なら "
             "`cv_danger_undetermined`＝判定保留（単桁差はノイズ grade）")
    L.append("- lstm_danger_holds: LSTM single vs LSTM robust に同じ有意性ゲート")
    L.append("")
    L.append(_md_table(verdicts))
    L.append("")
    L.append("> 注意: 集計 `cv_danger_holds`/`lstm_danger_holds` は衝突をシナリオ"
             "横断で合算し、かつ corner GT は seed 予算が少ない（avec/calib=20・"
             "±1SD=10）ため、単桁カウントの flip は Monte-Carlo ノイズと区別できない。"
             "有意性ゲート（Fisher p<0.05）を課しても集計はなお GT-artifact シナリオに"
             "汚染されるため、**主張②は必ず下記の per-scenario 分類（有意セル `*`）で"
             "読むこと**。主張①（`robust_gain_holds`）はシナリオ横断の集計でも頑健。")
    L.append("")

    # Sensitivity: do verdicts flip across GT? (see _sensitivity_status)
    L.append("## 感度（GT 間で判定が反転するか）")
    L.append("")
    for col, undet in [("robust_gain_holds", None),
                       ("cv_danger_holds", "cv_danger_undetermined"),
                       ("lstm_danger_holds", "lstm_danger_undetermined")]:
        L.append(f"- **{col}**: {_sensitivity_status(verdicts, col, undet)}")
    L.append("")
    L.append("> 注意: `cv_danger_holds`/`lstm_danger_holds` の反転は有意性ゲート後でも "
             "corner GT の少 seed 予算（10）に左右されやすい。集計 danger の反転は "
             "感度の *示唆* に留め、確定的な per-scenario 信号は下表の有意セル（`*`）で読む。"
             "`robust_gain_holds`（主張①）の不変性が最も信頼できる結論。")
    L.append("")

    # Per-scenario claim-(2): the honest, uncontaminated view.
    srows = rand_scenario_rows(master)
    srows.to_csv(root / "scenario_rand.csv", index=False)
    L.append("## 主張② シナリオ別（per-scenario・汚染なし）")
    L.append("")
    L.append("各 (GT, シナリオ) の rand 衝突数・run-level Fisher p と分類"
             "（single-danger=分布なし single が衝突しつつ robust 2種は無衝突＝真の主張"
             "②信号／mixed=single≫robust>0＝主張②方向は残るが robust も非ゼロ／"
             "GT-artifact=robust≧single＝弁別でなく GT 自体の衝突／no-conflict=無衝突。"
             "fisher_p=single 群 vs robust 群の run-level 片側 Fisher）:")
    L.append("")
    L.append("> **fisher_p の読み方（2つの caveat）**: (1) `class` は衝突 *カウント* から、"
             "`fisher_p` は collided-run の有意性から独立に決まる。よって "
             "`single-danger`（robust=0）でも `fisher_p` が非有意（少 run の偶然パターン）"
             "なことがある＝class 名だけで主張②を確定せず必ず `fisher_p`/`*` を併読する。"
             "(2) single 群は3計画器（cv/lstm/sgan）×seed を1 run 単位でプールするが、"
             "同一 seed・同一シナリオの3計画器は初期条件と RNG を共有し独立でない"
             "（pseudo-replication）。よって run-level n は約3倍に水増しされ、"
             "Fisher p は反保守的（楽観的＝真の p の下界）。有意セルは『示唆』として読み、"
             "確定的結論にはしない。")
    L.append("")
    if not srows.empty:
        order = {g: i for i, g in enumerate(_GT_ORDER)}
        # Unknown GT labels sort last (fillna) instead of becoming NaN and
        # scattering unpredictably.
        srows = srows.sort_values(
            by=["scenario", "gt_label"],
            key=lambda s: (s.map(order).fillna(len(order))
                           if s.name == "gt_label" else s))
        L.append(_md_table(srows[["scenario", "gt_label"] + SINGLE_CONDS
                                  + ROBUST_CONDS + ["fisher_p", "class"]]))
    L.append("")
    # Reading-narrative generated FROM the per-scenario table for every GT
    # present (review M9): never hand-written prose that can contradict the
    # shipped table.
    L.extend(_scenario_narrative(srows))
    L.append("")

    L.append("## 付録: 平均指標（means.csv 抜粋）")
    L.append("")
    L.append("### margin キャンペーン")
    L.append("")
    mm = means[means.campaign == "margin"].drop(columns=["campaign"])
    if not mm.empty:
        L.append(_md_table(mm))
    L.append("")
    L.append("### rand キャンペーン（衝突数）")
    L.append("")
    rm = means[means.campaign == "rand"][
        ["gt_label", "scenario", "condition", "n", "collisions", "min_dist_mean"]]
    if not rm.empty:
        L.append(_md_table(rm))
    L.append("")

    (root / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS),
                    help="Comma-separated scenario YAML paths")
    ap.add_argument("--campaigns", default="margin,rand",
                    help="Comma-separated: margin and/or rand")
    ap.add_argument("--gt", default="core", choices=["core", "all"],
                    help="core = avec/calib/calib_lo/calib_hi; "
                         "all also adds the off-diagonal +/-1SD corners")
    ap.add_argument("--include-offdiag", action="store_true",
                    help="Add the two off-diagonal +/-1SD corners (alias of --gt all)")
    ap.add_argument("--seeds-main", type=int, default=20,
                    help="Seeds for the avec/calib headline arms")
    ap.add_argument("--seeds-corner", type=int, default=10,
                    help="Seeds for the +/-1SD corner arms")
    ap.add_argument("--cruise", type=float, default=3.0,
                    help="Cruise target speed [m/s] (RQ2 calibration domain)")
    ap.add_argument("--total-time", type=float, default=60.0,
                    help="total_time [s] (low speed needs a longer cap)")
    ap.add_argument("--root", default="outputs/rq1b")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip running; rebuild master/verdicts/REPORT from cache")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    campaigns = [c.strip() for c in args.campaigns.split(",") if c.strip()]
    unknown_campaigns = [c for c in campaigns if c not in ("margin", "rand")]
    if unknown_campaigns:
        ap.error(f"Unknown campaign(s): {unknown_campaigns} (valid: margin, rand)")
    gts = list(GT_CORE)
    if args.gt == "all" or args.include_offdiag:
        gts += GT_OFFDIAG
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    arm_dfs = []
    total_failed = 0
    for campaign in campaigns:
        for gt in gts:
            outdir = root / campaign / gt["label"]
            if args.report_only:
                df = collect_rows(outdir)
                if df.empty:
                    continue
                df = df.copy()
                df["campaign"] = campaign
                df["gt_label"] = gt["label"]
                arm_dfs.append(df)
                continue
            seeds = _seeds_for(gt["label"], args.seeds_main, args.seeds_corner)
            df, failed = _run_arm(campaign, gt, scenarios, seeds, args.cruise,
                                  args.total_time, root)
            total_failed += failed
            if not df.empty:
                arm_dfs.append(df)

    if not arm_dfs:
        print("No runs/cache found; nothing to aggregate.", file=sys.stderr)
        sys.exit(1)
    master = pd.concat(arm_dfs, ignore_index=True)
    master.to_csv(root / "master_runs.csv", index=False)

    gt_labels = [g["label"] for g in gts]
    verdicts = build_verdicts(master, gt_labels)
    verdicts.to_csv(root / "verdicts.csv", index=False)
    print("\n=== RQ1b verdicts ===")
    print(verdicts.to_string(index=False))

    write_report(root, master, verdicts, gts, args.cruise)
    print(f"\nWrote {root}/REPORT.md, verdicts.csv, master_runs.csv, means.csv")

    # Machine-readable headline-test sidecar for the cross-RQ multiplicity ledger
    # (make_multiplicity_ledger.py). Deterministic: rand_scenario_rows groups by
    # sorted (gt, scenario) and rounds fisher_p, so re-running (incl. report-only)
    # overwrites this file byte-for-byte from the same cached runs.
    sidecar = root / "headline_tests.json"
    sidecar.write_text(json.dumps({
        "source": "RQ1b-rand",
        "generated_by": "run_rq1b_sensitivity.py",
        "tests": rq1b_headline_tests(rand_scenario_rows(master)),
    }, indent=2) + "\n")
    print(f"Wrote {sidecar}")

    if total_failed:
        print(f"\nWARNING: {total_failed} run(s) failed and were not cached "
              f"(they will be retried on the next invocation).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
