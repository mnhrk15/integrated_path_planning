#!/usr/bin/env python3
"""RQ3 analysis: verdicts, REPORT.md and ledger sidecar for the real loop.

Reads the campaign aggregates (``all_runs.csv`` -- regenerable from the run
cache via ``run_rq3_realloop.py --report-only`` -- and ``encounters.csv``)
and derives everything downstream as pure functions of those tables:

* **V1 reactivity confound** (headline): per-encounter PAIRED differences of
  each SFM arm against the replay reference at fixed (predictor, plan);
  seeds are collapsed to the encounter mean FIRST (no pseudo-replication),
  then sign + Wilcoxon over n<=26 encounter pairs
  (``run_rq2_evaluation._paired_stats``). The calib-vs-replay tests are the
  canonical family ``rq3_v1_reactivity`` (6 tests, user-approved 2026-07-03);
  avec/norep/calib13x are auxiliary controls.
* **V2 verdict preservation**: does the robust-vs-single gain direction and
  the most-dangerous-predictor identity hold across pedestrian arms?
  Tri-state discipline (invariant / power-limited undetermined / reversal)
  mirroring run_rq1b_sensitivity._sensitivity_status: a non-significant
  disagreement is NOT a reversal.
* **V3 robust gain on real geometry** (auxiliary): robust - single paired
  within the replay arm; non-replay arms and the Wilcoxon companions are
  emitted as auxiliary control families, and the V2 ranking gates' Wilcoxon
  p is persisted (v2_verdicts.csv ``gap_p`` + sidecar family
  ``rq3_v2_ranking_gates``) -- thesis cross-cut review M2a.

All REPORT prose is generated from the computed tables (never hand-written
claims), thresholds are module constants quoted verbatim, and the outputs are
byte-stable: rerunning on the same all_runs.csv reproduces every artifact
bit-for-bit.

Usage:
    python examples/make_rq3_report.py [--root outputs/rq3_realloop]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.multiplicity import adjust  # noqa: E402
from examples.run_rq2_evaluation import _paired_stats  # noqa: E402
from examples.run_rq3_realloop import PED_ARMS, PLAN_MODES, PREDICTORS  # noqa: E402

# ---------------------------------------------------------------------------
# Analysis constants (quoted verbatim in the REPORT)
# ---------------------------------------------------------------------------
ALPHA = 0.05                    # significance gate for the tri-state verdicts
REFERENCE_ARM = "replay"        # the non-SFM reference point (recorded peds)
CANONICAL_ARM = "calib"         # canonical family compares calib vs replay
CONTROL_ARMS = ["avec", "norep", "calib13x"]
PRIMARY_METRIC = "min_dist_m"   # paired metric for V1/V3 (same-time min sep)

V1_COLUMNS = [
    "ped_arm", "pred", "plan", "n_pairs", "n_arm_gt_replay", "n_arm_lt_replay",
    "mean_delta_m", "sign_p", "wilcoxon_p",
    "arm_coll_encs", "replay_coll_encs", "coll_arm_only", "coll_replay_only",
    "mcnemar_p",
]
V3_COLUMNS = [
    "ped_arm", "pred", "n_pairs", "n_robust_gt_single", "mean_delta_m",
    "sign_p", "wilcoxon_p", "single_coll_encs", "robust_coll_encs",
    "n_time_pairs", "mean_time_cost_s",
]
V2_COLUMNS = [
    "verdict_kind", "pred_or_plan", "ped_arm", "value", "significant",
    "detail", "gap_p",
]

STATUS_INVARIANT = "全アームで不変（反応性仮定に頑健）"
STATUS_INVARIANT_NS = ("全アームで方向不変だが全アーム非有意"
                       "（頑健・ただし検出力は限定的）")
STATUS_UNDETERMINED = ("方向不一致だが不一致側に有意性なし"
                       "（判定不能・検出力限界の可能性）")
STATUS_REVERSAL = "反転あり（反応性交絡の実証）"
STATUS_NODATA = "アーム欠損（判定不能）"
STATUS_DEGENERATE = "全アームで縮退（Δ=0・robust ≡ single）"


def _json_float(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else x


# ---------------------------------------------------------------------------
# Encounter-level collapse (seeds -> mean; the unit of analysis is the
# encounter, disclosed in the REPORT)
# ---------------------------------------------------------------------------

def encounter_means(runs: pd.DataFrame) -> pd.DataFrame:
    df = runs.copy()
    df["min_ttc_s"] = pd.to_numeric(df["min_ttc_s"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan)
    df["collided"] = (pd.to_numeric(df["collision_count"],
                                    errors="coerce") > 0).astype(float)
    df["goal"] = df["goal_reached"].astype(bool).astype(float)
    g = df.groupby(["ped_arm", "pred", "plan", "enc_id"], sort=True)
    out = g.agg(
        min_dist_m=("min_dist_m", "mean"),
        min_ttc_s=("min_ttc_s", "mean"),
        time_s=("time_s", "mean"),
        progress=("progress", "mean"),
        collision_rate=("collided", "mean"),
        goal_rate=("goal", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return out.sort_values(["ped_arm", "pred", "plan", "enc_id"],
                           kind="mergesort").reset_index(drop=True)


def _cell(em: pd.DataFrame, ped_arm: str, pred: str, plan: str) -> pd.DataFrame:
    m = em[(em.ped_arm == ped_arm) & (em.pred == pred) & (em.plan == plan)]
    return m.set_index("enc_id").sort_index()


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar (binomial) p for paired binary outcomes; NaN if b+c=0."""
    n = b + c
    if n == 0:
        return float("nan")
    return float(stats.binomtest(min(b, c), n, 0.5).pvalue)


def _mcnemar_family_bh(rows: List[Dict]) -> Dict:
    """Within-arm BH over the emitted McNemar family members.

    Mirrors headline_tests: the cv/robust duplicate is excluded, NaN p
    (no discordant pairs) drop out inside ``adjust``. Returns the survival
    summary the narrative quotes instead of an uncorrected minimum p.
    """
    ps = [r["mcnemar_p"] for r in rows
          if not (r["pred"] == "cv" and r["plan"] == "robust")]
    res = adjust(ps, alpha=ALPHA)
    finite_q = [q for q in res["bh_q"] if q is not None and not np.isnan(q)]
    return {
        "m": len(ps),
        "n_reject": int(sum(bool(x) for x in res["bh_reject"])),
        "min_q": min(finite_q) if finite_q else float("nan"),
    }


# ---------------------------------------------------------------------------
# V1: reactivity confound (SFM arm vs replay, paired per encounter)
# ---------------------------------------------------------------------------

def v1_rows(em: pd.DataFrame, arms: List[str], preds: List[str],
            plans: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for arm in arms:
        for pred in preds:
            for plan in plans:
                a = _cell(em, arm, pred, plan)
                r = _cell(em, REFERENCE_ARM, pred, plan)
                common = a.index.intersection(r.index)
                if len(common) == 0:
                    continue
                pools = {"replay": r.loc[common, PRIMARY_METRIC].tolist(),
                         "arm": a.loc[common, PRIMARY_METRIC].tolist()}
                s = _paired_stats(pools, "replay", "arm")  # d = arm - replay
                if s is None:
                    continue
                a_coll = (a.loc[common, "collision_rate"] > 0)
                r_coll = (r.loc[common, "collision_rate"] > 0)
                b = int((a_coll & ~r_coll).sum())   # arm-only collisions
                c = int((~a_coll & r_coll).sum())   # replay-only collisions
                rows.append({
                    "ped_arm": arm, "pred": pred, "plan": plan,
                    "n_pairs": s["n_pairs"],
                    "n_arm_gt_replay": s["n_real_gt_sim"],
                    "n_arm_lt_replay": s["n_real_lt_sim"],
                    "mean_delta_m": round(float(s["mean_gap"]), 4),
                    "sign_p": s["sign_p"],
                    "wilcoxon_p": s["wilcoxon_p"],
                    "arm_coll_encs": int(a_coll.sum()),
                    "replay_coll_encs": int(r_coll.sum()),
                    "coll_arm_only": b,
                    "coll_replay_only": c,
                    "mcnemar_p": _mcnemar_exact(b, c),
                })
    return rows


# ---------------------------------------------------------------------------
# V3 (and V2's robust-gain input): robust - single within one ped arm
# ---------------------------------------------------------------------------

def robust_gain_rows(em: pd.DataFrame, arms: List[str],
                     preds: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for arm in arms:
        for pred in preds:
            s_cell = _cell(em, arm, pred, "single")
            r_cell = _cell(em, arm, pred, "robust")
            common = s_cell.index.intersection(r_cell.index)
            if len(common) == 0:
                continue
            pools = {"single": s_cell.loc[common, PRIMARY_METRIC].tolist(),
                     "robust": r_cell.loc[common, PRIMARY_METRIC].tolist()}
            s = _paired_stats(pools, "single", "robust")  # d = robust - single
            if s is None:
                continue
            # Completion-time cost only over encounters where BOTH plans
            # reached the goal in every seed (censoring-honest pairs).
            both_goal = common[(s_cell.loc[common, "goal_rate"] == 1.0)
                               & (r_cell.loc[common, "goal_rate"] == 1.0)]
            time_cost = (r_cell.loc[both_goal, "time_s"]
                         - s_cell.loc[both_goal, "time_s"])
            rows.append({
                "ped_arm": arm, "pred": pred,
                "n_pairs": s["n_pairs"],
                "n_robust_gt_single": s["n_real_gt_sim"],
                "mean_delta_m": round(float(s["mean_gap"]), 4),
                "sign_p": s["sign_p"],
                "wilcoxon_p": s["wilcoxon_p"],
                # not in V3_COLUMNS (the CSV stays byte-identical); carried for
                # the ledger sidecar's wilcoxon family (thesis review M2a)
                "wilcoxon_stat": s["wilcoxon_stat"],
                "single_coll_encs": int((s_cell.loc[common, "collision_rate"]
                                         > 0).sum()),
                "robust_coll_encs": int((r_cell.loc[common, "collision_rate"]
                                         > 0).sum()),
                "n_time_pairs": int(len(both_goal)),
                "mean_time_cost_s": round(float(time_cost.mean()), 4)
                if len(both_goal) else float("nan"),
            })
    return rows


# ---------------------------------------------------------------------------
# V2: verdict preservation across pedestrian arms (tri-state)
# ---------------------------------------------------------------------------

def tristate(values: List[Optional[bool]],
             undetermined: List[bool]) -> str:
    """RQ1b _sensitivity_status discipline for one verdict across arms.

    ``values`` is the pure per-arm DIRECTION (True/False), ``undetermined``
    is True for arms whose direction lacks statistical significance. A
    reversal claim needs significant evidence on BOTH sides of a
    disagreement; a disagreement carried only by non-significant arms is
    reported as undetermined (a detection-power artifact must not read as a
    reversal). Agreement does not need significance to count as invariant,
    but all-non-significant agreement is annotated as such.
    """
    if not values or any(v is None for v in values):
        return STATUS_NODATA
    vals = [bool(v) for v in values]
    undet = [bool(u) for u in undetermined]
    if len(set(vals)) <= 1:
        return STATUS_INVARIANT_NS if all(undet) else STATUS_INVARIANT
    has_sig_pos = any(v and not u for v, u in zip(vals, undet))
    has_sig_neg = any((not v) and (not u) for v, u in zip(vals, undet))
    if has_sig_pos and has_sig_neg:
        return STATUS_REVERSAL
    return STATUS_UNDETERMINED


def v2_robust_gain_preservation(gain_rows: List[Dict], arms: List[str],
                                preds: List[str]) -> List[Dict]:
    """Per predictor: is the robust-gain direction preserved across arms?"""
    by = {(r["ped_arm"], r["pred"]): r for r in gain_rows}
    out: List[Dict] = []
    for pred in preds:
        vals, undet, details = [], [], []
        n_arms = n_ties = 0
        missing = False
        for arm in arms:
            r = by.get((arm, pred))
            if r is None:
                missing = True
                details.append(f"{arm}:欠損")
                continue
            n_arms += 1
            if r["mean_delta_m"] == 0:
                # Exact tie (e.g. the deterministic CV predictor, whose
                # 20-sample distribution collapses so robust == single
                # bit-for-bit): no direction to vote with.
                n_ties += 1
                details.append(f"{arm}:0(縮退)")
                continue
            positive = r["mean_delta_m"] > 0
            sig = (r["wilcoxon_p"] is not None
                   and not np.isnan(r["wilcoxon_p"])
                   and r["wilcoxon_p"] < ALPHA)
            vals.append(bool(positive))
            undet.append(not sig)
            details.append(f"{arm}:{'+' if positive else '-'}"
                           f"{'' if sig else '(n.s.)'}")
        if missing:
            value = STATUS_NODATA
        elif n_arms and n_ties == n_arms:
            value = STATUS_DEGENERATE
        else:
            value = tristate(vals, undet)
        out.append({
            "verdict_kind": "robust_gain_direction",
            "pred_or_plan": pred,
            "ped_arm": "ALL",
            "value": value,
            "significant": "",
            "detail": " ".join(details),
        })
    return out


def _ranking_for(em: pd.DataFrame, arm: str, plan: str,
                 preds: List[str]) -> Optional[Dict]:
    """Predictor ranking by encounter-mean min separation (asc = dangerous).

    The most-dangerous identity is significance-gated: paired Wilcoxon
    between the bottom two predictors' per-encounter min_dist.
    """
    cells = {p: _cell(em, arm, p, plan) for p in preds}
    cells = {p: c for p, c in cells.items() if len(c)}
    if len(cells) < 2:
        return None
    means = {p: float(c[PRIMARY_METRIC].mean()) for p, c in cells.items()}
    order = sorted(means, key=lambda p: means[p])
    bottom, runner = order[0], order[1]
    common = cells[bottom].index.intersection(cells[runner].index)
    d = (cells[runner].loc[common, PRIMARY_METRIC]
         - cells[bottom].loc[common, PRIMARY_METRIC]).to_numpy()
    if len(d) and np.any(d != 0):
        w = stats.wilcoxon(d)
        wp, wstat = float(w.pvalue), float(w.statistic)
    else:
        wp, wstat = float("nan"), float("nan")
    return {
        "order": order, "means": means, "most_dangerous": bottom,
        "runner": runner, "gap_n_pairs": int(len(d)),
        "gap_p": wp, "gap_stat": wstat,
        "significant": (not np.isnan(wp)) and wp < ALPHA,
    }


def v2_ranking_preservation(em: pd.DataFrame, arms: List[str],
                            preds: List[str],
                            plans: List[str]) -> List[Dict]:
    out: List[Dict] = []
    for plan in plans:
        vals, undet, details = [], [], []
        per_arm: Dict[str, Dict] = {}
        for arm in arms:
            r = _ranking_for(em, arm, plan, preds)
            per_arm[arm] = r
            if r is None:
                vals.append(None)
                undet.append(True)
                details.append(f"{arm}:欠損")
                continue
            details.append(
                f"{arm}:{'<'.join(r['order'])}"
                f"({'sig' if r['significant'] else 'n.s.'})")
        ref = per_arm.get(REFERENCE_ARM)
        if ref is not None:
            for arm in arms:
                r = per_arm[arm]
                if r is None:
                    continue
                vals.append(bool(r["most_dangerous"] == ref["most_dangerous"]))
                undet.append(not (r["significant"] and ref["significant"]))
        out.append({
            "verdict_kind": "most_dangerous_predictor",
            "pred_or_plan": plan,
            "ped_arm": "ALL",
            "value": tristate(vals, undet) if vals else STATUS_NODATA,
            "significant": "",
            "detail": " ".join(details),
        })
        for arm in arms:
            r = per_arm[arm]
            if r is None:
                continue
            out.append({
                "verdict_kind": "predictor_ranking",
                "pred_or_plan": plan,
                "ped_arm": arm,
                "value": "<".join(r["order"]),
                "significant": bool(r["significant"]),
                "gap_p": r["gap_p"],
                # not in V2_COLUMNS; carried for the ledger sidecar's
                # ranking-gate family (thesis review M2a)
                "gap_stat": r["gap_stat"],
                "gap_n_pairs": r["gap_n_pairs"],
                "most_dangerous": r["most_dangerous"],
                "runner": r["runner"],
                "detail": " ".join(f"{p}={r['means'][p]:.3f}"
                                   for p in r["order"]),
            })
    return out


# ---------------------------------------------------------------------------
# Ledger sidecar (rq3.* namespace)
# ---------------------------------------------------------------------------

def headline_tests(v1: List[Dict], gains: List[Dict],
                   v2: Optional[List[Dict]] = None) -> List[Dict]:
    """Ledger records: V1 (sign + McNemar), V3 (sign + Wilcoxon), V2 gates.

    Emission order is append-only relative to the pre-M2a sidecar: the first
    46 records (V1 sign/McNemar + replay V3 sign) are byte-identical to the
    frozen layout; the thesis cross-cut review M2a families
    (``rq3_v3_robust_real_ctrl`` / ``rq3_v3_robust_wilcoxon`` /
    ``rq3_v2_ranking_gates``) follow at the end, all auxiliary. The V1
    Wilcoxon companions stay passthrough fields of the sign records (the
    user-approved canonical design counts sign tests only); the V3 Wilcoxon
    companions ARE emitted because the thesis table quotes their p values.
    """
    tests: List[Dict] = []
    for r in v1:
        canonical = r["ped_arm"] == CANONICAL_ARM
        cv_degenerate = r["pred"] == "cv" and r["plan"] == "robust"
        rec = {
            "test_id": (f"rq3.v1.reactivity_sign.{r['ped_arm']}."
                        f"{r['pred']}.{r['plan']}"),
            "description": (
                f"Paired per-encounter sign test: {r['ped_arm']} SFM arm vs "
                f"replay reference, same-time min separation "
                f"({r['pred']}/{r['plan']}, seeds collapsed to encounter "
                f"means, n={r['n_pairs']})"),
            "family": ("rq3_v1_reactivity" if canonical
                       else "rq3_v1_reactivity_ctrl"),
            "protocol": "paired_encounters",
            "p_value": _json_float(r["sign_p"]),
            "statistic": float(r["n_arm_gt_replay"]),
            "sidedness": "two-sided",
            "n_pairs": r["n_pairs"],
            "mean_gap_m": _json_float(r["mean_delta_m"]),
            "wilcoxon_p": _json_float(r["wilcoxon_p"]),
            "headline": canonical,
            "note": ("canonical reactivity-confound measurement "
                     "(user-approved family, 2026-07-03)" if canonical else
                     "control arm for the reactivity axis (auxiliary)"),
        }
        if cv_degenerate:
            rec["note"] += ("; degenerate with the cv/single member (the CV "
                            "predictor is deterministic, so its 20-sample "
                            "distribution collapses and robust == single "
                            "bit-for-bit) -- kept in the family per the "
                            "approved 6-test design; with its p=1.0 here the "
                            "duplication can only be conservative (a "
                            "duplicated SMALL p could instead shift BH ranks "
                            "anti-conservatively, which is why the McNemar "
                            "family excludes its duplicate)")
        if not canonical:
            rec["auxiliary"] = True
        tests.append(rec)
        mp = _json_float(r["mcnemar_p"])
        if mp is not None and not cv_degenerate:
            # Family structure mirrors the sign-test split (canonical arm vs
            # controls) instead of one all-arm pool: a 23-test mixed family
            # would dilute the calibrated arm's signal with control-arm
            # hypotheses that are not the confirmatory question. The
            # bit-identical cv/robust duplicate is NOT emitted (its p would
            # re-enter the family and shift BH ranks both ways); the
            # cv/single record carries the disclosure.
            note = "rare-event companion to the min-separation sign test"
            if r["pred"] == "cv" and r["plan"] == "single":
                note += ("; the cv/robust cell is bit-identical (CV "
                         "distribution collapse) and is deliberately not "
                         "emitted as a second family member")
            tests.append({
                "test_id": (f"rq3.v1.collision_mcnemar.{r['ped_arm']}."
                            f"{r['pred']}.{r['plan']}"),
                "description": (
                    f"Exact McNemar on paired per-encounter collision "
                    f"occurrence, {r['ped_arm']} vs replay "
                    f"({r['pred']}/{r['plan']}; discordant pairs "
                    f"{r['coll_arm_only']}+{r['coll_replay_only']})"),
                "family": ("rq3_v1_collision_mcnemar" if canonical
                           else "rq3_v1_collision_mcnemar_ctrl"),
                "protocol": "paired_encounters",
                "auxiliary": True,
                "p_value": mp,
                "statistic": float(r["coll_arm_only"]),
                "sidedness": "two-sided",
                "n_pairs": r["n_pairs"],
                "headline": False,
                "note": note,
            })
    for r in gains:
        if r["ped_arm"] != REFERENCE_ARM:
            continue
        tests.append({
            "test_id": f"rq3.v3.robust_gain_sign.replay.{r['pred']}",
            "description": (
                f"Paired per-encounter sign test: robust vs true-single min "
                f"separation within the replay arm ({r['pred']}, "
                f"n={r['n_pairs']})"),
            "family": "rq3_v3_robust_real",
            "protocol": "paired_encounters",
            "auxiliary": True,
            "p_value": _json_float(r["sign_p"]),
            "statistic": float(r["n_robust_gt_single"]),
            "sidedness": "two-sided",
            "n_pairs": r["n_pairs"],
            "mean_gap_m": _json_float(r["mean_delta_m"]),
            "wilcoxon_p": _json_float(r["wilcoxon_p"]),
            "headline": False,
            "note": ("robust gain on real encounter geometry; auxiliary by "
                     "design (RQ1b claim-1 is the canonical robust-gain "
                     "family)"),
        })
    # --- thesis cross-cut review M2a: the remaining thesis-quoted p values ---
    # Non-replay V3 sign tests (thesis table 8.3 quotes them). The cv cells
    # are degenerate (deterministic predictor => robust == single bit-for-bit,
    # no direction): emitted with a null p so the ledger discloses the
    # degeneracy per arm, mirroring the replay.cv convention; NaN p's are not
    # counted in the family size (multiplicity.adjust drops them).
    for r in gains:
        if r["ped_arm"] == REFERENCE_ARM:
            continue
        rec = {
            "test_id": (f"rq3.v3.robust_gain_sign.{r['ped_arm']}."
                        f"{r['pred']}"),
            "description": (
                f"Paired per-encounter sign test: robust vs true-single min "
                f"separation within the {r['ped_arm']} SFM arm ({r['pred']}, "
                f"n={r['n_pairs']})"),
            "family": "rq3_v3_robust_real_ctrl",
            "protocol": "paired_encounters",
            "auxiliary": True,
            "p_value": _json_float(r["sign_p"]),
            "statistic": float(r["n_robust_gt_single"]),
            "sidedness": "two-sided",
            "n_pairs": r["n_pairs"],
            "mean_gap_m": _json_float(r["mean_delta_m"]),
            "headline": False,
            "note": ("robust gain under an SFM control arm (reactivity "
                     "sensitivity companion to rq3_v3_robust_real; ledger "
                     "registration of a thesis-quoted p, review M2a)"),
        }
        if _json_float(r["sign_p"]) is None:
            rec["note"] += ("; p undefined: degenerate cell -- every paired "
                            "difference is exactly zero (here the "
                            "deterministic CV predictor collapses to "
                            "robust == single bit-for-bit)")
        tests.append(rec)
    # V3 Wilcoxon companions over the SAME paired differences, all arms
    # (thesis table 8.3 quotes all 10 finite cells). Degenerate cv cells have
    # no Wilcoxon (all-zero d) and are not emitted -- the sign families carry
    # the degeneracy disclosure (mirrors the McNemar NaN-skip convention).
    for r in gains:
        wp = _json_float(r["wilcoxon_p"])
        if wp is None:
            continue
        tests.append({
            "test_id": (f"rq3.v3.robust_gain_wilcoxon.{r['ped_arm']}."
                        f"{r['pred']}"),
            "description": (
                f"Paired per-encounter Wilcoxon signed-rank: robust vs "
                f"true-single min separation within the {r['ped_arm']} arm "
                f"({r['pred']}, n={r['n_pairs']})"),
            "family": "rq3_v3_robust_wilcoxon",
            "protocol": "paired_encounters",
            "auxiliary": True,
            "p_value": wp,
            "statistic": _json_float(r.get("wilcoxon_stat")),
            "sidedness": "two-sided",
            "n_pairs": r["n_pairs"],
            "mean_gap_m": _json_float(r["mean_delta_m"]),
            "headline": False,
            "note": ("magnitude-aware companion over the SAME paired "
                     "differences as the V3 sign tests; ledger registration "
                     "of a thesis-quoted p (review M2a)"),
        })
    # V2 most-dangerous-predictor gates: the significance gate behind each
    # predictor_ranking verdict (previously only the significant boolean was
    # persisted in v2_verdicts.csv; the thesis table 8.2 relies on the gate).
    for r in (v2 or []):
        if r["verdict_kind"] != "predictor_ranking":
            continue
        rec = {
            "test_id": (f"rq3.v2.ranking_gap_wilcoxon.{r['ped_arm']}."
                        f"{r['pred_or_plan']}"),
            "description": (
                f"Paired per-encounter Wilcoxon signed-rank: bottom-two "
                f"predictor gap ({r['most_dangerous']} vs {r['runner']}) in "
                f"min separation, {r['ped_arm']} arm / {r['pred_or_plan']} "
                f"plan (n={r['gap_n_pairs']})"),
            "family": "rq3_v2_ranking_gates",
            "protocol": "paired_encounters",
            "auxiliary": True,
            "p_value": _json_float(r["gap_p"]),
            "statistic": _json_float(r.get("gap_stat")),
            "sidedness": "two-sided",
            "n_pairs": r["gap_n_pairs"],
            "headline": False,
            "note": ("significance gate of the V2 most-dangerous-predictor "
                     "verdict (tri-state input); a non-significant gate "
                     "reads as detection-power limit, not invariance "
                     "evidence. Ledger registration of a thesis-quoted "
                     "gate (review M2a)"),
        }
        if rec["p_value"] is None:
            rec["note"] += ("; p undefined: degenerate gap -- every paired "
                            "difference between the bottom-two predictors "
                            "is exactly zero")
        tests.append(rec)
    return tests


# ---------------------------------------------------------------------------
# REPORT.md (all verdict-bearing prose derived from the tables)
# ---------------------------------------------------------------------------

def _md_table(rows: List[Dict], cols: List[str]) -> List[str]:
    L = ["| " + " | ".join(cols) + " |",
         "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append("nan" if np.isnan(v) else f"{v:.4g}")
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return L


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "nan"
    return f"{p:.3g}"


def _v1_narrative(v1: List[Dict]) -> List[str]:
    L: List[str] = []
    calib_bh = None
    for arm in [CANONICAL_ARM] + CONTROL_ARMS:
        rows = [r for r in v1 if r["ped_arm"] == arm]
        if not rows:
            continue
        pos = [r for r in rows if r["mean_delta_m"] > 0]
        sig = [r for r in rows if r["sign_p"] is not None
               and not np.isnan(r["sign_p"]) and r["sign_p"] < ALPHA]
        deltas = [r["mean_delta_m"] for r in rows]
        arm_only = sum(r["coll_arm_only"] for r in rows)
        replay_only = sum(r["coll_replay_only"] for r in rows)
        bh = _mcnemar_family_bh(rows)
        if arm == CANONICAL_ARM:
            calib_bh = bh
        mc = (f"McNemar family BH（m={bh['m']}・cv/robust 縮退除外）: "
              f"{bh['n_reject']}/{bh['m']} 件が q<{ALPHA} で生存"
              + (f"・最小 q={bh['min_q']:.3g}"
                 if not np.isnan(bh["min_q"]) else ""))
        L.append(
            f"- `{arm}` vs replay: {len(rows)} セル中 {len(pos)} セルで "
            f"min-separation が replay より大（Δ範囲 "
            f"[{min(deltas):+.3f}, {max(deltas):+.3f}] m）、"
            f"符号検定 p<{ALPHA} は {len(sig)}/{len(rows)} セル。"
            f"衝突不一致の対（セル横断延べ・同一遭遇の重複含む）は "
            f"replay 側のみ {replay_only} 件 vs {arm} 側のみ {arm_only} 件。"
            f"{mc}。")
    if all(sum(r["coll_arm_only"] for r in v1 if r["ped_arm"] == a) == 0
           for a in {r["ped_arm"] for r in v1}):
        if calib_bh is not None and calib_bh["n_reject"] > 0:
            L.append(
                "- 全 SFM アーム・全セルで `coll_arm_only`=0 かつ canonical "
                "アーム（calib）の McNemar family が BH 生存: SFM 歩行者に"
                "置き換えると replay（記録実歩行者・非反応）で起きる衝突が"
                "消える方向にのみ不一致が発生する＝歩行者反応性の仮定が"
                "閉ループ安全結果を直接変える（反応性交絡）ことの"
                "対応対による測定。")
        else:
            q_txt = (f"最小 q={calib_bh['min_q']:.3g}"
                     if calib_bh is not None
                     and not np.isnan(calib_bh["min_q"]) else "q なし")
            L.append(
                f"- 全 SFM アーム・全セルで `coll_arm_only`=0（方向は一貫して"
                f"「SFM で衝突が消える」側）だが、canonical アーム"
                f"（{CANONICAL_ARM}）の McNemar family は BH を生存せず"
                f"（{q_txt}）＝方向的示唆にとどまり確証的主張はしない"
                f"（制御アームの family 生存状況は各行のとおりで、"
                f"確証枠には算入しない）。")
    return L


def censoring_rows(runs: pd.DataFrame) -> List[Dict]:
    """Per-arm censoring / recorded-ego deviation summary (review M2)."""
    out = []
    for arm, g in runs.groupby("ped_arm", sort=True):
        out.append({
            "ped_arm": arm,
            "n_runs": len(g),
            "censored_frac": round(float(g["censored"].mean()), 3),
            "goal_frac": round(float(g["goal_reached"].mean()), 3),
            "progress_mean": round(float(pd.to_numeric(
                g["progress"], errors="coerce").mean()), 3),
            "ego_dev_mean_m": round(float(pd.to_numeric(
                g["ego_dev_mean_m"], errors="coerce").mean()), 3),
            "ego_dev_max_m": round(float(pd.to_numeric(
                g["ego_dev_max_m"], errors="coerce").mean()), 3),
        })
    return out


def medoid_reference_rows(em: pd.DataFrame) -> List[Dict]:
    """Descriptive medoid-vs-true-single contrast (review F4 companion).

    Uses the medoid reference cells (sgan x replay/calib, user-approved
    scope). Paired per-encounter min-separation difference and collision
    counts; descriptive only (no ledger entry -- the F4 point is that the
    two 'single' definitions measurably differ, not a new hypothesis).
    """
    out = []
    for arm in sorted(em[em.plan == "medoid"]["ped_arm"].unique()):
        for pred in sorted(em[(em.plan == "medoid")
                              & (em.ped_arm == arm)]["pred"].unique()):
            med = _cell(em, arm, pred, "medoid")
            drw = _cell(em, arm, pred, "single")
            common = med.index.intersection(drw.index)
            if len(common) == 0:
                continue
            d = (med.loc[common, PRIMARY_METRIC]
                 - drw.loc[common, PRIMARY_METRIC])
            out.append({
                "ped_arm": arm, "pred": pred, "n_pairs": int(len(common)),
                "mean_medoid_minus_draw_m": round(float(d.mean()), 4),
                "max_abs_diff_m": round(float(d.abs().max()), 4),
                "medoid_coll_encs": int((med.loc[common, "collision_rate"]
                                         > 0).sum()),
                "draw_coll_encs": int((drw.loc[common, "collision_rate"]
                                       > 0).sum()),
            })
    return out


def write_report(runs: pd.DataFrame, census: pd.DataFrame,
                 v1: List[Dict], v2: List[Dict], v3: List[Dict],
                 gains: List[Dict], em: pd.DataFrame) -> str:
    arms_present = sorted(runs["ped_arm"].unique())
    n_runs = len(runs)
    n_enc = runs["enc_id"].nunique()
    n_censored = int(runs["censored"].sum())
    n_goal = int(runs["goal_reached"].sum())
    n_coll = int((runs["collision_count"] > 0).sum())

    L: List[str] = []
    L.append("# RQ3 REPORT: 実データ接地閉ループ（Closing the Loop の歩行者版）")
    L.append("")
    L.append("生成: `examples/make_rq3_report.py`（全 verdict は純関数、prose は表から機械生成）。"
             "入力: `outputs/rq3_realloop/all_runs.csv`（`run_rq3_realloop.py "
             "--report-only` でキャッシュから再構築可能）。")
    L.append("")
    L.append(f"判定しきい値: ALPHA={ALPHA}（tri-state の有意性ゲート）・"
             f"主指標={PRIMARY_METRIC}（同時刻 ego-歩行者中心間距離の最小値）・"
             f"対単位=encounter（seed は encounter 内平均で先に潰す＝擬似反復の排除）。")
    L.append("")

    L.append("## 0. 実験構成と器具開示")
    L.append("")
    L.append("プランナ駆動 ego を録画 CITR 遭遇ジオメトリに接地: 参照経路=録画 ego 軌道の"
             "スプライン（0.5 m 間引き）・目標速度=録画 ego 速度中央値・初期状態=録画開始"
             "フレーム・total_time=録画窓長（timeout=censoring）。歩行者アーム:")
    L.append("")
    arm_rows = []
    for name, spec in PED_ARMS.items():
        arm_rows.append({
            "arm": name, "kind": spec["ped_kind"],
            "sigma": spec["sigma"] if spec["sigma"] is not None else "-",
            "v0": spec["v0"] if spec["v0"] is not None else "-",
            "speed_regime": (spec["speed_regime"]
                             if spec["ped_kind"] == "sfm" else "replay"),
        })
    L += _md_table(arm_rows, ["arm", "kind", "sigma", "v0", "speed_regime"])
    L.append("")
    L.append("- 固定器具: scenario_01.yaml の verified フェイルセーフ/エンベロープ/"
             "プランナ定数（掃引はスコープ外）・SGAN/LSTM チェックポイント "
             "zara1_12_model.pt・衝突判定は同時刻位置のみ。")
    L.append("- 幾何: ego_radius=1.0 m / ped_radius=0.30 m（RQ2 較正整合）。"
             "実車寸法との差は制約 (limitation) 参照。")
    L.append("- 観測ウォームアップ: 全アームでフレーム0速度の等速バックキャスト"
             "（warmup_source=backcast、窓前実録画は ego NaN/在席非保証のため不使用）"
             "＝ t=0 の観測履歴と予測はアーム間で同一、差分は純粋にアーム動力学由来。")
    L.append("")

    L.append("## 1. ランと打ち切りの census")
    L.append("")
    L.append(f"- ラン総数 {n_runs}（アーム: {', '.join(arms_present)}）・"
             f"遭遇 {n_enc}/{len(census)}（適格 "
             f"{int(census['eligible'].sum())}/{len(census)}）。")
    L.append(f"- goal 到達 {n_goal}/{n_runs}・timeout 打ち切り（censored）"
             f" {n_censored}/{n_runs}・衝突ラン {n_coll}/{n_runs}。"
             "完了時間の対比較は両アーム goal 到達の遭遇のみで行う（censoring 対処）。")
    L.append("")
    cens = censoring_rows(runs)
    L += _md_table(cens, ["ped_arm", "n_runs", "censored_frac", "goal_frac",
                          "progress_mean", "ego_dev_mean_m", "ego_dev_max_m"])
    L.append("")
    prog_all = float(pd.to_numeric(runs["progress"], errors="coerce").mean())
    dev_all = float(pd.to_numeric(runs["ego_dev_mean_m"],
                                  errors="coerce").mean())
    L.append(f"**打ち切りの帰結（review M2）**: プランナ ego は録画ドライバより"
             f"保守的で、録画窓内に録画終端へ到達しないランが多数"
             f"（progress 平均 {prog_all:.2f}・録画 ego との時刻整合偏差 平均 "
             f"{dev_all:.2f} m）。したがって V1/V3 の対比較は「録画窓で切り"
             f"詰めた曝露・録画経路から乖離し得る ego」の下での測定であり、"
             f"遭遇後半の相互作用は部分的にしか観測されない。対内では両アーム"
             f"が同一の打ち切り規則・同一の録画窓を共有するため比較自体は"
             f"保存されるが、絶対値（衝突率・min_dist）は完走条件下の値では"
             f"ない点に注意。")
    L.append("")

    L.append("## 2. V1: 反応性交絡の直接測定（SFM アーム vs replay・対応差）")
    L.append("")
    L.append("Δ = SFM アーム − replay（正= SFM 歩行者が譲るぶん ego の余裕が水増しされる方向）。"
             f"canonical family = `rq3_v1_reactivity`（{CANONICAL_ARM} の全 "
             "pred×plan、BH-FDR は ledger 参照）。")
    L.append("")
    L += _md_table(v1, V1_COLUMNS)
    L.append("")
    L += _v1_narrative(v1)
    L.append("")

    L.append("## 3. V2: ベンチマーク判定の保存性（tri-state）")
    L.append("")
    L.append(f"replay アームの判定が headline 候補・SFM アームは感度幅。有意性ゲート"
             f"（Wilcoxon p<{ALPHA}）を通らない不一致は『検出力限界』であり反転とは"
             "読まない（RQ1b の tristate 規律）。")
    L.append("")
    L += _md_table(v2, V2_COLUMNS)
    L.append("")

    L.append("## 4. V3: robust 利得の実ジオメトリ検証（replay 参照点つき・auxiliary）")
    L.append("")
    L.append("Δ = robust − single（true-single draw、review F4 対応）。"
             "replay アーム＝『実際の歩行者がした行動』の下での利得。全アーム分を併記。")
    L.append("")
    L += _md_table(gains, V3_COLUMNS)
    L.append("")

    L.append("### 4.1 medoid 参考対比（review F4 の実測差・記述のみ）")
    L.append("")
    med_rows = medoid_reference_rows(em)
    if med_rows:
        L += _md_table(med_rows, ["ped_arm", "pred", "n_pairs",
                                  "mean_medoid_minus_draw_m", "max_abs_diff_m",
                                  "medoid_coll_encs", "draw_coll_encs"])
        L.append("")
        L.append("medoid（predict_single_best 既定＝分散抑制代表値）と "
                 "true-single draw は同一 seed でも異なる閉ループ軌道を生む"
                 "（AVEC/RQ1b の single 条件は medoid 相手の測定だったという "
                 "F4 の定量的裏付け）。ledger 検定は張らない（新仮説ではなく"
                 "計装の開示）。")
    else:
        L.append("（medoid 参考ランなし）")
    L.append("")

    min_sep_rec = float(census["recorded_min_sep_m"].min()) \
        if "recorded_min_sep_m" in census else float("nan")
    disc_radius = 1.0 + 0.30  # ego_radius + ped_radius (RQ2-consistent)
    n_below = int((census["recorded_min_sep_m"] < disc_radius).sum()) \
        if "recorded_min_sep_m" in census else 0
    L.append("## 5. 制約 (limitations)")
    L.append("")
    L.append(f"- **円板近似**: ego_radius=1.0 m は実車の外接円近似で、録画済み"
             f"実遭遇にも最接近 {min_sep_rec:.2f} m（< {disc_radius:.2f} m "
             f"判定半径和・{n_below} 遭遇）の事例がある＝『衝突』は保守的な"
             f"器具定義であり実接触ではない。")
    L.append("- **打ち切り窓上の測定**: §1 の通り、録画窓長を total_time と"
             "する設計は censoring を生む（対内で共有・開示済み）。")
    L.append("- **replay の非反応性**: replay 歩行者はプランナ ego に反応しない"
             "（Closing the Loop の log-replay と同じ設計選択を参照点として利用）。"
             "ego が録画から乖離した後の replay 軌道は反実仮想として読めない。"
             "バイアスの向きは既知: replay は歩行者側回避を過小評価（衝突を過大に）、"
             "SFM は過大評価（衝突を過小に）しうるため、両アームは実世界の挙動を"
             "挟み込む参照枠 (bracketing) として読む。")
    L.append("- **バックキャスト観測履歴**: 直線 8 フレームは SGAN の学習分布と異なるが、"
             "全アーム同一条件のため対比較は保存される。")
    L.append("- **フェイルセーフ定数**: S1-S3 で手調整された固定器具（未掃引）。")
    L.append("- **medoid 参考条件**: sgan × {replay, calib} のみ（計算量制御・"
             "ユーザー承認 2026-07-03）。")
    L.append("- **統計単位**: encounter（n<=26）。同一 encounter を共有する"
             "pred×plan セル間の検定は独立でない（family 補正は ledger の "
             "canonical/auxiliary 区分で処理）。")
    L.append("")
    L.append("## 6. ledger 登載")
    L.append("")
    L.append("`headline_tests.json`（namespace rq3.*）: canonical = "
             "`rq3_v1_reactivity`（6 検定）、auxiliary = "
             "`rq3_v1_reactivity_ctrl` / `rq3_v1_collision_mcnemar`（calib）"
             "/ `rq3_v1_collision_mcnemar_ctrl`（制御アーム） / "
             "`rq3_v3_robust_real` / `rq3_v3_robust_real_ctrl`"
             "（非 replay アームの V3 符号検定・cv は p 未定義で縮退開示） / "
             "`rq3_v3_robust_wilcoxon`（V3 全アームの Wilcoxon 併記） / "
             "`rq3_v2_ranking_gates`（V2 最危険予測器判定の有意性ゲート）。"
             "cv/robust の McNemar はビット同一縮退の"
             "ため未登載（cv/single の note 参照）。V1 の Wilcoxon 併記は"
             "符号検定レコードの passthrough フィールドに記録し別仮説として"
             "数えない（canonical 6 検定設計）。"
             "`examples/make_multiplicity_ledger.py` の再実行で台帳へ自動編入。")
    L.append("")
    L.append("**台帳への意図差分の開示**（静的記録: 2026-07-03 の台帳再生成時に"
             "機械検証した監査結果の転記であり、本 REPORT の再生成では"
             "再計算されない）: canonical family の追加により "
             "study-wide 補正（overall_* 列）は全既存行で再計算される"
             "（canonical 21→27 検定）。既存行の within-family 列は全行"
             "バイト不変を機械検証済み。overall 層の判定反転は 4 件・全て "
             "True→False（保守化方向・新規の主張は発生しない）: "
             "`rq1b.rand.fisher.avec.scenario_02`（既知の境界セル、"
             "within-family q=0.023 生存は不変）、"
             "`rq1b.rand.fisher_aggregate.avec.lstm`・"
             "`rq2.dut.multivehicle.closest_ks.avec_default`・"
             "`rq2cap.loco.closedloop.closest_sign.no_repulsion`"
             "（いずれも auxiliary 層）。")
    L.append("")
    L.append("**追加登載の開示（静的記録: 2026-07-16、修論横断レビュー M2a 対応）**: "
             "本文（修論 表8.2/8.3）に掲載していた p 値のうち台帳外だったもの"
             "（非 replay アームの V3 符号検定・V3 全アーム Wilcoxon・V2 判定"
             "ゲート Wilcoxon）を auxiliary family 3 つ（計 32 行・うち cv 縮退 "
             "4 行は p 未定義）として末尾追加。auxiliary は canonical の "
             "study-wide 補正プールに入らないため、canonical 27 行・研究横断"
             "生存 3 件・既存全行の within-family 列は不変（追加後の台帳再生成で"
             "機械検証済み。auxiliary 行の overall_* 列のみ aux プール内補正の"
             "再計算で変わるが、REPORT・修論とも非使用）。全新規 family は "
             "family 内 BH で `rq3_v2_ranking_gates` の replay/single ゲート"
             "（p=0.135・V2 の検出力限界開示と整合）を除き生存＝既存結論への"
             "影響なし。方向つき開示の規律に従い1点付記する: 非使用の "
             "auxiliary プール内 overall 層では、有限 p の拡大（87→117）で"
             "既存行 `rq2.dut.multivehicle.closest_ks.avec_default` の判定が"
             "1件だけ 非有意→有意（q 0.0575→0.0428）へ動く。この検定は KS "
             "診断（p 非主張・付録Bで overall 列を掲載しないと宣言済み）で"
             "あり、どの主張にも使われない。上の 2026-07-03 静的記録にある"
             "同 ID の True→False は canonical 21→27 拡大時の overall 層の"
             "別事象である。")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/rq3_realloop")
    args = ap.parse_args()
    root = Path(args.root)

    runs = pd.read_csv(root / "all_runs.csv")
    census = pd.read_csv(root / "encounters.csv")
    em = encounter_means(runs)

    arms = [a for a in PED_ARMS if a != REFERENCE_ARM
            and a in set(runs["ped_arm"])]
    preds = [p for p in PREDICTORS if p in set(runs["pred"])]
    plans = [p for p in PLAN_MODES if p in set(runs["plan"])]
    all_arms = ([REFERENCE_ARM] if REFERENCE_ARM in set(runs["ped_arm"])
                else []) + arms

    plans_regular = [p for p in plans if p in PLAN_MODES]
    v1 = v1_rows(em, arms, preds, plans_regular)
    gains = robust_gain_rows(em, all_arms, preds)
    v2 = (v2_robust_gain_preservation(gains, all_arms, preds)
          + v2_ranking_preservation(em, all_arms, preds, plans_regular))
    v3 = [r for r in gains if r["ped_arm"] == REFERENCE_ARM]

    pd.DataFrame(v1, columns=V1_COLUMNS).to_csv(root / "paired_v1.csv",
                                                index=False)
    pd.DataFrame(v2, columns=V2_COLUMNS).to_csv(root / "v2_verdicts.csv",
                                                index=False)
    pd.DataFrame(gains, columns=V3_COLUMNS).to_csv(root / "v3_robust.csv",
                                                   index=False)

    sidecar = {
        "source": "rq3_realloop",
        "generated_by": "make_rq3_report.py",
        "tests": headline_tests(v1, gains, v2),
    }
    with open(root / "headline_tests.json", "w") as f:
        json.dump(sidecar, f, indent=1)
        f.write("\n")

    report = write_report(runs, census, v1, v2, v3, gains, em)
    (root / "REPORT.md").write_text(report)
    print(f"wrote {root}/paired_v1.csv, v2_verdicts.csv, v3_robust.csv, "
          f"headline_tests.json, REPORT.md "
          f"({len(sidecar['tests'])} ledger tests)")


if __name__ == "__main__":
    main()
