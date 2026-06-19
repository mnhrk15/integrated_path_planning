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
import sys
from pathlib import Path

import pandas as pd

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


def rand_verdict(df):
    """Claim (2) CV danger: distributionless prediction collides more than
    distribution-aware (robust), per predictor (apples-to-apples)."""
    scenarios = sorted(df.scenario.unique())

    def tot(cond):
        return sum(_cond_collisions(df, sc, cond) for sc in scenarios)

    coll = {c: tot(c) for c in
            ["cv_single", "lstm_single", "lstm_robust_eps0.0",
             "sgan_single_inf1.00", "sgan_robust_eps0.0"]}
    return {
        # CV (no distribution at all) vs the distribution-aware SGAN robust.
        "cv_danger_holds": coll["cv_single"] > coll["sgan_robust_eps0.0"],
        # Same predictor, single vs robust over its own distribution.
        "lstm_danger_holds": coll["lstm_single"] > coll["lstm_robust_eps0.0"],
        "collisions_by_condition": coll,
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
        rows.append({"gt_label": gt, "scenario": sc, **coll,
                     "single_total": single_tot, "robust_total": robust_tot,
                     "class": klass})
    return pd.DataFrame(rows)


def build_verdicts(master, gt_labels):
    rows = []
    for gt in gt_labels:
        mdf = master[(master.campaign == "margin") & (master.gt_label == gt)]
        rdf = master[(master.campaign == "rand") & (master.gt_label == gt)]
        mv = margin_verdict(mdf) if not mdf.empty else None
        rv = rand_verdict(rdf) if not rdf.empty else None
        rows.append({
            "gt_label": gt,
            "robust_gain_holds": (mv["robust_gain_holds"] if mv else None),
            "dominating_inflations": (",".join(mv["dominating_inflations"])
                                      if mv else ""),
            "robust_collisions": (mv["robust_total_collisions"] if mv else None),
            "cv_danger_holds": (rv["cv_danger_holds"] if rv else None),
            "lstm_danger_holds": (rv["lstm_danger_holds"] if rv else None),
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
    L.append("- cv_danger_holds: CV single の衝突 > SGAN robust の衝突（＝主張②保持）")
    L.append("- lstm_danger_holds: LSTM single の衝突 > LSTM robust の衝突")
    L.append("")
    L.append(_md_table(verdicts))
    L.append("")
    L.append("> 注意: 集計 `cv_danger_holds`/`lstm_danger_holds` は衝突をシナリオ"
             "横断で合算するため、シナリオごとの内訳を隠す。主張②は必ず下記の "
             "per-scenario 分類で読むこと。主張①（`robust_gain_holds`）は"
             "シナリオ横断の集計でも頑健。")
    L.append("")

    # Sensitivity: do verdicts flip across GT? A None means the verdict was not
    # computed for that GT (missing/empty arm); report that as undetermined
    # rather than silently calling an uncomputed verdict "robust".
    def _sensitivity_status(col):
        raw = verdicts[col].tolist()
        vals = [v for v in raw if v is not None]
        if len(vals) < len(raw):
            return f"一部 GT 未計算（{len(vals)}/{len(raw)} GT のみ・判定不能）"
        return "反転あり（較正に感度あり）" if len(set(vals)) > 1 else "全 GT で不変（頑健）"
    L.append("## 感度（GT 間で判定が反転するか）")
    L.append("")
    for col in ["robust_gain_holds", "cv_danger_holds", "lstm_danger_holds"]:
        L.append(f"- **{col}**: {_sensitivity_status(col)}")
    L.append("")

    # Per-scenario claim-(2): the honest, uncontaminated view.
    srows = rand_scenario_rows(master)
    srows.to_csv(root / "scenario_rand.csv", index=False)
    L.append("## 主張② シナリオ別（per-scenario・汚染なし）")
    L.append("")
    L.append("各 (GT, シナリオ) の rand 衝突数と分類（single-danger=分布なし single "
             "が衝突しつつ robust 2種は無衝突＝真の主張②信号／mixed=single≫robust>0"
             "＝主張②方向は残るが robust も非ゼロ／GT-artifact=robust≧single＝弁別"
             "でなく GT 自体の衝突／no-conflict=無衝突）:")
    L.append("")
    if not srows.empty:
        order = {g: i for i, g in enumerate(
            ["avec", "calib", "calib_lo", "calib_hi",
             "calib_s-v+", "calib_s+v-"])}
        # Unknown GT labels sort last (fillna) instead of becoming NaN and
        # scattering unpredictably.
        srows = srows.sort_values(
            by=["scenario", "gt_label"],
            key=lambda s: (s.map(order).fillna(len(order))
                           if s.name == "gt_label" else s))
        L.append(_md_table(srows[["scenario", "gt_label"] + SINGLE_CONDS
                                  + ROBUST_CONDS + ["class"]]))
    L.append("")
    # The reading-narrative below is hand-written against the standard 3-scenario
    # / core-GT run. Emit it only when those three scenarios are actually present
    # so a partial run (subset of --scenarios/--campaigns) cannot ship prose that
    # contradicts its own tables. Re-run with the standard set to regenerate it.
    have_scenarios = set(srows.scenario.unique()) if not srows.empty else set()
    if {"scenario_01", "scenario_02", "scenario_03"} <= have_scenarios:
        L.append("**読み筋（per-scenario・3シナリオとも干渉成立する修正シナリオ）**:")
        L.append("- **S1（密交差）**: 両 GT で single-danger が残る（主に cv＝盲目予測"
                 "の本質的危険。協調的な実較正歩行者でも、ego が誤った単一軌道に "
                 "commit すると回避できない）。")
        L.append("- **S2（狭路すれ違い）**: AVEC GT では single 衝突・robust 0"
                 "（single-danger）→ **較正 GT では single も 0（no-conflict）＝主張②が"
                 "消失**。狭路では協調的な実較正歩行者が single 計画の危険を解消する。")
        L.append("- **S3（右折 yield）**: AVEC GT では single-danger（single 衝突・"
                 "robust 0）→ **較正 GT（弱い v0≈1.68）では交錯が厳しくなり single が"
                 "大幅悪化する一方 robust も完全には守れない（mixed: single≫robust>0）**。"
                 "single≫robust の方向（主張②）は残るが、弱い斥力が右折交錯を困難化し "
                 "robust の安全余裕も低下する。")
        L.append("")
        L.append("**主張②の結論（scenario-dependent）**: AVEC 反応モデルでは主張②"
                 "（分布なし計画は危険）は3シナリオすべてで成立。**実較正（より協調的）"
                 "反応モデル下では効果はシナリオ依存**＝狭路 S2 では消失（協調回避）、"
                 "密交差 S1（盲目 cv）と右折 S3（弱斥力で交錯困難）では残る。"
                 "「分布なし計画の危険度は反応モデルに依存する」が正直な結論で、"
                 "AVEC sim の一律な危険性主張は手調整反応モデルに一部依存していた。"
                 "robust 利得（主張①）は全 GT・全シナリオで頑健に生き残る。")
    else:
        L.append("（per-scenario 固定ナラティブは標準3シナリオ "
                 "(scenario_01/02/03) を実行したときのみ生成。今回の対象シナリオ"
                 "では省略。）")
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

    if total_failed:
        print(f"\nWARNING: {total_failed} run(s) failed and were not cached "
              f"(they will be retried on the next invocation).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
