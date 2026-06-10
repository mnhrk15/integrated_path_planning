#!/usr/bin/env python3
"""Generate REPORT.md and the trade-off figure for the margin-control experiment.

Reads <outdir>/all_runs.csv produced by examples/run_da_poc.py and writes:
  - tradeoff_curve.png : MinDist-vs-Time trade-off per scenario
  - welch_tests.csv    : Welch t-tests backing Experiments A and B
  - REPORT.md          : tables, sanity checks, and verdicts (Japanese)

Usage:
    python examples/make_margin_report.py [--outdir output/exp_margin_control]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.run_da_poc import CONDITIONS, BASELINE_LABEL  # noqa: E402

ROBUST_LABEL = "sgan_robust_eps0.0"
LSTM_SINGLE = "lstm_single"
LSTM_ROBUST = "lstm_robust_eps0.0"
INFLATION_LABELS = [c[0] for c in CONDITIONS
                    if c[1] == "sgan" and not c[2]]  # single-sample SGAN series
P_SIG = 0.05

# Mapping for the per-seed sanity check against the original PoC outputs.
POC_DIR_TMPL = "output/poc_da_{scenario}"
POC_LABEL_MAP = {BASELINE_LABEL: "baseline_single", ROBUST_LABEL: "da_eps0.0"}
SANITY_COLS = ["time_s", "speed_ms", "min_dist_m", "min_ttc_s",
               "collision_count", "ade", "fde"]


def scenario_total_time(scenario_stem: str) -> float:
    path = Path("scenarios") / f"{scenario_stem}.yaml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if "total_time" in data:
            return float(data["total_time"])
    return 30.0  # default_config.yaml fallback


def welch(a: pd.Series, b: pd.Series):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p)


def fmt_ms(g: pd.DataFrame, col: str, prec: int = 3) -> str:
    return f"{g[col].mean():.{prec}f}±{g[col].std(ddof=1):.{prec}f}"


def fmt_p(p: float) -> str:
    return f"{p:.3e}"


def sanity_check(df: pd.DataFrame) -> list:
    """Per-seed comparison against the original PoC outputs (behavior preservation)."""
    lines = []
    for scenario in sorted(df.scenario.unique()):
        poc_path = Path(POC_DIR_TMPL.format(scenario=scenario)) / "all_runs.csv"
        if not poc_path.exists():
            lines.append(f"- {scenario}: 旧PoC出力なし（{poc_path}）→ SKIP")
            continue
        poc = pd.read_csv(poc_path)
        for new_label, old_label in POC_LABEL_MAP.items():
            new = df[(df.scenario == scenario) & (df.condition == new_label)] \
                .set_index("seed").sort_index()
            old = poc[poc.condition == old_label].set_index("seed").sort_index()
            seeds = new.index.intersection(old.index)
            if len(seeds) == 0:
                lines.append(f"- {scenario} {new_label}: 共通シードなし → SKIP")
                continue
            diffs = (new.loc[seeds, SANITY_COLS] - old.loc[seeds, SANITY_COLS]).abs()
            max_diff = float(diffs.to_numpy().max())
            status = "PASS" if max_diff <= 1e-9 else "FAIL"
            lines.append(f"- {scenario} {new_label} ↔ 旧 {old_label} "
                         f"(n={len(seeds)}): max|Δ|={max_diff:.2e} → **{status}**")
    return lines


def experiment_a(df: pd.DataFrame, scenarios: list):
    """Welch tests of each inflation level vs the robust planner + verdicts."""
    rows = []
    for scenario in scenarios:
        sdf = df[df.scenario == scenario]
        robust = sdf[sdf.condition == ROBUST_LABEL]
        for label in INFLATION_LABELS:
            g = sdf[sdf.condition == label]
            if g.empty or robust.empty:
                continue
            row = {"scenario": scenario, "condition": label,
                   "inflation": g.inflation.iloc[0]}
            for col in ["min_dist_m", "time_s"]:
                delta, p = welch(g[col], robust[col])
                row[f"{col}_delta_vs_robust"] = delta
                row[f"{col}_p_vs_robust"] = p
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_robust_mean"] = robust[col].mean()
            rows.append(row)
    a = pd.DataFrame(rows)

    # Mean-based verdict: some inflation matches robust MinDist at no extra time
    # in ALL scenarios simultaneously.
    mean_ok_levels = []
    sig_block_levels = []
    for label in INFLATION_LABELS:
        if label == BASELINE_LABEL:
            continue
        sub = a[a.condition == label]
        if len(sub) < len(scenarios):
            continue
        if ((sub.min_dist_m_delta_vs_robust >= 0) &
                (sub.time_s_delta_vs_robust <= 0)).all():
            mean_ok_levels.append(label)
        # Significance-based: this level is blocked if in >=1 scenario it is
        # significantly worse on MinDist or significantly slower than robust.
        blocked = (((sub.min_dist_m_delta_vs_robust < 0) &
                    (sub.min_dist_m_p_vs_robust < P_SIG)) |
                   ((sub.time_s_delta_vs_robust > 0) &
                    (sub.time_s_p_vs_robust < P_SIG))).any()
        if blocked:
            sig_block_levels.append(label)

    candidates = [l for l in INFLATION_LABELS if l != BASELINE_LABEL]
    if mean_ok_levels:
        verdict_mean = ("保守性で説明可能: inflation "
                        f"{', '.join(mean_ok_levels)} が3シナリオ同時に "
                        "MinDist ≥ robust かつ Time ≤ robust を達成")
    else:
        verdict_mean = ("平均ベースでは、どの inflation も3シナリオ同時に "
                        "「MinDist ≥ robust かつ Time ≤ robust」を達成できない "
                        "→ 分布情報の寄与を示唆")
    if set(sig_block_levels) == set(candidates):
        verdict_sig = ("有意性ベース: 全 inflation が少なくとも1シナリオで "
                       "robust に有意に劣る（MinDist 低下 or Time 増加, p<0.05）"
                       " → 分布の形が情報を持つ証拠")
    else:
        free = sorted(set(candidates) - set(sig_block_levels))
        verdict_sig = ("有意性ベース: inflation "
                       f"{', '.join(free)} は robust に有意に劣るシナリオがない "
                       "（有意差のみでは robust 優位を主張できない）")
    return a, verdict_mean, verdict_sig


def experiment_b(df: pd.DataFrame, scenarios: list):
    """Per-seed robust gains (MinDist and Time) for SGAN vs LSTM, via Welch."""
    rows = []
    for scenario in scenarios:
        sdf = df[df.scenario == scenario]
        gains = {}
        for method, single_label, robust_label in [
                ("sgan", BASELINE_LABEL, ROBUST_LABEL),
                ("lstm", LSTM_SINGLE, LSTM_ROBUST)]:
            single = sdf[sdf.condition == single_label].set_index("seed")
            robust = sdf[sdf.condition == robust_label].set_index("seed")
            seeds = single.index.intersection(robust.index)
            if len(seeds) == 0:
                continue
            gains[method] = {
                "min_dist": (robust.loc[seeds, "min_dist_m"]
                             - single.loc[seeds, "min_dist_m"]),
                "time": (robust.loc[seeds, "time_s"]
                         - single.loc[seeds, "time_s"]),
            }
            # Within-method Welch: robust vs single
            d_md, p_md = welch(robust.loc[seeds, "min_dist_m"],
                               single.loc[seeds, "min_dist_m"])
            d_t, p_t = welch(robust.loc[seeds, "time_s"],
                             single.loc[seeds, "time_s"])
            rows.append({"scenario": scenario, "test": f"{method}_robust_vs_single",
                         "delta_min_dist": d_md, "p_min_dist": p_md,
                         "delta_time": d_t, "p_time": p_t, "n": len(seeds)})
        if "sgan" in gains and "lstm" in gains:
            d_md, p_md = welch(gains["sgan"]["min_dist"], gains["lstm"]["min_dist"])
            d_t, p_t = welch(gains["sgan"]["time"], gains["lstm"]["time"])
            rows.append({"scenario": scenario, "test": "gain_sgan_vs_gain_lstm",
                         "delta_min_dist": d_md, "p_min_dist": p_md,
                         "delta_time": d_t, "p_time": p_t,
                         "n": min(len(gains["sgan"]["min_dist"]),
                                  len(gains["lstm"]["min_dist"]))})
    return pd.DataFrame(rows)


def ade_invariance(df: pd.DataFrame, scenarios: list):
    """Welch test of ADE robust vs single within each method (should be ~unchanged)."""
    rows = []
    for scenario in scenarios:
        sdf = df[df.scenario == scenario]
        for method, single_label, robust_label in [
                ("sgan", BASELINE_LABEL, ROBUST_LABEL),
                ("lstm", LSTM_SINGLE, LSTM_ROBUST)]:
            single = sdf[sdf.condition == single_label]
            robust = sdf[sdf.condition == robust_label]
            if single.empty or robust.empty:
                continue
            d, p = welch(robust["ade"], single["ade"])
            rows.append({"scenario": scenario, "method": method,
                         "delta_ade": d, "p": p})
    return pd.DataFrame(rows)


def plot_tradeoff(df: pd.DataFrame, scenarios: list, out_path: Path):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.2),
                             squeeze=False)
    extra = [(ROBUST_LABEL, "*", "tab:red", "SGAN robust (eps=0)"),
             (LSTM_SINGLE, "s", "tab:green", "LSTM single"),
             (LSTM_ROBUST, "^", "tab:purple", "LSTM robust (eps=0)")]
    for ax, scenario in zip(axes[0], scenarios):
        sdf = df[df.scenario == scenario]
        xs, ys, xerr, yerr, anns = [], [], [], [], []
        for label in INFLATION_LABELS:
            g = sdf[sdf.condition == label]
            if g.empty:
                continue
            n = len(g)
            xs.append(g.time_s.mean())
            ys.append(g.min_dist_m.mean())
            xerr.append(g.time_s.std(ddof=1) / np.sqrt(n))
            yerr.append(g.min_dist_m.std(ddof=1) / np.sqrt(n))
            anns.append(f"{g.inflation.iloc[0]:.2f}")
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker="o", color="tab:blue",
                    capsize=2, label="SGAN single (inflation sweep)")
        for x, y, a in zip(xs, ys, anns):
            ax.annotate(a, (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        for label, marker, color, name in extra:
            g = sdf[sdf.condition == label]
            if g.empty:
                continue
            n = len(g)
            ax.errorbar([g.time_s.mean()], [g.min_dist_m.mean()],
                        xerr=[g.time_s.std(ddof=1) / np.sqrt(n)],
                        yerr=[g.min_dist_m.std(ddof=1) / np.sqrt(n)],
                        marker=marker, markersize=10, color=color,
                        capsize=2, linestyle="none", label=name)
        ax.set_title(scenario)
        ax.set_xlabel("Completion time mean [s]")
        ax.set_ylabel("MinDist mean [m]")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output/exp_margin_control")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    df = pd.read_csv(outdir / "all_runs.csv")
    scenarios = sorted(df.scenario.unique())

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True).stdout.strip()
    except Exception:
        git_head = "unknown"

    sanity_lines = sanity_check(df)
    exp_a, verdict_mean, verdict_sig = experiment_a(df, scenarios)
    exp_b = experiment_b(df, scenarios)
    ade_inv = ade_invariance(df, scenarios)
    plot_tradeoff(df, scenarios, outdir / "tradeoff_curve.png")

    welch_all = pd.concat([
        exp_a.assign(experiment="A"),
        exp_b.assign(experiment="B"),
    ], ignore_index=True)
    welch_all.to_csv(outdir / "welch_tests.csv", index=False)

    lines = []
    w = lines.append
    w("# 実験レポート: 分布情報と保守性の分離（★1）")
    w("")
    w(f"- コミット: `{git_head}`")
    n_seeds = df.groupby(['scenario', 'condition']).size()
    w(f"- ラン数: {len(df)}（シナリオ×条件あたり {n_seeds.min()}–{n_seeds.max()} シード）")
    w("- プロトコル: 3シナリオ × seeds 0..19、visualization 無効、"
      "Welch 両側 t 検定。評価メトリクス（MinDist/TTC/collision、ego_radius=1.0 m）は"
      "全条件共通で、プランナの feasibility 判定のみを操作。")
    w("")
    w("## Sanity: 挙動保存チェック（旧 PoC 出力との per-seed 照合）")
    w("")
    lines.extend(sanity_lines)
    w("")

    w("## 条件別サマリ（mean±std）")
    w("")
    for scenario in scenarios:
        sdf = df[df.scenario == scenario]
        total_time = scenario_total_time(scenario)
        w(f"### {scenario}")
        w("")
        w("| condition | n | Time [s] | MinDist [m] | MinTTC [s] | 衝突 | ADE [m] | Time飽和 |")
        w("|---|---|---|---|---|---|---|---|")
        for label, *_ in CONDITIONS:
            g = sdf[sdf.condition == label]
            if g.empty:
                continue
            saturated = int((g.time_s >= total_time - 0.1 - 1e-9).sum())
            w(f"| {label} | {len(g)} | {fmt_ms(g, 'time_s', 2)} | "
              f"{fmt_ms(g, 'min_dist_m')} | {fmt_ms(g, 'min_ttc_s')} | "
              f"{int(g.collision_count.sum())} | {fmt_ms(g, 'ade')} | {saturated} |")
        w("")

    w("## 実験A: 膨張マージン付き単一サンプル vs robust(ε=0)")
    w("")
    w("各 inflation 条件と robust の差（Δ = inflation条件 − robust、Welch 両側）:")
    w("")
    w("| scenario | inflation | ΔMinDist [m] | p | ΔTime [s] | p |")
    w("|---|---|---|---|---|---|")
    for _, r in exp_a.iterrows():
        w(f"| {r.scenario} | {r.inflation:.2f} | "
          f"{r.min_dist_m_delta_vs_robust:+.3f} | {fmt_p(r.min_dist_m_p_vs_robust)} | "
          f"{r.time_s_delta_vs_robust:+.3f} | {fmt_p(r.time_s_p_vs_robust)} |")
    w("")
    w("**判定（平均ベース）**: " + verdict_mean)
    w("")
    w("**判定（有意性ベース）**: " + verdict_sig)
    w("")
    w("![tradeoff](tradeoff_curve.png)")
    w("")

    w("## 実験B: LSTM 分布での robust 計画")
    w("")
    w("| scenario | 検定 | ΔMinDist [m] | p | ΔTime [s] | p | n |")
    w("|---|---|---|---|---|---|---|")
    for _, r in exp_b.iterrows():
        w(f"| {r.scenario} | {r.test} | {r.delta_min_dist:+.3f} | "
          f"{fmt_p(r.p_min_dist)} | {r.delta_time:+.3f} | "
          f"{fmt_p(r.p_time)} | {int(r.n)} |")
    w("")
    w("`gain_sgan_vs_gain_lstm` は per-seed の robust 利得 "
      "d(seed) = X_robust(seed) − X_single(seed)（X = MinDist / Time）を方法間で "
      "Welch 比較したもの（正 = SGAN の利得・コストが大きい）。同一シードは歩行者 "
      "SFM の初期条件を共有するが、ego の挙動差により厳密な対応ペアではない点に注意。")
    w("")

    w("## 付記")
    w("")
    w("- robust 条件でも予測器は同一だが、ego の挙動変化が歩行者の斥力反応 → 観測 → "
      "予測に波及するため、per-seed の ADE は厳密には一致しない。統計的不変性の確認"
      "（Welch、robust vs single、同一 method）:")
    for _, r in ade_inv.iterrows():
        w(f"    - {r.scenario} {r.method}: ΔADE={r.delta_ade:+.4f} m, p={fmt_p(r.p)}")
    w("- `Time飽和` は time_s ≥ total_time − dt のラン数（ゴール未到達のまま打切り）。"
      "飽和が多い条件では Time が右側打切りされており、真の走行時間コストは表の値"
      "より大きい（Time ≤ robust 側に有利な打切りなので、実験Aの判定を弱めない）。")
    w("- `min_ttc_s` に inf が含まれる場合は mean±std が inf になる（既存集計と同じ"
      "扱い）。")
    w("")

    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {outdir / 'REPORT.md'}")
    print(f"Wrote {outdir / 'welch_tests.csv'}")
    print(f"Wrote {outdir / 'tradeoff_curve.png'}")
    print("\n--- Verdicts ---")
    print(verdict_mean)
    print(verdict_sig)


if __name__ == "__main__":
    main()
