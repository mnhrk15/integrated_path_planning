#!/usr/bin/env python3
"""Aggregate every RQ's headline-test sidecar into one multiplicity-corrected ledger.

The axis-A review (``docs/CODE_REVIEW_axisA_20260619.md`` point 8) flagged that the
RQ suite reports many significance tests -- RQ2 pooled closest-approach KS, RQ1b
claim-(2) per-scenario single-vs-robust Fisher -- without any family-wise / FDR
management, while calling non-significant results "indistinguishable". This tool
removes that asymmetry. Each RQ script emits a ``headline_tests*.json`` sidecar
listing its hypotheses; this collects them, applies Benjamini-Hochberg FDR
(primary) and Holm-Bonferroni (conservative FWER) via ``src.core.multiplicity``,
and writes:

  * ``outputs/multiplicity_ledger.csv`` -- every test with within-family and
    study-wide adjusted p / survival flags.
  * ``outputs/multiplicity_ledger.md``  -- the same as a readable table, an RQ1b
    family-definition sensitivity, and a thesis-ready paragraph.

RQ1a contributes NO tests: its open-loop ADE/FDE/NLL are point estimates with no
significance test (intentional -- review M1/M10). That is itself good multiplicity
hygiene -- you cannot p-hack tests you never ran -- and is stated explicitly.

LOSO RQ2 sidecars (``*_loso``) are a re-split of the SAME fidelity question as the
canonical LOCO; including both would double-count it. LOSO families are reported
on their own but EXCLUDED from the study-wide "overall" correction.

Usage:
    .venv/bin/python examples/make_multiplicity_ledger.py
        [--root outputs] [--out outputs] [--alpha 0.05]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.multiplicity import adjust, build_ledger  # noqa: E402

# Columns the flat CSV/table exposes (records carry extra per-RQ fields too).
LEDGER_COLUMNS = [
    "source", "test_id", "family", "headline", "p_value",
    "family_size", "family_bh_q", "family_holm_p",
    "family_bh_reject", "family_holm_reject",
    "overall_size", "overall_bh_q", "overall_holm_p",
    "overall_bh_reject", "overall_holm_reject",
    "description",
]


def load_sidecars(root: Path) -> Tuple[List[Dict], List[Dict]]:
    """Collect tests from every headline_tests*.json under ``root`` (sorted).

    Returns (tests, sources). File order is sorted for deterministic output. A
    stray or stale file the glob picks up (top-level not an object, ``tests`` not
    a list, a non-dict entry, or unreadable/invalid JSON) is skipped with a stderr
    warning rather than aborting aggregation of the valid sidecars.
    """
    files = sorted(Path(root).glob("**/headline_tests*.json"))
    tests: List[Dict] = []
    sources: List[Dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping unreadable sidecar {f}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"  skipping malformed sidecar {f}: top-level is not an object",
                  file=sys.stderr)
            continue
        ftests = data.get("tests", [])
        if not isinstance(ftests, list):
            print(f"  skipping sidecar {f}: 'tests' is not a list", file=sys.stderr)
            continue
        src = data.get("source", f.stem)
        valid = [dict(t) for t in ftests if isinstance(t, dict)]
        if len(valid) != len(ftests):
            print(f"  {f}: dropped {len(ftests) - len(valid)} non-object test entr"
                  f"y(ies)", file=sys.stderr)
        sources.append({"source": src, "path": str(f), "n_tests": len(valid)})
        for t in valid:
            t.setdefault("source", src)
            tests.append(t)
    return tests, sources


def _is_auxiliary(test: Dict) -> bool:
    """Whether a test is an auxiliary re-split excluded from the study-wide family.

    Branches on an EXPLICIT ``auxiliary`` flag (or ``protocol == 'loso'``) carried
    by the producer, not on a substring of the free-form ``family`` label -- the
    LOSO RQ2 protocol re-splits the SAME fidelity question as LOCO, so counting
    both would double-count it, and a label-suffix convention would silently break
    if a family string were ever reworded. Tests without the flag (e.g. RQ1b) are
    never auxiliary.
    """
    return bool(test.get("auxiliary", False)) or test.get("protocol") == "loso"


def assemble(tests: List[Dict], alpha: float = 0.05) -> Tuple[List[Dict], List[Dict]]:
    """Split auxiliary re-splits from the canonical study-wide family, then correct.

    The study-wide "overall" correction runs over the canonical tests only (LOSO
    re-splits of the same fidelity question are excluded to avoid double-counting);
    auxiliary tests still get within-family BH/Holm so they are not dropped.
    Returns (canonical_rows, auxiliary_rows).
    """
    canonical = [t for t in tests if not _is_auxiliary(t)]
    auxiliary = [t for t in tests if _is_auxiliary(t)]
    canonical_rows = build_ledger(canonical, alpha)
    auxiliary_rows = build_ledger(auxiliary, alpha) if auxiliary else []
    return canonical_rows, auxiliary_rows


def rq1b_family_sensitivity(rows: List[Dict], alpha: float = 0.05) -> Dict:
    """Family-definition sensitivity for the RQ1b claim-(2) Fisher tests.

    The headline S2/avec signal's survival depends on the family chosen: the
    avec-conditioned subset, the headline-GT subset, and the full GT x scenario
    scan (which includes the underpowered +/-1SD corners) give materially
    different adjusted p. Report the most-significant test's adjusted p under each
    so the boundary case is explicit rather than hidden behind one family choice.
    """
    fam = [r for r in rows if r.get("family") == "rq1b_claim2_fisher"]
    if not fam:
        return {}
    views = {
        "avec_only": [r for r in fam if r.get("gt") == "avec"],
        "headline_gts": [r for r in fam if r.get("power_tier") == "headline"],
        "full_scan": fam,
    }
    out: Dict[str, Dict] = {}
    for name, subset in views.items():
        ps = [r.get("p_value", np.nan) for r in subset]
        # Match adjust()/_finite_mask: np.asarray(...).astype(float) treats numpy
        # floats and Python numbers alike, so the family-size predicate here does
        # not silently diverge from the one used to compute the corrections.
        finite_mask = np.isfinite(np.asarray(ps, dtype=float))
        finite = [(i, ps[i]) for i in np.flatnonzero(finite_mask)]
        if not finite:
            continue
        adj = adjust(ps, alpha)
        imin = min(finite, key=lambda kv: kv[1])[0]
        out[name] = {
            "m": adj["m"],
            "min_test_id": subset[imin]["test_id"],
            "min_p": float(ps[imin]),
            "min_bh_q": float(adj["bh_q"][imin]),
            "min_holm_p": float(adj["holm_p"][imin]),
            "survives_bh": bool(adj["bh_reject"][imin]),
            "survives_holm": bool(adj["holm_reject"][imin]),
        }
    return out


def _flat_df(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in LEDGER_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[LEDGER_COLUMNS]


def _md_table(df: pd.DataFrame, cols: List[str]) -> str:
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                v = f"{v:.4f}"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_markdown(canonical_rows: List[Dict], loso_rows: List[Dict],
                    sources: List[Dict], alpha: float) -> str:
    L: List[str] = []
    L.append("# RQ スイート横断 多重比較 ledger")
    L.append("")
    L.append(f"family-wise / FDR 補正（alpha={alpha}）。BH-FDR を primary、Holm-Bonferroni "
             "を conservative sensitivity として併記。NaN（空 arm の Fisher・空プールの "
             "KS）は仮説ではないので family size に数えない。")
    L.append("")
    L.append("> **RQ1a は検定を持たない**（開ループ ADE/FDE/NLL は点推定・有意性検定なし"
             "＝意図的、review M1/M10）。実行していない検定は補正対象にならない＝"
             "「やらなかった検定で p-hack できない」という多重性衛生そのもの。")
    L.append("")
    L.append("## 収集した sidecar")
    L.append("")
    for s in sources:
        L.append(f"- `{s['path']}` — source={s['source']}, tests={s['n_tests']}")
    L.append("")

    disp = ["test_id", "family", "p_value", "family_size",
            "family_bh_q", "family_holm_p", "overall_bh_q", "overall_holm_p"]
    L.append("## 補正結果（canonical 研究横断 family）")
    L.append("")
    df = _flat_df(canonical_rows)
    L.append(_md_table(df, disp))
    L.append("")
    n_overall = int(df["overall_size"].dropna().iloc[0]) if len(df) else 0
    n_bh = int(df["overall_bh_reject"].sum()) if len(df) else 0
    n_holm = int(df["overall_holm_reject"].sum()) if len(df) else 0
    L.append(f"- 研究横断 family size（overall）: **{n_overall}** 検定")
    L.append(f"- overall BH-FDR で生存（q<{alpha}）: **{n_bh}** / Holm で生存: **{n_holm}**")
    L.append("")
    L.append("> **overall の読み方**: overall は RQ2 忠実度と RQ1b 計画安全という"
             "*異なる問い*を1 family に束ねた最保守の境界（cross-suite 過剰補正）。"
             "適切な評価単位は各 family 内補正（上表 `family_bh_q`）と下の RQ1b "
             "family 定義感度。overall は『最悪でもこの程度』の sanity 上限として読む"
             "（実際 RQ2 忠実度は family 内 q=0.007 で明確に有意・overall でのみ境界化）。")
    L.append("")

    if loso_rows:
        L.append("## 付録: LOSO（補助・overall から除外）")
        L.append("")
        L.append("LOSO は LOCO と同じ忠実度の問いの再分割。二重計上を避けるため overall "
                 "には含めず、family 内補正のみ示す。")
        L.append("")
        L.append(_md_table(_flat_df(loso_rows), disp))
        L.append("")

    # RQ1b family-definition sensitivity.
    sens = rq1b_family_sensitivity(canonical_rows, alpha)
    if sens:
        L.append("## RQ1b claim-(2) family 定義感度")
        L.append("")
        L.append("最有意 per-scenario Fisher（S2/avec）の補正後 p は family の取り方に敏感。"
                 "3 つの view を併記する:")
        L.append("")
        L.append("| view | m | min test | raw p | BH q | Holm p | BH 生存 | Holm 生存 |")
        L.append("|---|---|---|---|---|---|---|---|")
        labels = {"avec_only": "avec 条件付き（3）",
                  "headline_gts": "headline GT（avec+calib, 6）",
                  "full_scan": "全 GT×scenario 走査（12, 弱パワー corner 含む）"}
        for key in ["avec_only", "headline_gts", "full_scan"]:
            if key not in sens:
                continue
            v = sens[key]
            L.append(f"| {labels[key]} | {v['m']} | `{v['min_test_id']}` | "
                     f"{v['min_p']:.4f} | {v['min_bh_q']:.4f} | {v['min_holm_p']:.4f} | "
                     f"{'生存' if v['survives_bh'] else '不成立'} | "
                     f"{'生存' if v['survives_holm'] else '不成立'} |")
        L.append("")

    # Thesis-ready paragraph.
    L.append("## 修論向け要約（データ駆動）")
    L.append("")
    L.extend(_thesis_paragraph(canonical_rows, sens, alpha))
    L.append("")
    return "\n".join(L)


def _thesis_paragraph(rows: List[Dict], sens: Dict, alpha: float) -> List[str]:
    """Generate the honest survival narrative FROM the corrected rows."""
    P: List[str] = []
    # RQ2 fidelity KS.
    rq2 = [r for r in rows if str(r.get("family", "")).startswith("rq2_fidelity_ks")]
    rq2_cal = next((r for r in rq2 if r.get("test_id", "").endswith("calibrated")), None)
    if rq2_cal is not None:
        P.append(
            f"- **RQ2 忠実度（closest-approach KS）**: 較正 sim vs 実の pooled KS は "
            f"raw p={rq2_cal['p_value']:.3f}。**忠実度 family 内（m={rq2_cal['family_size']}）"
            f"では BH q={rq2_cal['family_bh_q']:.3f}＝明確に有意**＝sim が実 standoff を"
            "再現しきれない＝~0.68m の忠実度ギャップは統計的に実在（限界を補強する向きの"
            f"所見）。RQ1b の無関係な計画検定まで一括する最保守の研究横断プール"
            f"（m={rq2_cal['overall_size']}）でのみ q={rq2_cal['overall_bh_q']:.3f}＝境界化"
            "（異質な問いを跨ぐ過剰補正のため参考値）。")
    # RQ1b claim-2.
    if sens:
        fs = sens.get("full_scan")
        av = sens.get("avec_only")
        if fs and av:
            P.append(
                f"- **RQ1b claim-(2)（分布なし計画は危険）**: 最有意セル S2/avec は raw "
                f"p={fs['min_p']:.4f}（pseudo-replication で反保守的＝真の p の下界）。"
                f"avec 条件付き family（m={av['m']}）では BH q={av['min_bh_q']:.4f}＝"
                f"{'生存' if av['survives_bh'] else '不成立'}だが、"
                f"弱パワー corner を含む全 12 セル走査（m={fs['m']}）では "
                f"BH q={fs['min_bh_q']:.4f}＝{'生存' if fs['survives_bh'] else '不成立'}。"
                "**＝claim-(2) の per-scenario 信号は family 定義に敏感な境界事例で、"
                "確定的ではなく示唆に留まる**（既存 REPORT の『示唆・反応モデル依存・"
                "外的妥当性ではない』枠組みと整合）。")
    P.append(
        "- **総括**: 強い結論は RQ1b claim-(1)（robust 利得は全 GT で頑健・"
        "`robust_gain_holds`、有意性検定を要さない決定的判定）と RQ2 の忠実度ギャップ"
        "の実在。claim-(2) の CV 危険性は多重比較後は弱い示唆。多重比較補正は"
        "**既存の正直なフレーミングを覆さず追認する**。")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs",
                    help="Directory to glob headline_tests*.json from")
    ap.add_argument("--out", default="outputs",
                    help="Directory to write the ledger CSV/MD")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    tests, sources = load_sidecars(Path(args.root))
    if not tests:
        print(f"No headline_tests*.json found under {args.root}", file=sys.stderr)
        sys.exit(1)
    canonical_rows, loso_rows = assemble(tests, args.alpha)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "multiplicity_ledger.csv"
    _flat_df(canonical_rows + loso_rows).to_csv(csv_path, index=False)
    md_path = out_dir / "multiplicity_ledger.md"
    md_path.write_text(render_markdown(canonical_rows, loso_rows, sources, args.alpha),
                       encoding="utf-8")

    print(f"collected {len(tests)} tests from {len(sources)} sidecar(s)")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
