#!/usr/bin/env python3
"""Compare the regenerated statistical benchmark against the stale CSVs.

The local (untracked) output/statistical_benchmark{,_s2,_s3} CSVs predate
commit 715d7b3 (ego-pedestrian repulsion fix + ADE/FDE rework) and do not
reproduce under the current code; the paper tables were produced from the
later output/comfort_s{1,2,3} campaigns, which the regenerated
output/statistical_benchmark_v2_s{1,2,3} reproduce bit-exactly. This script
documents the old->new drift and checks that the paper's headline claims
hold on the regenerated data:

  C1: zero collisions in all 123 runs
  C2: LSTM has the lowest stochastic-method ADE in S1 and S2
  C3: S2 MinDist ordering CV > LSTM > SGAN
  C4: S3 MinTTC saturates at ~0.85 s for all methods
  C5: S3 RMS jerk is roughly double the S1/S2 level
  C6: the principal S3 LSTM-SGAN effect stays significant (Welch)

Writes output/benchmark_regen_comparison/REPORT.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).parent.parent
OUTDIR = REPO / "output" / "benchmark_regen_comparison"

PAIRS = {
    "S1": ("output/statistical_benchmark", "output/statistical_benchmark_v2_s1"),
    "S2": ("output/statistical_benchmark_s2", "output/statistical_benchmark_v2_s2"),
    "S3": ("output/statistical_benchmark_s3", "output/statistical_benchmark_v2_s3"),
}

METRICS = ["time_s", "speed_ms", "min_dist_m", "min_ttc_s", "ade", "fde"]
NEW_ONLY = ["rms_jerk"]


def fmt_ms(g: pd.Series) -> str:
    if len(g) == 1:
        return f"{g.iloc[0]:.2f}"
    return f"{g.mean():.2f}±{g.std(ddof=1):.2f}"


def welch(a, b) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def main():
    old = {}
    new = {}
    for scen, (old_dir, new_dir) in PAIRS.items():
        old_csv = REPO / old_dir / "all_runs.csv"
        new_csv = REPO / new_dir / "all_runs.csv"
        if not new_csv.exists():
            sys.exit(f"Missing regenerated CSV: {new_csv}")
        old[scen] = pd.read_csv(old_csv) if old_csv.exists() else None
        new[scen] = pd.read_csv(new_csv)

    lines = [
        "# Paper-table regeneration: stale CSVs (pre-715d7b3) vs current code",
        "",
        "Old = local output/statistical_benchmark{,_s2,_s3} (stale pre-715d7b3",
        "precursors, NOT the paper tables). New = output/statistical_benchmark_",
        "v2_s{1,2,3}, regenerated with current code (single-circle footprint =",
        "paper geometry); these reproduce the paper campaign output/comfort_s*",
        "bit-exactly, so 'new' equals the published table values.",
        "",
        "## Mean±std per scenario / method",
        "",
    ]

    for scen in PAIRS:
        lines += [f"### {scen}", "",
                  "| metric | " + " | ".join(
                      f"{m} old → new" for m in ["CV", "LSTM", "SGAN"]) + " |",
                  "|---|---|---|---|"]
        for met in METRICS + NEW_ONLY:
            cells = []
            for method in ["CV", "LSTM", "SGAN"]:
                n_g = new[scen][new[scen]["method"] == method][met] \
                    if met in new[scen] else pd.Series(dtype=float)
                if old[scen] is not None and met in old[scen]:
                    o_g = old[scen][old[scen]["method"] == method][met]
                    cells.append(f"{fmt_ms(o_g)} → {fmt_ms(n_g)}")
                else:
                    cells.append(f"— → {fmt_ms(n_g)}")
            lines.append(f"| {met} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Paper-claim checks under current code", ""]

    verdicts = []

    # C1: zero collisions
    total = sum(len(new[s]) for s in PAIRS)
    coll = sum(int(new[s]["collision_count"].sum()) for s in PAIRS)
    verdicts.append(("C1 zero collisions (single-circle metric)",
                     coll == 0, f"{coll} collisions in {total} runs"))

    # C2: LSTM lowest stochastic ADE in S1, S2
    for scen in ["S1", "S2"]:
        d = new[scen]
        lstm = d[d.method == "LSTM"]["ade"]
        sgan = d[d.method == "SGAN"]["ade"]
        ok = lstm.mean() < sgan.mean()
        verdicts.append((f"C2 LSTM ADE < SGAN ADE in {scen}", ok,
                         f"LSTM {lstm.mean():.3f} vs SGAN {sgan.mean():.3f}, "
                         f"Welch p={welch(lstm, sgan):.2g}"))

    # C3: S2 MinDist ordering CV > LSTM > SGAN
    d = new["S2"]
    cv_md = d[d.method == "CV"]["min_dist_m"].mean()
    lstm_md = d[d.method == "LSTM"]["min_dist_m"].mean()
    sgan_md = d[d.method == "SGAN"]["min_dist_m"].mean()
    verdicts.append(("C3 S2 MinDist ordering CV > LSTM > SGAN",
                     cv_md > lstm_md > sgan_md,
                     f"CV {cv_md:.3f}, LSTM {lstm_md:.3f}, SGAN {sgan_md:.3f}"))

    # C4: S3 MinTTC ~0.85 for all methods
    d = new["S3"]
    ttcs = {m: d[d.method == m]["min_ttc_s"].mean() for m in ["CV", "LSTM", "SGAN"]}
    ok = all(abs(v - 0.85) < 0.05 for v in ttcs.values())
    verdicts.append(("C4 S3 MinTTC saturated near 0.85 s", ok,
                     ", ".join(f"{m} {v:.4f}" for m, v in ttcs.items())))

    # C5: S3 RMS jerk roughly double S1/S2
    if "rms_jerk" in new["S1"]:
        j = {s: new[s]["rms_jerk"].mean() for s in PAIRS}
        ratio = j["S3"] / max(j["S1"], j["S2"])
        verdicts.append(("C5 S3 RMS jerk >> S1/S2", ratio > 1.3,
                         f"S1 {j['S1']:.2f}, S2 {j['S2']:.2f}, S3 {j['S3']:.2f} "
                         f"(ratio {ratio:.2f})"))

    # C6: principal S3 LSTM-SGAN effect (time and min_dist)
    d = new["S3"]
    for met in ["time_s", "min_dist_m"]:
        p = welch(d[d.method == "LSTM"][met], d[d.method == "SGAN"][met])
        verdicts.append((f"C6 S3 LSTM vs SGAN {met} significant", p < 0.05,
                         f"Welch p={p:.2g}"))

    lines += ["| claim | holds? | detail |", "|---|---|---|"]
    for name, ok, detail in verdicts:
        lines.append(f"| {name} | {'YES' if ok else '**NO**'} | {detail} |")

    # Per-seed drift magnitude old vs new
    lines += ["", "## Per-seed drift old → new (max abs diff, stochastic methods)", ""]
    lines += ["| scenario | time_s | min_dist_m | ade |", "|---|---|---|---|"]
    for scen in PAIRS:
        if old[scen] is None:
            continue
        merged = new[scen].merge(old[scen], on=["method", "seed"],
                                 suffixes=("_new", "_old"))
        row = [scen]
        for met in ["time_s", "min_dist_m", "ade"]:
            row.append(f"{np.max(np.abs(merged[f'{met}_new'] - merged[f'{met}_old'])):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUTDIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
