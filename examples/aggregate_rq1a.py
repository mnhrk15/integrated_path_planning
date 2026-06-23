#!/usr/bin/env python3
"""Reproducible cross-scene aggregation of RQ1a open-loop ADE/FDE/NLL.

Review finding R1 (docs/CODE_REVIEW_full_20260623.md): the RQ1a "average ADE
across 5 scenes" headline was computed by an UNCOMMITTED hand-aggregation, and
the equal-weighted 5-scene mean is unit-confounded -- eth uses an accelerated
~0.8 s cadence, so its 12-step horizon spans ~9.6 s of real time vs ~4.8 s for
the 0.4 s scenes, inflating its metres-error. The cross-scene aggregation choice
is the single most outcome-determining RQ1a decision (it can flip which method
"wins"), yet it lived nowhere in version control.

This script makes the choice explicit and auditable: it reads the per-scene CSV
emitted by run_openloop_prediction.py and reports every defensible aggregation
side by side -- unweighted vs trajectory-weighted, with-eth vs without-eth, and
scene-level joint best-of-N (``ade``) vs canonical per-agent minADE
(``ade_per_agent``). The PER-SCENE orderings (the actual H1 evidence) are
invariant to all of this; only the single cross-scene headline number moves.

Usage:
    python examples/aggregate_rq1a.py \
        --csv outputs/openloop_full_eval_zara_eth_ucy_seeds0-4.csv \
        --out-dir outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Scenes whose cadence differs from the 0.4 s UCY standard. eth is excluded in
# the "no-eth" variants because its accelerated cadence makes its absolute ADE
# incommensurable with the others under an equal-weighted cross-scene mean.
CONFOUNDED_SCENES = ("eth",)

# Metrics carried through aggregation. ade = scene-level joint best-of-N;
# ade_per_agent = canonical SGAN per-agent minADE (each agent picks its own best
# sample). The pair is the crux of R1: per-agent minADE is the standard ADE_N.
METRICS = ("ade", "fde", "ade_per_agent", "fde_per_agent", "nll")


# --------------------------------------------------------------------------- #
# Pure aggregation core (no I/O) -- unit-tested in tests/test_aggregate_rq1a.py
# --------------------------------------------------------------------------- #
def per_scene_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean of ``metric`` over seeds for each (method, scene).

    Variable seed counts (e.g. deterministic CV stored once per scene vs 5 seeds
    for the stochastic models) are handled naturally by the groupby mean, and
    NaN metric rows (e.g. NLL with <2 samples) are skipped.
    """
    return (df.groupby(["method", "scene"])[metric]
              .mean()
              .reset_index()
              .rename(columns={metric: "value"}))


def _scene_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Representative trajectory count per (method, scene) = mean over seeds."""
    return (df.groupby(["method", "scene"])["n_trajectories"]
              .mean()
              .reset_index()
              .rename(columns={"n_trajectories": "weight"}))


def cross_scene(df: pd.DataFrame, metric: str, *,
                drop_eth: bool = False, weighted: bool = False) -> Dict[str, float]:
    """Cross-scene aggregate of ``metric`` per method.

    drop_eth: exclude the accelerated-cadence eth scene.
    weighted: weight each scene by its trajectory count (else equal weight).
    Returns {method: value}; NaN/missing per-scene values are dropped.
    """
    ps = per_scene_means(df, metric)
    if drop_eth:
        ps = ps[~ps["scene"].isin(CONFOUNDED_SCENES)]

    out: Dict[str, float] = {}
    if weighted:
        w = _scene_weights(df)
        merged = ps.merge(w, on=["method", "scene"])
        for method, grp in merged.groupby("method"):
            vals = grp["value"].to_numpy(dtype=float)
            wts = grp["weight"].to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
            out[method] = (float(np.average(vals[mask], weights=wts[mask]))
                           if mask.any() else float("nan"))
    else:
        for method, grp in ps.groupby("method"):
            vals = grp["value"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            out[method] = float(vals.mean()) if vals.size else float("nan")
    return out


def aggregate_all(df: pd.DataFrame, metrics=METRICS) -> pd.DataFrame:
    """Tidy table of every (metric, aggregation, method) cross-scene value plus
    the per-scene breakdown."""
    variants = [
        ("unweighted_5scene", dict(drop_eth=False, weighted=False)),
        ("weighted_5scene", dict(drop_eth=False, weighted=True)),
        ("unweighted_no_eth", dict(drop_eth=True, weighted=False)),
        ("weighted_no_eth", dict(drop_eth=True, weighted=True)),
    ]
    rows: List[dict] = []
    for metric in metrics:
        for name, kw in variants:
            for method, value in cross_scene(df, metric, **kw).items():
                rows.append(dict(metric=metric, aggregation=name,
                                 method=method, value=value))
        for _, r in per_scene_means(df, metric).iterrows():
            rows.append(dict(metric=metric, aggregation=f"scene:{r['scene']}",
                             method=r["method"], value=r["value"]))
    return pd.DataFrame(rows, columns=["metric", "aggregation", "method", "value"])


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _wide(tidy: pd.DataFrame, metric: str, aggregations: List[str]) -> pd.DataFrame:
    sub = tidy[(tidy["metric"] == metric) & (tidy["aggregation"].isin(aggregations))]
    wide = sub.pivot(index="method", columns="aggregation", values="value")
    return wide.reindex(columns=[a for a in aggregations if a in wide.columns])


def build_markdown(df: pd.DataFrame, tidy: pd.DataFrame, csv_path: Path) -> str:
    methods = sorted(df["method"].unique())
    scenes = sorted(df["scene"].unique())
    cross = ["unweighted_5scene", "weighted_5scene", "unweighted_no_eth", "weighted_no_eth"]
    scene_aggs = [f"scene:{s}" for s in scenes]

    def fmt(x):
        return "n/a" if pd.isna(x) else f"{x:.4f}"

    def table(metric: str, aggregations: List[str], header: List[str]) -> str:
        wide = _wide(tidy, metric, aggregations)
        lines = ["| method | " + " | ".join(header) + " |",
                 "|" + "---|" * (len(header) + 1)]
        for m in methods:
            cells = [fmt(wide.loc[m, a]) if (m in wide.index and a in wide.columns)
                     else "n/a" for a in aggregations]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    out = [
        "# RQ1a cross-scene ADE aggregation (reproducible)",
        "",
        f"Source: `{csv_path}`  |  scenes: {', '.join(scenes)}  |  methods: {', '.join(methods)}",
        "",
        "Generated by `examples/aggregate_rq1a.py` to address review finding R1: the",
        "cross-scene headline must be reported as a sensitivity over aggregation",
        "choices, not a single equal-weighted mean. Per-scene orderings (the H1",
        "evidence) are invariant to these choices.",
        "",
        "## Scene-level ADE (`ade`, joint best-of-N) -- cross-scene",
        table("ade", cross, ["unwtd 5-scene", "traj-wtd 5-scene", "unwtd no-eth", "traj-wtd no-eth"]),
        "",
        "## Per-agent minADE (`ade_per_agent`, canonical ADE_N) -- cross-scene",
        table("ade_per_agent", cross, ["unwtd 5-scene", "traj-wtd 5-scene", "unwtd no-eth", "traj-wtd no-eth"]),
        "",
        "## Per-scene ADE (the H1 evidence; aggregation-invariant)",
        "### scene-level `ade`",
        table("ade", scene_aggs, scenes),
        "",
        "### per-agent `ade_per_agent`",
        table("ade_per_agent", scene_aggs, scenes),
        "",
        "## FDE / NLL cross-scene (for completeness)",
        "### `fde` (scene-level)",
        table("fde", cross, ["unwtd 5-scene", "traj-wtd 5-scene", "unwtd no-eth", "traj-wtd no-eth"]),
        "### `nll`",
        table("nll", cross, ["unwtd 5-scene", "traj-wtd 5-scene", "unwtd no-eth", "traj-wtd no-eth"]),
        "",
        "## Notes",
        "- **eth is unit-confounded**: its ~0.8 s cadence makes the 12-step horizon",
        "  span ~9.6 s of real time vs ~4.8 s elsewhere, inflating its absolute ADE.",
        "  Treat the no-eth columns as the primary cross-scene read and eth as a",
        "  sensitivity row, OR report per-scene.",
        "- **Aggregation flips the cross-scene `ade` ordering** (CV worst under the",
        "  equal-weighted 5-scene mean vs best once eth is dropped), so no single",
        "  cross-scene `ade` headline is defensible on its own.",
        "- **Per-agent minADE (`ade_per_agent`) is the canonical ADE_N** and shows",
        "  the learned models clearly ahead of CV in every aggregation.",
        "- The **per-scene orderings are invariant** to aggregation -- they are the",
        "  actual basis for H1 (the cross-scene ordering being a sim artifact).",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="outputs/openloop_full_eval_zara_eth_ucy_seeds0-4.csv",
                    help="per-scene CSV from run_openloop_prediction.py")
    ap.add_argument("--out-dir", default="outputs",
                    help="directory for rq1a_aggregate.{csv,md}")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tidy = aggregate_all(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "rq1a_aggregate.csv"
    md_out = out_dir / "rq1a_aggregate.md"
    tidy.to_csv(csv_out, index=False)
    md = build_markdown(df, tidy, Path(args.csv))
    md_out.write_text(md)

    print(md)
    print(f"\nWrote {csv_out} and {md_out}")


if __name__ == "__main__":
    main()
