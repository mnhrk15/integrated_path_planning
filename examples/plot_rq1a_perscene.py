#!/usr/bin/env python3
"""Render the RQ1a per-scene ADE figure (thesis chapter 5, fig-rq1-perscene).

Reads the tidy aggregation table ``outputs/rq1a_aggregate.csv`` written by
aggregate_rq1a.py and draws the per-scene ADE of the three predictors as two
grouped-bar panels:

* (a) scene-level (joint) best-of-N ADE -- the closed-loop benchmark's metric
  convention; the per-scene ordering visibly flips across scenes.
* (b) per-agent min-ADE -- the standard per-agent best-of-N convention; the
  learned predictors lead CV on (almost) every scene.

The best method of each scene is marked with a star so the ordering (not the
absolute level) is what the eye compares. Exact values live in the thesis
table / the CSV, so bars carry no value labels.

Figure-independence rule: labels name only methods, scenes and metrics --
no benchmark-campaign or internal-hypothesis names are baked into the image.

Usage:
    .venv/bin/python examples/plot_rq1a_perscene.py
    .venv/bin/python examples/plot_rq1a_perscene.py \
        --csv outputs/rq1a_aggregate.csv --out figs/rq1a_perscene_ade.png
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
METHODS = ["cv", "lstm", "sgan"]
METHOD_LABELS = {"cv": "CV", "lstm": "LSTM", "sgan": "SGAN"}
METHOD_COLORS = {"cv": "#4C72B0", "lstm": "#DD8452", "sgan": "#55A868"}
PANELS = [
    ("ade", "(a) scene-level best-of-$N$ ADE [m]"),
    ("ade_per_agent", "(b) per-agent min-ADE [m]"),
]


def load_per_scene(csv_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return values[metric][method][scene] from the tidy aggregation CSV."""
    values: Dict[str, Dict[str, Dict[str, float]]] = {
        metric: {m: {} for m in METHODS} for metric, _ in PANELS
    }
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            metric = row["metric"]
            if metric not in values:
                continue
            agg = row["aggregation"]
            if not agg.startswith("scene:"):
                continue
            scene = agg.split(":", 1)[1]
            if scene in SCENES and row["method"] in METHODS and row["value"]:
                values[metric][row["method"]][scene] = float(row["value"])
    for metric, per_method in values.items():
        for method, per_scene in per_method.items():
            missing = [s for s in SCENES if s not in per_scene]
            if missing:
                raise SystemExit(
                    f"{csv_path}: missing {metric}/{method} for scenes {missing} "
                    "-- regenerate with examples/aggregate_rq1a.py"
                )
    return values


def draw(values: Dict[str, Dict[str, Dict[str, float]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    width = 0.26
    for ax, (metric, panel_title) in zip(axes, PANELS):
        for j, method in enumerate(METHODS):
            xs = [i + (j - 1) * width for i in range(len(SCENES))]
            ys = [values[metric][method][s] for s in SCENES]
            ax.bar(xs, ys, width=width, color=METHOD_COLORS[method],
                   label=METHOD_LABELS[method], zorder=2)
        # star the best (lowest-ADE) method of each scene
        for i, scene in enumerate(SCENES):
            best = min(METHODS, key=lambda m: values[metric][m][scene])
            j = METHODS.index(best)
            ax.plot(i + (j - 1) * width, values[metric][best][scene] + 0.045,
                    marker="*", markersize=11, color="#B8860B", zorder=3,
                    linestyle="none")
        ax.set_xticks(range(len(SCENES)))
        ax.set_xticklabels(SCENES)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ADE [m]")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path,
                        default=REPO / "outputs" / "rq1a_aggregate.csv")
    parser.add_argument("--out", type=Path,
                        default=REPO / "figs" / "rq1a_perscene_ade.png")
    args = parser.parse_args(argv)
    draw(load_per_scene(args.csv), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
