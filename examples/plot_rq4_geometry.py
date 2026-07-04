#!/usr/bin/env python3
"""Render the RQ3 real-loop encounter-geometry overview figure (thesis ch. 8).

Reads the committed trajectory tables under
``outputs/rq3_realloop/rq4_trajectories/`` (exported and cache-verified by
``export_rq4_trajectories.py``; no dataset access, no re-run) and draws a
2x2 grid, one panel per scenario family's representative encounter:

* recorded vehicle path (black dashed, start circle)
* recorded pedestrian trajectories (gray thin lines, start dots)
* planner-ego trajectory under the replay arm, SGAN predictor, seed 0,
  robust plan (blue) and single-sample plan (orange)

Figure-independence rule: labels name only arms, plan types, clip names and
axes -- no benchmark-campaign brand names, internal-hypothesis names,
p values or claim sentences are baked into the image.

Usage:
    .venv/bin/python examples/plot_rq4_geometry.py
    .venv/bin/python examples/plot_rq4_geometry.py \
        --trajdir outputs/rq3_realloop/rq4_trajectories \
        --out figs/rq4_geometry.png
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent

# Same fixed panel set as export_rq4_trajectories.TARGETS (one per family).
PANELS = [
    ("vci_back__back_interaction_02__e00", "rear approach"),
    ("vci_front__front_interaction_01__e00", "oncoming"),
    ("vci_lat_bi__bidirection_normal_driving_04__e00",
     "bidirectional crossing"),
    ("vci_lat_uni__unidirection_normal_driving_03__e00",
     "unidirectional crossing"),
]
COLOR_ROBUST = "#4C72B0"
COLOR_SINGLE = "#DD8452"
COLOR_RECORDED = "#111111"
COLOR_PED = "#9A9A9A"
PAD_M = 2.0          # margin around the recorded bounding box
MAX_EXTRA_M = 8.0    # cap on the extra room granted to planner excursions


def load_traj(csv_path: Path) -> Dict[str, np.ndarray]:
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} missing -- regenerate with "
            "examples/export_rq4_trajectories.py (needs the VCI dataset)")
    series: Dict[str, List[List[float]]] = defaultdict(list)
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            series[row["kind"]].append([float(row["t"]), float(row["x"]),
                                        float(row["y"])])
    return {k: np.asarray(v) for k, v in series.items()}


def panel_title(enc_id: str, family_note: str) -> str:
    scenario, clip, _ = enc_id.split("__")
    return f"{scenario} / {clip}\n({family_note})"


def draw(trajdir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2))
    for ax, (enc_id, family_note) in zip(axes.flat, PANELS):
        series = load_traj(trajdir / f"{enc_id}.csv")

        rec_xy = series["recorded_ego"][:, 1:]
        ped_keys = sorted(k for k in series if k.startswith("recorded_ped_"))
        ped_xy = np.vstack([series[k][:, 1:] for k in ped_keys])
        base = np.vstack([rec_xy, ped_xy])
        lo, hi = base.min(axis=0) - PAD_M, base.max(axis=0) + PAD_M

        for k in ped_keys:
            xy = series[k][:, 1:]
            ax.plot(xy[:, 0], xy[:, 1], color=COLOR_PED, linewidth=0.9,
                    alpha=0.8, zorder=2)
            ax.plot(xy[0, 0], xy[0, 1], linestyle="none", marker="o",
                    markersize=2.6, color=COLOR_PED, zorder=2)

        ax.plot(rec_xy[:, 0], rec_xy[:, 1], color=COLOR_RECORDED,
                linewidth=1.4, linestyle="--", zorder=3)
        ax.plot(rec_xy[0, 0], rec_xy[0, 1], linestyle="none", marker="o",
                markersize=6, markerfacecolor="white",
                markeredgecolor=COLOR_RECORDED, zorder=4)

        for plan, color in [("single", COLOR_SINGLE), ("robust", COLOR_ROBUST)]:
            xy = series[f"planner_ego_{plan}"][:, 1:]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.6, zorder=3)
            ax.plot(xy[-1, 0], xy[-1, 1], linestyle="none", marker="s",
                    markersize=3.5, color=color, zorder=4)
            # widen the view for planner excursions, capped so one outlier
            # cannot blow up the panel scale
            plo = np.maximum(xy.min(axis=0) - PAD_M, lo - MAX_EXTRA_M)
            phi = np.minimum(xy.max(axis=0) + PAD_M, hi + MAX_EXTRA_M)
            lo, hi = np.minimum(lo, plo), np.maximum(hi, phi)

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(panel_title(enc_id, family_note), fontsize=9.5)
        ax.set_xlabel("x [m]", fontsize=8.5)
        ax.set_ylabel("y [m]", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

    handles = [
        Line2D([], [], color=COLOR_RECORDED, linewidth=1.4, linestyle="--",
               marker="o", markersize=6, markerfacecolor="white",
               label="recorded vehicle (circle = start)"),
        Line2D([], [], color=COLOR_PED, linewidth=0.9,
               label="recorded pedestrians"),
        Line2D([], [], color=COLOR_ROBUST, linewidth=1.6, marker="s",
               markersize=3.5, label="planner ego, robust plan"
               " (square = end)"),
        Line2D([], [], color=COLOR_SINGLE, linewidth=1.6, marker="s",
               markersize=3.5, label="planner ego, single-sample plan"
               " (square = end)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajdir", type=Path,
                        default=REPO / "outputs" / "rq3_realloop"
                        / "rq4_trajectories")
    parser.add_argument("--out", type=Path,
                        default=REPO / "figs" / "rq4_geometry.png")
    args = parser.parse_args(argv)
    draw(args.trajdir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
