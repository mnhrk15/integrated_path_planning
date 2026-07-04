#!/usr/bin/env python3
"""Render the RQ3 real-loop V1 collision-discordance figure (thesis chapter 8).

Reads the committed paired table ``outputs/rq3_realloop/paired_v1.csv`` (no
recomputation) and draws two panels:

* (a) diverging horizontal bars per (SFM arm, predictor) cell: encounters
  whose collision occurs ONLY under the recorded (replay) pedestrians extend
  to the left, encounters whose collision occurs ONLY under the SFM arm
  extend to the right. The right side is empty in every cell (all
  ``coll_arm_only`` = 0), which is the one-sidedness the panel shows.
* (b) mean paired difference of the same-time min separation (SFM arm minus
  replay) per arm, one marker per (predictor, plan) cell, with the zero line
  as reference.

Figure-independence rule: labels name only arms, predictors, plan types and
metrics -- no benchmark-campaign brand names, internal-hypothesis names or
p values are baked into the image. Exact statistics live in the thesis
tables / the committed CSVs.

Usage:
    .venv/bin/python examples/plot_rq4_v1_collisions.py
    .venv/bin/python examples/plot_rq4_v1_collisions.py \
        --paired outputs/rq3_realloop/paired_v1.csv \
        --out figs/rq4_v1_collisions.png
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent

ARMS = ["calib", "avec", "norep", "calib13x"]
ARM_LABELS = {
    "calib": "calibrated",
    "avec": "hand-tuned",
    "norep": "no-repulsion",
    "calib13x": "calibrated-1.3x",
}
PREDS = ["cv", "lstm", "sgan"]
PRED_LABELS = {"cv": "CV", "lstm": "LSTM", "sgan": "SGAN"}
PLANS = ["single", "robust"]
PLAN_LABELS = {"single": "single-sample plan", "robust": "robust plan"}
COLOR_ROBUST = "#4C72B0"
COLOR_SINGLE = "#DD8452"
N_PAIRS = 26


def load_paired(csv_path: Path) -> Dict[Tuple[str, str, str], Dict]:
    """Return rows[(ped_arm, pred, plan)] = {replay_only, arm_only, delta}."""
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} missing -- regenerate with "
            "examples/make_rq3_report.py")
    rows: Dict[Tuple[str, str, str], Dict] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["ped_arm"], row["pred"], row["plan"])
            rows[key] = {
                "replay_only": int(row["coll_replay_only"]),
                "arm_only": int(row["coll_arm_only"]),
                "delta": float(row["mean_delta_m"]),
                "n_pairs": int(row["n_pairs"]),
            }
    missing = [(a, p, pl) for a in ARMS for p in PREDS for pl in PLANS
               if (a, p, pl) not in rows]
    if missing:
        raise SystemExit(
            f"{csv_path}: missing paired rows {missing} -- regenerate with "
            "examples/make_rq3_report.py")
    return rows


def draw(rows: Dict[Tuple[str, str, str], Dict], out_path: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # --- (a) diverging bars: replay-only (left) vs arm-only (right) ---
    cells = [(arm, pred) for arm in ARMS for pred in PREDS]
    bar_h = 0.34
    ys = list(range(len(cells)))[::-1]  # top-down reading order
    for (arm, pred), y in zip(cells, ys):
        for plan, dy, color in [("single", +0.5 * bar_h, COLOR_SINGLE),
                                ("robust", -0.5 * bar_h, COLOR_ROBUST)]:
            r = rows[(arm, pred, plan)]
            ax_a.barh(y + dy, -r["replay_only"], height=bar_h, color=color,
                      zorder=2)
            ax_a.barh(y + dy, r["arm_only"], height=bar_h, color=color,
                      zorder=2)
            if r["replay_only"]:
                ax_a.annotate(str(r["replay_only"]),
                              (-r["replay_only"] - 0.15, y + dy),
                              ha="right", va="center", fontsize=7,
                              color="#333333")
        arm_zero = sum(rows[(arm, pred, pl)]["arm_only"] for pl in PLANS)
        ax_a.annotate(str(arm_zero), (0.35, y), ha="left", va="center",
                      fontsize=7, color="#888888")
    ax_a.axvline(0.0, color="#333333", linewidth=0.8, zorder=3)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels(
        [f"{ARM_LABELS[a]} · {PRED_LABELS[p]}" for a, p in cells], fontsize=8)
    ax_a.set_xlim(-9, 3)
    ax_a.set_xticks([-8, -6, -4, -2, 0, 2])
    ax_a.set_xticklabels(["8", "6", "4", "2", "0", "2"])
    ax_a.set_xlabel(f"discordant-collision encounters (of {N_PAIRS})",
                    fontsize=9)
    ax_a.annotate("collision only under\nrecorded pedestrians (replay)",
                  (-8.7, ys[0] + 0.9), ha="left", va="bottom", fontsize=8,
                  color="#555555")
    ax_a.annotate("collision only\nunder SFM arm", (0.6, ys[0] + 0.9),
                  ha="left", va="bottom", fontsize=8, color="#555555")
    ax_a.set_ylim(ys[-1] - 0.8, ys[0] + 2.4)
    ax_a.set_title("(a) collision discordance vs replay reference",
                   fontsize=10.5)
    ax_a.grid(axis="x", alpha=0.3, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SINGLE,
                      label=PLAN_LABELS["single"]),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ROBUST,
                      label=PLAN_LABELS["robust"]),
    ], loc="lower left", fontsize=8)

    # --- (b) mean paired delta of min separation per arm ---
    pred_markers = {"cv": "s", "lstm": "o", "sgan": "^"}
    for i, arm in enumerate(ARMS):
        for pred in PREDS:
            for plan, color, dx in [("single", COLOR_SINGLE, -0.13),
                                    ("robust", COLOR_ROBUST, +0.13)]:
                r = rows[(arm, pred, plan)]
                ax_b.plot(i + dx, r["delta"], linestyle="none",
                          marker=pred_markers[pred], markersize=6,
                          color=color, alpha=0.85, zorder=3)
    ax_b.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--",
                 zorder=1)
    ax_b.set_xticks(range(len(ARMS)))
    ax_b.set_xticklabels([ARM_LABELS[a] for a in ARMS], fontsize=8,
                         rotation=15, ha="right")
    ax_b.set_xlim(-0.55, 3.55)
    ax_b.set_ylim(-0.28, 0.54)
    ax_b.set_ylabel("mean paired $\\Delta$ min. separation [m]\n"
                    "(SFM arm $-$ replay)", fontsize=9)
    ax_b.set_title("(b) same-time min-separation shift", fontsize=10.5)
    ax_b.grid(axis="y", alpha=0.3, zorder=0)
    ax_b.set_axisbelow(True)
    handles = [Line2D([], [], linestyle="none", marker=pred_markers[p],
                      color="#555555", markersize=6, label=PRED_LABELS[p])
               for p in PREDS]
    handles += [Line2D([], [], linestyle="none", marker="o",
                       color=COLOR_SINGLE, markersize=6,
                       label=PLAN_LABELS["single"]),
                Line2D([], [], linestyle="none", marker="o",
                       color=COLOR_ROBUST, markersize=6,
                       label=PLAN_LABELS["robust"])]
    ax_b.legend(handles=handles, loc="upper right", fontsize=7.5, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paired", type=Path,
                        default=REPO / "outputs" / "rq3_realloop"
                        / "paired_v1.csv")
    parser.add_argument("--out", type=Path,
                        default=REPO / "figs" / "rq4_v1_collisions.png")
    args = parser.parse_args(argv)
    draw(load_paired(args.paired), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
