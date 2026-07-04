#!/usr/bin/env python3
"""Render the RQ3 real-loop V3 robust-gain distribution figure (thesis ch. 8).

Reads the committed per-encounter delta table
``outputs/rq3_realloop/v3_encounter_deltas.csv`` (no recomputation; exported
and gate-checked against v3_robust.csv by ``export_rq4_encounter_deltas.py``)
and draws one strip panel per learned predictor:

* (a) LSTM, (b) SGAN: per-encounter paired delta of the same-time min
  separation (robust - single-sample plan) across the five pedestrian arms,
  26 encounter dots per arm plus a median bar. The replay arm (recorded
  pedestrians) sits leftmost as the reference position.

The deterministic CV predictor is omitted: its 20-sample distribution
collapses, so robust == single bit-for-bit and every delta is exactly zero
(disclosed in the thesis caption, not in the image).

Figure-independence rule: labels name only arms, predictors, plan types and
metrics -- no benchmark-campaign brand names, internal-hypothesis names or
p values are baked into the image.

Usage:
    .venv/bin/python examples/plot_rq4_v3_gain.py
    .venv/bin/python examples/plot_rq4_v3_gain.py \
        --deltas outputs/rq3_realloop/v3_encounter_deltas.csv \
        --out figs/rq4_v3_robust_gain.png
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

ARMS = ["replay", "calib", "avec", "norep", "calib13x"]
ARM_LABELS = {
    "replay": "replay\n(recorded)",
    "calib": "calibrated",
    "avec": "hand-tuned",
    "norep": "no-repulsion",
    "calib13x": "calibrated-\n1.3x",
}
PREDS = ["lstm", "sgan"]
PRED_LABELS = {"lstm": "LSTM", "sgan": "SGAN"}
N_PAIRS = 26
COLOR_DOT = "#4C72B0"
COLOR_MEDIAN = "#333333"


def load_deltas(csv_path: Path) -> Dict[Tuple[str, str], List[float]]:
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} missing -- regenerate with "
            "examples/export_rq4_encounter_deltas.py")
    deltas: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            deltas[(row["ped_arm"], row["pred"])].append(float(row["delta_m"]))
    for arm in ARMS:
        for pred in PREDS:
            n = len(deltas.get((arm, pred), []))
            if n != N_PAIRS:
                raise SystemExit(
                    f"{csv_path}: {arm}/{pred} has {n} rows, expected "
                    f"{N_PAIRS} -- regenerate with "
                    "examples/export_rq4_encounter_deltas.py")
    return deltas


def draw(deltas: Dict[Tuple[str, str], List[float]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    rng = np.random.default_rng(0)  # fixed seed: deterministic jitter
    for ax, pred, tag in zip(axes, PREDS, ["(a)", "(b)"]):
        for i, arm in enumerate(ARMS):
            vals = np.asarray(deltas[(arm, pred)])
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.plot(i + jitter, vals, linestyle="none", marker="o",
                    markersize=3.6, color=COLOR_DOT, alpha=0.55, zorder=2)
            med = float(np.median(vals))
            ax.plot([i - 0.26, i + 0.26], [med, med], color=COLOR_MEDIAN,
                    linewidth=2.0, zorder=3)
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--",
                   zorder=1)
        ax.axvline(0.5, color="#BBBBBB", linewidth=0.8, linestyle=":",
                   zorder=1)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([ARM_LABELS[a] for a in ARMS], fontsize=8)
        ax.set_title(f"{tag} {PRED_LABELS[pred]} predictor", fontsize=10.5)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("$\\Delta$ min. separation [m]\n"
                       "(robust $-$ single-sample plan)", fontsize=9)
    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=3.6,
                   color=COLOR_DOT, alpha=0.55,
                   label=f"encounter (n={N_PAIRS})"),
        plt.Line2D([], [], color=COLOR_MEDIAN, linewidth=2.0,
                   label="median"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deltas", type=Path,
                        default=REPO / "outputs" / "rq3_realloop"
                        / "v3_encounter_deltas.csv")
    parser.add_argument("--out", type=Path,
                        default=REPO / "figs" / "rq4_v3_robust_gain.png")
    args = parser.parse_args(argv)
    draw(load_deltas(args.deltas), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
