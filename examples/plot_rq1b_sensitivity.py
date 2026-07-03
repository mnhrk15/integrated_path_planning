#!/usr/bin/env python3
"""Render the RQ1b reaction-model sensitivity figure (thesis chapter 7).

Reads the committed campaign tables under ``outputs/rq1b/`` (no recomputation)
and draws two panels:

* (a) margin campaign, hand-tuned GT: mean min. distance of the robust plan
  (eps=0) vs the single-sample plan (no margin inflation) per scenario, with
  collided-run counts annotated on the bars. The same robust/single pair of
  the other five GT arms is overlaid as gray dots, so the eye checks that the
  direction of the gain is unchanged across the whole sweep.
* (b) rand campaign, scenario 2: collided-run fraction of the pooled
  single-sample group vs the robust group across the six GT reaction arms,
  with raw counts annotated. Scenario 2 under the hand-tuned arm is the only
  cell whose one-sided Fisher test is significant (p = 0.0078); the signal
  vanishes under the calibrated arms.

Figure-independence rule: labels name only arms, scenarios, plan types and
metrics -- no benchmark-campaign brand names or internal-hypothesis names are
baked into the image. Exact values live in the thesis tables / the CSVs, so
bars carry only the collided-run count annotations needed to read the panel.

Usage:
    .venv/bin/python examples/plot_rq1b_sensitivity.py
    .venv/bin/python examples/plot_rq1b_sensitivity.py \
        --means outputs/rq1b/means.csv \
        --scenario-rand outputs/rq1b/scenario_rand.csv \
        --out figs/rq1b_sensitivity.png
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

SCENARIOS = ["scenario_01", "scenario_02", "scenario_03"]
SCENARIO_LABELS = {"scenario_01": "S1", "scenario_02": "S2", "scenario_03": "S3"}

GT_ARMS = ["avec", "calib", "calib_lo", "calib_hi", "calib_loso_vmax",
           "calib_loso_smin"]
GT_LABELS = {
    "avec": "hand-tuned",
    "calib": "calibrated",
    "calib_lo": "−1SD",
    "calib_hi": "+1SD",
    "calib_loso_vmax": "LOSO $V_0$-max",
    "calib_loso_smin": "LOSO $\\sigma$-min",
}
MAIN_ARM = "avec"

ROBUST_COND = "sgan_robust_eps0.0"
SINGLE_COND = "sgan_single_inf1.00"
COLOR_ROBUST = "#4C72B0"
COLOR_SINGLE = "#DD8452"
COLOR_OTHER = "#8C8C8C"


def load_margin_pairs(csv_path: Path) -> Dict[str, Dict[str, Dict[str, Tuple[float, int, int]]]]:
    """Return pairs[gt][scenario][cond] = (min_dist_mean, collisions, n)."""
    pairs: Dict[str, Dict[str, Dict[str, Tuple[float, int, int]]]] = {
        gt: {s: {} for s in SCENARIOS} for gt in GT_ARMS
    }
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["campaign"] != "margin" or row["condition"] not in (
                    ROBUST_COND, SINGLE_COND):
                continue
            gt, scen = row["gt_label"], row["scenario"]
            if gt in pairs and scen in SCENARIOS:
                pairs[gt][scen][row["condition"]] = (
                    float(row["min_dist_mean"]), int(row["collisions"]),
                    int(row["n"]))
    for gt, per_scen in pairs.items():
        for scen, conds in per_scen.items():
            missing = [c for c in (ROBUST_COND, SINGLE_COND) if c not in conds]
            if missing:
                raise SystemExit(
                    f"{csv_path}: missing margin rows {gt}/{scen}/{missing} "
                    "-- regenerate with examples/run_rq1b_sensitivity.py")
    return pairs


def load_s2_rand(csv_path: Path) -> Dict[str, Dict[str, int]]:
    """Return counts[gt] = {single_collided, single_n, robust_collided, robust_n}."""
    counts: Dict[str, Dict[str, int]] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["scenario"] != "scenario_02" or row["gt_label"] not in GT_ARMS:
                continue
            counts[row["gt_label"]] = {
                "single_collided": int(row["single_collided_runs"]),
                "single_n": int(row["single_n"]),
                "robust_collided": int(row["robust_collided_runs"]),
                "robust_n": int(row["robust_n"]),
                "fisher_p": float(row["fisher_p"]),
            }
    missing = [gt for gt in GT_ARMS if gt not in counts]
    if missing:
        raise SystemExit(
            f"{csv_path}: missing scenario_02 rows for {missing} "
            "-- regenerate with examples/run_rq1b_sensitivity.py")
    return counts


def draw(pairs, s2_counts, out_path: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    width = 0.32

    # --- (a) margin campaign: robust vs single min distance, hand-tuned GT ---
    other_arms = [gt for gt in GT_ARMS if gt != MAIN_ARM]
    for j, (cond, color, label) in enumerate([
            (ROBUST_COND, COLOR_ROBUST, "robust plan"),
            (SINGLE_COND, COLOR_SINGLE, "single-sample plan")]):
        xs = [i + (j - 0.5) * width for i in range(len(SCENARIOS))]
        ys = [pairs[MAIN_ARM][s][cond][0] for s in SCENARIOS]
        ax_a.bar(xs, ys, width=width, color=color, label=label, zorder=2)
        for x, scen in zip(xs, SCENARIOS):
            dist, coll, n = pairs[MAIN_ARM][scen][cond]
            dots = [pairs[gt][scen][cond][0] for gt in other_arms]
            if coll:
                ax_a.annotate(f"{coll}/{n} collided",
                              (x, max([dist] + dots) + 0.07),
                              ha="center", fontsize=7.5, color="#333333",
                              zorder=4)
            ax_a.plot([x] * len(dots), dots, linestyle="none", marker="o",
                      markersize=3.5, color=COLOR_OTHER, zorder=3)
    ax_a.plot([], [], linestyle="none", marker="o", markersize=3.5,
              color=COLOR_OTHER, label="other 5 GT arms")
    ax_a.set_xticks(range(len(SCENARIOS)))
    ax_a.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax_a.set_ylim(0, 3.05)
    ax_a.set_ylabel("mean min. distance [m]")
    ax_a.set_title("(a) robust vs single-sample plan\n"
                   "(margin campaign, hand-tuned GT bars)", fontsize=10.5)
    ax_a.grid(axis="y", alpha=0.3, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper left", fontsize=8)

    # --- (b) rand campaign, scenario 2: collided-run fraction per GT arm ---
    for j, (group, color, label) in enumerate([
            ("single", COLOR_SINGLE, "single-sample group"),
            ("robust", COLOR_ROBUST, "robust group")]):
        xs = [i + (j - 0.5) * width for i in range(len(GT_ARMS))]
        fracs, notes = [], []
        for gt in GT_ARMS:
            coll = s2_counts[gt][f"{group}_collided"]
            n = s2_counts[gt][f"{group}_n"]
            fracs.append(coll / n)
            notes.append(f"{coll}/{n}")
        ax_b.bar(xs, fracs, width=width, color=color, label=label, zorder=2)
        # stagger the two groups' count labels so zero-height annotations
        # of adjacent bars do not run into each other
        note_dy = 0.004 if group == "single" else 0.016
        for x, frac, note in zip(xs, fracs, notes):
            ax_b.annotate(note, (x, frac + note_dy), ha="center", fontsize=7,
                          color="#333333", zorder=3)
    avec_p = s2_counts[MAIN_ARM]["fisher_p"]
    ax_b.annotate(f"Fisher $p={avec_p:.4f}$",
                  (0, s2_counts[MAIN_ARM]["single_collided"]
                   / s2_counts[MAIN_ARM]["single_n"] + 0.025),
                  ha="left", fontsize=8.5, color="#333333")
    ax_b.set_xticks(range(len(GT_ARMS)))
    ax_b.set_xticklabels([GT_LABELS[gt] for gt in GT_ARMS], fontsize=8,
                         rotation=20, ha="right")
    ax_b.set_ylabel("collided-run fraction")
    ax_b.set_ylim(0, 0.2)
    ax_b.set_title("(b) collided runs in scenario S2 per GT arm\n"
                   "(rand campaign, single-sample vs robust group)",
                   fontsize=10.5)
    ax_b.grid(axis="y", alpha=0.3, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--means", type=Path,
                        default=REPO / "outputs" / "rq1b" / "means.csv")
    parser.add_argument("--scenario-rand", type=Path,
                        default=REPO / "outputs" / "rq1b" / "scenario_rand.csv")
    parser.add_argument("--out", type=Path,
                        default=REPO / "figs" / "rq1b_sensitivity.png")
    args = parser.parse_args(argv)
    draw(load_margin_pairs(args.means), load_s2_rand(args.scenario_rand),
         args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
