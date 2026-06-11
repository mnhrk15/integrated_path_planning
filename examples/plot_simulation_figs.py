#!/usr/bin/env python3
"""Generate clean trajectory-map figures for the AVEC full paper (Figs 1-3).

Reads trajectory.npz per scenario (default output/scenario_0X/, overridable
with --inputs) and the scenario YAML map_config, then renders a
tightly-cropped, consistently styled figure per scenario into the paper's
figs/ directory. Companion to plot_lateral_analysis.py. Safe to re-run;
writes only the three PNGs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
FIGS = Path("/Users/mnhrk/Research/AVEC_FullPaper/figs")

# Per-scenario tweaks. "annotate_stop" marks the longest full stop (v < 0.05
# for at least 0.5 s) detected in the run, the yield behavior analyzed in the
# paper's discussion section.
SCENARIOS = {
    "scenario_01": {},
    "scenario_02": {"ylim": (-6.0, 6.0)},
    "scenario_03": {"annotate_stop": True},
}


def find_longest_stop(times, ego_v, v_stop=0.05, min_dur=0.5):
    """Return (t_start, t_end, idx_start) of the longest full stop, or None."""
    idx = np.where(np.asarray(ego_v, float) < v_stop)[0]
    if idx.size == 0:
        return None
    best = None
    for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        dur = times[seg[-1]] - times[seg[0]]
        if dur >= min_dur and (best is None or dur > best[1] - best[0]):
            best = (float(times[seg[0]]), float(times[seg[-1]]), int(seg[0]))
    return best


def load_map(scn):
    with open(REPO / "scenarios" / f"{scn}.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("map_config", {}) or {}


def plot_one(scn, opts, input_dir=None):
    npz_dir = Path(input_dir) if input_dir else REPO / "output" / scn
    data = np.load(npz_dir / "trajectory.npz", allow_pickle=True)
    ego_x = np.asarray(data["ego_x"], float)
    ego_y = np.asarray(data["ego_y"], float)
    ped = np.asarray(data["ped_positions"], float)  # [T, N, 2]
    mp = load_map(scn)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # --- static map ---
    for b in mp.get("road_borders", []):
        if len(b) == 4:
            x1, y1, x2, y2 = b
            ax.plot([x1, x2], [y1, y2], "-", color="0.25", lw=1.5, alpha=0.9, zorder=1)
    for ln in mp.get("lanes", []):
        if len(ln) == 4:
            x1, y1, x2, y2 = ln
            ax.plot([x1, x2], [y1, y2], "--", color="0.55", lw=0.8, alpha=0.5, zorder=1)
    has_cw = False
    for cw in mp.get("crosswalks", []):
        if len(cw) >= 4:
            x, y, w, h = cw[:4]
            ang = cw[4] if len(cw) > 4 else 0.0
            ax.add_patch(Rectangle((x, y), w, h, angle=ang, facecolor="none",
                                   edgecolor="0.5", hatch="///", lw=0.5, alpha=0.6, zorder=0))
            has_cw = True

    # --- pedestrian trajectories (thin faint lines) ---
    for n in range(ped.shape[1]):
        ax.plot(ped[:, n, 0], ped[:, n, 1], "-", color="tab:red", lw=0.8, alpha=0.35, zorder=2)

    # --- ego trajectory + start/end ---
    ax.plot(ego_x, ego_y, "-", color="tab:blue", lw=2.0, zorder=4)
    ax.plot(ego_x[0], ego_y[0], "o", color="green", ms=6, zorder=5)
    ax.plot(ego_x[-1], ego_y[-1], "o", color="red", ms=6, zorder=5)

    if opts.get("annotate_stop"):
        stop = find_longest_stop(np.asarray(data["times"], float),
                                 np.asarray(data["ego_v"], float))
        if stop is not None:
            t0, t1, i0 = stop
            sx, sy = float(ego_x[i0]), float(ego_y[i0])
            ax.plot(sx, sy, "*", color="black", ms=9, zorder=6)
            ax.annotate(f"yield stop ({t1 - t0:.1f} s)", xy=(sx, sy),
                        xytext=(sx + 6.0, sy - 5.0), fontsize=8, color="black",
                        zorder=6,
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # --- tight limits from ego + pedestrians (+ margin) ---
    xs = np.concatenate([ego_x, ped[..., 0].ravel()])
    ys = np.concatenate([ego_y, ped[..., 1].ravel()])
    xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    m = 2.5
    ax.set_xlim(xs.min() - m, xs.max() + m)
    if opts.get("ylim"):
        ax.set_ylim(*opts["ylim"])
    else:
        ax.set_ylim(ys.min() - m, ys.max() + m)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # --- legend (single row, above axes) ---
    handles = [
        Line2D([0], [0], color="tab:blue", lw=2.0, label="Ego"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", ms=6, label="Start"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", ms=6, label="End"),
        Line2D([0], [0], color="tab:red", lw=1.2, alpha=0.6, label="Pedestrians"),
    ]
    if has_cw:
        handles.append(Patch(facecolor="none", edgecolor="0.5", hatch="///", label="Crosswalk"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=len(handles), frameon=False, fontsize=8,
              handletextpad=0.4, columnspacing=1.2)

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"{scn}_sim.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  xlim=({xs.min()-m:.1f},{xs.max()+m:.1f}) ylim=({ys.min()-m:.1f},{ys.max()+m:.1f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs=3, metavar=("S1_DIR", "S2_DIR", "S3_DIR"), default=None,
        help="Directories containing trajectory.npz for scenarios 1-3 "
             "(default: output/scenario_0X)")
    args = parser.parse_args()
    input_dirs = args.inputs if args.inputs else [None] * 3
    for (scn, opts), input_dir in zip(SCENARIOS.items(), input_dirs):
        plot_one(scn, opts, input_dir=input_dir)


if __name__ == "__main__":
    main()
