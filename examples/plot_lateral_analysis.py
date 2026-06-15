#!/usr/bin/env python3
"""Generate the right-turn behavior figure for Scenario 3 (AVEC Full Paper Fig 4).

Reads trajectory.npz (default output/scenario_03/, overridable with --input)
and produces a 3-pane plot of speed v(t), Frenet lateral deviation d(t), and
yaw psi(t) for the SGAN run, with the full yield stop (v < 0.05 for at least
0.5 s) shaded in every pane.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.core.coordinate_converter import CoordinateConverter
from src.planning import CubicSpline2D

SCENARIO = REPO_ROOT / "scenarios" / "scenario_03.yaml"
OUT_PNG = Path("/Users/mnhrk/Research/AVEC_FullPaper/figs/scenario_03_lateral.png")


def stop_windows(times, ego_v, v_stop=0.05, min_dur=0.5):
    """All (t_start, t_end) windows where the ego is fully stopped."""
    idx = np.where(np.asarray(ego_v, float) < v_stop)[0]
    windows = []
    if idx.size:
        for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if times[seg[-1]] - times[seg[0]] >= min_dur:
                windows.append((float(times[seg[0]]), float(times[seg[-1]])))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(REPO_ROOT / "output" / "scenario_03"),
        help="Directory containing trajectory.npz (default: output/scenario_03)")
    args = parser.parse_args()

    cfg = load_config(str(SCENARIO))
    csp = CubicSpline2D(cfg.reference_waypoints_x, cfg.reference_waypoints_y)
    conv = CoordinateConverter(csp)

    data = np.load(Path(args.input) / "trajectory.npz", allow_pickle=True)
    times = data["times"]
    ego_x = data["ego_x"]
    ego_y = data["ego_y"]
    ego_v = data["ego_v"]
    ego_yaw = data["ego_yaw"]

    # d(t): signed lateral distance from the reference path
    d_values = np.zeros(len(times))
    for i in range(len(times)):
        rs, rx, ry, rtheta, _, _ = conv.find_nearest_point_on_path(
            float(ego_x[i]), float(ego_y[i])
        )
        dx = float(ego_x[i]) - rx
        dy = float(ego_y[i]) - ry
        sign = math.copysign(1.0, math.cos(rtheta) * dy - math.sin(rtheta) * dx)
        d_values[i] = sign * math.hypot(dx, dy)

    windows = stop_windows(times, ego_v)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5.0, 5.5))
    axes[0].plot(times, ego_v, color="C0")
    axes[0].set_ylabel(r"$v$ [m/s]")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, d_values, color="C1")
    axes[1].set_ylabel(r"$d$ [m]")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, ego_yaw, color="C2")
    axes[2].set_ylabel(r"$\psi$ [rad]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.3)

    for t0, t1 in windows:
        for ax in axes:
            ax.axvspan(t0, t1, color="red", alpha=0.12, zorder=0)

    if windows:
        t0, t1 = max(windows, key=lambda w: w[1] - w[0])
        axes[0].annotate(
            f"yield stop\n({t1 - t0:.1f} s)",
            xy=((t0 + t1) / 2.0, 0.1),
            xytext=(t1 + 1.0, 1.8),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    stops = ", ".join(f"{t0:.1f}-{t1:.1f}s" for t0, t1 in windows) or "none"
    print(f"Saved {OUT_PNG} (stop windows: {stops})")


if __name__ == "__main__":
    main()
