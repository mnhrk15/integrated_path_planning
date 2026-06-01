#!/usr/bin/env python3
"""Generate lateral-analysis figure for Scenario 3 (AVEC Full Paper Fig 4).

Reads output/scenario_03/trajectory.npz and produces a 3-pane plot of
x(t), Frenet lateral deviation d(t), and yaw psi(t) for the SGAN run,
with a dashed vertical line at the time the ego passes near (4, -10).
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.core.coordinate_converter import CoordinateConverter
from src.planning import CubicSpline2D

SCENARIO = REPO_ROOT / "scenarios" / "scenario_03.yaml"
TRAJ_NPZ = REPO_ROOT / "output" / "scenario_03" / "trajectory.npz"
OUT_PNG = Path("/Users/mnhrk/Research/AVEC_FullPaper/figs/scenario_03_lateral.png")


def main() -> None:
    cfg = load_config(str(SCENARIO))
    csp = CubicSpline2D(cfg.reference_waypoints_x, cfg.reference_waypoints_y)
    conv = CoordinateConverter(csp)

    data = np.load(TRAJ_NPZ, allow_pickle=True)
    times = data["times"]
    ego_x = data["ego_x"]
    ego_y = data["ego_y"]
    planned_yaw = data["planned_yaw"]

    # yaw(t): take first element of each step's planned yaw; backfill empties
    yaw = np.zeros(len(times))
    last = 0.0
    for i, py in enumerate(planned_yaw):
        if hasattr(py, "__len__") and len(py) > 0:
            yaw[i] = float(py[0])
            last = yaw[i]
        else:
            yaw[i] = last

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

    # Time of nearest approach to (4, -10)
    idx_mark = int(np.argmin(np.hypot(ego_x - 4.0, ego_y - (-10.0))))
    t_mark = float(times[idx_mark])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5.0, 5.5))
    axes[0].plot(times, ego_x, color="C0")
    axes[0].set_ylabel(r"$x$ [m]")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, d_values, color="C1")
    axes[1].set_ylabel(r"$d$ [m]")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, yaw, color="C2")
    axes[2].set_ylabel(r"$\psi$ [rad]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.axvline(t_mark, ls="--", color="red", lw=1.0)
    axes[0].annotate(
        f"$t={t_mark:.1f}$ s\n$(x,y)\\approx(4,-10)$",
        xy=(t_mark, ego_x[idx_mark]),
        xytext=(t_mark + 0.5, ego_x[idx_mark] - 6),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PNG} (marker at t={t_mark:.2f}s, idx={idx_mark})")


if __name__ == "__main__":
    main()
