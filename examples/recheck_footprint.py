#!/usr/bin/env python3
"""Post-hoc footprint recheck of saved trajectories (A-4, step 2 of *3).

Re-evaluates every saved trajectory.npz under a vehicle-shaped footprint
without re-running any simulation, to test whether the paper's "zero
collisions" finding survives replacing the legacy single-circle ego footprint
(radius 1.0 m, centre) with the drawn 4.5 x 2.0 m rectangle.

Three geometries are evaluated per timestep against all pedestrians:
  legacy : centre distance, collision if < ego_radius + ped_radius (1.2 m)
  multi  : 3-circle cover of the rectangle (offsets 0, +-1.5 m, r = 1.25 m),
           collision if any circle-centre distance < r + ped_radius (1.45 m)
  rect   : exact distance from the oriented rectangle to the pedestrian
           centre, collision if < ped_radius (footprint ground truth; the
           multi-circle check is conservative relative to this)

Ego heading uses the stored ego_yaw when present (new runs) and is otherwise
reconstructed by finite differences of (ego_x, ego_y), holding the last valid
heading through standstill phases.

Outputs per-trajectory rows to output/footprint_recheck/results.csv and a
summary to output/footprint_recheck/REPORT.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.footprint import EgoFootprint

REPO = Path(__file__).parent.parent
OUTDIR = REPO / "output" / "footprint_recheck"

# All saved runs use these values (verified across scenarios/*.yaml)
EGO_RADIUS = 1.0
PED_RADIUS = 0.2
VEHICLE_LENGTH = 4.5
VEHICLE_WIDTH = 2.0
N_CIRCLES = 3

FOOTPRINT = EgoFootprint.multi_circle(VEHICLE_LENGTH, VEHICLE_WIDTH, N_CIRCLES)


def reconstruct_yaw(ego_x: np.ndarray, ego_y: np.ndarray) -> np.ndarray:
    """Heading from finite differences, held through standstill steps."""
    dx = np.gradient(ego_x)
    dy = np.gradient(ego_y)
    moving = np.hypot(dx, dy) > 1e-4
    yaw = np.arctan2(dy, dx)
    if not moving.any():
        return np.zeros_like(ego_x)
    # Forward-fill the last moving heading; back-fill the leading segment
    last = yaw[moving][0]
    out = np.empty_like(yaw)
    for i in range(len(yaw)):
        if moving[i]:
            last = yaw[i]
        out[i] = last
    return out


def rect_clearance(ped_local: np.ndarray) -> np.ndarray:
    """Distance from pedestrian centres (vehicle frame) to the rectangle."""
    dx = np.maximum(np.abs(ped_local[:, 0]) - VEHICLE_LENGTH / 2, 0.0)
    dy = np.maximum(np.abs(ped_local[:, 1]) - VEHICLE_WIDTH / 2, 0.0)
    return np.hypot(dx, dy)


def recheck(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    ego_x = data["ego_x"]
    ego_y = data["ego_y"]
    ped_positions = data["ped_positions"]
    stored_min = data["min_distances"] if "min_distances" in data else None
    if "ego_yaw" in data:
        yaw = data["ego_yaw"]
        yaw_source = "stored"
    else:
        yaw = reconstruct_yaw(ego_x, ego_y)
        yaw_source = "reconstructed"

    n = len(ego_x)
    legacy_min = np.full(n, np.inf)
    multi_min = np.full(n, np.inf)   # min circle-centre distance
    rect_min = np.full(n, np.inf)    # min rectangle-surface distance

    for i in range(n):
        peds = np.asarray(ped_positions[i], dtype=float)
        if peds.size == 0:
            continue
        peds = peds.reshape(-1, 2)
        center = np.array([ego_x[i], ego_y[i]])
        legacy_min[i] = np.min(np.linalg.norm(peds - center, axis=1))

        centers = FOOTPRINT.circle_centers(ego_x[i], ego_y[i], yaw[i])
        multi_min[i] = np.min(
            np.linalg.norm(peds[None, :, :] - centers[:, None, :], axis=2)
        )

        c, s = np.cos(yaw[i]), np.sin(yaw[i])
        rot = np.array([[c, s], [-s, c]])  # world -> vehicle frame
        ped_local = (peds - center) @ rot.T
        rect_min[i] = np.min(rect_clearance(ped_local))

    legacy_collisions = int(np.sum(legacy_min < EGO_RADIUS + PED_RADIUS))
    multi_collisions = int(np.sum(multi_min < FOOTPRINT.radius + PED_RADIUS))
    rect_collisions = int(np.sum(rect_min < PED_RADIUS))

    crosscheck = (
        float(np.nanmax(np.abs(legacy_min - stored_min)))
        if stored_min is not None and np.isfinite(stored_min).all()
        else float("nan")
    )

    return {
        "run": str(npz_path.parent.relative_to(REPO / "output")),
        "steps": n,
        "yaw_source": yaw_source,
        "stored_vs_recomputed_maxdiff": crosscheck,
        "legacy_min_dist": float(np.min(legacy_min)),
        "legacy_collision_steps": legacy_collisions,
        "multi_min_dist": float(np.min(multi_min)),
        "multi_clearance": float(np.min(multi_min)) - (FOOTPRINT.radius + PED_RADIUS),
        "multi_collision_steps": multi_collisions,
        "rect_min_surface_dist": float(np.min(rect_min)),
        "rect_clearance": float(np.min(rect_min)) - PED_RADIUS,
        "rect_collision_steps": rect_collisions,
    }


def main():
    npz_files = sorted((REPO / "output").glob("*/trajectory.npz"))
    if not npz_files:
        sys.exit("No trajectory.npz found under output/")

    rows = [recheck(p) for p in npz_files]
    df = pd.DataFrame(rows)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / "results.csv", index=False)

    lines = [
        "# Footprint recheck (A-4): saved trajectories under a vehicle-shaped footprint",
        "",
        f"- Trajectories: {len(df)} (representative saved runs; the 123/480-run",
        "  statistical campaigns store only scalar metrics, so they are covered by",
        "  step 3 = re-simulation, not by this recheck)",
        f"- Legacy: single circle r={EGO_RADIUS} m (threshold {EGO_RADIUS + PED_RADIUS} m, centre distance)",
        f"- Multi-circle: {N_CIRCLES} circles r={FOOTPRINT.radius:.2f} m at offsets {np.round(FOOTPRINT.offsets, 2).tolist()} m"
        f" (threshold {FOOTPRINT.radius + PED_RADIUS:.2f} m)",
        f"- Rectangle (exact): {VEHICLE_LENGTH} x {VEHICLE_WIDTH} m oriented box vs pedestrian radius {PED_RADIUS} m",
        "",
        "| run | steps | yaw | xcheck max|Δ| | legacy MinDist | legacy col | multi clearance | multi col | rect clearance | rect col |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['run']} | {r['steps']} | {r['yaw_source']} | "
            f"{r['stored_vs_recomputed_maxdiff']:.2e} | "
            f"{r['legacy_min_dist']:.3f} | {r['legacy_collision_steps']} | "
            f"{r['multi_clearance']:+.3f} | {r['multi_collision_steps']} | "
            f"{r['rect_clearance']:+.3f} | {r['rect_collision_steps']} |"
        )

    n_multi = int((df["multi_collision_steps"] > 0).sum())
    n_rect = int((df["rect_collision_steps"] > 0).sum())
    lines += [
        "",
        "## Verdict",
        "",
        f"- Runs with multi-circle collisions: **{n_multi} / {len(df)}**",
        f"- Runs with exact-rectangle collisions: **{n_rect} / {len(df)}**",
        f"- Tightest multi-circle clearance: {df['multi_clearance'].min():+.3f} m"
        f" ({df.loc[df['multi_clearance'].idxmin(), 'run']})",
        f"- Tightest exact-rectangle clearance: {df['rect_clearance'].min():+.3f} m"
        f" ({df.loc[df['rect_clearance'].idxmin(), 'run']})",
    ]

    (OUTDIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUTDIR / 'results.csv'} and {OUTDIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
