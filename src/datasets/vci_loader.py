"""VCI DUT/CITR vehicle-crowd interaction loader for ego-repulsion calibration (RQ2).

The VCI datasets (fetched manually, see scripts/download_vci_data.sh) ship
per-sequence CSVs with a header row, comma-separated, pedestrians and vehicles
in separate files:

  pedestrians: id, frame, label, x_est, y_est, xv_est, yv_est
  vehicles:    id, frame, label, x_est, y_est, psi_est, vel_est

Positions are already in metres (filtered trajectories are pre-converted from
the 1920x1080 pixel frame via the per-folder ``.txt`` ratio file). DUT is
~23.98 fps; CITR's rate is unstated, so ``fps`` is a required-ish parameter.

Vehicle trajectories are what make calibration of the SFM ego repulsion
(sigma, v0) possible -- this is why calibration is DUT/CITR-only (ETH/UCY has no
vehicle). Tracks are resampled onto a 0.4 s grid (the SGAN cadence) and exposed
as fixed-population windows so they flow through ReplayPedestrianSource, the
observer, and the fidelity metrics unchanged.

The column/layout assumptions follow the published README; verify against the
real files on first download (the spike validates structure, not real values).
Raw data is gitignored and not redistributed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

DUT_FPS = 23.98  # DUT drone recording; CITR rate is unstated -> pass explicitly


@dataclass
class AgentTracks:
    """Per-agent tracks resampled onto a common time grid.

    positions[t, a] is NaN wherever agent ``a`` is absent at grid time ``t``
    (outside its recorded span); ``extra`` holds per-agent scalar channels such
    as vehicle speed/heading on the same [T, A] grid.
    """

    times: np.ndarray  # [T] grid times [s]
    ids: np.ndarray  # [A] sorted agent ids
    positions: np.ndarray  # [T, A, 2] metres (NaN where absent)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> [T, A]


def _read_agent_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {"id", "frame", "x_est", "y_est"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)} (got {list(df.columns)})")
    return df


def _resample_agents(
    df: pd.DataFrame,
    fps: float,
    target_dt: float,
    extra_cols: Dict[str, str],
    angular_cols: Tuple[str, ...] = (),
) -> AgentTracks:
    """Linearly resample every agent onto a shared target_dt grid.

    Channels listed in ``angular_cols`` (e.g. vehicle heading) are unwrapped
    before interpolation and wrapped back to (-pi, pi], so a +/-pi crossing is
    not interpolated through 0. Agent ids are coerced to int (matching the
    ETH/UCY loader). Duplicate (id, frame) rows are collapsed (keep first) so
    np.interp sees strictly increasing sample times.
    """
    if len(df) == 0:
        return AgentTracks(
            times=np.empty(0),
            ids=np.empty(0, dtype=int),
            positions=np.empty((0, 0, 2)),
            extra={name: np.empty((0, 0)) for name in extra_cols},
        )

    # One groupby (no per-id boolean scans, no get_group FutureWarning). Default
    # sort=True yields ascending keys, so enumeration index aligns with `ids`.
    groups = list(df.groupby(df["id"].astype(int), sort=True))
    ids = np.array([int(key) for key, _ in groups], dtype=int)

    t_all = df["frame"].to_numpy(dtype=float) / fps
    t_min, t_max = float(t_all.min()), float(t_all.max())
    # Build the grid from an explicit count (robust to float endpoint drift,
    # unlike np.arange(stop=t_max + eps)).
    n_t = int(np.floor((t_max - t_min) / target_dt + 1e-9)) + 1
    grid = t_min + target_dt * np.arange(n_t)
    n_a = len(ids)
    positions = np.full((n_t, n_a, 2), np.nan)
    extra = {name: np.full((n_t, n_a), np.nan) for name in extra_cols}

    for a, (_key, sub) in enumerate(groups):
        sub = sub.sort_values("frame").drop_duplicates(subset="frame", keep="first")
        t = sub["frame"].to_numpy(dtype=float) / fps
        if len(t) == 0:
            continue
        # Only fill grid points within this agent's recorded span; outside stays
        # NaN (no extrapolation). Small tolerance so a float-division endpoint
        # that mathematically equals t[0]/t[-1] is not excluded.
        mask = (grid >= t[0] - 1e-9) & (grid <= t[-1] + 1e-9)
        positions[mask, a, 0] = np.interp(grid[mask], t, sub["x_est"].to_numpy())
        positions[mask, a, 1] = np.interp(grid[mask], t, sub["y_est"].to_numpy())
        for name, col in extra_cols.items():
            vals = sub[col].to_numpy(dtype=float)
            if name in angular_cols:
                interp = np.interp(grid[mask], t, np.unwrap(vals))
                interp = (interp + np.pi) % (2 * np.pi) - np.pi  # wrap to (-pi, pi]
                extra[name][mask, a] = interp
            else:
                extra[name][mask, a] = np.interp(grid[mask], t, vals)

    return AgentTracks(times=grid, ids=ids, positions=positions, extra=extra)


def load_vci_pedestrians(
    path: Union[str, Path], fps: float = DUT_FPS, target_dt: float = 0.4
) -> AgentTracks:
    """Load and resample the pedestrians CSV of one VCI sequence."""
    return _resample_agents(_read_agent_csv(path), fps, target_dt, extra_cols={})


def load_vci_vehicles(
    path: Union[str, Path], fps: float = DUT_FPS, target_dt: float = 0.4
) -> AgentTracks:
    """Load and resample the vehicles CSV (carries 'psi' and 'vel' channels)."""
    df = _read_agent_csv(path)
    extra = {}
    if "psi_est" in df.columns:
        extra["psi"] = "psi_est"
    if "vel_est" in df.columns:
        extra["vel"] = "vel_est"
    # psi is a heading angle -> unwrap before interpolation.
    return _resample_agents(df, fps, target_dt, extra_cols=extra, angular_cols=("psi",))


def extract_fixed_windows(
    tracks: AgentTracks, seq_len: int, stride: int = 1, min_agents: int = 1
) -> List[np.ndarray]:
    """Return ``[seq_len, N, 2]`` windows of agents present (non-NaN) throughout.

    Population N is fixed within each window (agents with any NaN in the window
    are dropped) but varies between windows, matching the ETH/UCY loader.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    windows: List[np.ndarray] = []
    n_t = len(tracks.times)
    for start in range(0, n_t - seq_len + 1, stride):
        block = tracks.positions[start : start + seq_len]  # [seq_len, A, 2]
        present = ~np.any(np.isnan(block), axis=(0, 2))  # [A]
        if int(present.sum()) < min_agents:
            continue
        windows.append(block[:, present, :])
    return windows


def vehicle_speed_samples(tracks: AgentTracks) -> np.ndarray:
    """All finite vehicle speed samples [m/s] (the 'vel' channel).

    Used to quantify the speed-domain mismatch between calibration data
    (DUT/CITR vehicles ~1-3 m/s) and the AVEC ego (~5-6 m/s) -- a prerequisite
    for deciding the velocity-extrapolation strategy in RQ2.
    """
    vel = tracks.extra.get("vel")
    if vel is None:
        return np.array([])
    return vel[np.isfinite(vel)]
