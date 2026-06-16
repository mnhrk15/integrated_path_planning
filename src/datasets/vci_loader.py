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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

DUT_FPS = 23.98  # DUT drone recording; CITR rate is unstated -> pass explicitly

# Filtered (metre-converted) trajectory file suffixes, one pair per video clip.
# The clip stem is the file name with the suffix stripped, e.g.
# "intersection_01_traj_ped_filtered.csv" -> stem "intersection_01". Matching on
# the full suffix (not just ".csv") structurally excludes co-located ratio .txt
# / plot .pdf files (CITR) and raw px CSVs (no "_filtered").
PED_SUFFIX = "_traj_ped_filtered.csv"
VEH_SUFFIX = "_traj_veh_filtered.csv"


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


def agent_speed_samples(tracks: AgentTracks, dt: Optional[float] = None) -> np.ndarray:
    """Per-step speeds [m/s] from finite adjacent grid positions (ped or veh).

    The AgentTracks analogue of the ETH/UCY ``walking_speed_stats``: it differs
    by masking on NaN (an agent's absent span) rather than on a frame-gap mode,
    since VCI tracks are already on a uniform ``target_dt`` grid. Only steps
    whose *both* endpoints are finite are counted, so NaN never propagates into
    a spurious speed. Gives a sanity distribution that should peak near
    ~1.3 m/s for pedestrians; also an independent cross-check of the recorded
    vehicle 'vel' channel.

    ``dt`` defaults to the grid's own step (``times[1]-times[0]``) so the speed
    is never silently divided by a target_dt that differs from how the tracks
    were resampled; pass it only to override.
    """
    pos = tracks.positions  # [T, A, 2]
    if pos.shape[0] < 2:
        return np.array([])
    if dt is None:
        dt = float(tracks.times[1] - tracks.times[0])  # the grid's actual target_dt
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=2) / dt  # [T-1, A]; NaN where absent
    return step[np.isfinite(step)]


@dataclass
class ClipTracks:
    """One VCI video clip: paired pedestrian/vehicle tracks plus provenance.

    ``ped`` and ``veh`` are independently resampled :class:`AgentTracks` on each
    file's own clip-local time grid; either is None when that file is absent
    (a ped-only or veh-only clip). Clips are kept separate rather than merged
    into one AgentTracks because ids restart at 1 per clip (merging would alias
    different people) and frames restart at 0 (merging would collide their time
    grids). ``scenario`` is the CITR subfolder (e.g. 'vci_front'); None for the
    flat DUT layout.
    """

    clip: str  # clip stem, e.g. "intersection_01"
    dataset: str  # "dut" | "citr"
    scenario: Optional[str]  # CITR subfolder name, None for DUT
    ped: Optional[AgentTracks]
    veh: Optional[AgentTracks]
    ped_path: Optional[Path]
    veh_path: Optional[Path]
    fps: float


def _discover_clip_files(
    root: Union[str, Path], dataset: str
) -> Dict[Tuple[Optional[str], str], Dict[str, Path]]:
    """Recursively find ``*_traj_{ped,veh}_filtered.csv`` under ``root``.

    Returns ``{(scenario, stem): {"ped": path, "veh": path}}``. ``rglob`` is
    depth-agnostic, so the flat DUT ``trajectories_filtered/`` and the nested
    CITR ``trajectories_filtered/<scenario>/`` layouts are both walked by one
    pass -- and an unknown number of extra wrapper directories from a zip
    extraction is absorbed (the user only points ``root`` at datasets/vci_*).
    For CITR the scenario is the file's parent directory name (which
    disambiguates a stem reused across scenarios); for the flat DUT it is None.
    """
    root = Path(root)
    found: Dict[Tuple[Optional[str], str], Dict[str, Path]] = {}
    for suffix, key in ((PED_SUFFIX, "ped"), (VEH_SUFFIX, "veh")):
        for path in root.rglob("*" + suffix):
            stem = path.name[: -len(suffix)]
            scenario = None if dataset == "dut" else path.parent.name
            side = found.setdefault((scenario, stem), {})
            if key in side and side[key] != path:
                # Two files collapse onto the same (scenario, stem) key -- e.g. a
                # DUT zip with the clip duplicated under different wrapper dirs
                # (scenario is None for DUT, so the directory no longer
                # disambiguates). Keeping the last would drop a clip
                # nondeterministically (rglob order is filesystem-dependent), so
                # fail loudly rather than silently lose data.
                raise ValueError(
                    f"duplicate {key} file for clip {(scenario, stem)!r}: "
                    f"{side[key]} and {path}"
                )
            if side and any(existing.parent != path.parent for existing in side.values()):
                # The same (scenario, stem) key must represent one directory-local
                # ped/veh pair. Otherwise a ped-only file under one wrapper dir and
                # a veh-only file under another wrapper dir would be silently
                # paired into a synthetic clip that never existed.
                raise ValueError(
                    f"mixed directories for clip {(scenario, stem)!r}: "
                    f"{sorted(str(existing.parent) for existing in side.values())} "
                    f"and {path.parent}"
                )
            side[key] = path
    return found


def load_vci_clips(
    root: Union[str, Path],
    dataset: str,
    fps: Optional[float] = None,
    target_dt: float = 0.4,
    require_both: bool = False,
    strict: bool = True,
) -> List[ClipTracks]:
    """Scan ``root`` for all VCI clips and load each via the single-file API.

    ``dataset`` is "dut" (flat layout, fps defaults to :data:`DUT_FPS`) or
    "citr" (nested per-scenario layout; fps is unstated upstream so it is
    required). Each clip reuses :func:`load_vci_pedestrians` /
    :func:`load_vci_vehicles`, so column validation, resampling and heading
    unwrapping are unchanged. With ``require_both`` a clip missing either file
    is skipped; otherwise the absent side is None. With ``strict=False`` a clip
    whose CSV fails to parse/validate keeps that side as None (its path is still
    retained) instead of aborting the whole scan -- so a diagnostic caller can
    still report the offending file rather than crash on the first bad one.
    Returns clips in a deterministic (scenario, stem) order.
    """
    if dataset not in ("dut", "citr"):
        raise ValueError(f"dataset must be 'dut' or 'citr', got {dataset!r}")
    if fps is None:
        if dataset == "dut":
            fps = DUT_FPS
        else:
            raise ValueError(
                "CITR fps is unstated upstream; pass fps explicitly "
                "(estimate it via examples/inspect_vci_data.py)"
            )

    def _load(loader, path):
        if path is None:
            return None
        try:
            return loader(path, fps=fps, target_dt=target_dt)
        # Only data problems (bad columns -> ValueError, unparseable/empty CSV,
        # unreadable file) are demoted under strict=False; programming/resource
        # errors (AttributeError, MemoryError, ...) propagate so a real bug or an
        # OOM is never silently reported as a merely "bad file".
        except (ValueError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            if strict:
                raise
            return None

    discovered = _discover_clip_files(root, dataset)
    clips: List[ClipTracks] = []
    for scenario, stem in sorted(discovered, key=lambda k: (k[0] or "", k[1])):
        paths = discovered[(scenario, stem)]
        ped_path = paths.get("ped")
        veh_path = paths.get("veh")
        if require_both and (ped_path is None or veh_path is None):
            continue
        clips.append(
            ClipTracks(
                clip=stem,
                dataset=dataset,
                scenario=scenario,
                ped=_load(load_vci_pedestrians, ped_path),
                veh=_load(load_vci_vehicles, veh_path),
                ped_path=ped_path,
                veh_path=veh_path,
                fps=fps,
            )
        )
    return clips
