"""Align VCI clips and extract vehicle-pedestrian encounters for RQ2 calibration.

RQ2 calibrates the SFM ego repulsion (sigma, v0) so that *simulated* pedestrians
reacting to a *recorded* real vehicle reproduce how *real* pedestrians avoided it.
That requires the recorded vehicle and the recorded pedestrians on ONE common
time grid (the calibration harness drives the ego from the vehicle and the SFM
peds react). :func:`vci_loader.load_vci_clips` resamples the pedestrian and
vehicle files INDEPENDENTLY, each on its own ``t_min`` origin, so this module
re-aligns the single vehicle onto the pedestrian grid and slices out the
contiguous spans where a fixed pedestrian population actually interacts with the
vehicle.

Two design choices specific to calibration (vs RQ1a open-loop replay):
* CITR vehicle clips carry exactly one vehicle (verified: vci_front/back/lat_*).
  :func:`align_clip_to_grid` asserts this and interpolates that vehicle's
  position / heading / speed onto the pedestrian grid; DUT (multi-vehicle,
  natural) is out of scope for the first spike and raises.
* An "encounter" is a contiguous span where the vehicle is present, a fixed set
  of pedestrians is present throughout, and the closest ego-pedestrian approach
  drops below ``min_sep_threshold`` (an actual interaction, not two agents
  passing far apart). Calibrating on non-interacting spans would carry no
  information about the ego repulsion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .vci_loader import AgentTracks, ClipTracks


@dataclass
class AlignedClip:
    """One clip with the (single) vehicle resampled onto the pedestrian grid.

    All arrays share the pedestrian time grid ``times`` [T]. Ego channels are
    NaN at grid times outside the vehicle's recorded span; pedestrian channels
    are NaN where that pedestrian is absent (inherited from the loader).
    """

    clip: str
    times: np.ndarray  # [T] grid times [s]
    ego_xy: np.ndarray  # [T, 2] metres (NaN outside vehicle span)
    ego_psi: np.ndarray  # [T] heading [rad]
    ego_vel: np.ndarray  # [T] speed [m/s]
    ped_xy: np.ndarray  # [T, A, 2] metres (NaN where absent)
    ped_vel: np.ndarray  # [T, A, 2] m/s (NaN where absent)
    ped_ids: np.ndarray  # [A]
    dt: float


@dataclass
class Encounter:
    """A fixed-population interaction window (the unit of calibration).

    ``ped_xy``/``ped_vel`` are [T, N, 2] with N pedestrians all present
    throughout (no NaN). ``ego_*`` are the recorded vehicle over the same span.
    """

    clip: str
    times: np.ndarray  # [T]
    ego_xy: np.ndarray  # [T, 2]
    ego_psi: np.ndarray  # [T]
    ego_vel: np.ndarray  # [T]
    ped_xy: np.ndarray  # [T, N, 2]
    ped_vel: np.ndarray  # [T, N, 2]
    ped_ids: np.ndarray  # [N]
    dt: float
    min_separation: float  # closest ego-ped approach in the span [m]
    # Optional per-ped SFM goal [N, 2]; None lets the harness derive it from the
    # recorded trajectory. Stored here so it is a fixed boundary condition (the
    # same across all (sigma, v0) evaluations) and so synthetic recovery tests can
    # pin it explicitly.
    goals: Optional[np.ndarray] = None


def _interp_channel(
    src_t: np.ndarray, src_v: np.ndarray, dst_t: np.ndarray, angular: bool = False
) -> np.ndarray:
    """Linearly interpolate finite ``src_v`` onto ``dst_t``; NaN outside support.

    With ``angular=True`` the channel is a heading: it is unwrapped before
    interpolation and the result wrapped back to (-pi, pi], so a +/-pi crossing is
    not interpolated through 0.
    """
    finite = np.isfinite(src_t) & np.isfinite(src_v)
    out = np.full(dst_t.shape, np.nan)
    if int(finite.sum()) < 2:
        return out
    st = src_t[finite]
    sv = src_v[finite]
    order = np.argsort(st)
    st, sv = st[order], sv[order]
    if angular:
        sv = np.unwrap(sv)
    mask = (dst_t >= st[0] - 1e-9) & (dst_t <= st[-1] + 1e-9)
    interp = np.interp(dst_t[mask], st, sv)
    if angular:
        interp = (interp + np.pi) % (2 * np.pi) - np.pi
    out[mask] = interp
    return out


def _ped_velocities(ped: AgentTracks, dt: float) -> np.ndarray:
    """[T, A, 2] pedestrian velocities: recorded vx/vy if present, else finite-diff.

    Finite-differencing falls back to a forward difference (last step duplicates
    the previous), matching ReplayPedestrianSource. NaN where the pedestrian is
    absent is preserved either way.
    """
    if "vx" in ped.extra and "vy" in ped.extra:
        return np.stack([ped.extra["vx"], ped.extra["vy"]], axis=2)  # [T, A, 2]
    pos = ped.positions  # [T, A, 2]
    vel = np.full_like(pos, np.nan)
    if pos.shape[0] >= 2:
        vel[:-1] = (pos[1:] - pos[:-1]) / dt
        vel[-1] = vel[-2]
    return vel


def align_clip_to_grid(clip: ClipTracks) -> AlignedClip:
    """Resample the clip's single vehicle onto the pedestrian time grid.

    The pedestrian grid is the master (calibration evaluates the pedestrians, so
    they keep their native sampling); the lone vehicle's position/heading/speed
    are interpolated onto it. Heading is unwrap-interpolated. Speed uses the
    recorded ``vel`` channel when present, else the magnitude of the
    finite-differenced position. Raises if the clip lacks pedestrians/vehicle or
    carries more than one vehicle (DUT multi-vehicle is out of scope here).
    """
    if clip.ped is None or clip.veh is None:
        raise ValueError(f"clip {clip.clip!r} needs both pedestrian and vehicle tracks")
    ped: AgentTracks = clip.ped
    veh: AgentTracks = clip.veh
    if veh.positions.shape[1] != 1:
        raise ValueError(
            f"clip {clip.clip!r} has {veh.positions.shape[1]} vehicles; "
            "calibration assumes a single ego vehicle (CITR vci_* clips)"
        )

    times = ped.times
    dt = float(times[1] - times[0]) if len(times) >= 2 else 0.4

    veh_t = veh.times
    veh_xy = veh.positions[:, 0, :]  # [Tv, 2]
    ego_x = _interp_channel(veh_t, veh_xy[:, 0], times)
    ego_y = _interp_channel(veh_t, veh_xy[:, 1], times)
    ego_xy = np.stack([ego_x, ego_y], axis=1)

    if "psi" in veh.extra:
        ego_psi = _interp_channel(veh_t, veh.extra["psi"][:, 0], times, angular=True)
    else:
        # Heading from the velocity direction of the interpolated path.
        d = np.gradient(ego_xy, dt, axis=0)
        ego_psi = np.arctan2(d[:, 1], d[:, 0])

    if "vel" in veh.extra:
        ego_vel = _interp_channel(veh_t, veh.extra["vel"][:, 0], times)
    else:
        d = np.gradient(ego_xy, dt, axis=0)
        ego_vel = np.linalg.norm(d, axis=1)

    return AlignedClip(
        clip=clip.clip,
        times=times,
        ego_xy=ego_xy,
        ego_psi=ego_psi,
        ego_vel=ego_vel,
        ped_xy=ped.positions,
        ped_vel=_ped_velocities(ped, dt),
        ped_ids=ped.ids,
        dt=dt,
    )


def _contiguous_runs(mask: np.ndarray) -> List[slice]:
    """Return slices for each maximal run of True in a 1-D boolean mask."""
    runs: List[slice] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append(slice(start, i))
            start = None
    if start is not None:
        runs.append(slice(start, len(mask)))
    return runs


def extract_encounters(
    aligned: AlignedClip,
    min_sep_threshold: float = 8.0,
    min_len: int = 5,
) -> List[Encounter]:
    """Slice the aligned clip into fixed-population vehicle-pedestrian encounters.

    For each maximal contiguous span where the vehicle is present, keep the
    pedestrians present throughout the whole span (fixed N, no NaN) and emit an
    :class:`Encounter` when at least one such pedestrian exists, the span is at
    least ``min_len`` frames, and the closest ego-pedestrian approach within it
    is below ``min_sep_threshold`` (i.e. an actual interaction). Spans with no
    qualifying pedestrian or no real approach are dropped.
    """
    # Require every ego channel (position AND heading AND speed) finite, not just
    # position: when a vehicle file lacks psi/vel the gradient fallback leaves NaN
    # at the span boundaries, and gating on position alone would keep frames whose
    # ego_psi/ego_vel are NaN (poisoning the CLI speed diagnostic and any future
    # velocity-dependent term).
    ego_present = (
        np.isfinite(aligned.ego_xy).all(axis=1)
        & np.isfinite(aligned.ego_psi)
        & np.isfinite(aligned.ego_vel)
    )  # [T]
    encounters: List[Encounter] = []
    for span in _contiguous_runs(ego_present):
        length = span.stop - span.start
        if length < min_len:
            continue
        ego_xy = aligned.ego_xy[span]  # [L, 2]
        ego_psi = aligned.ego_psi[span]
        ego_vel = aligned.ego_vel[span]
        ped_xy = aligned.ped_xy[span]  # [L, A, 2]
        ped_vel = aligned.ped_vel[span]  # [L, A, 2]

        # Present throughout AND with finite velocity throughout: a recorded
        # vx/vy sample can be NaN at a frame where the position is present, which
        # would otherwise leak NaN into enc.ped_vel (and thence np.gradient /
        # _cruise_speeds / _far_goals downstream).
        present = ~np.any(np.isnan(ped_xy), axis=(0, 2)) & ~np.any(
            np.isnan(ped_vel), axis=(0, 2)
        )  # [A] present throughout
        if int(present.sum()) == 0:
            continue
        ped_xy = ped_xy[:, present, :]  # [L, N, 2]
        ped_vel = ped_vel[:, present, :]
        ped_ids = aligned.ped_ids[present]

        # Closest ego-ped approach over the span (centre-to-centre).
        dists = np.linalg.norm(ped_xy - ego_xy[:, None, :], axis=2)  # [L, N]
        min_sep = float(np.min(dists))
        if min_sep > min_sep_threshold:
            continue

        encounters.append(
            Encounter(
                clip=aligned.clip,
                times=aligned.times[span],
                ego_xy=ego_xy,
                ego_psi=ego_psi,
                ego_vel=ego_vel,
                ped_xy=ped_xy,
                ped_vel=ped_vel,
                ped_ids=ped_ids,
                dt=aligned.dt,
                min_separation=min_sep,
            )
        )
    return encounters


def encounters_from_clips(
    clips: List[ClipTracks],
    min_sep_threshold: float = 8.0,
    min_len: int = 5,
) -> List[Encounter]:
    """Align and extract encounters from many clips (vehicle-bearing clips only).

    Clips without both ped and vehicle tracks, or with multiple vehicles, are
    skipped (not an error) so a whole scenario directory can be passed in.
    """
    out: List[Encounter] = []
    for clip in clips:
        # Skip only the known non-calibratable layouts -- a clip missing either
        # side, or one with multiple vehicles (DUT). Pre-checking these (rather
        # than catching ValueError from align_clip_to_grid) means any *other*
        # error surfaces instead of being silently swallowed as a "skip".
        if clip.ped is None or clip.veh is None:
            continue
        if clip.veh.positions.shape[1] != 1:
            continue
        out.extend(
            extract_encounters(align_clip_to_grid(clip), min_sep_threshold, min_len)
        )
    return out
