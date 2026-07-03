"""RQ3 real-data-grounded closed loop: Encounter -> closed-loop building blocks.

Wires a VCI encounter (recorded ego + pedestrian trajectories on a 0.4 s grid,
:class:`~src.datasets.vci_encounter.Encounter`) into the planner-driven closed
loop (:class:`~src.simulation.integrated_simulator.IntegratedSimulator`):

* the RECORDED EGO trajectory becomes the reference path (deduped spline
  waypoints) and its median speed the target speed -- the planner replaces the
  recorded driver, everything else stays anchored to the recording;
* the PEDESTRIANS are swapped along the reactivity axis: ``replay`` (recorded
  trajectories, interpolated to the simulation dt, non-reactive) versus SFM
  arms (calibrated / hand-tuned / no-repulsion parameters, reacting to the
  planner ego through the explicit ego-repulsion force);
* the OBSERVER is seeded with a constant-velocity backcast of frame 0, so at
  t=0 every arm sees the identical observation history and the pedestrian
  source has not been advanced (the encounter geometry is preserved exactly).

This module holds pure construction/metric helpers only; the campaign loop,
arm table and caching live in ``examples/run_rq3_realloop.py``.

Geometry conventions shared with the calibration harness (RQ2): ego collision
radius 1.0 m, pedestrian/agent radius 0.30 m, cruise-speed regimes ``median``
(target = cap = recorded per-ped median, the calibration regime) and
``closedloop`` (target = frame-0 speed, cap = 1.3x, the S1-S3 deployment
regime).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import SimulationConfig, load_config
from ..core.data_structures import PedestrianState
from ..datasets.vci_encounter import Encounter
from .calibration_harness import (
    DEFAULT_AGENT_RADIUS,
    DEFAULT_EGO_RADIUS,
    _build_ped_sim,
    _floor,
    _resolve_goals,
)
from .integrated_simulator import IntegratedSimulator, PedestrianSimulator
from .replay_source import ReplayPedestrianSource

# ---------------------------------------------------------------------------
# Eligibility thresholds (RQ3 Phase 0). Quoted verbatim in the RQ3 REPORT.
# ---------------------------------------------------------------------------
MIN_PATH_LEN_M = 5.0      # goal-termination radius (2 m) + planner headroom
MIN_MEDIAN_SPEED = 0.3    # m/s; below this the target-speed anchor degenerates
MIN_STRAIGHTNESS = 0.6    # net displacement / arc length; folded-path guard
DEDUPE_MIN_DS = 0.5       # m; waypoint spacing floor for the reference spline

SPEED_REGIMES = ("median_cruise", "initial_13x")

DEFAULT_BASE_SCENARIO = "scenarios/scenario_01.yaml"


def dedupe_waypoints(ego_xy: np.ndarray,
                     min_ds: float = DEDUPE_MIN_DS) -> Tuple[np.ndarray, np.ndarray]:
    """Drop waypoints closer than ``min_ds`` to the last kept one.

    Near-duplicate points (recorded ego standing still) make the arc-length
    parameter of ``CubicSpline2D`` non-increasing and break the spline. The
    first point (= ego initial position) is always kept; the recorded end
    point is appended even when closer than ``min_ds`` so the spline
    terminates at the recorded goal. Raises if fewer than 2 distinct
    waypoints survive.
    """
    xy = np.asarray(ego_xy, dtype=float)
    keep = [0]
    for i in range(1, len(xy)):
        if np.hypot(*(xy[i] - xy[keep[-1]])) >= min_ds:
            keep.append(i)
    last = len(xy) - 1
    # Append the recorded end point unless it (near-)coincides with the last
    # kept knot: a knot pair separated by ~1e-9..1e-3 m would ill-condition
    # the spline end without adding geometry.
    if keep[-1] != last and np.hypot(*(xy[last] - xy[keep[-1]])) > 1e-3:
        keep.append(last)
    if len(keep) < 2:
        raise ValueError(
            f"reference path degenerates to {len(keep)} waypoint(s) after "
            f"dedupe (min_ds={min_ds}); recorded ego is essentially stationary")
    kept = xy[keep]
    return kept[:, 0].copy(), kept[:, 1].copy()


def _path_length(xy: np.ndarray) -> float:
    d = np.diff(np.asarray(xy, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def encounter_eligibility(enc: Encounter) -> Tuple[bool, str]:
    """(eligible, reason) census for one encounter; never raises.

    Ineligible encounters are skipped WITH a disclosed reason (censoring, not
    silent exclusion) -- the campaign writes the reason column verbatim.
    """
    ego_xy = np.asarray(enc.ego_xy, dtype=float)
    path_len = _path_length(ego_xy)
    net_disp = float(np.hypot(*(ego_xy[-1] - ego_xy[0])))
    straightness = net_disp / path_len if path_len > 1e-9 else 0.0
    try:
        xs, ys = dedupe_waypoints(ego_xy)
        path_len_dedup = _path_length(np.column_stack([xs, ys]))
        degenerate = False
    except ValueError:
        path_len_dedup = 0.0
        degenerate = True

    reasons = []
    if path_len_dedup < MIN_PATH_LEN_M:
        reasons.append(f"path_len<{MIN_PATH_LEN_M}")
    if float(np.nanmedian(enc.ego_vel)) < MIN_MEDIAN_SPEED:
        reasons.append(f"median_speed<{MIN_MEDIAN_SPEED}")
    if straightness <= MIN_STRAIGHTNESS:
        reasons.append(f"straightness<={MIN_STRAIGHTNESS}")
    if degenerate:
        reasons.append("spline_degenerate")
    return (not reasons, ";".join(reasons))


# ---------------------------------------------------------------------------
# Pedestrian sources
# ---------------------------------------------------------------------------

def _interp_to_sim_grid(enc: Encounter, sim_dt: float) -> Tuple[np.ndarray,
                                                                np.ndarray,
                                                                np.ndarray]:
    """Linearly interpolate recorded ped positions AND velocities to sim_dt.

    Velocities are interpolated from the recorded ``ped_vel`` (not re-derived
    from interpolated positions, which would be piecewise-constant). Returns
    (times_rel, ped_xy_q, ped_vel_q) with times_rel[0] = 0.
    """
    t_rel = np.asarray(enc.times, dtype=float) - float(enc.times[0])
    total = t_rel[-1]
    n_q = int(round(total / sim_dt)) + 1
    t_q = np.arange(n_q) * sim_dt
    T, N, _ = enc.ped_xy.shape
    xy_q = np.empty((n_q, N, 2))
    vel_q = np.empty((n_q, N, 2))
    for i in range(N):
        for a in range(2):
            xy_q[:, i, a] = np.interp(t_q, t_rel, enc.ped_xy[:, i, a])
            vel_q[:, i, a] = np.interp(t_q, t_rel, enc.ped_vel[:, i, a])
    return t_q, xy_q, vel_q


def build_replay_source(enc: Encounter,
                        sim_dt: float = 0.1) -> ReplayPedestrianSource:
    """Recorded pedestrians as a non-reactive replay source at the sim dt."""
    _, xy_q, vel_q = _interp_to_sim_grid(enc, sim_dt)
    return ReplayPedestrianSource(
        xy_q, dt=sim_dt, velocities=vel_q,
        goals=_resolve_goals(enc), ids=np.asarray(enc.ped_ids),
    )


def build_sfm_source(enc: Encounter, sigma: float, v0: float,
                     speed_regime: str = "median_cruise",
                     sim_dt: float = 0.1) -> PedestrianSimulator:
    """SFM pedestrians at (sigma, v0), starting from recorded frame 0.

    ``median_cruise`` reuses the calibration harness regime verbatim
    (target = cap = recorded per-ped median speed) -- the regime the
    calibrated parameters were fitted in. ``initial_13x`` reproduces the
    closed-loop deployment regime (target = frame-0 speed, cap = 1.3x),
    with the harness's speed floor so a ped recorded standing still is not
    frozen forever (disclosed deviation from the raw closed loop).
    """
    if speed_regime not in SPEED_REGIMES:
        raise ValueError(
            f"unknown speed_regime {speed_regime!r}; expected {SPEED_REGIMES}")
    if speed_regime == "median_cruise":
        return _build_ped_sim(enc, sigma, v0, DEFAULT_EGO_RADIUS,
                              DEFAULT_AGENT_RADIUS, dt=sim_dt)
    return _build_ped_sim(
        enc, sigma, v0, DEFAULT_EGO_RADIUS, DEFAULT_AGENT_RADIUS, dt=sim_dt,
        cruise_fn=lambda e: _floor(
            np.hypot(e.ped_vel[0, :, 0], e.ped_vel[0, :, 1])),
        cap_policy="closedloop",
    )


def observation_seed_backcast(enc: Encounter, obs_len: int = 8,
                              sgan_dt: float = 0.4) -> List[PedestrianState]:
    """Constant-velocity backcast observation history ending at frame 0.

    Positions ``p0 + v0 * t`` for t = -(obs_len-1)*sgan_dt .. 0.0 (the last
    state IS recorded frame 0 at timestamp 0.0). Every arm receives this same
    history, so t=0 predictions are identical across arms and differences
    emerge purely from the arm dynamics. Real pre-window history is NOT used
    (ego is NaN outside the encounter window and ped presence is not
    guaranteed there); provenance records ``warmup_source=backcast``.
    """
    p0 = np.asarray(enc.ped_xy[0], dtype=float)
    v0 = np.asarray(enc.ped_vel[0], dtype=float)
    goals = _resolve_goals(enc)
    ids = np.asarray(enc.ped_ids)
    states = []
    for k in range(obs_len):
        t = -(obs_len - 1 - k) * sgan_dt
        states.append(PedestrianState(
            positions=p0 + v0 * t, velocities=v0.copy(),
            goals=goals.copy(), ids=ids.copy(), timestamp=t,
        ))
    return states


# ---------------------------------------------------------------------------
# Config + simulator assembly
# ---------------------------------------------------------------------------

def encounter_config(enc: Encounter,
                     base_scenario: str = DEFAULT_BASE_SCENARIO) -> SimulationConfig:
    """SimulationConfig anchored to the recorded encounter.

    Loads ``base_scenario`` (scenario_01: the verified fail-safe / envelope /
    planner constants, kept as the FIXED instrument configuration and
    disclosed in the RQ3 REPORT) and overrides everything encounter-specific.
    ``ped_radius``/``obstacle_radius`` are set to the calibration-consistent
    0.30 m (scenario_01 uses 0.2; RQ2 calibrated with agent_radius 0.30).
    """
    config = load_config(base_scenario)
    xs, ys = dedupe_waypoints(enc.ego_xy)
    config.reference_waypoints_x = xs.tolist()
    config.reference_waypoints_y = ys.tolist()
    config.ego_initial_state = [
        float(enc.ego_xy[0, 0]), float(enc.ego_xy[0, 1]),
        float(enc.ego_psi[0]), float(enc.ego_vel[0]), 0.0,
    ]
    config.ego_target_speed = float(np.nanmedian(enc.ego_vel))
    # total_time = recorded window length, padded by half a sim step: the
    # runner computes n_steps = int(total_time / dt), and the grid-derived
    # window length (times[-1] - times[0]) can sit 1 ulp BELOW the exact
    # multiple of dt (e.g. 10.399999999999999 / 0.1 -> 103), silently dropping
    # the final recorded frame in 5/26 encounters. The +dt/2 pad makes the
    # truncation land exactly on the intended step count either way.
    n_steps = int(round(float(enc.times[-1] - enc.times[0]) / config.dt))
    config.total_time = (n_steps + 0.5) * config.dt
    # Pedestrians come from the injected source, never from the config.
    config.ped_initial_states = []
    config.ped_groups = []
    config.static_obstacles = []
    config.map_config = {}
    config.ped_radius = DEFAULT_AGENT_RADIUS
    config.obstacle_radius = DEFAULT_AGENT_RADIUS
    config.visualization_enabled = False
    return config


def build_realloop_simulator(
    enc: Encounter,
    ped_kind: str,
    pred_method: str,
    plan_mode: str,
    sigma: Optional[float] = None,
    v0: Optional[float] = None,
    speed_regime: str = "median_cruise",
    sgan_model_path: Optional[str] = None,
    base_scenario: str = DEFAULT_BASE_SCENARIO,
) -> Tuple[IntegratedSimulator, Dict]:
    """Assemble the closed loop for one (encounter, arm) cell.

    Args:
        ped_kind: 'replay' or 'sfm'.
        pred_method: 'cv' | 'lstm' | 'sgan'.
        plan_mode: 'single' (true single draw, review F4), 'robust'
            (chance-constrained over the full distribution, eps=0.0) or
            'medoid' (the historical predict_single_best default, reference).
        sigma, v0: SFM ego-repulsion parameters (required for ped_kind='sfm').
        sgan_model_path: resolved checkpoint path (None for cv).

    Returns:
        (simulator, provenance) -- provenance records exactly what was built.
    """
    config = encounter_config(enc, base_scenario)
    config.prediction_method = pred_method
    if sgan_model_path is not None:
        config.sgan_model_path = sgan_model_path

    if plan_mode == "single":
        config.distribution_aware_planning = False
        config.single_select = "draw"
    elif plan_mode == "robust":
        config.distribution_aware_planning = True
        config.chance_epsilon = 0.0
        config.single_select = "medoid"  # unused by the robust path
    elif plan_mode == "medoid":
        config.distribution_aware_planning = False
        config.single_select = "medoid"
    else:
        raise ValueError(f"unknown plan_mode {plan_mode!r}")

    if ped_kind == "replay":
        source = build_replay_source(enc, sim_dt=config.dt)
    elif ped_kind == "sfm":
        if sigma is None or v0 is None:
            raise ValueError("ped_kind='sfm' needs sigma and v0")
        source = build_sfm_source(enc, sigma, v0, speed_regime,
                                  sim_dt=config.dt)
    else:
        raise ValueError(f"unknown ped_kind {ped_kind!r}")

    seed_states = observation_seed_backcast(
        enc, obs_len=config.obs_len, sgan_dt=0.4)
    sim = IntegratedSimulator(config, pedestrian_source=source,
                              observation_seed=seed_states)

    n_samples = int(config.num_samples)
    if plan_mode == "robust":
        single_mode = f"distribution_of_{n_samples}"
    elif plan_mode == "single":
        single_mode = f"draw1_of_{n_samples}" if n_samples > 1 else "single_draw"
    else:
        single_mode = f"medoid_of_{n_samples}" if n_samples > 1 else "single_draw"

    provenance = {
        "ped_kind": ped_kind,
        "sigma": float(sigma) if sigma is not None else float("nan"),
        "v0": float(v0) if v0 is not None else float("nan"),
        "speed_regime": speed_regime if ped_kind == "sfm" else "replay",
        "warmup_source": "backcast",
        "pred": pred_method,
        "plan": plan_mode,
        "single_mode": single_mode,
        "chance_epsilon": float(config.chance_epsilon),
        "ego_target_speed": float(config.ego_target_speed),
        "total_time": float(config.total_time),
        "ped_radius": float(config.ped_radius),
        "ego_radius": float(getattr(config, "ego_radius", 1.0)),
        "base_scenario": base_scenario,
    }
    return sim, provenance


# ---------------------------------------------------------------------------
# Post-run metrics
# ---------------------------------------------------------------------------

def recorded_ego_deviation(history, enc: Encounter,
                           dt: float = 0.1) -> Dict[str, float]:
    """Time-aligned deviation of the planner ego from the recorded ego.

    Interpolates the recorded ego trajectory onto the simulation timestamps
    (both clocks start at the encounter window start) and reports the mean
    and max positional deviation plus a progress fraction: the planner ego's
    displacement projected on the recorded start->end axis, divided by the
    recorded net displacement (1.0 = reached the recorded end point; can
    exceed 1 slightly on overshoot). Pure function of (history, enc).

    ``SimulationResult.time`` is recorded BEFORE the clock increment, so the
    stored ego state physically belongs to ``r.time + dt``; the comparison
    uses that shifted time.
    """
    if not history:
        return {"ego_dev_mean_m": float("nan"), "ego_dev_max_m": float("nan"),
                "progress": float("nan")}
    t_rel = np.asarray(enc.times, dtype=float) - float(enc.times[0])
    times = np.array([r.time for r in history], dtype=float) + float(dt)
    ego = np.array([[r.ego_state.x, r.ego_state.y] for r in history])
    rec_x = np.interp(times, t_rel, enc.ego_xy[:, 0])
    rec_y = np.interp(times, t_rel, enc.ego_xy[:, 1])
    dev = np.hypot(ego[:, 0] - rec_x, ego[:, 1] - rec_y)

    start = np.asarray(enc.ego_xy[0], dtype=float)
    end = np.asarray(enc.ego_xy[-1], dtype=float)
    span = end - start
    net = float(np.hypot(*span))
    if net > 1e-9:
        u = span / net
        progress = float(np.dot(ego[-1] - start, u) / net)
    else:
        progress = float("nan")
    return {
        "ego_dev_mean_m": float(dev.mean()),
        "ego_dev_max_m": float(dev.max()),
        "progress": progress,
    }
