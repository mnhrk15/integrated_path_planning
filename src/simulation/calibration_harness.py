"""Calibrate the SFM ego repulsion (sigma, v0) against recorded VCI encounters.

This is the inverse of :class:`ReplayPedestrianSource`: there the pedestrians are
replayed and the ego is ignored (open-loop prediction, RQ1a); here the EGO is
fixed to the recorded vehicle trajectory and the SFM pedestrians REACT to it, so
that fitting (sigma, v0) makes the simulated avoidance match the real avoidance
(RQ2). The harness reuses :class:`PedestrianSimulator` directly (not the full
``IntegratedSimulator``, which drags in the planner/predictor/warmup); that class
already applies an explicit position-only ego repulsion
``magnitude = v0 * exp(-clearance / sigma)`` and accepts the parameters via
``social_force_params``.

Two pysocialforce-internal corrections are mandatory (verified against the
installed library), or (sigma, v0) would be fit to compensate for them:

* **Desired-speed inflation.** ``DesiredForce`` drives each ped toward its goal
  at ``max_speeds = 1.3 * initial_speeds`` (``scene.py`` / ``forces.py``). If the
  recorded speed is used as the initial speed, simulated peds cruise ~30% too
  fast. :func:`_set_cruise_speed` pins ``max_speeds`` to the recorded cruise
  speed (by setting ``initial_speeds = cruise / multiplier``), reusing the exact
  mechanism of the existing ``v0_randomization`` block.
* **Stop-when-arrived.** A ped within 0.5 m of its goal is frozen
  (``scene.py``) and braked (``DesiredForce`` goal_threshold). Goals are
  therefore placed FAR along each ped's recorded heading
  (:func:`_far_goals`) so the driving force keeps pulling at cruise speed across
  the whole window — direction is anchored, not destination.

Objective design (see the thesis plan / RQ2): the FITTER is a short-rollout
position error (:func:`objective_rollout_ade`) — driving the SFM peds with the
recorded ego and minimising their displacement from the recorded pedestrian
trajectory. The spike's smoke test settled this empirically: a teacher-forced
one-step radial-acceleration residual (:func:`objective_one_step`) is degenerate
here (its minimum is always ``v0 = 0`` — the instantaneous exponential ego force
is far stronger than the small, noisy per-step real radial acceleration, so
matching instantaneous force drives the repulsion to zero), whereas the rollout
ADE has a clean interior minimum (the integrated trajectory deflection does
reproduce the real avoidance at a moderate ``v0``). ``objective_one_step`` is
kept as a DIAGNOSTIC — its ``v0 -> 0`` verdict is itself a reportable finding
(the paper's hand-tuned force is too impulsive at the instantaneous level). The
avoidance-onset / min-separation KS metrics are the VALIDATION report
(:func:`fidelity_report`), not the optimisation target (the SFM forces are small
enough that the onset acceleration threshold often does not trigger in
simulation, making a KS objective ill-defined).
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

from ..core.data_structures import EgoVehicleState
from ..core.metrics import (
    avoidance_onset_distance,
    compare_distributions_ks,
    min_separation_series,
)
from .integrated_simulator import PedestrianSimulator
from ..datasets.vci_encounter import Encounter

DEFAULT_EGO_RADIUS = 1.0  # AVEC ego footprint radius [m]; held fixed (confounds sigma)
# Pedestrian radius [m]. MUST match the AVEC/RQ1b scenarios' agent_radius (0.30):
# the ego repulsion magnitude is v0*exp(-clearance/sigma) with
# clearance = distance - (ego_radius + agent_radius), so calibrating at a
# different agent_radius than the one RQ1b runs at shifts the clearance origin
# and re-scales the fitted sigma when the calibrated (sigma, v0) is injected into
# RQ1b (review M6: the old 0.35 vs scenario 0.30 mismatch was a +4-7% force bias,
# the same order as the sigma fold spread). Held fixed (it confounds sigma).
DEFAULT_AGENT_RADIUS = 0.30
GOAL_DISTANCE = 50.0  # far-goal distance along recorded heading [m]


def _floor(cruise: np.ndarray) -> np.ndarray:
    """Floor non-finite / non-positive desired speeds to a small positive value.

    A zero or NaN desired speed would make pysocialforce's stop-when-arrived
    freeze the ped, so every cruise estimator routes its result through this floor.
    """
    return np.where(np.isfinite(cruise) & (cruise > 1e-3), cruise, 1e-3)


def _cruise_speeds(ped_vel: np.ndarray) -> np.ndarray:
    """Per-ped representative walking speed [N] from recorded velocities [T,N,2].

    Uses the median per-step speed (robust to the noisy first/last samples), with
    a small floor so a momentarily-stationary ped still gets a non-zero desired
    speed (a zero would make pysocialforce's stop-when-arrived freeze it).
    """
    speeds = np.linalg.norm(ped_vel, axis=2)  # [T, N]
    with warnings.catch_warnings():
        # An all-NaN column (a ped absent throughout) yields NaN here; that is
        # floored just below, so silence the "All-NaN slice" RuntimeWarning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cruise = np.nanmedian(speeds, axis=0)  # [N]
    return _floor(cruise)


# A cruise-speed estimator maps a whole Encounter to a per-ped desired speed [N].
# The default (_cruise_speeds on the recorded velocity) and the free-walking /
# upper-quantile alternatives below all share this shape so they are swappable
# via the harness's ``cruise_fn`` hook (the RQ2 cruise-bias diagnostic, D).
CruiseEstimator = Callable[["Encounter"], np.ndarray]


def cruise_freewalk(
    enc: Encounter, ego_distance_threshold: float = 8.0, quantile: float = 0.5
) -> np.ndarray:
    """Per-ped desired speed [N] from FREE-WALKING frames only (RQ2 cruise bias).

    The default estimator (:func:`_cruise_speeds`) takes the median over the whole
    window, but the recorded speed already dips while the ped slows to avoid the
    ego, biasing the desired speed DOWN -- which in turn lets the fitter explain
    the observed avoidance with a weaker (lower-v0) repulsion. This estimator
    instead pools only the frames where the ped is farther than
    ``ego_distance_threshold`` from the ego (not yet reacting) and takes the
    ``quantile`` speed there. A ped with no such free-walking frame (always close)
    falls back to the all-frame median, so the floor / stop-when-arrived guarantee
    of the baseline is preserved.
    """
    speeds = np.linalg.norm(enc.ped_vel, axis=2)  # [T, N]
    dist = np.linalg.norm(enc.ped_xy - enc.ego_xy[:, None, :], axis=2)  # [T, N]
    free = (dist > ego_distance_threshold) & np.isfinite(speeds)  # [T, N]
    N = speeds.shape[1]
    out = np.empty(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN ped column
        for j in range(N):
            sj = speeds[:, j]
            if free[:, j].any():
                out[j] = np.quantile(sj[free[:, j]], quantile)
            else:  # never free-walking -> baseline median fallback
                finite = np.isfinite(sj)
                out[j] = np.median(sj[finite]) if finite.any() else 1e-3
    return _floor(out)


def cruise_upper_quantile(enc: Encounter, quantile: float = 0.85) -> np.ndarray:
    """Per-ped desired speed [N] as an upper ``quantile`` over ALL frames.

    The cheapest correction for the avoidance-slowdown bias: instead of selecting
    free-walking frames, it just shifts the per-ped speed statistic to an upper
    quantile so the (rarer) full-speed strides dominate the (avoidance) dips.
    """
    speeds = np.linalg.norm(enc.ped_vel, axis=2)  # [T, N]
    N = speeds.shape[1]
    out = np.empty(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j in range(N):
            sj = speeds[:, j]
            finite = np.isfinite(sj)
            out[j] = np.quantile(sj[finite], quantile) if finite.any() else 1e-3
    return _floor(out)


def _far_goals(ped_xy: np.ndarray, ped_vel: np.ndarray, distance: float = GOAL_DISTANCE) -> np.ndarray:
    """Goal [N,2] placed ``distance`` m ahead along each ped's recorded heading.

    Heading is the net start->end displacement direction (falling back to the
    initial velocity, then +x, for a barely-moving ped). The net displacement is
    used because the alternative -- the initial velocity extrapolated straight --
    ignores the ped's natural path curvature over the window and produces a sim
    that diverges from any real (non-straight) walk, swamping the ego-repulsion
    signal. Anchoring the *direction* of a far goal (not the destination) keeps
    the driving force pulling at cruise speed across the whole window instead of
    braking near a reached goal.

    LIMITATION: the net displacement of the *recorded* (already-avoided) path
    partially contains the avoidance, so the goal is mildly contaminated by the
    behaviour the ego repulsion is meant to generate; for a crossing ped the
    transient avoidance bump mostly cancels in start->end, but a permanent lane
    shift would leak in. The true intended goal is unobserved (a known RQ2
    limitation). The goal is held FIXED across (sigma, v0) so it never co-adapts.
    """
    start = ped_xy[0]  # [N, 2]
    net = ped_xy[-1] - ped_xy[0]  # [N, 2]
    heading = net.copy()
    norms = np.linalg.norm(heading, axis=1)
    weak = norms < 1e-3
    if np.any(weak):
        v0 = ped_vel[0]
        vnorm = np.linalg.norm(v0, axis=1)
        for j in np.where(weak)[0]:
            heading[j] = v0[j] if vnorm[j] > 1e-3 else np.array([1.0, 0.0])
    heading = heading / np.linalg.norm(heading, axis=1, keepdims=True)
    return start + heading * distance


def _resolve_goals(enc: Encounter) -> np.ndarray:
    """Encounter's explicit goals if set, else derived from the recorded path.

    Either way the goals are a fixed boundary condition (independent of the
    (sigma, v0) being evaluated), so a parameter sweep never moves the goals.
    """
    if enc.goals is not None:
        return np.asarray(enc.goals, dtype=float)
    return _far_goals(enc.ped_xy, enc.ped_vel)


def _set_cruise_speed(ped_sim: PedestrianSimulator, cruise: np.ndarray) -> None:
    """Pin pysocialforce ``max_speeds`` to ``cruise`` (cancels the 1.3x inflation).

    ``state.setter`` recomputes ``max_speeds = multiplier * initial_speeds`` on
    every state assignment but only sets ``initial_speeds`` once, so overriding
    ``initial_speeds = cruise / multiplier`` makes the recomputed ``max_speeds``
    equal ``cruise`` and persist across steps (same trick as the v0_randomization
    path in integrated_simulator).
    """
    peds = ped_sim.sim.peds
    multiplier = float(peds.max_speed_multiplier)
    peds.initial_speeds = cruise / multiplier
    peds.max_speeds = cruise.copy()


def _build_ped_sim(
    enc: Encounter,
    sigma: float,
    v0: float,
    ego_radius: float,
    agent_radius: float,
    dt: float,
    cruise_fn: Optional[CruiseEstimator] = None,
) -> PedestrianSimulator:
    """Construct a PedestrianSimulator for one encounter at given (sigma, v0).

    ``cruise_fn`` (default :func:`_cruise_speeds` on the recorded velocity)
    estimates each ped's desired walking speed; pass an alternative (e.g.
    :func:`cruise_freewalk`) for the RQ2 cruise-bias diagnostic. Default None
    reproduces the original behaviour bit-for-bit.
    """
    pos0 = enc.ped_xy[0]  # [N, 2]
    vel0 = enc.ped_vel[0]  # [N, 2]
    goals = _resolve_goals(enc)  # [N, 2]
    initial_states = np.hstack([pos0, vel0, goals])  # [N, 6]
    ped_sim = PedestrianSimulator(
        initial_states=initial_states,
        dt=dt,
        ego_radius=ego_radius,
        social_force_params={
            "ego_repulsion.sigma": sigma,
            "ego_repulsion.v0": v0,
            "agent_radius": agent_radius,
        },
        v0_randomization=False,
    )
    cruise = _cruise_speeds(enc.ped_vel) if cruise_fn is None else cruise_fn(enc)
    # Floor at the consumption point so the stop-when-arrived guarantee holds for
    # ANY cruise_fn (the built-ins self-floor, so this is a no-op for them).
    _set_cruise_speed(ped_sim, _floor(cruise))
    return ped_sim


def _interp_ego(enc: Encounter, i: int, frac: float) -> EgoVehicleState:
    """Recorded ego state interpolated a fraction ``frac`` into frame interval i->i+1.

    Position and speed interpolate linearly; heading interpolates on the unwrapped
    pair. (The current ego repulsion uses only position; yaw/v are carried so a
    later velocity-dependent term can use them without changing the harness.)
    """
    xy = enc.ego_xy[i] * (1 - frac) + enc.ego_xy[i + 1] * frac
    v = enc.ego_vel[i] * (1 - frac) + enc.ego_vel[i + 1] * frac
    p0, p1 = enc.ego_psi[i], enc.ego_psi[i + 1]
    p1u = p0 + ((p1 - p0 + np.pi) % (2 * np.pi) - np.pi)  # nearest unwrap of p1 to p0
    psi = p0 * (1 - frac) + p1u * frac
    psi = (psi + np.pi) % (2 * np.pi) - np.pi  # re-wrap to (-pi, pi]
    return EgoVehicleState(x=float(xy[0]), y=float(xy[1]), yaw=float(psi), v=float(v), a=0.0)


def simulate_encounter(
    enc: Encounter,
    sigma: float,
    v0: float,
    ego_radius: float = DEFAULT_EGO_RADIUS,
    agent_radius: float = DEFAULT_AGENT_RADIUS,
    dt: float = 0.1,
    cruise_fn: Optional[CruiseEstimator] = None,
) -> np.ndarray:
    """Roll out SFM pedestrians reacting to the recorded ego; return sim ped xy.

    The SFM integrates at a substep close to ``dt`` (default 0.1 s, matching AVEC
    usage), with the recorded ego linearly interpolated across the substeps of each
    recorded frame; pedestrian positions are recorded at the recorded-frame grid.
    The substep size is ``enc.dt / round(enc.dt / dt)`` rather than ``dt`` itself,
    so the simulated physical time advanced per recorded frame is EXACTLY ``enc.dt``
    even when ``dt`` does not evenly divide ``enc.dt`` (otherwise ``substeps * dt``
    would over/under-shoot the frame and silently mis-pace the rollout, comparing
    sim and recorded positions at different physical times). At the default
    ``enc.dt=0.4, dt=0.1`` this is exact (4 substeps of 0.1 s) — unchanged.

    The recorded ego force driving substep ``k`` is sampled at the substep MIDPOINT
    (``frac = (k + 0.5) / substeps``), not its end. Sampling the end
    (``(k + 1) / substeps``) placed the ego one substep AHEAD of the pedestrians
    integrating against it — a consistent ~half-frame phase lead that, because it
    is fixed across the (sigma, v0) sweep, shifted the fitted optimum rather than
    just adding noise. The midpoint is the second-order-accurate representative
    time for the piecewise-constant ego force the integrator applies over the
    substep, and is symmetric (no lead or lag) about the substep it drives.
    Returns ``[T, N, 2]`` aligned with ``enc.ped_xy`` (frame 0 = recorded start).
    """
    substeps = max(1, int(round(enc.dt / dt)))
    dt_sub = enc.dt / substeps
    ped_sim = _build_ped_sim(enc, sigma, v0, ego_radius, agent_radius, dt_sub, cruise_fn)
    T, N, _ = enc.ped_xy.shape
    sim_xy = np.empty((T, N, 2))
    sim_xy[0] = enc.ped_xy[0]
    for i in range(T - 1):
        for k in range(substeps):
            frac = (k + 0.5) / substeps
            ped_sim.step(_interp_ego(enc, i, frac), n=1)
        sim_xy[i + 1] = ped_sim.get_state().positions
    return sim_xy


def objective_rollout_ade(
    encounters: List[Encounter],
    sigma: float,
    v0: float,
    ego_radius: float = DEFAULT_EGO_RADIUS,
    agent_radius: float = DEFAULT_AGENT_RADIUS,
    dt: float = 0.1,
    interaction_distance: Optional[float] = None,
    cruise_fn: Optional[CruiseEstimator] = None,
) -> float:
    """Short-rollout displacement error vs the recorded pedestrians (the FITTER).

    Drives the SFM pedestrians with the recorded ego at (sigma, v0) via
    :func:`simulate_encounter` and returns the mean per-frame, per-pedestrian
    distance between simulated and recorded positions, pooled over encounters.
    This is dense (every frame, every ped) and has a smooth interior minimum in
    (sigma, v0), unlike the one-step force residual.

    Frame 0 is excluded: :func:`simulate_encounter` pins ``sim_xy[0]`` to the
    recorded start, so its error is exactly 0 for every (sigma, v0). Counting it
    only scaled the reported ADE down by a (parameter-independent) factor — it
    never moved the optimum, but it made the metric read artificially low,
    especially for short ``min_len`` encounters where one zero frame is a large
    fraction of the window.

    ``interaction_distance`` (optional) keeps only pedestrians that approach the
    ego within that distance at some recorded frame; the rest never feel the ego
    repulsion and only add a (sigma, v0)-independent baseline error that dilutes
    the signal. ``None`` keeps all pedestrians.
    """
    total = 0.0
    count = 0
    for enc in encounters:
        sim_xy = simulate_encounter(enc, sigma, v0, ego_radius, agent_radius, dt, cruise_fn)
        err = np.linalg.norm(sim_xy - enc.ped_xy, axis=2)  # [T, N]
        if interaction_distance is not None:
            dist = np.linalg.norm(enc.ped_xy - enc.ego_xy[:, None, :], axis=2)  # [T, N]
            keep = np.min(dist, axis=0) <= interaction_distance  # [N]
            err = err[:, keep]
        err = err[1:]  # drop the trivially-zero initial frame (sim_xy[0] == recorded start)
        total += float(err.sum())
        count += err.size
    if count == 0:
        return float("inf")
    return total / count


def _set_real_state(ped_sim: PedestrianSimulator, pos: np.ndarray, vel: np.ndarray, goals: np.ndarray) -> None:
    """Overwrite the simulator's pedestrian state to a recorded configuration.

    Teacher-forcing: the one-step objective evaluates the SFM force at the REAL
    positions/velocities each frame (no rollout), so goal drift and crowd
    rearrangement cannot accumulate. ``initial_speeds`` was overridden in
    :func:`_set_cruise_speed` and is preserved by the setter, so ``max_speeds``
    stays at the recorded cruise.
    """
    state = np.hstack([pos, vel, goals])  # [N, 6]; setter appends tau
    ped_sim.sim.peds.state = state


def objective_one_step(
    encounters: List[Encounter],
    sigma: float,
    v0: float,
    ego_radius: float = DEFAULT_EGO_RADIUS,
    agent_radius: float = DEFAULT_AGENT_RADIUS,
    clearance_min: float = 1e-3,
    max_distance: Optional[float] = None,
) -> float:
    """Teacher-forced one-step radial-acceleration residual (DIAGNOSTIC).

    Retained as a diagnostic, not the fitter (see module docstring): its minimum
    collapses to ``v0 = 0`` because the instantaneous exponential ego force
    dwarfs the small/noisy real per-step radial acceleration. Reported alongside
    the rollout-ADE calibration to show the paper's force is too impulsive
    instantaneously even where a moderate ``v0`` reproduces the trajectory.


    For every encounter, frame and pedestrian, compare the radial (ego->ped)
    component of the real acceleration (``np.gradient`` of recorded velocity, the
    same rule as ``avoidance_onset_distance``) against the radial component of the
    SFM force evaluated at the real configuration (``compute_forces`` + explicit
    ego repulsion; pysocialforce force is in acceleration units, so the two are
    comparable). Only samples with positive clearance contribute — at
    ``clearance <= 0`` the repulsion magnitude saturates at ``v0`` and carries no
    ``sigma`` information. Returns the mean squared residual (``inf`` if no sample
    qualifies, so a degenerate region is never chosen as a minimum).
    """
    total = 0.0
    count = 0
    radius_sum = ego_radius + agent_radius
    for enc in encounters:
        T, N, _ = enc.ped_xy.shape
        if T < 3 or N == 0:
            continue
        ped_sim = _build_ped_sim(enc, sigma, v0, ego_radius, agent_radius, dt=enc.dt)
        goals = _resolve_goals(enc)
        a_real = np.gradient(enc.ped_vel, enc.dt, axis=0)  # [T, N, 2]
        for t in range(T):
            _set_real_state(ped_sim, enc.ped_xy[t], enc.ped_vel[t], goals)
            ped_sim._overwrite_ego_state(
                EgoVehicleState(
                    x=float(enc.ego_xy[t, 0]), y=float(enc.ego_xy[t, 1]),
                    yaw=float(enc.ego_psi[t]), v=float(enc.ego_vel[t]), a=0.0,
                )
            )
            a_sim = ped_sim.sim.compute_forces() + ped_sim._compute_ego_repulsive_force()  # [N, 2]
            rel = enc.ped_xy[t] - enc.ego_xy[t]  # [N, 2]
            dist = np.linalg.norm(rel, axis=1)  # [N]
            valid = (dist > 1e-9) & (dist - radius_sum > clearance_min)
            if max_distance is not None:
                valid &= dist <= max_distance
            if not np.any(valid):
                continue
            u = rel[valid] / dist[valid, None]  # [V, 2] ego->ped unit vectors
            radial = np.sum((a_real[t, valid] - a_sim[valid]) * u, axis=1)  # [V]
            total += float(np.sum(radial * radial))
            count += int(valid.sum())
    if count == 0:
        return float("inf")
    return total / count


def _per_encounter_onset(onset_arrays: List[np.ndarray]) -> List[float]:
    """One scalar per encounter: the median onset distance, NaN if the encounter
    triggered no avoidance onset.

    Collapsing each encounter's per-ped onsets to a single median gives an
    INDEPENDENT unit per encounter (and, pooled across LOCO folds, per clip),
    which is what a two-sample KS assumes -- in contrast to concatenating every
    ped's onset, whose within-encounter samples react to the same ego trajectory
    and are autocorrelated (review m3/point5: that pooled KS p is anti-conservative).

    Uses nanmedian to honour this function's own NaN-means-no-onset contract: a
    non-empty array carrying a stray NaN (avoidance_onset_distance only emits
    finite distances, so this is defensive) must NOT collapse the whole encounter
    to NaN and be silently dropped from the KS as if it triggered no avoidance.
    """
    return [float(np.nanmedian(a)) if len(a) else float("nan") for a in onset_arrays]


def fidelity_report(
    encounters: List[Encounter],
    sigma: float,
    v0: float,
    ego_radius: float = DEFAULT_EGO_RADIUS,
    agent_radius: float = DEFAULT_AGENT_RADIUS,
    dt: float = 0.1,
    cruise_fn: Optional[CruiseEstimator] = None,
) -> Dict[str, float]:
    """Roll out at (sigma, v0) and compare simulated vs real avoidance (VALIDATION).

    Reports rollout ADE (mean ego-frame position error), the closest-approach
    distributions (one scalar per encounter, avoiding the autocorrelation that
    pooling per-step min-separation would introduce), the avoidance-onset
    distributions (pooled per ped), and two-sample KS statistics between
    sim and real for each. The KS p-value is reported but, with few encounters,
    only the statistic and the raw closest-approach values are informative.
    """
    sim_closest: List[float] = []
    real_closest: List[float] = []
    sim_onsets: List[np.ndarray] = []
    real_onsets: List[np.ndarray] = []
    ade_sum = 0.0
    ade_count = 0
    for enc in encounters:
        sim_xy = simulate_encounter(enc, sigma, v0, ego_radius, agent_radius, dt, cruise_fn)
        sim_sep = min_separation_series(enc.ego_xy, sim_xy)
        real_sep = min_separation_series(enc.ego_xy, enc.ped_xy)
        sim_closest.append(float(np.min(sim_sep)))
        real_closest.append(float(np.min(real_sep)))
        # Derive the onset acceleration the SAME way for sim and real -- both
        # from positions (ped_vel left None). Passing the recorded ped_vel for
        # real but finite-differencing sim would compute the two accelerations
        # with a different number of differentiations (grad(recorded v) vs
        # grad(grad(pos))), biasing the very sim-vs-real KS this metric exists for
        # (avoidance_onset_distance's docstring requires the two inputs agree).
        sim_onsets.append(avoidance_onset_distance(enc.ego_xy, sim_xy, dt=enc.dt))
        real_onsets.append(avoidance_onset_distance(enc.ego_xy, enc.ped_xy, dt=enc.dt))
        # Exclude frame 0 (sim_xy[0] == recorded start => error 0 for all params),
        # matching objective_rollout_ade so the reported and fitted ADE share a scale.
        frame_err = np.linalg.norm(sim_xy - enc.ped_xy, axis=2)[1:]  # [T-1, N]
        ade_sum += float(np.sum(frame_err))
        ade_count += frame_err.size

    sim_onset = np.concatenate(sim_onsets) if sim_onsets else np.array([])
    real_onset = np.concatenate(real_onsets) if real_onsets else np.array([])
    ks_closest, p_closest = compare_distributions_ks(np.array(sim_closest), np.array(real_closest))
    ks_onset, p_onset = compare_distributions_ks(sim_onset, real_onset)
    # Per-encounter onset: ONE median onset per encounter (NaN when that encounter
    # triggered no avoidance). Unlike the per-ped pool -- whose within-encounter
    # samples share a single ego trajectory and are autocorrelated, making the
    # pooled KS p anti-conservative (review m3/point5) -- this is an independent
    # unit per encounter (clip-independent once pooled across LOCO folds), so its
    # KS is a VALID two-sample test rather than a diagnostic-only one.
    onset_per_enc_sim = _per_encounter_onset(sim_onsets)
    onset_per_enc_real = _per_encounter_onset(real_onsets)
    return {
        "n_encounters": len(encounters),
        "rollout_ade": ade_sum / ade_count if ade_count else float("nan"),
        "mean_closest_sim": float(np.mean(sim_closest)) if sim_closest else float("nan"),
        "mean_closest_real": float(np.mean(real_closest)) if real_closest else float("nan"),
        "ks_closest": ks_closest,
        "p_closest": p_closest,
        "n_onset_sim": int(sim_onset.size),
        "n_onset_real": int(real_onset.size),
        "ks_onset": ks_onset,
        "p_onset": p_onset,
        # Raw per-encounter closest-approach scalars and per-ped onset distances
        # (review C1): with one held-out clip per LOCO fold the per-fold KS is a
        # degenerate n=1 statistic (always 1.0). Exposing the raw values lets the
        # evaluation POOL them across folds into a single, well-powered KS, and
        # report the honest mean standoff gap, instead of averaging n=1 KS=1.0.
        "closest_sim_raw": [float(x) for x in sim_closest],
        "closest_real_raw": [float(x) for x in real_closest],
        "onset_sim_raw": sim_onset.tolist(),
        "onset_real_raw": real_onset.tolist(),
        # Per-encounter (independent) onset scalars for the VALID onset KS.
        "onset_per_enc_sim_raw": onset_per_enc_sim,
        "onset_per_enc_real_raw": onset_per_enc_real,
    }
