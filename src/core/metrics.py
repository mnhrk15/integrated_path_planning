import numpy as np
from typing import Dict, List, Optional, Tuple

from scipy.stats import ks_2samp

from .data_structures import SimulationResult


SGAN_EVAL_DT = 0.4
SGAN_EVAL_STEPS = 12

# Bandwidth floor for the KDE-based NLL [m]. Keeps the mixture non-degenerate
# when all samples coincide at a step (e.g. converged predictions).
KDE_BANDWIDTH_FLOOR = 0.05

# Per-point lower bound on log-likelihood [nats] (KDE-NLL convention from the
# Trajectron++/planning-aware evaluation line). Without it the mean is
# dominated by ground-truth points far outside the sample support.
KDE_NLL_LOG_P_FLOOR = -20.0


def _steps_for_interval(interval: float, dt: float) -> int:
    """Return the number of simulation steps in an evaluation interval."""
    ratio = interval / dt
    rounded_ratio = int(round(ratio))
    if rounded_ratio <= 0 or not np.isclose(ratio, rounded_ratio):
        raise ValueError(f"Evaluation interval {interval} must be a multiple of dt={dt}")
    return rounded_ratio


def _standard_ade_fde_details(
    history: List[SimulationResult],
    dt: float,
    prediction_dt: float,
    prediction_steps: int,
) -> Tuple[float, float, float, float, int, int]:
    """Fixed-horizon best-of-N SGAN ADE/FDE: scene-level AND per-agent.

    Returns ``(ade, fde, ade_per_agent, fde_per_agent, max_samples, count)``.

    ``ade``/``fde`` are the historical SCENE-LEVEL joint best-of-N: one sample is
    chosen for the whole scene (``min`` over the ped+step mean). ``ade_per_agent``
    /``fde_per_agent`` are the canonical SGAN minADE/minFDE where EACH pedestrian
    picks its own best sample. The two differ for stochastic predictors (N>1) and
    are identical for a deterministic one (N=1, the per-ped min is the identity),
    so reporting both (review M1) shows whether the cv/lstm/sgan ordering is an
    artefact of the non-canonical scene-level joint selection -- the scene-level
    one inflates only the multi-sample methods.
    """
    stride = _steps_for_interval(prediction_dt, dt)
    pred_indices = stride * np.arange(1, prediction_steps + 1) - 1
    future_offsets = stride * np.arange(1, prediction_steps + 1)
    total_ade = 0.0
    total_fde = 0.0
    total_ade_pa = 0.0
    total_fde_pa = 0.0
    trajectory_count = 0
    max_samples = 0

    for i, result in enumerate(history):
        has_distribution = (
            result.predicted_distribution is not None
            and result.predicted_distribution.size > 0
        )
        has_single = (
            result.predicted_trajectories is not None
            and result.predicted_trajectories.size > 0
        )
        if not has_distribution and not has_single:
            continue

        samples = (
            result.predicted_distribution
            if has_distribution
            else result.predicted_trajectories[None, ...]
        )
        n_samples, n_peds, dense_steps, _ = samples.shape
        if dense_steps <= pred_indices[-1] or i + future_offsets[-1] >= len(history):
            continue

        gt_traj = np.stack(
            [history[i + offset].ped_state.positions for offset in future_offsets],
            axis=1,
        )
        if gt_traj.shape != (n_peds, prediction_steps, 2):
            continue

        displacement = np.linalg.norm(
            samples[:, :, pred_indices, :] - gt_traj[None, ...],
            axis=3,
        )
        ade_samples = np.mean(displacement, axis=(1, 2))
        fde_samples = np.mean(displacement[:, :, -1], axis=1)

        # Scene-level joint best-of-N: one sample chosen for the whole scene.
        total_ade += float(np.min(ade_samples)) * n_peds
        total_fde += float(np.min(fde_samples)) * n_peds
        # Per-agent best-of-N (canonical SGAN minADE/minFDE): each ped picks its
        # own best sample. min over the sample axis, after the per-ped step mean.
        total_ade_pa += float(np.sum(np.min(np.mean(displacement, axis=2), axis=0)))
        total_fde_pa += float(np.sum(np.min(displacement[:, :, -1], axis=0)))
        trajectory_count += n_peds
        max_samples = max(max_samples, n_samples)

    if trajectory_count == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0, 0
    return (
        total_ade / trajectory_count,
        total_fde / trajectory_count,
        total_ade_pa / trajectory_count,
        total_fde_pa / trajectory_count,
        max_samples,
        trajectory_count,
    )


def _kde_nll_details(
    history: List[SimulationResult],
    dt: float,
    prediction_dt: float,
    prediction_steps: int,
) -> Tuple[float, int]:
    """Mean KDE negative log-likelihood of the ground truth under the samples.

    For every prediction origin that carries a sample distribution (>= 2
    samples), a Gaussian-mixture KDE is fit per pedestrian and evaluation
    step (Scott's rule bandwidth per axis, floored at KDE_BANDWIDTH_FLOOR)
    and the log-density of the ground-truth position is accumulated. Like
    the standard ADE/FDE, evaluation happens at the predictor cadence and
    only over origins with a complete future horizon. Origins with a single
    forecast (e.g. the CV model) are skipped, so the metric is NaN there.
    """
    stride = _steps_for_interval(prediction_dt, dt)
    pred_indices = stride * np.arange(1, prediction_steps + 1) - 1
    future_offsets = stride * np.arange(1, prediction_steps + 1)
    total_log_lik = 0.0
    eval_count = 0

    for i, result in enumerate(history):
        dist = result.predicted_distribution
        if dist is None or dist.size == 0 or dist.shape[0] < 2:
            continue
        n_samples, n_peds, dense_steps, _ = dist.shape
        if dense_steps <= pred_indices[-1] or i + future_offsets[-1] >= len(history):
            continue

        gt_traj = np.stack(
            [history[i + offset].ped_state.positions for offset in future_offsets],
            axis=1,
        )
        if gt_traj.shape != (n_peds, prediction_steps, 2):
            continue

        samples = dist[:, :, pred_indices, :]  # [N, P, T, 2]
        if not np.any(np.ptp(samples, axis=0) > 0):
            # All samples identical: a deterministic predictor replicated into
            # a pseudo-distribution (e.g. CV) — NLL is not defined there.
            continue
        scott = n_samples ** (-1.0 / 6.0)  # Scott's rule, d=2
        bandwidth = np.maximum(
            samples.std(axis=0, ddof=1) * scott, KDE_BANDWIDTH_FLOOR
        )  # [P, T, 2]
        scaled = (samples - gt_traj[None, ...]) / bandwidth[None, ...]
        log_kernel = (
            -0.5 * np.sum(scaled**2, axis=3)
            - np.log(2.0 * np.pi * bandwidth[..., 0] * bandwidth[..., 1])[None, ...]
        )  # [N, P, T]
        peak = log_kernel.max(axis=0)  # [P, T]
        log_p = peak + np.log(np.mean(np.exp(log_kernel - peak[None, ...]), axis=0))
        log_p = np.maximum(log_p, KDE_NLL_LOG_P_FLOOR)
        total_log_lik += float(log_p.sum())
        eval_count += log_p.size

    if eval_count == 0:
        return float("nan"), 0
    return -total_log_lik / eval_count, eval_count


def calculate_kde_nll(
    history: List[SimulationResult],
    dt: float,
    prediction_dt: float = SGAN_EVAL_DT,
    prediction_steps: int = SGAN_EVAL_STEPS,
) -> Tuple[float, int]:
    """Calculate the KDE-based NLL of the ground truth under the predictions."""
    return _kde_nll_details(history, dt, prediction_dt, prediction_steps)


def calculate_standard_ade_fde(
    history: List[SimulationResult],
    dt: float,
    prediction_dt: float = SGAN_EVAL_DT,
    prediction_steps: int = SGAN_EVAL_STEPS,
) -> Tuple[float, float, int]:
    """Calculate SGAN-style fixed-horizon, scene-level best-of-N ADE/FDE.

    Only prediction origins with a complete future horizon are evaluated.
    Dense trajectories are sampled at the predictor cadence. Note that dense
    predictions are re-anchored to the evaluation origin (current time), so at
    origins where the last observation sample is stale (3 out of 4 steps at
    dt=0.1) the sampled values are linear interpolations between raw predictor
    outputs, and the last evaluation point lies up to `staleness` beyond the
    raw support (filled by the predictor's clamped tail extrapolation).
    minADE and minFDE select their best scene-level sample independently.
    """
    ade, fde, _ade_pa, _fde_pa, max_samples, _ = _standard_ade_fde_details(
        history,
        dt,
        prediction_dt,
        prediction_steps,
    )
    return ade, fde, max_samples


def calculate_ade_fde(
    history: List[SimulationResult],
    dt: float,
    prediction_dt: float = SGAN_EVAL_DT,
    prediction_steps: int = SGAN_EVAL_STEPS,
) -> Tuple[float, float, int]:
    """Backward-compatible alias for standard SGAN-style ADE/FDE."""
    return calculate_standard_ade_fde(history, dt, prediction_dt, prediction_steps)


def calculate_planning_ade_fde(
    history: List[SimulationResult],
) -> Tuple[float, float, int]:
    """Calculate rolling ADE/FDE for the single trajectory sent to the planner.

    This metric uses dense planner-resolution trajectories and the available
    future portion near the end of a simulation. It is intentionally separate
    from fixed-horizon SGAN evaluation.
    """
    total_ade = 0.0
    total_fde = 0.0
    trajectory_count = 0

    for i, result in enumerate(history):
        predictions = result.predicted_trajectories
        if predictions is None or predictions.size == 0:
            continue

        n_peds, n_steps, _ = predictions.shape
        eval_steps = min(n_steps, len(history) - (i + 1))
        if eval_steps == 0:
            continue

        gt_traj = np.stack(
            [history[i + 1 + k].ped_state.positions for k in range(eval_steps)],
            axis=1,
        )
        if gt_traj.shape != (n_peds, eval_steps, 2):
            continue

        displacement = np.linalg.norm(
            predictions[:, :eval_steps, :] - gt_traj,
            axis=2,
        )
        total_ade += float(np.sum(np.mean(displacement, axis=1)))
        total_fde += float(np.sum(displacement[:, -1]))
        trajectory_count += n_peds

    if trajectory_count == 0:
        return float("nan"), float("nan"), 0
    return (
        total_ade / trajectory_count,
        total_fde / trajectory_count,
        trajectory_count,
    )


def calculate_aggregate_metrics(
    history: List[SimulationResult],
    dt: float,
    prediction_dt: float = SGAN_EVAL_DT,
    prediction_steps: int = SGAN_EVAL_STEPS,
) -> Dict[str, float]:
    """Calculate aggregate metrics for the entire simulation."""
    min_distances = [r.metrics.get("min_distance", float("inf")) for r in history]
    ttc_list = [r.metrics.get("ttc", float("inf")) for r in history]
    ttc_valid = [t for t in ttc_list if t > 0 and t != float("inf")]

    jerks = [abs(r.ego_state.jerk) for r in history]
    accels = [abs(r.ego_state.a) for r in history]

    ade, fde, ade_pa, fde_pa, n_samples, ade_eval_count = _standard_ade_fde_details(
        history,
        dt,
        prediction_dt,
        prediction_steps,
    )
    planning_ade, planning_fde, planning_eval_count = calculate_planning_ade_fde(history)
    nll, nll_eval_count = _kde_nll_details(
        history,
        dt,
        prediction_dt,
        prediction_steps,
    )

    return {
        "min_dist": min(min_distances) if min_distances else 0.0,
        "collision_count": sum(1 for r in history if r.metrics.get("collision", False)),
        "min_ttc": min(ttc_valid) if ttc_valid else float("inf"),
        "max_jerk": max(jerks) if jerks else 0.0,
        "mean_jerk": np.mean(jerks) if jerks else 0.0,
        "rms_jerk": float(np.sqrt(np.mean(np.square(jerks)))) if jerks else 0.0,
        "max_accel": max(accels) if accels else 0.0,
        "mean_accel": np.mean(accels) if accels else 0.0,
        "ade": ade,
        "fde": fde,
        "ade_per_agent": ade_pa,
        "fde_per_agent": fde_pa,
        "pred_samples": n_samples,
        "ade_eval_count": ade_eval_count,
        "planning_ade": planning_ade,
        "planning_fde": planning_fde,
        "planning_eval_count": planning_eval_count,
        "nll": nll,
        "nll_eval_count": nll_eval_count,
    }


# ---------------------------------------------------------------------------
# Fidelity metrics for real-data calibration (MSc thesis, Axis A / RQ2).
#
# These compare ego-pedestrian *interaction* between a simulation and recorded
# ground truth: how close pedestrians get to the vehicle (min separation) and at
# what distance they begin to evade it (avoidance onset). A two-sample KS test
# then quantifies sim-vs-real distribution agreement. The core functions take
# plain position/velocity arrays so the same code applies to both AVEC
# SimulationResult histories and raw DUT/CITR recordings (enabling KS
# comparison). A fixed pedestrian population N within the window is assumed.
#
# NOTE: "distance"/"separation" here is centre-to-centre ego-pedestrian
# distance, NOT envelope breach; interpret as a relative comparison between
# conditions (see thesis limitations on the 1.2 m collision envelope).
# ---------------------------------------------------------------------------


def min_separation_series(ego_xy: np.ndarray, ped_xy: np.ndarray) -> np.ndarray:
    """Per-step minimum ego-pedestrian distance.

    Args:
        ego_xy: Ego positions [T, 2].
        ped_xy: Pedestrian positions [T, N, 2] (fixed N).
    Returns:
        [T] minimum distance to any pedestrian at each step (inf where N == 0).
    """
    ego_xy = np.asarray(ego_xy, dtype=float)
    ped_xy = np.asarray(ped_xy, dtype=float)
    if ego_xy.shape[0] != ped_xy.shape[0]:
        raise ValueError(
            f"ego_xy T={ego_xy.shape[0]} != ped_xy T={ped_xy.shape[0]}"
        )
    if ped_xy.shape[1] == 0:
        return np.full(ego_xy.shape[0], np.inf)
    dists = np.linalg.norm(ped_xy - ego_xy[:, None, :], axis=2)  # [T, N]
    return np.min(dists, axis=1)


def avoidance_onset_distance(
    ego_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_vel: Optional[np.ndarray] = None,
    dt: float = 0.4,
    accel_threshold: float = 0.3,
    max_distance: float = 5.0,
) -> np.ndarray:
    """Ego-pedestrian distance at which each pedestrian starts evading the ego.

    For every pedestrian, find the first step at which it accelerates *away*
    from the ego (acceleration component along the ego->ped direction exceeds
    ``accel_threshold``) while within ``max_distance``, and record the ego-ped
    distance there. This captures the reactive standoff that the SFM ego
    repulsion (sigma, v0) should reproduce when calibrated.

    Args:
        ego_xy: Ego positions [T, 2].
        ped_xy: Pedestrian positions [T, N, 2].
        ped_vel: Pedestrian velocities [T, N, 2]; finite-differenced if None.
        dt: Time step [s].
        accel_threshold: Min away-pointing acceleration [m/s^2] to count as onset.
        max_distance: Only consider steps within this ego-ped distance [m].
    Returns:
        1-D array of onset distances [m], one per pedestrian that evades.
    """
    ego_xy = np.asarray(ego_xy, dtype=float)
    ped_xy = np.asarray(ped_xy, dtype=float)
    T, N, _ = ped_xy.shape
    if T < 2 or N == 0:
        return np.array([])
    if ped_vel is None:
        vel = np.gradient(ped_xy, dt, axis=0)  # [T, N, 2]
    else:
        vel = np.asarray(ped_vel, dtype=float)
        if vel.shape != ped_xy.shape:
            raise ValueError(
                f"ped_vel shape {vel.shape} != ped_xy shape {ped_xy.shape}"
            )
    # Acceleration via np.gradient (central differences, one-sided at the ends)
    # so it is defined at every step (no fabricated boundary zero) AND computed
    # by the SAME rule whether the velocity was provided or finite-differenced
    # from positions -- the two branches must agree, with identical step count
    # and acc[t]<->ped_xy[t] alignment, or the sim-vs-real KS comparison the
    # metric exists for would be biased between its two inputs.
    acc = np.gradient(vel, dt, axis=0)  # [T, N, 2]

    onsets: List[float] = []
    for j in range(N):
        for t in range(T):
            rel = ped_xy[t, j] - ego_xy[t]
            dist = float(np.linalg.norm(rel))
            if dist < 1e-9 or dist > max_distance:
                continue
            away = float(np.dot(acc[t, j], rel / dist))
            if away > accel_threshold:
                onsets.append(dist)
                break
    return np.array(onsets)


def compare_distributions_ks(
    sim_samples: np.ndarray, real_samples: np.ndarray
) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test: (statistic, p-value).

    Inputs are flattened to 1-D and non-finite values are dropped. The test
    assumes the samples are i.i.d.: do NOT pool a strongly autocorrelated
    series (e.g. every per-step min-separation of a single encounter) without
    thinning, or the p-value is anti-conservative.

    A small p-value means the simulated and real samples differ. A large
    p-value only means equality could not be rejected -- it is NOT proof of a
    good match (low power from few samples also yields a large p), so always
    report the statistic alongside it. Returns (nan, nan) if either sample is
    empty after filtering.
    """
    sim = np.asarray(sim_samples, dtype=float)
    real = np.asarray(real_samples, dtype=float)
    sim = sim[np.isfinite(sim)]
    real = real[np.isfinite(real)]
    if sim.size == 0 or real.size == 0:
        return float("nan"), float("nan")
    result = ks_2samp(sim, real)
    return float(result.statistic), float(result.pvalue)


def calculate_min_separation(
    history: List[SimulationResult],
) -> Tuple[np.ndarray, float]:
    """Min-separation series and overall minimum from a SimulationResult history.

    Requires a fixed pedestrian population across the history (e.g. a replayed
    encounter window). Returns ([T] per-step min distance, overall min).
    """
    ego_xy = np.array([[r.ego_state.x, r.ego_state.y] for r in history], dtype=float)
    try:
        ped_xy = np.stack([r.ped_state.positions for r in history], axis=0)
    except ValueError as exc:
        raise ValueError(
            "calculate_min_separation requires a fixed pedestrian population "
            "across the history (pedestrian count varies between steps)"
        ) from exc
    series = min_separation_series(ego_xy, ped_xy)
    overall = float(np.min(series)) if series.size else float("inf")
    return series, overall
