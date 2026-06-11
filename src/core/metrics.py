import numpy as np
from typing import List, Dict, Tuple

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
) -> Tuple[float, float, int, int]:
    """Calculate fixed-horizon, scene-level best-of-N SGAN evaluation metrics."""
    stride = _steps_for_interval(prediction_dt, dt)
    pred_indices = stride * np.arange(1, prediction_steps + 1) - 1
    future_offsets = stride * np.arange(1, prediction_steps + 1)
    total_ade = 0.0
    total_fde = 0.0
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

        # SGAN minADE/minFDE each select one scene-level sample independently.
        total_ade += float(np.min(ade_samples)) * n_peds
        total_fde += float(np.min(fde_samples)) * n_peds
        trajectory_count += n_peds
        max_samples = max(max_samples, n_samples)

    if trajectory_count == 0:
        return float("nan"), float("nan"), 0, 0
    return (
        total_ade / trajectory_count,
        total_fde / trajectory_count,
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
    ade, fde, max_samples, _ = _standard_ade_fde_details(
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

    ade, fde, n_samples, ade_eval_count = _standard_ade_fde_details(
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
        "pred_samples": n_samples,
        "ade_eval_count": ade_eval_count,
        "planning_ade": planning_ade,
        "planning_fde": planning_fde,
        "planning_eval_count": planning_eval_count,
        "nll": nll,
        "nll_eval_count": nll_eval_count,
    }
