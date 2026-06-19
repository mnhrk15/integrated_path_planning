import numpy as np
import pytest

from src.core.data_structures import EgoVehicleState, PedestrianState, SimulationResult
from src.core.metrics import (
    _standard_ade_fde_details,
    calculate_aggregate_metrics,
    calculate_planning_ade_fde,
    calculate_standard_ade_fde,
)


def _result(positions, predictions=None, distribution=None):
    positions = np.asarray(positions, dtype=float)
    n_peds = len(positions)
    return SimulationResult(
        time=0.0,
        ego_state=EgoVehicleState(x=0.0, y=0.0, yaw=0.0, v=0.0, a=0.0),
        ped_state=PedestrianState(
            positions=positions,
            velocities=np.zeros((n_peds, 2)),
            goals=np.zeros((n_peds, 2)),
        ),
        predicted_trajectories=predictions,
        predicted_distribution=distribution,
    )


def test_standard_ade_fde_uses_complete_fixed_horizon_only():
    """Standard ADE/FDE should exclude truncated predictions near the run end."""
    perfect = np.array([[[1.0, 0.0], [2.0, 0.0]]])
    truncated_bad = np.array([[[100.0, 0.0], [100.0, 0.0]]])
    history = [
        _result([[0.0, 0.0]], predictions=perfect),
        _result([[1.0, 0.0]], predictions=truncated_bad),
        _result([[2.0, 0.0]]),
    ]

    ade, fde, samples = calculate_standard_ade_fde(
        history,
        dt=0.1,
        prediction_dt=0.1,
        prediction_steps=2,
    )

    assert ade == pytest.approx(0.0)
    assert fde == pytest.approx(0.0)
    assert samples == 1


def test_standard_ade_fde_selects_best_sample_after_scene_aggregation():
    """Different pedestrians cannot select different best samples for one metric."""
    distribution = np.array(
        [
            [[[1.0, 0.0]], [[3.0, 0.0]]],
            [[[3.0, 0.0]], [[1.0, 0.0]]],
        ]
    )
    history = [
        _result([[0.0, 0.0], [0.0, 0.0]], distribution=distribution),
        _result([[1.0, 0.0], [1.0, 0.0]]),
    ]

    ade, fde, samples = calculate_standard_ade_fde(
        history,
        dt=0.1,
        prediction_dt=0.1,
        prediction_steps=1,
    )

    assert ade == pytest.approx(1.0)
    assert fde == pytest.approx(1.0)
    assert samples == 2


def test_standard_minade_and_minfde_select_samples_independently():
    """SGAN-style minADE and minFDE may come from different scene-level samples."""
    distribution = np.array(
        [
            [[[1.0, 0.0], [12.0, 0.0]]],
            [[[7.0, 0.0], [8.0, 0.0]]],
        ]
    )
    history = [
        _result([[0.0, 0.0]], distribution=distribution),
        _result([[1.0, 0.0]]),
        _result([[2.0, 0.0]]),
    ]

    ade, fde, samples = calculate_standard_ade_fde(
        history,
        dt=0.1,
        prediction_dt=0.1,
        prediction_steps=2,
    )

    assert ade == pytest.approx(5.0)
    assert fde == pytest.approx(6.0)
    assert samples == 2


def test_per_agent_best_of_n_diverges_from_scene_level():
    """Review M1: with N>1 the per-agent best-of-N (canonical SGAN minADE) lets
    each ped pick its own best sample, so it can be far smaller than the
    scene-level joint min. Same 2-ped/2-sample crossing as the scene-level test:
    scene-level=1.0 (no single sample is best for both peds) but per-agent=0.0
    (each ped's own best sample is perfect)."""
    distribution = np.array(
        [
            [[[1.0, 0.0]], [[3.0, 0.0]]],
            [[[3.0, 0.0]], [[1.0, 0.0]]],
        ]
    )
    history = [
        _result([[0.0, 0.0], [0.0, 0.0]], distribution=distribution),
        _result([[1.0, 0.0], [1.0, 0.0]]),
    ]
    ade, fde, ade_pa, fde_pa, samples, count = _standard_ade_fde_details(
        history, dt=0.1, prediction_dt=0.1, prediction_steps=1)
    assert ade == pytest.approx(1.0) and fde == pytest.approx(1.0)       # scene-level
    assert ade_pa == pytest.approx(0.0) and fde_pa == pytest.approx(0.0)  # per-agent
    assert count == 2


def test_per_agent_equals_scene_level_for_single_sample():
    """A deterministic (N=1) predictor: the per-ped min over one sample is the
    identity, so per-agent == scene-level. The two metrics only diverge for N>1,
    which is exactly why reporting both isolates the best-of-N inflation."""
    predictions = np.array([[[1.5, 0.0], [2.5, 0.0]]])  # 1 ped, 2 steps, slight error
    history = [
        _result([[0.0, 0.0]], predictions=predictions),
        _result([[1.0, 0.0]]),
        _result([[2.0, 0.0]]),
    ]
    ade, fde, ade_pa, fde_pa, _s, _c = _standard_ade_fde_details(
        history, dt=0.1, prediction_dt=0.1, prediction_steps=2)
    assert ade == pytest.approx(0.5)
    assert ade_pa == pytest.approx(ade) and fde_pa == pytest.approx(fde)


def test_standard_ade_fde_returns_nan_when_no_complete_horizon():
    """Unevaluated standard metrics should not look like perfect zero error."""
    history = [
        _result([[0.0, 0.0]], predictions=np.array([[[1.0, 0.0]]])),
        _result([[1.0, 0.0]]),
    ]

    ade, fde, samples = calculate_standard_ade_fde(
        history,
        dt=0.1,
        prediction_dt=0.1,
        prediction_steps=2,
    )
    metrics = calculate_aggregate_metrics(
        history,
        dt=0.1,
        prediction_dt=0.1,
        prediction_steps=2,
    )

    assert np.isnan(ade)
    assert np.isnan(fde)
    assert samples == 0
    assert np.isnan(metrics["ade"])
    assert np.isnan(metrics["fde"])
    assert metrics["ade_eval_count"] == 0


def test_planning_ade_fde_uses_selected_dense_trajectory():
    """Planning metrics should evaluate the trajectory consumed by the planner."""
    selected = np.array([[[2.0, 0.0], [4.0, 0.0]]])
    perfect_distribution = np.array([[[[1.0, 0.0], [2.0, 0.0]]]])
    history = [
        _result(
            [[0.0, 0.0]],
            predictions=selected,
            distribution=perfect_distribution,
        ),
        _result([[1.0, 0.0]]),
        _result([[2.0, 0.0]]),
    ]

    standard_ade, standard_fde, _ = calculate_standard_ade_fde(
        history,
        dt=0.1,
        prediction_dt=0.1,
        prediction_steps=2,
    )
    planning_ade, planning_fde, count = calculate_planning_ade_fde(history)

    assert standard_ade == pytest.approx(0.0)
    assert standard_fde == pytest.approx(0.0)
    assert planning_ade == pytest.approx(1.5)
    assert planning_fde == pytest.approx(2.0)
    assert count == 1


def test_aggregate_metrics_exports_standard_and_planning_names():
    """Aggregate output should keep standard and planning metrics distinct."""
    predictions = np.array([[[1.0, 0.0]]])
    history = [
        _result([[0.0, 0.0]], predictions=predictions),
        _result([[1.0, 0.0]]),
    ]

    metrics = calculate_aggregate_metrics(
        history,
        dt=0.1,
        prediction_dt=0.1,
        prediction_steps=1,
    )

    assert metrics["ade"] == pytest.approx(0.0)
    assert metrics["fde"] == pytest.approx(0.0)
    assert metrics["planning_ade"] == pytest.approx(0.0)
    assert metrics["planning_fde"] == pytest.approx(0.0)
    assert metrics["ade_eval_count"] == 1
    assert metrics["planning_eval_count"] == 1


def test_aggregate_metrics_reports_comfort_accel_and_jerk():
    """Comfort metrics: mean |accel| and RMS jerk over the ego-state series."""
    history = [_result([[0.0, 0.0]]), _result([[1.0, 0.0]])]
    history[0].ego_state.a, history[0].ego_state.jerk = 1.0, 2.0
    history[1].ego_state.a, history[1].ego_state.jerk = -3.0, -4.0

    metrics = calculate_aggregate_metrics(
        history,
        dt=0.1,
        prediction_dt=0.1,
        prediction_steps=1,
    )

    # mean(|1|, |-3|) = 2.0 ; max(|a|) = 3.0
    assert metrics["mean_accel"] == pytest.approx(2.0)
    assert metrics["max_accel"] == pytest.approx(3.0)
    # sqrt(mean(2^2, 4^2)) = sqrt(10) ; mean(|jerk|) = 3.0 ; max = 4.0
    assert metrics["rms_jerk"] == pytest.approx(np.sqrt(10.0))
    assert metrics["mean_jerk"] == pytest.approx(3.0)
    assert metrics["max_jerk"] == pytest.approx(4.0)


def test_kde_nll_analytic_two_symmetric_samples():
    """Two samples symmetric around the GT, both bandwidths at the floor.

    With samples at (1 +- delta, 0), GT at (1, 0) and delta small enough that
    Scott's bandwidth floors at h on both axes, every kernel contributes
    log N(delta; h) and NLL = log(2 pi h^2) + 0.5 (delta/h)^2.
    """
    from src.core.metrics import KDE_BANDWIDTH_FLOOR, calculate_kde_nll

    delta = 0.01
    h = KDE_BANDWIDTH_FLOOR
    distribution = np.array([
        [[[1.0 + delta, 0.0]]],
        [[[1.0 - delta, 0.0]]],
    ])  # [N=2, P=1, T=1, 2]
    history = [
        _result([[0.0, 0.0]], distribution=distribution),
        _result([[1.0, 0.0]]),
    ]

    nll, count = calculate_kde_nll(history, dt=0.1, prediction_dt=0.1, prediction_steps=1)

    assert count == 1
    assert nll == pytest.approx(np.log(2 * np.pi * h**2) + 0.5 * (delta / h) ** 2)


def test_kde_nll_increases_when_gt_leaves_the_distribution():
    from src.core.metrics import calculate_kde_nll

    near = np.array([[[[1.0, 0.0]]], [[[1.2, 0.0]]]])
    far = np.array([[[[5.0, 0.0]]], [[[5.2, 0.0]]]])
    gt = [[1.1, 0.0]]
    h_near = [_result([[0.0, 0.0]], distribution=near), _result(gt)]
    h_far = [_result([[0.0, 0.0]], distribution=far), _result(gt)]

    nll_near, _ = calculate_kde_nll(h_near, dt=0.1, prediction_dt=0.1, prediction_steps=1)
    nll_far, _ = calculate_kde_nll(h_far, dt=0.1, prediction_dt=0.1, prediction_steps=1)

    assert nll_far > nll_near


def test_kde_nll_nan_without_distribution():
    """Single-forecast predictors (CV) carry no distribution: NLL is NaN."""
    from src.core.metrics import calculate_kde_nll

    history = [
        _result([[0.0, 0.0]], predictions=np.array([[[1.0, 0.0]]])),
        _result([[1.0, 0.0]]),
    ]

    nll, count = calculate_kde_nll(history, dt=0.1, prediction_dt=0.1, prediction_steps=1)

    assert count == 0
    assert np.isnan(nll)

    metrics = calculate_aggregate_metrics(history, dt=0.1, prediction_dt=0.1, prediction_steps=1)
    assert np.isnan(metrics["nll"])
    assert metrics["nll_eval_count"] == 0


def test_kde_nll_skips_replicated_deterministic_samples():
    """N identical samples (CV pseudo-distribution) must not produce an NLL."""
    from src.core.metrics import calculate_kde_nll

    same = np.array([[[[1.0, 0.0]]]] * 20)  # [N=20, P=1, T=1, 2], all identical
    history = [
        _result([[0.0, 0.0]], distribution=same),
        _result([[1.5, 0.0]]),
    ]

    nll, count = calculate_kde_nll(history, dt=0.1, prediction_dt=0.1, prediction_steps=1)

    assert count == 0
    assert np.isnan(nll)
