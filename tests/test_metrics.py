import numpy as np
import pytest

from src.core.data_structures import EgoVehicleState, PedestrianState, SimulationResult
from src.core.metrics import (
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
