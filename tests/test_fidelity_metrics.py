import numpy as np
import pytest

from src.core.data_structures import EgoVehicleState, PedestrianState, SimulationResult
from src.core.metrics import (
    avoidance_onset_distance,
    calculate_min_separation,
    compare_distributions_ks,
    min_separation_series,
)


def test_min_separation_series_picks_nearest_pedestrian():
    ego = np.array([[0.0, 0.0], [0.0, 0.0]])
    ped = np.array([[[3.0, 0.0], [0.0, 4.0]], [[1.0, 0.0], [0.0, 2.0]]])  # [T=2,N=2,2]
    np.testing.assert_allclose(min_separation_series(ego, ped), [3.0, 1.0])


def test_min_separation_series_handles_no_pedestrians():
    ego = np.zeros((3, 2))
    ped = np.zeros((3, 0, 2))
    assert np.all(np.isinf(min_separation_series(ego, ped)))


def test_avoidance_onset_detects_evasion():
    # Stationary ego at origin; a pedestrian approaches along +x then reverses
    # (accelerates away) once it has closed to ~3 m.
    ego = np.zeros((5, 2))
    px = [4.0, 3.0, 2.0, 2.5, 3.5]
    ped = np.array([[[x, 0.0]] for x in px])  # [5, 1, 2]
    onsets = avoidance_onset_distance(ego, ped, dt=0.4, accel_threshold=0.3, max_distance=5.0)
    assert onsets.size == 1
    assert onsets[0] == pytest.approx(3.0)


def test_avoidance_onset_provided_velocity_matches_finite_difference():
    # The two branches (ped_vel given vs finite-differenced) must agree: same
    # step count and same acc-to-position alignment, else the sim-vs-real KS
    # comparison is biased between its inputs.
    ego = np.zeros((5, 2))
    px = [4.0, 3.0, 2.0, 2.5, 3.5]
    ped = np.array([[[x, 0.0]] for x in px])
    onset_fd = avoidance_onset_distance(ego, ped, dt=0.4)
    onset_vel = avoidance_onset_distance(
        ego, ped, ped_vel=np.gradient(ped, 0.4, axis=0), dt=0.4
    )
    np.testing.assert_allclose(onset_fd, onset_vel)
    assert onset_fd.size == 1 and onset_fd[0] == pytest.approx(3.0)


def test_avoidance_onset_ignores_constant_velocity_approach():
    ego = np.zeros((5, 2))
    px = [5.0, 4.0, 3.0, 2.0, 1.0]  # steady approach, zero acceleration
    ped = np.array([[[x, 0.0]] for x in px])
    assert avoidance_onset_distance(ego, ped, dt=0.4).size == 0


def test_avoidance_onset_respects_max_distance():
    # Evasion happens far away (>max_distance) so it must not be counted.
    ego = np.zeros((4, 2))
    px = [10.0, 9.0, 9.5, 10.5]  # reverses at ~9 m
    ped = np.array([[[x, 0.0]] for x in px])
    assert avoidance_onset_distance(ego, ped, dt=0.4, max_distance=5.0).size == 0


def test_ks_same_distribution_high_p_different_low_p():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    c = rng.normal(3.0, 1.0, 500)
    _, p_same = compare_distributions_ks(a, b)
    _, p_diff = compare_distributions_ks(a, c)
    assert p_same > 0.05
    assert p_diff < 0.01


def test_ks_returns_nan_when_empty_after_filtering():
    stat, p = compare_distributions_ks([np.inf, np.nan], [1.0, 2.0])
    assert np.isnan(stat) and np.isnan(p)


def test_calculate_min_separation_from_history():
    def _res(ex, ped_pos):
        pos = np.array(ped_pos, dtype=float)
        return SimulationResult(
            time=0.0,
            ego_state=EgoVehicleState(x=ex, y=0.0, yaw=0.0, v=0.0, a=0.0),
            ped_state=PedestrianState(
                positions=pos,
                velocities=np.zeros_like(pos),
                goals=pos.copy(),
            ),
        )

    history = [_res(0.0, [[2.0, 0.0]]), _res(0.0, [[1.0, 0.0]])]
    series, overall = calculate_min_separation(history)
    np.testing.assert_allclose(series, [2.0, 1.0])
    assert overall == pytest.approx(1.0)


def test_per_encounter_onset_collapses_to_one_independent_scalar():
    from src.simulation.calibration_harness import _per_encounter_onset

    # Three encounters: the first has two per-ped onsets (median collapses them
    # to ONE independent scalar), the second one, the third NONE -> NaN. The
    # per-ped pool would be length 3 (autocorrelated within encounter 1); the
    # per-encounter view is length 3 with one independent unit per encounter.
    onset_arrays = [np.array([2.0, 4.0]), np.array([3.0]), np.array([])]
    got = _per_encounter_onset(onset_arrays)
    assert len(got) == 3
    assert got[0] == pytest.approx(3.0)  # median([2,4])
    assert got[1] == pytest.approx(3.0)
    assert np.isnan(got[2])  # no onset triggered -> NaN, dropped by the KS filter

    # A non-empty array with a stray NaN must NOT collapse the encounter to NaN
    # (nanmedian over the finite values) -- it DID trigger avoidance.
    got_nan = _per_encounter_onset([np.array([2.0, np.nan, 4.0])])
    assert got_nan[0] == pytest.approx(3.0)
