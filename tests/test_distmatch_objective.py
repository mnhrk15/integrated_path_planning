"""Regression tests for the distribution-matching objective (synthetic data).

The load-bearing guarantee: ``objective_multi(w_ade=1, w_dist=0, w_onset=0)``
is bit-for-bit ``objective_rollout_ade`` (same rollouts, same accumulation
order), so the weight sweep's w=0 arm IS the canonical fitter, not a
reimplementation of it.
"""
import numpy as np
import pytest

from src.core.metrics import (
    compare_distributions_emd,
    compare_distributions_energy,
)
from src.calibration import calibrate
from src.datasets.vci_encounter import Encounter
from src.simulation.calibration_harness import (
    _far_goals,
    objective_multi,
    objective_rollout_ade,
    simulate_encounter,
)
from tests.test_calibration_harness import make_encounter


# --------------------------------------------------------------------------- #
# metrics-layer distances
# --------------------------------------------------------------------------- #
def test_emd_matches_hand_computed_value_and_units():
    """EMD between two point masses is their distance in metres."""
    assert compare_distributions_emd([1.0], [3.5]) == pytest.approx(2.5)
    # identical samples -> zero; shift by d -> exactly d (translation property)
    a = np.array([1.0, 2.0, 4.0])
    assert compare_distributions_emd(a, a) == 0.0
    assert compare_distributions_emd(a, a + 0.7) == pytest.approx(0.7)


def test_distance_conventions_filter_nonfinite_and_empty():
    a = np.array([1.0, np.nan, 2.0])
    b = np.array([1.0, 2.0, np.inf])
    assert np.isfinite(compare_distributions_emd(a, b))
    assert np.isnan(compare_distributions_emd([], [1.0]))
    assert np.isnan(compare_distributions_energy([np.nan], [1.0]))


# --------------------------------------------------------------------------- #
# objective_multi
# --------------------------------------------------------------------------- #
def test_objective_multi_w_dist_zero_equals_rollout_ade_bitwise():
    enc = make_encounter(T=16, n_extra_far=1)
    for params in ((0.7, 3.5), (1.0, 0.5), (1.5, 0.0)):
        ade = objective_rollout_ade([enc], *params)
        multi = objective_multi([enc], *params, w_ade=1.0, w_dist=0.0, w_onset=0.0)
        assert multi == ade  # bitwise, not approx
    # ... and with the ADE-only interaction filter engaged
    ade_f = objective_rollout_ade([enc], 0.7, 3.5, interaction_distance=8.0)
    multi_f = objective_multi([enc], 0.7, 3.5, w_dist=0.0,
                              interaction_distance=8.0)
    assert multi_f == ade_f


def test_objective_multi_dist_term_is_nonnegative_and_deterministic():
    enc = make_encounter(T=16)
    ade = objective_rollout_ade([enc], 0.7, 3.5)
    multi = objective_multi([enc], 0.7, 3.5, w_ade=1.0, w_dist=1.0)
    assert multi >= ade  # EMD >= 0
    assert multi == objective_multi([enc], 0.7, 3.5, w_ade=1.0, w_dist=1.0)


def test_objective_multi_pure_distribution_arm_is_finite():
    enc = make_encounter(T=16)
    loss = objective_multi([enc], 0.7, 3.5, w_ade=0.0, w_dist=1.0)
    assert np.isfinite(loss) and loss >= 0.0


def test_objective_multi_empty_ade_pool_returns_inf():
    """interaction_distance below every ped's closest approach empties the ADE
    term -> inf (mirrors objective_rollout_ade's count==0 convention)."""
    enc = make_encounter(T=8, ped_start=(0.0, 40.0))  # never near the ego
    assert objective_multi([enc], 0.7, 3.5, interaction_distance=0.01) == float("inf")


def test_objective_multi_interaction_filter_leaves_closest_term_all_peds():
    """The filter applies to the ADE term ONLY: with a far ped added, the
    closest-approach pools still see every ped (mirrors fidelity_report)."""
    enc = make_encounter(T=16, n_extra_far=1)
    # pure-distribution losses with and without the ADE filter must be equal
    a = objective_multi([enc], 0.7, 3.5, w_ade=0.0, w_dist=1.0)
    b = objective_multi([enc], 0.7, 3.5, w_ade=0.0, w_dist=1.0,
                        interaction_distance=8.0)
    assert a == b


def test_objective_multi_unknown_metric_raises_before_rollout():
    enc = make_encounter(T=8)
    with pytest.raises(ValueError, match="unknown dist_metric"):
        objective_multi([enc], 0.7, 3.5, dist_metric="cramer")


def test_objective_multi_onset_fallback_keeps_loss_finite():
    """With w_onset>0 and no simulated onset (weak forces), the onset term must
    contribute the continuous fallback, not inf (Nelder-Mead cliff guard)."""
    enc = make_encounter(T=16)
    # v0=0: the sim ped never accelerates away -> empty sim onset pool
    loss = objective_multi([enc], 0.7, 0.0, w_ade=1.0, w_dist=0.0,
                           w_onset=1.0, onset_fallback=5.0)
    ade = objective_rollout_ade([enc], 0.7, 0.0)
    assert np.isfinite(loss)
    assert loss == pytest.approx(ade + 5.0)


def test_calibrate_with_objective_multi_recovers_params():
    """The grid+NM optimiser still recovers generating params through the
    multi-objective (the added EMD term is zero at the truth too)."""
    base = make_encounter(T=18)
    sigma_true, v0_true = 0.7, 3.0
    goals = _far_goals(base.ped_xy, base.ped_vel)
    sim_xy = simulate_encounter(base, sigma_true, v0_true)
    pseudo = Encounter(
        clip="pseudo", times=base.times, ego_xy=base.ego_xy, ego_psi=base.ego_psi,
        ego_vel=base.ego_vel, ped_xy=sim_xy, ped_vel=base.ped_vel,
        ped_ids=base.ped_ids, dt=base.dt, min_separation=0.0, goals=goals,
    )
    result = calibrate(
        lambda s, v: objective_multi([pseudo], s, v, w_ade=1.0, w_dist=1.0),
        grid_sigma=[0.3, 0.5, 0.7, 1.0, 1.5],
        grid_v0=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
    )
    assert result.grid_best == (sigma_true, v0_true)
    assert result.loss < 1e-3
