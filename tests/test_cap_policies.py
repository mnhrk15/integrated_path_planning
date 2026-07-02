"""Regression tests for the speed-cap policies (review F2, synthetic data).

The load-bearing guarantees, in order of importance:
1. The DEFAULT path is bit-for-bit unchanged: cap_policy omitted, cap_policy=None
   and cap_policy="median" are three spellings of the same code path, and
   capfit(m=1) reproduces it too (the permanent anchor for the m-sweep).
2. The decoupling shim actually decouples: "uncapped" lets a ped transiently
   exceed its cruise speed under repulsion (which "median" forbids), while the
   DesiredForce target (max_speeds outside the integration substep) stays at
   the cruise value.
3. "closedloop" reproduces the closed-loop regime: target = cap =
   max_speed_multiplier x cruise, i.e. peds walk ~1.3x the recorded speed.
"""
import numpy as np
import pytest

from src.simulation.calibration_harness import (
    CAP_POLICIES,
    UNCAPPED_SPEED,
    _build_ped_sim,
    fidelity_report,
    objective_one_step,
    objective_rollout_ade,
    simulate_encounter,
)
from tests.test_calibration_harness import make_encounter


CRUISE = 1.3  # make_encounter's constant recorded ped speed = its median cruise


def _reports_equal(a: dict, b: dict) -> bool:
    """Bit-level equality across every fidelity_report key (scalars + lists)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
        if va.shape != vb.shape:
            return False
        # NaN-positional equality (KS p-values can be NaN on tiny synthetic pools)
        if not np.array_equal(va, vb, equal_nan=True):
            return False
    return True


# --------------------------------------------------------------------------- #
# default-path preservation (the smoke gate's in-process counterpart)
# --------------------------------------------------------------------------- #
def test_cap_policy_none_median_and_omitted_are_bitwise_identical():
    enc = make_encounter(T=16)
    base = simulate_encounter(enc, 0.7, 3.5)
    assert np.array_equal(base, simulate_encounter(enc, 0.7, 3.5, cap_policy=None))
    assert np.array_equal(base, simulate_encounter(enc, 0.7, 3.5, cap_policy="median"))

    loss = objective_rollout_ade([enc], 0.7, 3.5)
    assert loss == objective_rollout_ade([enc], 0.7, 3.5, cap_policy=None)
    assert loss == objective_rollout_ade([enc], 0.7, 3.5, cap_policy="median")

    rep = fidelity_report([enc], 0.7, 3.5)
    assert _reports_equal(rep, fidelity_report([enc], 0.7, 3.5, cap_policy=None))
    assert _reports_equal(rep, fidelity_report([enc], 0.7, 3.5, cap_policy="median"))


def test_capfit_m1_equals_median_bitwise():
    """capfit(m=1) must equal the shim-less median path EXACTLY.

    This is the permanent anchor of the capfit m-sweep. m=1 is ALIASED to the
    median path inside _apply_cap_policy because pushing ``1.0*cruise`` through
    the shim is NOT bit-identical to the state setter's own
    ``multiplier*(cruise/multiplier)`` round trip -- with cruise=0.97 (below)
    the round trip loses 1 ulp, so a shim-based m=1 would silently diverge in
    the last bit. The alias makes the contract exact by construction; this
    test uses a NON-round-tripping cruise so any future de-aliasing regression
    is caught rather than masked by a lucky cruise value.
    """
    assert 1.3 * (0.97 / 1.3) != 0.97  # the round trip really is lossy here
    for ped_vel in ((0.0, 1.3), (0.0, 0.97)):
        enc = make_encounter(T=16, ped_vel=ped_vel)
        a = simulate_encounter(enc, 0.7, 3.5, cap_policy="median")
        b = simulate_encounter(enc, 0.7, 3.5, cap_policy="capfit", cap_multiplier=1.0)
        assert np.array_equal(a, b), ped_vel
        la = objective_rollout_ade([enc], 0.7, 3.5, cap_policy="median")
        lb = objective_rollout_ade([enc], 0.7, 3.5, cap_policy="capfit",
                                   cap_multiplier=1.0)
        assert la == lb, ped_vel


def test_capfit_above_one_engages_the_shim():
    """m>1 must actually decouple: under strong repulsion the capped speed can
    exceed the cruise target, which the median path forbids."""
    enc = make_encounter(T=16)
    sp_median = _frame_speeds(
        simulate_encounter(enc, 1.0, 6.0, cap_policy="median"), enc.dt)
    sp_m2 = _frame_speeds(
        simulate_encounter(enc, 1.0, 6.0, cap_policy="capfit",
                           cap_multiplier=2.0), enc.dt)
    assert sp_median.max() <= CRUISE + 1e-6
    assert sp_m2.max() > CRUISE + 0.05
    assert sp_m2.max() <= 2.0 * CRUISE + 1e-6  # ...but still capped at m*cruise


def test_unknown_cap_policy_raises():
    enc = make_encounter(T=8)
    with pytest.raises(ValueError, match="unknown cap_policy"):
        simulate_encounter(enc, 0.7, 3.5, cap_policy="bogus")
    assert "bogus" not in CAP_POLICIES


# --------------------------------------------------------------------------- #
# the decoupled policies do what they claim
# --------------------------------------------------------------------------- #
def _frame_speeds(sim_xy: np.ndarray, dt: float) -> np.ndarray:
    return np.linalg.norm(np.diff(sim_xy, axis=0), axis=2) / dt  # [T-1, N]


def test_uncapped_allows_transient_speed_above_cruise():
    """Under a strong repulsion the median policy pins the ped AT its cruise
    speed (cap = target = cruise), while uncapped lets it accelerate away."""
    enc = make_encounter(T=16)  # ego passes right by the crossing ped
    strong = dict(sigma=1.0, v0=6.0)
    sp_median = _frame_speeds(
        simulate_encounter(enc, cap_policy="median", **strong), enc.dt)
    sp_uncapped = _frame_speeds(
        simulate_encounter(enc, cap_policy="uncapped", **strong), enc.dt)
    # median: every frame displacement is capped at cruise * dt (small numeric
    # slack only); uncapped: the evasion must exceed the cruise speed clearly.
    assert sp_median.max() <= CRUISE + 1e-6
    assert sp_uncapped.max() > CRUISE + 0.05
    assert sp_uncapped.max() < UNCAPPED_SPEED  # sanity: not a numeric blow-up


def test_uncapped_keeps_cruise_speed_when_no_repulsion():
    """Decoupling must not touch the walking-speed target: with v0=0 the
    uncapped ped still walks at the recorded cruise, not at UNCAPPED_SPEED."""
    enc = make_encounter(T=16, ped_vel=(0.0, CRUISE))
    sim_xy = simulate_encounter(enc, sigma=0.7, v0=0.0, cap_policy="uncapped")
    sp = _frame_speeds(sim_xy, enc.dt)
    assert np.median(sp) == pytest.approx(CRUISE, abs=0.1)


def test_closedloop_cruises_at_multiplier_times_cruise():
    """closedloop reproduces the deployment regime: target = cap = 1.3 x cruise,
    so with v0=0 peds walk ~30% faster than recorded (the regime mismatch the
    arm exists to quantify -- the mirror of
    test_no_ego_control_walks_straight_at_recorded_speed)."""
    enc = make_encounter(T=16, ped_vel=(0.0, CRUISE))
    sim_xy = simulate_encounter(enc, sigma=0.7, v0=0.0, cap_policy="closedloop")
    sp = _frame_speeds(sim_xy, enc.dt)
    assert np.median(sp) == pytest.approx(1.3 * CRUISE, abs=0.15)


def test_substep_max_speeds_restored_after_each_step():
    """After every integration step the state setter must have restored
    max_speeds to the cruise target, so the NEXT compute_forces sees the
    unmodified DesiredForce target (the shim only swaps the cap in).

    Tolerance note: the setter restores ``1.3 * (cruise / 1.3)``, which is only
    bitwise-equal to cruise for round-tripping values like 1.3 -- asserting
    array_equal here would be a lucky-value over-claim (review finding), so
    the contract is 'restored to the cruise target up to the 1-ulp round trip'.
    """
    enc = make_encounter(T=8)
    ped_sim = _build_ped_sim(enc, 0.7, 3.5, ego_radius=1.0, agent_radius=0.3,
                             dt=0.1, cap_policy="uncapped")
    peds = ped_sim.sim.peds
    cruise_target = peds.max_speeds.copy()  # set by _set_cruise_speed
    assert np.allclose(cruise_target, CRUISE, atol=1e-9)
    from src.core.data_structures import EgoVehicleState
    ego = EgoVehicleState(x=-6.0, y=0.0, yaw=0.0, v=1.5, a=0.0)
    for _ in range(5):
        ped_sim.step(ego, n=1)
        assert np.allclose(peds.max_speeds, cruise_target, rtol=1e-12, atol=0.0)


def test_shim_copies_the_cap_array_defensively():
    """_install_cap_shim must snapshot the cap: a caller-side mutation of the
    array after installation (or an in-place edit by a future pysocialforce)
    must not change the effective cap mid-rollout."""
    from src.simulation.calibration_harness import _install_cap_shim

    enc = make_encounter(T=16)
    strong = dict(sigma=1.0, v0=6.0)
    ped_sim = _build_ped_sim(enc, strong["sigma"], strong["v0"],
                             ego_radius=1.0, agent_radius=0.3, dt=0.1)
    n_peds = enc.ped_xy.shape[1]
    cap_arr = np.full(n_peds, 10.0)
    _install_cap_shim(ped_sim, cap_arr)
    cap_arr[:] = 0.0  # a zero cap would freeze the ped via capped_velocity
    from src.core.data_structures import EgoVehicleState
    for i in range(6):
        ego = EgoVehicleState(x=float(enc.ego_xy[i, 0]), y=float(enc.ego_xy[i, 1]),
                              yaw=0.0, v=1.5, a=0.0)
        ped_sim.step(ego, n=1)
    moved = np.linalg.norm(ped_sim.get_state().positions - enc.ped_xy[0], axis=1)
    assert moved.max() > 0.5  # peds kept walking => the shim used its own copy


def test_closedloop_seeds_desired_speed_from_cruise_not_frame0():
    """closedloop must seed initial_speeds from the (noise-robust) per-ped
    CRUISE estimate, not from the frame-0 recorded velocity: with a noisy slow
    first frame the regime is still 1.3 x cruise. (Review finding: the
    constant-velocity fixture could not discriminate this; a pysocialforce-
    default regression -- seeding from frame-0 speed -- stayed green.)"""
    enc = make_encounter(T=16, ped_vel=(0.0, CRUISE))
    enc.ped_vel[0] = (0.0, 0.4)  # noisy slow frame 0; median cruise stays 1.3
    ped_sim = _build_ped_sim(enc, 0.7, 0.0, ego_radius=1.0, agent_radius=0.3,
                             dt=0.1, cap_policy="closedloop")
    peds = ped_sim.sim.peds
    assert np.allclose(peds.initial_speeds, CRUISE, atol=1e-9)   # not 0.4
    assert np.allclose(peds.max_speeds, 1.3 * CRUISE, atol=1e-9)


def test_capfit_nonpositive_multiplier_raises():
    """cap = m x cruise with m <= 0 would freeze every ped via capped_velocity;
    fail fast instead of optimising nonsense."""
    enc = make_encounter(T=8)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="cap_multiplier"):
            simulate_encounter(enc, 0.7, 3.5, cap_policy="capfit",
                               cap_multiplier=bad)


def test_simulate_is_deterministic_under_cap_policies():
    enc = make_encounter(T=12)
    for policy in CAP_POLICIES:
        a = simulate_encounter(enc, 0.7, 3.5, cap_policy=policy, cap_multiplier=1.5)
        b = simulate_encounter(enc, 0.7, 3.5, cap_policy=policy, cap_multiplier=1.5)
        assert np.array_equal(a, b), policy


def test_objective_one_step_unaffected_by_cap_policy():
    """The teacher-forced diagnostic never calls peds.step, and the shim is
    instance-local: running an uncapped rollout first must not leak into a
    subsequent one-step evaluation."""
    enc = make_encounter(T=12)
    before = objective_one_step([enc], 0.7, 3.5)
    simulate_encounter(enc, 0.7, 3.5, cap_policy="uncapped")
    after = objective_one_step([enc], 0.7, 3.5)
    assert before == after
