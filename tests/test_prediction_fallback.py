"""Regression guards for the prediction-failure safety contract (review R3).

Two claim-critical behaviours of ``IntegratedSimulator._update_prediction`` have
no test today, so a refactor could silently break them with the suite green:

1. When prediction raises, the constant-velocity (CV) fallback must still emit a
   FULL planning-horizon timeseries (never flattened to a single step), with the
   current pedestrian positions prepended at t=0. A flattened fallback would
   corrupt the same-time-only dynamic-collision check.
2. A *persistent* failure must NOT silently degrade every step to CV (which would
   be scored as the CV method and invalidate the cv/lstm/sgan comparison): after
   5 consecutive failures the step must raise ``RuntimeError``; the counter resets
   on any success.

The CV fallback lives at integrated_simulator.py:478-496, the t=0 prepend at
:502-513, the counter reset at :466 and the 5-strike guard at :469-476.
"""
import numpy as np
import pytest

from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator


class _Predictor:
    """Stub for ``predict_single_best``: raise for the first ``n_raise`` calls,
    then return a valid (trajectory, distribution=None) tuple."""

    def __init__(self, n_raise: int, n_peds: int, dense_steps: int = 12):
        self.n_raise = n_raise
        self.n_peds = n_peds
        self.dense_steps = dense_steps
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.n_raise:
            raise RuntimeError("injected prediction failure")
        return np.zeros((self.n_peds, self.dense_steps, 2)), None


@pytest.fixture(scope="module")
def sim():
    # scenario_01 is the same fixture test_long_run_stability uses; construction
    # warms up the observer so predictions are active immediately.
    config = load_config("scenarios/scenario_01.yaml")
    config.total_time = 2.0  # keep the build light; we drive _update_prediction directly
    return IntegratedSimulator(config)


def test_cv_fallback_is_full_horizon_constant_velocity(sim, monkeypatch):
    """One failure -> full-horizon CV timeseries with current positions at t=0."""
    sim._consecutive_prediction_failures = 0
    ped_state = sim.pedestrian_sim.get_state()
    assert sim.observer.is_ready
    n_peds = ped_state.n_peds
    assert n_peds > 0
    pos = np.asarray(ped_state.positions, dtype=float)
    vel = np.asarray(ped_state.velocities, dtype=float)

    monkeypatch.setattr(sim.predictor, "predict_single_best",
                        _Predictor(n_raise=1, n_peds=n_peds))

    _, _, dyn, _, _ = sim._update_prediction(ped_state)

    dt = sim.config.dt
    horizon = int(getattr(sim.config, "max_t", 5.0) / dt)
    # NOT flattened: a full horizon plus the prepended current position at t=0.
    assert dyn.shape == (n_peds, horizon + 1, 2)
    # t=0 entry is exactly the current pedestrian positions.
    np.testing.assert_allclose(dyn[:, 0, :], pos, rtol=0, atol=1e-9)
    # Every step k is the constant-velocity extrapolation pos + vel * (k * dt).
    for k in range(dyn.shape[1]):
        np.testing.assert_allclose(dyn[:, k, :], pos + vel * (k * dt),
                                   rtol=0, atol=1e-6)


def test_failure_counter_increments_then_resets_on_success(sim, monkeypatch):
    sim._consecutive_prediction_failures = 0
    ped_state = sim.pedestrian_sim.get_state()
    monkeypatch.setattr(sim.predictor, "predict_single_best",
                        _Predictor(n_raise=2, n_peds=ped_state.n_peds))

    sim._update_prediction(ped_state)
    assert sim._consecutive_prediction_failures == 1
    sim._update_prediction(ped_state)
    assert sim._consecutive_prediction_failures == 2
    # 3rd call succeeds -> the counter must reset to 0.
    sim._update_prediction(ped_state)
    assert sim._consecutive_prediction_failures == 0


def test_fifth_consecutive_failure_raises_runtimeerror(sim, monkeypatch):
    sim._consecutive_prediction_failures = 0
    ped_state = sim.pedestrian_sim.get_state()
    monkeypatch.setattr(sim.predictor, "predict_single_best",
                        _Predictor(n_raise=999, n_peds=ped_state.n_peds))

    # 4 consecutive failures degrade to the CV fallback without raising.
    for _ in range(4):
        sim._update_prediction(ped_state)
    assert sim._consecutive_prediction_failures == 4
    # The 5th consecutive failure must escalate instead of silently using CV.
    with pytest.raises(RuntimeError):
        sim._update_prediction(ped_state)
