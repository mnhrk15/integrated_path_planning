"""Tests for PedestrianObserver downsampling.

Regression tests for the elapsed-time double-counting bug: the observer must
sample at sgan_dt (0.4s) intervals when driven at the simulation dt (0.1s),
not at irregular 0.1-0.3s intervals.
"""

import numpy as np
import pytest

from src.core.data_structures import PedestrianState
from src.pedestrian.observer import PedestrianObserver


def make_state(t: float, speed: float = 1.2) -> PedestrianState:
    """Pedestrian walking at constant speed along +x."""
    pos = np.array([[speed * t, 0.0]])
    vel = np.array([[speed, 0.0]])
    goal = np.array([[100.0, 0.0]])
    return PedestrianState(positions=pos, velocities=vel, goals=goal, timestamp=t)


def drive(observer: PedestrianObserver, dt: float, n_steps: int, t0: float = 0.0):
    for k in range(1, n_steps + 1):
        observer.update(make_state(t0 + k * dt))


class TestObserverSamplingInterval:
    def test_samples_at_exact_sgan_dt_intervals(self):
        """Driven at dt=0.1, samples must be exactly 0.4s apart."""
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        drive(obs, dt=0.1, n_steps=200)

        intervals = np.diff(np.array(obs.timestamps))
        assert len(intervals) == 7  # deque holds obs_len=8 timestamps
        np.testing.assert_allclose(intervals, 0.4, atol=1e-9)

    def test_sample_timestamps_on_sgan_grid(self):
        """First sample at 0.4s, then 0.8, 1.2, ... (no early sampling)."""
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        sampled = []
        for k in range(1, 33):
            t = k * 0.1
            before = len(obs.history)
            obs.update(make_state(t))
            if len(obs.history) > before:
                sampled.append(t)
        np.testing.assert_allclose(
            sampled, [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2], atol=1e-9
        )

    def test_ready_after_warmup_step_count(self):
        """obs_len * sgan_dt / dt updates fill the history exactly."""
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        warmup_steps = int(8 * 0.4 / 0.1)
        drive(obs, dt=0.1, n_steps=warmup_steps - 1)
        assert not obs.is_ready
        obs.update(make_state(warmup_steps * 0.1))
        assert obs.is_ready

    def test_apparent_velocity_matches_true_speed(self):
        """Displacement between samples divided by sgan_dt recovers the true
        walking speed (the double-counting bug made this 0.3-0.9 m/s)."""
        speed = 1.2
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        drive(obs, dt=0.1, n_steps=100)

        traj = np.stack(list(obs.history), axis=0)  # (8, 1, 2)
        step_speeds = np.linalg.norm(np.diff(traj[:, 0, :], axis=0), axis=1) / 0.4
        np.testing.assert_allclose(step_speeds, speed, atol=1e-6)

    def test_no_float_drift_over_long_run(self):
        """1000 steps at dt=0.1: every sampling interval stays at 0.4s
        (guards the leftover-subtraction tolerance handling)."""
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        sampled = []
        for k in range(1, 1001):
            before = len(obs.history)
            obs.update(make_state(k * 0.1))
            if len(obs.history) > before:
                sampled.append(k * 0.1)
        intervals = np.diff(sampled)
        np.testing.assert_allclose(intervals, 0.4, atol=1e-9)

    def test_dt_equal_to_sgan_dt_samples_every_step(self):
        obs = PedestrianObserver(obs_len=8, dt=0.4, sgan_dt=0.4)
        drive(obs, dt=0.4, n_steps=8)
        assert obs.is_ready
        np.testing.assert_allclose(np.diff(np.array(obs.timestamps)), 0.4, atol=1e-9)

    def test_nonzero_start_time(self):
        """Timestamps that do not start at zero (warmup clock) still sample
        every sgan_dt of elapsed time."""
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        drive(obs, dt=0.1, n_steps=100, t0=3.2)
        intervals = np.diff(np.array(obs.timestamps))
        np.testing.assert_allclose(intervals, 0.4, atol=1e-9)

    def test_reset_clears_accumulator_and_reference_time(self):
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        drive(obs, dt=0.1, n_steps=10)
        obs.reset()
        assert len(obs.history) == 0
        assert obs.accumulated_time == 0.0
        assert obs._last_update_timestamp is None
        # After reset, sampling cadence starts over identically
        drive(obs, dt=0.1, n_steps=32, t0=5.0)
        assert obs.is_ready
        np.testing.assert_allclose(np.diff(np.array(obs.timestamps)), 0.4, atol=1e-9)


class TestObserverSeed:
    """RQ3: observer.seed() installs a pre-computed history directly."""

    def _seed_states(self, obs_len=8, sgan_dt=0.4):
        # Backcast contract: timestamps -(obs_len-1)*sgan_dt .. 0.0
        return [make_state(-(obs_len - 1 - k) * sgan_dt) for k in range(obs_len)]

    def test_seed_makes_observer_ready(self):
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        obs.seed(self._seed_states())
        assert obs.is_ready
        assert obs.last_sample_time == 0.0
        np.testing.assert_allclose(np.diff(np.array(obs.timestamps)), 0.4,
                                   atol=1e-9)

    def test_seed_then_run_samples_on_sgan_grid(self):
        """After seeding to t=0, run-time sampling lands at 0.4, 0.8, ..."""
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        obs.seed(self._seed_states())
        sampled = []
        for k in range(1, 13):
            t = k * 0.1
            before = list(obs.timestamps)
            obs.update(make_state(t))
            if list(obs.timestamps) != before:
                sampled.append(t)
        np.testing.assert_allclose(sampled, [0.4, 0.8, 1.2], atol=1e-9)

    def test_seed_too_short_raises(self):
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        with pytest.raises(ValueError, match="obs_len"):
            obs.seed(self._seed_states()[:5])

    def test_seed_non_increasing_timestamps_raise(self):
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        states = self._seed_states()
        states[3] = make_state(states[2].timestamp)  # duplicate timestamp
        with pytest.raises(ValueError, match="increasing"):
            obs.seed(states)

    def test_seed_replaces_prior_history(self):
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        drive(obs, dt=0.1, n_steps=100)  # arbitrary prior state
        obs.seed(self._seed_states())
        assert len(obs.history) == 8
        assert obs.last_sample_time == 0.0
        assert obs.accumulated_time == 0.0

    def test_seed_positions_are_copies(self):
        """Mutating the caller's arrays must not corrupt the history."""
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        states = self._seed_states()
        obs.seed(states)
        states[-1].positions[:] = 999.0
        assert not np.allclose(obs.history[-1], 999.0)

    def test_seed_wrong_spacing_raises(self):
        """Seed states must be exactly sgan_dt apart (each is one history
        frame); a 0.1s-spaced seed would silently corrupt CV velocity
        estimates (review m4)."""
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        states = [make_state(-(8 - 1 - k) * 0.1) for k in range(8)]
        with pytest.raises(ValueError, match="sgan_dt"):
            obs.seed(states)

    def test_seed_not_ending_at_zero_raises(self):
        obs = PedestrianObserver(obs_len=8, dt=0.1, sgan_dt=0.4)
        states = [make_state(0.4 * (k + 1)) for k in range(8)]  # ends at 3.2
        with pytest.raises(ValueError, match="0.0"):
            obs.seed(states)
