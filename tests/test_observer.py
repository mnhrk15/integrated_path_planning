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
