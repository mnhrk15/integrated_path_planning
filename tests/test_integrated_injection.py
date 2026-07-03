"""RQ3 injection contract: pedestrian_source / observation_seed kwargs.

The real-data-grounded closed loop (RQ3) injects a pre-built pedestrian
source (ReplayPedestrianSource or an externally configured SFM simulator)
into IntegratedSimulator and seeds the observer with a pre-computed history
instead of running the warmup loop. These tests pin the contract:

* with ``observation_seed`` the source is NOT advanced during __init__
  (t=0 stays anchored to the injected frame 0),
* the default path (no kwargs) still runs the original warmup,
* an injected replay + CV run is bit-identical across rebuilds.
"""

import numpy as np
import pytest

from src.config import SimulationConfig
from src.core.data_structures import PedestrianState
from src.simulation.integrated_simulator import (
    IntegratedSimulator,
    PedestrianSimulator,
)
from src.simulation.replay_source import ReplayPedestrianSource

DT = 0.1
SGAN_DT = 0.4
OBS_LEN = 8
TOTAL_TIME = 2.0


def _config(with_peds: bool = False) -> SimulationConfig:
    """Minimal straight-road CV config (no YAML, no visualization)."""
    config = SimulationConfig()
    config.dt = DT
    config.total_time = TOTAL_TIME
    config.obs_len = OBS_LEN
    config.pred_len = 8
    config.num_samples = 1
    config.prediction_method = 'cv'
    config.visualization_enabled = False
    config.reference_waypoints_x = [0.0, 10.0, 20.0, 30.0]
    config.reference_waypoints_y = [0.0, 0.0, 0.0, 0.0]
    config.ego_initial_state = [0.0, 0.0, 0.0, 2.0, 0.0]
    config.ego_target_speed = 2.0
    if with_peds:
        # One pedestrian far from the road (no interaction with the ego).
        config.ped_initial_states = [[5.0, 12.0, 1.0, 0.0, 30.0, 12.0]]
    return config


def _replay_source(n_frames: int = 40) -> ReplayPedestrianSource:
    """One pedestrian walking +x at y=10 (clear of the ego), dt=0.1."""
    t = np.arange(n_frames) * DT
    traj = np.stack([np.column_stack([2.0 + 1.2 * t, np.full_like(t, 10.0)])],
                    axis=1)  # [T, 1, 2]
    return ReplayPedestrianSource(traj, dt=DT)


def _sfm_source() -> PedestrianSimulator:
    return PedestrianSimulator(
        initial_states=np.array([[5.0, 12.0, 1.0, 0.0, 30.0, 12.0]]),
        dt=DT,
        ego_radius=1.0,
        v0_randomization=False,
    )


def _observation_seed(n_peds: int = 1) -> list:
    """Backcast contract: obs_len states, sgan_dt apart, last timestamp 0.0."""
    states = []
    for k in range(OBS_LEN):
        t = -(OBS_LEN - 1 - k) * SGAN_DT
        pos = np.tile([2.0 + 1.2 * t, 10.0], (n_peds, 1))
        vel = np.tile([1.2, 0.0], (n_peds, 1))
        goal = np.tile([50.0, 10.0], (n_peds, 1))
        states.append(PedestrianState(positions=pos, velocities=vel,
                                      goals=goal, timestamp=t))
    return states


def _signature(history):
    ego = np.array([[r.ego_state.x, r.ego_state.y, r.ego_state.yaw,
                     r.ego_state.v, r.ego_state.a] for r in history])
    ped = np.concatenate([r.ped_state.positions.ravel() for r in history
                          if r.ped_state is not None])
    return ego, ped


class TestSeededInjectionDoesNotAdvanceSource:
    def test_replay_source_stays_at_frame_zero(self):
        source = _replay_source()
        sim = IntegratedSimulator(_config(), pedestrian_source=source,
                                  observation_seed=_observation_seed())
        assert source._idx == 0
        assert source.time == 0.0
        assert sim.observer.is_ready
        assert sim.observer.last_sample_time == 0.0

    def test_sfm_source_clock_stays_at_zero(self):
        source = _sfm_source()
        pos_before = source.get_state().positions.copy()
        sim = IntegratedSimulator(_config(), pedestrian_source=source,
                                  observation_seed=_observation_seed())
        assert source.time == 0.0
        np.testing.assert_array_equal(source.get_state().positions, pos_before)
        assert sim.observer.is_ready

    def test_first_step_advances_source_by_one_frame(self):
        source = _replay_source()
        sim = IntegratedSimulator(_config(), pedestrian_source=source,
                                  observation_seed=_observation_seed())
        sim.step()
        assert source._idx == 1
        assert source.time == pytest.approx(DT)


class TestDefaultPathPreserved:
    def test_default_warmup_still_advances_internal_sfm(self):
        """No kwargs -> the original warmup runs (source advanced 3.2s)."""
        sim = IntegratedSimulator(_config(with_peds=True))
        warmup_time = OBS_LEN * SGAN_DT
        assert sim.pedestrian_sim.time == pytest.approx(warmup_time)
        assert sim.observer.is_ready

    def test_injected_source_without_seed_uses_warmup(self):
        """pedestrian_source alone (no observation_seed) keeps the warmup
        loop, which advances the injected source."""
        source = _replay_source(n_frames=80)
        IntegratedSimulator(_config(), pedestrian_source=source)
        warmup_steps = int(OBS_LEN * SGAN_DT / DT)
        assert source._idx == warmup_steps


class TestInjectedRunDeterminism:
    def _run(self):
        sim = IntegratedSimulator(_config(), pedestrian_source=_replay_source(),
                                  observation_seed=_observation_seed())
        return _signature(sim.run())

    def test_replay_cv_run_is_bit_identical_across_rebuilds(self):
        ego1, ped1 = self._run()
        ego2, ped2 = self._run()
        np.testing.assert_array_equal(ego1, ego2)
        np.testing.assert_array_equal(ped1, ped2)
        assert len(ego1) > 0

    def test_replayed_positions_follow_the_recording(self):
        """The ped positions consumed by the loop are the recorded frames
        (frame k at step k), not SFM output."""
        source = _replay_source()
        sim = IntegratedSimulator(_config(), pedestrian_source=source,
                                  observation_seed=_observation_seed())
        history = sim.run()
        for k, r in enumerate(history[:10], start=1):
            expected = source.trajectories[k]
            np.testing.assert_allclose(r.ped_state.positions, expected)


class TestSeedWithoutPedestrians:
    def test_observation_seed_without_any_ped_source_raises(self):
        """A seed with no pedestrian source would be silently discarded
        (review m3): fail fast instead."""
        with pytest.raises(ValueError, match="pedestrian source"):
            IntegratedSimulator(_config(with_peds=False),
                                observation_seed=_observation_seed())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
