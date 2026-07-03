"""Tests for src/simulation/realloop.py (RQ3 encounter -> closed loop)."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.datasets.vci_encounter import Encounter
from src.simulation.calibration_harness import (
    DEFAULT_AGENT_RADIUS,
    _cruise_speeds,
)
from src.simulation.realloop import (
    build_realloop_simulator,
    build_replay_source,
    build_sfm_source,
    dedupe_waypoints,
    encounter_config,
    encounter_eligibility,
    observation_seed_backcast,
    recorded_ego_deviation,
)

DT = 0.4


def make_encounter(n_frames=20, n_peds=2, ego_speed=2.5,
                   ped_speed=1.2) -> Encounter:
    """Straight-line ego along +x; peds walking +y off to the side."""
    t = np.arange(n_frames) * DT
    ego_xy = np.column_stack([ego_speed * t, np.zeros_like(t)])
    ped_xy = np.stack(
        [np.column_stack([np.full_like(t, 4.0 + 3.0 * i),
                          -4.0 + ped_speed * t]) for i in range(n_peds)],
        axis=1)  # [T, N, 2]
    ped_vel = np.zeros_like(ped_xy)
    ped_vel[..., 1] = ped_speed
    return Encounter(
        clip="synthetic", times=t, ego_xy=ego_xy,
        ego_psi=np.zeros(n_frames),
        ego_vel=np.full(n_frames, float(ego_speed)),
        ped_xy=ped_xy, ped_vel=ped_vel,
        ped_ids=np.arange(n_peds), dt=DT, min_separation=4.0,
    )


class TestDedupeWaypoints:
    def test_drops_stationary_cluster_keeps_endpoints(self):
        xy = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0],
                       [1.0, 0.0], [2.0, 0.0], [2.1, 0.0]])
        xs, ys = dedupe_waypoints(xy, min_ds=0.5)
        assert xs[0] == 0.0 and ys[0] == 0.0          # first point kept
        assert xs[-1] == 2.1                          # recorded end appended
        assert np.all(np.diff(xs) > 0)
        d = np.hypot(np.diff(xs), np.diff(ys))
        assert np.all(d[:-1] >= 0.5)                  # spacing floor (end exempt)

    def test_exactly_stationary_raises(self):
        """Identical points cannot form a spline (ds=0) -> fail fast."""
        xy = np.tile([[3.0, 3.0]], (10, 1))
        with pytest.raises(ValueError, match="stationary"):
            dedupe_waypoints(xy, min_ds=0.5)

    def test_jittered_stationary_is_degenerate_and_ineligible(self):
        """Sub-mm jitter around one point: the end-point append refuses a
        ~1e-4 m knot pair (would ill-condition the spline end), so dedupe
        degenerates -> eligibility reports it, never a raise mid-campaign."""
        enc = make_encounter()
        n = len(enc.times)
        jitter = 1e-4 * np.random.default_rng(0).normal(size=(n, 2))
        enc.ego_xy = np.tile([[3.0, 3.0]], (n, 1)) + jitter
        with pytest.raises(ValueError, match="stationary"):
            dedupe_waypoints(enc.ego_xy, min_ds=0.5)
        ok, reason = encounter_eligibility(enc)
        assert not ok
        assert "spline_degenerate" in reason

    def test_endpoint_within_a_millimetre_is_not_appended(self):
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0005, 0.0]])
        xs, ys = dedupe_waypoints(xy, min_ds=0.5)
        assert xs.tolist() == [0.0, 1.0]  # 0.5 mm end knot refused


class TestReplayInterpolation:
    def test_grid_nodes_match_recording_exactly(self):
        enc = make_encounter()
        src = build_replay_source(enc, sim_dt=0.1)
        n_q = 4 * (len(enc.times) - 1) + 1
        assert src.n_frames == n_q
        np.testing.assert_allclose(src.trajectories[::4], enc.ped_xy,
                                   atol=1e-12)

    def test_midpoints_are_linear(self):
        enc = make_encounter()
        src = build_replay_source(enc, sim_dt=0.1)
        mid = 0.5 * (enc.ped_xy[:-1] + enc.ped_xy[1:])
        np.testing.assert_allclose(src.trajectories[2::4], mid, atol=1e-12)

    def test_velocities_from_recorded_ped_vel_not_position_diff(self):
        enc = make_encounter()
        # Recorded velocity deliberately differs from the position slope.
        enc.ped_vel[..., 1] = 9.9
        src = build_replay_source(enc, sim_dt=0.1)
        np.testing.assert_allclose(src.velocities[..., 1], 9.9)

    def test_ids_forwarded(self):
        enc = make_encounter()
        src = build_replay_source(enc, sim_dt=0.1)
        np.testing.assert_array_equal(src.ids, enc.ped_ids)


class TestObservationSeedBackcast:
    def test_contract(self):
        enc = make_encounter()
        states = observation_seed_backcast(enc, obs_len=8, sgan_dt=0.4)
        assert len(states) == 8
        ts = [s.timestamp for s in states]
        np.testing.assert_allclose(np.diff(ts), 0.4, atol=1e-12)
        assert ts[-1] == 0.0
        # Last state IS recorded frame 0.
        np.testing.assert_allclose(states[-1].positions, enc.ped_xy[0])
        # Backcast: p0 + v0 * t (t negative).
        np.testing.assert_allclose(
            states[0].positions,
            enc.ped_xy[0] + enc.ped_vel[0] * ts[0], atol=1e-12)


class TestEligibility:
    def test_healthy_encounter_is_eligible(self):
        ok, reason = encounter_eligibility(make_encounter())
        assert ok and reason == ""

    def test_short_path_flagged(self):
        ok, reason = encounter_eligibility(
            make_encounter(n_frames=6, ego_speed=0.5))
        assert not ok
        assert "path_len" in reason

    def test_slow_ego_flagged(self):
        enc = make_encounter(ego_speed=0.1)
        ok, reason = encounter_eligibility(enc)
        assert not ok
        assert "median_speed" in reason

    def test_folded_path_flagged(self):
        enc = make_encounter()
        n = len(enc.times)
        # Out-and-back: net displacement << arc length.
        fold = np.concatenate([np.linspace(0, 10, n // 2),
                               np.linspace(10, 0.5, n - n // 2)])
        enc.ego_xy = np.column_stack([fold, np.zeros(n)])
        ok, reason = encounter_eligibility(enc)
        assert not ok
        assert "straightness" in reason


class TestEncounterConfig:
    def test_fields_anchored_to_recording(self):
        enc = make_encounter()
        config = encounter_config(enc)
        assert config.ped_radius == DEFAULT_AGENT_RADIUS == 0.30
        assert config.obstacle_radius == 0.30
        assert config.ego_target_speed == pytest.approx(
            float(np.nanmedian(enc.ego_vel)))
        # total_time is the recorded window padded by half a sim step, so the
        # runner's int(total_time/dt) lands EXACTLY on the recorded step
        # count for every float representation of the window length.
        n_steps_expected = 4 * (len(enc.times) - 1)
        assert int(config.total_time / config.dt) == n_steps_expected
        assert config.total_time == pytest.approx(
            float(enc.times[-1] - enc.times[0]) + 0.5 * config.dt, abs=1e-6)

    def test_total_time_survives_float_truncation_corner(self):
        """5/26 real encounters have window lengths 1 ulp below the exact dt
        multiple (e.g. 10.399999999999999): the naive int(window/dt) drops
        the final recorded frame. The padded total_time must not."""
        enc = make_encounter(n_frames=27)  # (T-1)*0.4 = 10.4 -> the bad case
        enc.times = enc.times + 0.0  # keep grid floats as produced
        # Reproduce the grid-accumulation artifact explicitly:
        enc.times = np.linspace(0.0, 10.399999999999999, 27)
        config = encounter_config(enc)
        assert int(config.total_time / config.dt) == 104  # not 103
        assert config.ego_initial_state == pytest.approx(
            [enc.ego_xy[0, 0], enc.ego_xy[0, 1], 0.0, 2.5, 0.0])
        assert config.ped_initial_states == []
        assert config.static_obstacles == []
        assert config.visualization_enabled is False
        # Waypoints are the deduped recorded path.
        xs, ys = dedupe_waypoints(enc.ego_xy)
        assert config.reference_waypoints_x == pytest.approx(xs.tolist())
        assert config.reference_waypoints_y == pytest.approx(ys.tolist())


class TestSfmSource:
    def test_median_cruise_pins_max_speeds_to_recorded_median(self):
        enc = make_encounter()
        sim = build_sfm_source(enc, sigma=1.0, v0=1.5,
                               speed_regime="median_cruise", sim_dt=0.1)
        expected = _cruise_speeds(enc.ped_vel)
        np.testing.assert_allclose(sim.sim.peds.max_speeds, expected)

    def test_initial_13x_regime(self):
        enc = make_encounter()
        sim = build_sfm_source(enc, sigma=1.0, v0=1.5,
                               speed_regime="initial_13x", sim_dt=0.1)
        peds = sim.sim.peds
        frame0 = np.hypot(enc.ped_vel[0, :, 0], enc.ped_vel[0, :, 1])
        multiplier = float(peds.max_speed_multiplier)
        np.testing.assert_allclose(peds.initial_speeds, frame0)
        np.testing.assert_allclose(peds.max_speeds, multiplier * frame0)

    def test_initial_13x_floors_stationary_peds(self):
        enc = make_encounter()
        enc.ped_vel[0] = 0.0  # recorded standing still at frame 0
        sim = build_sfm_source(enc, sigma=1.0, v0=1.5,
                               speed_regime="initial_13x", sim_dt=0.1)
        assert np.all(sim.sim.peds.initial_speeds > 0.0)

    def test_starts_at_recorded_frame_zero(self):
        enc = make_encounter()
        sim = build_sfm_source(enc, sigma=1.0, v0=1.5, sim_dt=0.1)
        np.testing.assert_allclose(sim.get_state().positions, enc.ped_xy[0])

    def test_unknown_regime_raises(self):
        with pytest.raises(ValueError, match="speed_regime"):
            build_sfm_source(make_encounter(), 1.0, 1.5, "freerun")


class TestBuildRealloopSimulator:
    def test_replay_cv_single(self):
        enc = make_encounter()
        sim, prov = build_realloop_simulator(enc, "replay", "cv", "single")
        assert prov["ped_kind"] == "replay"
        assert prov["speed_regime"] == "replay"
        assert prov["warmup_source"] == "backcast"
        assert prov["single_mode"].startswith("draw1_of_")
        assert sim.observer.is_ready
        assert sim.predictor.single_select == "draw"
        assert sim.distribution_aware_planning is False

    def test_sfm_robust_provenance(self):
        enc = make_encounter()
        sim, prov = build_realloop_simulator(
            enc, "sfm", "cv", "robust", sigma=1.168, v0=1.712)
        assert prov["sigma"] == pytest.approx(1.168)
        assert prov["v0"] == pytest.approx(1.712)
        assert prov["speed_regime"] == "median_cruise"
        assert prov["single_mode"].startswith("distribution_of_")
        assert sim.distribution_aware_planning is True
        # Effective SFM params reached the pedestrian simulator.
        assert sim.pedestrian_sim.ego_repulsion_sigma == pytest.approx(1.168)
        assert sim.pedestrian_sim.ego_repulsion_v0 == pytest.approx(1.712)

    def test_medoid_reference_mode(self):
        enc = make_encounter()
        sim, prov = build_realloop_simulator(enc, "replay", "cv", "medoid")
        assert prov["single_mode"].startswith("medoid_of_")
        assert sim.predictor.single_select == "medoid"

    def test_sfm_without_params_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            build_realloop_simulator(make_encounter(), "sfm", "cv", "single")

    def test_unknown_plan_mode_raises(self):
        with pytest.raises(ValueError, match="plan_mode"):
            build_realloop_simulator(make_encounter(), "replay", "cv", "safe")


class TestRecordedEgoDeviation:
    def _history_from(self, times, xy):
        return [SimpleNamespace(time=float(t),
                                ego_state=SimpleNamespace(x=float(p[0]),
                                                          y=float(p[1])))
                for t, p in zip(times, xy)]

    def test_perfect_tracking_gives_zero_dev_and_unit_progress(self):
        """History times are pre-increment (state at r.time + dt), so a
        perfectly tracking ego has its positions at t_rel and times t_rel-dt."""
        enc = make_encounter()
        t_rel = enc.times - enc.times[0]
        history = self._history_from(t_rel - 0.1, enc.ego_xy)
        out = recorded_ego_deviation(history, enc, dt=0.1)
        assert out["ego_dev_mean_m"] == pytest.approx(0.0, abs=1e-9)
        assert out["ego_dev_max_m"] == pytest.approx(0.0, abs=1e-9)
        assert out["progress"] == pytest.approx(1.0)

    def test_lagging_ego_shows_partial_progress(self):
        enc = make_encounter()
        t_rel = enc.times - enc.times[0]
        half = enc.ego_xy * 0.5  # covers half the recorded displacement
        history = self._history_from(t_rel - 0.1, half)
        out = recorded_ego_deviation(history, enc, dt=0.1)
        assert out["progress"] == pytest.approx(0.5)
        assert out["ego_dev_max_m"] > 0.0

    def test_empty_history(self):
        out = recorded_ego_deviation([], make_encounter())
        assert np.isnan(out["progress"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
