"""Tests for the smooth-braking redesign (S4 handover missions 2-3):

- Dedicated short-horizon brake candidates outside the Ti x tv grid
- Their held-at-rest padding to max_t (same-time collision soundness)
- The configurable last-resort emergency-stop deceleration
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.core.data_structures import EgoVehicleState, FrenetState
from src.planning.cubic_spline import CubicSpline2D
from src.planning.frenet_planner import (
    FrenetPlanner, BRAKE_T_MIN, BRAKE_T_STEP,
)
from src.simulation.integrated_simulator import IntegratedSimulator


def make_straight_planner(**kwargs):
    xs = np.linspace(0, 80, 30)
    spline = CubicSpline2D(xs.tolist(), [0.0] * 30)
    defaults = dict(max_speed=10.0, max_accel=2.0, max_curvature=0.2,
                    dt=0.1, d_road_w=0.5, max_road_width=3.0,
                    robot_radius=1.0, obstacle_radius=0.2,
                    min_t=4.0, max_t=5.0, d_t_s=1.39, n_s_sample=1)
    defaults.update(kwargs)
    return FrenetPlanner(spline, **defaults)


class TestBrakeCandidates:
    def test_ladder_generated_below_min_t(self):
        planner = make_straight_planner()
        state = FrenetState(s=5.0, s_d=5.0, s_dd=0.0, d=0.5, d_d=0.0, d_dd=0.0)
        cands = planner._generate_brake_candidates(state, target_speed=5.0)
        expected = len(np.arange(BRAKE_T_MIN, planner.min_t - 1e-9, BRAKE_T_STEP))
        assert len(cands) == expected > 0
        for fp in cands:
            # Padded to the full prediction horizon (same-time collision
            # checks must cover a pedestrian reaching the stop point late).
            assert fp.t[-1] == pytest.approx(planner.max_t)
            # Comes to rest and stays at rest.
            assert fp.s_d[-1] == pytest.approx(0.0, abs=1e-9)
            assert abs(fp.s[-1] - fp.s[-2]) < 1e-9
            # In-lane stop: lateral offset held at the current value.
            assert fp.d[-1] == pytest.approx(state.d, abs=1e-9)

    def test_stopping_distance_fills_short_range(self):
        """From 5 m/s the grid's shortest stop travels v*min_t/2 = 10 m; the
        ladder must offer stops well inside that range."""
        planner = make_straight_planner()
        state = FrenetState(s=0.0, s_d=5.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        cands = planner._generate_brake_candidates(state, target_speed=5.0)
        stop_distances = [fp.s[-1] - state.s for fp in cands]
        assert min(stop_distances) < 3.0
        assert max(stop_distances) < 5.0 * planner.min_t / 2.0

    def test_no_candidates_at_standstill(self):
        planner = make_straight_planner()
        state = FrenetState(s=5.0, s_d=0.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        assert planner._generate_brake_candidates(state, target_speed=5.0) == []

    def test_brake_candidates_respect_accel_gate(self):
        """The short stops peak at 1.5*v/T m/s^2; in NORMAL (max_accel 2.0)
        the hard ones must be rejected by the ordinary accel check so each
        fail-safe state only ever adopts stops it is allowed to."""
        planner = make_straight_planner(max_accel=2.0)
        state = FrenetState(s=5.0, s_d=5.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        cands = planner._generate_brake_candidates(state, target_speed=5.0)
        cands = planner._calc_global_paths(cands)
        result = planner._check_paths(cands, np.empty((0, 2)))
        # From 5 m/s, T=1.0 peaks at 7.5 m/s^2: must land in max_accel_error.
        assert len(result['max_accel_error']) > 0

    def test_plan_yields_short_stop_when_wall_inside_min_t_distance(self):
        """A blocking wall 6 m ahead: every Ti-grid candidate (>= 10 m to
        stop from 5 m/s) collides, so without the brake ladder planning
        fails outright and braking falls to the fail-safe slam. With it,
        plan() returns a smooth path stopping short of the wall."""
        planner = make_straight_planner(max_accel=8.0)
        ys = np.arange(-3.5, 3.6, 0.25)
        wall = np.stack([np.full_like(ys, 16.0), ys], axis=1)
        path = planner.plan(
            EgoVehicleState(x=10.0, y=0.0, yaw=0.0, v=5.0, a=0.0),
            wall, np.empty((0, 0, 2)), target_speed=5.0)
        assert path is not None
        assert path.v[-1] == pytest.approx(0.0, abs=0.05)
        # Stops short of the wall by at least the combined radius.
        assert max(path.x) < 16.0 - 1.0


class TestStopDistanceDirective:
    def test_filter_keeps_only_short_stops(self):
        """With max_stop_distance the lazy long-horizon stops are rejected
        and a short brake-ladder stop is selected instead — the planner can
        no longer procrastinate a commanded stop."""
        planner = make_straight_planner(max_accel=8.0)
        ego = EgoVehicleState(x=10.0, y=0.0, yaw=0.0, v=3.0, a=0.0)

        lazy = planner.plan(ego, np.empty((0, 2)), np.empty((0, 0, 2)),
                            target_speed=0.0)
        committed = planner.plan(ego, np.empty((0, 2)), np.empty((0, 0, 2)),
                                 target_speed=0.0, max_stop_distance=2.5)

        assert lazy is not None and committed is not None
        lazy_travel = lazy.s[-1] - lazy.s[0]
        committed_travel = committed.s[-1] - committed.s[0]
        # The unconstrained stop stretches over the whole horizon (~ v*T/2),
        # the committed one must fit the room.
        assert lazy_travel > 4.0
        assert committed_travel <= 2.5 + 1e-6
        assert abs(committed.v[-1]) < 0.15
        assert planner.last_check_stats['stop_distance_error'] > 0

    def test_infeasible_room_fails_plan(self):
        """No candidate can stop within 5 cm from 5 m/s: the directive must
        surface as a planning failure (fail-safe escalation), not silently
        pick a path that overruns the room."""
        planner = make_straight_planner(max_accel=8.0)
        ego = EgoVehicleState(x=10.0, y=0.0, yaw=0.0, v=5.0, a=0.0)
        path = planner.plan(ego, np.empty((0, 2)), np.empty((0, 0, 2)),
                            target_speed=0.0, max_stop_distance=0.05)
        assert path is None

    def test_hold_in_place_when_already_stopped(self):
        """Nearly stopped with a tiny room: the hold-in-place candidate
        passes, so waiting at standstill does not become a planning failure."""
        planner = make_straight_planner(max_accel=8.0)
        ego = EgoVehicleState(x=10.0, y=0.0, yaw=0.0, v=0.05, a=0.0)
        path = planner.plan(ego, np.empty((0, 2)), np.empty((0, 0, 2)),
                            target_speed=0.0, max_stop_distance=0.3)
        assert path is not None
        assert (path.s[-1] - path.s[0]) <= 0.3 + 1e-6


class TestForwardClearanceMetric:
    def _metrics(self, ped_positions, yaw=0.0):
        from src.core.data_structures import (
            compute_safety_metrics_static, PedestrianState)
        ego = EgoVehicleState(x=0.0, y=0.0, yaw=yaw, v=2.0, a=0.0)
        pos = np.asarray(ped_positions, dtype=float)
        peds = PedestrianState(
            positions=pos, velocities=np.zeros_like(pos),
            goals=np.zeros_like(pos), timestamp=0.0)
        return compute_safety_metrics_static(
            ego, peds, ego_radius=1.0, ped_radius=0.2)

    def test_clearance_ahead_ignores_pedestrians_behind(self):
        m = self._metrics([[-1.5, 0.0], [3.0, 0.0]])
        assert m['clearance'] == pytest.approx(1.5 - 1.2)
        assert m['clearance_ahead'] == pytest.approx(3.0 - 1.2)

    def test_clearance_ahead_inf_when_all_behind(self):
        m = self._metrics([[-1.5, 0.0], [-3.0, 1.0]])
        assert m['clearance_ahead'] == float('inf')
        assert np.isfinite(m['clearance'])

    def test_clearance_ahead_follows_heading(self):
        # Heading +y: the pedestrian at +x is no longer ahead.
        m = self._metrics([[3.0, 0.0], [0.0, 2.0]], yaw=np.pi / 2)
        assert m['clearance_ahead'] == pytest.approx(2.0 - 1.2)


class TestEmergencyDecelConfig:
    def make_sim(self, **config_kwargs):
        sim = IntegratedSimulator.__new__(IntegratedSimulator)
        defaults = dict(ego_max_accel=2.0, dt=0.1)
        defaults.update(config_kwargs)
        sim.config = SimpleNamespace(**defaults)
        sim.time = 0.0
        sim.ego_state = EgoVehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0, a=0.0)
        return sim

    def test_config_key_overrides_legacy_rate(self):
        # With a tiny clearance the rate saturates at the configured cap
        # (3.0), not the legacy 2x ego_max_accel.
        sim = self.make_sim(ego_emergency_decel=3.0)
        sim._last_clearance = 0.1
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state.v == pytest.approx(5.0 - 3.0 * 0.1)
        assert sim.ego_state.a == pytest.approx(-3.0)

    def test_none_falls_back_to_twice_max_accel(self):
        sim = self.make_sim(ego_emergency_decel=None)
        sim._last_clearance = 0.1
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state.v == pytest.approx(5.0 - 4.0 * 0.1)
        assert sim.ego_state.a == pytest.approx(-4.0)

    def test_adaptive_rate_uses_available_clearance(self):
        """v=5, clearance 5.2, margin 0.2 -> required = 25/(2*5.0) = 2.5:
        between the bounds, so the stop is exactly as firm as needed."""
        sim = self.make_sim(ego_emergency_decel=4.0)
        sim._last_clearance = 5.2
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state.a == pytest.approx(-2.5)

    def test_adaptive_rate_floors_at_max_accel_when_room_is_ample(self):
        """Plenty of room ahead: never brake softer than the planner's
        ordinary limit."""
        sim = self.make_sim(ego_emergency_decel=4.0)
        sim._last_clearance = 100.0
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state.a == pytest.approx(-2.0)

    def test_nonfinite_clearance_falls_back_to_max_rate(self):
        """Planning failed but nothing is ahead (clearance_ahead = inf):
        the threat is lateral or static and invisible to the forward
        clearance, so the stop must use the legacy maximum rate, not the
        ego_max_accel floor (half as firm)."""
        sim = self.make_sim(ego_emergency_decel=4.0)  # clearance unset -> inf
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state.a == pytest.approx(-4.0)

        sim = self.make_sim(ego_emergency_decel=4.0)
        sim._last_clearance = float('inf')
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state.a == pytest.approx(-4.0)

        # None cap -> legacy 2x ego_max_accel is the fallback rate too.
        sim = self.make_sim(ego_emergency_decel=None)
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state.a == pytest.approx(-4.0)

    def test_adaptive_rate_saturates_at_cap_when_room_is_gone(self):
        sim = self.make_sim(ego_emergency_decel=4.0)
        sim._last_clearance = 0.2
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state.a == pytest.approx(-4.0)
