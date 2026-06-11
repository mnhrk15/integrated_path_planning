"""Regression tests for planner/simulator robustness guards.

Covers three fixes from the 2026-06-11 code review:
- Emergency stop integrates vehicle motion (braking distance > 0) and does not
  alias the recorded history (C-3).
- Frenet candidates crossing the 1 - kappa_ref * d <= eps singularity are
  rejected (M-9).
- Geometrically discontinuous candidate paths (teleport jumps from exploding
  Frenet initial conditions) are rejected.
"""

from types import SimpleNamespace

import numpy as np

from src.core.data_structures import EgoVehicleState, FrenetPath
from src.planning.cubic_spline import CubicSpline2D
from src.planning.frenet_planner import FrenetPlanner
from src.simulation.integrated_simulator import IntegratedSimulator


def make_emergency_sim(v=5.0, yaw=0.0):
    sim = IntegratedSimulator.__new__(IntegratedSimulator)
    sim.config = SimpleNamespace(ego_max_accel=2.0, dt=0.1)
    sim.time = 1.0
    sim.ego_state = EgoVehicleState(x=10.0, y=-3.0, yaw=yaw, v=v, a=0.0)
    return sim


class TestEmergencyStopKinematics:
    def test_position_integrates_during_braking(self):
        sim = make_emergency_sim(v=5.0, yaw=0.0)
        sim._apply_emergency_stop(old_a=0.0)
        np.testing.assert_allclose(sim.ego_state.x, 10.5, atol=1e-12)
        np.testing.assert_allclose(sim.ego_state.y, -3.0, atol=1e-12)
        np.testing.assert_allclose(sim.ego_state.v, 5.0 - 4.0 * 0.1, atol=1e-12)

    def test_position_integrates_along_heading(self):
        yaw = np.pi / 2
        sim = make_emergency_sim(v=2.0, yaw=yaw)
        sim._apply_emergency_stop(old_a=0.0)
        np.testing.assert_allclose(sim.ego_state.x, 10.0, atol=1e-9)
        np.testing.assert_allclose(sim.ego_state.y, -3.0 + 0.2, atol=1e-9)

    def test_braking_distance_over_full_stop(self):
        """v0=4, dec=4 m/s2 -> stop distance ~ v0^2/(2*dec) = 2.0 m (old code: 0 m)."""
        sim = make_emergency_sim(v=4.0)
        start_x = sim.ego_state.x
        for _ in range(20):
            sim._apply_emergency_stop(old_a=sim.ego_state.a)
        assert sim.ego_state.v == 0.0
        travelled = sim.ego_state.x - start_x
        assert 1.5 < travelled < 2.5

    def test_no_history_aliasing(self):
        """The state object must be replaced, not mutated in place."""
        sim = make_emergency_sim(v=5.0)
        recorded = sim.ego_state          # simulates the reference held by history
        v_before = recorded.v
        x_before = recorded.x
        sim._apply_emergency_stop(old_a=0.0)
        assert sim.ego_state is not recorded
        assert recorded.v == v_before
        assert recorded.x == x_before


def make_arc_planner(radius=5.0, span=1.5 * np.pi, **kwargs):
    """Reference path: arc with curvature 1/radius, long enough (default
    ~23.6 m) that 4-5 s candidate horizons stay inside the spline domain."""
    theta = np.linspace(0.0, span, 60)
    xs = radius * np.sin(theta)
    ys = radius * (1.0 - np.cos(theta))
    spline = CubicSpline2D(xs.tolist(), ys.tolist())
    defaults = dict(max_speed=13.9, max_accel=8.0, max_curvature=10.0,
                    dt=0.1, d_road_w=0.5, max_road_width=7.0,
                    robot_radius=1.0, min_t=4.0, max_t=5.0,
                    d_t_s=1.39, n_s_sample=1)
    defaults.update(kwargs)
    return FrenetPlanner(spline, **defaults)


class TestSingularityGuard:
    def test_candidates_beyond_curvature_center_are_invalidated(self):
        """On a radius-5 left-turn arc (kappa = +1/5), candidates whose lateral
        offset reaches d >= (1-eps)*5 = +4.75 m cross the curvature center
        while still inside the spline domain: their converted geometry is
        degenerate (pi-flipped heading, near-zero speed) and the whole
        candidate must be dropped."""
        planner = make_arc_planner(radius=5.0)
        from src.core.data_structures import FrenetState
        fstate = FrenetState(s=2.0, s_d=3.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        paths = planner._generate_frenet_paths(fstate, target_speed=3.0)
        paths = planner._calc_global_paths(paths)
        # Identify candidates that hit the singular band INSIDE the spline
        # domain (s within bounds) — these must be emptied. Use Frenet-side
        # arrays which are never truncated.
        s_max = planner.csp.s[-1]
        singular_present = []
        for fp in paths:
            d_arr = np.asarray(fp.d)
            s_arr = np.asarray(fp.s)
            in_domain = s_arr <= s_max
            if np.any((d_arr[in_domain] / 5.0) >= 1.0 - 0.05):
                singular_present.append(fp)
        assert len(singular_present) > 0, "test setup: no singular candidates generated"
        for fp in singular_present:
            assert len(fp.x) == 0, (
                f"singular candidate (max d={max(fp.d):.2f}) survived with "
                f"{len(fp.x)} converted points"
            )

    def test_out_of_domain_paths_are_truncated_not_dropped(self):
        """Candidates that merely overrun the spline end must keep their valid
        prefix (legacy truncation), NOT be invalidated by the singularity
        guard — otherwise the planner returns None ~one horizon before every
        goal and the vehicle can never finish a scenario."""
        xs = np.linspace(0, 60, 25)
        spline = CubicSpline2D(xs.tolist(), [0.0] * 25)
        planner = FrenetPlanner(spline, max_speed=10.0, max_accel=8.0,
                                max_curvature=10.0, dt=0.1, d_road_w=0.5,
                                max_road_width=7.0, robot_radius=1.0,
                                min_t=4.0, max_t=5.0, d_t_s=1.39, n_s_sample=1)
        # 6 m/s * 4 s = 24 m horizon from x=45 overruns the 60 m spline.
        path = planner.plan(
            EgoVehicleState(x=45.0, y=0.0, yaw=0.0, v=6.0, a=0.0),
            np.empty((0, 2)), np.empty((0, 0, 2)), target_speed=6.0,
        )
        assert path is not None, "plan() must still succeed near the spline end"
        assert len(path.x) >= 2

    def test_straight_reference_unaffected(self):
        """kappa=0: the guard must never trigger on a straight road."""
        xs = np.linspace(0, 50, 20)
        spline = CubicSpline2D(xs.tolist(), [0.0] * 20)
        planner = FrenetPlanner(spline, max_speed=13.9, max_accel=8.0,
                                max_curvature=10.0, dt=0.1, d_road_w=0.5,
                                max_road_width=7.0, robot_radius=1.0,
                                min_t=4.0, max_t=5.0, d_t_s=1.39, n_s_sample=1)
        path = planner.plan(
            EgoVehicleState(x=5.0, y=0.0, yaw=0.0, v=5.0, a=0.0),
            np.empty((0, 2)), np.empty((0, 0, 2)), target_speed=5.0,
        )
        assert path is not None and len(path.x) > 1


class TestContinuityGuard:
    def _path(self, xs, ys):
        n = len(xs)
        return FrenetPath(t=[0.1 * i for i in range(n)],
                          x=list(xs), y=list(ys),
                          yaw=[0.0] * n, v=[1.0] * n, a=[0.0] * n, c=[0.0] * n)

    def test_teleporting_path_rejected(self):
        planner = make_arc_planner()
        good = self._path([0.0, 0.1, 0.2], [0.0, 0.0, 0.0])
        teleport = self._path([0.0, 0.1, 5000.0], [0.0, 0.0, 0.0])
        result = planner._check_paths([good, teleport], np.empty((0, 2)))
        all_categorized = [fp for fps in result.values() for fp in fps]
        assert good in all_categorized
        assert teleport not in all_categorized

    def test_nonfinite_path_rejected(self):
        planner = make_arc_planner()
        bad = self._path([0.0, 0.1, 0.2], [0.0, 0.0, 0.0])
        bad.v = [1.0, float("nan"), 1.0]
        result = planner._check_paths([bad], np.empty((0, 2)))
        assert all(bad not in fps for fps in result.values())
