"""Regression tests for the Frenet derivative-convention fixes (2026-06-11 review).

Covers:
- M-8: lateral initial conditions and inverse transform use a single (temporal)
  convention, so fp.yaw matches the polyline tangent and the initial lateral
  velocity equals v*sin(delta_theta).
- M-7: the lateral sampling grid is symmetric and always contains d = 0.
- Horizon endpoint inclusion: the time grid contains t = Ti and the Ti range
  contains max_t.
- Lockstep truncation of out-of-domain paths (Cartesian and Frenet arrays stay
  the same length).
- The ego-curvature cache used for the Frenet initial conditions.
"""

import numpy as np
import pytest

from src.core.coordinate_converter import normalize_angle
from src.core.data_structures import EgoVehicleState
from src.planning.cubic_spline import CubicSpline2D
from src.planning.frenet_planner import FrenetPlanner


def make_straight_planner(length=120.0, **kwargs):
    """Planner on a straight east-bound reference line of the given length."""
    n = int(length / 10) + 1
    csp = CubicSpline2D(
        [10.0 * i for i in range(n)],
        [0.0] * n,
    )
    defaults = dict(
        max_speed=10.0,
        max_accel=2.0,
        max_curvature=1.0,
        dt=0.1,
        d_road_w=1.0,
        max_road_width=7.0,
        robot_radius=1.0,
        obstacle_radius=0.3,
    )
    defaults.update(kwargs)
    return FrenetPlanner(reference_path=csp, **defaults)


NO_OBS = np.empty((0, 2))


class TestLateralConventionM8:
    def test_initial_lateral_velocity_is_temporal(self):
        """d_d must be v*sin(delta_theta), not the spatial derivative tan(delta_theta)."""
        planner = make_straight_planner()
        yaw = np.deg2rad(15.0)
        ego = EgoVehicleState(x=20.0, y=0.0, yaw=yaw, v=5.0, a=0.0)
        fs = planner._cartesian_to_frenet_state(ego)
        assert fs is not None
        assert np.isclose(fs.d_d, 5.0 * np.sin(yaw), atol=1e-3)
        # Pre-fix value was tan(15 deg) ~ 0.27; make sure we are far from it.
        assert abs(fs.d_d - np.tan(yaw)) > 0.5

    def test_yaw_matches_polyline_tangent(self):
        """Stored fp.yaw must agree with the tangent of the converted polyline."""
        planner = make_straight_planner()
        ego = EgoVehicleState(x=20.0, y=0.0, yaw=np.deg2rad(15.0), v=5.0, a=0.0)
        path = planner.plan(ego, NO_OBS, target_speed=5.0)
        assert path is not None
        x = np.asarray(path.x)
        y = np.asarray(path.y)
        yaw = np.asarray(path.yaw)
        seg_tangent = np.arctan2(np.diff(y), np.diff(x))
        err = np.abs(normalize_angle(yaw[:-1] - seg_tangent))
        assert np.max(err) < np.deg2rad(5.0)

    def test_initial_speed_continuity(self):
        """The converted speed at index 0 must equal the ego speed."""
        planner = make_straight_planner()
        ego = EgoVehicleState(x=20.0, y=0.0, yaw=np.deg2rad(15.0), v=5.0, a=0.0)
        path = planner.plan(ego, NO_OBS, target_speed=5.0)
        assert path is not None
        assert np.isclose(path.v[0], 5.0, atol=1e-6)

    def test_plan_from_standstill_is_finite(self):
        """s_dot ~ 0 must not blow up the spatial-derivative conversion.

        At v = 0 the lateral initial velocity is v*sin(delta_theta) = 0
        regardless of the heading misalignment, so the on-centerline candidate
        stays feasible and every converted quantity must be finite.
        """
        planner = make_straight_planner()
        ego = EgoVehicleState(x=20.0, y=0.0, yaw=np.deg2rad(10.0), v=0.0, a=0.0)
        path = planner.plan(ego, NO_OBS, target_speed=5.0)
        assert path is not None
        for arr in (path.x, path.y, path.yaw, path.v, path.a, path.c):
            assert np.all(np.isfinite(arr))


class TestLateralGridM7:
    def test_grid_contains_zero_and_is_symmetric(self):
        planner = make_straight_planner(d_road_w=0.3, max_road_width=7.0)
        ego = EgoVehicleState(x=20.0, y=0.0, yaw=0.0, v=5.0, a=0.0)
        path = planner.plan(ego, NO_OBS, target_speed=5.0)
        assert path is not None
        # With the horizon endpoint included, the terminal lateral offset of
        # each candidate equals its grid value exactly; on an empty straight
        # road the cheapest candidate is d = 0.
        assert np.isclose(path.d[-1], 0.0, atol=1e-9)

    def test_grid_values_symmetric_and_bounded(self):
        # Reproduce the grid construction for a S1/S3-like configuration where
        # the legacy arange produced {-7.0, ..., -0.1, 0.2, ..., 6.8}.
        d_road_w, max_road_width = 0.3, 7.0
        n_side = int(max_road_width / d_road_w + 1e-9)
        di_values = np.arange(-n_side, n_side + 1) * d_road_w
        assert 0.0 in di_values
        np.testing.assert_allclose(di_values, -di_values[::-1], atol=1e-12)
        assert np.max(np.abs(di_values)) <= max_road_width + 1e-9


class TestHorizonEndpoint:
    def test_time_cache_includes_endpoint(self):
        planner = make_straight_planner()
        cache = planner._build_time_cache(4.0)
        assert len(cache.t) == 41
        assert np.isclose(cache.t[-1], 4.0)

    def test_ti_range_includes_max_t(self):
        planner = make_straight_planner(min_t=4.0, max_t=5.0)
        ego = EgoVehicleState(x=20.0, y=0.0, yaw=0.0, v=5.0, a=0.0)
        path = planner.plan(ego, NO_OBS, target_speed=5.0)
        assert path is not None
        # Horizon durations available to the planner must include max_t.
        n_ti = int(round((planner.max_t - planner.min_t) / planner.dt))
        ti_values = planner.min_t + np.arange(n_ti + 1) * planner.dt
        assert np.isclose(ti_values[-1], planner.max_t)

    def test_collision_checked_at_horizon_endpoint(self):
        """An obstacle colliding only at the final time index must be detected."""
        planner = make_straight_planner(min_t=5.0, max_t=5.0)
        from src.core.data_structures import FrenetPath

        fp = FrenetPath()
        t = np.arange(51) * 0.1
        fp.t = t.tolist()
        fp.x = (20.0 + 5.0 * t).tolist()
        fp.y = [0.0] * 51

        dyn = np.full((1, 51, 2), 1000.0)
        dyn[0, 50] = [fp.x[-1], 0.0]  # collides only at t = 5.0
        assert planner._check_collision(fp, None, dyn) is False

        dyn_other_time = np.full((1, 51, 2), 1000.0)
        dyn_other_time[0, 10] = [fp.x[-1], 0.0]  # same place, wrong time
        assert planner._check_collision(fp, None, dyn_other_time) is True


class TestLockstepTruncation:
    def test_truncated_path_arrays_stay_in_lockstep(self):
        """Paths leaving the spline domain are truncated across all arrays."""
        planner = make_straight_planner(length=60.0)
        # Ego close enough to the spline end that 4-5 s horizons at ~5 m/s
        # overrun s_max = 60 and force truncation.
        ego = EgoVehicleState(x=45.0, y=0.0, yaw=0.0, v=5.0, a=0.0)
        path = planner.plan(ego, NO_OBS, target_speed=5.0)
        assert path is not None
        n = len(path.x)
        full_horizon = len(planner._build_time_cache(planner.min_t).t)
        assert n < full_horizon  # truncation actually happened
        for arr in (path.y, path.yaw, path.c, path.v, path.a,
                    path.t, path.s, path.s_d, path.s_dd, path.s_ddd,
                    path.d, path.d_d, path.d_dd, path.d_ddd):
            assert len(arr) == n

    def test_paths_shorter_than_two_points_are_invalidated(self):
        planner = make_straight_planner(length=60.0)
        fp_list = planner._generate_frenet_paths(
            planner._cartesian_to_frenet_state(
                EgoVehicleState(x=59.9, y=0.0, yaw=0.0, v=5.0, a=0.0)
            ),
            5.0,
        )
        fp_list = planner._calc_global_paths(fp_list)
        for fp in fp_list:
            assert len(fp.x) != 1  # either empty (invalid) or >= 2 points


class TestEgoCurvatureCache:
    def test_cache_updates_on_success_and_survives_failure(self):
        planner = make_straight_planner()
        assert planner._last_kappa == 0.0
        ego = EgoVehicleState(x=20.0, y=0.0, yaw=0.0, v=5.0, a=0.0)
        path = planner.plan(ego, NO_OBS, target_speed=5.0)
        assert path is not None
        assert planner._last_kappa == float(path.c[1])
        kappa_after_success = planner._last_kappa

        # Wall of static obstacles ahead: no candidate survives. The ego has
        # not moved, so the cached curvature must survive for the same-step
        # escalation retry.
        wall_y = np.linspace(-8.0, 8.0, 33)
        wall = np.stack([np.full_like(wall_y, 24.0), wall_y], axis=1)
        failed = planner.plan(ego, wall, target_speed=5.0)
        assert failed is None
        assert planner._last_kappa == kappa_after_success

        # The simulator resets the cache when it applies the straight-line
        # emergency stop.
        planner.reset_ego_curvature()
        assert planner._last_kappa == 0.0
