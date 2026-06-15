"""Tests for the multi-circle ego footprint (A-4: collision-geometry rework)."""
import numpy as np
import pytest

from src.core.footprint import EgoFootprint, footprint_from_config
from src.core.data_structures import (
    EgoVehicleState,
    PedestrianState,
    compute_safety_metrics_static,
)
from src.config import SimulationConfig


def _ego(x=0.0, y=0.0, yaw=0.0, v=5.0):
    return EgoVehicleState(x=x, y=y, yaw=yaw, v=v, a=0.0)


def _peds(positions, velocities=None):
    positions = np.array(positions, dtype=float)
    if velocities is None:
        velocities = np.zeros_like(positions)
    return PedestrianState(
        positions=positions,
        velocities=np.array(velocities, dtype=float),
        goals=np.zeros_like(positions),
        timestamp=0.0,
    )


class TestEgoFootprintGeometry:
    def test_three_circle_cover_dimensions(self):
        fp = EgoFootprint.multi_circle(4.5, 2.0, 3)
        assert np.allclose(fp.offsets, [-1.5, 0.0, 1.5])
        assert fp.radius == pytest.approx(np.hypot(0.75, 1.0))  # 1.25

    def test_cover_contains_rectangle_corners(self):
        # Every rectangle corner must lie inside at least one circle
        for n in (2, 3, 4):
            fp = EgoFootprint.multi_circle(4.5, 2.0, n)
            centers = fp.circle_centers(0.0, 0.0, 0.0)
            corners = np.array([[2.25, 1.0], [2.25, -1.0], [-2.25, -1.0], [-2.25, 1.0]])
            for corner in corners:
                dists = np.linalg.norm(centers - corner, axis=1)
                assert np.min(dists) <= fp.radius + 1e-12

    def test_circle_centers_follow_heading(self):
        fp = EgoFootprint.multi_circle(4.5, 2.0, 3)
        centers = fp.circle_centers(10.0, 5.0, np.pi / 2)
        assert np.allclose(centers, [[10.0, 3.5], [10.0, 5.0], [10.0, 6.5]])


class TestSafetyMetricsWithFootprint:
    def test_legacy_path_unchanged_without_footprint(self):
        ego = _ego()
        peds = _peds([[3.0, 0.0]], [[-1.0, 0.0]])
        m = compute_safety_metrics_static(ego, peds, ego_radius=1.0, ped_radius=0.2)
        assert m['min_distance'] == pytest.approx(3.0)
        assert not m['collision']
        assert m['clearance'] == pytest.approx(3.0 - 1.2)
        # Head-on approach at 6 m/s closing speed: (3.0 - 1.2) / 6.0
        assert m['ttc'] == pytest.approx(1.8 / 6.0)

    def test_multi_circle_detects_front_overlap_missed_by_single_circle(self):
        # Pedestrian 2.2 m ahead of the vehicle centre: outside the legacy
        # 1.2 m circle but inside the rectangle nose (half-length 2.25 m)
        ego = _ego()
        peds = _peds([[2.2, 0.0]])
        legacy = compute_safety_metrics_static(ego, peds, 1.0, 0.2)
        assert not legacy['collision']

        fp = EgoFootprint.multi_circle(4.5, 2.0, 3)
        multi = compute_safety_metrics_static(ego, peds, 1.0, 0.2, footprint=fp)
        # Front circle at (1.5, 0): distance 0.7 < 1.25 + 0.2
        assert multi['collision']
        assert multi['min_distance'] == pytest.approx(0.7)

    def test_multi_circle_no_false_positive_beside_vehicle(self):
        # Pedestrian 1.5 m beside the centre: clear of the 1.0 m-wide half-body
        # plus circle slack, at the same lateral distance as the legacy check
        ego = _ego()
        peds = _peds([[0.0, 1.5]])
        fp = EgoFootprint.multi_circle(4.5, 2.0, 3)
        m = compute_safety_metrics_static(ego, peds, 1.0, 0.2, footprint=fp)
        assert not m['collision']

    def test_multi_circle_respects_heading(self):
        # Same pedestrian, vehicle rotated 90 deg: the nose now points +y,
        # so a pedestrian at (2.2, 0) sits beside the vehicle, not ahead
        ego = _ego(yaw=np.pi / 2)
        peds = _peds([[2.2, 0.0]])
        fp = EgoFootprint.multi_circle(4.5, 2.0, 3)
        m = compute_safety_metrics_static(ego, peds, 1.0, 0.2, footprint=fp)
        assert not m['collision']

    def test_empty_pedestrians(self):
        ego = _ego()
        peds = _peds(np.empty((0, 2)))
        fp = EgoFootprint.multi_circle(4.5, 2.0, 3)
        m = compute_safety_metrics_static(ego, peds, 1.0, 0.2, footprint=fp)
        assert m['min_distance'] == float('inf')
        assert not m['collision']
        assert m['ttc'] == float('inf')


class TestPlannerFootprint:
    @staticmethod
    def _planner(**kwargs):
        from unittest.mock import MagicMock
        from src.planning.frenet_planner import FrenetPlanner
        from src.planning.cubic_spline import CubicSpline2D
        spline = MagicMock(spec=CubicSpline2D)
        spline.s = [0, 100.0]
        return FrenetPlanner(reference_path=spline, dt=0.1, **kwargs)

    @staticmethod
    def _straight_path(n=10, dt=0.1):
        from src.planning.frenet_planner import FrenetPath
        fp = FrenetPath()
        fp.x = [i * 1.0 for i in range(n)]
        fp.y = [0.0] * n
        fp.t = [i * dt for i in range(n)]
        fp.yaw = [0.0] * n
        return fp

    def test_geometry_expands_points_per_circle(self):
        fp_model = EgoFootprint.multi_circle(4.5, 2.0, 3)
        planner = self._planner(robot_radius=1.0, obstacle_radius=0.2, footprint=fp_model)
        path = self._straight_path(n=10)
        points, t, _, _, sq_rubicon, _ = planner._path_collision_geometry(path)
        assert points.shape == (30, 2)
        assert len(t) == 30
        assert sq_rubicon == pytest.approx((1.25 + 0.2) ** 2)

    def test_static_nose_collision_detected_only_with_footprint(self):
        # Path ends at x=9; obstacle at (10.4, 0) is 1.4 m beyond the last
        # path point: outside the legacy 1.2 m radius but inside the reach of
        # the nose circle (offset 1.5 m, threshold 1.45 m)
        obstacle = np.array([[10.4, 0.0]])
        path_kwargs = dict(robot_radius=1.0, obstacle_radius=0.2)

        legacy = self._planner(**path_kwargs)
        assert legacy._check_collision(self._straight_path(), obstacle)

        fp_model = EgoFootprint.multi_circle(4.5, 2.0, 3)
        multi = self._planner(**path_kwargs, footprint=fp_model)
        assert not multi._check_collision(self._straight_path(), obstacle)

    def test_distribution_check_uses_footprint(self):
        # Dynamic sample sitting at the nose for the whole horizon
        fp_model = EgoFootprint.multi_circle(4.5, 2.0, 3)
        planner = self._planner(robot_radius=1.0, obstacle_radius=0.2, footprint=fp_model)
        path = self._straight_path()
        distribution = np.full((1, 1, 10, 2), [10.4, 0.0])
        assert not planner._check_collision_distribution(
            path, np.empty((0, 2)), distribution, epsilon=0.0
        )

    def test_yaw_shorter_than_path_is_padded(self):
        fp_model = EgoFootprint.multi_circle(4.5, 2.0, 3)
        planner = self._planner(robot_radius=1.0, obstacle_radius=0.2, footprint=fp_model)
        path = self._straight_path(n=10)
        path.yaw = path.yaw[:-1]  # heading from diffs is one element short
        points, t, _, _, _, _ = planner._path_collision_geometry(path)
        assert points.shape == (30, 2)


class TestFootprintConfig:
    def test_default_config_is_legacy_circle(self):
        config = SimulationConfig()
        assert config.ego_footprint == "circle"
        assert footprint_from_config(config) is None

    def test_multi_circle_from_config(self):
        config = SimulationConfig(ego_footprint="multi_circle")
        fp = footprint_from_config(config)
        assert isinstance(fp, EgoFootprint)
        assert len(fp.offsets) == 3
        assert fp.radius == pytest.approx(1.25)
