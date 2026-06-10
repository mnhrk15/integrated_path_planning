"""Ego vehicle footprint models for collision evaluation.

The legacy model is a single circle of radius ``ego_radius`` centred on the
vehicle position. ``EgoFootprint`` covers the vehicle rectangle
(``vehicle_length`` x ``vehicle_width``, centred on the position as drawn by
the animator) with ``n_circles`` equal circles placed along the heading axis,
so that the union of circles contains the full rectangle.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EgoFootprint:
    """Multi-circle cover of the ego vehicle rectangle.

    Attributes:
        offsets: Longitudinal circle-centre offsets from the vehicle centre [m]
        radius: Common circle radius [m]
    """
    offsets: np.ndarray
    radius: float

    @classmethod
    def multi_circle(cls, vehicle_length: float, vehicle_width: float,
                     n_circles: int) -> "EgoFootprint":
        """Cover an L x W rectangle with n equal circles along its long axis.

        The rectangle is split into n segments of length L/n; each circle is
        the circumscribed circle of one segment, giving the tightest equal-
        radius cover with centres on the axis:
        radius = sqrt((L/(2n))^2 + (W/2)^2).
        """
        if n_circles < 1:
            raise ValueError(f"n_circles must be >= 1, got {n_circles}")
        seg = vehicle_length / n_circles
        offsets = -vehicle_length / 2 + seg / 2 + seg * np.arange(n_circles)
        radius = float(np.hypot(seg / 2, vehicle_width / 2))
        return cls(offsets=offsets, radius=radius)

    def circle_centers(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Circle centres [n_circles, 2] for a vehicle pose (x, y, yaw)."""
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        return np.array([x, y]) + self.offsets[:, None] * direction


def rectangle_surface_distance(points: np.ndarray, length: float,
                               width: float) -> np.ndarray:
    """Distance from points (vehicle frame, [n, 2]) to an L x W rectangle.

    Zero for points inside the rectangle; the rectangle is centred on the
    origin with its long axis along +x.
    """
    dx = np.maximum(np.abs(points[:, 0]) - length / 2, 0.0)
    dy = np.maximum(np.abs(points[:, 1]) - width / 2, 0.0)
    return np.hypot(dx, dy)


def world_to_vehicle_frame(points: np.ndarray, x: float, y: float,
                           yaw: float) -> np.ndarray:
    """Transform world-frame points [n, 2] into the vehicle frame at (x, y, yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    return (points - np.array([x, y])) @ rot.T


def footprint_from_config(config) -> "EgoFootprint | None":
    """Build the ego footprint from a SimulationConfig.

    Returns None for the legacy single-circle mode, in which callers fall back
    to the ``ego_radius`` centre circle.
    """
    if config.ego_footprint == "circle":
        return None
    return EgoFootprint.multi_circle(
        config.vehicle_length, config.vehicle_width, config.ego_footprint_n_circles
    )
