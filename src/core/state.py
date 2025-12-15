"""Legacy-compatible state dataclasses for tests and simple mocks."""

from dataclasses import dataclass


@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    v: float
    a: float
    timestamp: float = 0.0


@dataclass
class PedestrianState:
    x: float
    y: float
    vx: float
    vy: float
    gx: float
    gy: float
    timestamp: float = 0.0


__all__ = ["EgoState", "PedestrianState"]
