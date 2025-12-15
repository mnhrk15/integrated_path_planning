"""Core module for fundamental data structures and utilities."""

from .data_structures import (
    EgoVehicleState,
    PedestrianState,
    FrenetState,
    FrenetPath,
    ObstacleSet,
    SimulationResult,
)
from .coordinate_converter import (
    CartesianFrenetConverter,
    CoordinateConverter,
    normalize_angle,
)

__all__ = [
    'EgoVehicleState',
    'PedestrianState',
    'FrenetState',
    'FrenetPath',
    'ObstacleSet',
    'SimulationResult',
    'CartesianFrenetConverter',
    'CoordinateConverter',
    'normalize_angle',
]
