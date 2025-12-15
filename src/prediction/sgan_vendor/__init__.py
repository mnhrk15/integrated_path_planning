"""Vendorized minimal SGAN components for inference only."""

from .models import TrajectoryGenerator
from .utils import relative_to_abs

__all__ = ["TrajectoryGenerator", "relative_to_abs"]
