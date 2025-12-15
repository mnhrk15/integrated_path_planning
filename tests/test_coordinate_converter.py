"""Tests for coordinate converter."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.coordinate_converter import normalize_angle, CartesianFrenetConverter
from src.planning.cubic_spline import CubicSpline2D


def test_normalize_angle():
    """Test angle normalization."""
    assert abs(normalize_angle(0.0)) < 1e-6
    assert abs(normalize_angle(2 * np.pi)) < 1e-6
    assert abs(normalize_angle(np.pi) - np.pi) < 1e-6
    assert abs(normalize_angle(-np.pi) - (-np.pi)) < 1e-6
    assert abs(normalize_angle(3 * np.pi) - (-np.pi)) < 1e-6


def test_cubic_spline_creation():
    """Test cubic spline creation."""
    x = [0.0, 10.0, 20.0, 30.0]
    y = [0.0, 5.0, 0.0, -5.0]
    
    csp = CubicSpline2D(x, y)
    
    # Test position calculation
    px, py = csp.calc_position(0.0)
    assert px == x[0]
    assert py == y[0]
    
    # Test yaw calculation
    yaw = csp.calc_yaw(0.0)
    assert yaw is not None


def test_coordinate_converter():
    """Test coordinate converter."""
    # Create simple straight path
    x = [0.0, 10.0, 20.0]
    y = [0.0, 0.0, 0.0]
    csp = CubicSpline2D(x, y)
    
    from src.core.coordinate_converter import CoordinateConverter
    converter = CoordinateConverter(csp)
    
    # Test obstacle conversion
    ped_traj = np.array([
        [[5.0, 2.0], [6.0, 2.0], [7.0, 2.0]],  # Pedestrian 1
        [[15.0, -2.0], [16.0, -2.0], [17.0, -2.0]]  # Pedestrian 2
    ])  # Shape: (2, 3, 2)
    
    # Test pass through
    traj = np.zeros((1, 10, 2))
    res = converter.pass_through_obstacle(traj)
    assert np.allclose(traj, res)
    
    # Test valid input shape
    with pytest.raises(ValueError):
        converter.pass_through_obstacle(np.zeros((10, 2)))  # Wrong shape


def test_nearest_point_at_boundaries():
    """Nearest point should resolve endpoints correctly."""
    x = [0.0, 10.0, 20.0]
    y = [0.0, 0.0, 0.0]
    csp = CubicSpline2D(x, y)
    from src.core.coordinate_converter import CoordinateConverter
    converter = CoordinateConverter(csp)

    rs, rx, ry, rtheta, rkappa, rdkappa = converter.find_nearest_point_on_path(0.0, 5.0)
    assert abs(rs - 0.0) < 1e-2
    rs2, *_ = converter.find_nearest_point_on_path(20.0, 5.0)
    assert abs(rs2 - csp.s[-1]) < 1e-2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
