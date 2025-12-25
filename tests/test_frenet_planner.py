
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.planning.frenet_planner import FrenetPlanner, FrenetPath
from src.core.data_structures import EgoVehicleState, FrenetState
from src.planning.cubic_spline import CubicSpline2D

@pytest.fixture
def mock_spline():
    """Create a mock CubicSpline2D with a straight line."""
    spline = MagicMock(spec=CubicSpline2D)
    
    # Setup a straight line from (0,0) to (100,0) for simplicity
    # length ~ 100m
    spline.s = [0, 100.0]
    
    # Mock calc_position: returns (s, 0)
    # Handle both scalar and array inputs
    def side_effect_position(s):
        if np.isscalar(s):
            return (s, 0.0) if 0 <= s <= 100 else (None, None)
        else:
            s_arr = np.array(s)
            mask = (s_arr >= 0) & (s_arr <= 100)
            
            x = np.where(mask, s_arr, np.nan)
            y = np.where(mask, 0.0, np.nan)
            return x, y
            
    spline.calc_position.side_effect = side_effect_position
    
    # Mock calc_yaw: returns 0 (facing East)
    # Support array for consistency
    def side_effect_yaw(s):
        if np.isscalar(s):
            return 0.0
        return np.zeros_like(s)
    spline.calc_yaw.side_effect = side_effect_yaw
    
    # Mock curvature: straight line -> 0
    def side_effect_zero(s):
        if np.isscalar(s):
            return 0.0
        return np.zeros_like(s)
        
    spline.calc_curvature.side_effect = side_effect_zero
    spline.calc_curvature_rate.side_effect = side_effect_zero
    
    return spline

@pytest.fixture
def planner(mock_spline):
    """Create a FrenetPlanner instance."""
    return FrenetPlanner(
        reference_path=mock_spline,
        max_speed=10.0,
        max_accel=2.0,
        max_curvature=1.0,
        dt=0.1,
        d_road_w=1.0,
        max_road_width=7.0
    )

def test_initialization(planner):
    """Test proper initialization."""
    assert planner.max_speed == 10.0
    assert planner.dt == 0.1
    assert planner.d_road_w == 1.0

def test_cartesian_to_frenet(planner):
    """Test conversion logic using the mock spline."""
    # State: x=10, y=2 (2m left of path), v=5, heading=0
    ego_state = EgoVehicleState(x=10.0, y=2.0, yaw=0.0, v=5.0, a=0.0)
    
    # Result should be s=10, d=2 (approx)
    # Since we use a real CoordinateConverter inside, we rely on its math.
    # But we mocked the spline, so we need to be careful if Converter calls spline methods that aren't mocked.
    # The Planner's _cartesian_to_frenet_state calls csp.calc_position repeatedly to find s.
    # Our mock calc_position works for s.
    
    frenet_state = planner._cartesian_to_frenet_state(ego_state)
    
    assert frenet_state is not None
    assert np.isclose(frenet_state.s, 10.0, atol=0.5) # Approximate check due to search grid
    assert np.isclose(frenet_state.d, 2.0, atol=0.1)
    
def test_path_generation(planner):
    """Test that paths are generated."""
    frenet_state = FrenetState(s=0, s_d=5, s_dd=0, d=0, d_d=0, d_dd=0)
    target_speed = 6.0
    
    batch = planner._generate_frenet_paths_vectorized(frenet_state, target_speed)
    
    # Should generate a batch dictionary with valid arrays
    assert isinstance(batch, dict)
    assert 's' in batch
    assert 't' in batch
    assert batch['s'].size > 0
    
    # Check structure
    N = batch['s'].shape[0]
    M = batch['s'].shape[1]
    assert N > 0
    assert M > 0
    assert batch['mask'].shape == (N, M)

def test_collision_check_static(planner):
    """Test collision with static obstacles."""
    # Create batch data mimicking a path
    # Path passing through (10, 0) to (20, 0)
    # x: [10, 10.5, ..., 20], y: [0...0]
    
    x_line = np.linspace(10.0, 20.0, 21)
    y_line = np.zeros(21)
    t_line = np.zeros(21) # Dummy time
    
    # Create batch of size 1
    batch = {
        'x': x_line[np.newaxis, :], # [1, 21]
        'y': y_line[np.newaxis, :],
        't': t_line[np.newaxis, :],
        'mask': np.ones((1, 21), dtype=bool)
    }
    
    # Obstacle at (15, 0) -> Collision
    static_obs = np.array([[15.0, 0.0]])
    
    valid_mask = planner._check_collision_vectorized(batch, static_obs, None)
    # Should be False (invalid due to collision)
    assert not valid_mask[0]
    
    # Obstacle at (15, 5) -> No Collision
    static_obs_safe = np.array([[15.0, 5.0]])
    valid_mask_safe = planner._check_collision_vectorized(batch, static_obs_safe, None)
    assert valid_mask_safe[0]

def test_collision_check_dynamic(planner):
    """Test collision with dynamic obstacles."""
    # Path of 2 steps
    # t=0, t=1 (dt=planner.dt which needs to match time indices)
    # Planner dt=0.1. So indices are t/0.1.
    # If we want test at t=0 and t=1s (index 10), we need array up to 1s.
    # Let's simple create a path with t=[0, 0.1] -> indices 0, 1
    
    x_line = np.array([10.0, 11.0])
    y_line = np.array([0.0, 0.0])
    t_line = np.array([0.0, 0.1])
    
    batch = {
        'x': x_line[np.newaxis, :], # [1, 2]
        'y': y_line[np.newaxis, :],
        't': t_line[np.newaxis, :],
        'mask': np.ones((1, 2), dtype=bool)
    }
    
    # Dynamic obs: 1 obstacle, 10 steps (indices 0..9)
    # At index 0 (t=0), pos=(10,0) -> Collision
    dynamic_obs = np.zeros((1, 10, 2)) 
    dynamic_obs[0, 0, :] = [10.0, 0.0] # Collision at t=0
    dynamic_obs[0, 1, :] = [100.0, 100.0] 
    
    valid_mask = planner._check_collision_vectorized(batch, None, dynamic_obs)
    assert not valid_mask[0]
    
    # Move obstacle away at t=0
    dynamic_obs[0, 0, :] = [10.0, 5.0]
    valid_mask_safe = planner._check_collision_vectorized(batch, None, dynamic_obs)
    assert valid_mask_safe[0]

def test_plan_end_to_end(planner):
    """Test the full plan() method."""
    ego_state = EgoVehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0, a=0.0)
    static_obs = np.empty((0, 2))
    
    # Mock global path calculation dependencies that might fail with simple mock
    # The _calc_global_paths calls csp.calc_position, etc.
    # Our mock does minimal job. Let's see if it holds.
    
    path = planner.plan(ego_state, static_obs, target_speed=5.0)
    
    # Should find a path on a clear straight road
    assert path is not None
    assert len(path.x) > 0
    # Final speed should be close to target
    assert np.isclose(path.v[-1], 5.0, atol=1.0)

def test_path_validity_out_of_bounds(planner):
    """Test that paths extending beyond reference path (NaNs) are rejected."""
    # Create batch where s goes out of bounds (mock spline valid 0..100)
    # Path: s starts at 90, speed 20m/s, duration 5s -> ends at 190 (invalid)
    
    # 5 steps, t=0..4
    t_vals = np.array([0, 1, 2, 3, 4])
    s_vals = np.array([90, 110, 130, 150, 170]) # > 100 is invalid
    
    # Create batch
    batch = {
        's': s_vals[np.newaxis, :],
        's_d': np.full((1, 5), 5.0), # Valid speed (within max_speed=10)
        's_dd': np.zeros((1, 5)),
        'd': np.zeros((1, 5)),
        'd_d': np.zeros((1, 5)),
        'd_dd': np.zeros((1, 5)),
        't': t_vals[np.newaxis, :],
        'v': np.full((1, 5), 5.0),
        'a': np.zeros((1, 5)),
        'c': np.zeros((1, 5)),
        'mask': np.ones((1, 5), dtype=bool)
    }
    
    # calc_global_paths logic would fill x, y with NaNs for s > 100
    # Let's verify what check_paths does when x, y are NaNs
    batch['x'] = np.array([[90, np.nan, np.nan, np.nan, np.nan]])
    # y, yaw also likely NaN
    batch['y'] = np.array([[0, np.nan, np.nan, np.nan, np.nan]])
    batch['yaw'] = np.array([[0, np.nan, np.nan, np.nan, np.nan]])
    
    valid_mask = planner._check_paths_vectorized(batch, None, None, None)
    
    # Should be False because path is invalid (NaNs)
    # Current BUG: It might be True because checks mask NaNs?
    assert not valid_mask[0]

def test_path_validity_nan_kinematics(planner):
    """Test that paths with NaN kinematics (v, a, c) are rejected."""
    # Create batch with VALID positions but NaN velocity
    t_vals = np.array([0, 1, 2, 3, 4])
    s_vals = np.array([10, 12, 14, 16, 18]) # Valid s
    
    batch = {
        's': s_vals[np.newaxis, :],
        's_d': np.full((1, 5), 5.0),
        's_dd': np.zeros((1, 5)),
        'd': np.zeros((1, 5)),
        'd_d': np.zeros((1, 5)),
        'd_dd': np.zeros((1, 5)),
        't': t_vals[np.newaxis, :],
        'x': s_vals[np.newaxis, :], # Valid x (straight line)
        'y': np.zeros((1, 5)),      # Valid y
        'yaw': np.zeros((1, 5)),    # Valid yaw
        # Velocity has a NaN
        'v': np.array([[5.0, np.nan, 5.0, 5.0, 5.0]]),
        'a': np.zeros((1, 5)),
        'c': np.zeros((1, 5)),
        'mask': np.ones((1, 5), dtype=bool)
    }
    
    valid_mask = planner._check_paths_vectorized(batch, None, None, None)
    
    # Should be False because v contains NaN
    assert not valid_mask[0]
