
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
    spline.calc_position.side_effect = lambda s: (s, 0.0) if 0 <= s <= 100 else (None, None)
    
    # Mock calc_yaw: returns 0 (facing East)
    spline.calc_yaw.return_value = 0.0
    
    # Mock curvature: straight line -> 0
    spline.calc_curvature.return_value = 0.0
    spline.calc_curvature_rate.return_value = 0.0
    
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
    
    paths = planner._generate_frenet_paths(frenet_state, target_speed)
    
    # Should generate multiple paths (longitudinal x lateral combinations)
    assert len(paths) > 0
    
    # Check structure of a path
    fp = paths[0]
    assert len(fp.t) > 0
    assert len(fp.s) == len(fp.t)
    assert len(fp.d) == len(fp.t)

def test_collision_check_static(planner):
    """Test collision with static obstacles."""
    # Create a path passing through (10, 0) to (20, 0) with high density
    fp = FrenetPath()
    # Create points every 0.5m: 10.0, 10.5, ..., 20.0
    # 21 points
    fp.x = list(np.linspace(10.0, 20.0, 21))
    fp.y = [0.0] * 21
    fp.t = [0.0] * 21 # Time doesn't matter for static, but needs to match length
    
    # Obstacle at (15, 0) -> Collision
    static_obs = np.array([[15.0, 0.0]])
    assert planner._check_collision(fp, static_obs) is False
    
    # Obstacle at (15, 5) -> No Collision
    static_obs_safe = np.array([[15.0, 5.0]])
    assert planner._check_collision(fp, static_obs_safe) is True

def test_collision_check_dynamic(planner):
    """Test collision with dynamic obstacles."""
    fp = FrenetPath()
    fp.t = [0.0, 1.0] # t=0, t=1 (dt=dt of planner, likely 0.1? No, from fixture dt=0.1)
    # BUT in check_collision it uses self.dt to map time.
    # fp.t values are actual times.
    fp.x = [10.0, 11.0] # Moving 10m/s? No, just positions.
    fp.y = [0.0, 0.0]
    
    # Dynamic obs: 1 obstacle, 10 steps
    # At t=0 (index 0), pos=(10,0) -> Collision
    dynamic_obs = np.zeros((1, 10, 2)) 
    dynamic_obs[0, 0, :] = [10.0, 0.0] # Collision at t=0
    dynamic_obs[0, 1, :] = [100.0, 100.0] 
    
    assert planner._check_collision(fp, None, dynamic_obs) is False
    
    # Move obstacle away at t=0
    dynamic_obs[0, 0, :] = [10.0, 5.0]
    assert planner._check_collision(fp, None, dynamic_obs) is True

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
