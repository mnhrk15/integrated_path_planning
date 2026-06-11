
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
    def calc_pos_side_effect(s):
        s_arr = np.atleast_1d(s)
        x = np.where((s_arr >= 0) & (s_arr <= 100), s_arr, np.nan)
        y = np.where((s_arr >= 0) & (s_arr <= 100), 0.0, np.nan)
        
        # Return scalars if input was scalar
        if np.isscalar(s) or s_arr.size == 1:
            if np.isnan(x[0]):
                return None, None
            return float(x[0]), float(y[0])
        return x, y

    spline.calc_position.side_effect = calc_pos_side_effect
    
    # Mock calc_yaw: returns 0 (facing East)
    def calc_yaw_side_effect(s):
        s_arr = np.atleast_1d(s)
        yaw = np.zeros_like(s_arr)
        if np.isscalar(s) or s_arr.size == 1:
            return 0.0
        return yaw

    spline.calc_yaw.side_effect = calc_yaw_side_effect
    
    # Mock curvature: straight line -> 0
    def return_zero(s):
        s_arr = np.atleast_1d(s)
        z = np.zeros_like(s_arr)
        if np.isscalar(s) or s_arr.size == 1:
            return 0.0
        return z

    spline.calc_curvature.side_effect = return_zero
    spline.calc_curvature_rate.side_effect = return_zero
    
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
    assert np.isclose(frenet_state.s, 10.0, atol=0.05) # Stricter check
    assert np.isclose(frenet_state.d, 2.0, atol=0.05)
    
    # Test off-grid point to verify sub-grid precision
    # Grid is likely 0.1m step. try x=10.053
    ego_state_off = EgoVehicleState(x=10.053, y=2.0, yaw=0.0, v=5.0, a=0.0)
    fs_off = planner._cartesian_to_frenet_state(ego_state_off)
    assert np.isclose(fs_off.s, 10.053, atol=0.01) # 1cm precision expected
    assert np.isclose(fs_off.d, 2.0, atol=0.01)
    
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

def test_speed_and_accel_checks_skip_index_zero(planner):
    """Index 0 is the current state: tightened limits must not reject on it."""
    def make_fp(v, a):
        fp = FrenetPath()
        n = len(v)
        fp.x = [float(i) for i in range(n)]
        fp.y = [0.0] * n
        fp.t = [0.1 * i for i in range(n)]
        fp.v = v
        fp.a = a
        fp.c = [0.0] * n
        return fp

    overrides = {'max_speed': 5.0, 'max_accel': 2.0}
    no_obs = np.empty((0, 2))

    # Over-limit only at index 0 (current state): kept as 'ok'.
    fp = make_fp(v=[6.0, 4.0, 4.0], a=[-4.0, 1.0, 0.0])
    result = planner._check_paths([fp], no_obs, None, overrides)
    assert result['ok'] == [fp]

    # Over-limit at index 1 (plan-controlled): rejected.
    fp_speed = make_fp(v=[4.0, 6.0, 4.0], a=[0.0, 1.0, 0.0])
    result = planner._check_paths([fp_speed], no_obs, None, overrides)
    assert result['max_speed_error'] == [fp_speed]

    fp_accel = make_fp(v=[4.0, 4.0, 4.0], a=[0.0, -4.0, 0.0])
    result = planner._check_paths([fp_accel], no_obs, None, overrides)
    assert result['max_accel_error'] == [fp_accel]

def _make_kinematic_fp(c, v, yaw=None, x=None, d=None, s=None, dt=0.1):
    """Hand-built FrenetPath with consistent kinematics for check tests."""
    fp = FrenetPath()
    n = len(c)
    if x is None:
        # Integrate positions from speeds so per-sample arc lengths match v.
        x = [0.0]
        for i in range(1, n):
            x.append(x[-1] + v[i] * dt)
    fp.x = x
    fp.y = [0.0] * n
    fp.yaw = yaw if yaw is not None else [0.0] * n
    fp.t = [dt * i for i in range(n)]
    fp.v = v
    fp.a = [0.0] * n
    fp.c = c
    fp.d = d if d is not None else [0.0] * n
    fp.s = s if s is not None else list(x)
    return fp

def test_curvature_check_skips_index_zero_and_low_speed(planner):
    """c[0] is the current state and may transiently exceed the kinematic
    limit; it must not veto the candidate. From index 1 on the pointwise limit
    applies at driving speed; at or below LOW_SPEED_CURVATURE_GATE the limit
    applies to the per-sample yaw change with a floored arc length, so a
    stopped, yaw-misaligned vehicle can realign (parking-speed steering) but
    cannot pivot in place."""
    no_obs = np.empty((0, 2))

    # Over-limit only at index 0: kept as 'ok' (planner max_curvature = 1.0;
    # v=2.0 keeps v^2*kappa below the lateral-acceleration limit).
    fp = _make_kinematic_fp(c=[1.5, 0.5, 0.5], v=[2.0, 2.0, 2.0])
    result = planner._check_paths([fp], no_obs, None, None)
    assert result['ok'] == [fp]

    # Over-limit at index 1 at speed: rejected.
    fp_curv = _make_kinematic_fp(c=[0.5, 1.5, 0.5], v=[2.0, 2.0, 2.0])
    result = planner._check_paths([fp_curv], no_obs, None, None)
    assert result['max_curvature_error'] == [fp_curv]

    # Restart from standstill: pointwise curvature spikes while v <= 0.5
    # (indices 1-2, mm-scale arc lengths) but the per-sample yaw change is
    # tiny. Must be kept.
    fp_restart = _make_kinematic_fp(
        c=[0.1, 2.0, 1.4, 0.3, 0.1],
        v=[0.0, 0.05, 0.4, 1.2, 3.0],
        yaw=[0.0, 0.002, 0.004, 0.006, 0.008])
    result = planner._check_paths([fp_restart], no_obs, None, None)
    assert result['ok'] == [fp_restart]

    # Snap pivot: 0.3 rad heading change in one near-standstill sample
    # (cap is max(kappa*ds, 0.1 rad)). Rejected.
    fp_pivot = _make_kinematic_fp(
        c=[0.1, 2.0, 1.4, 0.3, 0.1],
        v=[0.0, 0.05, 0.4, 1.2, 3.0],
        yaw=[0.0, 0.3, 0.31, 0.32, 0.33])
    result = planner._check_paths([fp_pivot], no_obs, None, None)
    assert result['max_curvature_error'] == [fp_pivot]

    # Sideways slide: 0.3 m of lateral displacement with ~zero longitudinal
    # progress while nearly stopped — the metric-relevant exploit (evading a
    # collision without rolling). Rejected.
    fp_slide = _make_kinematic_fp(
        c=[0.1, 2.0, 1.4, 0.3, 0.1],
        v=[0.0, 0.05, 0.4, 1.2, 3.0],
        d=[0.0, 0.3, 0.6, 0.7, 0.7],
        s=[0.0, 0.001, 0.04, 0.16, 0.46])
    result = planner._check_paths([fp_slide], no_obs, None, None)
    assert result['max_curvature_error'] == [fp_slide]

    # Same pointwise spike at driving speed: rejected.
    fp_fast = _make_kinematic_fp(c=[0.1, 2.0, 1.4, 0.3, 0.1],
                                 v=[3.0, 3.0, 3.0, 3.0, 3.0])
    result = planner._check_paths([fp_fast], no_obs, None, None)
    assert result['max_curvature_error'] == [fp_fast]

def test_lateral_accel_check(planner):
    """v^2*kappa must not exceed max_lat_accel (default 3.0); EMERGENCY may
    relax it via constraint_overrides."""
    no_obs = np.empty((0, 2))

    # 8 m/s at kappa 0.05 -> 3.2 m/s^2 > 3.0: rejected (within the curvature
    # limit 1.0, so this is caught by the lateral-acceleration check alone).
    fp = _make_kinematic_fp(c=[0.0, 0.05, 0.05], v=[8.0, 8.0, 8.0])
    result = planner._check_paths([fp], no_obs, None, None)
    assert result['max_lat_accel_error'] == [fp]

    # Index 0 is exempt (current state).
    fp0 = _make_kinematic_fp(c=[0.05, 0.01, 0.01], v=[8.0, 8.0, 8.0])
    result = planner._check_paths([fp0], no_obs, None, None)
    assert result['ok'] == [fp0]

    # EMERGENCY-style override admits the same path.
    fp2 = _make_kinematic_fp(c=[0.0, 0.05, 0.05], v=[8.0, 8.0, 8.0])
    result = planner._check_paths([fp2], no_obs, None, {'max_lat_accel': 6.0})
    assert result['ok'] == [fp2]

def test_road_corridor_check_whole_path(planner):
    """|d(t)| must stay within max_road_width over the whole path, not only
    at the terminal offset (the sampling grid only pins d(T))."""
    no_obs = np.empty((0, 2))

    # Mid-path overshoot beyond max_road_width (7.0): rejected.
    fp = _make_kinematic_fp(c=[0.0, 0.0, 0.0], v=[4.0, 4.0, 4.0],
                            d=[0.0, 7.5, 6.9])
    result = planner._check_paths([fp], no_obs, None, None)
    assert result['road_bound_error'] == [fp]

    # Index 0 outside the corridor (e.g. ego starts off-grid) is tolerated.
    fp0 = _make_kinematic_fp(c=[0.0, 0.0, 0.0], v=[4.0, 4.0, 4.0],
                             d=[7.5, 6.9, 6.0])
    result = planner._check_paths([fp0], no_obs, None, None)
    assert result['ok'] == [fp0]

def test_lateral_candidates_bounded_by_max_road_width(mock_spline):
    """Lateral offsets must stay within ±max_road_width (treated as the
    allowed offset from the reference line, i.e. road half-width minus the
    vehicle footprint)."""
    narrow = FrenetPlanner(
        reference_path=mock_spline,
        max_speed=10.0,
        max_accel=2.0,
        max_curvature=1.0,
        dt=0.1,
        d_road_w=0.3,
        max_road_width=1.2
    )
    fstate = FrenetState(s=5.0, s_d=5.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    paths = narrow._generate_frenet_paths(fstate, target_speed=5.0)
    assert len(paths) > 0
    terminal_d = {round(fp.d[-1], 6) for fp in paths if len(fp.d)}
    # Full symmetric grid: 9 distinct offsets at 0.3 m spacing, extremes at
    # exactly +-1.2, reference line included.
    assert terminal_d == {round(0.3 * i, 6) for i in range(-4, 5)}

def test_collision_check_dynamic_same_time_only(planner):
    """Collision requires SAME-TIME co-location, not mere spatial overlap.

    Discriminating cases for the core invariant (M-16): a flattened
    implementation (ignoring time) or one that always reads obstacle index 0
    must fail at least one of these assertions.
    """
    # Combined radius is robot 2.0 + obstacle 0.3 = 2.3 m, so consecutive
    # samples are spaced 3 m apart to stay clear of each other.
    fp = FrenetPath()
    fp.x = [10.0, 13.0, 16.0]
    fp.y = [0.0, 0.0, 0.0]
    fp.t = [0.0, 0.1, 0.2]

    far = [100.0, 100.0]

    # (a) Obstacle occupies the final path point but at t=0 only: the path is
    # at (10,0) then (6 m away), so this must NOT collide. A flattened check
    # (or an always-index-0 check) reports a collision here.
    obs_wrong_time = np.array([[[16.0, 0.0], far, far]])
    assert planner._check_collision(fp, None, obs_wrong_time) is True

    # (b) Same position, same time (t=0.2 <-> obstacle index 2): collision.
    obs_same_time = np.array([[far, far, [16.0, 0.0]]])
    assert planner._check_collision(fp, None, obs_same_time) is False

    # (c) Collision only at a non-zero index (t=0.1 <-> index 1): an
    # implementation that only ever checks index 0 misses this.
    obs_mid_time = np.array([[far, [13.0, 0.0], far]])
    assert planner._check_collision(fp, None, obs_mid_time) is False

def test_collision_check_distribution_chance_constrained(planner):
    """Chance-constrained check tolerates up to floor(epsilon*N) colliding samples."""
    fp = FrenetPath()
    fp.x = [10.0, 11.0]
    fp.y = [0.0, 0.0]
    fp.t = [0.0, 0.1]  # -> time indices [0, 1]

    # 4 samples, 1 pedestrian, 2 steps. Only sample 0 collides (at path point 0).
    far = [[100.0, 100.0], [100.0, 100.0]]
    hit = [[10.0, 0.0], [100.0, 100.0]]
    dist = np.array([hit, far, far, far])[:, None, :, :]  # [4, 1, 2, 2]
    assert dist.shape == (4, 1, 2, 2)

    # epsilon=0 (robust): a single colliding sample makes the path infeasible.
    assert planner._check_collision_distribution(fp, None, dist, 0.0) is False
    # epsilon=0.25 -> floor(0.25*4)=1 violation allowed -> feasible.
    assert planner._check_collision_distribution(fp, None, dist, 0.25) is True
    # epsilon=0.2 -> floor(0.2*4)=0 -> infeasible again.
    assert planner._check_collision_distribution(fp, None, dist, 0.2) is False

def test_collision_check_distribution_static_is_hard(planner):
    """Static obstacles remain hard constraints regardless of epsilon."""
    fp = FrenetPath()
    fp.x = [10.0, 11.0]
    fp.y = [0.0, 0.0]
    fp.t = [0.0, 0.1]

    far = [[100.0, 100.0], [100.0, 100.0]]
    dist = np.array([far, far])[:, None, :, :]  # no dynamic collisions
    static_obs = np.array([[10.0, 0.0]])  # directly on the path

    # Even when all dynamic violations are allowed, the static obstacle blocks it.
    assert planner._check_collision_distribution(fp, static_obs, dist, 1.0) is False
    # Without the static obstacle the path is feasible.
    assert planner._check_collision_distribution(fp, None, dist, 0.0) is True

@pytest.fixture
def inflated_planner_pair(mock_spline):
    """Planners sharing geometry (combined radius 1.3m) with/without margin inflation."""
    kwargs = dict(
        reference_path=mock_spline,
        max_speed=10.0,
        max_accel=2.0,
        max_curvature=1.0,
        dt=0.1,
        d_road_w=1.0,
        max_road_width=7.0,
        robot_radius=1.0,
        obstacle_radius=0.3,
    )
    nominal = FrenetPlanner(**kwargs)
    inflated = FrenetPlanner(**kwargs, collision_margin_inflation=1.2)
    return nominal, inflated

def _straight_fp():
    fp = FrenetPath()
    fp.x = [10.0, 11.0]
    fp.y = [0.0, 0.0]
    fp.t = [0.0, 0.1]
    return fp

def test_margin_inflation_rejects_borderline_dynamic(inflated_planner_pair):
    """Inflation 1.2 widens the dynamic check radius from 1.3 to 1.56m."""
    nominal, inflated = inflated_planner_pair
    fp = _straight_fp()
    # Pedestrian 1.4m laterally away at t=0: outside 1.3m, inside 1.56m.
    dynamic_obs = np.full((1, 2, 2), 100.0)
    dynamic_obs[0, 0, :] = [10.0, 1.4]

    assert nominal._check_collision(fp, None, dynamic_obs) is True
    assert inflated._check_collision(fp, None, dynamic_obs) is False

def test_margin_inflation_not_applied_to_distribution(inflated_planner_pair):
    """The chance-constrained check keeps the nominal radius even when inflation is set."""
    _, inflated = inflated_planner_pair
    fp = _straight_fp()
    near_miss = [[10.0, 1.4], [100.0, 100.0]]  # 1.4m > nominal 1.3m
    dist = np.array([near_miss, near_miss])[:, None, :, :]  # [2, 1, 2, 2]

    # Robust (eps=0) distribution check passes: no sample is within 1.3m.
    assert inflated._check_collision_distribution(fp, None, dist, 0.0) is True
    # The same geometry fails the inflated single-sample check.
    assert inflated._check_collision(fp, None, np.array(near_miss)[None, :, :]) is False

def test_margin_inflation_static_unaffected(inflated_planner_pair):
    """Static obstacles always use the nominal radius."""
    nominal, inflated = inflated_planner_pair
    fp = _straight_fp()
    static_obs = np.array([[10.0, 1.4]])  # 1.4m away, outside nominal 1.3m

    assert nominal._check_collision(fp, static_obs) is True
    assert inflated._check_collision(fp, static_obs) is True
    # Sanity: a truly colliding static obstacle is still rejected.
    assert inflated._check_collision(fp, np.array([[10.0, 0.5]])) is False

def test_default_inflation_preserves_geometry(planner):
    """With inflation 1.0 the geometry matches the pre-inflation implementation."""
    fp = _straight_fp()
    default_geom = planner._path_collision_geometry(fp)
    explicit_geom = planner._path_collision_geometry(fp, 1.0)

    for a, b in zip(default_geom, explicit_geom):
        assert np.array_equal(a, b)

    path_points, _, path_min, path_max, sq_rubicon, sq_rubicon_dyn = default_geom
    radius = max(planner.robot_radius + planner.obstacle_radius, 1e-6)
    assert sq_rubicon == radius ** 2
    assert sq_rubicon_dyn == sq_rubicon
    assert np.array_equal(path_min, np.min(path_points, axis=0) - radius)
    assert np.array_equal(path_max, np.max(path_points, axis=0) + radius)

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
