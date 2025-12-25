"""Frenet Optimal Trajectory Planner.

This module implements the Frenet-Frame based optimal trajectory generation
for autonomous driving.

Reference:
Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios 
in a Frenet Frame" (2010)
"""

import copy
import numpy as np
from typing import List, Optional, Tuple, Dict
from loguru import logger

from ..core.data_structures import FrenetPath, FrenetState, EgoVehicleState
from ..core.coordinate_converter import CartesianFrenetConverter
from .cubic_spline import CubicSpline2D

from .polynomial_solver import VectorizedPolynomialSolver


# Planning parameters
MAX_SPEED = 50.0 / 3.6  # Maximum speed [m/s]
MAX_ACCEL = 2.0  # Maximum acceleration [m/s²]
MAX_CURVATURE = 1.0  # Maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # Maximum road width [m]
D_ROAD_W = 0.5  # Road width sampling distance [m]
DT = 0.2  # Time tick [s]
MAX_T = 5.0  # Max prediction time [s]
MIN_T = 4.0  # Min prediction time [s]
TARGET_SPEED = 30.0 / 3.6  # Target speed [m/s]
D_T_S = 5.0 / 3.6  # Target speed sampling width [m/s]
N_S_SAMPLE = 1  # Sampling number of target speed

# Cost weights
K_J = 0.1  # Jerk cost
K_T = 0.1  # Time cost
K_D = 1.0  # Lateral offset cost
K_S_DOT = 1.0  # Speed difference cost
K_LAT = 1.0  # Lateral cost weight
K_LON = 1.0  # Longitudinal cost weight

# Safety
ROBOT_RADIUS = 2.0  # Robot radius [m]


class FrenetPlanner:
    """Frenet Optimal Trajectory Planner.
    
    This planner generates optimal trajectories in the Frenet frame along
    a reference path, avoiding obstacles while optimizing for comfort and efficiency.
    
    Args:
        reference_path: CubicSpline2D representing the reference path
        max_speed: Maximum allowed speed [m/s]
        max_accel: Maximum allowed acceleration [m/s²]
        max_curvature: Maximum allowed curvature [1/m]
        dt: Time step [s]
        d_road_w: Lateral sampling width [m]
        max_road_width: Maximum road width to check [m]
    """
    
    def __init__(
        self,
        reference_path: CubicSpline2D,
        max_speed: float = MAX_SPEED,
        max_accel: float = MAX_ACCEL,
        max_curvature: float = MAX_CURVATURE,
        dt: float = DT,
        d_road_w: float = D_ROAD_W,
        max_road_width: float = MAX_ROAD_WIDTH,
        robot_radius: float = ROBOT_RADIUS,
        obstacle_radius: float = 0.3,
        min_t: float = MIN_T,
        max_t: float = MAX_T,
        d_t_s: float = D_T_S,
        n_s_sample: int = N_S_SAMPLE,
        **kwargs
    ):
        self.csp = reference_path
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_curvature = max_curvature
        self.dt = dt
        self.d_road_w = d_road_w
        self.max_road_width = max_road_width
        self.converter = CartesianFrenetConverter()
        self.robot_radius = robot_radius
        self.obstacle_radius = obstacle_radius
        
        # Time horizon parameters (use provided values or fallback to module constants)
        self.min_t = min_t
        self.max_t = max_t
        self.d_t_s = d_t_s
        self.n_s_sample = n_s_sample

        # Cost weights (default to module constants; can be overridden via kwargs)
        self.k_j = kwargs.get("k_j", K_J)
        self.k_t = kwargs.get("k_t", K_T)
        self.k_d = kwargs.get("k_d", K_D)
        self.k_s_dot = kwargs.get("k_s_dot", K_S_DOT)
        self.k_lat = kwargs.get("k_lat", K_LAT)
        self.k_lon = kwargs.get("k_lon", K_LON)
        
        logger.info(f"Frenet Planner initialized with dt={dt}s, "
                   f"max_speed={max_speed}m/s, max_accel={max_accel}m/s², "
                   f"robot_radius={robot_radius}m, obstacle_radius={obstacle_radius}m, "
                   f"time_horizon=[{min_t:.1f}, {max_t:.1f}]s")
    
    def plan(
        self,
        ego_state: EgoVehicleState,
        static_obstacles: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray] = None,
        target_speed: float = TARGET_SPEED,
        constraint_overrides: Optional[Dict[str, float]] = None
    ) -> Optional[FrenetPath]:
        """Vectorized Plan optimal trajectory from current state.
        
        Args:
            ego_state: Current ego vehicle state
            static_obstacles: Static obstacle positions [n_obstacles, 2]
            dynamic_obstacles: Dynamic obstacles with time dimension [n_obs, time_steps, 2]
            target_speed: Desired target speed [m/s]
            constraint_overrides: Optional dictionary to override constraints
            
        Returns:
            Best trajectory, or None if no valid trajectory found
        """
        # Convert ego state to Frenet frame
        frenet_state = self._cartesian_to_frenet_state(ego_state)
        
        if frenet_state is None:
            logger.warning("Failed to convert ego state to Frenet frame")
            return None
        
        # 1. Generate candidate paths (Vectorized)
        # Returns dict of tensors
        batch = self._generate_frenet_paths_vectorized(frenet_state, target_speed)
        
        if batch['s'].size == 0:
            logger.warning("No candidate paths generated")
            return None

        # 2. Calculate Global Coordinates (Vectorized)
        batch = self._calc_global_paths_vectorized(batch)
        
        # 3. Check Validity (Vectorized)
        valid_mask = self._check_paths_vectorized(
            batch, static_obstacles, dynamic_obstacles, constraint_overrides
        )
        
        if not np.any(valid_mask):
            logger.warning("No valid path found (all checks failed)")
            return None
            
        # 4. Select Best Path
        best_path = self._select_best_path_vectorized(batch, valid_mask)
        
        if best_path is not None:
            logger.debug(f"Found valid vectorized path with cost {best_path.cost:.2f}")
        
        return best_path

    def _generate_frenet_paths_vectorized(
        self,
        frenet_state: FrenetState,
        target_speed: float
    ) -> Dict[str, np.ndarray]:
        """Generate candidate Frenet paths using vectorized operations."""
        # Parameter Space
        # Note: Using instance vars
        max_s = self.csp.s[-1]
        s_remaining = max_s - frenet_state.s
        if s_remaining <= 0.0:
            return {'s': np.array([])}

        # Velocities
        # Use round to avoid floating point issues with arange
        v_min = target_speed - self.d_t_s * self.n_s_sample
        v_max = target_speed + self.d_t_s * self.n_s_sample
        # Includes end point logic similar to original code if needed, but original used arange which excludes endpoint usually unless steps align perfectly.
        # Original: np.arange(v_min, v_max + d_t_s, d_t_s) almost implies including max
        v_samples = np.arange(v_min, v_max + 0.001, self.d_t_s)
        v_samples = v_samples[v_samples >= 0.0]  # Remove negative velocities
        if v_samples.size == 0:
            return {'s': np.array([])}

        # Goal-aware sampling: shrink horizon and add stop candidate near the end of the spline
        t_min = self.min_t
        t_max = self.max_t
        slowest_end_speed = float(np.min(v_samples))
        if slowest_end_speed > 0.0 and s_remaining < slowest_end_speed * self.max_t:
            max_t_by_dist = s_remaining / slowest_end_speed
            t_max = min(t_max, max_t_by_dist)
            t_min = min(t_min, t_max)
            if not np.isclose(v_samples, 0.0).any():
                v_samples = np.sort(np.concatenate([v_samples, np.array([0.0])]))

        # Ensure we have at least one time sample
        if t_max < self.dt:
            t_max = self.dt
        if t_min > t_max:
            t_min = t_max
        t_samples = np.arange(t_min, t_max + 1e-5, self.dt)
        
        # Lateral offsets
        d_samples = np.arange(-self.max_road_width, self.max_road_width, self.d_road_w)
        
        # Meshgrid
        # Shape: (N_T, N_V, N_D)
        T_grid, V_grid, D_grid = np.meshgrid(t_samples, v_samples, d_samples, indexing='ij')
        
        # Flatten for solver
        T_flat = T_grid.ravel()
        V_flat = V_grid.ravel()
        D_flat = D_grid.ravel()
        
        N = len(T_flat)
        if N == 0:
            return {'s': np.array([])}
            
        # --- Solve Longitudinal (Quartic) ---
        # s_start, s_d, s_dd -> v_end=V_flat, a_end=0
        lon_coeffs = VectorizedPolynomialSolver.solve_quartic_batch(
            frenet_state.s, frenet_state.s_d, frenet_state.s_dd,
            V_flat, np.zeros(N), T_flat
        )
        # lon_coeffs: [N, 5] -> a0..a4
        
        # --- Solve Lateral (Quintic) ---
        # d_start... -> d_end=D_flat, d_d=0, d_dd=0
        lat_coeffs = VectorizedPolynomialSolver.solve_quintic_batch(
            frenet_state.d, frenet_state.d_d, frenet_state.d_dd,
            D_flat, np.zeros(N), np.zeros(N), T_flat
        )
        # lat_coeffs: [N, 6] -> a0..a5
        
        # --- Evaluate Trajectories ---
        # Time steps [0, dt, ..., MAX_T_in_batch]
        max_duration = np.max(T_flat)
        n_steps = int(np.floor(max_duration / self.dt)) + 1 # +1 to include 0.0
        
        # Time tensor [N, n_steps]
        t_base = np.arange(n_steps) * self.dt
        t_tensor = np.broadcast_to(t_base, (N, n_steps))
        
        # Mask for valid times: t <= T
        # Add small epsilon for float comparison safety
        mask = t_tensor <= (T_flat[:, None] + 1e-5)
        
        # Evaluation helper
        # We assume t_tensor is used for powers.
        t2 = t_tensor ** 2
        t3 = t2 * t_tensor
        t4 = t3 * t_tensor
        t5 = t4 * t_tensor
        
        # Longitudinal s(t)
        # a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
        # a: [N, 5]
        s = (lon_coeffs[:, 0, None] + 
             lon_coeffs[:, 1, None] * t_tensor + 
             lon_coeffs[:, 2, None] * t2 + 
             lon_coeffs[:, 3, None] * t3 + 
             lon_coeffs[:, 4, None] * t4)
             
        s_d = (lon_coeffs[:, 1, None] + 
               2 * lon_coeffs[:, 2, None] * t_tensor + 
               3 * lon_coeffs[:, 3, None] * t2 + 
               4 * lon_coeffs[:, 4, None] * t3)
               
        s_dd = (2 * lon_coeffs[:, 2, None] + 
                6 * lon_coeffs[:, 3, None] * t_tensor + 
                12 * lon_coeffs[:, 4, None] * t2)
                
        s_ddd = (6 * lon_coeffs[:, 3, None] + 
                 24 * lon_coeffs[:, 4, None] * t_tensor)
                 
        # Lateral d(t)
        # a0 + ... + a5*t^5
        d = (lat_coeffs[:, 0, None] + 
             lat_coeffs[:, 1, None] * t_tensor + 
             lat_coeffs[:, 2, None] * t2 + 
             lat_coeffs[:, 3, None] * t3 + 
             lat_coeffs[:, 4, None] * t4 + 
             lat_coeffs[:, 5, None] * t5)
             
        d_d = (lat_coeffs[:, 1, None] + 
               2 * lat_coeffs[:, 2, None] * t_tensor + 
               3 * lat_coeffs[:, 3, None] * t2 + 
               4 * lat_coeffs[:, 4, None] * t3 + 
               5 * lat_coeffs[:, 5, None] * t4)
               
        d_dd = (2 * lat_coeffs[:, 2, None] + 
                6 * lat_coeffs[:, 3, None] * t_tensor + 
                12 * lat_coeffs[:, 4, None] * t2 + 
                20 * lat_coeffs[:, 5, None] * t3)
                
        d_ddd = (6 * lat_coeffs[:, 3, None] + 
                 24 * lat_coeffs[:, 4, None] * t_tensor + 
                 60 * lat_coeffs[:, 5, None] * t2)

        # Extend mask with spline range validity to truncate paths at boundary
        s_in_range = (s >= 0.0) & (s <= max_s)
        mask = mask & s_in_range

        # Apply mask (set invalid to NaN to avoid processing)
        # We use NaN for invalid steps, which propagates through logic
        # But for costs, we sum ignoring NaNs or assume 0 contribution?
        # Better: keep values but valid mask handles logic.
        # Actually, setting to NaN is safer for coordinate conversion checks.
        
        # HOWEVER: s_d, s_dd etc are used for costs.
        # Cost is integral over [0, T]. So we should Zero out values beyond T?
        # Cost Logic: sum(J^2). If we zero out J beyond T, sum is correct.
        
        s_ddd_masked = np.where(mask, s_ddd, 0.0)
        d_ddd_masked = np.where(mask, d_ddd, 0.0)
        
        # --- Cost Calculation (Vectorized) ---
        # Jp = sum(d_ddd^2)
        Jp = np.sum(d_ddd_masked ** 2, axis=1)
        
        # Js = sum(s_ddd^2)
        Js = np.sum(s_ddd_masked ** 2, axis=1)
        
        # Jd = d[T]^2
        # We need final State. 
        # Since T matches grid, we can just use the value at index corresponding to T roughly?
        # Or better, evaluate at T exactly using the coefficients to be precise.
        # But t_tensor has discrete steps.
        # Let's use the discrete last valid point.
        # Row-wise, the last True in mask.
        # Actually, simpler: The coefficients define the exact curve.
        # d_end was a constraint! So d(T) = D_flat.
        # s_d_end was constraint! s_d(T) = V_flat.
        # s_d(T) is target_velocity of this path?
        # Wait, s_d_end in Quartic IS V_flat.
        
        Jd = D_flat ** 2
        Jv = (target_speed - V_flat) ** 2
        Jt = T_flat
        
        lat_cost = self.k_j * Jp + self.k_t * Jt + self.k_d * Jd
        lon_cost = self.k_j * Js + self.k_t * Jt + self.k_s_dot * Jv
        total_cost = self.k_lat * lat_cost + self.k_lon * lon_cost
        
        # Pack into batch dict
        # We keep the full time arrays (with NaNs for invalid)
        
        # Set invalid values to NaN for safety in conversion
        # Use copy to avoid setting original params if shared (not shared here)
        s[~mask] = np.nan
        d[~mask] = np.nan
        s_d[~mask] = np.nan
        d_d[~mask] = np.nan
        s_dd[~mask] = np.nan
        d_dd[~mask] = np.nan
        
        return {
            't': t_tensor,
            's': s,
            's_d': s_d,
            's_dd': s_dd,
            'd': d,
            'd_d': d_d,
            'd_dd': d_dd,
            'target_speed': V_flat, # Note: V_flat is ending speed, not necessarily constant target
            'cost': total_cost,
            'mask': mask,
            'T': T_flat
        }
    
    def _cartesian_to_frenet_state(
        self, 
        ego_state: EgoVehicleState
    ) -> Optional[FrenetState]:
        """Convert ego vehicle state from Cartesian to Frenet frame.
        
        Args:
            ego_state: Ego vehicle state in Cartesian coordinates
            
        Returns:
            Ego vehicle state in Frenet coordinates, or None if conversion fails
        """
        # Find nearest point on reference path
        s_samples = np.linspace(0, self.csp.s[-1], 1000)
        min_dist = float('inf')
        best_s = 0.0
        
        for s in s_samples:
            px, py = self.csp.calc_position(s)
            if px is None or py is None:
                continue
            dist = np.hypot(ego_state.x - px, ego_state.y - py)
            if dist < min_dist:
                min_dist = dist
                best_s = s
        
        # Get reference point properties
        rs = best_s
        rx, ry = self.csp.calc_position(rs)
        rtheta = self.csp.calc_yaw(rs)
        rkappa = self.csp.calc_curvature(rs)
        rdkappa = self.csp.calc_curvature_rate(rs)
        
        if any(v is None for v in [rx, ry, rtheta, rkappa, rdkappa]):
            return None
        
        # Convert to Frenet
        try:
            s_condition, d_condition = self.converter.cartesian_to_frenet(
                rs, rx, ry, rtheta, rkappa, rdkappa,
                ego_state.x, ego_state.y, ego_state.v, ego_state.a,
                ego_state.yaw, 0.0  # Assume curvature is 0 for simplicity
            )
            
            return FrenetState(
                s=s_condition[0],
                s_d=s_condition[1],
                s_dd=s_condition[2],
                d=d_condition[0],
                d_d=d_condition[1],
                d_dd=d_condition[2]
            )
        except Exception as e:
            logger.error(f"Error converting to Frenet frame: {e}")
            return None
    
    def _calc_global_paths_vectorized(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate global coordinates for batch paths (fully vectorized)."""
        s = batch['s'] # [N, M]
        d = batch['d']
        s_d = batch['s_d']
        d_d = batch['d_d']
        s_dd = batch['s_dd']
        d_dd = batch['d_dd']
        
        # Flatten for spline calculation
        # We process all points including NaNs (spline returns NaNs for them)
        shape = s.shape
        s_flat = s.ravel()
        
        # Reference path calculations
        ix, iy = self.csp.calc_position(s_flat)
        i_yaw = self.csp.calc_yaw(s_flat)
        i_k = self.csp.calc_curvature(s_flat)
        i_dk = self.csp.calc_curvature_rate(s_flat)
        
        # Prepare conditions tuple for converter
        # Flatten derivatives as well
        s_condition = (s_flat, s_d.ravel(), s_dd.ravel())
        d_condition = (d.ravel(), d_d.ravel(), d_dd.ravel())
        
        # Vectorized conversion
        # Note: converter.frenet_to_cartesian handles arrays
        x, y, yaw, k, v, a = self.converter.frenet_to_cartesian(
            s_flat, ix, iy, i_yaw, i_k, i_dk,
            s_condition, d_condition
        )
        
        # Reshape back to [N, M] and store in batch
        batch['x'] = x.reshape(shape)
        batch['y'] = y.reshape(shape)
        batch['yaw'] = yaw.reshape(shape)
        batch['c'] = k.reshape(shape)
        batch['v'] = v.reshape(shape)
        batch['a'] = a.reshape(shape)
        
        return batch

    def _check_paths_vectorized(
        self,
        batch: Dict[str, np.ndarray],
        static_obstacles: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray],
        constraint_overrides: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """Check validity of batch paths (vectorized).
        
        Returns:
            Boolean mask [N] where True indicates valid path
        """
        # Resolve constraints
        c_max_speed = self.max_speed
        c_max_accel = self.max_accel
        c_max_curvature = self.max_curvature
        
        if constraint_overrides:
            c_max_speed = constraint_overrides.get('max_speed', c_max_speed)
            c_max_accel = constraint_overrides.get('max_accel', c_max_accel)
            c_max_curvature = constraint_overrides.get('max_curvature', c_max_curvature)
            
        # Get tensors [N, M]
        v = batch['v']
        a = batch['a']
        c = batch['c']
        x = batch['x']
        y = batch['y']
        yaw = batch['yaw']
        mask = batch['mask'] # Valid time steps

        # Reject paths that have no valid samples at all.
        has_valid_samples = np.any(mask, axis=1)
        if not np.any(has_valid_samples):
            return has_valid_samples
        
        # Mask NaNs in constraints to avoid False positives (NaN > max is False)
        # But we need to ensure we don't count invalid parts.
        # Check ONLY valid parts.
        
        # any() over axis 1 (time steps)
        # We need to ignore masked values.
        # Fill masked values with safe values (0) before checking max
        
        v_filled = np.where(mask, v, 0.0)
        a_filled = np.where(mask, a, 0.0)
        c_filled = np.where(mask, c, 0.0)
        
        # 1. Max Speed Check
        # If any point in valid range exceeds max
        speed_invalid = np.any(v_filled > c_max_speed, axis=1)
        
        # 2. Max Accel Check
        accel_invalid = np.any(np.abs(a_filled) > c_max_accel, axis=1)
        
        # 3. Max Curvature Check
        curv_invalid = np.any(np.abs(c_filled) > c_max_curvature, axis=1)
        
        # 4. NaN / Out of bounds Check
        # If x, y, yaw, v, a, or c are NaN within valid mask, path is invalid
        nan_invalid = np.any(
            (np.isnan(x) | np.isnan(y) | np.isnan(yaw) | np.isnan(v) | np.isnan(a) | np.isnan(c)) & mask, 
            axis=1
        )
        
        kinematic_valid = has_valid_samples & ~(speed_invalid | accel_invalid | curv_invalid | nan_invalid)
        
        if not np.any(kinematic_valid):
            return kinematic_valid # All False
            
        # 4. Collision Check on remaining candidates
        # To save computation, only check collision for kinematically valid paths
        # BUT for full vectorization structure, maybe check all OR subset?
        # Let's perform check on ALL for simplicity of code structure, logic is vectorized.
        # Or use indexing? Indexing is better if valid set is small.
        # Let's check all, optimization comes from numpy.
        
        collision_valid = self._check_collision_vectorized(batch, static_obstacles, dynamic_obstacles)
        
        return kinematic_valid & collision_valid

    def _check_collision_vectorized(
        self,
        batch: Dict[str, np.ndarray],
        static_obstacles: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray]
    ) -> np.ndarray:
        """Vectorized collision check."""
        x = batch['x'] # [N, M]
        y = batch['y']
        mask = batch['mask']
        t = batch['t']
        
        N, M = x.shape
        
        # Points: [N, M, 2]
        # Only check valid points?
        # We can fill invalid x,y with infinity?
        # Or just let them be NaNs. Distance to NaN is NaN. NaN <= radius is False.
        # So NaN is "safe" for collision check if comparators handle it right.
        # dist <= radius. If dist is NaN, False. OK.
        
        path_points = np.stack([x, y], axis=2) # [N, M, 2]
        
        inflated_radius = max(self.robot_radius + self.obstacle_radius, 1e-6)
        sq_rubicon = inflated_radius ** 2
        
        # 1. Static Obstacles
        # [N, M, 1, 2] - [1, 1, N_obs, 2] -> [N, M, N_obs, 2]
        # Memory heavy if N=thousands, M=25, Obs=100 -> 1000 * 25 * 100 * 2 * 8 bytes ≈ 40MB. OK.
        
        if static_obstacles is not None and len(static_obstacles) > 0:
            # We can flatten N*M to simplify broadcasting?
            # [N*M, 2] - [N_obs, 2] -> [N*M, N_obs, 2]
            # dists: [N*M, N_obs] -> min axis 1 -> [N*M]
            # reshape [N, M] -> check any.
            
            diff = path_points[:, :, np.newaxis, :] - static_obstacles[np.newaxis, np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=3) # [N, M, N_obs]
            
            # Check collisions
            # any point colliding?
            # sq_dists <= sq_rubicon
            
            # Mask out invalid time steps (NaNs are False comparison usually, but verify)
            # NaNs in sq_dists results in False for <=.
            # So invalid points won't trigger collision.
            
            collisions = np.any(sq_dists <= sq_rubicon, axis=2) # [N, M] (collision with ANY obs)
            path_collision = np.any(collisions, axis=1) # [N] (collision at ANY time)
            
            if np.any(path_collision):
                # If any path collides, markup.
                # We need to return valid mask
                # So if path_collision is True, valid is False.
                pass
            else:
                path_collision = np.zeros(N, dtype=bool)
        else:
            path_collision = np.zeros(N, dtype=bool)
            
        if np.all(path_collision):
            return ~path_collision
            
        # 2. Dynamic Obstacles
        if (dynamic_obstacles is not None and 
            dynamic_obstacles.size > 0 and 
            dynamic_obstacles.shape[-1] == 2):
            
            n_obs, n_time_obs, _ = dynamic_obstacles.shape
            
            # Time alignment
            # t: [N, M]. Indices: round(t/dt)
            time_indices = np.round(t / self.dt).astype(int)
            time_indices = np.clip(time_indices, 0, n_time_obs - 1)
            
            # Get obstacles at relevant times
            # dynamic_obstacles: [n_obs, n_time_obs, 2]
            # We want obs for each path point (n, m)
            # Obs pos: [n_obs, 2] for each (n, m)
            
            # This is tricky to broadcast simply.
            # We effectively want [N, M, n_obs, 2]
            # Obs[o, time_indices[n,m], :]
            
            # Transpose obs to [n_time_obs, n_obs, 2]
            obs_T = dynamic_obstacles.transpose(1, 0, 2)
            
            # Gather
            # obs_T[time_indices[n,m]] -> [N, M, n_obs, 2]
            relevant_obs = obs_T[time_indices] 
            
            diff = path_points[:, :, np.newaxis, :] - relevant_obs
            sq_dists = np.sum(diff ** 2, axis=3) # [N, M, n_obs]
            
            dyn_collisions = np.any(sq_dists <= sq_rubicon, axis=2) # [N, M]
            dyn_path_collision = np.any(dyn_collisions, axis=1) # [N]
            
            path_collision = path_collision | dyn_path_collision
            
        return ~path_collision

    def _select_best_path_vectorized(
        self, 
        batch: Dict[str, np.ndarray], 
        valid_mask: np.ndarray
    ) -> Optional[FrenetPath]:
        """Select best path from batch and convert to object."""
        if not np.any(valid_mask):
            return None
            
        costs = batch['cost']
        # Set invalid costs to infinity
        valid_costs = np.where(valid_mask, costs, float('inf'))
        
        best_idx = np.argmin(valid_costs)
        
        if valid_costs[best_idx] == float('inf'):
            return None
            
        # Create FrenetPath object for best index
        return self._extract_path_from_batch(batch, best_idx)

    def _extract_path_from_batch(self, batch: Dict[str, np.ndarray], idx: int) -> FrenetPath:
        """Create FrenetPath object from batch at index."""
        fp = FrenetPath()
        
        mask = batch['mask'][idx]
        valid_len = np.sum(mask)
        
        # Helper to slice and listify
        def sl(key):
            # Take array at idx, mask it, tolist
            arr = batch[key][idx]
            return arr[mask].tolist()
            
        fp.t = sl('t')
        fp.s = sl('s')
        fp.s_d = sl('s_d')
        fp.s_dd = sl('s_dd')
        # fp.s_ddd = sl('s_ddd') # Not stored in batch usually? I calculated only masked for cost...
        # If I need s_ddd, I should store it or recalculate.
        # Cost calculation created local s_ddd.
        # Let's skip s_ddd/d_ddd for now or if critical, add to batch.
        # Existing code doesn't explicitly store ddd in FrenetPath usually? Ah, dataclass has it.
        # If I don't fill it, it's fine?
        # Let's leave empty or fill if easy.
        
        fp.d = sl('d')
        fp.d_d = sl('d_d')
        fp.d_dd = sl('d_dd')
        
        fp.x = sl('x')
        fp.y = sl('y')
        fp.yaw = sl('yaw')
        fp.c = sl('c')
        fp.v = sl('v')
        fp.a = sl('a')
        
        fp.cost = float(batch['cost'][idx])
        
        return fp
