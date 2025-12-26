"""Frenet Optimal Trajectory Planner.

This module implements the Frenet-Frame based optimal trajectory generation
for autonomous driving.

Reference:
Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios 
in a Frenet Frame" (2010)
"""

import copy
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from loguru import logger

from ..core.data_structures import FrenetPath, FrenetState, EgoVehicleState
from ..core.coordinate_converter import CartesianFrenetConverter
from .cubic_spline import CubicSpline2D


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


@dataclass(frozen=True)
class TimeCache:
    """Cached time powers and inverse matrices for polynomial generation."""
    t: np.ndarray
    t2: np.ndarray
    t3: np.ndarray
    t4: np.ndarray
    t5: np.ndarray
    quartic_A_inv: np.ndarray
    quintic_A_inv: np.ndarray


@dataclass(frozen=True)
class LongitudinalProfile:
    """Precomputed longitudinal trajectory components for a target velocity."""
    t: np.ndarray
    s: np.ndarray
    s_d: np.ndarray
    s_dd: np.ndarray
    s_ddd: np.ndarray


@dataclass(frozen=True)
class LateralProfile:
    """Precomputed lateral trajectory components for a target offset."""
    d: np.ndarray
    d_d: np.ndarray
    d_dd: np.ndarray
    d_ddd: np.ndarray


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
        """Plan optimal trajectory from current state.
        
        Args:
            ego_state: Current ego vehicle state
            static_obstacles: Static obstacle positions [n_obstacles, 2]
            dynamic_obstacles: Dynamic obstacles with time dimension [n_obs, time_steps, 2]
            target_speed: Desired target speed [m/s]
            constraint_overrides: Optional dictionary to override constraints (e.g. max_accel)
            
        Returns:
            Best trajectory, or None if no valid trajectory found
        """
        # Convert ego state to Frenet frame
        frenet_state = self._cartesian_to_frenet_state(ego_state)
        
        if frenet_state is None:
            logger.warning("Failed to convert ego state to Frenet frame")
            return None
        
        # Generate candidate paths
        fp_list = self._generate_frenet_paths(
            frenet_state, 
            target_speed
        )
        
        # Convert to global coordinates
        fp_list = self._calc_global_paths(fp_list)
        
        # Check path validity
        fp_dict = self._check_paths(fp_list, static_obstacles, dynamic_obstacles, constraint_overrides)
        
        # Select best path
        best_path = self._select_best_path(fp_dict)
        
        if best_path is not None:
            logger.debug(f"Found valid path with cost {best_path.cost:.2f}")
        else:
            logger.warning("No valid path found")
        
        return best_path
    
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
    
    def _generate_frenet_paths(
        self,
        frenet_state: FrenetState,
        target_speed: float
    ) -> List[FrenetPath]:
        """Generate candidate Frenet paths.
        
        Args:
            frenet_state: Current state in Frenet frame
            target_speed: Target speed [m/s]
            
        Returns:
            List of candidate Frenet paths
        """
        frenet_paths = []
        
        # Sample different time horizons (use instance variables instead of module constants)
        for Ti in np.arange(self.min_t, self.max_t, self.dt):
            time_cache = self._build_time_cache(Ti)
            tv_values = np.arange(
                target_speed - self.d_t_s * self.n_s_sample,
                target_speed + self.d_t_s * self.n_s_sample,
                self.d_t_s
            )
            tv_values = tv_values[tv_values >= 0.0]
            if tv_values.size == 0:
                continue

            di_values = np.arange(-self.max_road_width, self.max_road_width, self.d_road_w)
            if di_values.size == 0:
                continue

            lon_profiles = self._build_longitudinal_profiles(
                frenet_state, tv_values, Ti, time_cache
            )
            lat_profiles = self._build_lateral_profiles(
                frenet_state, di_values, Ti, time_cache
            )
            if not lon_profiles or not lat_profiles:
                continue

            for lon_profile in lon_profiles:
                for lat_profile in lat_profiles:
                    fp = FrenetPath(
                        t=lon_profile.t,
                        s=lon_profile.s,
                        s_d=lon_profile.s_d,
                        s_dd=lon_profile.s_dd,
                        s_ddd=lon_profile.s_ddd,
                        d=lat_profile.d,
                        d_d=lat_profile.d_d,
                        d_dd=lat_profile.d_dd,
                        d_ddd=lat_profile.d_ddd,
                    )
                    fp.cost = self._calculate_cost(fp, target_speed)
                    frenet_paths.append(fp)
        
        return frenet_paths
    
    def _generate_longitudinal_trajectory(
        self,
        frenet_state: FrenetState,
        target_velocity: float,
        time: float,
        time_cache: "TimeCache"
    ) -> FrenetPath:
        """Generate longitudinal trajectory using quartic polynomial.
        
        Args:
            frenet_state: Current Frenet state
            target_velocity: Target velocity [m/s]
            time: Time horizon [s]
            
        Returns:
            Frenet path with longitudinal trajectory
        """
        fp = FrenetPath()
        
        t = time_cache.t
        t2 = time_cache.t2
        t3 = time_cache.t3
        t4 = time_cache.t4
        a0 = frenet_state.s
        a1 = frenet_state.s_d
        a2 = frenet_state.s_dd / 2.0
        b = np.array([
            target_velocity - a1 - 2.0 * a2 * time,
            -2.0 * a2
        ])
        a3, a4 = time_cache.quartic_A_inv @ b
        fp.t = t.tolist()
        fp.s = (a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4).tolist()
        fp.s_d = (a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3).tolist()
        fp.s_dd = (2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t2).tolist()
        fp.s_ddd = (6.0 * a3 + 24.0 * a4 * t).tolist()
        
        return fp
    
    def _generate_lateral_trajectory(
        self,
        fp_lon: FrenetPath,
        lateral_offset: float,
        frenet_state: FrenetState,
        time: float,
        time_cache: "TimeCache"
    ) -> FrenetPath:
        """Generate lateral trajectory using quintic polynomial.
        
        Args:
            fp_lon: Longitudinal trajectory
            lateral_offset: Target lateral offset [m]
            frenet_state: Current Frenet state
            time: Time horizon [s]
            
        Returns:
            Complete Frenet path
        """
        fp = copy.deepcopy(fp_lon)
        
        t = time_cache.t
        t2 = time_cache.t2
        t3 = time_cache.t3
        t4 = time_cache.t4
        t5 = time_cache.t5
        a0 = frenet_state.d
        a1 = frenet_state.d_d
        a2 = frenet_state.d_dd / 2.0
        b = np.array([
            lateral_offset - a0 - a1 * time - a2 * time * time,
            -a1 - 2.0 * a2 * time,
            -2.0 * a2
        ])
        a3, a4, a5 = time_cache.quintic_A_inv @ b
        fp.d = (a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5).tolist()
        fp.d_d = (a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3 + 5.0 * a5 * t4).tolist()
        fp.d_dd = (2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t2 + 20.0 * a5 * t3).tolist()
        fp.d_ddd = (6.0 * a3 + 24.0 * a4 * t + 60.0 * a5 * t2).tolist()
        
        return fp

    def _build_time_cache(
        self,
        time: float
    ) -> "TimeCache":
        """Precompute time powers and coefficient inverses for vectorized evaluation."""
        t = np.arange(0.0, time, self.dt)
        t2 = t * t
        t3 = t2 * t
        t4 = t2 * t2
        t5 = t4 * t
        t_scalar = float(time)
        quartic_A = np.array([
            [3.0 * t_scalar ** 2, 4.0 * t_scalar ** 3],
            [6.0 * t_scalar, 12.0 * t_scalar ** 2]
        ])
        quintic_A = np.array([
            [t_scalar ** 3, t_scalar ** 4, t_scalar ** 5],
            [3.0 * t_scalar ** 2, 4.0 * t_scalar ** 3, 5.0 * t_scalar ** 4],
            [6.0 * t_scalar, 12.0 * t_scalar ** 2, 20.0 * t_scalar ** 3]
        ])
        return TimeCache(
            t=t,
            t2=t2,
            t3=t3,
            t4=t4,
            t5=t5,
            quartic_A_inv=np.linalg.inv(quartic_A),
            quintic_A_inv=np.linalg.inv(quintic_A)
        )

    def _build_longitudinal_profiles(
        self,
        frenet_state: FrenetState,
        target_velocities: np.ndarray,
        time: float,
        time_cache: TimeCache
    ) -> List[LongitudinalProfile]:
        t = time_cache.t
        t2 = time_cache.t2
        t3 = time_cache.t3
        t4 = time_cache.t4

        a0 = frenet_state.s
        a1 = frenet_state.s_d
        a2 = frenet_state.s_dd / 2.0

        tv_values = np.asarray(target_velocities, dtype=float)
        b = np.column_stack([
            tv_values - a1 - 2.0 * a2 * time,
            np.full(tv_values.shape, -2.0 * a2)
        ])
        coeffs = b @ time_cache.quartic_A_inv.T
        a3 = coeffs[:, 0][:, None]
        a4 = coeffs[:, 1][:, None]

        s = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4
        s_d = a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3
        s_dd = 2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t2
        s_ddd = 6.0 * a3 + 24.0 * a4 * t

        profiles = []
        for idx in range(tv_values.shape[0]):
            profiles.append(LongitudinalProfile(
                t=t,
                s=s[idx],
                s_d=s_d[idx],
                s_dd=s_dd[idx],
                s_ddd=s_ddd[idx],
            ))
        return profiles

    def _build_lateral_profiles(
        self,
        frenet_state: FrenetState,
        lateral_offsets: np.ndarray,
        time: float,
        time_cache: TimeCache
    ) -> List[LateralProfile]:
        t = time_cache.t
        t2 = time_cache.t2
        t3 = time_cache.t3
        t4 = time_cache.t4
        t5 = time_cache.t5

        a0 = frenet_state.d
        a1 = frenet_state.d_d
        a2 = frenet_state.d_dd / 2.0

        di_values = np.asarray(lateral_offsets, dtype=float)
        b = np.column_stack([
            di_values - a0 - a1 * time - a2 * time * time,
            np.full(di_values.shape, -a1 - 2.0 * a2 * time),
            np.full(di_values.shape, -2.0 * a2)
        ])
        coeffs = b @ time_cache.quintic_A_inv.T
        a3 = coeffs[:, 0][:, None]
        a4 = coeffs[:, 1][:, None]
        a5 = coeffs[:, 2][:, None]

        d = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
        d_d = a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3 + 5.0 * a5 * t4
        d_dd = 2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t2 + 20.0 * a5 * t3
        d_ddd = 6.0 * a3 + 24.0 * a4 * t + 60.0 * a5 * t2

        profiles = []
        for idx in range(di_values.shape[0]):
            profiles.append(LateralProfile(
                d=d[idx],
                d_d=d_d[idx],
                d_dd=d_dd[idx],
                d_ddd=d_ddd[idx],
            ))
        return profiles
    
    def _calculate_cost(
        self,
        fp: FrenetPath,
        target_speed: float
    ) -> float:
        """Calculate total cost of a trajectory.
        
        Args:
            fp: Frenet path
            target_speed: Target speed [m/s]
            
        Returns:
            Total cost
        """
        # Lateral costs
        Jp = np.sum(np.square(fp.d_ddd))  # Lateral jerk
        Jd = (fp.d[-1]) ** 2  # Final lateral offset
        
        # Longitudinal costs
        Js = np.sum(np.square(fp.s_ddd))  # Longitudinal jerk
        Jv = (target_speed - fp.s_d[-1]) ** 2  # Speed deviation
        
        # Time cost
        Jt = fp.t[-1]
        
        # Combine costs
        lat_cost = self.k_j * Jp + self.k_t * Jt + self.k_d * Jd
        lon_cost = self.k_j * Js + self.k_t * Jt + self.k_s_dot * Jv
        
        total_cost = self.k_lat * lat_cost + self.k_lon * lon_cost
        
        return total_cost
    
    def _calc_global_paths(
        self,
        fp_list: List[FrenetPath]
    ) -> List[FrenetPath]:
        """Convert Frenet paths to global coordinates.
        
        Args:
            fp_list: List of Frenet paths
            
        Returns:
            List of paths with global coordinates filled
        """
        for fp in fp_list:
            for i in range(len(fp.s)):
                ix, iy = self.csp.calc_position(fp.s[i])
                if ix is None or iy is None:
                    break
                
                i_yaw = self.csp.calc_yaw(fp.s[i])
                i_kappa = self.csp.calc_curvature(fp.s[i])
                i_dkappa = self.csp.calc_curvature_rate(fp.s[i])
                
                if any(v is None for v in [i_yaw, i_kappa, i_dkappa]):
                    break
                
                s_condition = [fp.s[i], fp.s_d[i], fp.s_dd[i]]
                d_condition = [fp.d[i], fp.d_d[i], fp.d_dd[i]]
                
                try:
                    x, y, theta, kappa, v, a = self.converter.frenet_to_cartesian(
                        fp.s[i], ix, iy, i_yaw, i_kappa, i_dkappa,
                        s_condition, d_condition
                    )
                    
                    fp.x.append(x)
                    fp.y.append(y)
                    fp.yaw.append(theta)
                    fp.c.append(kappa)
                    fp.v.append(v)
                    fp.a.append(a)
                except Exception as e:
                    logger.debug(f"Error converting to global: {e}")
                    break
        
        return fp_list
    
    def _check_paths(
        self,
        fp_list: List[FrenetPath],
        static_obstacles: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray] = None,
        constraint_overrides: Optional[Dict[str, float]] = None
    ) -> dict:
        """Check path validity and categorize paths.
        
        Args:
            fp_list: List of Frenet paths
            static_obstacles: Static obstacle positions [n_obstacles, 2]
            dynamic_obstacles: Dynamic obstacle trajectories [n_obs, time_steps, 2]
            constraint_overrides: Optional overrides for max_speed, max_accel, max_curvature
            
        Returns:
            Dictionary categorizing paths by validity
        """
        path_dict = {
            'max_speed_error': [],
            'max_accel_error': [],
            'max_curvature_error': [],
            'collision_error': [],
            'ok': []
        }
        
        # Resolve constraints
        c_max_speed = self.max_speed
        c_max_accel = self.max_accel
        c_max_curvature = self.max_curvature
        
        if constraint_overrides:
            c_max_speed = constraint_overrides.get('max_speed', c_max_speed)
            c_max_accel = constraint_overrides.get('max_accel', c_max_accel)
            c_max_curvature = constraint_overrides.get('max_curvature', c_max_curvature)
        
        for fp in fp_list:
            if len(fp.x) == 0:
                continue
            
            # Check speed limit (use generator expression for early termination)
            if any(v > c_max_speed for v in fp.v):
                path_dict['max_speed_error'].append(fp)
            # Check acceleration limit (use generator expression for early termination)
            elif any(abs(a) > c_max_accel for a in fp.a):
                path_dict['max_accel_error'].append(fp)
            # Check curvature limit (use generator expression for early termination)
            elif any(abs(c) > c_max_curvature for c in fp.c):
                path_dict['max_curvature_error'].append(fp)
            # Check collision
            elif not self._check_collision(fp, static_obstacles, dynamic_obstacles):
                path_dict['collision_error'].append(fp)
            else:
                path_dict['ok'].append(fp)
        
        return path_dict
    
    def _check_collision(
        self,
        fp: FrenetPath,
        static_obstacles: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray] = None
    ) -> bool:
        """Check if path collides with obstacles (vectorized implementation).
        
        Args:
            fp: Frenet path
            static_obstacles: Static obstacle positions [n_obstacles, 2]
            dynamic_obstacles: Dynamic obstacle trajectories [n_obs, time_steps, 2]
            
        Returns:
            True if no collision, False otherwise
        """
        if len(fp.x) == 0:
            return True
        
        # Ensure consistency between t and x/y/s
        # x, y calculation might stop early if path goes out of bounds, 
        # but t is pre-calculated. We must synchronize them.
        min_len = min(len(fp.x), len(fp.t))
        
        path_x = np.array(fp.x[:min_len])
        path_y = np.array(fp.y[:min_len])
        path_t = np.array(fp.t[:min_len])
        
        path_points = np.stack([path_x, path_y], axis=1)  # [n_path, 2]
        
        inflated_radius = max(
            self.robot_radius + self.obstacle_radius,
            1e-6
        )
        sq_rubicon = inflated_radius ** 2

        # Coarse AABB filter to skip unnecessary distance checks
        path_min = np.min(path_points, axis=0) - inflated_radius
        path_max = np.max(path_points, axis=0) + inflated_radius
        
        # 1. Static obstacles: Check all path points against all obstacles
        # Shape: (n_path, 1, 2) - (1, n_static, 2) -> (n_path, n_static, 2)
        if static_obstacles is not None and len(static_obstacles) > 0:
            static_mask = (
                (static_obstacles[:, 0] >= path_min[0]) &
                (static_obstacles[:, 0] <= path_max[0]) &
                (static_obstacles[:, 1] >= path_min[1]) &
                (static_obstacles[:, 1] <= path_max[1])
            )
            if not np.any(static_mask):
                static_candidates = None
            else:
                static_candidates = static_obstacles[static_mask]

            if static_candidates is not None and len(static_candidates) > 0:
                diff = path_points[:, None, :] - static_candidates[None, :, :]
                sq_dists = np.sum(diff ** 2, axis=2)
                if np.any(sq_dists <= sq_rubicon):
                    return False

        # 2. Dynamic obstacles: Check time-aligned points
        if (dynamic_obstacles is not None and 
            dynamic_obstacles.size > 0 and 
            dynamic_obstacles.shape[-1] == 2):
            
            # Coarse filter by obstacle trajectory AABB
            obs_min = np.min(dynamic_obstacles, axis=1)
            obs_max = np.max(dynamic_obstacles, axis=1)
            dyn_mask = (
                (obs_max[:, 0] >= path_min[0]) &
                (obs_min[:, 0] <= path_max[0]) &
                (obs_max[:, 1] >= path_min[1]) &
                (obs_min[:, 1] <= path_max[1])
            )
            if not np.any(dyn_mask):
                return True

            dynamic_candidates = dynamic_obstacles[dyn_mask]
            n_obs, n_time, _ = dynamic_candidates.shape
            
            # Map each path point index to a time index in the obstacle array
            # time_indices[i] corresponds to the time step for path point i
            time_indices = np.round(path_t / self.dt).astype(int)
            
            # Clamp indices to valid range [0, n_time - 1]
            time_indices = np.clip(time_indices, 0, n_time - 1)
            
            # Select relevant obstacle positions for each path point
            # Shape: [n_path, n_obs, 2]
            # dynamic_obstacles is [n_obs, n_time, 2] -> transpose to [n_time, n_obs, 2]
            # Then fancy index with time_indices -> [n_path, n_obs, 2]
            relevant_obs = dynamic_candidates.transpose(1, 0, 2)[time_indices]
            
            # Vectorized distance check
            # path_points: [n_path, 2] -> [n_path, 1, 2]
            # relevant_obs: [n_path, n_obs, 2]
            diff = path_points[:, None, :] - relevant_obs
            sq_dists = np.sum(diff ** 2, axis=2)
            
            if np.any(sq_dists <= sq_rubicon):
                return False

        return True
    
    def _select_best_path(
        self,
        path_dict: dict
    ) -> Optional[FrenetPath]:
        """Select the best valid path.
        
        Args:
            path_dict: Dictionary of categorized paths
            
        Returns:
            Best path, or None if no valid path exists
        """
        if len(path_dict['ok']) == 0:
            return None
        
        # Find minimum cost path
        min_cost = float('inf')
        best_path = None
        
        for fp in path_dict['ok']:
            if fp.cost < min_cost:
                min_cost = fp.cost
                best_path = fp
        
        return best_path
