"""Frenet Optimal Trajectory Planner.

This module implements the Frenet-Frame based optimal trajectory generation
for autonomous driving.

Reference:
Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios 
in a Frenet Frame" (2010)
"""

import copy
from itertools import islice

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

# Curvature is only constrained at samples faster than this [m/s]. Below it
# the per-sample arc length is sub-centimetre, so the discrete curvature
# estimate of a candidate that re-aligns a stopped, yaw-misaligned vehicle
# with the reference tangent diverges (Δyaw over ~mm of travel) even though
# the manoeuvre is an ordinary near-standstill steering correction. Without
# this gate a vehicle that stops with any heading offset can never produce a
# feasible restart candidate and deadlocks in EMERGENCY.
LOW_SPEED_CURVATURE_GATE = 0.5


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
    
    # Reject candidates whose lateral offset approaches the curvature center of
    # the reference path (Frenet->Cartesian singular at 1 - kappa_ref * d = 0)
    SINGULARITY_EPS = 0.05

    # Below this longitudinal speed [m/s] the spatial derivative d' = d_dot/s_dot
    # is ill-defined; fall back to d' = d'' = 0 (heading on the reference tangent).
    EPS_S_DOT = 1e-3

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
        # Use high-level CoordinateConverter which handles nearest point search
        from ..core.coordinate_converter import CoordinateConverter
        self.converter = CoordinateConverter(reference_path)
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

        # Chance-constrained planning: allowed fraction of colliding prediction
        # samples (0.0 = robust/worst-case). Only used when a prediction
        # distribution is supplied to plan().
        self.chance_epsilon = float(kwargs.get("chance_epsilon", 0.0))

        # Margin inflation for the single-sample dynamic collision check only
        # (1.0 = no inflation). Not applied to the chance-constrained
        # distribution check nor to static obstacles.
        self.collision_margin_inflation = float(
            kwargs.get("collision_margin_inflation", 1.0)
        )

        # Optional multi-circle ego footprint (EgoFootprint). When set, every
        # collision check evaluates all footprint circles (radius
        # footprint.radius) placed along the candidate-path heading instead of
        # the single robot_radius circle at the path point.
        self.footprint = kwargs.get("footprint", None)

        # Ego curvature for the Frenet initial conditions: curvature at index 1
        # of the previously adopted path (the simulator advances to index 1).
        # Kept unchanged on a failed plan — the ego has not moved, so the
        # same-step escalation retry must see the same curvature. The simulator
        # calls reset_ego_curvature() when it applies the straight-line
        # emergency stop instead.
        self._last_kappa = 0.0

        logger.info(f"Frenet Planner initialized with dt={dt}s, "
                   f"max_speed={max_speed}m/s, max_accel={max_accel}m/s², "
                   f"robot_radius={robot_radius}m, obstacle_radius={obstacle_radius}m, "
                   f"margin_inflation={self.collision_margin_inflation}, "
                   f"footprint={'multi_circle' if self.footprint is not None else 'circle'}, "
                   f"time_horizon=[{min_t:.1f}, {max_t:.1f}]s")
    
    def plan(
        self,
        ego_state: EgoVehicleState,
        static_obstacles: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray] = None,
        target_speed: float = TARGET_SPEED,
        constraint_overrides: Optional[Dict[str, float]] = None,
        dynamic_obstacles_distribution: Optional[np.ndarray] = None
    ) -> Optional[FrenetPath]:
        """Plan optimal trajectory from current state.

        Args:
            ego_state: Current ego vehicle state
            static_obstacles: Static obstacle positions [n_obstacles, 2]
            dynamic_obstacles: Dynamic obstacles with time dimension [n_obs, time_steps, 2]
            target_speed: Desired target speed [m/s]
            constraint_overrides: Optional dictionary to override constraints (e.g. max_accel)
            dynamic_obstacles_distribution: Optional sampled prediction distribution
                [n_samples, n_obs, time_steps, 2]. When provided, collision checking
                is chance-constrained over the samples (see ``chance_epsilon``) instead
                of using the single representative ``dynamic_obstacles``.

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
        fp_dict = self._check_paths(
            fp_list, static_obstacles, dynamic_obstacles, constraint_overrides,
            dynamic_obstacles_distribution
        )

        # Rejection breakdown of the last planning call. Diagnostic only (e.g.
        # to verify predictions actually constrain the chosen path); never
        # feeds back into planning.
        self.last_check_stats = {k: len(v) for k, v in fp_dict.items()}

        # Select best path
        best_path = self._select_best_path(fp_dict)
        
        if best_path is not None:
            logger.debug(f"Found valid path with cost {best_path.cost:.2f}")
        else:
            logger.warning("No valid path found")

        if best_path is not None and len(best_path.c) > 1:
            self._last_kappa = float(best_path.c[1])

        return best_path

    def reset_ego_curvature(self):
        """Reset the cached ego curvature (the ego is moving straight).

        Called by the simulator after a straight-line emergency stop, where the
        previously adopted path's curvature no longer describes the ego motion.
        """
        self._last_kappa = 0.0
    
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
        try:
            rs, rx, ry, rtheta, rkappa, rdkappa = self.converter.find_nearest_point_on_path(
                ego_state.x, ego_state.y
            )
        except Exception as e:
            logger.error(f"Failed to find nearest point: {e}")
            return None
        
        # Convert to Frenet
        try:
            s_condition, d_condition = self.converter.cartesian_to_frenet(
                rs, rx, ry, rtheta, rkappa, rdkappa,
                ego_state.x, ego_state.y, ego_state.v, ego_state.a,
                ego_state.yaw, self._last_kappa
            )

            # cartesian_to_frenet returns Apollo-convention lateral derivatives
            # taken w.r.t. arc length (d' = dd/ds), but the lateral quintic is
            # built on the time grid, so convert to time derivatives here:
            # d_dot = d'*s_dot (= v*sin(delta_theta)), d_ddot = d''*s_dot^2 + d'*s_ddot.
            s, s_d, s_dd = s_condition
            d, d_p, d_pp = d_condition
            d_d = d_p * s_d
            d_dd = d_pp * s_d ** 2 + d_p * s_dd

            return FrenetState(s=s, s_d=s_d, s_dd=s_dd, d=d, d_d=d_d, d_dd=d_dd)
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
        
        # Sample different time horizons, inclusive of max_t. Floor (not
        # round) so the grid never overshoots max_t when (max_t - min_t) is
        # not an integer multiple of dt: predictions only cover max_t, and a
        # longer horizon would be collision-checked against clamped, stale
        # obstacle samples.
        n_ti = int((self.max_t - self.min_t) / self.dt + 1e-9)
        for Ti in self.min_t + np.arange(n_ti + 1) * self.dt:
            time_cache = self._build_time_cache(Ti)
            tv_values = np.arange(
                target_speed - self.d_t_s * self.n_s_sample,
                target_speed + self.d_t_s * self.n_s_sample,
                self.d_t_s
            )
            tv_values = tv_values[tv_values >= 0.0]
            if tv_values.size == 0:
                continue

            # Symmetric lateral grid centred on the reference line: always
            # contains d = 0 exactly and never exceeds the road half-width.
            n_side = int(self.max_road_width / self.d_road_w + 1e-9)
            di_values = np.arange(-n_side, n_side + 1) * self.d_road_w
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
        # Inclusive grid: the terminal point t = time carries the boundary
        # conditions (d(Ti) = di, s_d(Ti) = tv) and the terminal costs.
        n_steps = int(round(time / self.dt))
        t = np.arange(n_steps + 1) * self.dt
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
        """Convert Frenet paths to global coordinates (Vectorized).
        
        Args:
            fp_list: List of Frenet paths
            
        Returns:
            List of paths with global coordinates filled
        """
        if not fp_list:
            return []

        # 1. Flatten all path points
        # Pre-allocate arrays for better performance
        total_points = sum(len(fp.s) for fp in fp_list)
        if total_points == 0:
            return fp_list

        all_s = np.zeros(total_points)
        all_s_d = np.zeros(total_points)
        all_s_dd = np.zeros(total_points)
        all_d = np.zeros(total_points)
        all_d_d = np.zeros(total_points)
        all_d_dd = np.zeros(total_points)
        
        split_indices = np.zeros(len(fp_list), dtype=int)
        
        cursor = 0
        for i, fp in enumerate(fp_list):
            n = len(fp.s)
            all_s[cursor:cursor+n] = fp.s
            all_s_d[cursor:cursor+n] = fp.s_d
            all_s_dd[cursor:cursor+n] = fp.s_dd
            all_d[cursor:cursor+n] = fp.d
            all_d_d[cursor:cursor+n] = fp.d_d
            all_d_dd[cursor:cursor+n] = fp.d_dd
            cursor += n
            split_indices[i] = cursor

        # 2. Batch calculate reference points using vectorized CubicSpline
        # Note: CubicSpline2D handles bounds checks internally (returns NaN)
        ix, iy = self.csp.calc_position(all_s)
        i_yaw = self.csp.calc_yaw(all_s)
        i_kappa = self.csp.calc_curvature(all_s)
        i_dkappa = self.csp.calc_curvature_rate(all_s)

        # 3. Batch conversion
        # CoordinateConverter.frenet_to_cartesian is now vectorized.
        # The lateral arrays hold time derivatives (d_dot, d_ddot) while
        # frenet_to_cartesian expects spatial ones: d' = d_dot/s_dot,
        # d'' = (d_ddot - d'*s_ddot)/s_dot^2. Near standstill the ratio is
        # ill-defined, so fall back to d' = d'' = 0 (heading aligned with the
        # reference tangent); the continuity guard in _check_paths covers the rest.
        moving = np.abs(all_s_d) > self.EPS_S_DOT
        safe_s_d = np.where(moving, all_s_d, 1.0)
        d_prime = np.where(moving, all_d_d / safe_s_d, 0.0)
        d_pprime = np.where(
            moving,
            (all_d_dd - d_prime * all_s_dd) / (safe_s_d * safe_s_d),
            0.0
        )

        s_condition = (all_s, all_s_d, all_s_dd)
        d_condition = (all_d, d_prime, d_pprime)

        try:
             x, y, theta, kappa, v, a = self.converter.frenet_to_cartesian(
                all_s, ix, iy, i_yaw, i_kappa, i_dkappa,
                s_condition, d_condition
            )
        except Exception as e:
            logger.error(f"Vectorized conversion failed: {e}")
            # Fallback or empty return?
            # If conversion fails for one point, it might be due to NaNs.
            # We continue, let the NaN values propagate, and filter later if needed.
            # But here we just re-raise or return broken paths.
            # Usually vector calls shouldn't raise exceptions unless dimension mismatch.
            return fp_list

        # Frenet->Cartesian is singular at 1 - kappa_ref * d <= 0 (the lateral
        # offset reaches the curvature center): heading flips by pi and the
        # converted velocity/acceleration are meaningless, yet finite, so they
        # would pass the constraint checks. Invalidate any candidate containing
        # such a point by NaN-ing its first sample (the unflatten step below
        # then drops the whole path). Out-of-domain points (NaN curvature from
        # the spline) are NOT singular: they keep the legacy behaviour of
        # truncating the path to its valid prefix.
        one_minus_kappa_d = 1.0 - i_kappa * all_d
        singular = np.isfinite(one_minus_kappa_d) & (one_minus_kappa_d <= self.SINGULARITY_EPS)
        if np.any(singular):
            seg_start = 0
            for seg_end in split_indices:
                if np.any(singular[seg_start:seg_end]):
                    x[seg_start] = np.nan
                seg_start = seg_end

        # 4. Unflatten results and populate paths
        # x, y, etc. are numpy arrays. Split them back.
        # np.split expects indices where splits occur (excluding 0, up to end)
        # split_indices contains end indices for each segment. 
        # We need to drop the last one for np.split, or just iterate with cursor.
        # Iterating with cursor is faster than np.split for many small arrays
        
        cursor = 0
        for i, end_index in enumerate(split_indices):
            fp = fp_list[i]
            # Ensure we cast to list as per data structure expectation
            # Use tolist() which is fast for numpy elements
            # Truncate path if it contains NaNs (e.g. goes out of spline bounds)
            # This replicates the behavior of the loop-based implementation which stopped
            # appending when calc_position returned None.
            x_segment = x[cursor:end_index]
            nan_mask = np.isnan(x_segment)
            
            if np.any(nan_mask):
                first_nan_idx = int(np.argmax(nan_mask))

                # Truncate the valid prefix in lockstep across the Cartesian
                # AND Frenet arrays, so every kept sample is covered by the
                # constraint and collision checks (no unchecked tail) and
                # len(x) == len(t) holds for every emitted path. A path needs
                # at least indices 0 and 1 to be executable (the simulator
                # advances to index 1); shorter ones are emptied entirely so
                # the failure reaches the state machine instead of a
                # degenerate "successful" plan. Note: fp.cost was computed on
                # the un-truncated profile, so terminal-cost terms refer to
                # the original horizon end.
                keep = first_nan_idx if first_nan_idx >= 2 else 0
                fp.x = x_segment[:keep].tolist()
                fp.y = y[cursor:end_index][:keep].tolist()
                fp.yaw = theta[cursor:end_index][:keep].tolist()
                fp.c = kappa[cursor:end_index][:keep].tolist()
                fp.v = v[cursor:end_index][:keep].tolist()
                fp.a = a[cursor:end_index][:keep].tolist()
                for name in ("t", "s", "s_d", "s_dd", "s_ddd",
                             "d", "d_d", "d_dd", "d_ddd"):
                    setattr(fp, name, getattr(fp, name)[:keep])
            else:
                # No NaNs, copy full reference
                fp.x = x_segment.tolist()
                fp.y = y[cursor:end_index].tolist()
                fp.yaw = theta[cursor:end_index].tolist()
                fp.c = kappa[cursor:end_index].tolist()
                fp.v = v[cursor:end_index].tolist()
                fp.a = a[cursor:end_index].tolist()
            

        
            cursor = end_index
        
        return fp_list
    
    def _check_paths(
        self,
        fp_list: List[FrenetPath],
        static_obstacles: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray] = None,
        constraint_overrides: Optional[Dict[str, float]] = None,
        dynamic_obstacles_distribution: Optional[np.ndarray] = None
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

            # Defensive guard: every constraint below assumes the Cartesian and
            # Frenet arrays describe the same samples. _calc_global_paths keeps
            # them in lockstep; reject externally constructed paths that don't.
            if len(fp.x) != len(fp.t):
                continue

            # Drop paths carrying non-finite kinematics: NaN compares False
            # against every limit below and would silently pass all checks.
            if not (np.all(np.isfinite(fp.v)) and np.all(np.isfinite(fp.a))
                    and np.all(np.isfinite(fp.c))):
                continue

            # Drop geometrically discontinuous paths: when the ego heading is
            # nearly orthogonal to the reference tangent, the Frenet initial
            # conditions (tan(dtheta)) explode and the converted positions jump
            # by kilometres while v/a/c stay finite. A physical path cannot move
            # more than ~max_speed*dt per sample.
            if len(fp.x) >= 2:
                step_len = np.hypot(np.diff(fp.x), np.diff(fp.y))
                if np.max(step_len) > max(c_max_speed, self.max_speed) * self.dt * 3.0:
                    continue

            # Check speed/acceleration limits from index 1 on: index 0 is the
            # current state, which the planner cannot change; rejecting on it
            # would discard all candidates whenever a tightened limit (e.g.
            # CAUTION max_speed, or CAUTION max_accel right after an emergency
            # brake at 2x max_accel) lies below the present value and force a
            # needless EMERGENCY escalation.
            if any(v > c_max_speed for v in islice(fp.v, 1, None)):
                path_dict['max_speed_error'].append(fp)
            elif any(abs(a) > c_max_accel for a in islice(fp.a, 1, None)):
                path_dict['max_accel_error'].append(fp)
            # Curvature is also checked from index 1: c[0] is the current state,
            # which can transiently exceed the (non-relaxable, kinematic) limit
            # right after an emergency manoeuvre and must not veto every candidate.
            # Samples below LOW_SPEED_CURVATURE_GATE are exempt: their discrete
            # curvature is a numerical artefact of sub-centimetre arc lengths
            # (near-standstill steering), not a violated turning radius.
            elif any(abs(c) > c_max_curvature and v > LOW_SPEED_CURVATURE_GATE
                     for c, v in islice(zip(fp.c, fp.v), 1, None)):
                path_dict['max_curvature_error'].append(fp)
            # Check collision (chance-constrained over the distribution when provided)
            elif not self._path_is_collision_free(
                fp, static_obstacles, dynamic_obstacles, dynamic_obstacles_distribution
            ):
                path_dict['collision_error'].append(fp)
            else:
                path_dict['ok'].append(fp)
        
        return path_dict
    
    def _path_is_collision_free(
        self,
        fp: FrenetPath,
        static_obstacles: np.ndarray,
        dynamic_obstacles: Optional[np.ndarray],
        dynamic_distribution: Optional[np.ndarray]
    ) -> bool:
        """Route to single-sample or chance-constrained collision checking."""
        if dynamic_distribution is not None and dynamic_distribution.size > 0:
            return self._check_collision_distribution(
                fp, static_obstacles, dynamic_distribution, self.chance_epsilon
            )
        return self._check_collision(fp, static_obstacles, dynamic_obstacles)

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
        geom = self._path_collision_geometry(fp, self.collision_margin_inflation)
        if geom is None:
            return True
        path_points, path_t, path_min, path_max, sq_rubicon, sq_rubicon_dyn = geom

        if self._hits_static(path_points, path_min, path_max, static_obstacles, sq_rubicon):
            return False
        if self._hits_dynamic(path_points, path_t, path_min, path_max, dynamic_obstacles, sq_rubicon_dyn):
            return False
        return True

    def _check_collision_distribution(
        self,
        fp: FrenetPath,
        static_obstacles: np.ndarray,
        dynamic_distribution: np.ndarray,
        epsilon: float
    ) -> bool:
        """Chance-constrained collision check over a sampled prediction distribution.

        A path is feasible if it stays collision-free under at least a
        ``(1 - epsilon)`` fraction of the sampled futures (``epsilon = 0`` enforces
        the robust/worst-case constraint that no sample collides). Static obstacles
        are always treated as hard constraints.

        Args:
            fp: Frenet path
            static_obstacles: Static obstacle positions [n_obstacles, 2]
            dynamic_distribution: Sampled trajectories [n_samples, n_obs, time_steps, 2]
            epsilon: Allowed fraction of colliding samples

        Returns:
            True if the path satisfies the chance constraint, False otherwise
        """
        # Margin inflation is intentionally NOT applied here: the chance
        # constraint consumes the raw sampled distribution.
        geom = self._path_collision_geometry(fp)
        if geom is None:
            return True
        path_points, path_t, path_min, path_max, sq_rubicon, _ = geom

        # Static obstacles are sample-independent and remain hard constraints.
        if self._hits_static(path_points, path_min, path_max, static_obstacles, sq_rubicon):
            return False

        if dynamic_distribution is None or dynamic_distribution.size == 0:
            return True

        n_samples = dynamic_distribution.shape[0]
        max_violations = int(np.floor(epsilon * n_samples))
        violations = 0
        for k in range(n_samples):
            if self._hits_dynamic(
                path_points, path_t, path_min, path_max,
                dynamic_distribution[k], sq_rubicon
            ):
                violations += 1
                if violations > max_violations:
                    return False
        return True

    def _path_collision_geometry(self, fp: FrenetPath, dynamic_margin_inflation: float = 1.0):
        """Precompute path points, time stamps, AABB, and squared safety radii.

        ``dynamic_margin_inflation`` scales the combined radius used against
        dynamic obstacles only; static obstacles always use the nominal radius.

        With a multi-circle footprint, each path point is expanded into one
        point per footprint circle (offset along the path heading) and the
        safety radius becomes ``footprint.radius + obstacle_radius``; the
        downstream static/dynamic checks are unchanged because ``path_t`` is
        expanded in lockstep.

        Returns ``None`` for an empty path, otherwise a tuple
        ``(path_points, path_t, path_min, path_max, sq_rubicon, sq_rubicon_dyn)``.
        """
        if len(fp.x) == 0:
            return None

        # x/y may stop early if the path leaves the spline domain, but t is
        # pre-computed, so synchronize their lengths.
        min_len = min(len(fp.x), len(fp.t))
        path_x = np.array(fp.x[:min_len])
        path_y = np.array(fp.y[:min_len])
        path_t = np.array(fp.t[:min_len])
        path_points = np.stack([path_x, path_y], axis=1)  # [n_path, 2]

        if self.footprint is None:
            ego_radius = self.robot_radius
        else:
            ego_radius = self.footprint.radius
            # fp.yaw can be shorter than fp.x (heading from point differences);
            # pad by holding the last value.
            yaw = np.array(fp.yaw[:min_len])
            if len(yaw) < min_len:
                pad = yaw[-1] if len(yaw) > 0 else 0.0
                yaw = np.concatenate([yaw, np.full(min_len - len(yaw), pad)])
            directions = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)  # [n_path, 2]
            # [n_circles, n_path, 2] -> [n_circles * n_path, 2]
            offset_points = (
                path_points[None, :, :]
                + self.footprint.offsets[:, None, None] * directions[None, :, :]
            )
            n_circles = len(self.footprint.offsets)
            path_points = offset_points.reshape(n_circles * min_len, 2)
            path_t = np.tile(path_t, n_circles)

        inflated_radius = max(ego_radius + self.obstacle_radius, 1e-6)
        dyn_radius = inflated_radius * dynamic_margin_inflation
        sq_rubicon = inflated_radius ** 2
        sq_rubicon_dyn = dyn_radius ** 2
        aabb_radius = max(inflated_radius, dyn_radius)
        path_min = np.min(path_points, axis=0) - aabb_radius
        path_max = np.max(path_points, axis=0) + aabb_radius
        return path_points, path_t, path_min, path_max, sq_rubicon, sq_rubicon_dyn

    def _hits_static(self, path_points, path_min, path_max, static_obstacles, sq_rubicon) -> bool:
        """Return True if any path point collides with a static obstacle."""
        if static_obstacles is None or len(static_obstacles) == 0:
            return False

        static_mask = (
            (static_obstacles[:, 0] >= path_min[0]) &
            (static_obstacles[:, 0] <= path_max[0]) &
            (static_obstacles[:, 1] >= path_min[1]) &
            (static_obstacles[:, 1] <= path_max[1])
        )
        if not np.any(static_mask):
            return False

        static_candidates = static_obstacles[static_mask]
        diff = path_points[:, None, :] - static_candidates[None, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        return bool(np.any(sq_dists <= sq_rubicon))

    def _hits_dynamic(self, path_points, path_t, path_min, path_max, dynamic_obstacles, sq_rubicon) -> bool:
        """Return True if the path collides with one set of time-stamped obstacles.

        ``dynamic_obstacles`` has shape [n_obs, time_steps, 2].
        """
        if (dynamic_obstacles is None or
                dynamic_obstacles.size == 0 or
                dynamic_obstacles.shape[-1] != 2):
            return False

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
            return False

        dynamic_candidates = dynamic_obstacles[dyn_mask]
        n_time = dynamic_candidates.shape[1]

        # Map each path point index to a time index in the obstacle array.
        time_indices = np.round(path_t / self.dt).astype(int)
        time_indices = np.clip(time_indices, 0, n_time - 1)

        # [n_obs, n_time, 2] -> [n_time, n_obs, 2] -> fancy index -> [n_path, n_obs, 2]
        relevant_obs = dynamic_candidates.transpose(1, 0, 2)[time_indices]
        diff = path_points[:, None, :] - relevant_obs
        sq_dists = np.sum(diff ** 2, axis=2)
        return bool(np.any(sq_dists <= sq_rubicon))

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
