"""Frenet Optimal Trajectory Planner.

This module implements the Frenet-Frame based optimal trajectory generation
for autonomous driving.

Reference:
Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios 
in a Frenet Frame" (2010)
"""

import copy
import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

from ..core.data_structures import FrenetPath, FrenetState, EgoVehicleState
from ..core.coordinate_converter import CartesianFrenetConverter
from .cubic_spline import CubicSpline2D
from .quintic_polynomial import QuinticPolynomial, QuarticPolynomial


# Planning parameters
MAX_SPEED = 50.0 / 3.6  # Maximum speed [m/s]
MAX_ACCEL = 2.0  # Maximum acceleration [m/s²]
MAX_CURVATURE = 1.0  # Maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # Maximum road width [m]
D_ROAD_W = 1.0  # Road width sampling distance [m]
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
    """
    
    def __init__(
        self,
        reference_path: CubicSpline2D,
        max_speed: float = MAX_SPEED,
        max_accel: float = MAX_ACCEL,
        max_curvature: float = MAX_CURVATURE,
        dt: float = DT,
        robot_radius: float = ROBOT_RADIUS,
        obstacle_radius: float = 0.3,
        safety_buffer: float = 0.0,
        **kwargs
    ):
        self.csp = reference_path
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_curvature = max_curvature
        self.dt = dt
        self.converter = CartesianFrenetConverter()
        self.robot_radius = robot_radius
        self.obstacle_radius = obstacle_radius
        self.safety_buffer = safety_buffer

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
                   f"buffer={safety_buffer}m")
    
    def plan(
        self,
        ego_state: EgoVehicleState,
        obstacles: np.ndarray,
        target_speed: float = TARGET_SPEED
    ) -> Optional[FrenetPath]:
        """Plan optimal trajectory from current state.
        
        Args:
            ego_state: Current ego vehicle state
            obstacles: Obstacle positions [n_obstacles, 2]
            target_speed: Desired target speed [m/s]
            
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
        fp_dict = self._check_paths(fp_list, obstacles)
        
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
        
        # Sample different time horizons
        for Ti in np.arange(MIN_T, MAX_T, DT):
            # Longitudinal planning (velocity keeping)
            for tv in np.arange(
                target_speed - D_T_S * N_S_SAMPLE,
                target_speed + D_T_S * N_S_SAMPLE,
                D_T_S
            ):
                # Generate longitudinal trajectory
                fp_lon = self._generate_longitudinal_trajectory(
                    frenet_state, tv, Ti
                )
                
                # Sample different lateral positions
                for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):
                    # Generate lateral trajectory
                    fp = self._generate_lateral_trajectory(
                        fp_lon, di, frenet_state, Ti
                    )
                    
                    # Calculate cost
                    fp.cost = self._calculate_cost(fp, target_speed)
                    
                    frenet_paths.append(fp)
        
        return frenet_paths
    
    def _generate_longitudinal_trajectory(
        self,
        frenet_state: FrenetState,
        target_velocity: float,
        time: float
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
        
        lon_qp = QuarticPolynomial(
            frenet_state.s,
            frenet_state.s_d,
            frenet_state.s_dd,
            target_velocity,
            0.0,
            time
        )
        
        fp.t = [t for t in np.arange(0.0, time, DT)]
        fp.s = [lon_qp.calc_point(t) for t in fp.t]
        fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
        fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
        fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
        
        return fp
    
    def _generate_lateral_trajectory(
        self,
        fp_lon: FrenetPath,
        lateral_offset: float,
        frenet_state: FrenetState,
        time: float
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
        
        lat_qp = QuinticPolynomial(
            frenet_state.d,
            frenet_state.d_d,
            frenet_state.d_dd,
            lateral_offset,
            0.0,
            0.0,
            time
        )
        
        fp.d = [lat_qp.calc_point(t) for t in fp.t]
        fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
        fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
        fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
        
        return fp
    
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
        Jp = sum(np.power(fp.d_ddd, 2))  # Lateral jerk
        Jd = (fp.d[-1]) ** 2  # Final lateral offset
        
        # Longitudinal costs
        Js = sum(np.power(fp.s_ddd, 2))  # Longitudinal jerk
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
        obstacles: np.ndarray
    ) -> dict:
        """Check path validity and categorize paths.
        
        Args:
            fp_list: List of Frenet paths
            obstacles: Obstacle positions [n_obstacles, 2]
            
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
        
        for fp in fp_list:
            if len(fp.x) == 0:
                continue
            
            # Check speed limit
            if any([v > self.max_speed for v in fp.v]):
                path_dict['max_speed_error'].append(fp)
            # Check acceleration limit
            elif any([abs(a) > self.max_accel for a in fp.a]):
                path_dict['max_accel_error'].append(fp)
            # Check curvature limit
            elif any([abs(c) > self.max_curvature for c in fp.c]):
                path_dict['max_curvature_error'].append(fp)
            # Check collision
            elif not self._check_collision(fp, obstacles):
                path_dict['collision_error'].append(fp)
            else:
                path_dict['ok'].append(fp)
        
        return path_dict
    
    def _check_collision(
        self,
        fp: FrenetPath,
        obstacles: np.ndarray
    ) -> bool:
        """Check if path collides with obstacles.
        
        Args:
            fp: Frenet path
            obstacles: Obstacle positions [n_obstacles, 2]
            
        Returns:
            True if no collision, False otherwise
        """
        if len(obstacles) == 0:
            return True
        
        inflated_radius = max(
            self.robot_radius + self.obstacle_radius + self.safety_buffer,
            1e-6
        )
        
        for i in range(len(fp.x)):
            for obs in obstacles:
                dist = np.hypot(fp.x[i] - obs[0], fp.y[i] - obs[1])
                if dist <= inflated_radius:
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
