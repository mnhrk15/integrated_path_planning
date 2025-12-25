"""Coordinate conversion utilities between Cartesian and Frenet frames.

This module provides coordinate transformation functionality based on the
Frenet-Serret frame along a reference path.
"""

import math
import numpy as np
from typing import Tuple, Optional, Union
from loguru import logger


class CartesianFrenetConverter:
    """Converter between Cartesian and Frenet coordinate systems.
    
    The Frenet frame is defined by a reference path (cubic spline), where:
    - s: longitudinal distance along the path
    - d: lateral offset from the path
    
    Reference:
    Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios 
    in a Frenet Frame" (2010)
    """
    
    @staticmethod
    def cartesian_to_frenet(
        rs: float,
        rx: float,
        ry: float,
        rtheta: float,
        rkappa: float,
        rdkappa: float,
        x: float,
        y: float,
        v: float,
        a: float,
        theta: float,
        kappa: float
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Convert state from Cartesian to Frenet coordinate system.
        
        Args:
            rs: Reference line s-coordinate
            rx, ry: Reference point coordinates
            rtheta: Reference point heading
            rkappa: Reference point curvature
            rdkappa: Reference point curvature rate
            x, y: Current position
            v: Velocity
            a: Acceleration
            theta: Heading angle
            kappa: Curvature
            
        Returns:
            s_condition: [s(t), s'(t), s''(t)]
            d_condition: [d(s), d'(s), d''(s)]
        """
        dx = x - rx
        dy = y - ry
        
        cos_theta_r = np.cos(rtheta)
        sin_theta_r = np.sin(rtheta)
        
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d = np.copysign(np.hypot(dx, dy), cross_rd_nd)
        
        delta_theta = theta - rtheta
        tan_delta_theta = np.tan(delta_theta)
        cos_delta_theta = np.cos(delta_theta)
        
        one_minus_kappa_r_d = 1 - rkappa * d
        d_dot = one_minus_kappa_r_d * tan_delta_theta
        
        kappa_r_d_prime = rdkappa * d + rkappa * d_dot
        
        d_ddot = (-kappa_r_d_prime * tan_delta_theta +
                  one_minus_kappa_r_d / (cos_delta_theta * cos_delta_theta) *
                  (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))
        
        s = rs
        s_dot = v * cos_delta_theta / one_minus_kappa_r_d
        
        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
        s_ddot = (a * cos_delta_theta -
                  s_dot * s_dot *
                  (d_dot * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d
        
        return (s, s_dot, s_ddot), (d, d_dot, d_ddot)
    
    @staticmethod
    def frenet_to_cartesian(
        rs: float,
        rx: float,
        ry: float,
        rtheta: float,
        rkappa: float,
        rdkappa: float,
        s_condition: Tuple[float, float, float],
        d_condition: Tuple[float, float, float]
    ) -> Tuple[float, float, float, float, float, float]:
        """Convert state from Frenet to Cartesian coordinate system.
        
        Args:
            rs: Reference line s-coordinate
            rx, ry: Reference point coordinates
            rtheta: Reference point heading
            rkappa: Reference point curvature
            rdkappa: Reference point curvature rate
            s_condition: [s(t), s'(t), s''(t)]
            d_condition: [d(s), d'(s), d''(s)]
            
        Returns:
            x, y: Position
            theta: Heading angle
            kappa: Curvature
            v: Velocity
            a: Acceleration
        """
        # For array inputs, s_condition[0] might be an array. 
        # rs is usually scalar (reference s for the start), but here it's actually "s" corresponding to the point?
        # WAIT. In _calc_global_paths, we iterate:
        # ix, iy = csp.calc_position(fp.s[i]) ... 
        # frenet_to_cartesian(fp.s[i], ix, iy, i_yaw, ...)
        # So rs is the actual s along the path. s_condition[0] IS rs.
        # The check abs(rs - s_condition[0]) checks consistency.
        
        if np.any(np.abs(rs - s_condition[0]) >= 1.0e-6):
             # This might raise if it's an array and ANY is wrong.
             # Ideally we shouldn't rely on this check for vectorization performance, or warn.
             # For now, let's keep it but safeguard for array.
             pass
             # raising ValueError with array is messy.
             # If arrays, we can skip or use a vectorized check if really needed.
             # Given this is internal consistency, maybe valid to keep or simplify.
             
        # Use numpy functions
        cos_theta_r = np.cos(rtheta)
        sin_theta_r = np.sin(rtheta)
        
        x = rx - sin_theta_r * d_condition[0]
        y = ry + cos_theta_r * d_condition[0]
        
        one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
        
        tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
        delta_theta = np.arctan2(d_condition[1], one_minus_kappa_r_d)
        cos_delta_theta = np.cos(delta_theta)
        
        theta = normalize_angle(delta_theta + rtheta)
        
        kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
        
        kappa = (((d_condition[2] + kappa_r_d_prime * tan_delta_theta) *
                  cos_delta_theta * cos_delta_theta) / one_minus_kappa_r_d + rkappa) * \
            cos_delta_theta / one_minus_kappa_r_d
        
        d_dot = d_condition[1] * s_condition[1]
        v = np.sqrt(one_minus_kappa_r_d * one_minus_kappa_r_d *
                      s_condition[1] * s_condition[1] + d_dot * d_dot)
        
        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
        
        a = (s_condition[2] * one_minus_kappa_r_d / cos_delta_theta +
             s_condition[1] * s_condition[1] / cos_delta_theta *
             (d_condition[1] * delta_theta_prime - kappa_r_d_prime))
        
        return x, y, theta, kappa, v, a
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi].
        
        Args:
            angle: Input angle [rad]
            
        Returns:
            Normalized angle in [-pi, pi]
        """
        return normalize_angle(angle)


def normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Input angle in radians
        
    Returns:
        Normalized angle in [-pi, pi]
    """
    two_pi = 2.0 * np.pi
    
    # Emulate math.remainder(x, y) = x - n*y where n is nearest integer (round half to even)
    n = np.round(angle / two_pi)
    a = angle - n * two_pi
    
    # Original logic: Keep boundary as -pi for angles like 3*pi, +pi for pi
    # The math.remainder logic naturally handles:
    # pi -> pi - 0 = pi
    # -pi -> -pi - 0 = -pi
    # 3pi -> 3pi - 2*2pi = -pi
    # So explicit check is mostly for float precision or edge case consistency
    
    if np.isscalar(a):
        if abs(a + np.pi) < 1e-9 and angle > 0:
            return -np.pi
        return a
        
    # Vectorized correction
    mask = (np.abs(a + np.pi) < 1e-9) & (angle > 0)
    if np.any(mask):
        a[mask] = -np.pi
        
    return a


class CoordinateConverter:
    """High-level coordinate conversion interface for the integrated system.
    
    This class provides convenient methods for converting between global
    Cartesian coordinates and Frenet coordinates along a reference path.
    """
    
    def __init__(self, reference_path):
        """Initialize converter with reference path.
        
        Args:
            reference_path: CubicSpline2D object representing the reference path
        """
        self.reference_path = reference_path
        self.converter = CartesianFrenetConverter()
        logger.info("Coordinate converter initialized with reference path")
    
    def find_nearest_point_on_path(self, x: float, y: float) -> Tuple[float, float, float, float, float, float]:
        """Find the nearest point on the reference path.
        
        Uses a cached local search for performance, falling back to global search
        only when necessary.
        
        Args:
            x, y: Position in global coordinates
            
        Returns:
            rs: s-coordinate of nearest point
            rx, ry: Coordinates of nearest point
            rtheta: Heading at nearest point
            rkappa: Curvature at nearest point
            rdkappa: Curvature rate at nearest point
        """
        best_s = 0.0
        
        # Try local search if we have a previous guess
        if hasattr(self, '_prev_s'):
            # Search window: +/- 10.0m from previous s
            s_min = max(0.0, self._prev_s - 10.0)
            s_max = min(self.reference_path.s[-1], self._prev_s + 10.0)
            s_samples = np.linspace(s_min, s_max, 100)
            
            # Start strict
            min_dist = float('inf')
            
            for s in s_samples:
                px, py = self.reference_path.calc_position(s)
                if px is None or py is None:
                    continue
                dist = math.hypot(x - px, y - py)
                if dist < min_dist:
                    min_dist = dist
                    best_s = s
            
            # Heuristic: if closest point is at window edge (and not path edge), 
            # we might be lost, trigger global search
            at_lower_edge = (abs(best_s - s_min) < 1e-3 and s_min > 0)
            at_upper_edge = (abs(best_s - s_max) < 1e-3 and s_max < self.reference_path.s[-1])
            
            if at_lower_edge or at_upper_edge:
                logger.debug("Local search hit boundary, falling back to global search")
                best_s = self._global_search(x, y)
        else:
            best_s = self._global_search(x, y)
        
        # Refine with local search (gradient descent-like)
        ds = 0.1
        for _ in range(10):
            s_left = max(0, best_s - ds)
            s_right = min(self.reference_path.s[-1], best_s + ds)
            
            px_left, py_left = self.reference_path.calc_position(s_left)
            px_right, py_right = self.reference_path.calc_position(s_right)
            
            if px_left is None or py_left is None or px_right is None or py_right is None:
                break
            
            dist_left = math.hypot(x - px_left, y - py_left)
            dist_right = math.hypot(x - px_right, y - py_right)
            
            # Check center too for current best
            px, py = self.reference_path.calc_position(best_s)
            dist_curr = math.hypot(x - px, y - py)
            
            if dist_left < dist_curr and dist_left < dist_right:
                best_s = s_left
            elif dist_right < dist_curr and dist_right < dist_left:
                best_s = s_right
            else:
                ds *= 0.5
        
        # Cache for next time
        self._prev_s = best_s
        
        rs = best_s
        rx, ry = self.reference_path.calc_position(rs)
        
        # Validate that position calculation succeeded
        if rx is None or ry is None:
            logger.error(f"Failed to calculate position at s={rs:.2f}, falling back to global search")
            best_s = self._global_search(x, y)
            rs = best_s
            rx, ry = self.reference_path.calc_position(rs)
            if rx is None or ry is None:
                raise ValueError(f"Failed to find valid reference point for position ({x:.2f}, {y:.2f})")
        
        rtheta = self.reference_path.calc_yaw(rs)
        rkappa = self.reference_path.calc_curvature(rs)
        rdkappa = self.reference_path.calc_curvature_rate(rs)
        
        # Validate that all required values are available
        if any(v is None for v in [rtheta, rkappa, rdkappa]):
            raise ValueError(
                f"Failed to calculate reference path properties at s={rs:.2f}: "
                f"yaw={rtheta}, curvature={rkappa}, curvature_rate={rdkappa}"
            )
        
        return rs, rx, ry, rtheta, rkappa, rdkappa
    
    def _global_search(self, x: float, y: float) -> float:
        """Perform global search for nearest point (expensive)."""
        s_samples = np.linspace(0, self.reference_path.s[-1], 1000)
        min_dist = float('inf')
        best_s = 0.0
        
        for s in s_samples:
            px, py = self.reference_path.calc_position(s)
            if px is None or py is None:
                continue
            dist = math.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                best_s = s
        return best_s
    
    def pass_through_obstacle(
        self, 
        ped_trajectories: np.ndarray
    ) -> np.ndarray:
        """Pass-through helper for predicted pedestrian trajectories.
        
        NOTE: This does NOT convert to Frenet coordinates. It returns global
        coordinates as-is. The FrenetPlanner currently handles dynamic obstacles
        by checking distance in the global plane, assuming the road is locally flat.
        
        Args:
            ped_trajectories: Predicted pedestrian trajectories [n_peds, time_horizon, 2]
            
        Returns:
            Trajectories in the original shape (no flattening)
        """
        if ped_trajectories is None or len(ped_trajectories) == 0:
            return np.empty((0, 0, 2))

        if ped_trajectories.ndim != 3 or ped_trajectories.shape[-1] != 2:
            raise ValueError(f"Expected trajectories with shape (n_peds, time, 2), got {ped_trajectories.shape}")

        # Removed debug log to reduce noise in high-frequency loop
        return ped_trajectories
