"""Cubic spline path planning.

Based on the implementation from PythonRobotics:
https://github.com/AtsushiSakai/PythonRobotics
"""

import math
import numpy as np
import bisect
from typing import Tuple, List, Optional


class CubicSpline1D:
    """1D Cubic Spline interpolation.
    
    Interpolates a 1D function using cubic splines with natural boundary conditions.
    
    Args:
        x: x coordinates for data points (must be sorted in ascending order)
        y: y coordinates for data points
    """
    
    def __init__(self, x: List[float], y: List[float]):
        x = np.array(x)
        y = np.array(y)
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")
        
        self.a = y
        self.b = None
        self.c = None
        self.d = None
        self.x = x
        self.y = y
        self.nx = len(x)
        
        # Calculate coefficient c
        A = self._calc_A(h)
        B = self._calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)
        
        # Calculate coefficients b and d
        self.d = (self.c[1:] - self.c[:-1]) / (3.0 * h)
        self.b = (self.a[1:] - self.a[:-1]) / h - h * (2.0 * self.c[:-1] + self.c[1:]) / 3.0
    
    def calc_position(self, x: float) -> np.ndarray:
        """Calculate y position for given x.
        
        Args:
            x: x position(s) to calculate y
            
        Returns:
            y position(s) for given x. Returns NaN for x outside range.
        """
        x = np.atleast_1d(x)
        
        # Initialize result with NaNs
        result = np.full_like(x, np.nan, dtype=float)
        
        # Identify valid indices
        valid_mask = (x >= self.x[0]) & (x <= self.x[-1])
        if not np.any(valid_mask):
            return result[0] if result.ndim == 0 else result # Return scalar if input was scalar
            
        x_valid = x[valid_mask]
        
        i = self._search_index(x_valid)
        dx = x_valid - self.x[i]
        
        y_valid = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
            
        result[valid_mask] = y_valid
        
        # Return scalar if input was scalar
        return result[0] if result.size == 1 and x.shape == (1,) else result
    
    def calc_first_derivative(self, x: float) -> np.ndarray:
        """Calculate first derivative at given x.
        
        Args:
            x: x position(s)
            
        Returns:
            First derivative(s). Returns NaN for x outside range.
        """
        x = np.atleast_1d(x)
        result = np.full_like(x, np.nan, dtype=float)
        
        valid_mask = (x >= self.x[0]) & (x <= self.x[-1])
        if not np.any(valid_mask):
            return result[0] if result.size == 1 and x.shape == (1,) else result
            
        x_valid = x[valid_mask]
        i = self._search_index(x_valid)
        dx = x_valid - self.x[i]
        
        dy_valid = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        result[valid_mask] = dy_valid
        
        return result[0] if result.size == 1 and x.shape == (1,) else result
    
    def calc_second_derivative(self, x: float) -> np.ndarray:
        """Calculate second derivative at given x.
        
        Args:
            x: x position(s)
            
        Returns:
            Second derivative(s). Returns NaN for x outside range.
        """
        x = np.atleast_1d(x)
        result = np.full_like(x, np.nan, dtype=float)
        
        valid_mask = (x >= self.x[0]) & (x <= self.x[-1])
        if not np.any(valid_mask):
            return result[0] if result.size == 1 and x.shape == (1,) else result
            
        x_valid = x[valid_mask]
        i = self._search_index(x_valid)
        dx = x_valid - self.x[i]
        
        ddy_valid = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        result[valid_mask] = ddy_valid
        
        return result[0] if result.size == 1 and x.shape == (1,) else result
    
    def calc_third_derivative(self, x: float) -> np.ndarray:
        """Calculate third derivative at given x.
        
        Args:
            x: x position(s)
            
        Returns:
            Third derivative(s). Returns NaN for x outside range.
        """
        x = np.atleast_1d(x)
        result = np.full_like(x, np.nan, dtype=float)
        
        valid_mask = (x >= self.x[0]) & (x <= self.x[-1])
        if not np.any(valid_mask):
            return result[0] if result.size == 1 and x.shape == (1,) else result
            
        x_valid = x[valid_mask]
        i = self._search_index(x_valid)
        
        # dddy is constant for each segment
        dddy_valid = 6.0 * self.d[i]
        result[valid_mask] = dddy_valid
        
        return result[0] if result.size == 1 and x.shape == (1,) else result
    
    def _search_index(self, x: np.ndarray) -> np.ndarray:
        """Search data segment index for given x (vectorized)."""
        # np.searchsorted finds indices where elements should be inserted to maintain order
        # We need the index *less than* x, so we subtract 1.
        # But for x == self.x[0], it returns 0 (insertion before first element), 
        # so index becomes -1 if we just subtract.
        # However, we only call this with x >= self.x[0], so it works out EXCEPT for x=x[0].
        # For x=x[0], searchsorted(side='right') would give index 1, -1 = 0. Correct.
        idx = np.searchsorted(self.x, x, side='right') - 1
        
        # Check boundary condition
        idx = np.clip(idx, 0, self.nx - 2)
        return idx
    
    def _calc_A(self, h: np.ndarray) -> np.ndarray:
        """Calculate matrix A for spline coefficient c."""
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]
        
        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A
    
    def _calc_B(self, h: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Calculate matrix B for spline coefficient c."""
        B = np.zeros(self.nx)
        B[1:-1] = 3.0 * (a[2:] - a[1:-1]) / h[1:] - 3.0 * (a[1:-1] - a[:-2]) / h[:-1]
        return B


class CubicSpline2D:
    """2D Cubic Spline path.
    
    Creates a smooth 2D path through a set of waypoints using cubic splines.
    The path is parameterized by arc length s.
    
    Args:
        x: x coordinates of waypoints
        y: y coordinates of waypoints
    """
    
    def __init__(self, x: List[float], y: List[float]):
        self.s = self._calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)
    
    def _calc_s(self, x: List[float], y: List[float]) -> List[float]:
        """Calculate cumulative arc length along the path."""
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s
    
    def calc_position(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate position at given arc length.
        
        Args:
            s: Arc length from start
            
        Returns:
            (x, y) position using arrays.
        """
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)
        return x, y
    
    def calc_curvature(self, s: float) -> Optional[float]:
        """Calculate curvature at given arc length.
        
        Args:
            s: Arc length from start
            
        Returns:
            Curvature, or None if s is outside the path range
        """
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        
        if np.any(np.isnan(dx)) or np.any(np.isnan(ddx)) or np.any(np.isnan(dy)) or np.any(np.isnan(ddy)):
            # Propagate NaNs (or handle if needed, but vector ops usually handle NaNs fine)
            pass
            
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))
        return k
    
    def calc_curvature_rate(self, s: float) -> Optional[float]:
        """Calculate rate of change of curvature at given arc length.
        
        Args:
            s: Arc length from start
            
        Returns:
            Curvature rate, or None if s is outside the path range
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        dddx = self.sx.calc_third_derivative(s)
        dddy = self.sy.calc_third_derivative(s)
        
        a = dx * ddy - dy * ddx
        b = dx * dddy - dy * dddx
        c = dx * ddx + dy * ddy
        d = dx * dx + dy * dy
        return (b * d - 3.0 * a * c) / (d * d * d)
    
    def calc_yaw(self, s: float) -> np.ndarray:
        """Calculate heading angle at given arc length.
        
        Args:
            s: Arc length from start
            
        Returns:
            Heading angle in radians. Returns NaN for s outside range.
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        
        yaw = np.arctan2(dy, dx)
        return yaw


def calc_spline_course(
    x: List[float], 
    y: List[float], 
    ds: float = 0.1
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Calculate a smooth path using cubic spline interpolation.
    
    Args:
        x: x coordinates of waypoints
        y: y coordinates of waypoints
        ds: Distance between interpolated points
        
    Returns:
        rx: Interpolated x coordinates
        ry: Interpolated y coordinates
        ryaw: Heading angles
        rk: Curvatures
        s: Arc length parameters
    """
    sp = CubicSpline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))
    
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))
    
    return rx, ry, ryaw, rk, s
