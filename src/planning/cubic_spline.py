"""Cubic spline path planning.

Based on the implementation from PythonRobotics:
https://github.com/AtsushiSakai/PythonRobotics
"""

import math
import numpy as np
import bisect
from typing import Tuple, List, Optional, Union


class CubicSpline1D:
    """1D Cubic Spline interpolation.
    
    Interpolates a 1D function using cubic splines with natural boundary conditions.
    
    Args:
        x: x coordinates for data points (must be sorted in ascending order)
        y: y coordinates for data points
    """
    
    def __init__(self, x: List[float], y: List[float]):
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")
        
        self.a = list(y)
        self.b = []
        self.c = []
        self.d = []
        self.x = x
        self.y = y
        self.nx = len(x)
        
        # Calculate coefficient c
        A = self._calc_A(h)
        B = self._calc_B(h, self.a)
        self.c = np.linalg.solve(A, B).tolist()
        
        # Calculate coefficients b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) \
                - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)
    
    def calc_position(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        """Calculate y position for given x.
        
        Args:
            x: x position(s) to calculate y
            
        Returns:
            y position(s) for given x. 
            Returns None if x is scalar and out of range.
            Returns array with NaNs if x is array and out of range.
        """
        if np.isscalar(x):
            if x < self.x[0] or x > self.x[-1]:
                return None
            
            i = self._search_index(x)
            dx = x - self.x[i]
            position = self.a[i] + self.b[i] * dx + \
                self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
            return position
        
        # Array input
        x = np.asarray(x)
        mask = (x >= self.x[0]) & (x <= self.x[-1])
        
        # Compute results for valid points
        res = np.full_like(x, np.nan, dtype=float)
        
        if np.any(mask):
            x_valid = x[mask]
            i = self._search_index(x_valid)
            dx = x_valid - np.array(self.x)[i]
            
            a = np.array(self.a)[i]
            b = np.array(self.b)[i]
            c = np.array(self.c)[i]
            d = np.array(self.d)[i]
            
            res[mask] = a + b * dx + c * dx ** 2.0 + d * dx ** 3.0
            
        return res
    
    def calc_first_derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        """Calculate first derivative at given x.
        
        Args:
            x: x position(s) to calculate first derivative
            
        Returns:
            First derivative(s) for given x.
        """
        if np.isscalar(x):
            if x < self.x[0] or x > self.x[-1]:
                return None
            
            i = self._search_index(x)
            dx = x - self.x[i]
            dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
            return dy

        # Array input
        x = np.asarray(x)
        mask = (x >= self.x[0]) & (x <= self.x[-1])
        
        res = np.full_like(x, np.nan, dtype=float)
        
        if np.any(mask):
            x_valid = x[mask]
            i = self._search_index(x_valid)
            dx = x_valid - np.array(self.x)[i]
            
            b = np.array(self.b)[i]
            c = np.array(self.c)[i]
            d = np.array(self.d)[i]
            
            res[mask] = b + 2.0 * c * dx + 3.0 * d * dx ** 2.0
            
        return res
    
    def calc_second_derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        """Calculate second derivative at given x.
        
        Args:
            x: x position(s) to calculate second derivative
            
        Returns:
            Second derivative(s) for given x.
        """
        if np.isscalar(x):
            if x < self.x[0] or x > self.x[-1]:
                return None
            
            i = self._search_index(x)
            dx = x - self.x[i]
            ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
            return ddy

        # Array input
        x = np.asarray(x)
        mask = (x >= self.x[0]) & (x <= self.x[-1])
        
        res = np.full_like(x, np.nan, dtype=float)
        
        if np.any(mask):
            x_valid = x[mask]
            i = self._search_index(x_valid)
            dx = x_valid - np.array(self.x)[i]
            
            c = np.array(self.c)[i]
            d = np.array(self.d)[i]
            
            res[mask] = 2.0 * c + 6.0 * d * dx
            
        return res
    
    def calc_third_derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        """Calculate third derivative at given x.
        
        Args:
            x: x position(s) to calculate third derivative
            
        Returns:
            Third derivative(s) for given x.
        """
        if np.isscalar(x):
            if x < self.x[0] or x > self.x[-1]:
                return None
            
            i = self._search_index(x)
            dddy = 6.0 * self.d[i]
            return dddy

        # Array input
        x = np.asarray(x)
        mask = (x >= self.x[0]) & (x <= self.x[-1])
        
        res = np.full_like(x, np.nan, dtype=float)
        
        if np.any(mask):
            x_valid = x[mask]
            i = self._search_index(x_valid)
            d = np.array(self.d)[i]
            res[mask] = 6.0 * d
            
        return res
    
    def _search_index(self, x: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Search data segment index for given x."""
        if np.isscalar(x):
            idx = bisect.bisect(self.x, x) - 1
            return min(max(idx, 0), self.nx - 2)
        
        # Array input
        idx = np.searchsorted(self.x, x, side='right') - 1
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
    
    def _calc_B(self, h: np.ndarray, a: List[float]) -> np.ndarray:
        """Calculate matrix B for spline coefficient c."""
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] \
                - 3.0 * (a[i + 1] - a[i]) / h[i]
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
    
    def calc_position(self, s: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray, None]]:
        """Calculate position at given arc length.
        
        Args:
            s: Arc length from start
            
        Returns:
            (x, y) position
        """
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)
        return x, y
    
    def calc_curvature(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        """Calculate curvature at given arc length.
        
        Args:
            s: Arc length from start
            
        Returns:
            Curvature
        """
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        
        if np.isscalar(s):
            if dx is None or ddx is None or dy is None or ddy is None:
                return None
            k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))
            return k
        
        # Array input
        # Note: dx, dy, etc. will contain NaNs where s was out of bounds
        denom = (dx ** 2 + dy ** 2) ** 1.5
        # Avoid division by zero warnings if any
        with np.errstate(invalid='ignore', divide='ignore'):
             k = (ddy * dx - ddx * dy) / denom
        return k
    
    def calc_curvature_rate(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        """Calculate rate of change of curvature at given arc length.
        
        Args:
            s: Arc length from start
            
        Returns:
            Curvature rate
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        dddx = self.sx.calc_third_derivative(s)
        dddy = self.sy.calc_third_derivative(s)
        
        if np.isscalar(s):
            if any(v is None for v in [dx, dy, ddx, ddy, dddx, dddy]):
                return None
            
            a = dx * ddy - dy * ddx
            b = dx * dddy - dy * dddx
            c = dx * ddx + dy * ddy
            d = dx * dx + dy * dy
            return (b * d - 3.0 * a * c) / (d * d * d)
        
        # Array input
        with np.errstate(invalid='ignore', divide='ignore'):
            a = dx * ddy - dy * ddx
            b = dx * dddy - dy * dddx
            c = dx * ddx + dy * ddy
            d = dx * dx + dy * dy
            return (b * d - 3.0 * a * c) / (d * d * d)
    
    def calc_yaw(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        """Calculate heading angle at given arc length.
        
        Args:
            s: Arc length from start
            
        Returns:
            Heading angle in radians
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        
        if np.isscalar(s):
            if dx is None or dy is None:
                return None
            yaw = math.atan2(dy, dx)
            return yaw
            
        # Array input
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
