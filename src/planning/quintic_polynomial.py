"""Quintic and Quartic polynomial trajectory generation.

These polynomials are used for smooth trajectory planning in the Frenet frame.
"""

import numpy as np
from typing import List


class QuinticPolynomial:
    """Quintic polynomial for trajectory generation.
    
    The polynomial has the form:
    x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    
    Args:
        xs: Start position
        vxs: Start velocity
        axs: Start acceleration
        xe: End position
        vxe: End velocity
        axe: End acceleration
        time: Time duration
    """
    
    def __init__(
        self,
        xs: float,
        vxs: float,
        axs: float,
        xe: float,
        vxe: float,
        axe: float,
        time: float
    ):
        # Initial conditions
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        
        # Solve for remaining coefficients using end conditions
        A = np.array([
            [time ** 3, time ** 4, time ** 5],
            [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
            [6 * time, 12 * time ** 2, 20 * time ** 3]
        ])
        b = np.array([
            xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
            vxe - self.a1 - 2 * self.a2 * time,
            axe - 2 * self.a2
        ])
        x = np.linalg.solve(A, b)
        
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]
    
    def calc_point(self, t: float) -> float:
        """Calculate position at time t.
        
        Args:
            t: Time
            
        Returns:
            Position
        """
        xt = (self.a0 + self.a1 * t + self.a2 * t ** 2 +
              self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5)
        return xt
    
    def calc_first_derivative(self, t: float) -> float:
        """Calculate velocity at time t.
        
        Args:
            t: Time
            
        Returns:
            Velocity (first derivative)
        """
        xt = (self.a1 + 2 * self.a2 * t +
              3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4)
        return xt
    
    def calc_second_derivative(self, t: float) -> float:
        """Calculate acceleration at time t.
        
        Args:
            t: Time
            
        Returns:
            Acceleration (second derivative)
        """
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3
        return xt
    
    def calc_third_derivative(self, t: float) -> float:
        """Calculate jerk at time t.
        
        Args:
            t: Time
            
        Returns:
            Jerk (third derivative)
        """
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        return xt


class QuarticPolynomial:
    """Quartic polynomial for trajectory generation.
    
    The polynomial has the form:
    x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
    
    This is typically used for longitudinal planning when the end acceleration
    is not specified (set to 0 by the optimizer).
    
    Args:
        xs: Start position
        vxs: Start velocity
        axs: Start acceleration
        vxe: End velocity
        axe: End acceleration
        time: Time duration
    """
    
    def __init__(
        self,
        xs: float,
        vxs: float,
        axs: float,
        vxe: float,
        axe: float,
        time: float
    ):
        # Initial conditions
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        
        # Solve for remaining coefficients
        A = np.array([
            [3 * time ** 2, 4 * time ** 3],
            [6 * time, 12 * time ** 2]
        ])
        b = np.array([
            vxe - self.a1 - 2 * self.a2 * time,
            axe - 2 * self.a2
        ])
        x = np.linalg.solve(A, b)
        
        self.a3 = x[0]
        self.a4 = x[1]
    
    def calc_point(self, t: float) -> float:
        """Calculate position at time t.
        
        Args:
            t: Time
            
        Returns:
            Position
        """
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4
        return xt
    
    def calc_first_derivative(self, t: float) -> float:
        """Calculate velocity at time t.
        
        Args:
            t: Time
            
        Returns:
            Velocity (first derivative)
        """
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3
        return xt
    
    def calc_second_derivative(self, t: float) -> float:
        """Calculate acceleration at time t.
        
        Args:
            t: Time
            
        Returns:
            Acceleration (second derivative)
        """
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2
        return xt
    
    def calc_third_derivative(self, t: float) -> float:
        """Calculate jerk at time t.
        
        Args:
            t: Time
            
        Returns:
            Jerk (third derivative)
        """
        xt = 6 * self.a3 + 24 * self.a4 * t
        return xt
