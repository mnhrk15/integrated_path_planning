
import numpy as np
from typing import Tuple

class VectorizedPolynomialSolver:
    """Vectorized solver for Quartic and Quintic polynomials.
    
    Efficiently solves ax = b for multiple sets of boundary conditions
    and time durations simultaneously.
    """
    
    @staticmethod
    def solve_quartic_batch(
        s_start: float,
        s_d_start: float,
        s_dd_start: float,
        v_end: np.ndarray,
        a_end: np.ndarray,
        T: np.ndarray
    ) -> np.ndarray:
        """Vectorized solver for Quartic Polynomials (Longitudinal).
        
        x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
        
        Args:
            s_start: Start position (scalar)
            s_d_start: Start velocity (scalar)
            s_dd_start: Start acceleration (scalar)
            v_end: Optimized target velocities [N]
            a_end: Optimized target accelerations [N]
            T: Time durations [N]
            
        Returns:
            Coefficients array [N, 5] where each row is [a0, a1, a2, a3, a4]
        """
        N = len(T)
        
        # a0, a1, a2 are determined by start conditions
        a0 = np.full(N, s_start)
        a1 = np.full(N, s_d_start)
        a2 = np.full(N, s_dd_start / 2.0)
        
        # Build matrix A and vector b for remaining coefficients [a3, a4]
        # Equations based on v_end and a_end constraints at t=T
        
        # A matrix elements
        # 3*T^2, 4*T^3
        # 6*T,   12*T^2
        
        T2 = T * T
        T3 = T2 * T
        
        A = np.zeros((N, 2, 2))
        A[:, 0, 0] = 3.0 * T2
        A[:, 0, 1] = 4.0 * T3
        A[:, 1, 0] = 6.0 * T
        A[:, 1, 1] = 12.0 * T2
        
        # b vector elements
        # v_end - a1 - 2*a2*T
        # a_end - 2*a2
        
        b = np.zeros((N, 2))
        b[:, 0] = v_end - a1 - 2.0 * a2 * T
        b[:, 1] = a_end - 2.0 * a2
        
        # Solve Ax = b
        # Since A is 2x2, we can implement analytical inverse for speed
        # det = ad - bc
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        
        # Inverse A:
        # 1/det * [[d, -b], [-c, a]]
        
        inv_A = np.zeros_like(A)
        inv_A[:, 0, 0] = A[:, 1, 1] / det
        inv_A[:, 0, 1] = -A[:, 0, 1] / det
        inv_A[:, 1, 0] = -A[:, 1, 0] / det
        inv_A[:, 1, 1] = A[:, 0, 0] / det
        
        # x = inv_A @ b
        # [N, 2, 2] @ [N, 2, 1] -> [N, 2, 1]
        x = np.einsum('ijk,ik->ij', inv_A, b)
        
        a3 = x[:, 0]
        a4 = x[:, 1]
        
        # Return stacked coefficients [N, 5]
        return np.column_stack([a0, a1, a2, a3, a4])

    @staticmethod
    def solve_quintic_batch(
        d_start: float,
        d_d_start: float,
        d_dd_start: float,
        d_end: np.ndarray,
        d_d_end: np.ndarray,
        d_dd_end: np.ndarray,
        T: np.ndarray
    ) -> np.ndarray:
        """Vectorized solver for Quintic Polynomials (Lateral).
        
        x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        
        Args:
            d_start: Start position (scalar)
            d_d_start: Start velocity (scalar)
            d_dd_start: Start acceleration (scalar)
            d_end: Target end positions [N]
            d_d_end: Target end velocities [N]
            d_dd_end: Target end accelerations [N]
            T: Time durations [N]
            
        Returns:
            Coefficients array [N, 6] where each row is [a0, a1, a2, a3, a4, a5]
        """
        N = len(T)
        
        a0 = np.full(N, d_start)
        a1 = np.full(N, d_d_start)
        a2 = np.full(N, d_dd_start / 2.0)
        
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        
        # Matrix A for [a3, a4, a5]
        # Eq 1 (Pos): T^3,   T^4,    T^5
        # Eq 2 (Vel): 3T^2,  4T^3,   5T^4
        # Eq 3 (Acc): 6T,    12T^2,  20T^3
        
        A = np.zeros((N, 3, 3))
        A[:, 0, 0] = T3
        A[:, 0, 1] = T4
        A[:, 0, 2] = T5
        
        A[:, 1, 0] = 3.0 * T2
        A[:, 1, 1] = 4.0 * T3
        A[:, 1, 2] = 5.0 * T4
        
        A[:, 2, 0] = 6.0 * T
        A[:, 2, 1] = 12.0 * T2
        A[:, 2, 2] = 20.0 * T3
        
        # Vector b
        # d_end - a0 - a1*T - a2*T^2
        # d_d_end - a1 - 2*a2*T
        # d_dd_end - 2*a2
        
        b = np.zeros((N, 3))
        b[:, 0] = d_end - a0 - a1 * T - a2 * T2
        b[:, 1] = d_d_end - a1 - 2.0 * a2 * T
        b[:, 2] = d_dd_end - 2.0 * a2
        
        # Solve Ax = b
        # Using numpy's solve is robust for 3x3, but manual inverse might be slightly faster or simpler to vectorize blindly
        # np.linalg.solve handles (N, 3, 3) and (N, 3) automatically
        
        # Calculate determinant (size N)
        # A is (N, 3, 3)
        # det = a(ei − fh) − b(di − fg) + c(dh − eg)
        # Indices: 0,1,2 for rows/cols
        
        # Unpack A for readability
        # row 0
        a00, a01, a02 = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
        # row 1
        a10, a11, a12 = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
        # row 2
        a20, a21, a22 = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]
        
        # Minors
        m00 = a11 * a22 - a12 * a21
        m01 = a10 * a22 - a12 * a20
        m02 = a10 * a21 - a11 * a20
        
        det = a00 * m00 - a01 * m01 + a02 * m02
        
        # Inverse A is 1/det * Adjugate(A)
        # Adjugate is transpose of cofactor matrix
        
        inv_det = 1.0 / det
        
        inv_A = np.zeros_like(A)
        
        # Row 0 of inv_A (Col 0 of Cofactor)
        inv_A[:, 0, 0] = (a11 * a22 - a12 * a21) * inv_det
        inv_A[:, 0, 1] = (a02 * a21 - a01 * a22) * inv_det
        inv_A[:, 0, 2] = (a01 * a12 - a02 * a11) * inv_det
        
        # Row 1 of inv_A (Col 1 of Cofactor)
        inv_A[:, 1, 0] = (a12 * a20 - a10 * a22) * inv_det
        inv_A[:, 1, 1] = (a00 * a22 - a02 * a20) * inv_det
        inv_A[:, 1, 2] = (a02 * a10 - a00 * a12) * inv_det
        
        # Row 2 of inv_A (Col 2 of Cofactor)
        inv_A[:, 2, 0] = (a10 * a21 - a11 * a20) * inv_det
        inv_A[:, 2, 1] = (a01 * a20 - a00 * a21) * inv_det
        inv_A[:, 2, 2] = (a00 * a11 - a01 * a10) * inv_det
        
        # x = inv_A @ b
        x = np.einsum('ijk,ik->ij', inv_A, b)
        
        a3 = x[:, 0]
        a4 = x[:, 1]
        a5 = x[:, 2]
        
        return np.column_stack([a0, a1, a2, a3, a4, a5])

    @staticmethod
    def evaluate_polynomial_batch(
        coefficients: np.ndarray,
        t: np.ndarray,
        order: int = 0
    ) -> np.ndarray:
        """Evaluate polynomials at given time steps.
        
        Args:
            coefficients: Polynomial coefficients [N, C] where C is 5 (Quartic) or 6 (Quintic)
            t: Time points to evaluate [N] or [N, T_steps]
            order: Derivative order (0=pos, 1=vel, 2=acc, 3=jerk)
            
        Returns:
            Evaluated values at t
        """
        # Assume coefficients are [a0, a1, a2, a3, a4, (a5)]
        
        # If t is [N, T_steps], we need to broadcast coeffs to [N, 1, C]
        # and t to [N, T_steps]
        
        # Simplified implementation: calculate terms manually based on order
        
        # Expand dims for broadcasting if t has time steps
        if t.ndim == 2:
            coeffs = coefficients[:, np.newaxis, :]  # [N, 1, C]
            t_pow = t[:, :, np.newaxis] # [N, T_steps, 1]
        else:
            coeffs = coefficients # [N, C]
            t_pow = t[:, np.newaxis] # [N, 1] or however it broadcasts with coeffs
            
        # This generic evaluation might be complex to optimize perfectly generic.
        # Let's write specific evaluators for speed in the planner loop 
        # OR just use simple arithmetic broadcasting.
        if coefficients.ndim != 2:
            raise ValueError("coefficients must have shape [N, C]")

        if t.ndim not in (1, 2):
            raise ValueError("t must have shape [N] or [N, T_steps]")

        num_coeffs = coefficients.shape[1]
        if num_coeffs not in (5, 6):
            raise ValueError("coefficients must have 5 (quartic) or 6 (quintic) terms")

        if t.shape[0] != coefficients.shape[0]:
            raise ValueError("t and coefficients must have matching first dimension")

        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3")

        if t.ndim == 1:
            t_eval = t[:, np.newaxis]
            squeeze = True
        else:
            t_eval = t
            squeeze = False

        t1 = t_eval
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t3 * t1

        a0 = coefficients[:, 0][:, np.newaxis]
        a1 = coefficients[:, 1][:, np.newaxis]
        a2 = coefficients[:, 2][:, np.newaxis]
        a3 = coefficients[:, 3][:, np.newaxis]
        a4 = coefficients[:, 4][:, np.newaxis]
        a5 = coefficients[:, 5][:, np.newaxis] if num_coeffs == 6 else None

        if order == 0:
            if num_coeffs == 5:
                values = a0 + a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4
            else:
                t5 = t4 * t1
                values = a0 + a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
        elif order == 1:
            if num_coeffs == 5:
                values = a1 + 2.0 * a2 * t1 + 3.0 * a3 * t2 + 4.0 * a4 * t3
            else:
                values = (
                    a1 + 2.0 * a2 * t1 + 3.0 * a3 * t2 +
                    4.0 * a4 * t3 + 5.0 * a5 * t4
                )
        elif order == 2:
            if num_coeffs == 5:
                values = 2.0 * a2 + 6.0 * a3 * t1 + 12.0 * a4 * t2
            else:
                values = 2.0 * a2 + 6.0 * a3 * t1 + 12.0 * a4 * t2 + 20.0 * a5 * t3
        else:
            if num_coeffs == 5:
                values = 6.0 * a3 + 24.0 * a4 * t1
            else:
                values = 6.0 * a3 + 24.0 * a4 * t1 + 60.0 * a5 * t2

        if squeeze:
            return values[:, 0]

        return values
