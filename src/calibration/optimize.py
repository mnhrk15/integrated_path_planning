"""Two-parameter (sigma, v0) calibration: coarse grid then Nelder-Mead refine.

A coarse grid is run first for two reasons beyond robustness to a non-convex
loss: it gives the global picture and it exposes the identifiability *ridge*
(the ego repulsion magnitude ``v0 * exp(-clearance / sigma)`` confounds ``v0``
and ``1/sigma`` when the data spans a narrow clearance band) — the grid loss
surface is itself a thesis figure. Nelder-Mead then refines from the grid
minimum. scipy is already a dependency (``metrics.compare_distributions_ks``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass
class CalibrationResult:
    """Result of a (sigma, v0) calibration."""

    sigma: float
    v0: float
    loss: float
    grid_sigma: np.ndarray  # [S]
    grid_v0: np.ndarray  # [V]
    grid_loss: np.ndarray  # [S, V]
    grid_best: tuple  # (sigma, v0) of the grid minimum
    refined: bool  # True if Nelder-Mead improved on the grid minimum


def calibrate(
    objective: Callable[[float, float], float],
    grid_sigma: Sequence[float],
    grid_v0: Sequence[float],
    refine: bool = True,
    max_iter: int = 60,
) -> CalibrationResult:
    """Minimise ``objective(sigma, v0)`` over a grid, optionally refining with NM.

    ``objective`` must return a finite loss for feasible parameters and may return
    ``inf`` for degenerate ones; the grid search ignores non-finite cells. The
    Nelder-Mead step penalises ``sigma <= 0`` / ``v0 < 0`` so it stays in the
    feasible region. The returned ``loss`` is the best of the grid and refined
    minima (so refinement never makes the result worse). A Nelder-Mead result is
    accepted whenever it strictly improves the loss, regardless of ``res.success``:
    at ``max_iter`` the solver routinely reports ``success=False`` ("Maximum number
    of iterations exceeded") on a noisy ADE surface even though it found a better
    interior point, and gating on ``success`` would silently throw that improvement
    away and fall back to the on-grid minimum.
    """
    grid_sigma = np.asarray(grid_sigma, dtype=float)
    grid_v0 = np.asarray(grid_v0, dtype=float)
    grid_loss = np.empty((len(grid_sigma), len(grid_v0)))
    for i, s in enumerate(grid_sigma):
        for k, v in enumerate(grid_v0):
            grid_loss[i, k] = objective(float(s), float(v))

    finite = np.isfinite(grid_loss)
    if not np.any(finite):
        raise ValueError("objective returned non-finite loss on the entire grid")
    masked = np.where(finite, grid_loss, np.inf)
    si, vi = np.unravel_index(np.argmin(masked), masked.shape)
    best_sigma = float(grid_sigma[si])
    best_v0 = float(grid_v0[vi])
    best_loss = float(masked[si, vi])
    grid_best = (best_sigma, best_v0)

    refined = False
    if refine:
        def penalised(x: np.ndarray) -> float:
            s, v = float(x[0]), float(x[1])
            if s <= 1e-3 or v < 0:
                return 1e12
            loss = objective(s, v)
            return loss if np.isfinite(loss) else 1e12

        res = minimize(
            penalised,
            x0=np.array([best_sigma, best_v0]),
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-6},
        )
        # Accept any finite improvement, even when res.success is False (the
        # iteration cap is hit before formal convergence). res.fun < best_loss
        # guarantees res.x is feasible, since the penalty maps the infeasible
        # region to 1e12, far above any finite ADE.
        if np.isfinite(res.fun) and float(res.fun) < best_loss:
            best_sigma, best_v0 = float(res.x[0]), float(res.x[1])
            best_loss = float(res.fun)
            refined = True

    return CalibrationResult(
        sigma=best_sigma,
        v0=best_v0,
        loss=best_loss,
        grid_sigma=grid_sigma,
        grid_v0=grid_v0,
        grid_loss=grid_loss,
        grid_best=grid_best,
        refined=refined,
    )
