"""Regression guard for CubicSpline2D.calc_curvature_rate (review R5).

dk/ds for a planar curve is (b*d - 3*a*c) / d**2.5 (NOT / d**3), where
a = x'y'' - y'x'',  b = x'y''' - y'x''',  c = x'x'' + y'y'',  d = x'^2 + y'^2.
The buggy denominator d**3 differs from the correct d**2.5 by a factor of
sqrt(d) (the speed), so it only matters on a CURVED path. The existing planner
tests exercise straight reference lines only (a=b=c=0 => dk/ds=0 for either
formula), which is exactly why the bug went undetected. This test compares
calc_curvature_rate against a central finite difference of calc_curvature on a
genuinely curved spline.
"""
import numpy as np

from src.planning.cubic_spline import CubicSpline2D


def _curved_spline() -> CubicSpline2D:
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [0.0, 0.6, 1.5, 1.2, 0.3, 1.0, 2.0]
    return CubicSpline2D(xs, ys)


def test_curvature_rate_matches_finite_difference_on_curved_spline():
    sp = _curved_spline()
    eps = 1e-4
    for s in np.linspace(0.5, sp.s[-1] - 0.5, 8):
        fd = (sp.calc_curvature(s + eps) - sp.calc_curvature(s - eps)) / (2 * eps)
        got = sp.calc_curvature_rate(s)
        np.testing.assert_allclose(
            got, fd, rtol=0, atol=1e-4,
            err_msg=f"dk/ds mismatch at s={s:.3f}: got {got}, finite-diff {fd}")


def test_curvature_rate_zero_on_straight_line():
    # Straight line: curvature is identically 0, so its rate of change is 0.
    sp = CubicSpline2D([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    for s in np.linspace(0.2, sp.s[-1] - 0.2, 5):
        np.testing.assert_allclose(sp.calc_curvature_rate(s), 0.0, atol=1e-9)
