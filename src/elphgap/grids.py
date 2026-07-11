"""Shared omega-grid validation for the public solver entry points.

Every kernel in this package integrates 1/omega- or 1/omega^2-weighted
quantities, so omega = 0 produces 0/0 -> NaN that only surfaces much later
as NaN moments or an eigensolver failure. Validate once at the API boundary
and fail with an actionable message instead.
"""

from __future__ import annotations

import numpy as np


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights: sum(w * f) == np.trapezoid(f, x).

    float64 cast included — an integer grid would otherwise truncate the
    half-interval endpoint weights to integers.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.zeros_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def validate_grid(omega: np.ndarray, name: str = "omega") -> np.ndarray:
    """Check that omega is a 1-D, strictly positive, strictly increasing grid.

    Returns the grid as a float64 array (integer grids would otherwise
    silently truncate trapezoid weights). Raises ValueError with a
    remediation hint on the common failure modes.
    """
    omega = np.asarray(omega, dtype=np.float64)
    if omega.ndim != 1 or omega.size < 2:
        raise ValueError(f"{name} must be a 1-D grid with at least 2 points, got shape {omega.shape}")
    if not np.all(np.isfinite(omega)):
        raise ValueError(f"{name} must be finite (found inf/NaN entries)")
    if not np.all(omega > 0.0):
        raise ValueError(
            f"{name} must be strictly positive: the a2F/omega moments and the "
            "2*omega/(omega^2 + nu^2) kernels are singular at omega = 0. Drop "
            f"the omega <= 0 head first (e.g. keep = {name} > 0)."
        )
    if not np.all(np.diff(omega) > 0.0):
        raise ValueError(f"{name} must be strictly increasing (trapezoidal integration grid)")
    return omega
