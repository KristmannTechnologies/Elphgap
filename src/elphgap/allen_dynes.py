"""Moments of alpha^2F and the Allen-Dynes Tc formula.

Conventions (Allen & Dynes, PRB 12, 905 (1975)):
    lambda   = 2 * Int d(omega) a2F(omega) / omega
    omega_log = exp( (2/lambda) * Int d(omega) a2F(omega) ln(omega) / omega )
    omega_2  = sqrt( (2/lambda) * Int d(omega) a2F(omega) * omega )
    Tc = (f1*f2) * (omega_log / 1.20) * exp( -1.04(1+lambda) / (lambda - mu*(1+0.62 lambda)) )
with the strong-coupling and shape corrections
    f1 = [1 + (lambda/L1)^(3/2)]^(1/3),          L1 = 2.46 (1 + 3.8 mu*)
    f2 = 1 + (r - 1) lambda^2 / (lambda^2 + L2^2), L2 = 1.82 (1 + 6.3 mu*) r,  r = omega_2/omega_log
Setting f1 = f2 = 1 recovers the McMillan form.

Frequencies in meV, Tc returned in K.
"""

from __future__ import annotations

import numpy as np

from .units import MEV_TO_K


def moments(omega: np.ndarray, a2f: np.ndarray) -> tuple[float, float, float]:
    """Return (lambda, omega_log [meV], omega_2 [meV]) by trapezoidal integration."""
    lam = 2.0 * np.trapezoid(a2f / omega, omega)
    wlog = np.exp(2.0 / lam * np.trapezoid(a2f * np.log(omega) / omega, omega))
    w2 = np.sqrt(2.0 / lam * np.trapezoid(a2f * omega, omega))
    return float(lam), float(wlog), float(w2)


def tc_allen_dynes(
    lam: float, wlog_mev: float, w2_mev: float | None = None, mu_star: float = 0.10
) -> float:
    """Allen-Dynes Tc in K. With w2_mev=None the shape correction f2 is skipped
    (f1 still applies); for the plain McMillan form use tc_mcmillan."""
    denom = lam - mu_star * (1.0 + 0.62 * lam)
    if denom <= 0:
        return 0.0
    f1 = (1.0 + (lam / (2.46 * (1.0 + 3.8 * mu_star))) ** 1.5) ** (1.0 / 3.0)
    f2 = 1.0
    if w2_mev is not None:
        r = w2_mev / wlog_mev
        l2 = 1.82 * (1.0 + 6.3 * mu_star) * r
        f2 = 1.0 + (r - 1.0) * lam**2 / (lam**2 + l2**2)
    tc_mev = f1 * f2 * (wlog_mev / 1.20) * np.exp(-1.04 * (1.0 + lam) / denom)
    return float(tc_mev * MEV_TO_K)


def tc_mcmillan(lam: float, wlog_mev: float, mu_star: float = 0.10) -> float:
    """McMillan Tc in K: the Allen-Dynes exponential with f1 = f2 = 1."""
    denom = lam - mu_star * (1.0 + 0.62 * lam)
    if denom <= 0:
        return 0.0
    tc_mev = (wlog_mev / 1.20) * np.exp(-1.04 * (1.0 + lam) / denom)
    return float(tc_mev * MEV_TO_K)
