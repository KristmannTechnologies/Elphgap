"""Linearized isotropic Migdal-Eliashberg Tc solver (imaginary axis).

Formulation (Allen & Dynes 1975; Carbotte, RMP 62, 1027 (1990)): with fermionic
Matsubara frequencies w_n = pi*T*(2n+1) and the electron-phonon kernel

    lambda(j) = 2 * Int d(omega) omega * a2F(omega) / (omega^2 + nu_j^2),
    nu_j = 2*pi*T*j,

the linearized gap equations, folded onto n >= 0 using gap parity
Delta(-n-1) = Delta(n), become the eigenproblem

    rho * x_n = (pi*T / (w_n * Z_n)) * sum_m B_nm x_m,
    B_nm = lambda(|n-m|) + lambda(n+m+1) - 2*mu_star,
    Z_n  = 1 + (pi*T / w_n) * (lambda(0) + 2 * sum_{j=1..n} lambda(j)),

with x_m = Delta_m / w_m. Tc is the temperature where the largest eigenvalue
rho(T) crosses 1. Since diag(w_n Z_n) > 0, the kernel is symmetrized as
S = D^{-1/2} B D^{-1/2} and solved with a dense symmetric eigensolver.

Conventions: omega in meV, T in K at the public API (converted internally to
meV). Matsubara cutoff omega_c = cutoff_factor * max(omega); the matrix size
N = omega_c / (2*pi*T) is capped at n_max, which sets a material-dependent
floor on resolvable Tc (reported via the `censored` flag). mu_star is used at
this cutoff for both Eliashberg and Allen-Dynes (standard practice, exact
mu* cutoff-rescaling neglected).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .units import K_TO_MEV, MEV_TO_K


def lambda_table(omega: np.ndarray, a2f: np.ndarray, t_mev: float, jmax: int) -> np.ndarray:
    """lambda(j) for j = 0..jmax at temperature t (meV)."""
    nu = 2.0 * np.pi * t_mev * np.arange(jmax + 1)
    integrand = 2.0 * omega * a2f / (omega**2 + nu[:, None] ** 2)
    return np.trapezoid(integrand, omega, axis=1)


def _matrix_size(t_mev: float, omega_c: float, n_max: int) -> int:
    n = int(np.ceil(omega_c / (2.0 * np.pi * t_mev)))
    return max(4, min(n_max, n))


def max_eigenvalue(
    omega: np.ndarray,
    a2f: np.ndarray,
    t_mev: float,
    mu_star: float,
    omega_c: float,
    n_max: int,
) -> float:
    """Largest eigenvalue rho(T) of the linearized Eliashberg kernel."""
    n_mat = _matrix_size(t_mev, omega_c, n_max)
    lam = lambda_table(omega, a2f, t_mev, 2 * n_mat)
    n = np.arange(n_mat)
    wn = np.pi * t_mev * (2 * n + 1)
    z = 1.0 + (np.pi * t_mev / wn) * (lam[0] + 2.0 * np.concatenate(([0.0], np.cumsum(lam[1 : n_mat]))))
    b = lam[np.abs(n[:, None] - n[None, :])] + lam[n[:, None] + n[None, :] + 1] - 2.0 * mu_star
    d_inv_sqrt = 1.0 / np.sqrt(wn * z / (np.pi * t_mev))
    s = d_inv_sqrt[:, None] * b * d_inv_sqrt[None, :]
    return float(eigh(s, eigvals_only=True, subset_by_index=[n_mat - 1, n_mat - 1])[0])


@dataclass
class TcResult:
    tc_kelvin: float
    censored: bool  # True if Tc fell below the resolvable floor (n_max cap)
    rho_at_floor: float | None = None


def tc_eliashberg(
    omega: np.ndarray,
    a2f: np.ndarray,
    mu_star: float = 0.10,
    cutoff_factor: float = 10.0,
    n_max: int = 4096,
    t_max_kelvin: float = 2000.0,
    rtol: float = 1e-3,
) -> TcResult:
    """Tc from bisection on rho(T) = 1. rho decreases monotonically with T."""
    omega_c = cutoff_factor * float(omega[-1])
    # Resolvable floor: temperature at which the (capped) matrix still reaches omega_c.
    t_floor_mev = omega_c / (2.0 * np.pi * n_max)
    t_floor_k = max(t_floor_mev * MEV_TO_K, 1e-3)

    def rho(t_k: float) -> float:
        return max_eigenvalue(omega, a2f, t_k * K_TO_MEV, mu_star, omega_c, n_max)

    rho_floor = rho(t_floor_k)
    if rho_floor < 1.0:
        return TcResult(tc_kelvin=0.0, censored=True, rho_at_floor=rho_floor)

    t_lo = t_floor_k
    t_hi = 2.0 * t_floor_k
    while rho(t_hi) > 1.0:
        t_lo = t_hi
        t_hi *= 2.0
        if t_hi > t_max_kelvin:
            raise RuntimeError(f"Tc bracket exceeded {t_max_kelvin} K")

    while (t_hi - t_lo) / t_hi > rtol:
        t_mid = 0.5 * (t_lo + t_hi)
        if rho(t_mid) > 1.0:
            t_lo = t_mid
        else:
            t_hi = t_mid
    return TcResult(tc_kelvin=0.5 * (t_lo + t_hi), censored=False)
