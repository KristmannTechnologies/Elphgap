"""Anisotropic (band-resolved) Migdal-Eliashberg solver (imaginary axis) — reference.

Band/pocket-resolved gap Δ(k,iω_n) and renormalization Z(k,iω_n), solved
self-consistently. The GPU/JAX port (eliashberg_aniso_jax.py) is a mechanical
lift of this numpy reference (same array structure, one extra band axis on top
of the isotropic solver's Matsubara axis), so correctness is pinned here first.

Anisotropic imaginary-axis equations (Margine & Giustino, PRB 87, 024505 (2013);
Allen & Mitrović 1982), with a Fermi-surface weight w_k (Σ_k w_k = 1):

    Z(k,n)   = 1 + (πT/ω_n) Σ_{k',n'} w_{k'} · [ω_{n'}/R(k',n')] · λ(k,k',n−n')
    Z(k,n)Δ(k,n) = πT Σ_{k',n'} w_{k'} · [Δ(k',n')/R(k',n')] · [λ(k,k',n−n') − μ*]
    R(k',n') = sqrt(ω_{n'}² + Δ(k',n')²)

    λ(k,k',n−n') = ∫dω  2ω α²F_{k,k'}(ω) / ((ω_n − ω_{n'})² + ω²)   (depends on n−n' only)

ω_n = πT(2n+1). μ* is applied on the same Matsubara cutoff as the isotropic
solver (eliashberg.py).

ISOTROPIC LIMIT (the hard self-test): if α²F_{k,k'}(ω) = α²F(ω) for all k,k',
then λ is k-independent, Δ(k,n)=Δ(n), Z(k,n)=Z(n), and Σ_{k'} w_{k'}=1 collapses
the k'-sum — recovering the isotropic ME equations whose linearization is
exactly the isotropic solver. Hence Tc(aniso, isotropic input) matches the
isotropic Tc up to the Tc-extraction method (see the tc_aniso docstring).

Frequencies/energies in meV, T in K at the public API. See test_aniso.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import K_TO_MEV, MEV_TO_K


def lambda_kernel(omega: np.ndarray, a2f_pairs: np.ndarray, t_mev: float, jmax: int) -> np.ndarray:
    """λ(k,k',m) for m = 0..jmax.

    a2f_pairs: (K, K, G) anisotropic α²F on the shared ω-grid, or (G,) isotropic.
    Returns (K,K,jmax+1) or (jmax+1,) matching the input rank.
    """
    nu = 2.0 * np.pi * t_mev * np.arange(jmax + 1)  # bosonic differences ν_m
    # kernel_m(ω) = 2ω/((ν_m)² + ω²); integrate against α²F over ω.
    g = 2.0 * omega / (nu[:, None] ** 2 + omega[None, :] ** 2)  # (jmax+1, G)
    if a2f_pairs.ndim == 1:
        return np.trapezoid(g * a2f_pairs[None, :], omega, axis=1)  # (jmax+1,)
    # (K,K,G) x (M,G) -> (K,K,M)
    return np.trapezoid(g[None, None, :, :] * a2f_pairs[:, :, None, :], omega, axis=3)


@dataclass
class AnisoState:
    delta: np.ndarray  # (K, N) meV
    z: np.ndarray  # (K, N)
    converged: bool
    iterations: int
    max_gap_mev: float


def solve_gap_at_T(
    omega: np.ndarray,
    a2f_pairs: np.ndarray,
    weights: np.ndarray,
    t_kelvin: float,
    mu_star: float = 0.10,
    cutoff_factor: float = 10.0,
    n_max: int = 512,
    delta0_mev: float = 1.0,
    mixing: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> AnisoState:
    """Self-consistent Δ(k,n), Z(k,n) at fixed T via damped fixed-point iteration.

    weights: (K,) Fermi-surface weights w_j (∝ partial DOS N_j), normalized to sum 1.
    a2f_pairs: (K,K,G) or (G,) [isotropic, broadcast over the K×K block].

    COUPLING CONVENTION (important — easy to get wrong for multi-band input):
    the kernels are summed as  Σ_j w_j · λ_solver(i,j) · (...)  and likewise
    Σ_j w_j · mu_star[i,j], i.e. a2f_pairs/mu_star carry the PER-PAIR coupling
    *density* and w_j supplies the target-band DOS. So the standard band-resolved
    couplings (which already include N_j, e.g. Golubov's MgB2 λ_ij with row-sum
    λ_σ = λ_σσ+λ_σπ) map as:
        λ_ij^std  = w_j · λ_solver(i,j)     ->  pass a2f_pairs[i,j] for λ_ij^std / w_j
        μ*_ij^std = w_j · mu_star[i,j]       ->  pass mu_star[i,j] = μ*_ij^std / w_j
    Feeding λ_ij^std directly makes the solver see w_j·λ_ij ≈ half the intended
    coupling and Tc collapses. (Single-band K=1 has w=1, so this is a no-op there;
    see benchmarks/mgb2_twoband.py and test_two_band_mgb2_literature.)
    """
    t_mev = t_kelvin * K_TO_MEV
    omega_c = cutoff_factor * float(omega[-1])
    n_mat = max(4, min(n_max, int(np.ceil(omega_c / (2.0 * np.pi * t_mev)))))
    k = len(weights)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    # mu_star: scalar or (K,K) matrix in THIS solver's convention, where the
    # standard band-resolved Coulomb is mu*_ij^std = w_j * mu_star[i,j].
    mu = np.broadcast_to(np.asarray(mu_star, dtype=np.float64), (k, k))

    # Bosonic kernel λ(k,k',m), m up to 2*n_mat (needs |n-n'| and n+n'+1 not used here).
    lam = lambda_kernel(omega, a2f_pairs, t_mev, 2 * n_mat)  # (K,K,M) or (M,)
    iso = lam.ndim == 1

    n = np.arange(n_mat)
    wn = np.pi * t_mev * (2 * n + 1)  # (N,) fermionic Matsubara, n>=0
    # Folding to n>=0 with gap parity Δ(-n-1)=Δ(n): contributions from n' and -n'-1.
    # |n - n'| and (n + n' + 1) index the bosonic kernel.
    abs_idx = np.abs(n[:, None] - n[None, :])  # (N,N)
    sum_idx = n[:, None] + n[None, :] + 1  # (N,N)

    def lam_pair(ki, kj):
        base = lam if iso else lam[ki, kj]
        return base[abs_idx], base[sum_idx]  # each (N,N)

    delta = np.full((k, n_mat), delta0_mev)
    z = np.ones((k, n_mat))

    for it in range(max_iter):
        r = np.sqrt(wn[None, :] ** 2 + delta**2)  # (K,N) = R(k',n')
        gz = wn[None, :] / r  # (K,N)
        gd = delta / r  # (K,N)

        new_z = np.ones((k, n_mat))
        new_zd = np.zeros((k, n_mat))
        for ki in range(k):
            acc_z = np.zeros(n_mat)
            acc_zd = np.zeros(n_mat)
            for kj in range(k):
                lam_abs, lam_sum = lam_pair(ki, kj)  # (N,N) over (n,n')
                # Z: ω_n'/R is ODD under n'<->-n'-1 -> lam_abs - lam_sum
                kz = lam_abs - lam_sum
                acc_z += w[kj] * (kz @ gz[kj])
                # Δ: Δ_n'/R is EVEN -> lam_abs + lam_sum; μ* (per band pair) on the fold
                kd = lam_abs + lam_sum - 2.0 * mu[ki, kj]
                acc_zd += w[kj] * (kd @ gd[kj])
            new_z[ki] = 1.0 + (np.pi * t_mev / wn) * acc_z
            new_zd[ki] = (np.pi * t_mev) * acc_zd
        new_delta = new_zd / new_z

        step = np.max(np.abs(new_delta - delta))
        delta = (1 - mixing) * delta + mixing * new_delta
        z = (1 - mixing) * z + mixing * new_z
        if step < tol:
            return AnisoState(delta, z, True, it + 1, float(np.max(np.abs(delta))))

    return AnisoState(delta, z, False, max_iter, float(np.max(np.abs(delta))))


def tc_aniso(
    omega: np.ndarray,
    a2f_pairs: np.ndarray,
    weights: np.ndarray,
    mu_star: float = 0.10,
    gap_threshold_mev: float = 1e-3,
    cutoff_factor: float = 10.0,
    n_max: int = 512,
    t_lo: float | None = None,
    t_hi: float | None = None,
    t_max_kelvin: float = 2000.0,
    rtol: float = 2e-3,
    **solve_kw,
) -> float:
    """Tc [K]: highest T with a nontrivial self-consistent gap, via bisection.

    METHOD (heuristic — read before quoting Tc): unlike the isotropic solver,
    which bisects on the leading eigenvalue of the LINEARIZED kernel, this
    bisects on whether the full nonlinear gap solution survives above
    `gap_threshold_mev`. Near Tc the fixed-point iteration slows down
    critically, so an un-converged transient can still sit above the
    threshold: at the default settings this biases Tc HIGH by a few percent
    relative to the linearized-kernel Tc (pinned to <6 % by the isotropic-
    limit test; tighten gap_threshold_mev / raise max_iter to trade accuracy
    against cost, but note that too-tight thresholds can fail to bracket).
    Gap values Δ(T) away from Tc are unaffected. A linearized eigenvalue
    bisection for the anisotropic kernel is planned.

    The lower bracket defaults to the resolvable Matsubara floor
    (omega_c / (2*pi*n_max), as in the isotropic solver), NOT a hardcoded
    value — otherwise sub-kelvin but resolvable Tc would be falsely reported
    as 0. Returns 0.0 only if the gap has collapsed already at the floor
    (truly normal / below the resolvable floor). A max-gap above
    gap_threshold counts as superconducting; the threshold rejects the
    trivial Δ=0 fixed point.
    """
    omega_c = cutoff_factor * float(omega[-1])
    t_floor_k = max(omega_c / (2.0 * np.pi * n_max) * MEV_TO_K, 1e-3)
    lo = t_floor_k if t_lo is None else t_lo

    def is_sc(t):
        st = solve_gap_at_T(omega, a2f_pairs, weights, t, mu_star=mu_star,
                            cutoff_factor=cutoff_factor, n_max=n_max, **solve_kw)
        # Gap magnitude is the physical SC criterion; the convergence flag is a
        # numerical detail (near Tc, critical slowing-down leaves a finite gap
        # un-converged but still clearly nonzero — that state IS superconducting).
        return st.max_gap_mev > gap_threshold_mev

    if not is_sc(lo):
        return 0.0  # below the resolvable floor -> treat as normal (censored)

    # Expand the upper bracket upward from the floor until the gap closes.
    hi = 2.0 * lo if t_hi is None else t_hi
    while is_sc(hi):
        lo = hi
        hi *= 2.0
        if hi > t_max_kelvin:
            raise RuntimeError(f"gap still open at {t_max_kelvin} K; check input")

    while (hi - lo) / hi > rtol:
        t_mid = 0.5 * (lo + hi)
        if is_sc(t_mid):
            lo = t_mid
        else:
            hi = t_mid
    return 0.5 * (lo + hi)
