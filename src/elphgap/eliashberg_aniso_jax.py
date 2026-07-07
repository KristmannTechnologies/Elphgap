"""GPU/JAX backend for the anisotropic Migdal-Eliashberg solver.

Mechanical lift of eliashberg_aniso.py (numpy reference): same physics, the
double k,k' sum expressed as einsum/tensordot so it runs as batched dense
linear algebra on GPU. The self-consistent Δ(k,n), Z(k,n) iteration is a
jit'd damped fixed-point with a fixed iteration count (lax-friendly, no
data-dependent Python control flow); the Tc bisection stays in Python and
calls the jit'd solver per temperature (as in the batched isotropic backend).

The anisotropic kernel K(k,k',n,n') applied to (Δ/R, ω/R) is a dense
(K·N)×(K·N) contraction — here one einsum per iteration, i.e. exactly the
kind of batched dense linear algebra GPUs are good at. Validated against the
numpy reference in test_aniso_jax.py.

float64 by default (ELPHGAP_JAX_X64=1) for parity; float32 for speed runs.
"""

from __future__ import annotations

import os

if os.environ.get("ELPHGAP_JAX_X64", "1") == "1":
    import jax

    jax.config.update("jax_enable_x64", True)
else:
    import jax

from functools import partial

import jax.numpy as jnp
import numpy as np

from .units import K_TO_MEV, MEV_TO_K


def _build_lambda_pairs(omega, a2f_pairs, t_mev, n_mat):
    """λ kernels folded onto (K,K,N,N): returns (lam_abs, lam_sum) as jnp arrays.

    Built once per temperature on host (numpy) — cheap relative to the iteration —
    then handed to the jit'd step. a2f_pairs: (K,K,G) array.
    """
    jmax = 2 * n_mat
    nu = 2.0 * np.pi * t_mev * np.arange(jmax + 1)
    g = 2.0 * omega / (nu[:, None] ** 2 + omega[None, :] ** 2)  # (M,G)
    lam = np.trapezoid(g[None, None] * a2f_pairs[:, :, None, :], omega, axis=3)  # (K,K,M)
    n = np.arange(n_mat)
    abs_idx = np.abs(n[:, None] - n[None, :])
    sum_idx = n[:, None] + n[None, :] + 1
    lam_abs = lam[:, :, abs_idx]  # (K,K,N,N)
    lam_sum = lam[:, :, sum_idx]
    return jnp.asarray(lam_abs), jnp.asarray(lam_sum)


@partial(jax.jit, static_argnames=("n_iter",))
def _iterate(lam_abs, lam_sum, w, wn, t_mev, mu_star, delta0, mixing, n_iter):
    """Damped fixed-point for Δ(k,n), Z(k,n). Fixed n_iter for jit-friendliness.

    lam_abs/lam_sum: (K,K,N,N); w: (K,); wn: (N,). Kernels:
      Z:  (lam_abs - lam_sum) contracted with ω_n'/R   (odd fold)
      ZΔ: (lam_abs + lam_sum - 2μ*) contracted with Δ_n'/R  (even fold)
    """
    k = w.shape[0]
    n_mat = wn.shape[0]
    kz = lam_abs - lam_sum  # (K,K,N,N)
    # mu_star: scalar or (K,K); broadcast onto the (K,K,1,1) pair axis
    mu = jnp.asarray(mu_star)
    mu = jnp.broadcast_to(mu, (k, k))[:, :, None, None]
    kd = lam_abs + lam_sum - 2.0 * mu

    def body(_, carry):
        delta, z = carry
        r = jnp.sqrt(wn[None, :] ** 2 + delta**2)  # (K,N)
        gz = wn[None, :] / r
        gd = delta / r
        # acc_z[ki,n] = Σ_kj w[kj] Σ_n' kz[ki,kj,n,n'] gz[kj,n']  (w indexed by j=kj)
        acc_z = jnp.einsum("j,ijnm,jm->in", w, kz, gz)
        acc_zd = jnp.einsum("j,ijnm,jm->in", w, kd, gd)
        new_z = 1.0 + (jnp.pi * t_mev / wn)[None, :] * acc_z
        new_zd = (jnp.pi * t_mev) * acc_zd
        new_delta = new_zd / new_z
        return ((1 - mixing) * delta + mixing * new_delta,
                (1 - mixing) * z + mixing * new_z)

    delta = jnp.full((k, n_mat), delta0)
    z = jnp.ones((k, n_mat))
    delta, z = jax.lax.fori_loop(0, n_iter, body, (delta, z))
    return delta, z


def solve_gap_at_T(
    omega, a2f_pairs, weights, t_kelvin, mu_star=0.10,
    cutoff_factor=10.0, n_max=512, delta0_mev=1.0, mixing=0.2, n_iter=2000,
):
    """Δ(k,n), Z(k,n) at fixed T. Returns (delta, z, max_gap_mev) with numpy delta."""
    t_mev = t_kelvin * K_TO_MEV
    omega = np.asarray(omega, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    a2f_pairs = np.asarray(a2f_pairs, dtype=np.float64)
    if a2f_pairs.ndim == 1:
        # Isotropic spectrum: broadcast over all K×K band pairs (keep K from
        # weights), matching the numpy backend — so a band-resolved μ* matrix
        # still applies. K=1 reduces to the plain isotropic case.
        kk = len(weights)
        a2f_pairs = np.broadcast_to(a2f_pairs, (kk, kk, a2f_pairs.shape[0]))
    omega_c = cutoff_factor * float(omega[-1])
    n_mat = max(4, min(n_max, int(np.ceil(omega_c / (2.0 * np.pi * t_mev)))))
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    wn = np.pi * t_mev * (2 * np.arange(n_mat) + 1)

    lam_abs, lam_sum = _build_lambda_pairs(omega, np.asarray(a2f_pairs, dtype=np.float64), t_mev, n_mat)
    delta, z = _iterate(lam_abs, lam_sum, jnp.asarray(w), jnp.asarray(wn),
                        t_mev, mu_star, delta0_mev, mixing, n_iter)
    delta = np.asarray(delta)
    return delta, np.asarray(z), float(np.max(np.abs(delta)))


def tc_aniso(
    omega, a2f_pairs, weights, mu_star=0.10, gap_threshold_mev=1e-3,
    cutoff_factor=10.0, n_max=512, t_max_kelvin=2000.0, rtol=2e-3, **solve_kw,
):
    """Tc [K] via bisection; lower bracket from the Matsubara floor (see numpy ref)."""
    omega = np.asarray(omega, dtype=np.float64)
    omega_c = cutoff_factor * float(omega[-1])
    t_floor = max(omega_c / (2.0 * np.pi * n_max) * MEV_TO_K, 1e-3)

    def is_sc(t):
        _, _, g = solve_gap_at_T(omega, a2f_pairs, weights, t, mu_star=mu_star,
                                 cutoff_factor=cutoff_factor, n_max=n_max, **solve_kw)
        return g > gap_threshold_mev

    if not is_sc(t_floor):
        return 0.0
    lo, hi = t_floor, 2.0 * t_floor
    while is_sc(hi):
        lo, hi = hi, 2.0 * hi
        if hi > t_max_kelvin:
            raise RuntimeError(f"gap still open at {t_max_kelvin} K")
    while (hi - lo) / hi > rtol:
        mid = 0.5 * (lo + hi)
        if is_sc(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
