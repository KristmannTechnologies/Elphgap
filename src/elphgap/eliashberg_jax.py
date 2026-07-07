"""Batched JAX backend for the linearized isotropic Eliashberg Tc solver.

Same physics as eliashberg.py, vectorized over materials for GPU execution:
all materials run bracket expansion + bisection in lockstep, each step being
one batched symmetric eigendecomposition of shape (B, N, N). Static matrix
size N (jit requirement): Matsubara frequencies beyond each material's cutoff
omega_c are masked out (rows/cols zeroed, diagonal pushed to -1e3 so masked
modes never carry the maximum eigenvalue). The static N sets a resolvable
Tc floor of omega_c / (2*pi*N) per material; materials whose rho at the floor
is < 1 are reported censored (tc = 0).

XLA-compile hygiene (a naive broadcast/gather formulation triggered
pathologically slow GPU fusion compiles): the lambda(j) integral is expressed
as an einsum over precomputed trapezoid weights, and grids should be padded
to ONE global length across all batches (`grid_pad_to`) so the whole run
compiles exactly once per batch shape.

Runs unchanged on CPU and CUDA. float64 enabled for parity with the
reference backend; switch ELPHGAP_JAX_X64=0 for a float32 speed run.
"""

from __future__ import annotations

import os
from functools import partial

if os.environ.get("ELPHGAP_JAX_X64", "1") == "1":
    import jax

    jax.config.update("jax_enable_x64", True)
else:
    import jax

import jax.numpy as jnp
import numpy as np

from .units import MEV_TO_K


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    w = np.zeros_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def pad_batch(materials, grid_pad_to: int | None = None):
    """Stack variable-length (omega, a2f) grids into padded arrays.

    Returns (omega, a2f, weights); weights are trapezoid weights on each
    material's TRUE grid, zero on the padded tail, so integrals are exact.
    """
    g = grid_pad_to or max(len(m.omega) for m in materials)
    omega = np.full((len(materials), g), 1.0)
    a2f = np.zeros((len(materials), g))
    w = np.zeros((len(materials), g))
    for i, m in enumerate(materials):
        k = len(m.omega)
        omega[i, :k] = m.omega
        a2f[i, :k] = m.a2f
        w[i, :k] = _trapezoid_weights(m.omega)
        # benign tail values (never weighted): keep omega positive to avoid 0-division
        if k < g:
            omega[i, k:] = m.omega[-1]
    return jnp.asarray(omega), jnp.asarray(a2f), jnp.asarray(w)


@partial(jax.jit, static_argnames=("n_mat",))
def _lambda_table(omega, a2f, w, t_mev, n_mat):
    """lambda(j), j = 0..2*n_mat, per material. einsum keeps the fusion simple."""
    j = jnp.arange(2 * n_mat + 1)
    nu2 = (2.0 * jnp.pi * t_mev[:, None] * j[None, :]) ** 2  # (B, 2N+1)
    kern = 2.0 * omega[:, None, :] * a2f[:, None, :] / (omega[:, None, :] ** 2 + nu2[:, :, None])
    return jnp.einsum("bjg,bg->bj", kern, w)


@partial(jax.jit, static_argnames=("n_mat",))
def _rho_from_lambda(lam, t_mev, mu_star, omega_c, n_mat):
    n = jnp.arange(n_mat)
    wn = jnp.pi * t_mev[:, None] * (2 * n + 1)[None, :]  # (B, N)
    z = 1.0 + (jnp.pi * t_mev[:, None] / wn) * (
        lam[:, :1]
        + 2.0 * jnp.concatenate([jnp.zeros_like(lam[:, :1]), jnp.cumsum(lam[:, 1:n_mat], axis=1)], axis=1)
    )
    b = lam[:, jnp.abs(n[:, None] - n[None, :])] + lam[:, n[:, None] + n[None, :] + 1] - 2.0 * mu_star
    d_inv_sqrt = 1.0 / jnp.sqrt(wn * z / (jnp.pi * t_mev[:, None]))
    s = d_inv_sqrt[:, :, None] * b * d_inv_sqrt[:, None, :]
    mask = (wn <= omega_c[:, None]).astype(s.dtype)
    s = s * mask[:, :, None] * mask[:, None, :]
    s = s - (1.0 - mask[:, None, :]) * jnp.eye(n_mat)[None] * 1e3  # bury masked modes
    return jnp.linalg.eigvalsh(s)[:, -1]


def rho_batched(omega, a2f, w, t_mev, mu_star, omega_c, n_mat):
    """Largest kernel eigenvalue per material (two small jit stages)."""
    lam = _lambda_table(omega, a2f, w, t_mev, n_mat)
    return _rho_from_lambda(lam, t_mev, mu_star, omega_c, n_mat)


def tc_batched(
    materials,
    mu_star: float = 0.10,
    cutoff_factor: float = 10.0,
    n_mat: int = 1024,
    n_expand: int = 14,
    n_bisect: int = 25,
    grid_pad_to: int | None = None,
):
    """Tc [K] for a batch of materials; returns (tc_kelvin, censored) numpy arrays."""
    omega, a2f, w = pad_batch(materials, grid_pad_to)
    omega_max = jnp.asarray([m.omega[-1] for m in materials])
    omega_c = cutoff_factor * omega_max
    mu = jnp.asarray(mu_star)

    def rho(t):
        return rho_batched(omega, a2f, w, t, mu, omega_c, n_mat)

    t_floor = omega_c / (2.0 * jnp.pi * n_mat)  # meV; full matrix reaches cutoff here
    censored = np.asarray(rho(t_floor) < 1.0)

    # Lockstep bracket expansion: while rho(t_hi) > 1, shift bracket up by 2x.
    t_lo, t_hi = t_floor, 2.0 * t_floor
    for _ in range(n_expand):
        above = rho(t_hi) > 1.0
        t_lo = jnp.where(above, t_hi, t_lo)
        t_hi = jnp.where(above, 2.0 * t_hi, t_hi)

    for _ in range(n_bisect):
        t_mid = 0.5 * (t_lo + t_hi)
        above = rho(t_mid) > 1.0
        t_lo = jnp.where(above, t_mid, t_lo)
        t_hi = jnp.where(above, t_hi, t_mid)

    tc_k = np.asarray(0.5 * (t_lo + t_hi)) * MEV_TO_K
    tc_k[censored] = 0.0
    return tc_k, censored
