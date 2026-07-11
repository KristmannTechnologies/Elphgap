"""Parity: JAX/GPU anisotropic backend vs. the numpy reference.

Same physics, same fixed point. The numpy ref stops on convergence; the JAX
version runs a fixed iteration count, so we give it enough iters to reach the
same damped fixed point, then require tight agreement on the gaps and Tc.
"""

import numpy as np
import pytest

from elphgap.eliashberg_aniso import solve_gap_at_T as solve_np
from elphgap.eliashberg_aniso import tc_aniso as tc_np
from elphgap.eliashberg_aniso_jax import solve_gap_at_T as solve_jx
from elphgap.eliashberg_aniso_jax import tc_aniso as tc_jx


def einstein(lam, w_e, sigma=0.05, n=2000):
    omega = np.linspace(max(0.05, w_e - 12 * sigma), w_e + 12 * sigma, n)
    gauss = np.exp(-0.5 * ((omega - w_e) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return omega, 0.5 * lam * w_e * gauss


def two_pocket():
    we = 20.0
    o, a_s = einstein(1.4, we)
    _, a_w = einstein(0.5, we)
    _, a_i = einstein(0.15, we)
    pairs = np.stack([np.stack([a_s, a_i]), np.stack([a_i, a_w])])  # (2,2,G)
    return o, pairs, np.array([0.5, 0.5])


def test_solve_parity_two_pocket():
    o, pairs, w = two_pocket()
    st = solve_np(o, pairs, w, 8.0, mu_star=0.10, n_max=256, max_iter=4000)
    dj, zj, gj = solve_jx(o, pairs, w, 8.0, mu_star=0.10, n_max=256, n_iter=4000)
    assert st.converged
    # gap per pocket at n=0 must agree
    assert np.abs(dj[0, 0]) == pytest.approx(np.abs(st.delta[0, 0]), rel=1e-3, abs=1e-3)
    assert np.abs(dj[1, 0]) == pytest.approx(np.abs(st.delta[1, 0]), rel=1e-3, abs=1e-3)
    # Z parity too: Z is a returned observable (mass renormalization), not
    # just an internal variable, so pin it as well.
    assert np.allclose(zj, st.z, rtol=1e-4, atol=1e-4)


def test_solve_parity_isotropic():
    o, a = einstein(1.0, 20.0)
    st = solve_np(o, a, np.array([1.0]), 6.0, mu_star=0.10, n_max=256, max_iter=4000)
    dj, _, gj = solve_jx(o, a, np.array([1.0]), 6.0, mu_star=0.10, n_max=256, n_iter=4000)
    assert gj == pytest.approx(st.max_gap_mev, rel=1e-3, abs=1e-3)


def test_isotropic_spectrum_band_resolved_mu_parity():
    """Edge case: 1-D (isotropic) spectrum + K=2 bands + (2,2) μ* matrix.
    JAX must keep the K bands (not collapse to 1) and match numpy."""
    from elphgap.eliashberg_aniso import solve_gap_at_T as solve_np
    o, a = einstein(1.2, 20.0)
    w = np.array([0.4, 0.6])
    mu = np.array([[0.10, 0.13], [0.09, 0.12]])
    sn = solve_np(o, a, w, 8.0, mu_star=mu, n_max=256, max_iter=4000)
    dj, _, gj = solve_jx(o, a, w, 8.0, mu_star=mu, n_max=256, n_iter=4000)
    assert dj.shape == sn.delta.shape and dj.shape[0] == 2  # K=2 bands preserved
    assert np.allclose(dj[:, 0], sn.delta[:, 0], rtol=2e-3, atol=2e-3)


def test_tc_parity_two_pocket():
    # Same iteration budget in both so the gap-collapse near Tc is resolved
    # consistently (critical slowing-down needs many iters); both use the
    # threshold-based SC criterion. Tc via gap-collapse is threshold-sensitive,
    # hence the few-percent tolerance.
    o, pairs, w = two_pocket()
    tcn = tc_np(o, pairs, w, mu_star=0.10, n_max=256, max_iter=8000)
    tcj = tc_jx(o, pairs, w, mu_star=0.10, n_max=256, n_iter=8000)
    assert tcj == pytest.approx(tcn, rel=0.04)


def test_jax_weak_coupling_floor_no_false_positive():
    """The JAX path has no convergence flag, so its floor decision uses the
    linearized eigenvalue: the lam=0.35 normal-state transient (numpy
    regression test) must classify as 0 K here too."""
    o, a = einstein(0.35, 20.0)
    assert tc_jx(o, a, np.array([1.0]), mu_star=0.10, n_max=512) == 0.0


def test_jax_unconverged_below_threshold_uses_eigenvalue_guard():
    """Mirror of the numpy guard test: with a tiny seed and a tiny fixed
    iteration budget the gap never grows past the threshold, so the bracket
    would collapse onto the floor (~1.5 K instead of ~12.7 K). Sub-threshold
    iterates must be routed through the linearized-eigenvalue criterion."""
    from elphgap import tc_eliashberg

    o, a = einstein(0.8, 20.0)
    tc_ref = tc_eliashberg(o, a, mu_star=0.10, n_max=256).tc_kelvin
    tc = tc_jx(o, a, np.array([1.0]), mu_star=0.10, n_max=256,
               delta0_mev=1e-4, n_iter=5)
    assert tc == pytest.approx(tc_ref, rel=0.02)


def test_jax_marginal_sc_floor_not_censored():
    """Converse guard direction: a marginally unstable (rho barely > 1),
    resolvable superconductor must NOT be censored to 0 K just because the
    fixed iteration budget cannot grow the mode — lam=0.36 has a genuine
    sub-kelvin Tc just above the floor."""
    o, a = einstein(0.36, 20.0)
    tcn = tc_np(o, a, np.array([1.0]), mu_star=0.10, n_max=512)
    tcj = tc_jx(o, a, np.array([1.0]), mu_star=0.10, n_max=512)
    assert tcn > 0.0 and tcj > 0.0
    assert tcj == pytest.approx(tcn, rel=0.08)
