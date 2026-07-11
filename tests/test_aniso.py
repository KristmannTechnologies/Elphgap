"""Correctness anchors for the anisotropic Migdal-Eliashberg solver.

The decisive self-tests: with k-independent (isotropic) input,
tc_aniso_linearized must reduce to the isotropic solver EXACTLY (same matrix,
same closed-form Z — pinned to rel. 1e-6), and the nonlinear solver's
gap-collapse Tc must match it to a few percent — because the isotropic solver
IS the linearization of these equations and Tc is exactly where the
nontrivial gap branches off. Further tests pin the un-converged-transient
guard (no false SC near the floor), the exact normal-state Z, and that
genuinely anisotropic input (two Fermi-surface pockets with different
coupling) yields two distinct gaps.
"""

import numpy as np
import pytest

from elphgap import K_TO_MEV, tc_eliashberg
from elphgap.eliashberg_aniso import (
    lambda_kernel,
    solve_gap_at_T,
    tc_aniso,
    tc_aniso_linearized,
)


def einstein(lam: float, w_e: float, sigma: float = 0.05, n: int = 2000):
    omega = np.linspace(max(0.05, w_e - 12 * sigma), w_e + 12 * sigma, n)
    gauss = np.exp(-0.5 * ((omega - w_e) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return omega, 0.5 * lam * w_e * gauss


@pytest.mark.parametrize("lam,w_e", [(0.8, 20.0), (1.2, 25.0)])
def test_isotropic_limit_matches_isotropic_solver(lam, w_e):
    """K=1 (one Fermi-surface point, w=1) is the exact isotropic reduction."""
    omega, a2f = einstein(lam, w_e)
    weights = np.array([1.0])
    a2f_pairs = a2f  # 1-D -> isotropic broadcast

    tc1 = tc_eliashberg(omega, a2f, mu_star=0.10).tc_kelvin
    tc2 = tc_aniso(omega, a2f_pairs, weights, mu_star=0.10,
                   t_lo=0.5 * tc1, t_hi=1.5 * tc1, n_max=256)
    # The gap-collapse Tc heuristic tracks the linearized Tc to a few percent
    # (biased high near Tc by critical slowing-down; see tc_aniso docstring).
    assert tc2 == pytest.approx(tc1, rel=0.06)


@pytest.mark.parametrize("lam,w_e", [(0.8, 20.0), (1.2, 25.0)])
def test_linearized_iso_limit_is_exact(lam, w_e):
    """tc_aniso_linearized with K=1 uses the SAME matrix, Z closed form, matrix
    sizing, floor and bisection as the isotropic solver -> exact agreement."""
    omega, a2f = einstein(lam, w_e)
    tc1 = tc_eliashberg(omega, a2f, mu_star=0.10, n_max=512).tc_kelvin
    res = tc_aniso_linearized(omega, a2f, np.array([1.0]), mu_star=0.10, n_max=512)
    assert not res.censored
    assert res.tc_kelvin == pytest.approx(tc1, rel=1e-6)


def test_weak_coupling_floor_no_false_positive():
    """Regression (the one that matters): lam=0.35, mu*=0.10 is NORMAL at the
    resolvable floor (isotropic rho_floor = 0.9925 < 1), but the un-guarded
    gap-collapse heuristic reported Tc ~0.76 K from a slowly decaying transient
    of the delta0_mev=1 seed (0 K with delta0_mev=1e-3 or max_iter=4000 — a
    seed/budget-dependent classification, not a few-percent bias)."""
    omega, a2f = einstein(0.35, 20.0)
    w = np.array([1.0])
    iso = tc_eliashberg(omega, a2f, mu_star=0.10, n_max=512)
    assert iso.censored and iso.rho_at_floor < 1.0  # precondition: truly normal
    assert tc_aniso(omega, a2f, w, mu_star=0.10, n_max=512) == 0.0
    lin = tc_aniso_linearized(omega, a2f, w, mu_star=0.10, n_max=512)
    assert lin.censored and lin.tc_kelvin == 0.0
    assert lin.rho_at_floor == pytest.approx(iso.rho_at_floor, rel=1e-9)


def test_normal_state_z_exact_and_convergence_covers_z():
    """Two guarantees at once: (1) converged=True means Z is converged too
    (delta0_mev=0 previously returned after ONE iteration with Z far off);
    (2) with the exact Matsubara tail, the aniso normal-state Z equals the
    isotropic solver's closed-form (untruncated) Z, not the matrix-truncated
    sum (which is visibly smaller, e.g. 1.991 vs 2.000 at these settings)."""
    omega, a2f = einstein(1.0, 20.0)
    st = solve_gap_at_T(omega, a2f, np.array([1.0]), 30.0, mu_star=0.10,
                        n_max=256, delta0_mev=0.0)
    assert st.converged
    assert st.iterations > 10  # Z must actually be iterated to its fixed point
    t_mev = 30.0 * K_TO_MEV
    n_mat = st.z.shape[1]
    lam = lambda_kernel(omega, a2f, t_mev, 2 * n_mat)
    wn = np.pi * t_mev * (2 * np.arange(n_mat) + 1)
    z_exact = 1.0 + (np.pi * t_mev / wn) * (
        lam[0] + 2.0 * np.concatenate(([0.0], np.cumsum(lam[1:n_mat])))
    )
    assert np.allclose(st.z[0], z_exact, rtol=1e-5)


def test_unconverged_below_threshold_uses_eigenvalue_guard():
    """Converse of the false-positive regression: a slowly growing unstable
    mode (tiny seed, tiny iteration budget) is un-converged BELOW the
    threshold. Transient magnitude proves nothing in either direction, so it
    must be routed through the linearized-eigenvalue guard instead of being
    classified normal (previously -> Tc = 0 for a clearly SC material)."""
    omega, a2f = einstein(0.8, 20.0)
    w = np.array([1.0])
    tc_ref = tc_eliashberg(omega, a2f, mu_star=0.10, n_max=256).tc_kelvin
    tc = tc_aniso(omega, a2f, w, mu_star=0.10, n_max=256,
                  delta0_mev=1e-4, max_iter=5)
    assert tc > 0.0
    # With every bracket point un-converged, is_sc reduces to the linearized
    # criterion, so Tc must land on the isotropic (linearized) value.
    assert tc == pytest.approx(tc_ref, rel=0.02)


def test_matrix_free_eigenvalue_matches_dense():
    """Above dense_max_dim the leading eigenvalue is computed matrix-free
    (FFT Toeplitz/Hankel folds + Lanczos/Arnoldi) instead of materializing
    the (K·N)² kernel — the K~20, n_max~1024 configurations would otherwise
    OOM on multi-GB dense temporaries. Both paths must agree to solver
    precision, on the symmetric AND the general (asymmetric-mu*) branch."""
    from elphgap import K_TO_MEV as k2m
    from elphgap.eliashberg_aniso import max_eigenvalue_aniso

    we = 20.0
    o, a_s = einstein(1.4, we)
    _, a_w = einstein(0.5, we)
    _, a_i = einstein(0.15, we)
    pairs = np.stack([np.stack([a_s, a_i]), np.stack([a_i, a_w])])
    w = np.array([0.4, 0.6])
    args = (o, pairs, w, 8.0 * k2m, 0.10, 10.0 * o[-1], 128)
    dense = max_eigenvalue_aniso(*args)
    mfree = max_eigenvalue_aniso(*args, dense_max_dim=1)  # force matrix-free
    assert mfree == pytest.approx(dense, rel=1e-10)
    mu_a = np.array([[0.10, 0.13], [0.09, 0.12]])  # general (Arnoldi) branch
    args_a = (o, pairs, w, 8.0 * k2m, mu_a, 10.0 * o[-1], 128)
    dense_a = max_eigenvalue_aniso(*args_a)
    mfree_a = max_eigenvalue_aniso(*args_a, dense_max_dim=1)
    assert mfree_a == pytest.approx(dense_a, rel=1e-10)


def test_matrix_free_zero_kernel_returns_zero():
    """Degenerate but valid input (a2F = 0, mu* = 0): rho must be 0 on the
    matrix-free path too, not an ARPACK 'starting vector is zero' breakdown;
    tc_aniso_linearized reports it censored."""
    from elphgap import K_TO_MEV as k2m
    from elphgap.eliashberg_aniso import max_eigenvalue_aniso

    o = np.linspace(1.0, 20.0, 200)
    a = np.zeros((2, 2, o.size))
    w = np.array([0.5, 0.5])
    rho = max_eigenvalue_aniso(o, a, w, 4.0 * k2m, 0.0, 10.0 * o[-1], 64,
                               dense_max_dim=1)  # force matrix-free
    assert rho == 0.0
    res = tc_aniso_linearized(o, a, w, mu_star=0.0, n_max=64)
    assert res.censored and res.tc_kelvin == 0.0


def test_linearized_asymmetric_mu_general_path():
    """An asymmetric mu* matrix takes the general (non-eigh) eigensolver path;
    with a real leading eigenvalue it must return a sane rho. (A complex
    leading pair raises ValueError instead of silently taking real parts.)"""
    from elphgap import K_TO_MEV as k2m
    from elphgap.eliashberg_aniso import max_eigenvalue_aniso

    omega, a2f = einstein(1.0, 20.0)
    mu = np.array([[0.10, 0.13], [0.09, 0.12]])  # asymmetric on purpose
    rho = max_eigenvalue_aniso(omega, a2f, np.array([0.4, 0.6]), 8.0 * k2m,
                               mu, 10.0 * omega[-1], 128)
    assert np.isfinite(rho) and rho > 0.5


def test_default_bracket_finds_subkelvin_tc():
    """Regression: a resolvable sub-kelvin Tc must NOT be reported as 0 under
    the DEFAULT bracket. lambda~0.36 gives an isotropic Tc of ~0.8 K."""
    omega, a2f = einstein(0.36, 20.0)
    w = np.array([1.0])
    tc1 = tc_eliashberg(omega, a2f, mu_star=0.10, n_max=512).tc_kelvin
    assert 0.0 < tc1 < 1.0  # precondition: genuinely sub-kelvin but resolvable
    tc2 = tc_aniso(omega, a2f, w, mu_star=0.10, n_max=512)  # no explicit t_lo
    assert tc2 == pytest.approx(tc1, rel=0.08)


def test_gap_opens_below_tc_closes_above():
    omega, a2f = einstein(1.0, 20.0)
    w = np.array([1.0])
    tc = tc_aniso(omega, a2f, w, mu_star=0.10, n_max=256)
    below = solve_gap_at_T(omega, a2f, w, 0.6 * tc, mu_star=0.10, n_max=256)
    above = solve_gap_at_T(omega, a2f, w, 1.3 * tc, mu_star=0.10, n_max=256)
    assert below.converged and below.max_gap_mev > 0.5
    assert above.max_gap_mev < 0.1  # gap collapsed above Tc


def test_mu_matrix_uniform_equals_scalar():
    """A uniform (K,K) mu* matrix must reproduce the scalar-mu* result exactly."""
    o, pairs, w = (lambda: None), None, None
    we = 20.0
    o, a_s = einstein(1.4, we)
    _, a_w = einstein(0.5, we)
    _, a_i = einstein(0.15, we)
    pairs = np.stack([np.stack([a_s, a_i]), np.stack([a_i, a_w])])
    w = np.array([0.5, 0.5])
    sca = solve_gap_at_T(o, pairs, w, 8.0, mu_star=0.12, n_max=256)
    mat = solve_gap_at_T(o, pairs, w, 8.0, mu_star=np.full((2, 2), 0.12), n_max=256)
    assert np.allclose(sca.delta, mat.delta, atol=1e-9)


def test_two_pocket_anisotropy_gives_two_gaps():
    """Two Fermi pockets, stronger/weaker intra-pocket coupling, weak inter-pocket:
    must produce two distinct gap magnitudes (MgB2-like qualitative structure)."""
    we = 20.0
    o_s, a_strong = einstein(1.4, we)
    _, a_weak = einstein(0.5, we)
    _, a_inter = einstein(0.15, we)
    # a2f_pairs[i,j] on shared grid o_s
    a2f_pairs = np.stack([
        np.stack([a_strong, a_inter]),
        np.stack([a_inter, a_weak]),
    ])  # (2,2,G)
    weights = np.array([0.5, 0.5])
    # Below the (higher) Tc the strong pocket should carry a clearly larger gap.
    st = solve_gap_at_T(o_s, a2f_pairs, weights, 8.0, mu_star=0.10, n_max=256)
    assert st.converged
    gap_strong = np.abs(st.delta[0, 0])
    gap_weak = np.abs(st.delta[1, 0])
    assert gap_strong > gap_weak * 1.5  # two well-separated gaps
    assert gap_weak > 0.05  # weak pocket still gapped (proximity via inter-coupling)


def test_two_band_mgb2_literature():
    """Quantitative anchor + convention guard: the canonical MgB2 two-band model
    (Golubov 2002) fed through the solver must give Δσ≈7 meV, a clearly smaller Δπ,
    and Tc in the ~40-60 K range (single-mode 2-band Eliashberg overestimates the
    real 39 K, as in the literature). This pins the per-pair coupling convention
    λ_ij^std = w_j·λ_solver(i,j): the bug of feeding λ_ij directly collapses Δσ to
    ~0.8 meV / Tc~6 K, far outside these bounds. See benchmarks/mgb2_twoband.py."""
    lam = np.array([[1.02, 0.21], [0.16, 0.45]])  # [[σσ,σπ],[πσ,ππ]]
    w = np.array([0.44, 0.56])                     # DOS fractions (σ, π)
    we, mu_std = 60.0, 0.13
    o_s, _ = einstein(lam[0, 0] / w[0], we)
    a = np.stack([np.stack([einstein(lam[i, j] / w[j], we)[1] for j in range(2)])
                  for i in range(2)])               # (2,2,G), per-pair density
    mu = np.array([[mu_std / w[0], mu_std / w[1]],
                   [mu_std / w[0], mu_std / w[1]]])  # μ*_ij^std/w_j
    st = solve_gap_at_T(o_s, a, w, 4.0, mu_star=mu, n_max=512)
    assert st.converged
    d_sigma, d_pi = np.abs(st.delta[0, 0]), np.abs(st.delta[1, 0])
    assert 5.0 < d_sigma < 11.0, f"Δσ={d_sigma:.2f} meV out of MgB2 range"
    assert d_sigma > 2.5 * d_pi, f"gaps not well separated: Δσ={d_sigma:.2f} Δπ={d_pi:.2f}"
    tc = tc_aniso(o_s, a, w, mu_star=mu, n_max=512)
    assert 38.0 < tc < 65.0, f"Tc={tc:.1f} K out of MgB2 2-band range"
    # Linearized (exact) Tc must land in the same range; the gap-collapse
    # heuristic is documented as biased slightly HIGH relative to it.
    tc_lin = tc_aniso_linearized(o_s, a, w, mu_star=mu, n_max=512).tc_kelvin
    assert 38.0 < tc_lin < 65.0, f"Tc_lin={tc_lin:.1f} K out of MgB2 2-band range"
    assert tc_lin <= tc * 1.02, f"heuristic Tc={tc:.1f} below linearized {tc_lin:.1f}"
    assert tc == pytest.approx(tc_lin, rel=0.06)
