"""Correctness anchors for the anisotropic Migdal-Eliashberg solver.

The decisive self-test: with k-independent (isotropic) input, the anisotropic
nonlinear solver's Tc must match the isotropic solver's linearized Tc —
because the isotropic solver IS the linearization of these equations and Tc
is exactly where the nontrivial gap branches off (agreement is within a few
percent, limited by the gap-collapse Tc heuristic; see tc_aniso). A second
test checks that genuinely anisotropic input (two Fermi-surface pockets with
different coupling) yields two distinct gaps.
"""

import numpy as np
import pytest

from elphgap import tc_eliashberg
from elphgap.eliashberg_aniso import solve_gap_at_T, tc_aniso


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
