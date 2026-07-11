#!/usr/bin/env python3
"""Exercise the anisotropic ME solver on the canonical MgB2 TWO-BAND model.

This is a SOLVER check against a real material's textbook parametrization. We
feed the literature 2-band electron-phonon coupling matrix for MgB2 into the
solver and check that it reproduces the qualitative two-gap structure (well
separated Δσ/Δπ, Δσ in the right few-meV range). Absolute Tc comes out HIGH
(mid-40s to ~60 K vs. the real 39 K) — the known artifact of collapsing each
α²F_ij onto a single Einstein mode. This is a trend scan, not a fit.

Literature coupling matrix (Golubov et al., J. Phys.: Condens. Matter 14, 1353
(2002); consistent with Choi et al., Nature 418, 758 (2002) and Margine &
Giustino, PRB 87, 024505 (2013)):
    λ_σσ ≈ 1.02   λ_σπ ≈ 0.21
    λ_πσ ≈ 0.16   λ_ππ ≈ 0.45
  -> per-band totals λ_σ=Σ_j λ_σj ≈ 1.23, λ_π ≈ 0.61; DOS-averaged λ ≈ 0.75
     (EPW's standard MgB2 example yields isotropic λ ≈ 0.8 -- same order).
DOS fractions at E_F: N_σ/N ≈ 0.44, N_π/N ≈ 0.56.

CAVEATS (honest):
  - We approximate each α²F_ij(ω) by a SINGLE Einstein mode at a representative
    frequency (the E2g bond-stretching mode dominates σ-coupling at ~67 meV; π
    couples to lower modes too). A single ω is a simplification -> quantitative
    Tc/gaps are approximate. We scan ω and μ* to show the trend, not to tune a fit.
  - μ* convention: this solver uses μ*_ij^std = w_j · mu_star[i,j], so we pass
    mu_star[i,j] = μ*_std / w_j to realize a standard uniform μ*_std.

Run:  python benchmarks/mgb2_twoband.py
"""
import numpy as np
from elphgap.eliashberg_aniso import solve_gap_at_T, tc_aniso, tc_aniso_linearized

# --- literature MgB2 2-band model -------------------------------------------------
LAM = np.array([[1.02, 0.21],     # [[σσ, σπ],
                [0.16, 0.45]])    #  [πσ, ππ]]
W = np.array([0.44, 0.56])        # DOS fractions (σ, π)
MU_STD = 0.13                     # standard Coulomb pseudopotential (uniform)
TARGET = "Exp/theory: Δσ≈7.0, Δπ≈2.3 meV, Tc≈39 K"


def einstein(lam, w_e, sigma=0.5, n=1500):
    """α²F(ω) as a narrow Gaussian at w_e carrying coupling `lam` (2∫α²F/ω = lam)."""
    omega = np.linspace(max(0.5, w_e - 10 * sigma), w_e + 10 * sigma, n)
    g = np.exp(-0.5 * ((omega - w_e) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return omega, 0.5 * lam * w_e * g


def build_pairs(w_e):
    """2x2 α²F matrix on a shared grid, each channel an Einstein peak at w_e.

    CONVENTION: this solver evaluates Σ_j w_j·λ_solver(i,j), so the standard band
    coupling (Golubov's λ_ij, which already includes the target-band DOS N_j) maps
    to λ_solver(i,j) = λ_ij^std / w_j -- i.e. a2f_pairs[i,j] must encode λ_ij/w_j.
    (Same rule as μ*: μ*_ij^std = w_j·mu_star[i,j].) Feeding λ_ij directly makes the
    solver see w_j·λ_ij ~ half the intended coupling -> Tc collapses (the trap).
    """
    omega, _ = einstein(LAM[0, 0] / W[0], w_e)
    a = np.zeros((2, 2, omega.size))
    for i in range(2):
        for j in range(2):
            _, aij = einstein(LAM[i, j] / W[j], w_e)
            a[i, j] = aij
    return omega, a


def mu_matrix(mu_std):
    # mu_star[i,j] = mu_std / w_j   so that solver's mu*_ij^std = w_j*mu_star = mu_std
    return np.array([[mu_std / W[0], mu_std / W[1]],
                     [mu_std / W[0], mu_std / W[1]]])


def gaps_at(omega, a, mu, t_kelvin=4.0):
    st = solve_gap_at_T(omega, a, W, t_kelvin, mu_star=mu, n_max=512)
    if not st.converged:
        return None, None, False
    # delta[k, 0] = leading-Matsubara gap per band (meV)
    return abs(st.delta[0, 0]), abs(st.delta[1, 0]), True


def main():
    print(__doc__.split("Run:")[0])
    print(f"  {TARGET}\n")
    # Tc_lin = linearized-kernel bisection (exact, recommended for quoting Tc);
    # Tc_gap = nonlinear gap-collapse heuristic (biased slightly high near Tc).
    print(f"{'ω_E[meV]':>8} {'μ*_std':>7} {'Δσ[meV]':>9} {'Δπ[meV]':>9} {'Tc_lin[K]':>9} {'Tc_gap[K]':>9} {'conv':>5}")
    for w_e in (60.0, 67.0, 75.0):
        for mu_std in (0.10, 0.13, 0.16):
            omega, a = build_pairs(w_e)
            mu = mu_matrix(mu_std)
            ds, dp, ok = gaps_at(omega, a, mu, t_kelvin=4.0)
            try:
                tc_lin = tc_aniso_linearized(omega, a, W, mu_star=mu, n_max=512).tc_kelvin
                tc = tc_aniso(omega, a, W, mu_star=mu, n_max=512)
            except Exception:
                tc_lin, tc = float("nan"), float("nan")
            if ok:
                print(f"{w_e:8.0f} {mu_std:7.2f} {ds:9.2f} {dp:9.2f} {tc_lin:9.1f} {tc:9.1f} {'yes':>5}")
            else:
                print(f"{w_e:8.0f} {mu_std:7.2f} {'--':>9} {'--':>9} {tc_lin:9.1f} {tc:9.1f} {'NO':>5}")

    print("\nInterpretation:")
    print("  - Two distinct gaps (Δσ >> Δπ) across the whole scan = the solver")
    print("    reproduces MgB2's two-band structure from the literature couplings.")
    print("  - Quantitative Δσ/Δπ/Tc depend on the (single-mode) α²F model + μ*;")
    print("    the converged EPW α²F(ω) shape (not a single Einstein peak) is the")
    print("    next refinement. This run validates the SOLVER, not a specific α²F.")


if __name__ == "__main__":
    main()
