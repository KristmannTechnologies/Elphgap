"""Analytic anchors: Einstein spectrum a2F = (lam*w_E/2) * delta(w - w_E).

On a grid the delta is a narrow Gaussian; moments must then satisfy
lambda -> lam, omega_log -> w_E, omega_2 -> w_E exactly in the narrow limit.
The Allen-Dynes formula is checked against a hand-evaluated value, and the
Eliashberg solver against AD within the ~10-15% accuracy AD was fitted to
for moderate coupling (Allen & Dynes 1975).
"""

import numpy as np
import pytest

from elphgap import MEV_TO_K, moments, tc_allen_dynes, tc_eliashberg


def einstein_spectrum(lam: float, w_e: float, sigma: float = 0.05, n: int = 4000):
    omega = np.linspace(w_e - 12 * sigma, w_e + 12 * sigma, n)
    gauss = np.exp(-0.5 * ((omega - w_e) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    a2f = 0.5 * lam * w_e * gauss
    return omega, a2f


@pytest.mark.parametrize("lam,w_e", [(0.5, 10.0), (1.0, 25.0), (2.0, 5.0)])
def test_moments_einstein(lam, w_e):
    omega, a2f = einstein_spectrum(lam, w_e)
    lam_c, wlog_c, w2_c = moments(omega, a2f)
    assert lam_c == pytest.approx(lam, rel=1e-3)
    assert wlog_c == pytest.approx(w_e, rel=1e-3)
    assert w2_c == pytest.approx(w_e, rel=1e-3)


def test_allen_dynes_hand_value():
    # lam=1, mu*=0.1, Einstein (r=1 -> f2=1): hand evaluation of the formula.
    # L1 = 2.46*(1+0.38) = 3.3948; f1 = (1 + (1/3.3948)^1.5)^(1/3) = 1.050680
    # exp(-1.04*2/(1 - 0.1*1.62)) = exp(-2.482100) = 0.0835676
    # Tc = f1 * (w_log/1.2) * 0.0835676 ; w_log = 10 meV -> Tc in K:
    expected_mev = 1.050680 * (10.0 / 1.2) * 0.0835676
    assert tc_allen_dynes(1.0, 10.0, 10.0, mu_star=0.10) == pytest.approx(
        expected_mev * MEV_TO_K, rel=1e-4
    )


def test_allen_dynes_f2_hand_value():
    # Shape correction actually exercised (r != 1): lam=1, mu*=0.1, w_log=10,
    # w_2=20 meV -> r=2. Hand evaluation:
    # L2 = 1.82*(1+0.63)*2 = 5.933200; L2^2 = 35.202862
    # f2 = 1 + (2-1)*1/(1+35.202862) = 1.02762213
    # f1 = 1.050680, exp(-2.482100) = 0.0835676 (as in the r=1 test)
    expected_mev = 1.050680 * 1.0276221 * (10.0 / 1.2) * 0.0835676
    assert tc_allen_dynes(1.0, 10.0, 20.0, mu_star=0.10) == pytest.approx(
        expected_mev * MEV_TO_K, rel=1e-4
    )


def test_mcmillan_excludes_strong_coupling_factors():
    # McMillan = AD exponential with f1 = f2 = 1; for lam=1, mu*=0.1, w_log=10 meV:
    # Tc = (10/1.2) * exp(-2.482100) = 0.696434 meV (no f1 = 1.050680 factor).
    from elphgap import tc_mcmillan

    expected_mev = (10.0 / 1.2) * 0.0835676
    assert tc_mcmillan(1.0, 10.0, mu_star=0.10) == pytest.approx(expected_mev * MEV_TO_K, rel=1e-4)
    assert tc_mcmillan(1.0, 10.0) < tc_allen_dynes(1.0, 10.0, 10.0)


def test_allen_dynes_zero_below_threshold():
    # Denominator lam - mu*(1+0.62 lam) <= 0 -> no superconductivity.
    assert tc_allen_dynes(0.1, 10.0, 10.0, mu_star=0.10) == 0.0


@pytest.mark.parametrize("lam", [0.7, 1.0, 1.5])
def test_eliashberg_close_to_allen_dynes_moderate_coupling(lam):
    w_e = 20.0
    omega, a2f = einstein_spectrum(lam, w_e)
    tc_ad = tc_allen_dynes(lam, w_e, w_e, mu_star=0.10)
    res = tc_eliashberg(omega, a2f, mu_star=0.10)
    assert not res.censored
    assert res.tc_kelvin == pytest.approx(tc_ad, rel=0.15)


def test_eliashberg_monotonic_in_lambda():
    w_e = 20.0
    tcs = []
    for lam in (0.6, 1.0, 1.6):
        omega, a2f = einstein_spectrum(lam, w_e)
        tcs.append(tc_eliashberg(omega, a2f).tc_kelvin)
    assert tcs[0] < tcs[1] < tcs[2]


def test_eliashberg_censored_for_weak_coupling():
    # lam=0.25, mu*=0.13: Tc is essentially zero -> below resolvable floor.
    omega, a2f = einstein_spectrum(0.25, 10.0)
    res = tc_eliashberg(omega, a2f, mu_star=0.13, n_max=512)
    assert res.censored


def test_t_max_kelvin_reachable_tc_not_rejected():
    """Regression: a Tc below t_max_kelvin must be found even when the bracket
    doubling would overshoot t_max (lam=1, w_E=100 meV -> Tc ~93 K; the
    unclamped bracket jumped ~58 K -> ~116 K and raised at t_max=100 K)."""
    omega, a2f = einstein_spectrum(1.0, 100.0)
    unlimited = tc_eliashberg(omega, a2f, mu_star=0.10, n_max=128).tc_kelvin
    assert 90.0 < unlimited < 100.0  # precondition for the clamp scenario
    capped = tc_eliashberg(omega, a2f, mu_star=0.10, n_max=128, t_max_kelvin=100.0)
    assert capped.tc_kelvin == pytest.approx(unlimited, rel=2e-3)
    with pytest.raises(RuntimeError, match="t_max_kelvin"):
        tc_eliashberg(omega, a2f, mu_star=0.10, n_max=128, t_max_kelvin=50.0)


def test_t_max_kelvin_initial_bracket_clamped():
    """Regression: with t_max between the floor and the FIRST bracket endpoint
    (2*floor), the expansion loop never ran and bisection silently returned a
    Tc ABOVE t_max_kelvin. n_max=31 puts the floor at ~60 K, true Tc ~93 K:
    t_max=80 K must raise, not return ~93 K."""
    omega, a2f = einstein_spectrum(1.0, 100.0)
    tc = tc_eliashberg(omega, a2f, mu_star=0.10, n_max=31).tc_kelvin
    assert tc > 80.0  # precondition: Tc really above the cap
    with pytest.raises(RuntimeError, match="t_max_kelvin"):
        tc_eliashberg(omega, a2f, mu_star=0.10, n_max=31, t_max_kelvin=80.0)


def test_omega_grid_validation():
    """omega <= 0 or a non-increasing grid must fail loudly at the API boundary,
    not as NaN moments / eigensolver errors much later."""
    omega = np.linspace(0.0, 10.0, 100)  # omega[0] = 0: singular kernels
    a2f = np.ones_like(omega)
    with pytest.raises(ValueError, match="strictly positive"):
        moments(omega, a2f)
    with pytest.raises(ValueError, match="strictly positive"):
        tc_eliashberg(omega, a2f)
    with pytest.raises(ValueError, match="strictly increasing"):
        moments(np.array([1.0, 3.0, 2.0]), np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        moments(np.array([1.0, 2.0, np.inf]), np.ones(3))
