"""Parity: batched JAX backend vs. reference numpy backend.

The synthetic test below is self-contained; the DB test additionally checks
real materials and auto-skips unless the external BETE-NET database is
present (see docs/BENCHMARKS.md).
"""

from pathlib import Path

import numpy as np
import pytest

DB = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "betenet_database.json"


def _einstein_material(idx: int, lam: float, w_e: float):
    from elphgap.io import Material

    omega = np.linspace(max(0.05, w_e - 6.0), w_e + 6.0, 1500)
    g = np.exp(-0.5 * ((omega - w_e) / 0.5) ** 2)
    a2f = g * lam / (2.0 * np.trapezoid(g / omega, omega))
    return Material(index=idx, comp=f"S{idx}", comp_name=f"synthetic-{idx}",
                    omega=omega, a2f=a2f, lambda_ref=lam, wlog_ref=w_e, wsq_ref=w_e)


def test_tc_batched_matches_reference_on_synthetic_spectra():
    """Self-contained parity for the batched isotropic path (no external DB)."""
    from elphgap import tc_eliashberg
    from elphgap.eliashberg_jax import tc_batched

    mats = [_einstein_material(0, 0.6, 15.0),
            _einstein_material(1, 1.0, 20.0),
            _einstein_material(2, 1.6, 25.0)]
    tc_jax, censored = tc_batched(mats, n_mat=512)
    for m, tj, cj in zip(mats, tc_jax, censored):
        ref = tc_eliashberg(m.omega, m.a2f, n_max=512)
        assert cj == ref.censored, m.comp_name
        if not ref.censored:
            assert tj == pytest.approx(ref.tc_kelvin, rel=0.02), m.comp_name


@pytest.mark.skipif(not DB.exists(), reason="BETE-NET database not downloaded")
def test_jax_matches_numpy_on_db_materials():
    from elphgap import load_database, tc_eliashberg
    from elphgap.eliashberg_jax import tc_batched

    mats = load_database(str(DB))
    # Nb plus a few mid-Tc materials with compact grids to keep the test fast.
    picks = [m for m in mats if m.comp in ("Nb", "V", "Ta", "AlB2")][:4]
    assert len(picks) >= 3

    tc_jax, censored = tc_batched(picks, n_mat=1024)
    for m, tj, cj in zip(picks, tc_jax, censored):
        ref = tc_eliashberg(m.omega, m.a2f, n_max=1024)
        assert cj == ref.censored, m.comp
        if not ref.censored:
            assert tj == pytest.approx(ref.tc_kelvin, rel=0.02), m.comp
