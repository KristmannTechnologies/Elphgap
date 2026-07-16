"""CLI tests: subprocess smoke, JSON manifest schema, and exit codes (0/2/3/4/5).

Invoked as `python -m elphgap ...` so the tests pass whether elphgap is
installed or only on PYTHONPATH (the src dir is forwarded to the child).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import elphgap

SRC_DIR = str(Path(elphgap.__file__).resolve().parents[1])  # .../src


def run(*args):
    env = {**os.environ, "PYTHONPATH": SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, "-m", "elphgap", *args], capture_output=True, text=True, env=env
    )


def _cum_lambda(omega, a2f):
    wpos = np.where(omega > 0, omega, 1e-9)
    integ = 2.0 * a2f / wpos
    integ[omega <= 0] = 0.0
    return np.concatenate(([0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(omega))))


def _write_epw(path, omega, a2f_cols, smearings, lam=None):
    """Full real-layout EPW prefix.a2f: header, data, five footer records in order."""
    if lam is None:
        lam = [_cum_lambda(omega, a) for a in a2f_cols]
    cols = [omega, *a2f_cols, *lam]
    n = len(a2f_cols)
    parts = [f" w[meV] a2f and integrated 2*a2f/w for {n:4d} smearing values"]
    parts += ["".join(f"{c[i]:14.7f}" for c in cols) for i in range(omega.size)]
    parts += [
        " Integrated el-ph coupling", "  #  " + "  ".join(f"{c[-1]:.6f}" for c in lam),
        " Phonon smearing (meV)", "  #  " + "  ".join(f"{s:.6f}" for s in smearings),
        " Electron smearing (eV)   0.0500000",
        " Fermi window (eV)   0.3000000",
        f" Summed el-ph coupling   {lam[0][-1]:.6f}",
    ]
    Path(path).write_text("\n".join(parts) + "\n")
    return str(path)


@pytest.fixture
def pb(tmp_path):
    w = np.linspace(0.0, 12.0, 121)
    wpos = np.where(w > 0, w, 1e-9)
    g = np.exp(-0.5 * ((w - 4.5) / 1.0) ** 2) + 0.75 * np.exp(-0.5 * ((w - 8.5) / 1.2) ** 2)
    g[0] = 0.0
    a2f = g * (1.2 / (2.0 * np.trapezoid((g / wpos)[1:], w[1:])))
    return _write_epw(tmp_path / "pb.a2f", w, [a2f], [0.5])


@pytest.fixture
def sweep(tmp_path):
    w = np.linspace(0.0, 12.0, 121)
    wpos = np.where(w > 0, w, 1e-9)
    g = np.exp(-0.5 * ((w - 6.0) / 1.5) ** 2)
    g[0] = 0.0
    a = g * (1.2 / (2.0 * np.trapezoid((g / wpos)[1:], w[1:])))
    return _write_epw(tmp_path / "sweep.a2f", w, [a, 0.95 * a], [0.1, 0.2])


def test_version():
    r = run("--version")
    assert r.returncode == 0 and "elphgap" in r.stdout


def test_inspect_human_shows_smearings(pb):
    r = run("inspect", pb)
    assert r.returncode == 0
    assert "lambda" in r.stdout and "a2F smearing columns" in r.stdout and "omega_log" in r.stdout


def test_inspect_json_manifest(pb):
    r = run("inspect", pb, "--json")
    assert r.returncode == 0
    d = json.loads(r.stdout)
    assert d["schema_version"] == "1" and d["command"] == "inspect"
    assert d["elphgap_version"] == elphgap.__version__
    assert d["format"]["name"] == "epw" and d["format"]["n_smearings"] == 1
    assert len(d["input"]["sha256"]) == 64 and d["input"]["bytes"] > 0
    assert d["spectrum"]["lambda"] > 0 and d["spectrum"]["omega_units"] == "meV"
    assert isinstance(d["smearings"], list) and d["smearings"][0]["column"] == 2
    assert all("code" in w and "message" in w for w in d["warnings"])  # structured


def test_tc_json_manifest_and_conventions(pb):
    r = run("tc", pb, "--json")
    assert r.returncode == 0
    d = json.loads(r.stdout)
    assert d["command"] == "tc" and d["tc"]["censored"] is False and d["tc"]["tc_K"] > 0
    c = d["conventions"]
    assert c["mu_star"] == 0.10 and c["n_max"] == 4096  # library default, not 512
    assert "mu_star_convention" in c and c["output_units"] == "K"
    assert c["omega_c_meV"] == pytest.approx(10.0 * d["spectrum"]["omega_max_meV"])
    assert c["t_floor_K"] > 0 and c["t_max_K"] == 2000.0 and c["rtol"] == 1e-3


def test_tc_fast_sets_nmax_512(pb):
    d = json.loads(run("tc", pb, "--fast", "--json").stdout)
    assert d["conventions"]["n_max"] == 512


def test_tc_human_states_units(pb):
    r = run("tc", pb)
    assert r.returncode == 0
    assert " K" in r.stdout and "meV" in r.stdout and "output units: K" in r.stdout


def test_tc_multismearing_requires_column(sweep):
    r = run("tc", sweep, "--format", "epw")
    assert r.returncode == 4 and "column_required" in r.stderr
    ok = run("tc", sweep, "--format", "epw", "--column", "3")
    assert ok.returncode == 0


def test_inspect_multismearing_shows_all(sweep):
    d = json.loads(run("inspect", sweep, "--format", "epw", "--json").stdout)
    assert d["format"]["n_smearings"] == 2 and len(d["smearings"]) == 2
    assert {r["column"] for r in d["smearings"]} == {2, 3}


def test_tc_censored_exit3(tmp_path):
    # Genuinely weak coupling (lambda ~ 0.25): Tc falls below the --fast floor.
    w = np.linspace(0.0, 12.0, 121)
    wpos = np.where(w > 0, w, 1e-9)
    g = np.exp(-0.5 * ((w - 6.0) / 1.5) ** 2)
    g[0] = 0.0
    a2f = g * (0.25 / (2.0 * np.trapezoid((g / wpos)[1:], w[1:])))
    weak = _write_epw(tmp_path / "weak.a2f", w, [a2f], [0.5])
    r = run("tc", weak, "--mu-star", "0.16", "--fast", "--json")
    assert r.returncode == 3
    d = json.loads(r.stdout)
    assert d["tc"]["censored"] is True and d["tc"]["tc_K"] == 0.0


def test_tc_solver_error_exit5(tmp_path):
    # Extreme coupling (lambda ~ 50) at high omega: bisection cannot bracket Tc
    # below t_max -> RuntimeError -> exit 5.
    w = np.linspace(1.0, 600.0, 300)
    g = np.exp(-0.5 * ((w - 400.0) / 20.0) ** 2)
    a2f = g * (50.0 / (2.0 * np.trapezoid(g / w, w)))
    extreme = _write_epw(tmp_path / "extreme.a2f", w, [a2f], [0.5])
    r = run("tc", extreme, "--format", "epw", "--n-max", "8")
    assert r.returncode == 5 and "error[solver_no_bracket]" in r.stderr


def test_cumulative_column_exit4(pb):
    r = run("inspect", pb, "--column", "3")
    assert r.returncode == 4 and "column_is_lambda" in r.stderr


def test_bad_params_exit4(pb):
    assert run("tc", pb, "--mu-star", "1.5").returncode == 4
    assert run("tc", pb, "--cutoff-factor", "0").returncode == 4
    assert run("tc", pb, "--n-max", "2").returncode == 4
    assert run("inspect", pb, "--format", "zzz").returncode == 4


def test_negative_default_exit2_clamp_ok(tmp_path):
    w = np.array([1.0, 2.0, 3.0])
    p = _write_epw(tmp_path / "neg.a2f", w, [np.array([0.1, -0.05, 0.3])], [0.5])
    assert run("inspect", p, "--format", "epw").returncode == 2
    assert run("inspect", p, "--format", "epw", "--clamp-negative").returncode == 0


def test_missing_file_exit2(tmp_path):
    r = run("inspect", str(tmp_path / "nope.a2f"))
    assert r.returncode == 2 and "error[unreadable]" in r.stderr


def test_json_has_no_nan(pb):
    out = run("tc", pb, "--json").stdout
    assert "NaN" not in out and "Infinity" not in out
    json.loads(out)  # strict parse succeeds


def test_usage_errors_exit4(pb):
    assert run("inspect", pb, "--bogus").returncode == 4
    assert run().returncode == 4


# --- re-gate blockers ------------------------------------------------------ #

def test_nonfinite_params_exit4(pb):
    assert run("tc", pb, "--mu-star", "nan").returncode == 4
    assert run("tc", pb, "--cutoff-factor", "inf").returncode == 4
    assert run("tc", pb, "--cutoff-factor", "nan").returncode == 4
    assert run("inspect", pb, "--clip-below", "nan").returncode == 4


def test_multismearing_negative_col2_column_required_first(tmp_path):
    # tc on a sweep whose default column 2 is negative: exit 4 (column_required),
    # NOT exit 2 (negative) — the missing choice is checked first.
    w = np.array([1.0, 5.0, 10.0])
    p = _write_epw(tmp_path / "s.a2f", w, [np.array([-0.05, 0.40, 0.10]), np.array([0.09, 0.38, 0.09])], [0.1, 0.2])
    r = run("tc", p, "--format", "epw")
    assert r.returncode == 4 and "column_required" in r.stderr


def test_broken_midblock_exit2(tmp_path):
    p = tmp_path / "b.a2f"
    p.write_text("1.0 0.1 0.1\nBROKEN 0.2 0.3\n3.0 0.3 0.6\n")
    r = run("inspect", str(p), "--format", "epw")
    assert r.returncode == 2 and "malformed_row" in r.stderr


def test_broken_edge_lines_exit2(tmp_path):
    first = tmp_path / "first.a2f"
    first.write_text("BROKEN 0.1 0.0\n1.0 0.3 0.3\n2.0 0.1 0.4\n")
    assert run("inspect", str(first), "--format", "epw").returncode == 2
    last = tmp_path / "last.a2f"
    last.write_text("1.0 0.3 0.3\n2.0 0.1 0.4\nBROKEN 0.1 0.0\n")
    assert run("inspect", str(last), "--format", "epw").returncode == 2


def test_nonfinite_footer_same_exit_human_and_json(tmp_path):
    # The gate divergence: QE lambda=nan must be exit 2 in BOTH human and JSON.
    p = tmp_path / "q.dos"
    p.write_text("#  frequencies in Rydberg\n0.001 0.05 0.05\n0.002 0.30 0.30\n lambda = nan   Delta = 1e-3\n")
    human = run("inspect", str(p))
    js = run("inspect", str(p), "--json")
    assert human.returncode == 2 and js.returncode == 2
    assert "non_finite_footer" in human.stderr and "non_finite_footer" in js.stderr


def _qe_file(tmp_path):
    p = tmp_path / "a2F.dos"
    p.write_text(
        "#  frequencies in Rydberg\n0.001 0.05 0.05\n0.002 0.30 0.30\n"
        "0.003 0.60 0.60\n0.004 0.20 0.20\n lambda = 0.9   Delta = 0.001\n"
    )
    return str(p)


def test_qe_human_reports_ry_conversion(tmp_path):
    r = run("tc", _qe_file(tmp_path))
    assert r.returncode in (0, 3)
    assert "input omega: Ry -> meV" in r.stdout  # not "meV"


def test_human_manifest_is_complete(pb):
    # Human tc must carry the same manifest fields as JSON: cleaning + smearing.
    r = run("tc", pb, "--mu-star", "0.10")
    assert r.returncode == 0
    for token in ("cleaning", "clip <=", "clamped", "smearing", "a2F smearing columns", "conventions"):
        assert token in r.stdout, token
