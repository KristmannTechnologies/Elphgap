"""CLI tests: subprocess smoke, JSON schema, and exit codes (0/2/3/4).

Invoked as `python -m elphgap ...` so the tests pass whether elphgap is
installed or only on PYTHONPATH (the src dir is forwarded to the child).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import elphgap

SRC_DIR = str(Path(elphgap.__file__).resolve().parents[1])  # .../src


def run(*args, cwd=None):
    env = {**os.environ, "PYTHONPATH": SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, "-m", "elphgap", *args],
        capture_output=True, text=True, env=env, cwd=cwd,
    )


@pytest.fixture
def qe(tmp_path):
    p = tmp_path / "a2F.dos"
    p.write_text(
        "#  frequencies in Rydberg\n"
        + "\n".join(
            f"{w:.6f}  {a:.6f}"
            for w, a in [(0.001, 0.05), (0.002, 0.30), (0.003, 0.60), (0.004, 0.35), (0.005, 0.10)]
        )
        + "\n"
    )
    return str(p)


@pytest.fixture
def epw_strong(tmp_path):
    # lambda ~ 1.2, omega peak ~ 8 meV -> a finite, resolvable Tc.
    import numpy as np

    w = np.linspace(0.0, 20.0, 81)
    g = np.exp(-0.5 * ((w - 8.0) / 2.0) ** 2)
    g[0] = 0.0
    a2f = np.zeros_like(w)
    a2f[1:] = 1.2 / (2.0 * np.trapezoid((g / np.where(w > 0, w, 1))[1:], w[1:])) * g[1:]
    p = tmp_path / "strong.a2f"
    p.write_text("# w[meV] a2F\n" + "\n".join(f"{wi:.6f} {ai:.6f}" for wi, ai in zip(w, a2f)) + "\n")
    return str(p)


def test_version():
    r = run("--version")
    assert r.returncode == 0 and "elphgap" in r.stdout


def test_inspect_human(qe):
    r = run("inspect", qe)
    assert r.returncode == 0
    assert "lambda" in r.stdout and "omega_log" in r.stdout


def test_inspect_json_schema(qe):
    r = run("inspect", qe, "--json")
    assert r.returncode == 0
    doc = json.loads(r.stdout)
    assert doc["command"] == "inspect"
    assert doc["elphgap_version"] == elphgap.__version__
    assert doc["format"]["name"] == "qe"
    assert doc["input"]["sha256"] and len(doc["input"]["sha256"]) == 64
    assert doc["spectrum"]["lambda"] > 0
    assert "omega_log_mev" in doc["spectrum"]


def test_tc_json_schema_and_value(epw_strong):
    r = run("tc", epw_strong, "--format", "epw", "--json")
    assert r.returncode == 0
    doc = json.loads(r.stdout)
    assert doc["command"] == "tc"
    assert doc["censored"] is False
    assert doc["tc_kelvin"] > 0
    conv = doc["conventions"]
    assert conv["mu_star"] == 0.10 and conv["cutoff_factor"] == 10.0 and conv["n_max"] == 512
    assert conv["omega_c_mev"] == pytest.approx(10.0 * doc["spectrum"]["omega_max_mev"])


def test_tc_censored_exit3(epw_strong):
    # Weak effective coupling via very high mu* + tiny n_max -> below the floor.
    r = run("tc", epw_strong, "--format", "epw", "--mu-star", "0.9", "--n-max", "8", "--json")
    assert r.returncode == 3
    assert json.loads(r.stdout)["censored"] is True


def test_bad_mu_star_exit4(epw_strong):
    r = run("tc", epw_strong, "--mu-star", "1.5")
    assert r.returncode == 4 and "mu-star" in r.stderr


def test_column_one_exit4(qe):
    r = run("inspect", qe, "--column", "1")
    assert r.returncode == 4 and "column" in r.stderr


def test_missing_file_exit2(tmp_path):
    r = run("inspect", str(tmp_path / "nope.a2f"))
    assert r.returncode == 2 and "parse error" in r.stderr


def test_broken_file_exit2(tmp_path):
    p = tmp_path / "broken.a2f"
    p.write_text("1.0 0.1\n2.0 oops\n3.0 0.3\n")
    r = run("inspect", str(p), "--format", "epw")
    assert r.returncode == 2


def test_unknown_flag_exit4(qe):
    r = run("inspect", qe, "--bogus")
    assert r.returncode == 4


def test_no_subcommand_exit4():
    r = run()
    assert r.returncode == 4
