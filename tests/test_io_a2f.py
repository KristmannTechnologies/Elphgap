"""Parser tests for read_a2f: real EPW (1+2N) / QE layouts, detection, fail-closed.

Fixtures use the authentic layouts. A complete EPW prefix.a2f is: the
' w[meV] ... for N smearing values' header, N a2F + N cumulative-lambda columns,
then the five footer records (Integrated el-ph coupling, Phonon smearing (meV),
Electron smearing (eV), Fermi window (eV), Summed el-ph coupling) in that order.
A complete QE a2F.dos has the 'frequencies in Rydberg' header and a
' lambda = ... Delta = ...' footer.
"""

import hashlib

import numpy as np
import pytest

from elphgap import A2FColumnError, A2FParseError, moments, read_a2f
from elphgap.io import RY_TO_MEV


def _cumulative_lambda(omega, a2f):
    wpos = np.where(omega > 0, omega, 1e-9)
    integ = 2.0 * a2f / wpos
    integ[omega <= 0] = 0.0
    return np.concatenate(([0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(omega))))


def _write_epw(tmp_path, name, omega, a2f_cols, smearings, *, lam_cols=None, header=True, header_n=None, footer=True):
    """Write a real-layout EPW prefix.a2f (full header + five footer records by default)."""
    omega = np.asarray(omega, dtype=float)
    a2f_cols = [np.asarray(a, dtype=float) for a in a2f_cols]
    n = len(a2f_cols)
    if lam_cols is None:
        lam_cols = [_cumulative_lambda(omega, a) for a in a2f_cols]
    lam_cols = [np.asarray(x, dtype=float) for x in lam_cols]
    cols = [omega, *a2f_cols, *lam_cols]
    parts = []
    if header:
        parts.append(f" w[meV] a2f and integrated 2*a2f/w for {(n if header_n is None else header_n):4d} smearing values")
    parts += ["".join(f"{c[i]:14.7f}" for c in cols) for i in range(omega.size)]
    if footer:
        integ = "  ".join(f"{c[-1]:.7f}" for c in lam_cols)
        smear = "  ".join(f"{s:.7f}" for s in smearings)
        parts += [
            " Integrated el-ph coupling", f"  #  {integ}",
            " Phonon smearing (meV)", f"  #  {smear}",
            " Electron smearing (eV)   0.0500000",
            " Fermi window (eV)   0.3000000",
            f" Summed el-ph coupling   {lam_cols[0][-1]:.7f}",
        ]
    p = tmp_path / name
    p.write_text("\n".join(parts) + "\n")
    return str(p)


def _write_qe(tmp_path, name="a2F.dos", n_modes=2, footer=True):
    omega_ry = np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.005])
    tot = np.array([0.0, 0.05, 0.30, 0.60, 0.35, 0.10])
    cols = [omega_ry, tot] + [tot / n_modes] * n_modes
    lines = ["".join(f"{c[i]:16.6E}" for c in cols) for i in range(omega_ry.size)]
    text = (
        "\n # Eliashberg function a2F (per both spin)\n #  frequencies in Rydberg\n"
        " # DOS normalized to E in Rydberg: a2F_total, a2F(mode)\n\n"
        + "\n".join(lines)
        + ("\n lambda =    1.05    Delta =  0.001\n" if footer else "\n")
    )
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _pb_like(n=241):
    w = np.linspace(0.0, 12.0, n)
    wpos = np.where(w > 0, w, 1e-9)
    g = np.exp(-0.5 * ((w - 4.5) / 1.0) ** 2) + 0.75 * np.exp(-0.5 * ((w - 8.5) / 1.2) ** 2)
    g[0] = 0.0
    a2f = g * (1.16 / (2.0 * np.trapezoid((g / wpos)[1:], w[1:])))
    return w, a2f


# --- QE -------------------------------------------------------------------- #

def test_qe_detection_units_and_footer(tmp_path):
    spec = read_a2f(_write_qe(tmp_path))
    assert spec.fmt == "qe" and spec.detected and spec.detection == "header"
    assert spec.units_in == "Ry" and spec.column == 2
    assert spec.column_kinds[:2] == ["omega", "a2f_total"] and spec.column_kinds[2] == "a2f_mode"
    assert spec.omega[0] == pytest.approx(0.001 * RY_TO_MEV, rel=1e-9)
    assert spec.lambda_footer == pytest.approx(1.05)
    assert any(w.code == "dropped_nonpositive" for w in spec.warnings)


def test_qe_per_mode_column_warns(tmp_path):
    spec = read_a2f(_write_qe(tmp_path), column=3)
    assert spec.column == 3
    assert any(w.code == "per_mode_column" for w in spec.warnings)


def test_qe_header_and_lambda_footer_mandatory(tmp_path):
    # Missing lambda footer (truncated QE run) -> exit 2.
    with pytest.raises(A2FParseError) as e:
        read_a2f(_write_qe(tmp_path, footer=False))
    assert e.value.code == "qe_lambda_footer_missing" and e.value.exit_code == 2
    # Missing the "frequencies in Rydberg" header -> exit 2.
    p = tmp_path / "noh.dos"
    p.write_text("0.001 0.05\n0.002 0.30\n lambda = 1.0   Delta = 0.001\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(p), fmt="qe")
    assert e.value.code == "qe_header_missing"


# --- EPW valid ------------------------------------------------------------- #

def test_epw_detection_columns_and_smearing(tmp_path):
    w, a2f = _pb_like()
    spec = read_a2f(_write_epw(tmp_path, "pb.a2f", w, [a2f], [0.5]))
    assert spec.fmt == "epw" and spec.n_columns == 3 and spec.n_smearings == 1 and spec.column == 2
    assert spec.column_kinds == ["omega", "a2f", "lambda_cumulative"]
    assert spec.smearing_meV == [0.5]
    lam, _, _ = moments(spec.omega, spec.a2f)
    assert lam == pytest.approx(1.16, rel=1e-3)
    assert not any(w.code == "lambda_crosscheck" for w in spec.warnings)
    assert spec.lambda_from_file[2] == pytest.approx(lam, rel=1e-3)


def test_epw_multi_smearing_requires_choice(tmp_path):
    w, a2f = _pb_like(120)
    spec = read_a2f(_write_epw(tmp_path, "sweep.a2f", w, [a2f, 0.9 * a2f], [0.1, 0.2]))
    assert spec.n_smearings == 2 and spec.primary_a2f_columns == [2, 3]
    assert spec.requires_column_choice
    assert spec.column_kinds == ["omega", "a2f", "a2f", "lambda_cumulative", "lambda_cumulative"]


def test_epw_cumulative_lambda_column_rejected(tmp_path):
    w, a2f = _pb_like(60)
    path = _write_epw(tmp_path, "pb.a2f", w, [a2f], [0.5])
    with pytest.raises(A2FColumnError) as e:
        read_a2f(path, column=3)  # column 3 is cumulative lambda
    assert e.value.code == "column_is_lambda" and e.value.exit_code == 4


def test_epw_lambda_crosscheck_warns_on_mismatch(tmp_path):
    w, a2f = _pb_like(120)
    lam = _cumulative_lambda(w, a2f) * 2.0  # inconsistent lambda column
    spec = read_a2f(_write_epw(tmp_path, "pb.a2f", w, [a2f], [0.5], lam_cols=[lam]))
    assert any(wn.code == "lambda_crosscheck" for wn in spec.warnings)


def test_no_magnitude_fallback_low_energy_mev_not_rescaled(tmp_path):
    w = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    a = np.array([0.0, 0.3, 0.6, 0.3, 0.1])
    spec = read_a2f(_write_epw(tmp_path, "lowe.a2f", w, [a], [0.05]), fmt="epw")
    assert spec.omega.max() == pytest.approx(0.8)  # meV, unscaled


def test_sha256_is_of_raw_bytes(tmp_path):
    w, a2f = _pb_like(60)
    path = _write_epw(tmp_path, "pb.a2f", w, [a2f], [0.5])
    spec = read_a2f(path)
    with open(path, "rb") as fh:
        raw = fh.read()
    assert spec.sha256 == hashlib.sha256(raw).hexdigest() and spec.n_bytes == len(raw)


def test_shipped_pb_example():
    import elphgap

    spec = read_a2f(elphgap.example_a2f_path())
    assert spec.fmt == "epw" and spec.n_columns == 3 and spec.smearing_meV == [0.5]
    lam, wlog, _ = moments(spec.omega, spec.a2f)
    assert 1.0 < lam < 1.3 and 3.0 < wlog < 8.0
    assert not any(w.code == "lambda_crosscheck" for w in spec.warnings)


# --- detection / format policy -------------------------------------------- #

def test_no_signature_autodetect_requires_format(tmp_path):
    p = tmp_path / "x.dat"
    p.write_text("5.0 0.1 0.2\n10.0 0.4 0.6\n15.0 0.1 0.7\n")
    with pytest.raises(A2FColumnError) as e:
        read_a2f(str(p))
    assert e.value.code == "format_undetectable" and e.value.exit_code == 4


def test_unknown_format_is_param_error(tmp_path):
    p = tmp_path / "x.dat"
    p.write_text("1 1\n2 2\n")
    with pytest.raises(A2FColumnError) as e:
        read_a2f(str(p), fmt="wat")
    assert e.value.code == "unknown_format" and e.value.exit_code == 4


def test_missing_file(tmp_path):
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(tmp_path / "nope.a2f"))
    assert e.value.code == "unreadable" and e.value.exit_code == 2


# --- early structural errors (fire before completeness) -------------------- #

def test_epw_column_count_must_be_odd(tmp_path):
    p = tmp_path / "bad.a2f"
    p.write_text("1.0 0.1\n2.0 0.4\n3.0 0.1\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(p), fmt="epw")
    assert e.value.code == "epw_column_count" and e.value.exit_code == 2


def test_nan_inf_in_data_rejected(tmp_path):
    p = tmp_path / "nan.a2f"
    p.write_text("1.0 0.1 0.1\n2.0 nan 0.2\n3.0 0.3 0.5\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(p), fmt="epw")
    assert e.value.code == "non_finite" and e.value.exit_code == 2


def test_malformed_and_ragged_and_empty(tmp_path):
    for name, text, code in [
        ("m.a2f", "1.0 0.1 0.1\n2.0 oops 0.2\n", "malformed_row"),
        ("r.a2f", "1.0 0.1 0.1\n2.0 0.2\n3.0 0.3 0.3\n", "ragged_columns"),
        ("e.a2f", "5.0 0.3 0.3\n", "no_data"),
    ]:
        p = tmp_path / name
        p.write_text(text)
        with pytest.raises(A2FParseError) as e:
            read_a2f(str(p), fmt="epw")
        assert e.value.code == code and e.value.exit_code == 2


def test_qe_single_column_is_clean_error(tmp_path):
    p = tmp_path / "one.dos"
    p.write_text("#  frequencies in Rydberg\n0.001\n0.002\n0.003\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(p))
    assert e.value.code == "too_few_columns" and e.value.exit_code == 2


# --- data cleaning / column policy (complete files) ------------------------ #

def test_negative_a2f_rejected_by_default_clamp_opt_in(tmp_path):
    w = np.array([1.0, 2.0, 3.0])
    a = np.array([0.10, -0.05, 0.30])
    path = _write_epw(tmp_path, "neg.a2f", w, [a], [0.5])
    with pytest.raises(A2FParseError) as e:
        read_a2f(path, fmt="epw")
    assert e.value.code == "negative_a2f" and e.value.exit_code == 2
    spec = read_a2f(path, fmt="epw", clamp_negative=True)
    assert (spec.a2f >= 0).all() and spec.clamped_negative == 1
    assert spec.most_negative_a2f == pytest.approx(-0.05)
    assert any(wn.code == "clamped_negative" for wn in spec.warnings)


def test_clip_below_negative_and_nonfinite_are_param_errors(tmp_path):
    w, a2f = _pb_like(60)
    path = _write_epw(tmp_path, "pb.a2f", w, [a2f], [0.5])
    for bad, code in [(-1.0, "clip_negative"), (float("nan"), "clip_not_finite")]:
        with pytest.raises(A2FColumnError) as e:
            read_a2f(path, clip_below_mev=bad)
        assert e.value.code == code and e.value.exit_code == 4


def test_clip_below_drops_and_warns(tmp_path):
    w = np.array([0.5, 1.5, 5.0, 10.0])
    a = np.array([0.1, 0.2, 0.4, 0.1])
    spec = read_a2f(_write_epw(tmp_path, "c.a2f", w, [a], [0.5]), fmt="epw", clip_below_mev=2.0)
    assert spec.omega.min() == pytest.approx(5.0)
    assert any(wn.code == "dropped_below_clip" for wn in spec.warnings)


def test_non_increasing_after_clean(tmp_path):
    w = np.array([5.0, 5.0, 6.0])
    a = np.array([0.1, 0.2, 0.3])
    with pytest.raises(A2FParseError) as e:
        read_a2f(_write_epw(tmp_path, "dup.a2f", w, [a], [0.5]), fmt="epw")
    assert e.value.code == "not_increasing"


def test_footer_smearing_count_hard_mismatch(tmp_path):
    w, a2f = _pb_like(60)  # N=2 block, but only one phonon smearing value
    path = _write_epw(tmp_path, "m.a2f", w, [a2f, 0.9 * a2f], [0.5])
    with pytest.raises(A2FParseError) as e:
        read_a2f(path, fmt="epw")
    assert e.value.code == "epw_smearing_count_mismatch" and e.value.exit_code == 2


def test_lambda_file_zero_with_positive_a2f_warns(tmp_path):
    w = np.array([1.0, 5.0, 10.0])
    a = np.array([0.10, 0.40, 0.10])
    spec = read_a2f(_write_epw(tmp_path, "z.a2f", w, [a], [0.5], lam_cols=[np.zeros(3)]), fmt="epw")
    assert any(wn.code == "lambda_crosscheck" for wn in spec.warnings)


def test_non_monotonic_lambda_column_warns(tmp_path):
    w = np.array([1.0, 5.0, 10.0])
    a = np.array([0.10, 0.40, 0.10])
    lam = np.array([0.90, 0.50, 0.10])  # decreasing -> not a valid cumulative
    spec = read_a2f(_write_epw(tmp_path, "nm.a2f", w, [a], [0.5], lam_cols=[lam]), fmt="epw")
    assert any(wn.code == "epw_lambda_not_monotonic" for wn in spec.warnings)


def test_require_column_before_column_specific_errors(tmp_path):
    # Multi-smearing with a negative default column 2: require_column (exit 4)
    # must fire before the negative-column data error (exit 2).
    w = np.array([1.0, 5.0, 10.0])
    path = _write_epw(tmp_path, "s.a2f", w, [np.array([-0.05, 0.4, 0.1]), np.array([0.09, 0.38, 0.09])], [0.1, 0.2])
    with pytest.raises(A2FColumnError) as e:
        read_a2f(path, fmt="epw", require_column=True)
    assert e.value.code == "column_required" and e.value.exit_code == 4


def test_non_selected_negatives_reported_not_clamped(tmp_path):
    w = np.array([1.0, 5.0, 10.0])
    path = _write_epw(tmp_path, "n.a2f", w, [np.array([0.10, 0.40, 0.10]), np.array([-0.05, 0.38, 0.09])], [0.1, 0.2])
    spec = read_a2f(path, fmt="epw", column=2)
    assert spec.negatives_by_column[3] == 1
    assert (spec.a2f_by_column[3] < 0).any()
    assert any(wn.code == "negative_a2f_other" for wn in spec.warnings)


# --- grammar: edge lines, finiteness (gate 3) ------------------------------ #

def test_broken_first_and_last_line(tmp_path):
    for name, text in [
        ("first", "BROKEN 0.1 0.0\n1.0 0.3 0.3\n2.0 0.1 0.4\n"),
        ("last", "1.0 0.3 0.3\n2.0 0.1 0.4\nBROKEN 0.1 0.0\n"),
    ]:
        p = tmp_path / f"{name}.a2f"
        p.write_text(text)
        with pytest.raises(A2FParseError) as e:
            read_a2f(str(p), fmt="epw")
        assert e.value.code == "malformed_row" and e.value.exit_code == 2


def test_block_grammar_and_commented_footer(tmp_path):
    for text in [
        "1.0 0.10 0.10\nBROKEN 0.2 0.3\n3.0 0.30 0.60\n",   # interleaved corruption
        "1.0 0.3 0.3\n2.0 0.4 0.5\n# Phonon smearing (meV)\n3.0 0.1 0.6\n",  # commented footer then data
    ]:
        p = tmp_path / "g.a2f"
        p.write_text(text)
        with pytest.raises(A2FParseError) as e:
            read_a2f(str(p), fmt="epw")
        assert e.value.code == "malformed_row" and e.value.exit_code == 2


def test_qe_lambda_nan_and_epw_footer_inf_rejected(tmp_path):
    q = tmp_path / "q.dos"
    q.write_text("#  frequencies in Rydberg\n0.001 0.05 0.05\n0.002 0.30 0.30\n lambda = nan   Delta = 0.001\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(q))
    assert e.value.code == "non_finite_footer" and e.value.exit_code == 2
    ic = tmp_path / "ic.a2f"
    ic.write_text("1.0 0.3 0.3\n2.0 0.1 0.4\n Integrated el-ph coupling\n  #  inf\n Phonon smearing (meV)\n  #  0.5\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(ic), fmt="epw")
    assert e.value.code == "non_finite_footer"


# --- gate 4: mandatory + ordered records, strict value binding ------------- #

def test_epw_header_n_must_match(tmp_path):
    for hn in (0, 2, 999999999):
        p = tmp_path / f"h{hn}.a2f"  # 3-column block => N=1, header claims otherwise
        p.write_text(f" w[meV] a2f and integrated 2*a2f/w for {hn} smearing values\n1.0 0.3 0.3\n2.0 0.1 0.4\n")
        with pytest.raises(A2FParseError) as e:
            read_a2f(str(p), fmt="epw")
        assert e.value.code == "epw_header_n_mismatch" and e.value.exit_code == 2


def test_epw_header_missing(tmp_path):
    # header omitted, footer present -> truncation/omission caught
    w = np.array([1.0, 2.0, 3.0])
    path = _write_epw(tmp_path, "nh.a2f", w, [np.array([0.3, 0.1, 0.05])], [0.5], header=False)
    with pytest.raises(A2FParseError) as e:
        read_a2f(path, fmt="epw")
    assert e.value.code == "epw_header_missing" and e.value.exit_code == 2


def test_epw_footer_truncated_incomplete(tmp_path):
    # The core truncation repro: header + data, cleanly cut before the footer.
    w = np.array([1.0, 2.0, 3.0])
    path = _write_epw(tmp_path, "trunc.a2f", w, [np.array([0.3, 0.1, 0.05])], [0.5], footer=False)
    with pytest.raises(A2FParseError) as e:
        read_a2f(path, fmt="epw")
    assert e.value.code == "epw_footer_incomplete" and e.value.exit_code == 2


def test_epw_footer_only_one_record(tmp_path):
    p = tmp_path / "one.a2f"
    p.write_text(
        " w[meV] a2f and integrated 2*a2f/w for    1 smearing values\n"
        "1.0 0.3 0.30\n2.0 0.1 0.40\n Fermi window (eV)   0.3\n"
    )
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(p), fmt="epw")
    assert e.value.code == "epw_footer_incomplete"


def test_epw_footer_out_of_order(tmp_path):
    p = tmp_path / "ord.a2f"
    p.write_text(
        " w[meV] a2f and integrated 2*a2f/w for    1 smearing values\n"
        "1.0 0.3 0.30\n2.0 0.1 0.40\n"
        " Integrated el-ph coupling\n  #  0.40\n Phonon smearing (meV)\n  #  0.5\n"
        " Fermi window (eV)   0.3\n Electron smearing (eV)   0.05\n Summed el-ph coupling   0.40\n"  # electron/fermi swapped
    )
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(p), fmt="epw")
    assert e.value.code == "epw_footer_order" and e.value.exit_code == 2


def test_complete_mini_epw_parses(tmp_path):
    w = np.array([1.0, 2.0, 3.0])
    spec = read_a2f(_write_epw(tmp_path, "full.a2f", w, [np.array([0.30, 0.10, 0.05])], [0.5]))
    assert spec.n_smearings == 1 and spec.n_columns == 3


def test_footer_before_data_and_duplicate_footer(tmp_path):
    fb = tmp_path / "fb.a2f"
    fb.write_text(" Phonon smearing (meV)\n  #  0.5\n1.0 0.3 0.3\n2.0 0.1 0.4\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(fb), fmt="epw")
    assert e.value.code == "footer_before_data"
    dup = tmp_path / "dup.a2f"
    dup.write_text("1.0 0.3 0.3\n2.0 0.1 0.4\n Fermi window (eV)   0.3\n Fermi window (eV)   0.3\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(dup), fmt="epw")
    assert e.value.code == "epw_duplicate_footer"


def test_qe_lambda_value_must_be_bound_same_line(tmp_path):
    # "lambda = BROKEN Delta = 0.001" must NOT take 0.001.
    p = tmp_path / "q.dos"
    p.write_text("#  frequencies in Rydberg\n0.001 0.05 0.05\n0.002 0.30 0.30\n lambda = BROKEN   Delta = 0.001\n")
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(p))
    assert e.value.code == "footer_malformed" and e.value.exit_code == 2


def test_epw_pending_value_not_collected_across_comments(tmp_path):
    # 'Integrated el-ph coupling' then '# BROKEN' then '# run 2026': must not borrow 2026.
    p = tmp_path / "b.a2f"
    p.write_text(
        " w[meV] a2f and integrated 2*a2f/w for    1 smearing values\n"
        "1.0 0.3 0.30\n2.0 0.1 0.40\n"
        " Integrated el-ph coupling\n# BROKEN\n# run 2026\n Phonon smearing (meV)\n  #  0.5\n"
    )
    with pytest.raises(A2FParseError) as e:
        read_a2f(str(p), fmt="epw")
    assert e.value.code == "footer_malformed"
