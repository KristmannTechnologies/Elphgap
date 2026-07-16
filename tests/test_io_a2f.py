"""Parser tests for read_a2f: QE + EPW formats, autodetection, cleaning, errors."""

import numpy as np
import pytest

from elphgap import A2FParseError, moments, read_a2f
from elphgap.io import RY_TO_MEV

QE_HEADER = "# Eliashberg function a2F (per spin)\n#  frequencies in Rydberg\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _qe_file(tmp_path, name="a2F.dos"):
    # omega[Ry], a2F_total, a2F_mode1. 0.003 Ry ~ 40.8 meV.
    body = "\n".join(
        f"{w:.6f}  {a:.6f}  {a:.6f}"
        for w, a in [
            (0.0000, 0.00), (0.0010, 0.05), (0.0020, 0.30),
            (0.0030, 0.60), (0.0040, 0.35), (0.0050, 0.10), (0.0060, 0.02),
        ]
    )
    return _write(tmp_path, name, QE_HEADER + body + "\n lambda = 1.05\n")


def _epw_file(tmp_path, name="pb.a2f", ncols=3):
    # omega[meV] + `ncols` a2F smearing columns; EPW-style comment footer.
    header = "# a2F(w) for different phonon smearings\n#   w[meV]   a2F...\n"
    lines = []
    for w in np.linspace(0.0, 20.0, 41):
        vals = "  ".join(f"{np.exp(-0.5*((w-10.0)/2.0)**2)*(1.0+0.05*c):.6f}" for c in range(ncols))
        lines.append(f"{w:.6f}  {vals}")
    footer = "# Integrated el-ph coupling\n#  1.10  1.11  1.12\n# Phonon smearing (meV)\n#  0.05  0.10  0.15\n"
    return _write(tmp_path, name, header + "\n".join(lines) + "\n" + footer)


def test_qe_units_and_detection(tmp_path):
    spec = read_a2f(_qe_file(tmp_path))
    assert spec.fmt == "qe" and spec.detected and spec.detection == "header"
    assert spec.units_in == "Ry" and spec.column == 2
    # omega converted Ry -> meV; 0.0 row dropped, so first kept is 0.001 Ry.
    assert spec.omega[0] == pytest.approx(0.001 * RY_TO_MEV, rel=1e-9)
    assert spec.omega.max() == pytest.approx(0.006 * RY_TO_MEV, rel=1e-9)
    assert any("omega <= 0" in w for w in spec.warnings)


def test_epw_units_column_and_smearing_warning(tmp_path):
    spec = read_a2f(_epw_file(tmp_path), fmt="epw")
    assert spec.fmt == "epw" and spec.units_in == "meV"
    assert spec.column == 2 and spec.n_columns == 4
    assert spec.omega.max() == pytest.approx(20.0)
    assert any("smearing" in w.lower() for w in spec.warnings)


def test_epw_autodetect_via_footer(tmp_path):
    # No "meV"/"Rydberg" string anywhere (the usual "Phonon smearing (meV)" line
    # is omitted on purpose), so detection must fall to the EPW footer marker.
    text = "# a2F\n 5.0 0.1\n 10.0 0.4\n 15.0 0.1\n# Integrated el-ph coupling\n#  1.0\n"
    spec = read_a2f(_write(tmp_path, "x.a2f", text))
    assert spec.fmt == "epw" and spec.detection == "footer"


def test_autodetect_via_magnitude(tmp_path):
    # No decisive header. Small omega -> Rydberg (qe); large -> meV (epw).
    small = "0.001 0.1\n0.002 0.4\n0.003 0.1\n"
    large = "5.0 0.1\n10.0 0.4\n15.0 0.1\n"
    assert read_a2f(_write(tmp_path, "s.dat", small)).fmt == "qe"
    assert read_a2f(_write(tmp_path, "l.dat", large)).fmt == "epw"
    assert read_a2f(_write(tmp_path, "s2.dat", small)).detection == "magnitude"


def test_forced_format_overrides_detection(tmp_path):
    # Rydberg header but forced epw: omega must NOT be scaled by RY_TO_MEV.
    spec = read_a2f(_qe_file(tmp_path), fmt="epw")
    assert spec.fmt == "epw" and not spec.detected and spec.detection == "forced"
    assert spec.omega.max() == pytest.approx(0.006)  # meV, unscaled


def test_column_override(tmp_path):
    spec = read_a2f(_epw_file(tmp_path), fmt="epw", column=3)
    assert spec.column == 3
    # column 3 is scaled up 5 % vs column 2 -> larger lambda.
    lam3, _, _ = moments(spec.omega, spec.a2f)
    lam2, _, _ = moments(*(lambda s: (s.omega, s.a2f))(read_a2f(_epw_file(tmp_path), fmt="epw")))
    assert lam3 > lam2


def test_negative_a2f_clamped_with_warning(tmp_path):
    text = "1.0 0.10\n2.0 -0.05\n3.0 0.30\n4.0 0.10\n"
    spec = read_a2f(_write(tmp_path, "neg.a2f", text), fmt="epw")
    assert (spec.a2f >= 0.0).all()
    assert any("clamped" in w and "negative" in w for w in spec.warnings)


def test_clip_below_drops_and_warns(tmp_path):
    text = "0.5 0.1\n1.5 0.2\n5.0 0.4\n10.0 0.1\n"
    spec = read_a2f(_write(tmp_path, "c.a2f", text), fmt="epw", clip_below_mev=2.0)
    assert spec.omega.min() == pytest.approx(5.0)
    assert any("<= 2" in w and "clip_below" in w for w in spec.warnings)


def test_broken_row_raises(tmp_path):
    text = "1.0 0.1\n2.0 oops\n3.0 0.3\n"
    with pytest.raises(A2FParseError, match="numeric data row"):
        read_a2f(_write(tmp_path, "b.a2f", text), fmt="epw")


def test_ragged_columns_raise(tmp_path):
    text = "1.0 0.1 0.1\n2.0 0.2\n3.0 0.3 0.3\n"
    with pytest.raises(A2FParseError, match="inconsistent column count"):
        read_a2f(_write(tmp_path, "r.a2f", text), fmt="epw")


def test_column_beyond_width_raises(tmp_path):
    with pytest.raises(A2FParseError, match="only 2 column"):
        read_a2f(_write(tmp_path, "n.a2f", "1.0 0.1\n2.0 0.2\n3.0 0.3\n"), fmt="epw", column=5)


def test_too_few_points_raises(tmp_path):
    with pytest.raises(A2FParseError, match="need at least 2"):
        read_a2f(_write(tmp_path, "one.a2f", "# c\n5.0 0.3\n"), fmt="epw")


def test_unknown_format_raises(tmp_path):
    with pytest.raises(A2FParseError, match="unknown format"):
        read_a2f(_write(tmp_path, "x.a2f", "1 1\n2 2\n"), fmt="wat")


def test_missing_file_raises():
    with pytest.raises(A2FParseError, match="cannot read"):
        read_a2f("/nonexistent/path/does-not-exist.a2f")


def test_non_increasing_after_cleaning_raises(tmp_path):
    text = "5.0 0.1\n5.0 0.2\n6.0 0.3\n"  # duplicate omega
    with pytest.raises(A2FParseError, match="strictly increasing"):
        read_a2f(_write(tmp_path, "dup.a2f", text), fmt="epw")


def test_shipped_pb_example_parses():
    import elphgap
    from pathlib import Path

    example = Path(elphgap.__file__).resolve().parents[2] / "examples" / "pb_like.a2f"
    if not example.exists():
        pytest.skip("examples/pb_like.a2f not present (sdist layout)")
    spec = read_a2f(str(example), fmt="epw")
    lam, wlog, _ = moments(spec.omega, spec.a2f)
    assert 1.0 < lam < 1.3  # Pb-like
    assert 3.0 < wlog < 8.0  # meV
