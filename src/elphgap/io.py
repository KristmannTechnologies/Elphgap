"""Loader for the BETE-NET training database (github.com/henniggroup/BETE-NET).

The file is a column-oriented pandas JSON dump: each column maps row-index
strings to values. Relevant columns: comp, comp_name, Freq_meV (alpha^2F
frequency grid), a2F, and the stored aggregates lambda, w_log, w_sq (meV).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    index: int
    comp: str
    comp_name: str
    omega: np.ndarray  # meV, ascending
    a2f: np.ndarray
    lambda_ref: float
    wlog_ref: float  # meV
    wsq_ref: float  # meV, sqrt(<omega^2>)


def load_database(path: str) -> list[Material]:
    with open(path) as f:
        db = json.load(f)
    materials = []
    for i in db["comp"]:
        omega = np.asarray(db["Freq_meV"][i], dtype=np.float64)
        a2f = np.asarray(db["a2F"][i], dtype=np.float64)
        # Guard against non-physical grid points (omega <= 0 would blow up 1/omega moments).
        mask = omega > 0
        materials.append(
            Material(
                index=int(i),
                comp=db["comp"][i],
                comp_name=db["comp_name"][i],
                omega=omega[mask],
                a2f=a2f[mask],
                lambda_ref=float(db["lambda"][i]),
                wlog_ref=float(db["w_log"][i]),
                wsq_ref=float(db["w_sq"][i]),
            )
        )
    materials.sort(key=lambda m: m.index)
    return materials


# --------------------------------------------------------------------------- #
# alpha^2F file readers (QE ph.x a2F.dos, EPW prefix.a2f)                      #
# --------------------------------------------------------------------------- #

RY_TO_MEV = 13605.693  # 1 Rydberg in meV; QE a2F.dos frequencies are in Ry.


class A2FParseError(ValueError):
    """An alpha^2F file could not be parsed into a usable (omega, a2F) grid.

    Subclasses ValueError so callers may catch parse failures and grid
    validation errors (from grids.validate_grid) with a single except clause.
    """


@dataclass
class A2FSpectrum:
    """A parsed isotropic alpha^2F(omega) spectrum plus provenance.

    omega and a2f are cleaned (omega > clip threshold, a2F clamped to >= 0) and
    ready to hand to moments()/tc_eliashberg(). Every deviation from the raw file
    (dropped rows, clamped values, auto-detected format, smearing-column choice)
    is recorded in `warnings` so nothing is silently altered.
    """

    omega: np.ndarray  # meV, strictly positive and increasing
    a2f: np.ndarray  # dimensionless, >= 0
    fmt: str  # "qe" or "epw"
    units_in: str  # native frequency unit of the file: "Ry" or "meV"
    column: int  # 1-based column index used for a2F
    n_columns: int  # number of numeric columns in the data block
    n_points_raw: int  # data rows in the file, before cleaning
    detected: bool  # True if fmt was auto-detected (not user-forced)
    detection: str  # how fmt was chosen: "header" / "footer" / "magnitude" / "forced"
    warnings: list[str]  # human-readable, never-silent notices


def _detect_format(comment_text: str, omega_max_raw: float) -> tuple[str, str]:
    """Guess ("qe"|"epw", how) from the comment text, else the frequency magnitude.

    QE ph.x a2F.dos files carry "frequencies in Rydberg" in the header; EPW
    prefix.a2f files quote omega in meV and end with "Integrated el-ph coupling"
    / "Phonon smearing" footers. If neither string is present, fall back to the
    raw magnitude of column 1: phonon energies are tens-to-hundreds of meV but a
    tiny fraction of a Rydberg, so omega_max < 1 implies Rydberg (QE).
    """
    if "rydberg" in comment_text or "[ry]" in comment_text or "(ry)" in comment_text:
        return "qe", "header"
    if "mev" in comment_text:
        return "epw", "header"
    if "phonon smearing" in comment_text or "el-ph coupling" in comment_text:
        return "epw", "footer"
    if omega_max_raw < 1.0:
        return "qe", "magnitude"
    return "epw", "magnitude"


def read_a2f(
    path: str,
    fmt: str = "auto",
    column: int | None = None,
    clip_below_mev: float | None = None,
) -> A2FSpectrum:
    """Read an isotropic alpha^2F(omega) spectrum from a QE or EPW output file.

    Supported formats
    -----------------
    qe  : Quantum ESPRESSO ph.x / lambda a2F.dos files. The comment header
          contains "frequencies in Rydberg"; column 1 is omega [Ry] and column 2
          is the total a2F (further columns are per-mode contributions).
          Frequencies are converted Ry -> meV via RY_TO_MEV = 13605.693.
    epw : EPW ``prefix.a2f`` files. Column 1 is omega [meV]; columns 2, 3, ...
          are a2F(omega) evaluated at successive *phonon smearing* values, and
          the file ends with ``# Integrated el-ph coupling`` and
          ``# Phonon smearing (meV)`` comment footers. Layout per the EPW
          tutorials (https://docs.epw-code.org, "FCC lead" / "Superconducting
          MgB2"): the smearing column is a PHYSICS choice, so with no explicit
          `column` this reader takes column 2 (first smearing) and WARNS; select
          another with `column=N` after checking convergence.

    Parameters
    ----------
    path : file to read.
    fmt : "auto" (default; detect from header/footer, else frequency magnitude),
          "qe", or "epw".
    column : 1-based a2F column. None -> column 2 for both formats.
    clip_below_mev : drop rows with omega <= this value [meV]. None (default)
          drops only omega <= 0 (non-physical: the a2F/omega moments are singular
          at omega = 0). Set e.g. 1.0 to cut low-omega numerical junk.

    Cleaning is never silent: dropped rows and a2F values clamped to >= 0 are
    counted in the returned `warnings`.

    Raises
    ------
    A2FParseError : unreadable file, non-numeric data row, ragged columns, a
          requested column beyond the data, or fewer than 2 usable points.
    """
    try:
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
    except OSError as exc:
        raise A2FParseError(f"cannot read {path!r}: {exc}") from exc
    text = raw_bytes.decode("utf-8", errors="replace")

    fmt = fmt.lower()
    if fmt not in ("auto", "qe", "epw"):
        raise A2FParseError(f"unknown format {fmt!r}; use 'qe', 'epw', or 'auto'")

    comment_parts: list[str] = []
    rows: list[list[float]] = []
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        cut = len(raw_line)
        for marker in ("#", "!"):
            i = raw_line.find(marker)
            if i != -1:
                cut = min(cut, i)
        if cut < len(raw_line):
            comment_parts.append(raw_line[cut:])
        data_part = raw_line[:cut].strip()
        if not data_part:
            continue
        tokens = data_part.split()
        try:
            float(tokens[0])
        except ValueError:
            # A leading non-numeric token (e.g. "lambda = 1.2") is a text
            # footer/label, not data: record it and move on.
            comment_parts.append(data_part)
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError as exc:
            raise A2FParseError(
                f"{path}:{lineno}: expected a numeric data row, got {data_part!r}"
            ) from exc

    if len(rows) < 2:
        raise A2FParseError(
            f"{path}: found {len(rows)} numeric data row(s); need at least 2. "
            "Expected a QE a2F.dos or EPW .a2f spectrum (two or more columns)."
        )

    ncol = len(rows[0])
    for k, row in enumerate(rows):
        if len(row) != ncol:
            raise A2FParseError(
                f"{path}: inconsistent column count (first data row has {ncol}, "
                f"data row {k + 1} has {len(row)})."
            )
    data = np.asarray(rows, dtype=np.float64)
    n_points_raw = data.shape[0]
    comment_text = "\n".join(comment_parts).lower()

    if fmt == "auto":
        detected = True
        fmt, detection = _detect_format(comment_text, float(np.max(data[:, 0])))
    else:
        detected = False
        detection = "forced"

    units_in = "Ry" if fmt == "qe" else "meV"
    col = 2 if column is None else int(column)
    if col < 2:
        raise A2FParseError(f"column must be >= 2 (column 1 is omega); got {col}")
    if col > ncol:
        raise A2FParseError(
            f"requested a2F column {col} but the file has only {ncol} column(s)."
        )

    omega = data[:, 0] * (RY_TO_MEV if fmt == "qe" else 1.0)
    a2f = data[:, col - 1].copy()

    warnings: list[str] = []
    if fmt == "epw" and column is None and ncol > 2:
        warnings.append(
            f"EPW file has {ncol - 1} a2F columns (phonon-smearing sweep); using "
            "column 2 (first smearing). The smearing choice is physics — set "
            "column=N explicitly after checking convergence."
        )

    threshold = 0.0 if clip_below_mev is None else float(clip_below_mev)
    keep = omega > threshold
    n_drop = int((~keep).sum())
    if n_drop:
        if clip_below_mev is None:
            warnings.append(
                f"dropped {n_drop} row(s) with omega <= 0 meV (non-physical; the "
                "a2F/omega moments are singular at omega = 0)."
            )
        else:
            warnings.append(
                f"dropped {n_drop} row(s) with omega <= {threshold:g} meV (clip_below_mev)."
            )
    omega = omega[keep]
    a2f = a2f[keep]

    neg = a2f < 0.0
    n_neg = int(neg.sum())
    if n_neg:
        warnings.append(
            f"clamped {n_neg} negative a2F value(s) to 0 (numerical noise or an "
            "unconverged double-delta integration)."
        )
        a2f[neg] = 0.0

    if omega.size < 2:
        raise A2FParseError(
            f"{path}: only {omega.size} point(s) left after removing omega <= "
            f"{threshold:g} meV; nothing to integrate."
        )
    if not np.all(np.diff(omega) > 0.0):
        raise A2FParseError(
            f"{path}: frequency column is not strictly increasing after cleaning "
            "(duplicate or unsorted omega); elphgap needs a monotone grid."
        )

    return A2FSpectrum(
        omega=omega,
        a2f=a2f,
        fmt=fmt,
        units_in=units_in,
        column=col,
        n_columns=ncol,
        n_points_raw=n_points_raw,
        detected=detected,
        detection=detection,
        warnings=warnings,
    )
