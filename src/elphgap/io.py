"""I/O helpers: the BETE-NET training database loader and alpha^2F file readers.

The BETE-NET file (github.com/henniggroup/BETE-NET) is a column-oriented pandas
JSON dump: each column maps row-index strings to values. Relevant columns: comp,
comp_name, Freq_meV (alpha^2F frequency grid), a2F, and the stored aggregates
lambda, w_log, w_sq (meV).

`read_a2f` parses an isotropic alpha^2F spectrum from a Quantum ESPRESSO
matdyn.x `a2F.dos` file or an EPW `prefix.a2f` file. Both readers are
fail-closed: an unrecognized layout, non-finite value, or ambiguous format is a
hard error, never a silent reinterpretation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .allen_dynes import moments


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
# alpha^2F file readers (QE matdyn.x a2F.dos, EPW prefix.a2f)                  #
# --------------------------------------------------------------------------- #
#
# EPW prefix.a2f (EPW/src/supercond.f90, subroutine evaluate_a2f_lambda):
#   WRITE(f12.7, 20f12.7)  wsph*1000, a2f(iwph, :), l_a2f(iwph, :)
# => column 1 = omega [meV], then N columns of a2F (one per phonon smearing),
#    then N columns of CUMULATIVE lambda(omega). Total columns = 1 + 2N. The
#    lambda columns are NOT a2F and must never be fed to the solver. Footer:
#    "Integrated el-ph coupling" / "Phonon smearing (meV)" / "Electron smearing
#    (eV)" / "Fermi window (eV)" / "Summed el-ph coupling".
#
# QE a2F.dos (PHonon/PH/matdyn.f90):
#   header "#  frequencies in Rydberg"; WRITE  E, dos_tot, dos_a2F(1:nmodes)
# => column 1 = omega [Ry], column 2 = TOTAL a2F, columns 3.. = per-mode a2F.
#    No cumulative-lambda columns. Footer " lambda = ...".
#
# Sources: https://docs.epw-code.org/tutorials/tutorial_04/index.html,
# https://gitlab.com/QEF/q-e (EPW/src/supercond.f90, PHonon/PH/matdyn.f90).

RY_TO_MEV = 13605.693  # 1 Rydberg in meV; QE a2F.dos frequencies are in Ry.
_LAMBDA_CROSSCHECK_TOL = 0.05  # warn if 2*int(a2F/w) differs >5% from the file's cumulative lambda


@dataclass(frozen=True)
class A2FWarning:
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


class A2FError(ValueError):
    """Base for alpha^2F reader failures; carries a stable `code` and `exit_code`."""

    exit_code = 2

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class A2FParseError(A2FError):
    """The file is broken or does not match the declared/official layout (exit 2)."""

    exit_code = 2


class A2FColumnError(A2FError):
    """An invalid user choice: format, column, or clip parameter (exit 4)."""

    exit_code = 4


@dataclass
class A2FSpectrum:
    """A parsed isotropic alpha^2F(omega) spectrum plus full provenance.

    `omega`/`a2f` are the cleaned SELECTED column, ready for moments()/tc. Every
    deviation from the raw bytes (format detection, column choice, dropped rows,
    clamped values, lambda cross-check) is recorded structurally so nothing is
    silently reinterpreted.
    """

    omega: np.ndarray  # meV, strictly positive and increasing
    a2f: np.ndarray  # dimensionless, >= 0 (selected column)
    fmt: str  # "qe" or "epw"
    units_in: str  # native frequency unit of the file: "Ry" or "meV"
    sha256: str  # of the raw input bytes (single read)
    n_bytes: int
    column: int  # 1-based column selected as a2F
    n_columns: int  # numeric columns in the data block
    column_kinds: list[str]  # per-column role: omega/a2f/a2f_total/a2f_mode/lambda_cumulative
    n_smearings: int  # EPW: N; QE: 1 (the total)
    primary_a2f_columns: list[int]  # 1-based columns selectable as a smearing's a2F
    smearing_meV: list[float] | None  # EPW footer smearing values (len N) or None
    a2f_by_column: dict[int, np.ndarray]  # cleaned a2F for each primary column (inspect table)
    lambda_from_file: dict[int, float]  # integrated lambda per primary column, from cumulative-lambda col
    lambda_footer: float | None  # QE footer "lambda =" value, if present
    n_points_raw: int
    dropped_below_clip: int
    clip_threshold_mev: float
    clamped_negative: int  # in the SELECTED column
    most_negative_a2f: float  # most negative raw a2F seen in the selected column (0.0 if none)
    detected: bool  # True if fmt was auto-detected (not user-forced)
    detection: str  # "header" / "footer" / "forced"
    warnings: list[A2FWarning] = field(default_factory=list)

    @property
    def requires_column_choice(self) -> bool:
        """True when more than one smearing a2F column exists (choice is physics)."""
        return len(self.primary_a2f_columns) > 1


def _parse_labeled_floats(comment_parts: list[str], label: str) -> list[float] | None:
    """Return the floats associated with the first comment line containing `label`.

    Handles both footer styles: QE writes " lambda = 0.9" (value on the label
    line), EPW writes "Phonon smearing (meV)" then "  #  0.05  0.10 ..." (values
    on the next line). Non-float tokens (labels, units, '#', '=') are ignored.
    Returns None if no floats are found near the label.
    """
    low = label.lower()
    for i, part in enumerate(comment_parts):
        if low in part.lower():
            for cand in comment_parts[i : i + 3]:
                vals: list[float] = []
                for t in cand.replace("#", " ").replace("=", " ").split():
                    try:
                        vals.append(float(t))
                    except ValueError:
                        pass  # ignore label words and unit tags
                if vals:
                    return vals
    return None


def _detect_format(comment_text: str, fmt: str) -> tuple[str, bool, str]:
    """Resolve (fmt, detected, how) from an explicit choice or unambiguous signatures.

    Auto-detection uses ONLY decisive header/footer strings — there is no
    magnitude fallback (it would rescale a legitimate low-energy meV spectrum by
    13605x). If neither signature is present and fmt is "auto", the caller must
    pass --format.
    """
    qe_sig = "frequencies in rydberg" in comment_text or "in rydberg" in comment_text
    epw_sig = any(
        s in comment_text
        for s in ("phonon smearing", "integrated el-ph coupling", "summed el-ph coupling", "fermi window")
    )
    if fmt == "auto":
        if qe_sig and not epw_sig:
            return "qe", True, "header"
        if epw_sig and not qe_sig:
            return "epw", True, "footer"
        raise A2FColumnError(
            "format_undetectable",
            "cannot auto-detect the format (no decisive QE 'frequencies in Rydberg' "
            "header or EPW footer signature). Pass --format qe|epw explicitly.",
        )
    return fmt, False, "forced"


def read_a2f(
    path: str,
    fmt: str = "auto",
    column: int | None = None,
    clip_below_mev: float = 0.0,
    clamp_negative: bool = False,
) -> A2FSpectrum:
    """Read an isotropic alpha^2F(omega) spectrum from a QE or EPW output file.

    Formats
    -------
    qe  : Quantum ESPRESSO matdyn.x `a2F.dos`. Header "frequencies in Rydberg";
          column 1 = omega [Ry] (-> meV via RY_TO_MEV), column 2 = total a2F,
          columns 3.. = per-mode a2F.
    epw : EPW `prefix.a2f`. Column 1 = omega [meV], columns 2..N+1 = a2F at each
          of N phonon smearings, columns N+2..2N+1 = CUMULATIVE lambda(omega).
          The layout is validated to be exactly 1+2N columns; lambda columns are
          rejected as a2F and used only as a cross-check.

    Parameters
    ----------
    fmt : "auto" (detect from header/footer signatures; no magnitude fallback),
          "qe", or "epw".
    column : 1-based a2F column. None -> first a2F column (2). For EPW with N>1
          smearings the caller should require an explicit choice.
    clip_below_mev : drop rows with omega <= this [meV]. Must be >= 0. Default 0
          drops only omega <= 0 (the a2F/omega moments are singular there).
    clamp_negative : if False (default), any negative a2F in the selected column
          is a hard error; if True, negatives are clamped to 0 and counted.

    Raises A2FParseError (exit 2) for broken/malformed/non-official files and
    A2FColumnError (exit 4) for invalid format/column/clip choices.
    """
    if fmt.lower() not in ("auto", "qe", "epw"):
        raise A2FColumnError("unknown_format", f"unknown format {fmt!r}; use 'qe', 'epw', or 'auto'")
    if clip_below_mev < 0.0:
        raise A2FColumnError("clip_negative", f"clip-below must be >= 0 meV; got {clip_below_mev}")

    try:
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
    except OSError as exc:
        raise A2FParseError("unreadable", f"cannot read {path!r}: {exc}") from exc
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    n_bytes = len(raw_bytes)
    text = raw_bytes.decode("utf-8", errors="replace")

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
            comment_parts.append(data_part)  # text footer/label, e.g. "lambda = 1.0"
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError as exc:
            raise A2FParseError(
                "malformed_row", f"{path}:{lineno}: expected a numeric data row, got {data_part!r}"
            ) from exc

    if len(rows) < 2:
        raise A2FParseError(
            "no_data",
            f"{path}: found {len(rows)} numeric data row(s); need >= 2 (a QE a2F.dos or EPW .a2f block).",
        )
    ncol = len(rows[0])
    for k, row in enumerate(rows):
        if len(row) != ncol:
            raise A2FParseError(
                "ragged_columns",
                f"{path}: inconsistent column count (row 1 has {ncol}, data row {k + 1} has {len(row)}).",
            )
    data = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise A2FParseError("non_finite", f"{path}: data block contains NaN or Inf; refusing to guess.")
    comment_text = "\n".join(comment_parts).lower()

    fmt, detected, detection = _detect_format(comment_text, fmt.lower())
    warnings: list[A2FWarning] = []

    # Column roles per format.
    lambda_footer: float | None = None
    smearing_meV: list[float] | None = None
    lambda_cols: dict[int, int] = {}
    if fmt == "qe":
        units_in, omega_scale = "Ry", RY_TO_MEV
        column_kinds = ["omega", "a2f_total"] + ["a2f_mode"] * (ncol - 2)
        primary_a2f_columns = [2]
        n_smearings = 1
        lam_vals = _parse_labeled_floats(comment_parts, "lambda =")
        if lam_vals:
            lambda_footer = lam_vals[0]
        if detection == "forced" and any(s in comment_text for s in ("phonon smearing", "el-ph coupling")):
            warnings.append(A2FWarning("format_forced_mismatch", "forced --format qe but the file carries EPW footer markers."))
    else:  # epw
        units_in, omega_scale = "meV", 1.0
        if ncol < 3 or ncol % 2 == 0:
            raise A2FParseError(
                "epw_column_count",
                f"{path}: an EPW prefix.a2f has 1+2N columns (omega, N a2F, N cumulative lambda); "
                f"got {ncol}. Columns N+2..2N+1 are lambda(omega), not a2F.",
            )
        n_smearings = (ncol - 1) // 2
        column_kinds = ["omega"] + ["a2f"] * n_smearings + ["lambda_cumulative"] * n_smearings
        primary_a2f_columns = list(range(2, n_smearings + 2))
        lambda_cols = {c: n_smearings + c for c in primary_a2f_columns}  # a2F col -> its cumulative-lambda col
        smearing_meV = _parse_labeled_floats(comment_parts, "phonon smearing (mev)")
        if smearing_meV is not None and len(smearing_meV) != n_smearings:
            warnings.append(
                A2FWarning("smearing_count_mismatch", f"footer lists {len(smearing_meV)} smearing values but the block implies N={n_smearings}.")
            )
            smearing_meV = None
        if detection == "forced" and "frequencies in rydberg" in comment_text:
            warnings.append(A2FWarning("format_forced_mismatch", "forced --format epw but the file declares Rydberg frequencies."))

    # Column selection.
    if column is None:
        col = primary_a2f_columns[0]
    else:
        col = int(column)
        if col < 1 or col > ncol:
            raise A2FColumnError("column_out_of_range", f"--column {col} out of range 1..{ncol}.")
        kind = column_kinds[col - 1]
        if kind == "omega":
            raise A2FColumnError("column_is_omega", f"--column {col} is the omega axis, not a2F.")
        if kind == "lambda_cumulative":
            raise A2FColumnError(
                "column_is_lambda",
                f"--column {col} is a CUMULATIVE lambda(omega) column, not a2F. a2F columns are {primary_a2f_columns}.",
            )
        if kind == "a2f_mode":
            warnings.append(A2FWarning("per_mode_column", f"--column {col} is a per-mode PARTIAL a2F (QE), not the total in column 2."))

    if fmt == "epw" and len(primary_a2f_columns) > 1 and column is not None and col in primary_a2f_columns:
        s_txt = f" (smearing {smearing_meV[col - 2]:g} meV)" if smearing_meV else ""
        warnings.append(A2FWarning("smearing_column", f"using EPW smearing column {col}{s_txt} of {n_smearings}."))

    # Clean the omega grid.
    omega_raw = data[:, 0] * omega_scale
    keep = omega_raw > clip_below_mev
    dropped = int((~keep).sum())
    if dropped:
        if clip_below_mev == 0.0:
            warnings.append(A2FWarning("dropped_nonpositive", f"dropped {dropped} row(s) with omega <= 0 meV (singular moments)."))
        else:
            warnings.append(A2FWarning("dropped_below_clip", f"dropped {dropped} row(s) with omega <= {clip_below_mev:g} meV (--clip-below)."))
    omega = omega_raw[keep]
    if omega.size < 2:
        raise A2FParseError("too_few_points", f"{path}: only {omega.size} point(s) left after clip <= {clip_below_mev:g} meV.")
    if not np.all(np.diff(omega) > 0.0):
        raise A2FParseError("not_increasing", f"{path}: omega is not strictly increasing after cleaning (duplicate/unsorted).")

    # Clean a2F for every primary column (inspect table); apply the negative
    # policy strictly to the SELECTED column (the one that reaches the solver).
    a2f_by_column: dict[int, np.ndarray] = {}
    clamped_selected = 0
    most_negative_selected = 0.0
    for c in sorted(set(primary_a2f_columns) | {col}):  # include a QE per-mode selection
        vals = data[keep, c - 1].copy()
        n_neg = int((vals < 0.0).sum())
        if c == col and n_neg:
            most_negative_selected = float(vals.min())
            if not clamp_negative:
                raise A2FParseError(
                    "negative_a2f",
                    f"{path}: selected a2F column {c} has {n_neg} negative value(s) (min {vals.min():.3e}). "
                    f"Pass --clamp-negative to clamp to 0 (only appropriate for numerical noise).",
                )
            clamped_selected = n_neg
            warnings.append(A2FWarning("clamped_negative", f"clamped {n_neg} negative a2F value(s) to 0 in column {c} (min {vals.min():.3e})."))
        vals[vals < 0.0] = 0.0  # non-selected columns clamp for display only
        a2f_by_column[c] = vals

    a2f = a2f_by_column[col]
    if not np.any(a2f > 0.0):
        raise A2FParseError("no_positive_a2f", f"{path}: selected a2F column {col} has no positive values.")

    # lambda cross-check against the file's cumulative-lambda column (EPW only).
    lambda_from_file: dict[int, float] = {}
    if fmt == "epw":
        for c in primary_a2f_columns:
            lambda_from_file[c] = float(data[keep, lambda_cols[c] - 1][-1])
        lam_a2f = 2.0 * float(np.trapezoid(a2f / omega, omega))
        lam_file = lambda_from_file[col]
        if lam_file > 1e-9 and abs(lam_a2f - lam_file) / lam_file > _LAMBDA_CROSSCHECK_TOL:
            warnings.append(
                A2FWarning(
                    "lambda_crosscheck",
                    f"2*int(a2F/omega) = {lam_a2f:.3f} deviates >5% from the file's cumulative "
                    f"lambda = {lam_file:.3f} (column {col}); check the a2F/lambda column mapping.",
                )
            )

    return A2FSpectrum(
        omega=omega,
        a2f=a2f,
        fmt=fmt,
        units_in=units_in,
        sha256=sha256,
        n_bytes=n_bytes,
        column=col,
        n_columns=ncol,
        column_kinds=column_kinds,
        n_smearings=n_smearings,
        primary_a2f_columns=primary_a2f_columns,
        smearing_meV=smearing_meV,
        a2f_by_column=a2f_by_column,
        lambda_from_file=lambda_from_file,
        lambda_footer=lambda_footer,
        n_points_raw=int(data.shape[0]),
        dropped_below_clip=dropped,
        clip_threshold_mev=float(clip_below_mev),
        clamped_negative=clamped_selected,
        most_negative_a2f=most_negative_selected,
        detected=detected,
        detection=detection,
        warnings=warnings,
    )


def lambda_of(omega: np.ndarray, a2f: np.ndarray) -> float:
    """lambda = 2 * integral a2F/omega domega for a cleaned (omega, a2F) pair."""
    return float(moments(omega, a2f)[0])
