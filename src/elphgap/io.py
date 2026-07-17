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
import re
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
_MONO_TOL = 1e-6  # relative tolerance for the cumulative-lambda monotonicity check


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
    a2f_by_column: dict[int, np.ndarray]  # a2F per primary column (selected: policy-cleaned; others: raw)
    negatives_by_column: dict[int, int]  # count of negative a2F values per primary column (never silent)
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


# Fixed QE/EPW header/footer line forms (EPW/src/supercond.f90, PHonon/PH/matdyn.f90).
# Regexes are anchored on the STRIPPED, lowercased line: internal whitespace in the
# fixed label text is strict (single space, exactly as the official WRITE literals),
# while numeric padding/spacing is lenient (\s+). SAMELINE records carry their value
# in group(1) (QE also captures Delta); NEXTLINE records take their values from the
# immediately following commented line.
_EPW_HEADER_RE = re.compile(r"^w\[mev\] a2f and integrated 2\*a2f/w for\s+(\d+) smearing values$")
_EPW_FOOTER = (
    ("integrated el-ph coupling", re.compile(r"^integrated el-ph coupling$"), "nextline"),
    ("phonon smearing", re.compile(r"^phonon smearing \(mev\)$"), "nextline"),
    ("electron smearing", re.compile(r"^electron smearing \(ev\)\s+(\S+)$"), "sameline"),
    ("fermi window", re.compile(r"^fermi window \(ev\)\s+(\S+)$"), "sameline"),
    # "DOS (eV)" (N(EF)): written by the EPW 6.0 release a2f writer, absent in
    # develop/5.3.1 -> OPTIONAL, but pinned to this canonical slot.
    ("dos", re.compile(r"^dos \(ev\)\s+(\S+)$"), "sameline"),
    ("summed el-ph coupling", re.compile(r"^summed el-ph coupling\s+(\S+)$"), "sameline"),
)
_EPW_FOOTER_ORDER = [rec[0] for rec in _EPW_FOOTER]
_EPW_FOOTER_OPTIONAL = {"dos"}
_EPW_FOOTER_MANDATORY = [r for r in _EPW_FOOTER_ORDER if r not in _EPW_FOOTER_OPTIONAL]
_QE_LAMBDA_RE = re.compile(r"^lambda\s*=\s*(\S+)\s+delta\s*=\s*(\S+)$")


def _match_footer(low: str) -> tuple[str, str, list[str]] | None:
    """(label, kind, captured groups) for a known footer LABEL line, else None."""
    for label, rx, kind in _EPW_FOOTER:
        m = rx.match(low)
        if m:
            return label, kind, list(m.groups())
    m = _QE_LAMBDA_RE.match(low)
    if m:
        return "lambda", "sameline", list(m.groups())
    return None


def _value_line_floats(commented: str) -> list[float] | None:
    """Floats on a commented value line '  #  0.05  0.10'; None if any token is not a float."""
    body = commented.lstrip()
    if body.startswith("#"):
        body = body[1:]
    body = body.strip()
    if not body:
        return None
    out: list[float] = []
    for tok in body.split():
        try:
            out.append(float(tok))
        except ValueError:
            return None
    return out


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
        for s in ("phonon smearing", "integrated el-ph coupling", "summed el-ph coupling", "fermi window", "smearing values", "2*a2f/w")
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
    require_column: bool = False,
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
    require_column : if True and no `column` is given for a multi-smearing EPW
          file, raise column_required (exit 4) BEFORE selecting/validating any
          default column (used by `tc`; `inspect` leaves it False and shows all).

    Raises A2FParseError (exit 2) for broken/malformed/non-official files and
    A2FColumnError (exit 4) for invalid format/column/clip choices.
    """
    if fmt.lower() not in ("auto", "qe", "epw"):
        raise A2FColumnError("unknown_format", f"unknown format {fmt!r}; use 'qe', 'epw', or 'auto'")
    if not np.isfinite(clip_below_mev):
        raise A2FColumnError("clip_not_finite", f"clip-below must be finite; got {clip_below_mev}")
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

    # Grammar: a header (comments; for EPW the single header line is MANDATORY),
    # then a CONTIGUOUS numeric data block, then a footer of KNOWN QE/EPW lines
    # (mandatory per format, in canonical order). Any
    # uncommented line that is neither numeric nor a recognized header/footer form
    # is malformed_row (exit 2), including at the block edges; numeric data after a
    # recognized footer (commented or not) is likewise malformed_row. Footer values
    # are captured by immediate label->value association, never lookahead-borrowed.
    comment_parts: list[str] = []
    rows: list[list[float]] = []
    state = "pre"  # "pre" (header) -> "data" -> "post" (footer)
    header_n_values: list[int] = []
    footer_values: dict[str, list[float]] = {}
    footer_order: list[str] = []
    pending_footer: str | None = None  # NEXTLINE label awaiting its commented value line
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        cut = len(raw_line)
        for marker in ("#", "!"):
            idx = raw_line.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        commented = raw_line[cut:] if cut < len(raw_line) else None
        data_part = raw_line[:cut].strip()

        # A pending NEXTLINE footer (Integrated / Phonon smearing) MUST be filled by
        # the immediately following line, and only by its commented value line — no
        # borrowing across comments or blanks.
        if pending_footer is not None:
            vals = _value_line_floats(commented) if commented is not None else None
            if data_part or vals is None:
                raise A2FParseError("footer_malformed", f"{path}:{lineno}: footer '{pending_footer}' value line missing or non-numeric.")
            footer_values[pending_footer] = vals
            pending_footer = None
            comment_parts.append(commented)
            continue

        if commented is not None:
            comment_parts.append(commented)
            # A footer marker (even commented) after data ends the block, so a stray
            # numeric row below it is rejected rather than silently appended.
            if state == "data" and _match_footer(commented.lstrip(" #").strip().lower()) is not None:
                state = "post"
        if not data_part:
            continue

        try:
            float(data_part.split()[0])
            is_numeric = True
        except ValueError:
            is_numeric = False

        if is_numeric:
            if state == "post":
                raise A2FParseError(
                    "malformed_row",
                    f"{path}:{lineno}: numeric data {data_part!r} after the footer began — corrupt or interleaved rows.",
                )
            try:
                rows.append([float(t) for t in data_part.split()])
            except ValueError as exc:
                raise A2FParseError(
                    "malformed_row", f"{path}:{lineno}: expected a numeric data row, got {data_part!r}"
                ) from exc
            state = "data"
            continue

        # Uncommented non-numeric line: the EPW header or a known footer label only.
        comment_parts.append(data_part)
        low = data_part.lower()
        m = _EPW_HEADER_RE.match(low)
        if m:
            if state != "pre":
                raise A2FParseError("malformed_row", f"{path}:{lineno}: EPW header line after data began: {data_part!r}")
            header_n_values.append(int(m.group(1)))
            continue
        fm = _match_footer(low)
        if fm is None:
            raise A2FParseError(
                "malformed_row",
                f"{path}:{lineno}: unrecognized non-numeric line {data_part!r} (not a known QE/EPW header/footer form).",
            )
        label, kind, groups = fm
        if state == "pre":
            raise A2FParseError("footer_before_data", f"{path}:{lineno}: footer line {data_part!r} before any data.")
        state = "post"
        footer_order.append(label)
        if kind == "nextline":
            pending_footer = label  # value on the immediately following commented line
        else:  # sameline: the captured value(s) are bound to THIS line and must be numbers
            bound: list[float] = []
            for g in groups:
                try:
                    bound.append(float(g))
                except ValueError:
                    raise A2FParseError("footer_malformed", f"{path}:{lineno}: footer '{label}' value {g!r} is not a number.")
            footer_values[label] = bound

    if pending_footer is not None:
        raise A2FParseError("footer_malformed", f"{path}: footer '{pending_footer}' has no value line before end of file.")
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
    if ncol < 2:
        raise A2FParseError("too_few_columns", f"{path}: data block has {ncol} column(s); need >= 2 (omega + a2F).")
    data = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise A2FParseError("non_finite", f"{path}: data block contains NaN or Inf; refusing to guess.")

    # Exactly one of each header/footer line (duplicates are contradictory).
    if len(header_n_values) > 1:
        raise A2FParseError("epw_duplicate_header", f"{path}: {len(header_n_values)} EPW header lines; expected at most 1.")
    if len(footer_order) != len(set(footer_order)):
        dup = next(x for x in footer_order if footer_order.count(x) > 1)
        raise A2FParseError("epw_duplicate_footer", f"{path}: footer label {dup!r} appears more than once.")
    # Every semantically-read footer value must be finite (reader-level; human == JSON).
    for label, vals in footer_values.items():
        if not np.all(np.isfinite(vals)):
            raise A2FParseError("non_finite_footer", f"{path}: footer '{label}' contains NaN/Inf.")

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
        # Mandatory records: the "frequencies in Rydberg" header and the lambda footer.
        if "frequencies in rydberg" not in comment_text:
            raise A2FParseError("qe_header_missing", f"{path}: QE a2F.dos is missing the 'frequencies in Rydberg' header.")
        if "lambda" not in footer_order:
            raise A2FParseError("qe_lambda_footer_missing", f"{path}: QE a2F.dos is missing the ' lambda = ... Delta = ...' footer (truncated run?).")
        lambda_footer = footer_values["lambda"][0]
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
        # Mandatory header: the ' w[meV] ... for N smearing values' line.
        # A truncated writer run (cut at a record boundary) fails here instead of
        # silently yielding a short spectrum and wrong Tc.
        if not header_n_values:
            raise A2FParseError("epw_header_missing", f"{path}: EPW prefix.a2f is missing the ' w[meV] ... smearing values' header (truncated run?).")
        # Mandatory footer: the canonical records in writer order, complete
        # ("dos" is optional -- EPW 6.0 release writes it, develop/5.3.1 do not).
        missing = [r for r in _EPW_FOOTER_MANDATORY if r not in footer_order]
        if missing:
            raise A2FParseError("epw_footer_incomplete", f"{path}: EPW footer missing record(s) {missing} (truncated run?).")
        if footer_order != [r for r in _EPW_FOOTER_ORDER if r in footer_order]:
            raise A2FParseError("epw_footer_order", f"{path}: EPW footer records out of canonical order: {footer_order}.")
        # Header-N vs data: N == (ncol-1)/2 normally. Known writer artifact: the
        # EPW 6.0 a2f_iso path stamps the template's smearing count (e.g. 10)
        # while writing a single smearing. Tolerate the mismatch ONLY when the
        # complete footer independently confirms the data's N (its per-smearing
        # records list exactly n_smearings values) -- truncation still dies at
        # the missing-footer check above.
        if header_n_values[0] != n_smearings:
            confirm = footer_values.get("phonon smearing")
            if confirm is not None and len(confirm) == n_smearings:
                warnings_pending_header = A2FWarning(
                    "epw_header_n_mismatch_tolerated",
                    f"header declares N={header_n_values[0]} but the data block has N={n_smearings} "
                    f"(1+2N columns), confirmed by the footer smearing count -- known EPW 6.0 "
                    f"a2f_iso header template artifact.",
                )
            else:
                raise A2FParseError(
                    "epw_header_n_mismatch",
                    f"{path}: header declares N={header_n_values[0]} smearings but the {ncol}-column block "
                    f"implies N={n_smearings} (1+2N).",
                )
        else:
            warnings_pending_header = None
        if warnings_pending_header is not None:
            warnings.append(warnings_pending_header)
        # Footer counts that encode N must also match.
        smearing_meV = footer_values.get("phonon smearing")
        for label in ("phonon smearing", "integrated el-ph coupling"):
            vals = footer_values.get(label)
            if vals is not None and len(vals) != n_smearings:
                raise A2FParseError(
                    "epw_smearing_count_mismatch",
                    f"{path}: footer '{label}' lists {len(vals)} value(s) but the {ncol}-column block "
                    f"implies N={n_smearings} smearings (1+2N).",
                )
        if detection == "forced" and "frequencies in rydberg" in comment_text:
            warnings.append(A2FWarning("format_forced_mismatch", "forced --format epw but the file declares Rydberg frequencies."))

    # Demand an explicit column for a smearing sweep BEFORE selecting/validating
    # any default column (so a missing --column is exit 4, ahead of any
    # column-specific data error such as a negative default column).
    if require_column and column is None and len(primary_a2f_columns) > 1:
        opts = ", ".join(
            f"col {c}" + (f" ({smearing_meV[c - 2]:g} meV)" if smearing_meV else "")
            for c in primary_a2f_columns
        )
        raise A2FColumnError(
            "column_required",
            f"this EPW file has {n_smearings} a2F smearing columns; choose one with --column N "
            f"(no canonical default). Options: {opts}. Run `inspect` to see each column's lambda.",
        )

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

    # a2F per primary column. The SELECTED column carries the negative policy
    # (reject by default; clamp only with clamp_negative). Non-selected columns
    # are kept RAW (never silently clamped); their negative count is reported.
    a2f_by_column: dict[int, np.ndarray] = {}
    negatives_by_column: dict[int, int] = {}
    clamped_selected = 0
    most_negative_selected = 0.0
    for c in sorted(set(primary_a2f_columns) | {col}):  # include a QE per-mode selection
        vals = data[keep, c - 1].copy()
        n_neg = int((vals < 0.0).sum())
        negatives_by_column[c] = n_neg
        if c == col and n_neg:
            most_negative_selected = float(vals.min())
            if not clamp_negative:
                raise A2FParseError(
                    "negative_a2f",
                    f"{path}: selected a2F column {c} has {n_neg} negative value(s) (min {vals.min():.3e}). "
                    f"Pass --clamp-negative to clamp to 0 (only appropriate for numerical noise).",
                )
            clamped_selected = n_neg
            vals = np.where(vals < 0.0, 0.0, vals)
            warnings.append(A2FWarning("clamped_negative", f"clamped {n_neg} negative a2F value(s) to 0 in column {c} (min {most_negative_selected:.3e})."))
        elif n_neg:
            warnings.append(A2FWarning("negative_a2f_other", f"a2F column {c} (not selected) has {n_neg} negative value(s), left unclamped."))
        a2f_by_column[c] = vals

    a2f = a2f_by_column[col]
    if not np.any(a2f > 0.0):
        raise A2FParseError("no_positive_a2f", f"{path}: selected a2F column {col} has no positive values.")

    # EPW: for EVERY smearing, compare 2*int(a2F/omega) (from RAW a2F, as the
    # file's cumulative lambda was built) against the file's cumulative-lambda
    # column, and check that column is non-decreasing.
    lambda_from_file: dict[int, float] = {}
    if fmt == "epw":
        for c in primary_a2f_columns:
            lam_col = data[keep, lambda_cols[c] - 1]
            lam_file = float(lam_col[-1])
            lambda_from_file[c] = lam_file
            dmin = float(np.min(np.diff(lam_col))) if lam_col.size > 1 else 0.0
            if dmin < -_MONO_TOL * max(1.0, abs(lam_file)):
                warnings.append(A2FWarning("epw_lambda_not_monotonic", f"cumulative-lambda column for a2F col {c} is not non-decreasing (min step {dmin:.3e}); the layout may be wrong."))
            lam_a2f = 2.0 * float(np.trapezoid(data[keep, c - 1] / omega, omega))
            if lam_file <= 1e-9:
                if lam_a2f > 1e-6:
                    warnings.append(A2FWarning("lambda_crosscheck", f"a2F col {c}: file cumulative lambda ~ 0 ({lam_file:.3e}) but 2*int(a2F/omega) = {lam_a2f:.3f} > 0; check the a2F/lambda mapping."))
            elif abs(lam_a2f - lam_file) / lam_file > _LAMBDA_CROSSCHECK_TOL:
                warnings.append(A2FWarning("lambda_crosscheck", f"a2F col {c}: 2*int(a2F/omega) = {lam_a2f:.3f} deviates >5% from the file's cumulative lambda = {lam_file:.3f}."))

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
        negatives_by_column=negatives_by_column,
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
