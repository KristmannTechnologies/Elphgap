"""Command-line interface for elphgap (isotropic Migdal-Eliashberg path only).

Subcommands
-----------
    elphgap inspect <file>   parse an alpha^2F spectrum; report format, units,
                             every a2F smearing column and its lambda, cleaning,
                             moments, and the input SHA256.
    elphgap tc      <file>   solve the isotropic Migdal-Eliashberg Tc and print
                             the full convention manifest (mu*, Matsubara cutoff,
                             n_max, t_floor/t_max/rtol, censoring, units).

Both commands share ONE manifest (identical fields in the human and --json
views). Anisotropic / band-resolved solves are Python-API only
(elphgap.tc_aniso_linearized): the CLI does not accept full EPW ephmat input.

Exit codes
----------
    0  ok
    2  broken / non-official input (unparseable, wrong EPW column count, NaN/Inf,
       negative a2F without --clamp-negative, empty spectrum)
    3  Tc censored (fell below the resolvable floor set by --n-max)
    4  invalid user choice (format, column, missing --column for a smearing sweep,
       out-of-range parameter, usage error)
    5  solver error (bisection could not bracket Tc within t_max)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .allen_dynes import moments
from .eliashberg import tc_eliashberg
from .io import A2FColumnError, A2FError, A2FParseError, A2FSpectrum, read_a2f
from .units import MEV_TO_K

SCHEMA_VERSION = "1"

EXIT_OK = 0
EXIT_PARSE = 2
EXIT_CENSORED = 3
EXIT_PARAMS = 4
EXIT_SOLVER = 5

# Reported/used tc_eliashberg conventions (kept explicit so the manifest is authoritative).
N_MAX_DEFAULT = 4096
N_MAX_FAST = 512
T_MAX_KELVIN = 2000.0
RTOL = 1e-3


class ParamError(A2FError):
    """An out-of-range or missing CLI parameter (exit 4)."""

    exit_code = 4


class _SolverError(A2FError):
    """The Eliashberg bisection could not bracket Tc within t_max (exit 5)."""

    exit_code = 5


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit 4 (invalid user choice), not 2."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(EXIT_PARAMS, f"{self.prog}: error: {message}\n")


def _resolve_n_max(args: argparse.Namespace) -> int:
    if args.n_max is not None:
        if args.n_max < 4:
            raise ParamError("bad_n_max", f"--n-max must be >= 4; got {args.n_max}")
        return args.n_max
    return N_MAX_FAST if args.fast else N_MAX_DEFAULT


def _validate_tc_params(args: argparse.Namespace) -> None:
    if not np.isfinite(args.mu_star) or not (0.0 <= args.mu_star < 1.0):
        raise ParamError("bad_mu_star", f"--mu-star must be a finite value in [0, 1); got {args.mu_star}")
    if not np.isfinite(args.cutoff_factor) or args.cutoff_factor <= 0.0:
        raise ParamError("bad_cutoff", f"--cutoff-factor must be a finite value > 0; got {args.cutoff_factor}")


def _lambda(omega: np.ndarray, a2f: np.ndarray) -> float:
    # Direct lambda moment; tolerates the raw (possibly negative) non-selected
    # columns without the sqrt(<omega^2>) NaN that full moments() would hit.
    return 2.0 * float(np.trapezoid(a2f / omega, omega))


def _smearing_rows(spec: A2FSpectrum) -> list[dict]:
    """Per-a2F-column table (all EPW smearings, or the QE total): lambda by both routes."""
    rows = []
    for c in spec.primary_a2f_columns:
        smear = spec.smearing_meV[c - 2] if (spec.fmt == "epw" and spec.smearing_meV) else None
        rows.append(
            {
                "column": c,
                "smearing_meV": smear,
                "lambda_from_a2f": _lambda(spec.omega, spec.a2f_by_column[c]),
                "lambda_from_file": spec.lambda_from_file.get(c),
                "n_negative": spec.negatives_by_column.get(c, 0),
                "selected": c == spec.column,
            }
        )
    return rows


def _build_manifest(command: str, path: str, spec: A2FSpectrum, tc: dict | None = None) -> dict:
    lam, wlog, w2 = moments(spec.omega, spec.a2f)
    if not all(np.isfinite(v) for v in (lam, wlog, w2)):
        raise A2FParseError("non_finite_moments", "computed spectral moments are not finite; check the input a2F.")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "elphgap_version": __version__,
        "command": command,
        "input": {"path": path, "sha256": spec.sha256, "bytes": spec.n_bytes},
        "format": {
            "name": spec.fmt,
            "detected": spec.detected,
            "detection": spec.detection,
            "units_in": spec.units_in,
            "n_columns": spec.n_columns,
            "n_smearings": spec.n_smearings,
            "a2f_column": spec.column,
            "smearing_meV": (spec.smearing_meV[spec.column - 2] if (spec.fmt == "epw" and spec.smearing_meV) else None),
            "n_mode_columns": (spec.n_columns - 2 if spec.fmt == "qe" else 0),
        },
        "cleaning": {
            "clip_below_meV": spec.clip_threshold_mev,
            "dropped_below_clip": spec.dropped_below_clip,
            "clamped_negative": spec.clamped_negative,
            "most_negative_a2f": spec.most_negative_a2f,
        },
        "spectrum": {
            "n_points": int(spec.omega.size),
            "n_points_raw": spec.n_points_raw,
            "omega_units": "meV",
            "omega_min_meV": float(spec.omega[0]),
            "omega_max_meV": float(spec.omega[-1]),
            "lambda": lam,
            "lambda_from_file": spec.lambda_from_file.get(spec.column),
            "lambda_footer": spec.lambda_footer,
            "omega_log_meV": wlog,
            "omega_log_K": wlog * MEV_TO_K,
            "omega_2_meV": w2,
        },
        "smearings": _smearing_rows(spec),
        "warnings": [w.as_dict() for w in spec.warnings],
    }
    if command == "tc" and tc is not None:
        manifest["tc"] = tc["result"]
        manifest["conventions"] = tc["conventions"]
    return manifest


def _print_warnings(warnings: list[dict]) -> None:
    if warnings:
        print("  warnings")
        for w in warnings:
            print(f"    ! [{w['code']}] {w['message']}")


def _print_smearings(spec: A2FSpectrum, rows: list[dict]) -> None:
    if spec.fmt == "epw":
        print(f"  a2F smearing columns ({spec.n_smearings})")
        print("    col   smearing[meV]   lambda(a2F)   lambda(file)   neg")
        for r in rows:
            smear = f"{r['smearing_meV']:.4g}" if r["smearing_meV"] is not None else "  ?"
            lfile = f"{r['lambda_from_file']:.4f}" if r["lambda_from_file"] is not None else "   ?"
            neg = str(r["n_negative"]) if r["n_negative"] else "-"
            sel = "  <- selected" if r["selected"] else ""
            print(f"    {r['column']:<5} {smear:>10}   {r['lambda_from_a2f']:>10.4f}   {lfile:>10}   {neg:>3}{sel}")
    else:  # qe
        r = rows[0]
        lf = f"{spec.lambda_footer:.4f}" if spec.lambda_footer is not None else "n/a"
        neg = f"   negatives: {r['n_negative']}" if r["n_negative"] else ""
        print(f"  a2F column   {r['column']} (total)   lambda(a2F)={r['lambda_from_a2f']:.4f}   lambda(footer)={lf}{neg}")
        n_modes = spec.n_columns - 2
        if n_modes > 0:
            print(f"  per-mode columns: {n_modes} (partial a2F in columns 3..{spec.n_columns}; select with --column)")


def _print_common(args: argparse.Namespace, spec: A2FSpectrum, manifest: dict) -> None:
    """Shared human block — the same fields the JSON manifest carries."""
    fm, sp, cl = manifest["format"], manifest["spectrum"], manifest["cleaning"]
    tag = "forced" if spec.detection == "forced" else f"auto-detected via {spec.detection}"
    print(f"  input        {args.file}")
    print(f"  sha256       {spec.sha256}  ({spec.n_bytes} bytes)")
    print(f"  format       {spec.fmt} [{tag}]   input omega: {spec.units_in} -> meV   columns {spec.n_columns}   smearings {spec.n_smearings}")
    sm = f"  (smearing {fm['smearing_meV']:g} meV)" if fm["smearing_meV"] is not None else ""
    print(f"  a2F column   {spec.column}{sm}")
    clean = f"  cleaning     clip <= {cl['clip_below_meV']:g} meV   dropped {cl['dropped_below_clip']}   clamped {cl['clamped_negative']}"
    if cl["clamped_negative"]:
        clean += f"   most_negative {cl['most_negative_a2f']:.3e}"
    print(clean)
    print(f"  points       {sp['n_points']} of {sp['n_points_raw']} raw       omega {sp['omega_min_meV']:.4g} … {sp['omega_max_meV']:.4g} meV")
    print(f"  lambda       {sp['lambda']:.4f}       omega_log {sp['omega_log_meV']:.4g} meV ({sp['omega_log_K']:.4g} K)       omega_2 {sp['omega_2_meV']:.4g} meV")
    _print_smearings(spec, manifest["smearings"])


def _run_inspect(args: argparse.Namespace) -> int:
    spec = read_a2f(args.file, fmt=args.format, column=args.column,
                    clip_below_mev=args.clip_below, clamp_negative=args.clamp_negative)
    manifest = _build_manifest("inspect", args.file, spec)
    if args.json:
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return EXIT_OK
    print(f"elphgap {__version__} · inspect   (schema {SCHEMA_VERSION})")
    _print_common(args, spec, manifest)
    _print_warnings(manifest["warnings"])
    return EXIT_OK


def _run_tc(args: argparse.Namespace) -> int:
    _validate_tc_params(args)
    n_max = _resolve_n_max(args)
    # require_column=True: read_a2f raises column_required (exit 4) for a smearing
    # sweep BEFORE selecting/validating any default column or its data.
    spec = read_a2f(args.file, fmt=args.format, column=args.column,
                    clip_below_mev=args.clip_below, clamp_negative=args.clamp_negative,
                    require_column=True)

    omega_c = args.cutoff_factor * float(spec.omega[-1])
    t_floor_k = max(omega_c / (2.0 * np.pi * n_max) * MEV_TO_K, 1e-3)
    try:
        res = tc_eliashberg(spec.omega, spec.a2f, mu_star=args.mu_star, cutoff_factor=args.cutoff_factor,
                            n_max=n_max, t_max_kelvin=T_MAX_KELVIN, rtol=RTOL)
    except RuntimeError as exc:
        raise _SolverError("solver_no_bracket", str(exc)) from exc

    tc = {
        "result": {"tc_K": res.tc_kelvin, "censored": res.censored, "rho_at_floor": res.rho_at_floor},
        "conventions": {
            "mu_star": args.mu_star,
            "mu_star_convention": "constant mu*, applied at the Matsubara cutoff omega_c",
            "cutoff_factor": args.cutoff_factor,
            "omega_c_meV": omega_c,
            "n_max": n_max,
            "t_floor_K": t_floor_k,
            "t_max_K": T_MAX_KELVIN,
            "rtol": RTOL,
            "output_units": "K",
        },
    }
    manifest = _build_manifest("tc", args.file, spec, tc=tc)
    if args.json:
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return EXIT_CENSORED if res.censored else EXIT_OK

    cv = manifest["conventions"]
    print(f"elphgap {__version__} · Tc (isotropic Migdal-Eliashberg)   (schema {SCHEMA_VERSION})")
    _print_common(args, spec, manifest)
    if res.censored:
        rho = f"{res.rho_at_floor:.4g}" if res.rho_at_floor is not None else "n/a"
        print(f"  Tc           censored — below the resolvable floor (rho(T_floor) = {rho} < 1)")
    else:
        print(f"  Tc           {res.tc_kelvin:.4g} K")
    print("  conventions")
    print(f"    mu*             {cv['mu_star']:.3f}  ({cv['mu_star_convention']})")
    print(f"    cutoff_factor   {cv['cutoff_factor']:g}")
    print(f"    omega_c         {cv['omega_c_meV']:.4g} meV  (= {cv['cutoff_factor']:g} × omega_max)")
    print(f"    n_max           {cv['n_max']}")
    print(f"    t_floor         {cv['t_floor_K']:.4g} K       t_max {cv['t_max_K']:g} K       rtol {cv['rtol']:g}")
    print(f"    censored        {'yes' if res.censored else 'no'}       output units: K")
    _print_warnings(manifest["warnings"])
    if res.censored:
        print("  note        raise --n-max to lower the floor, or the coupling is too weak for a finite Tc at this mu*.", file=sys.stderr)
    return EXIT_CENSORED if res.censored else EXIT_OK


def _build_parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="elphgap", description="Isotropic Migdal-Eliashberg CLI for alpha^2F spectra.")
    parser.add_argument("--version", action="version", version=f"elphgap {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{inspect,tc}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("file", help="alpha^2F file (QE matdyn a2F.dos or EPW prefix.a2f)")
    common.add_argument("--format", choices=("auto", "qe", "epw"), default="auto",
                        help="input format (default: auto-detect from header/footer signatures)")
    common.add_argument("--column", type=int, default=None,
                        help="1-based a2F column (EPW smearing choice is physics; required for a sweep)")
    common.add_argument("--clip-below", type=float, default=0.0, metavar="MEV",
                        help="drop rows with omega <= MEV (default 0: drop only omega <= 0)")
    common.add_argument("--clamp-negative", action="store_true",
                        help="clamp negative a2F to 0 (default: reject). Only for numerical noise.")
    common.add_argument("--json", action="store_true", help="machine-readable JSON manifest")

    p_inspect = sub.add_parser("inspect", parents=[common], help="parse a spectrum; report all smearings and lambda")
    p_inspect.set_defaults(func=_run_inspect)

    p_tc = sub.add_parser("tc", parents=[common], help="solve the isotropic Migdal-Eliashberg Tc")
    p_tc.add_argument("--mu-star", type=float, default=0.10, help="Coulomb pseudopotential mu* (default 0.10)")
    p_tc.add_argument("--cutoff-factor", type=float, default=10.0, help="Matsubara cutoff omega_c = factor × omega_max (default 10)")
    p_tc.add_argument("--n-max", type=int, default=None, help=f"max Matsubara matrix size (default {N_MAX_DEFAULT})")
    p_tc.add_argument("--fast", action="store_true", help=f"shortcut for --n-max {N_MAX_FAST} (higher Tc floor, faster)")
    p_tc.set_defaults(func=_run_tc)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except A2FError as exc:
        print(f"elphgap: error[{exc.code}]: {exc}", file=sys.stderr)
        return exc.exit_code
    except ValueError as exc:
        # grids.validate_grid and friends: treat as broken input.
        print(f"elphgap: error[invalid_input]: {exc}", file=sys.stderr)
        return EXIT_PARSE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
