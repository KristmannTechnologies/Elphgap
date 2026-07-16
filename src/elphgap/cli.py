"""Command-line interface for elphgap (isotropic path only).

Subcommands
-----------
    elphgap inspect <file>   parse an alpha^2F spectrum; report format, units,
                             column, grid, lambda, omega_log, input SHA256.
    elphgap tc      <file>   solve the isotropic Migdal-Eliashberg Tc and print
                             every convention (mu*, Matsubara cutoff, n_max, ...).

Anisotropic / band-resolved solves are Python-API only
(elphgap.tc_aniso_linearized): the CLI deliberately does not accept full EPW
ephmat / k-resolved input. See docs/limitations.md.

Exit codes
----------
    0  ok
    2  parse / format error (unreadable file, bad columns, empty spectrum)
    3  Tc censored (fell below the resolvable floor set by --n-max)
    4  invalid parameters / usage error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from . import __version__
from .allen_dynes import moments
from .eliashberg import tc_eliashberg
from .io import A2FParseError, A2FSpectrum, read_a2f
from .units import MEV_TO_K

EXIT_OK = 0
EXIT_PARSE = 2
EXIT_CENSORED = 3
EXIT_PARAMS = 4


class ParamError(ValueError):
    """An out-of-range CLI parameter (maps to exit code 4)."""


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit 4 (invalid parameters), not 2."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(EXIT_PARAMS, f"{self.prog}: error: {message}\n")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_params(args: argparse.Namespace) -> None:
    """Reject on-their-face-invalid parameters before touching the file (exit 4)."""
    if args.column is not None and args.column < 2:
        raise ParamError(
            f"--column must be >= 2 (column 1 is omega); got {args.column}"
        )
    if getattr(args, "mu_star", None) is not None and not (0.0 <= args.mu_star < 1.0):
        raise ParamError(f"--mu-star must be in [0, 1); got {args.mu_star}")
    if getattr(args, "cutoff_factor", None) is not None and args.cutoff_factor <= 0.0:
        raise ParamError(f"--cutoff-factor must be > 0; got {args.cutoff_factor}")
    if getattr(args, "n_max", None) is not None and args.n_max < 4:
        raise ParamError(f"--n-max must be >= 4; got {args.n_max}")


def _format_line(spec: A2FSpectrum) -> str:
    tag = "forced" if spec.detection == "forced" else f"auto-detected via {spec.detection}"
    return f"{spec.fmt} (column {spec.column})  [{tag}; override with --format]"


def _base_report(command: str, path: str, spec: A2FSpectrum) -> dict:
    lam, wlog_mev, w2_mev = moments(spec.omega, spec.a2f)
    return {
        "elphgap_version": __version__,
        "command": command,
        "input": {"path": path, "sha256": _sha256(path)},
        "format": {
            "name": spec.fmt,
            "detected": spec.detected,
            "detection": spec.detection,
            "units_in": spec.units_in,
            "column": spec.column,
            "n_columns": spec.n_columns,
        },
        "spectrum": {
            "n_points": int(spec.omega.size),
            "n_points_raw": spec.n_points_raw,
            "omega_min_mev": float(spec.omega[0]),
            "omega_max_mev": float(spec.omega[-1]),
            "lambda": lam,
            "omega_log_mev": wlog_mev,
            "omega_log_kelvin": wlog_mev * MEV_TO_K,
            "omega_2_mev": w2_mev,
        },
        "warnings": list(spec.warnings),
    }


def _print_warnings(warnings: list[str]) -> None:
    if warnings:
        print("  warnings")
        for w in warnings:
            print(f"    ! {w}")


def _run_inspect(args: argparse.Namespace) -> int:
    spec = read_a2f(args.file, fmt=args.format, column=args.column, clip_below_mev=args.clip_below)
    report = _base_report("inspect", args.file, spec)
    if args.json:
        print(json.dumps(report, indent=2))
        return EXIT_OK

    s = report["spectrum"]
    dropped = spec.n_points_raw - s["n_points"]
    print(f"elphgap {__version__} · inspect")
    print(f"  input       {args.file}")
    print(f"  sha256      {report['input']['sha256']}")
    print(f"  format      {_format_line(spec)}")
    print(f"  units       {spec.units_in} -> meV")
    print(f"  column      {spec.column} of {spec.n_columns}")
    pts = f"  points      {s['n_points']}"
    if dropped:
        pts += f"  ({dropped} dropped from {spec.n_points_raw} raw)"
    print(pts)
    print(f"  omega       {s['omega_min_mev']:.4g} … {s['omega_max_mev']:.4g} meV")
    print(f"  lambda      {s['lambda']:.4f}")
    print(f"  omega_log   {s['omega_log_mev']:.4g} meV  ({s['omega_log_kelvin']:.4g} K)")
    print(f"  omega_2     {s['omega_2_mev']:.4g} meV")
    print(f"  solver      elphgap {__version__}")
    _print_warnings(report["warnings"])
    return EXIT_OK


def _run_tc(args: argparse.Namespace) -> int:
    spec = read_a2f(args.file, fmt=args.format, column=args.column, clip_below_mev=args.clip_below)
    report = _base_report("tc", args.file, spec)

    omega_c = args.cutoff_factor * float(spec.omega[-1])
    res = tc_eliashberg(
        spec.omega,
        spec.a2f,
        mu_star=args.mu_star,
        cutoff_factor=args.cutoff_factor,
        n_max=args.n_max,
    )
    report["tc_kelvin"] = res.tc_kelvin
    report["censored"] = res.censored
    report["conventions"] = {
        "mu_star": args.mu_star,
        "cutoff_factor": args.cutoff_factor,
        "omega_c_mev": omega_c,
        "n_max": args.n_max,
    }
    if res.rho_at_floor is not None:
        report["rho_at_floor"] = res.rho_at_floor

    if args.json:
        print(json.dumps(report, indent=2))
        return EXIT_CENSORED if res.censored else EXIT_OK

    s = report["spectrum"]
    print(f"elphgap {__version__} · Tc (isotropic Migdal-Eliashberg)")
    print(f"  input       {args.file}")
    print(f"  sha256      {report['input']['sha256']}")
    print(f"  format      {_format_line(spec)}")
    print(f"  lambda      {s['lambda']:.4f}")
    print(f"  omega_log   {s['omega_log_mev']:.4g} meV")
    if res.censored:
        rho = f"{res.rho_at_floor:.4g}" if res.rho_at_floor is not None else "n/a"
        print(f"  Tc          censored — below the resolvable floor (rho(T_floor) = {rho} < 1)")
    else:
        print(f"  Tc          {res.tc_kelvin:.4g} K")
    print("  conventions")
    print(f"    mu*             {args.mu_star:.3f}")
    print(f"    cutoff_factor   {args.cutoff_factor:g}")
    print(f"    omega_c         {omega_c:.4g} meV  (= {args.cutoff_factor:g} × omega_max)")
    print(f"    n_max           {args.n_max}")
    print(f"    censored        {'yes' if res.censored else 'no'}")
    _print_warnings(report["warnings"])
    if res.censored:
        print(
            "  note        raise --n-max to lower the resolvable-Tc floor, or the "
            "coupling is simply too weak for a finite Tc at this mu*.",
            file=sys.stderr,
        )
    return EXIT_CENSORED if res.censored else EXIT_OK


def _build_parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="elphgap", description="Isotropic Migdal-Eliashberg CLI for alpha^2F spectra.")
    parser.add_argument("--version", action="version", version=f"elphgap {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{inspect,tc}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("file", help="alpha^2F file (QE a2F.dos or EPW .a2f)")
    common.add_argument(
        "--format", choices=("auto", "qe", "epw"), default="auto",
        help="input format (default: auto-detect from header/footer, else magnitude)",
    )
    common.add_argument(
        "--column", type=int, default=None,
        help="1-based a2F column (default: 2). EPW smearing choice is physics.",
    )
    common.add_argument(
        "--clip-below", type=float, default=None, metavar="MEV",
        help="drop rows with omega <= MEV (default: drop only omega <= 0)",
    )
    common.add_argument("--json", action="store_true", help="machine-readable JSON output")

    p_inspect = sub.add_parser(
        "inspect", parents=[common], help="parse a spectrum and report lambda, omega_log, ...",
    )
    p_inspect.set_defaults(func=_run_inspect)

    p_tc = sub.add_parser(
        "tc", parents=[common], help="solve the isotropic Migdal-Eliashberg Tc",
    )
    p_tc.add_argument("--mu-star", type=float, default=0.10, help="Coulomb pseudopotential mu* (default: 0.10)")
    p_tc.add_argument(
        "--cutoff-factor", type=float, default=10.0,
        help="Matsubara cutoff omega_c = cutoff_factor × omega_max (default: 10)",
    )
    p_tc.add_argument("--n-max", type=int, default=512, help="max Matsubara matrix size (default: 512)")
    p_tc.set_defaults(func=_run_tc)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_params(args)
    except ParamError as exc:
        print(f"elphgap: error: {exc}", file=sys.stderr)
        return EXIT_PARAMS
    try:
        return args.func(args)
    except A2FParseError as exc:
        print(f"elphgap: parse error: {exc}", file=sys.stderr)
        return EXIT_PARSE
    except ValueError as exc:
        # grids.validate_grid and friends raise ValueError on a bad spectrum.
        print(f"elphgap: error: {exc}", file=sys.stderr)
        return EXIT_PARSE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
