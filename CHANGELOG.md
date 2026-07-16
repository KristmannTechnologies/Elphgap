# Changelog

## v0.1.1 (unreleased)

### Added — command line and α²F I/O
- **`elphgap` console script.** `elphgap inspect <file>` reports the detected
  format, units, every a2F smearing column and its λ (from the spectrum and, for
  EPW, cross-checked against the file's cumulative-λ column), the frequency grid,
  cleaning counts, moments, and the input SHA256. `elphgap tc <file>` solves the
  isotropic Migdal-Eliashberg Tc and prints the full convention manifest (μ*,
  Matsubara cutoff ω_c = cutoff_factor·ω_max, n_max, t_floor/t_max/rtol,
  censoring, units). `--json` emits the same fields (schema_version 1). Exit
  codes: 0 ok, 2 broken input, 3 Tc censored, 4 invalid user choice, 5 solver
  error. Isotropic path only — anisotropic solves stay in the Python API.
- **α²F parsers** (`elphgap.read_a2f`, fail-closed):
  - EPW `prefix.a2f` is parsed to its real layout — column 1 = ω [meV], then
    `N` a2F columns, then `N` cumulative-λ(ω) columns (validated as `1+2N`).
    Cumulative-λ columns are rejected as a2F and used only for a λ cross-check
    (warns on >5% disagreement); a smearing sweep (N>1) requires an explicit
    `--column`.
  - QE `a2F.dos` (written by **matdyn.x**): ω [Ry]→meV, column 2 = total a2F,
    further columns = per-mode a2F.
  - Format detection uses only decisive header/footer signatures (no frequency-
    magnitude fallback); otherwise `--format` is required. NaN/Inf and negative
    a2F are rejected by default (`--clamp-negative` opts in), and the JSON is
    strict (`allow_nan=False`). `--clip-below` cuts a low-ω tail explicitly.
- **Packaged example.** `examples/pb_like.a2f` (synthetic, real EPW layout) ships
  as package data; `elphgap.example_a2f_path()` locates it after a pip install.
- **Docs** (`docs/`): quickstart, scope/limitations, troubleshooting, and a
  corrected GPL-3.0 usage FAQ. README gains a command-line section.
- **Packaging/CI.** Version sourced from `elphgap.__version__`; PEP 639 license
  metadata (`license = "GPL-3.0-or-later"`, `license-files`); Documentation/
  Issues/Changelog URLs. CI runs the full suite on Ubuntu+macOS × CPython
  3.11–3.14 plus a fresh wheel/sdist install (no JAX) that runs the advertised
  examples; a prepared, dispatch-only PyPI Trusted-Publishing workflow is
  included (SHA-pinned, tag==version guarded — it does not publish on merge).

### Solver numerics
- **No solver-numerics change in the v0.1.1 preparation itself.** The diff
  `main..v0.1.1-prep` touches only I/O, CLI, docs, packaging, and CI (plus the
  additive `__version__`). The isotropic/anisotropic Eliashberg kernels, grids,
  and unit constants are byte-identical to `main`.
- Note: relative to the **v0.1.0 tag**, a v0.1.1 release also ships the
  post-review solver hardening that already landed on `main` — see below.

## Between v0.1.0 and v0.1.1 — solver hardening on `main` (commit 0e40901)

These solver-numerics changes were made after review and are already on `main`;
they are part of what a v0.1.1 release ships relative to the v0.1.0 tag, and are
**not** part of the v0.1.1-prep I/O diff.
- NEW linearized anisotropic Tc (`tc_aniso_linearized` / `max_eigenvalue_aniso`):
  bisection on the leading eigenvalue of the folded linearized kernel; reduces to
  the isotropic solver exactly in the isotropic limit (recommended for quoting an
  anisotropic Tc).
- Fixed `tc_aniso` false positives/negatives near the resolvable floor
  (classification now uses the linearized eigenvalue, not a transient gap
  magnitude); exact closed-form normal-state Z (untruncated Matsubara tail) in
  both nonlinear anisotropic solvers.
- `t_max_kelvin` bracket clamping; ω-grid validation at all public entry points;
  integer-grid trapezoid-weight fix; asymmetric-pair-matrix guard.
- Tests expanded (23 → 36).

## v0.1.0 (2026-07-08)
- Initial release: isotropic + anisotropic Migdal-Eliashberg gap solvers
  (NumPy reference + JAX/GPU, batched isotropic path), Allen-Dynes/McMillan
  estimates, spectral moments, BETE-NET database loader.
- Self-contained validation (pytest): isotropic-limit consistency (~6 %,
  gap-collapse Tc vs. linearized Tc), NumPy↔JAX parity, MgB₂ two-band
  literature model (Golubov 2002).
- Optional external-data benchmarks (data not shipped): BETE-NET moment/Tc
  parity, EPW gap-output post-processing, GPU throughput.
- Self-contained anisotropic wall-clock benchmark
  (`benchmarks/aniso_speedup.py`): NumPy reference vs. JAX per-iteration
  cost on synthetic multi-band spectra.
- Known limitation: anisotropic Tc is extracted via a nonlinear
  gap-collapse threshold heuristic (can bias Tc high by a few percent at
  default settings); a linearized eigenvalue bisection is planned.
