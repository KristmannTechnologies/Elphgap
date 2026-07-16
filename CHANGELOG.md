# Changelog

## v0.1.1 (unreleased)
- **CLI** (`elphgap` console script): `elphgap inspect <file>` reports the
  detected format, units, chosen column, frequency range, grid size, λ, ω_log,
  and the input SHA256; `elphgap tc <file>` solves the isotropic
  Migdal-Eliashberg Tc and prints every convention (μ*, Matsubara cutoff
  ω_c = cutoff_factor·ω_max, n_max, censored flag). `--json` on both for
  machine-readable output. Exit codes: 0 ok, 2 parse/format error, 3 Tc
  censored (below the resolvable floor), 4 invalid parameters. Isotropic path
  only — anisotropic solves stay in the Python API.
- **α²F parsers** (`elphgap.read_a2f`): Quantum ESPRESSO ph.x `a2F.dos`
  (frequencies in Rydberg → meV) and EPW `prefix.a2f` (meV, per-smearing
  columns) with header/footer autodetection. No silent clipping: ω≤0 rows are
  dropped and negative α²F values clamped to 0, both counted in a `warnings`
  list; the EPW smearing-column choice is surfaced as a warning and selectable
  with `--column`. `--clip-below` cuts low-ω numerical junk explicitly.
- **Docs** (`docs/`): quickstart with a synthetic Pb-like end-to-end example
  (`examples/pb_like.a2f`), scope/limitations, troubleshooting, and a
  GPL-3.0 commercial-use FAQ. README gains an Install/CLI section.
- **Packaging/CI**: version is now sourced from `elphgap.__version__`; GitHub
  Actions run the pytest suite on macOS, and a prepared (inactive)
  trusted-publishing workflow is included.
- No solver-numerics changes: the isotropic/anisotropic Eliashberg kernels,
  grids, and unit constants are unchanged from v0.1.0.

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
