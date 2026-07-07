# Changelog

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
