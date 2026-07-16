# Scope and limitations (CLI)

The `elphgap` command line is a deliberately narrow front end to the isotropic
solver. Read this before quoting any number it prints. (The Python API has the
full anisotropic machinery and its own caveats — see the
[README](../README.md#scope-and-limitations).)

## Isotropic only

`elphgap inspect` and `elphgap tc` read a single isotropic α²F(ω) column and
solve the **isotropic** linearized Migdal-Eliashberg equations
(`tc_eliashberg`). There is no CLI path for band/pocket-resolved αᵢⱼ²F or full
k-resolved EPW `ephmat` files. For an anisotropic Tc use the Python API
(`elphgap.tc_aniso_linearized`); the CLI will not silently "average" an
anisotropic input into an isotropic one.

## What the physics assumes

- **Harmonic phonons.** The α²F you supply is taken as given. Strongly
  anharmonic / soft-mode systems can shift λ substantially; SSCHA-class
  corrections live *upstream* of this solver, in how α²F was computed.
- **Migdal approximation** (ω_ph ≪ E_F). Not checked; suspect on
  flat-band / low-E_F systems.
- **Garbage in, garbage out.** elphgap does not compute the electron-phonon
  coupling. The physical quality of Tc is bounded by your DFPT/EPW convergence,
  the smearing column you pick, and μ*.

## Conventions the CLI pins (and prints)

- **μ\*** is an input, not a prediction. It enters at the Matsubara cutoff.
  Report Tc as a band over μ\* ∈ {0.10, 0.13, 0.16}, never a point value.
  Default: `--mu-star 0.10`.
- **Matsubara cutoff** ω_c = `cutoff-factor` · ω_max, with ω_max the largest
  frequency in your (cleaned) grid. Default `--cutoff-factor 10`. The matrix
  size N = ⌈ω_c / (2πT)⌉ is capped at `--n-max` (default 512). The infinite
  Matsubara sum in the mass-renormalization Z_n is kept in closed form; only
  the kernel matrix is truncated.
- **Smearing column.** EPW `prefix.a2f` files carry α²F at several phonon
  smearing values. The CLI defaults to column 2 (first smearing) and warns;
  choosing the converged column is your physics call (`--column N`).

## Censoring semantics (`tc`, exit code 3)

`--n-max` sets a *resolvable-Tc floor*: the temperature at which the capped
matrix still spans ω_c. If the leading Eliashberg eigenvalue is already below 1
at that floor, the true Tc (if any) lies below what this cutoff can resolve.
elphgap then reports `censored: yes`, prints `rho(T_floor)`, and **exits 3** —
it does **not** invent a small Tc. A censored result means one of:

1. the coupling is genuinely too weak for a finite Tc at this μ\*, or
2. `--n-max` is too small — raise it to lower the floor and re-run.

`censored` is never a silent 0 K: it is a distinct state in both the
human-readable and `--json` output.

## Numerics not modeled

- **Imaginary (Matsubara) axis only** — no real-axis continuation, so no
  spectral functions / tunneling DOS from the CLI.
- **No uncertainty propagation.** Different BLAS / SciPy / platform stacks can
  move the last digits; reproduce locally before citing.

## Not peer-reviewed

This is research software (v0.x); interfaces may change between 0.x releases.
Validate against the shipped tests and your own convergence studies, and please
open an issue with counter-examples.
