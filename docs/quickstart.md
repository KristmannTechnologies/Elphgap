# Quickstart

This walks through the `elphgap` command line on a small α²F file, end to end.
The CLI is **isotropic only**; anisotropic / band-resolved solves stay in the
Python API (see [limitations](limitations.md)).

## Install

elphgap has no PyPI package. Build a wheel from a clone and install it:

```bash
git clone https://github.com/KristmannTechnologies/Elphgap.git && cd Elphgap
python -m pip install build
python -m build                     # writes dist/elphgap-<version>-py3-none-any.whl + .tar.gz
pip install dist/elphgap-*.whl      # or: pip install -e .   (editable, from source)
```

The reference solvers and the CLI need only NumPy + SciPy. For GPU/JAX execution
see the [README](../README.md#install).

Check the console script is on your PATH:

```bash
elphgap --version        # -> elphgap 0.1.1
```

## A Pb-like example, end to end

The repo ships a **synthetic** Eliashberg spectral function,
[`examples/pb_like.a2f`](../examples/pb_like.a2f). It is a two-Gaussian toy —
*not* a real EPW/DFPT run — written in the EPW `prefix.a2f` column layout
(ω [meV] in column 1, α²F per phonon smearing in columns 2+) and tuned to a
Pb-like λ ≈ 1.16. Use it to exercise the tool; use your own DFPT/EPW output for
physics.

### 1. Inspect the spectrum

```bash
elphgap inspect examples/pb_like.a2f
```

```
elphgap 0.1.1 · inspect
  input       examples/pb_like.a2f
  sha256      8c2eb2f06e13985368a44c47491e4ced1102aa7b2175821bc5f1b67cc783464b
  format      epw (column 2)  [auto-detected via header; override with --format]
  units       meV -> meV
  column      2 of 4
  points      240  (1 dropped from 241 raw)
  omega       0.05 … 12 meV
  lambda      1.1600
  omega_log   5.106 meV  (59.25 K)
  omega_2     5.942 meV
  solver      elphgap 0.1.1
  warnings
    ! dropped 1 row(s) with omega <= 0 meV (non-physical; the a2F/omega moments are singular at omega = 0).
```

The dropped-row warning is expected: EPW files start at ω = 0, where the
`a2F/ω` moment integrand is singular, so that row is discarded (and reported —
elphgap never clips silently). The file has three α²F columns (a phonon-smearing
sweep); `inspect` uses column 2 by default. **Which smearing column you trust is
physics** — pick one explicitly with `--column N` after checking convergence.

### 2. Solve for Tc

```bash
elphgap tc examples/pb_like.a2f --format epw --mu-star 0.10
```

```
elphgap 0.1.1 · Tc (isotropic Migdal-Eliashberg)
  input       examples/pb_like.a2f
  sha256      8c2eb2f06e13985368a44c47491e4ced1102aa7b2175821bc5f1b67cc783464b
  format      epw (column 2)  [forced; override with --format]
  lambda      1.1600
  omega_log   5.106 meV
  Tc          5.903 K
  conventions
    mu*             0.100
    cutoff_factor   10
    omega_c         120 meV  (= 10 × omega_max)
    n_max           512
    censored        no
```

Always quote Tc as a **band over μ\***, not a single number. On this file:

| μ\* | Tc (isotropic ME) |
|-----|-------------------|
| 0.10 | 5.90 K |
| 0.13 | 5.46 K |
| 0.16 | 5.09 K |

Add `--json` to either command for machine-readable output (stable keys:
`elphgap_version`, `input.sha256`, `format`, `spectrum.lambda`,
`spectrum.omega_log_mev`, and for `tc` also `tc_kelvin`, `censored`,
`conventions`).

## Reference: the real EPW Pb tutorial

The synthetic file is tuned to resemble lead. For a *real* fcc-Pb calculation,
the EPW "FCC lead" superconductivity tutorial
([docs.epw-code.org](https://docs.epw-code.org)) reports, at μ\* = 0.1:
λ ≈ 1.158, ω_log ≈ 4.8 meV, McMillan Tc ≈ 4.37 K, Allen-Dynes-modified-McMillan
Tc ≈ 4.75 K, and a full isotropic-ME Tc of order 5 K (experimental Pb Tc =
7.2 K). Our synthetic λ = 1.16 / ME Tc ≈ 5.9 K at μ\* = 0.1 sits in the same
ballpark by construction — treat it as a smoke test, not a reproduction of the
tutorial. To reproduce the tutorial numbers, run EPW, then point
`elphgap tc your_prefix.a2f --format epw` at its output.

## The same thing from Python

```python
from elphgap import read_a2f, moments, tc_eliashberg

spec = read_a2f("examples/pb_like.a2f", fmt="epw", column=2)
for w in spec.warnings:
    print("warning:", w)

lam, wlog, w2 = moments(spec.omega, spec.a2f)
res = tc_eliashberg(spec.omega, spec.a2f, mu_star=0.10, cutoff_factor=10.0, n_max=512)
print(f"lambda={lam:.3f}  omega_log={wlog:.2f} meV  Tc={res.tc_kelvin:.2f} K "
      f"(censored={res.censored})")
```

## Next

- [limitations.md](limitations.md) — what the isotropic CLI does and does not model.
- [troubleshooting.md](troubleshooting.md) — format/units/censoring errors.
- [license-and-commercial-use.md](license-and-commercial-use.md) — GPL-3.0 in plain terms.
