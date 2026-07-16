# Quickstart

This walks through the `elphgap` command line on a small α²F file, end to end.
The CLI is **isotropic only**; anisotropic / band-resolved solves stay in the
Python API (see [limitations](limitations.md)).

## Install

elphgap has no PyPI package yet. Build a wheel from a clone and install it:

```bash
git clone https://github.com/KristmannTechnologies/Elphgap.git && cd Elphgap
python -m pip install build
python -m build                     # dist/elphgap-<version>-py3-none-any.whl + .tar.gz
pip install dist/elphgap-*.whl      # or: pip install -e .   (editable, from source)
elphgap --version                   # -> elphgap 0.1.1
```

The reference solvers and the CLI need only NumPy + SciPy. For GPU/JAX execution
see the [README](../README.md#install).

## A Pb-like example, end to end

A **synthetic** Eliashberg spectral function ships as package data, so it works
after a plain install (no clone needed). Locate it with `example_a2f_path()`:

```bash
EX=$(python -c "import elphgap; print(elphgap.example_a2f_path())")
```

It is a two-Gaussian toy — *not* a real EPW/DFPT run — written in the **real EPW
`prefix.a2f` layout** for a single smearing: column 1 = ω [meV], column 2 =
α²F(ω), column 3 = cumulative λ(ω). Tuned to a Pb-like λ ≈ 1.16. Use it to
exercise the tool; use your own DFPT/EPW output for physics.

### 1. Inspect the spectrum

```bash
elphgap inspect "$EX"
```

```
elphgap 0.1.1 · inspect   (schema 1)
  input        .../elphgap/examples/pb_like.a2f
  sha256       55fc50f9...4dc159667  (9488 bytes)
  format       epw [auto-detected via footer]   units meV -> meV   columns 3
  a2F column   2  (smearing 0.5 meV)
  points       240 of 241 raw  (1 dropped, clip <= 0 meV)
  omega        0.05 … 12 meV
  lambda       1.1600       omega_log 5.106 meV (59.25 K)       omega_2 5.942 meV
  a2F smearing columns (1)
    col   smearing[meV]   lambda(a2F)   lambda(file)
    2            0.5       1.1600       1.1600  <- selected
  warnings
    ! [dropped_nonpositive] dropped 1 row(s) with omega <= 0 meV (singular moments).
```

`inspect` lists **every** a2F smearing column and its λ — both computed from the
spectrum (`lambda(a2F)`) and read from the file's cumulative-λ column
(`lambda(file)`); a >5% disagreement is warned. The dropped-row warning is
expected: EPW files start at ω = 0, where the `a2F/ω` moment is singular, so that
row is discarded (and reported — elphgap never clips silently).

### 2. Solve for Tc

```bash
elphgap tc "$EX" --mu-star 0.10
```

```
elphgap 0.1.1 · Tc (isotropic Migdal-Eliashberg)   (schema 1)
  format       epw [auto-detected via footer]   a2F column 2   input omega: meV
  lambda       1.1600       omega_log 5.106 meV
  Tc           5.903 K
  conventions
    mu*             0.100  (constant mu*, applied at the Matsubara cutoff omega_c)
    cutoff_factor   10
    omega_c         120 meV  (= 10 × omega_max)
    n_max           4096
    t_floor         0.05411 K       t_max 2000 K       rtol 0.001
    censored        no       output units: K
```

Always quote Tc as a **band over μ\***, not a single number. On this file:

| μ\* | Tc (isotropic ME) |
|-----|-------------------|
| 0.10 | 5.90 K |
| 0.13 | 5.46 K |
| 0.16 | 5.09 K |

Add `--json` to either command for the same fields as a machine-readable
manifest (`schema_version` "1"; keys include `input.sha256`, `format`,
`spectrum.lambda`, `smearings[]`, and for `tc` also `tc.tc_K`, `tc.censored`,
`conventions`). `--fast` swaps the default `n_max` 4096 → 512 (faster, higher Tc
floor).

### Multi-smearing EPW files

A real EPW convergence sweep writes several α²F columns (one per phonon
smearing), i.e. `1 + 2N` columns for `N` smearings. There is no canonical
default, so `tc` **requires** `--column N` in that case (and errors, exit 4,
listing the options); `inspect` shows all of them so you can pick the converged
one.

## Reference: the real EPW Pb tutorial

The synthetic file is tuned to resemble lead. For a *real* fcc-Pb calculation,
the EPW "FCC lead" superconductivity tutorial
([docs.epw-code.org](https://docs.epw-code.org)) reports, at μ\* = 0.1:
λ ≈ 1.158, ω_log ≈ 4.8 meV, McMillan Tc ≈ 4.37 K, Allen-Dynes-modified-McMillan
Tc ≈ 4.75 K, and a full isotropic-ME Tc of order 5 K (experimental Pb Tc =
7.2 K). Our synthetic λ = 1.16 / ME Tc ≈ 5.9 K at μ\* = 0.1 sits in the same
ballpark by construction — a smoke test, not a reproduction. To reproduce the
tutorial numbers, run EPW and point `elphgap tc your_prefix.a2f --format epw`
at its output.

## The same thing from Python

```python
import elphgap
from elphgap import read_a2f, moments, tc_eliashberg

spec = read_a2f(elphgap.example_a2f_path(), fmt="epw", column=2)
for w in spec.warnings:
    print("warning:", w.code, "-", w.message)

lam, wlog, w2 = moments(spec.omega, spec.a2f)
res = tc_eliashberg(spec.omega, spec.a2f, mu_star=0.10, cutoff_factor=10.0, n_max=4096)
print(f"lambda={lam:.3f}  omega_log={wlog:.2f} meV  Tc={res.tc_kelvin:.2f} K "
      f"(censored={res.censored})")
```

## Next

- [limitations.md](limitations.md) — what the isotropic CLI does and does not model.
- [troubleshooting.md](troubleshooting.md) — format/units/censoring/exit codes.
- [license-and-commercial-use.md](license-and-commercial-use.md) — GPL-3.0 in plain terms.
