# elphgap — GPU-accelerated anisotropic Migdal-Eliashberg gap solver

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21284954.svg)](https://doi.org/10.5281/zenodo.21284954)

**elphgap** solves the (an)isotropic Migdal-Eliashberg gap equations on the
imaginary (Matsubara) axis for phonon-mediated superconductors — in
[JAX](https://github.com/google/jax), on CPU or GPU (batched over materials
on the isotropic path; the anisotropic solver is jit-compiled per material). We are not
aware of another open-source implementation that accelerates the
*anisotropic* Migdal-Eliashberg gap-equation solve itself on GPUs: the
open-source anisotropic solvers we know of ([EPW](https://epw-code.org/),
[EPIq](https://arxiv.org/abs/2306.15462), elphbolt via its bundled
`superconda` app) run the gap solve on CPU, and the GPU efforts in this
ecosystem accelerate upstream or adjacent kernels — Wannier interpolation
([EPW-GPU, arXiv:2603.10295](https://arxiv.org/abs/2603.10295)), Boltzmann
transport ([Perturbo-GPU, arXiv:2511.03683](https://arxiv.org/abs/2511.03683))
— not the gap equation. GPU codes for *other* superconductivity formalisms
(SuperConga: quasiclassical Eilenberger; DCA++: Hubbard-model pairing)
solve different equations. If you know of a prior GPU Migdal-Eliashberg
gap solver, please open an issue and we will link it here. NumPy reference
implementations of every solver are included and tested against the JAX
paths.

**What it does:** given an isotropic α²F(ω) *or* band/pocket-resolved
α²Fᵢⱼ(ω) spectra with DOS weights (designed for a few to ~20 bands; the
shipped tests exercise K ≤ 2; a literature λᵢⱼ matrix can be encoded as
single-mode spectra, see `benchmarks/mgb2_twoband.py`), it
computes superconducting gaps Δ(T) and Tc, plus Allen-Dynes / McMillan
estimates from spectral moments. Tc is found by bisection on the leading
eigenvalue of the linearized kernel — isotropic (`tc_eliashberg`) and
anisotropic (`tc_aniso_linearized`, the recommended way to quote an
anisotropic Tc). A nonlinear gap-collapse bisection (`tc_aniso`) is also
available; it is a threshold heuristic (see *Scope and limitations*).

**What it does NOT do:** it does not compute the electron-phonon coupling
itself. Garbage in, garbage out — the physical quality of Tc is bounded by
the quality of your α²F/λᵢⱼ input (DFPT/EPW convergence, harmonicity, μ*).

> **Status: research software (v0.x).** Provided as is, without warranty of
> any kind (see LICENSE). Every solver path ships with tests and benchmark
> scripts — run them on your machine before trusting any number, and treat
> results as inputs to your own convergence and sanity checks, not as
> answers. Numerical output depends strongly on the supplied α²F/λᵢⱼ, the
> Matsubara cutoff, and μ*; none of these choices can be made for you.
> Interfaces may change between 0.x versions.

---

## Install

From source (no PyPI package):

```bash
git clone https://github.com/KristmannTechnologies/Elphgap.git && cd Elphgap
pip install -e ".[cpu-jax,test]"   # JAX on CPU + test deps
pytest                             # verify (~2 min): suite passes; one test
                                   # auto-skips without the external BETE-NET DB
```

The reference solvers alone need only NumPy + SciPy: `pip install -e .`

For GPU execution, first install a CUDA-enabled JAX build following the
[official JAX install instructions](https://docs.jax.dev/en/latest/installation.html)
for your CUDA version (e.g. `pip install -U "jax[cuda12]"`), then
`pip install -e .`.

## Quickstart

```python
import numpy as np
from elphgap import moments, tc_allen_dynes, tc_eliashberg

# alpha^2F as two columns (omega [meV], a2F) — from your own DFPT/EPW run:
#   w, a2f = np.loadtxt("a2F.dat", unpack=True)
#   keep = w > 1.0            # cut numerical low-omega junk first
#   w, a2f = w[keep], a2f[keep]
# Here, self-contained: a synthetic Einstein mode at 60 meV with lambda = 1.
w = np.linspace(1.0, 120.0, 2000)
g = np.exp(-0.5 * ((w - 60.0) / 1.5) ** 2)
a2f = g / (2.0 * np.trapezoid(g / w, w))      # normalized so lambda = 1

lam, wlog, wsq = moments(w, a2f)
result = tc_eliashberg(w, a2f, mu_star=0.13)
print(f"lambda={lam:.3f}  omega_log={wlog:.1f} meV")
print(f"Tc (Allen-Dynes): {tc_allen_dynes(lam, wlog, wsq, mu_star=0.13):.1f} K")
print(f"Tc (isotropic ME): {result.tc_kelvin:.1f} K")
```

Anisotropic (multi-band) usage: see `benchmarks/mgb2_twoband.py` — the
canonical MgB₂ two-band model (Golubov 2002 coupling matrix) end-to-end.

## Command line (isotropic)

`pip install -e .` (or a built wheel — `python -m build` then
`pip install dist/elphgap-*.whl`) puts an `elphgap` console script on your PATH.
It reads an isotropic α²F spectrum from a Quantum ESPRESSO `a2F.dos`
(frequencies in Ry) or EPW `prefix.a2f` (meV, per-smearing columns) file:

```bash
elphgap inspect examples/pb_like.a2f          # format, units, λ, ω_log, input SHA256
elphgap tc examples/pb_like.a2f --mu-star 0.10 # isotropic Migdal-Eliashberg Tc + conventions
elphgap tc examples/pb_like.a2f --json         # machine-readable
```

Isotropic only — for an anisotropic Tc use the Python API
(`tc_aniso_linearized`). The CLI never clips silently (ω≤0 rows and negative
α²F are dropped/clamped *and reported*), makes the EPW smearing-column choice
explicit (`--column N`), and uses exit codes 0/2/3/4 (ok / parse error /
Tc censored / bad parameters). Full walkthrough and the synthetic Pb-like
example: [`docs/quickstart.md`](docs/quickstart.md); scope, troubleshooting, and
a GPL-3.0 usage FAQ are under [`docs/`](docs/).

## μ* convention (read this before feeding literature λᵢⱼ)

The anisotropic solver sums `Σⱼ wⱼ · λ_solver(i,j)`. Standard band-resolved
couplings λᵢⱼ (which already contain the partial DOS Nⱼ) must be passed as
**λᵢⱼ / wⱼ**, and a uniform standard μ* as **μ*/wⱼ**. Feeding standard
matrices directly collapses Tc. This is documented in the
`solve_gap_at_T` docstring and pinned by `test_two_band_mgb2_literature`.

## Validation

The pytest suite is self-contained: `pip install -e ".[cpu-jax,test]" &&
pytest` must pass on a fresh clone. Rows marked **(external data)** were
produced on our hardware with data that is *not* shipped in this repo (the
BETE-NET database, our own EPW runs, a GPU) — treat them as reported
results, not as claims reproducible from a bare clone;
`docs/BENCHMARKS.md` explains how to obtain the data and rerun them.

| Check | Result |
|---|---|
| Isotropic limit of aniso solver vs isotropic solver | `tc_aniso_linearized` reduces to the isotropic solver EXACTLY (same matrix, same closed-form Z; pinned to rel. 1e-6 by test). The nonlinear gap-collapse `tc_aniso` tracks it to a few % (see limitations; ≤0.4 % on the MgB₂ scan) |
| NumPy ↔ JAX parity (iso + aniso) | gaps agree to rel. 1e-3, aniso Tc to ~4 %, batched-iso Tc to ~2 % on synthetic spectra (tests); the same batched path is additionally checked against DB materials when the external BETE-NET DB is present (auto-skipped otherwise) |
| MgB₂ two-band literature model (Golubov 2002) | two separated gaps, Δσ≈7–11 meV over the ω_E/μ* scan (exp: ~7, closest at μ*=0.16); Tc high by the known single-Einstein-mode simplification — trend scan, no fit (`benchmarks/mgb2_twoband.py`) |
| Spectral moments | unit-tested to 1e-3 against analytic Einstein-mode values (tests); parity ≲1e-3 vs. BETE-NET reference values over 806 materials **(external data)** (`benchmarks/db_parity.py`) |
| Tc(ME) vs. Allen-Dynes across 806 materials | median ratio 1.20; AD accurate ~10 % for λ≳0.8, up to 2× off at λ≈0.35 **(external data)** |
| MgB₂ two-gap vs. EPW | `benchmarks/epw_gap_histogram.py` post-processes EPW's anisotropic gap output for side-by-side comparison with the two-band model — bring your own EPW run; **no head-to-head comparison is shipped** |

## Performance

The core anisotropic solve is reproducible on any CUDA GPU **from a bare
clone** — `benchmarks/aniso_speedup.py` is self-contained (synthetic
multi-band spectra, no external data). It times the *identical* JAX solver
on CPU and GPU, so the numbers below isolate the backend. They are **not** a
comparison against the shipped NumPy reference, which is written for
readability, is far slower, and is not used as a performance baseline.

Anisotropic gap iteration, identical float64 JAX code, one NVIDIA A100 vs.
the same host's CPU (measured; your hardware will differ — rerun
`python benchmarks/aniso_speedup.py`):

| Problem size (K bands × N Matsubara) | GPU | CPU (same code) | speedup |
|---|---|---|---|
| K=16 × 512  | ~1.0 ms/iter  | ~53 ms/iter  | ~50× |
| K=16 × 1024 | ~0.95 ms/iter | ~101 ms/iter | ~106× |
| K=20 × 1024 | ~1.5 ms/iter  | ~213 ms/iter | ~140× |

The advantage grows with band count and Matsubara size; below K≈8 the GPU is
underutilized and the two backends are comparable, so there is little to
gain on small problems. The self-consistent solve runs this iteration
thousands of times per temperature, so the per-iteration gain carries to
end-to-end anisotropic Tc.

Batched **isotropic** Tc throughput additionally needs the external BETE-NET
database (**external data**; rerun `benchmarks/gpu_speedup.py` on yours
before citing): on one A100, float32 at n_max=1024, the full 806-material
database solves in ~440 s (~1.8 materials/s); float64 at n_max=1024 is
~1.7 materials/s.

## Scope and limitations

- Imaginary-axis (Matsubara) formulation; no real-axis continuation
  (spectral functions/tunneling DOS) yet.
- **`tc_aniso` (gap-collapse) is a threshold heuristic**: bisection on where
  the nonlinear gap collapses below `gap_threshold_mev`. Near Tc the
  fixed-point iteration slows down critically, which biases the extracted Tc
  high by a few percent at default settings (gap values Δ(T) away from Tc
  are unaffected). Un-converged states are NOT classified by their transient
  gap magnitude — no finite iterate can certify SC vs. normal near marginal
  stability — but by the linearized kernel's leading eigenvalue at that
  temperature (the exact criterion). Without that guard, a slowly decaying
  normal-state transient near the resolvable floor was misclassified as
  superconducting (a categorical error, not a percent-level bias; pinned by
  regression tests in both backends). Prefer **`tc_aniso_linearized`** to
  quote Tc: it bisects the leading eigenvalue of the linearized kernel,
  exactly like the isotropic solver, and reports sub-floor materials as
  censored.
- ω-grids must be strictly positive and increasing (the a2F/ω moments and
  Matsubara kernels are singular at ω=0); the public entry points validate
  this and fail with an actionable error instead of returning NaN.
- Inputs are assumed **harmonic**; strongly anharmonic soft-mode systems can
  shift λ substantially (known field-wide caveat — SSCHA-class corrections
  are upstream of this solver).
- μ* is an input parameter with a cutoff convention; report Tc as a band
  over μ* ∈ {0.10, 0.13, 0.16} rather than a point value.
- Migdal approximation assumed valid (ω_ph ≪ E_F); check before trusting
  results on flat-band/low-E_F systems.
- Anisotropic mode takes a coarse-grained band/pocket coupling structure
  (2–20 bands), not the full k-resolved EPW ephmat files.
- Reference values in the validation table were produced on our hardware
  with the pinned settings in `benchmarks/`; different BLAS/JAX/driver
  stacks can shift results at the last digits. Reproduce locally before
  citing.
- This package has not been through external peer review. Independent
  verification, bug reports, and counter-examples are explicitly welcome —
  please open an issue.

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later; see LICENSE) —
the same copyleft license used by Quantum ESPRESSO and EPW. No warranty, no
liability; see the license text for the full disclaimer. If you use results from this code in a
publication, you are responsible for validating them against the included
benchmarks and for your own convergence studies.

## Cite

See `CITATION.cff` (Zenodo DOI on release). Please also cite the methods
this builds on: Allen & Dynes (1975), Margine & Giustino PRB 87, 024505
(2013) for the anisotropic ME formulation, and EPW/IsoME if you use them to
generate inputs or cross-checks. If you run the BETE-NET database
benchmarks, cite Gibson et al., npj Comput. Mater. 11, 7 (2025).
