# Running the benchmarks

- `mgb2_twoband.py` — self-contained (literature coupling matrix):
  `python benchmarks/mgb2_twoband.py`
- `aniso_speedup.py` — self-contained wall-clock benchmark of the
  anisotropic solve on synthetic multi-band spectra: per-iteration cost of
  the NumPy reference vs. the JAX backend across K bands × N Matsubara
  sizes, plus an optional end-to-end Tc timing (`--tc`). Run it once with a
  CUDA JAX build and once with `JAX_PLATFORMS=cpu` to isolate the GPU
  contribution; `ELPHGAP_JAX_X64=0` switches the JAX path to float32.
- `db_parity.py` — needs the BETE-NET `database.json`, which is **not
  shipped** with this repo. Obtain it from the BETE-NET project
  (github.com/henniggroup/BETE-NET), mind its license/terms, and cite
  Gibson et al., npj Comput. Mater. 11, 7 (2025) if you use it. Run:
  `python benchmarks/db_parity.py --db /path/to/database.json [--limit N]`
  (default path: `benchmarks/data/betenet_database.json`, gitignored).
  The comparison plot needs `matplotlib` (optional — the script skips
  plotting if it is missing). Outputs go to `results/`.
- `gpu_speedup.py` — batched Tc throughput on CPU or CUDA; same `--db`
  handling as above. Requires a JAX build matching your hardware.
  `ELPHGAP_JAX_X64=0` switches to float32 for speed runs (check the parity
  impact on your data first).
- `epw_gap_histogram.py` — post-processes an EPW anisotropic-gap output
  directory (`*.imag_aniso_gap0_*` files) into the two-gap histogram:
  `python benchmarks/epw_gap_histogram.py /path/to/epw_out`. Bring your own
  EPW run; no EPW data is shipped.

Performance numbers quoted in the README were measured on our hardware —
rerun on yours before citing.
