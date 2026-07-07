"""Wall-clock benchmark: anisotropic gap solve, NumPy reference vs. JAX backend.

Self-contained (synthetic multi-band spectra, no external data). Measures the
per-iteration cost of the self-consistent Δ(k,n), Z(k,n) fixed point at fixed
T across problem sizes (K bands × N Matsubara points), plus an optional
end-to-end tc_aniso comparison at K=2.

Method: each backend is timed at I and 2·I iterations with identical inputs;
per-iteration time = (t_2I − t_I) / I. The difference cancels one-time costs
(λ-kernel build, JAX jit compilation, dispatch overhead), which are reported
separately (`jax_first_call_s` includes compilation). The two backends do the
same O(K²·N²) work per iteration.

JAX runs on whatever backend is available — run once with CUDA and once with
JAX_PLATFORMS=cpu to isolate the GPU contribution. ELPHGAP_JAX_X64=0 switches
the JAX path to float32 (the NumPy reference is float64-only; use --iters-np 0
to skip it on repeat runs).

Run:  python benchmarks/aniso_speedup.py [--sizes 2x512,8x1024] [--tc]
      [--iters-np 50] [--iters-jax 100] [--out results/aniso_speedup.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MU_STAR = 0.10
T_KELVIN = 1.0  # low T so n_mat = n_max for every size below (cutoff 10*120 meV)


def synthetic_problem(k: int, grid: int = 1200):
    """K-band α²Fᵢⱼ(ω): Gaussian modes, near-band-diagonal coupling, λ_eff ≈ 1.5.

    Deterministic (no RNG) so timings are comparable across machines/runs.
    """
    w = np.linspace(1.0, 120.0, grid)
    centers = np.linspace(25.0, 95.0, k) if k > 1 else np.array([60.0])
    modes = np.exp(-0.5 * ((w[None, :] - centers[:, None]) / 4.0) ** 2)  # (K,G)
    pair = 0.5 * (modes[:, None, :] + modes[None, :, :])  # (K,K,G), symmetric
    idx = np.arange(k)
    amp = 0.6 ** np.abs(idx[:, None] - idx[None, :])  # near-diagonal dominance
    a2f = amp[:, :, None] * pair
    weights = np.full(k, 1.0 / k)
    # scale so the weighted effective coupling  λ_i = Σ_j w_j·2∫α²F_ij/ω dω  averages 1.5
    lam_ij = 2.0 * np.trapezoid(a2f / w[None, None, :], w, axis=2)
    lam_i = lam_ij @ weights
    a2f *= 1.5 / lam_i.mean()
    return w, a2f, weights


def time_call(fn, repeats: int = 1) -> float:
    """Best-of-N wall time (min suppresses scheduler noise on sub-ms calls)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="2x512,4x512,8x512,16x512,8x1024,16x1024,20x1024",
                        help="comma-separated KxN cases (K bands, N Matsubara points)")
    parser.add_argument("--iters-np", type=int, default=50,
                        help="base iteration count for the NumPy reference (0 = skip)")
    parser.add_argument("--iters-jax", type=int, default=100)
    parser.add_argument("--tc", action="store_true",
                        help="also run one end-to-end tc_aniso comparison at K=2")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "aniso_speedup.json")
    args = parser.parse_args()

    import jax

    from elphgap.eliashberg_aniso import solve_gap_at_T as solve_np
    from elphgap.eliashberg_aniso_jax import solve_gap_at_T as solve_jx

    backend = jax.default_backend()
    x64 = jax.config.jax_enable_x64
    print(f"jax backend: {backend} | devices: {jax.devices()} | float64: {x64}")

    cases = []
    for spec in args.sizes.split(","):
        k, n = (int(v) for v in spec.strip().split("x"))
        w, a2f, weights = synthetic_problem(k)
        common = dict(mu_star=MU_STAR, n_max=n)
        row = {"k": k, "n_mat": n}

        i_j = args.iters_jax
        run_j = lambda it: solve_jx(w, a2f, weights, T_KELVIN, n_iter=it, **common)
        row["jax_first_call_s"] = time_call(lambda: run_j(i_j))  # includes jit compile

        def diff_jax(it):
            t1 = time_call(lambda: run_j(it), repeats=3)
            run_j(2 * it)  # warm-up for the 2I shape…
            t2 = time_call(lambda: run_j(2 * it), repeats=3)  # …then time it
            return t2 - t1

        dt = diff_jax(i_j)
        # resolution guard: on fast devices the per-iteration cost is µs-scale,
        # so grow the iteration block until the differenced time clears
        # dispatch/timer noise by a wide margin.
        while dt < 0.2 and i_j < 200_000:
            i_j *= 4
            dt = diff_jax(i_j)
        _, _, gap_j = run_j(i_j)
        row["jax_iters_used"] = i_j
        row["jax_per_iter_s"] = dt / i_j
        row["max_gap_jax_mev"] = gap_j

        if args.iters_np > 0:
            i_n = args.iters_np
            run_n = lambda it: solve_np(w, a2f, weights, T_KELVIN, tol=0.0, max_iter=it, **common)
            t1 = time_call(lambda: run_n(i_n))
            state = None

            def run_2i():
                nonlocal state
                state = run_n(2 * i_n)

            t2 = time_call(run_2i)
            row["numpy_per_iter_s"] = (t2 - t1) / i_n
            row["max_gap_numpy_mev"] = state.max_gap_mev
            row["speedup"] = row["numpy_per_iter_s"] / row["jax_per_iter_s"]

        cases.append(row)
        msg = f"K={k:>2} N={n:>4}  jax {row['jax_per_iter_s'] * 1e3:8.2f} ms/iter"
        if "speedup" in row:
            msg += (f"  numpy {row['numpy_per_iter_s'] * 1e3:8.2f} ms/iter"
                    f"  speedup {row['speedup']:6.1f}x")
        print(msg, flush=True)

    tc = None
    if args.tc:
        from elphgap.eliashberg_aniso import tc_aniso as tc_np
        from elphgap.eliashberg_aniso_jax import tc_aniso as tc_jx

        w, a2f, weights = synthetic_problem(2)
        t_jx = time.perf_counter()
        tc_j = tc_jx(w, a2f, weights, mu_star=MU_STAR)
        t_jx = time.perf_counter() - t_jx
        t_np = time.perf_counter()
        tc_n = tc_np(w, a2f, weights, mu_star=MU_STAR)
        t_np = time.perf_counter() - t_np
        tc = {"k": 2, "tc_numpy_k": tc_n, "tc_jax_k": tc_j,
              "wall_numpy_s": t_np, "wall_jax_s": t_jx}
        print(f"tc_aniso end-to-end (K=2): numpy {tc_n:.2f} K in {t_np:.1f} s | "
              f"jax {tc_j:.2f} K in {t_jx:.1f} s", flush=True)

    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"backend": backend, "devices": [str(d) for d in jax.devices()],
                   "float64": bool(x64), "t_kelvin": T_KELVIN, "mu_star": MU_STAR,
                   "iters_np": args.iters_np, "iters_jax": args.iters_jax,
                   "cases": cases, "tc_end_to_end": tc}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
