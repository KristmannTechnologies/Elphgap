"""Throughput benchmark: full-database Eliashberg Tc via the batched JAX backend.

Identical code path on CPU and CUDA. Reports materials/second and total wall
time; cross-checks a few Tc values against results/db_parity.json if present.

The BETE-NET database is NOT shipped — see docs/BENCHMARKS.md.

Run:  python benchmarks/gpu_speedup.py [--db PATH] [--batch 32] [--n-mat 1024]
GPU:  ELPHGAP_JAX_X64=0 for a float32 speed run (check parity impact first).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from elphgap import load_database
from elphgap.eliashberg_jax import tc_batched

ROOT = Path(__file__).resolve().parents[1]
DB_DEFAULT = ROOT / "benchmarks" / "data" / "betenet_database.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_DEFAULT,
                        help="path to the BETE-NET database.json (not shipped)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n-mat", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(
            f"database not found: {args.db}\n"
            "Download it from the BETE-NET project (github.com/henniggroup/"
            "BETE-NET) and pass --db /path/to/database.json"
        )
    print("jax backend:", jax.default_backend(), "| devices:", jax.devices())
    mats = load_database(str(args.db))
    if args.limit:
        mats = mats[: args.limit]
    # Sort by grid length so padding waste inside a batch stays small.
    mats = sorted(mats, key=lambda m: len(m.omega))

    # One global grid length + constant batch shape => XLA compiles exactly once.
    g_max = max(len(m.omega) for m in mats)
    rows = []
    t0 = time.perf_counter()
    for i in range(0, len(mats), args.batch):
        chunk = mats[i : i + args.batch]
        n_real = len(chunk)
        if n_real < args.batch:  # pad tail chunk to keep the jit shape constant
            chunk = chunk + [chunk[-1]] * (args.batch - n_real)
        tc, censored = tc_batched(chunk, n_mat=args.n_mat, grid_pad_to=g_max)
        rows += [
            {"comp_name": m.comp_name, "tc_me_k": float(t), "censored": bool(c)}
            for m, t, c in list(zip(chunk, tc, censored))[:n_real]
        ]
        done = min(i + args.batch, len(mats))
        print(f"{done}/{len(mats)}  ({done / (time.perf_counter() - t0):.2f} materials/s)", flush=True)
    wall = time.perf_counter() - t0
    print(f"TOTAL {len(mats)} materials in {wall:.1f} s = {len(mats) / wall:.2f} materials/s")

    out = ROOT / "results" / "gpu_speedup.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"backend": jax.default_backend(), "n_mat": args.n_mat,
                   "wall_seconds": wall, "rows": rows}, f)
    print("wrote", out)

    ref_path = ROOT / "results" / "db_parity.json"
    if ref_path.exists():
        ref = {r["comp_name"]: r for r in json.load(open(ref_path))["rows"]}
        devs = [abs(r["tc_me_k"] - ref[r["comp_name"]]["tc_me_k"]) / ref[r["comp_name"]]["tc_me_k"]
                for r in rows
                if r["comp_name"] in ref and not r["censored"]
                and not ref[r["comp_name"]]["censored"] and ref[r["comp_name"]]["tc_me_k"] > 1.0]
        if devs:
            print(f"parity vs numpy reference (Tc>1K): median {np.median(devs):.2%}, max {max(devs):.2%}")


if __name__ == "__main__":
    main()
