"""Validation of the elphgap solver against the BETE-NET database.

1. Moment parity: recompute lambda, w_log, w_2 from a2F and compare to the
   stored reference columns (validates parsing + integration).
2. Tc for all materials: Allen-Dynes and full isotropic Eliashberg.
3. Outputs: results/db_parity.json, results/tc_comparison.png (plot needs
   matplotlib; skipped if missing).

The database is NOT shipped with this repo — obtain it from the BETE-NET
project (github.com/henniggroup/BETE-NET) and mind its license/terms.

Run:  python benchmarks/db_parity.py --db /path/to/database.json [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from elphgap import load_database, moments, tc_allen_dynes, tc_eliashberg, tc_mcmillan

ROOT = Path(__file__).resolve().parents[1]
DB_DEFAULT = ROOT / "benchmarks" / "data" / "betenet_database.json"
RESULTS = ROOT / "results"
MU_STAR = 0.10


def process(mat) -> dict:
    lam, wlog, w2 = moments(mat.omega, mat.a2f)
    t0 = time.perf_counter()
    me = tc_eliashberg(mat.omega, mat.a2f, mu_star=MU_STAR)
    return {
        "index": mat.index,
        "comp": mat.comp,
        "comp_name": mat.comp_name,
        "lambda": lam,
        "wlog_mev": wlog,
        "w2_mev": w2,
        "lambda_ref": mat.lambda_ref,
        "wlog_ref": mat.wlog_ref,
        "w2_ref": mat.wsq_ref,
        "tc_ad_k": tc_allen_dynes(lam, wlog, w2, MU_STAR),
        "tc_mcmillan_k": tc_mcmillan(lam, wlog, MU_STAR),
        "tc_me_k": me.tc_kelvin,
        "censored": me.censored,
        "me_seconds": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_DEFAULT,
                        help="path to the BETE-NET database.json (not shipped)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(
            f"database not found: {args.db}\n"
            "Download it from the BETE-NET project (github.com/henniggroup/"
            "BETE-NET) and pass --db /path/to/database.json"
        )
    mats = load_database(str(args.db))
    if args.limit:
        mats = mats[: args.limit]
    print(f"{len(mats)} materials, mu*={MU_STAR}")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(process, mats, chunksize=4))
    wall = time.perf_counter() - t0

    lam_dev = max(abs(r["lambda"] - r["lambda_ref"]) / r["lambda_ref"] for r in rows)
    wlog_dev = max(abs(r["wlog_mev"] - r["wlog_ref"]) / r["wlog_ref"] for r in rows)
    w2_dev = max(abs(r["w2_mev"] - r["w2_ref"]) / r["w2_ref"] for r in rows)
    resolved = [r for r in rows if not r["censored"]]
    print(f"moment parity: max rel dev lambda={lam_dev:.2e}, w_log={wlog_dev:.2e}, w_2={w2_dev:.2e}")
    print(f"Eliashberg resolved: {len(resolved)}/{len(rows)} (rest censored: Tc below floor)")
    print(f"wall time {wall:.0f} s, mean {np.mean([r['me_seconds'] for r in rows]):.1f} s/material/core")

    ratios = [r["tc_me_k"] / r["tc_ad_k"] for r in resolved if r["tc_ad_k"] > 0.5]
    if ratios:
        print(f"Tc(ME)/Tc(AD) for Tc_AD>0.5K: median {np.median(ratios):.3f}, "
              f"p10 {np.percentile(ratios, 10):.3f}, p90 {np.percentile(ratios, 90):.3f}")
    else:
        print("Tc(ME)/Tc(AD): no resolved materials with Tc_AD > 0.5 K in this subset")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "db_parity.json"
    with open(out, "w") as f:
        json.dump(
            {
                "mu_star": MU_STAR,
                "n_materials": len(rows),
                "moment_parity": {"lambda_max_rel_dev": lam_dev, "wlog_max_rel_dev": wlog_dev,
                                  "w2_max_rel_dev": w2_dev},
                "wall_seconds": wall,
                "rows": rows,
            },
            f,
        )
    print(f"wrote {out}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping the comparison plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ad = np.array([r["tc_ad_k"] for r in resolved])
    me = np.array([r["tc_me_k"] for r in resolved])
    lam_arr = np.array([r["lambda"] for r in resolved])
    sc = axes[0].scatter(ad, me, c=lam_arr, s=10, cmap="viridis", vmin=0.3, vmax=2.0)
    lims = [0.05, max(ad.max(), me.max()) * 1.3]
    axes[0].plot(lims, lims, "k--", lw=0.8)
    axes[0].set(xscale="log", yscale="log", xlim=lims, ylim=lims,
                xlabel="Tc Allen-Dynes [K]", ylabel="Tc Eliashberg (isotropic) [K]",
                title=f"BETE-NET DB, {len(resolved)} materials, mu*={MU_STAR}")
    fig.colorbar(sc, ax=axes[0], label="lambda")
    mask = ad > 0.5
    axes[1].scatter(lam_arr[mask], me[mask] / ad[mask], s=10, alpha=0.6)
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="lambda", ylabel="Tc(ME) / Tc(AD)",
                title="Allen-Dynes accuracy vs coupling strength")
    fig.tight_layout()
    fig.savefig(RESULTS / "tc_comparison.png", dpi=160)
    print(f"wrote {RESULTS / 'tc_comparison.png'}")


if __name__ == "__main__":
    main()
