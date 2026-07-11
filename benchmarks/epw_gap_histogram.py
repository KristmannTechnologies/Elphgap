#!/usr/bin/env python3
"""Analyze EPW anisotropic Eliashberg gap output (two-gap detection + a2F lambda).

Usage:
    python benchmarks/epw_gap_histogram.py <epw_out_dir>

<epw_out_dir> must contain EPW's aniso outputs, e.g.
    MgB2.imag_aniso_gap0_<T>   (Matsubara-axis gap distribution rho(delta_nk))
    MgB2.pade_aniso_gap0_<T>   (Pade-continued real-axis version)
    MgB2.a2f                   (anisotropic Eliashberg spectral function)

It reports the gap-value histogram, splits into low/high (pi/sigma) clusters, and
integrates a2F -> lambda, omega_log — useful for a side-by-side comparison of
EPW's anisotropic gap structure with this package's two-band results. Bring your
own EPW run; no EPW data is shipped. Pure-stdlib + numpy (no EPW dependency).
"""
import sys
import glob
import os
import numpy as np

TRAP = getattr(np, "trapezoid", getattr(np, "trapz", None))


def load_gap0(path):
    """Columns: T+dist, delta_nk[meV], T[K], dist(scaled), dist(not scaled)."""
    g, w = [], []
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        c = s.split()
        if len(c) < 5:
            continue
        try:
            g.append(float(c[1]))
            w.append(float(c[4]))
        except ValueError:
            continue
    return np.array(g), np.array(w)


def histogram(g, w, bin_mev=0.5, hi=14.0):
    edges = np.arange(0, hi + bin_mev / 2, bin_mev)
    hist = np.zeros(len(edges) - 1)
    for gg, ww in zip(g, w):
        i = int(gg / bin_mev)
        if 0 <= i < len(hist):
            hist[i] += ww
    return edges, hist


def report_gap(tag, path, split_mev=4.5):
    g, w = load_gap0(path)
    if len(g) == 0:
        print(f"  [{tag}] no data in {path}")
        return
    print(f"\n=== {tag}: {os.path.basename(path)} ===")
    print(f"rows={len(g)}  range=[{g.min():.2f},{g.max():.2f}] meV  total_w={w.sum():.0f}")
    edges, hist = histogram(g, w)
    mx = hist.max()
    for i, h in enumerate(hist):
        if h > 0.01 * mx:
            print(f"  {edges[i]:4.1f}-{edges[i+1]:4.1f} meV | {h:9.0f} {'#'*int(50*h/mx)}")
    lo, hiM = g < split_mev, g >= split_mev
    if w[lo].sum() > 0:
        print(f"  pi  (<{split_mev}): Δ={np.average(g[lo], weights=w[lo]):.2f} meV  "
              f"frac={w[lo].sum()/w.sum()*100:.0f}%")
    if w[hiM].sum() > 0:
        print(f"  sig(>={split_mev}): Δ={np.average(g[hiM], weights=w[hiM]):.2f} meV  "
              f"frac={w[hiM].sum()/w.sum()*100:.0f}%")
        # NOTE: a fixed-threshold split is NOT a modality test — any weight on
        # both sides (broad unimodal peak, noisy tail) lands here. Judge
        # bimodality from the histogram above; the split just summarizes the
        # sigma/pi means for MgB2, where the two-gap structure is known.
        print(f"  => weight on both sides of the {split_mev} meV split "
              "(inspect histogram for actual bimodality)" if w[lo].sum() > 0
              else "  => single high cluster")
    else:
        print(f"  sig(>={split_mev}): EMPTY -- no second (sigma) gap")


def report_a2f(path):
    rows = []
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#") or "lambda" in s.lower() or "Integrated" in s \
                or "smearing" in s.lower():
            continue
        try:
            rows.append([float(x) for x in s.split()])
        except ValueError:
            continue
    a = np.array(rows)
    if a.ndim != 2 or a.shape[1] < 2:
        print(f"\n[a2f] unexpected shape in {path}")
        return
    w = a[:, 0]
    pos = w > 1e-6
    a2f = a[:, 1]
    lam = 2 * TRAP(a2f[pos] / w[pos], w[pos])
    wlog = np.exp(2 / lam * TRAP(a2f[pos] / w[pos] * np.log(w[pos]), w[pos]))
    print(f"\n=== a2F: {os.path.basename(path)} ===")
    print(f"freq range [{w.min():.1f},{w.max():.1f}] meV  ->  lambda={lam:.3f}  "
          f"wlog={wlog:.1f} meV")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1]
    if not os.path.isdir(d):
        sys.exit(f"error: not a directory: {d}")
    found = 0
    for f in sorted(glob.glob(os.path.join(d, "*imag_aniso_gap0_*"))):
        if f.endswith((".cube", ".frmsf")):
            continue
        report_gap("IMAG (Matsubara)", f)
        found += 1
    for f in sorted(glob.glob(os.path.join(d, "*pade_aniso_gap0_*"))):
        if f.endswith((".cube", ".frmsf")):
            continue
        report_gap("PADE (real-axis)", f)
        found += 1
    for a2f in sorted(glob.glob(os.path.join(d, "*.a2f"))):
        report_a2f(a2f)
        found += 1
    if found == 0:
        sys.exit(f"error: no EPW aniso-gap or .a2f files found in {d}")


if __name__ == "__main__":
    main()
