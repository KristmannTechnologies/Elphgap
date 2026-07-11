"""Anisotropic (band-resolved) Migdal-Eliashberg solver (imaginary axis) — reference.

Band/pocket-resolved gap Δ(k,iω_n) and renormalization Z(k,iω_n), solved
self-consistently. The GPU/JAX port (eliashberg_aniso_jax.py) is a mechanical
lift of this numpy reference (same array structure, one extra band axis on top
of the isotropic solver's Matsubara axis), so correctness is pinned here first.

Anisotropic imaginary-axis equations (Margine & Giustino, PRB 87, 024505 (2013);
Allen & Mitrović 1982), with a Fermi-surface weight w_k (Σ_k w_k = 1):

    Z(k,n)   = 1 + (πT/ω_n) Σ_{k',n'} w_{k'} · [ω_{n'}/R(k',n')] · λ(k,k',n−n')
    Z(k,n)Δ(k,n) = πT Σ_{k',n'} w_{k'} · [Δ(k',n')/R(k',n')] · [λ(k,k',n−n') − μ*]
    R(k',n') = sqrt(ω_{n'}² + Δ(k',n')²)

    λ(k,k',n−n') = ∫dω  2ω α²F_{k,k'}(ω) / ((ω_n − ω_{n'})² + ω²)   (depends on n−n' only)

ω_n = πT(2n+1). μ* is applied on the same Matsubara cutoff as the isotropic
solver (eliashberg.py).

ISOTROPIC LIMIT (the hard self-test): if α²F_{k,k'}(ω) = α²F(ω) for all k,k',
then λ is k-independent, Δ(k,n)=Δ(n), Z(k,n)=Z(n), and Σ_{k'} w_{k'}=1 collapses
the k'-sum — recovering the isotropic ME equations whose linearization is
exactly the isotropic solver. The Z Matsubara sum includes the exact tail
beyond the truncated matrix (see _z_tail), matching the isotropic solver's
closed-form Z, so tc_aniso_linearized reduces to the isotropic Tc EXACTLY
(pinned by test), and the nonlinear tc_aniso differs from it only by the
gap-collapse Tc-extraction heuristic (see the tc_aniso docstring).

Frequencies/energies in meV, T in K at the public API. See test_aniso.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .eliashberg import TcResult, _matrix_size
from .grids import trapezoid_weights, validate_grid
from .units import K_TO_MEV, MEV_TO_K


def lambda_kernel(omega: np.ndarray, a2f_pairs: np.ndarray, t_mev: float, jmax: int) -> np.ndarray:
    """λ(k,k',m) for m = 0..jmax.

    a2f_pairs: (K, K, G) anisotropic α²F on the shared ω-grid, or (G,) isotropic.
    Returns (K,K,jmax+1) or (jmax+1,) matching the input rank.
    """
    nu = 2.0 * np.pi * t_mev * np.arange(jmax + 1)  # bosonic differences ν_m
    # kernel_m(ω) = 2ω/((ν_m)² + ω²); integrate against α²F over ω.
    g = 2.0 * omega / (nu[:, None] ** 2 + omega[None, :] ** 2)  # (jmax+1, G)
    if a2f_pairs.ndim == 1:
        return np.trapezoid(g * a2f_pairs[None, :], omega, axis=1)  # (jmax+1,)
    # (M,G) x (K,K,G) -> (K,K,M): contract over ω with trapezoid weights —
    # broadcasting to (K,K,M,G) before np.trapezoid would allocate K²·M·G
    # floats (~8 GB at K=20, n_mat=1024, G~1200).
    tw = trapezoid_weights(omega)
    return np.einsum("mg,ijg->ijm", g * tw[None, :], a2f_pairs, optimize=True)


def _z_tail(lam_kk: np.ndarray, w: np.ndarray, n_mat: int) -> np.ndarray:
    """Exact tail of the Z Matsubara sum beyond the truncated (N x N) matrix.

    In the tail n' >= N (ω_{n'} >= ω_c >> Δ) the factor ω_{n'}/R(n') is 1
    exactly in the normal state and to O(Δ²/ω_c²) in the SC state, so

        tail(n) = Σ_{n'=N..∞} [λ(n'-n) - λ(n+n'+1)] = Σ_{l=N-n}^{N+n} λ(l)
                = C(N+n) - C(N-n-1),   C(m) = Σ_{l=0..m} λ(l)

    (the two infinite sums cancel beyond l = N+n). Adding it makes Z match
    the isotropic solver's closed-form (untruncated) Matsubara sum. O(K²N)
    via cumulative sums — no (K,K,N,N) folded tensors. Needs λ up to
    l = 2N-1 < len(lam_kk). Returns the w-weighted tail, shape (K, N).
    """
    c = np.cumsum(lam_kk, axis=2)  # (K,K,M)
    n = np.arange(n_mat)
    tail = c[:, :, n_mat + n] - c[:, :, n_mat - n - 1]  # (K,K,N)
    return np.einsum("j,ijn->in", w, tail)


@dataclass
class AnisoState:
    delta: np.ndarray  # (K, N) meV
    z: np.ndarray  # (K, N)
    converged: bool  # residual of BOTH Delta and Z below tol
    iterations: int
    max_gap_mev: float


def solve_gap_at_T(
    omega: np.ndarray,
    a2f_pairs: np.ndarray,
    weights: np.ndarray,
    t_kelvin: float,
    mu_star: float = 0.10,
    cutoff_factor: float = 10.0,
    n_max: int = 512,
    delta0_mev: float = 1.0,
    mixing: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> AnisoState:
    """Self-consistent Δ(k,n), Z(k,n) at fixed T via damped fixed-point iteration.

    weights: (K,) Fermi-surface weights w_j (∝ partial DOS N_j), normalized to sum 1.
    a2f_pairs: (K,K,G) or (G,) [isotropic, broadcast over the K×K block].

    COUPLING CONVENTION (important — easy to get wrong for multi-band input):
    the kernels are summed as  Σ_j w_j · λ_solver(i,j) · (...)  and likewise
    Σ_j w_j · mu_star[i,j], i.e. a2f_pairs/mu_star carry the PER-PAIR coupling
    *density* and w_j supplies the target-band DOS. So the standard band-resolved
    couplings (which already include N_j, e.g. Golubov's MgB2 λ_ij with row-sum
    λ_σ = λ_σσ+λ_σπ) map as:
        λ_ij^std  = w_j · λ_solver(i,j)     ->  pass a2f_pairs[i,j] for λ_ij^std / w_j
        μ*_ij^std = w_j · mu_star[i,j]       ->  pass mu_star[i,j] = μ*_ij^std / w_j
    Feeding λ_ij^std directly makes the solver see w_j·λ_ij ≈ half the intended
    coupling and Tc collapses. (Single-band K=1 has w=1, so this is a no-op there;
    see benchmarks/mgb2_twoband.py and test_two_band_mgb2_literature.)
    """
    omega = validate_grid(omega)
    t_mev = t_kelvin * K_TO_MEV
    omega_c = cutoff_factor * float(omega[-1])
    n_mat = _matrix_size(t_mev, omega_c, n_max)
    k = len(weights)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    # mu_star: scalar or (K,K) matrix in THIS solver's convention, where the
    # standard band-resolved Coulomb is mu*_ij^std = w_j * mu_star[i,j].
    mu = np.broadcast_to(np.asarray(mu_star, dtype=np.float64), (k, k))

    # Bosonic kernel λ(k,k',m), m up to 2*n_mat (needs |n-n'| and n+n'+1 not used here).
    lam = lambda_kernel(omega, a2f_pairs, t_mev, 2 * n_mat)  # (K,K,M) or (M,)
    iso = lam.ndim == 1

    n = np.arange(n_mat)
    wn = np.pi * t_mev * (2 * n + 1)  # (N,) fermionic Matsubara, n>=0
    # Folding to n>=0 with gap parity Δ(-n-1)=Δ(n): contributions from n' and -n'-1.
    # |n - n'| and (n + n' + 1) index the bosonic kernel.
    abs_idx = np.abs(n[:, None] - n[None, :])  # (N,N)
    sum_idx = n[:, None] + n[None, :] + 1  # (N,N)

    def lam_pair(ki, kj):
        base = lam if iso else lam[ki, kj]
        return base[abs_idx], base[sum_idx]  # each (N,N)

    # Exact Z tail beyond the (N x N) block, so Z matches the isotropic
    # solver's closed-form (untruncated) Matsubara sum. Iteration-independent.
    lam_kk = np.broadcast_to(lam, (k, k) + lam.shape) if iso else lam
    z_tail = _z_tail(lam_kk, w, n_mat)  # (K,N)

    delta = np.full((k, n_mat), delta0_mev)
    z = np.ones((k, n_mat))

    for it in range(max_iter):
        r = np.sqrt(wn[None, :] ** 2 + delta**2)  # (K,N) = R(k',n')
        gz = wn[None, :] / r  # (K,N)
        gd = delta / r  # (K,N)

        new_z = np.ones((k, n_mat))
        new_zd = np.zeros((k, n_mat))
        for ki in range(k):
            acc_z = np.zeros(n_mat)
            acc_zd = np.zeros(n_mat)
            for kj in range(k):
                lam_abs, lam_sum = lam_pair(ki, kj)  # (N,N) over (n,n')
                # Z: ω_n'/R is ODD under n'<->-n'-1 -> lam_abs - lam_sum
                kz = lam_abs - lam_sum
                acc_z += w[kj] * (kz @ gz[kj])
                # Δ: Δ_n'/R is EVEN -> lam_abs + lam_sum; μ* (per band pair) on the fold
                kd = lam_abs + lam_sum - 2.0 * mu[ki, kj]
                acc_zd += w[kj] * (kd @ gd[kj])
            new_z[ki] = 1.0 + (np.pi * t_mev / wn) * (acc_z + z_tail[ki])
            new_zd[ki] = (np.pi * t_mev) * acc_zd
        new_delta = new_zd / new_z

        # Converged means BOTH fields are stationary: Z is damped separately
        # from Delta, so a Delta-only residual can report converged=True with
        # Z still far from its fixed point (e.g. delta0_mev=0).
        step = max(np.max(np.abs(new_delta - delta)), np.max(np.abs(new_z - z)))
        delta = (1 - mixing) * delta + mixing * new_delta
        z = (1 - mixing) * z + mixing * new_z
        if step < tol:
            return AnisoState(delta, z, True, it + 1, float(np.max(np.abs(delta))))

    return AnisoState(delta, z, False, max_iter, float(np.max(np.abs(delta))))


def _reject_complex_leading(ev: np.ndarray) -> float:
    """Return the leading (max real part) eigenvalue, rejecting complex pairs.

    A complex leading pair has no real crossing rho = 1: discarding the
    imaginary part would hand the bisection a number that is not an
    eigenvalue of the kernel at all.
    """
    lead = ev[np.argmax(ev.real)]
    if abs(lead.imag) > 1e-9 * max(1.0, abs(lead.real)):
        raise ValueError(
            "asymmetric a2f_pairs/mu_star produced a complex leading eigenvalue; "
            "the linearized Tc is only defined for detailed-balance-symmetric "
            "pair matrices (lam_ij = lam_ji, mu*_ij = mu*_ji in solver convention)"
        )
    return float(lead.real)


def max_eigenvalue_aniso(
    omega: np.ndarray,
    a2f_pairs: np.ndarray,
    weights: np.ndarray,
    t_mev: float,
    mu_star,
    omega_c: float,
    n_max: int,
    dense_max_dim: int = 2048,
) -> float:
    """Largest eigenvalue rho(T) of the LINEARIZED anisotropic kernel.

    Linearizing the gap equation at Delta -> 0, with the exact normal-state Z
    (closed-form infinite Matsubara sum), gives the (K·N)×(K·N) eigenproblem

        rho·x(i,n) = (πT / (ω_n Z(i,n))) Σ_{j,m} w_j B(ij,nm) x(j,m),
        B(ij,nm)   = λ_ij(|n−m|) + λ_ij(n+m+1) − 2 μ*_ij,
        Z(i,n)     = 1 + (πT/ω_n) (Λ_i(0) + 2 Σ_{l=1..n} Λ_i(l)),
        Λ_i(m)     = Σ_j w_j λ_ij(m).

    K=1 reduces EXACTLY to the isotropic solver's matrix (same sizing, same Z).
    For symmetric λ/μ* pair matrices (physical inputs in this solver's
    convention are — detailed balance N_i λ_ij = N_j λ_ji) the weighted kernel
    is diagonally similar to a symmetric matrix and solved symmetrically;
    otherwise the largest real part of the general spectrum is returned.

    For K·N <= dense_max_dim (default 2048, i.e. up to 4 bands at n_max=512)
    the matrix is built densely (exact eigh / eigvals). Above that, a
    MATRIX-FREE Lanczos/Arnoldi iteration is used instead: the |n−m| fold is
    a Toeplitz matvec and the n+m+1 fold a Hankel matvec, both applied by FFT
    (scipy.linalg.matmul_toeplitz) — O(K²N) memory and O(K²·N·log N) per
    matvec, so larger band counts (8 bands at n_max=512, K~20 at n_max~1024)
    never hit the dense (K·N)² memory/cubic-runtime cliff. Both paths agree
    to solver precision (pinned by test).
    """
    omega = validate_grid(omega)
    n_mat = _matrix_size(t_mev, omega_c, n_max)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    k = len(w)
    lam = lambda_kernel(omega, np.asarray(a2f_pairs, dtype=np.float64), t_mev, 2 * n_mat)
    if lam.ndim == 1:
        lam = np.broadcast_to(lam, (k, k, lam.shape[0]))
    mu = np.broadcast_to(np.asarray(mu_star, dtype=np.float64), (k, k))

    n = np.arange(n_mat)
    wn = np.pi * t_mev * (2 * n + 1)
    lam_i = np.einsum("j,ijm->im", w, lam)  # (K,M)
    z = 1.0 + (np.pi * t_mev / wn)[None, :] * (
        lam_i[:, :1] + 2.0 * np.concatenate([np.zeros((k, 1)), np.cumsum(lam_i[:, 1:n_mat], axis=1)], axis=1)
    )  # (K,N) exact normal-state Z
    d = wn[None, :] * z / (np.pi * t_mev)  # (K,N), > 0
    sym = np.allclose(lam, lam.transpose(1, 0, 2)) and np.allclose(mu, mu.T)

    if k * n_mat <= dense_max_dim:
        abs_idx = np.abs(n[:, None] - n[None, :])
        sum_idx = n[:, None] + n[None, :] + 1
        b = lam[:, :, abs_idx] + lam[:, :, sum_idx] - 2.0 * mu[:, :, None, None]  # (K,K,N,N)
        if sym:
            # Diagonal similarity x -> sqrt(w_i d(i,n))·x symmetrizes the kernel.
            r_scale = np.sqrt(w[:, None] / d)  # (K,N)
            s = b * r_scale[:, None, :, None] * r_scale[None, :, None, :]
            s = s.transpose(0, 2, 1, 3).reshape(k * n_mat, k * n_mat)
            return float(eigh(s, eigvals_only=True, subset_by_index=[k * n_mat - 1, k * n_mat - 1])[0])
        m = (b * w[None, :, None, None] / d[:, None, :, None]).transpose(0, 2, 1, 3).reshape(k * n_mat, k * n_mat)
        return _reject_complex_leading(np.linalg.eigvals(m))

    if not (np.any(lam) or np.any(mu)):
        # Zero kernel (a2F ≡ 0, mu* = 0): rho = 0 exactly. ARPACK would
        # otherwise break down on the zero operator ("starting vector is
        # zero") instead of reporting the censored result.
        return 0.0

    # Matrix-free path. B_ij x = Toeplitz(λ_ij)·x + Hankel(λ_ij)·x − 2μ_ij Σx,
    # with Hankel(λ)[n,m] = λ(n+m+1) applied as a Toeplitz on the reversed
    # vector: Σ_m λ(n+m+1) x_m = Σ_m' λ(N+n−m') x_{N−1−m'} (first column
    # λ(N+n), first row λ(N−m); λ is tabulated up to 2N so all indices exist).
    from scipy.linalg import matmul_toeplitz
    from scipy.sparse.linalg import LinearOperator, eigs, eigsh

    lam_t = np.ascontiguousarray(lam[:, :, :n_mat])            # λ(|n−m|) col = row
    lam_h_col = np.ascontiguousarray(lam[:, :, n_mat:2 * n_mat])  # λ(N+n)
    lam_h_row = np.ascontiguousarray(lam[:, :, n_mat:0:-1])       # λ(N−m)

    def apply_b(y):  # (K,N) -> (K,N): out_i = Σ_j B_ij y_j (no w, no d)
        out = np.zeros((k, n_mat))
        for i in range(k):
            for j in range(k):
                yj = y[j]
                out[i] += matmul_toeplitz((lam_t[i, j], lam_t[i, j]), yj)
                out[i] += matmul_toeplitz((lam_h_col[i, j], lam_h_row[i, j]), yj[::-1])
                out[i] -= 2.0 * mu[i, j] * yj.sum()
        return out

    if sym:
        r_scale = np.sqrt(w[:, None] / d)  # (K,N)

        def matvec(v):
            x = v.reshape(k, n_mat)
            return (r_scale * apply_b(r_scale * x)).ravel()

        op = LinearOperator((k * n_mat, k * n_mat), matvec=matvec, dtype=np.float64)
        return float(eigsh(op, k=1, which="LA", return_eigenvectors=False)[0])

    def matvec(v):
        x = v.reshape(k, n_mat)
        return (apply_b(w[:, None] * x) / d).ravel()

    op = LinearOperator((k * n_mat, k * n_mat), matvec=matvec, dtype=np.float64)
    return _reject_complex_leading(eigs(op, k=1, which="LR", return_eigenvectors=False))


def tc_aniso_linearized(
    omega: np.ndarray,
    a2f_pairs: np.ndarray,
    weights: np.ndarray,
    mu_star=0.10,
    cutoff_factor: float = 10.0,
    n_max: int = 512,
    t_max_kelvin: float = 2000.0,
    rtol: float = 1e-3,
    dense_max_dim: int = 2048,
) -> TcResult:
    """Anisotropic Tc from bisection on rho(T) = 1 of the linearized kernel.

    Same method as the isotropic tc_eliashberg — to which it reduces exactly
    for isotropic input (pinned by test) — and the recommended way to quote an
    anisotropic Tc: it is free of the seed/iteration dependence of the
    gap-collapse heuristic (see tc_aniso) and reports sub-floor materials as
    censored instead of guessing. Cost: one leading-eigenvalue solve per
    temperature — dense up to K·N = dense_max_dim, matrix-free (FFT folds +
    Lanczos, O(K²N) memory) above, so 8 bands at n_max=512 or K~20 at
    n_max~1024 stay usable (see max_eigenvalue_aniso).
    """
    omega = validate_grid(omega)
    omega_c = cutoff_factor * float(omega[-1])
    t_floor_k = max(omega_c / (2.0 * np.pi * n_max) * MEV_TO_K, 1e-3)

    def rho(t_k: float) -> float:
        return max_eigenvalue_aniso(omega, a2f_pairs, weights, t_k * K_TO_MEV, mu_star,
                                    omega_c, n_max, dense_max_dim=dense_max_dim)

    rho_floor = rho(t_floor_k)
    if rho_floor < 1.0:
        return TcResult(tc_kelvin=0.0, censored=True, rho_at_floor=rho_floor)
    if t_floor_k >= t_max_kelvin:
        raise RuntimeError(f"rho(T) > 1 already at the resolvable floor {t_floor_k} K >= t_max_kelvin={t_max_kelvin} K")

    # Bracket endpoints never exceed t_max_kelvin, so bisection cannot return
    # a Tc above the requested maximum.
    t_lo = t_floor_k
    t_hi = min(2.0 * t_floor_k, t_max_kelvin)
    while rho(t_hi) > 1.0:
        if t_hi >= t_max_kelvin:
            raise RuntimeError(f"rho(T) still > 1 at t_max_kelvin={t_max_kelvin} K; Tc above requested maximum")
        t_lo = t_hi
        t_hi = min(2.0 * t_hi, t_max_kelvin)

    while (t_hi - t_lo) / t_hi > rtol:
        t_mid = 0.5 * (t_lo + t_hi)
        if rho(t_mid) > 1.0:
            t_lo = t_mid
        else:
            t_hi = t_mid
    return TcResult(tc_kelvin=0.5 * (t_lo + t_hi), censored=False)


def tc_aniso(
    omega: np.ndarray,
    a2f_pairs: np.ndarray,
    weights: np.ndarray,
    mu_star: float = 0.10,
    gap_threshold_mev: float = 1e-3,
    cutoff_factor: float = 10.0,
    n_max: int = 512,
    t_lo: float | None = None,
    t_hi: float | None = None,
    t_max_kelvin: float = 2000.0,
    rtol: float = 2e-3,
    **solve_kw,
) -> float:
    """Tc [K]: highest T with a nontrivial self-consistent gap, via bisection.

    METHOD (heuristic — read before quoting Tc): unlike tc_aniso_linearized /
    the isotropic solver, which bisect on the leading eigenvalue of the
    LINEARIZED kernel, this bisects on whether the full nonlinear gap solution
    survives above `gap_threshold_mev`. Near Tc the fixed-point iteration
    slows down critically, so an un-converged transient can sit above the
    threshold: at the default settings this biases Tc HIGH by a few percent
    relative to the linearized-kernel Tc (pinned by the isotropic-limit test;
    tighten gap_threshold_mev / raise max_iter to trade accuracy against cost,
    but note that too-tight thresholds can fail to bracket). Gap values Δ(T)
    away from Tc are unaffected. Prefer tc_aniso_linearized when the number
    you need is Tc itself.

    Un-converged states are NOT taken at face value: a slowly decaying
    normal-state transient of the delta0_mev seed can otherwise be
    misclassified as superconducting near the resolvable floor — a
    categorical, seed/iteration-budget-dependent error, not a small bias
    (e.g. lambda=0.35: linearized rho_floor=0.993 says normal, yet the
    un-guarded heuristic reported ~0.76 K). Near marginal stability
    (rho ~ 1) no finite iterate can certify either direction, so ambiguous
    states are decided by the exact criterion — the linearized kernel's
    stability at that T (max_eigenvalue_aniso; see is_sc below).

    The lower bracket defaults to the resolvable Matsubara floor
    (omega_c / (2*pi*n_max), as in the isotropic solver), NOT a hardcoded
    value — otherwise sub-kelvin but resolvable Tc would be falsely reported
    as 0. Returns 0.0 only if the gap has collapsed already at the floor
    (truly normal / below the resolvable floor). A max-gap above
    gap_threshold counts as superconducting; the threshold rejects the
    trivial Δ=0 fixed point.
    """
    omega = validate_grid(omega)
    omega_c = cutoff_factor * float(omega[-1])
    t_floor_k = max(omega_c / (2.0 * np.pi * n_max) * MEV_TO_K, 1e-3)
    lo = t_floor_k if t_lo is None else t_lo

    def is_sc(t):
        st = solve_gap_at_T(omega, a2f_pairs, weights, t, mu_star=mu_star,
                            cutoff_factor=cutoff_factor, n_max=n_max, **solve_kw)
        if st.converged:
            return st.max_gap_mev > gap_threshold_mev
        # An un-converged transient proves nothing in EITHER direction: a
        # decaying normal-state remnant of the seed can sit above the
        # threshold (false positive), and a slowly growing unstable mode can
        # still sit below it when the budget runs out (false negative).
        # Decide by the exact criterion instead: stability of the Delta = 0
        # state under the linearized kernel at this T.
        return max_eigenvalue_aniso(omega, a2f_pairs, weights, t * K_TO_MEV,
                                    mu_star, omega_c, n_max) > 1.0

    if not is_sc(lo):
        return 0.0  # below the resolvable floor -> treat as normal (censored)
    if lo >= t_max_kelvin:
        raise RuntimeError(f"gap open already at the lower bracket {lo} K >= t_max_kelvin={t_max_kelvin} K")

    # Expand the upper bracket upward from the floor until the gap closes;
    # endpoints are clamped to t_max_kelvin so bisection cannot exceed it.
    hi = min(2.0 * lo if t_hi is None else t_hi, t_max_kelvin)
    while is_sc(hi):
        if hi >= t_max_kelvin:
            raise RuntimeError(f"gap still open at t_max_kelvin={t_max_kelvin} K; check input")
        lo = hi
        hi = min(2.0 * hi, t_max_kelvin)

    while (hi - lo) / hi > rtol:
        t_mid = 0.5 * (lo + hi)
        if is_sc(t_mid):
            lo = t_mid
        else:
            hi = t_mid
    return 0.5 * (lo + hi)
