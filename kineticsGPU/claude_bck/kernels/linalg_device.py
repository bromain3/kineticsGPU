"""
linalg_device.py  — device-side linear algebra for per-thread CUDA kernels.

Contains only pure linear-algebra primitives; no chemistry imports.

_gmres_jvp_device lives in integrator_kernel.py (not here) because it calls
_eval_jvp (chemistry_device) and _adiabatic_jvp_TC (integrator_kernel), and
Numba does not support imports inside device functions — putting it there
avoids a circular import while keeping all symbols in one place.

Public API
──────────
  _zero, _copy, _axpy, _dot, _norm
  _lu_factor_inplace, _lu_solve
  _matvec_dense
  _arnoldi_givens_step        ← shared by both GMRES variants
  _gmres_backsolve            ← shared by both GMRES variants
  _gmres_givens_device        ← dense-J GMRES (LU path only)
"""

import math
from numba import cuda, float64, int32


# ─────────────────────────────────────────────────────────────────────────────
#  Scalar / vector helpers
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _zero(v, n):
    for i in range(n):
        v[i] = 0.0


@cuda.jit(device=True, inline=True)
def _copy(src, dst, n):
    for i in range(n):
        dst[i] = src[i]


@cuda.jit(device=True, inline=True)
def _axpy(alpha, x, y, n):
    """y += alpha * x"""
    for i in range(n):
        y[i] += alpha * x[i]


@cuda.jit(device=True, inline=True)
def _dot(x, y, n):
    s = float64(0.0)
    for i in range(n):
        s += x[i] * y[i]
    return s


@cuda.jit(device=True, inline=True)
def _norm(x, n):
    s = float64(0.0)
    for i in range(n):
        s += x[i] * x[i]
    return math.sqrt(s)


# ─────────────────────────────────────────────────────────────────────────────
#  LU factorization (partial pivoting)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _lu_factor_inplace(A_flat, piv, n):
    """
    In-place LU with partial pivoting.
    A_flat : 1-D view of (n,n) row-major matrix — modified in place.
    piv    : 1-D int32 array of length n.
    """
    for k in range(n):
        max_val = math.fabs(A_flat[k * n + k])
        max_row = k
        for i in range(k + 1, n):
            v = math.fabs(A_flat[i * n + k])
            if v > max_val:
                max_val = v
                max_row = i
        piv[k] = max_row
        if max_row != k:
            for j in range(n):
                tmp = A_flat[k * n + j]
                A_flat[k * n + j] = A_flat[max_row * n + j]
                A_flat[max_row * n + j] = tmp
        if math.fabs(A_flat[k * n + k]) > 1e-300:
            inv_pivot = 1.0 / A_flat[k * n + k]
            for i in range(k + 1, n):
                A_flat[i * n + k] *= inv_pivot
                for j in range(k + 1, n):
                    A_flat[i * n + j] -= A_flat[i * n + k] * A_flat[k * n + j]


@cuda.jit(device=True)
def _lu_solve(LU_flat, piv, rhs, x, n):
    """
    Solve LU·x = rhs using pre-factored LU and pivot vector.
    rhs, x : 1-D arrays of length n  (rhs is not modified).
    """
    for i in range(n):
        x[i] = rhs[i]
    for k in range(n):
        p = piv[k]
        if p != k:
            tmp  = x[k]
            x[k] = x[p]
            x[p] = tmp
    for i in range(1, n):
        s = x[i]
        for j in range(i):
            s -= LU_flat[i * n + j] * x[j]
        x[i] = s
    for i in range(n - 1, -1, -1):
        s = x[i]
        for j in range(i + 1, n):
            s -= LU_flat[i * n + j] * x[j]
        d = LU_flat[i * n + i]
        x[i] = s / d if math.fabs(d) > 1e-300 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Dense matrix-vector product  w = J_flat · v
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _matvec_dense(J_flat, v, w, n):
    """w[i] = sum_j J_flat[i*n + j] * v[j]"""
    for i in range(n):
        s = float64(0.0)
        base = i * n
        for j in range(n):
            s += J_flat[base + j] * v[j]
        w[i] = s


# ─────────────────────────────────────────────────────────────────────────────
#  Arnoldi + Givens step
#
#  Shared by _gmres_givens_device (below) and _gmres_jvp_device (integrator_kernel.py).
#  On entry : gmres_w  holds  A·V[j]  (matvec result, filled by caller).
#  On exit  : V[j+1] normalised, H/cs/sn/g updated.
#  Returns  : |g[j+1]|  (Givens-rotated residual estimate).
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _arnoldi_givens_step(gmres_V, gmres_H, gmres_cs, gmres_sn, gmres_g,
                         gmres_w, j, n, R):
    # Gram-Schmidt orthogonalisation
    for i in range(j + 1):
        vi_base = i * n
        h_ij = float64(0.0)
        for k in range(n):
            h_ij += gmres_w[k] * gmres_V[vi_base + k]
        gmres_H[i * R + j] = h_ij
        for k in range(n):
            gmres_w[k] -= h_ij * gmres_V[vi_base + k]

    # h_{j+1,j} = ||w||
    h_j1j = float64(0.0)
    for k in range(n):
        h_j1j += gmres_w[k] * gmres_w[k]
    h_j1j = math.sqrt(h_j1j)
    gmres_H[(j + 1) * R + j] = h_j1j

    # Normalise → V[j+1]
    vj1_base = (j + 1) * n
    if h_j1j > 1e-300:
        inv_h = float64(1.0) / h_j1j
        for k in range(n):
            gmres_V[vj1_base + k] = gmres_w[k] * inv_h
    else:
        for k in range(n):
            gmres_V[vj1_base + k] = float64(0.0)

    # Apply previous Givens rotations to column j
    for i in range(j):
        h_i  = gmres_H[i * R + j]
        h_i1 = gmres_H[(i + 1) * R + j]
        c    = gmres_cs[i]
        s    = gmres_sn[i]
        gmres_H[i * R + j]       =  c * h_i + s * h_i1
        gmres_H[(i + 1) * R + j] = -s * h_i + c * h_i1

    # New Givens rotation to eliminate H[j+1, j]
    h_jj   = gmres_H[j * R + j]
    h_j1j2 = gmres_H[(j + 1) * R + j]
    r_diag = math.sqrt(h_jj * h_jj + h_j1j2 * h_j1j2)
    if r_diag > 1e-300:
        c_new = h_jj   / r_diag
        s_new = h_j1j2 / r_diag
    else:
        c_new = float64(1.0)
        s_new = float64(0.0)

    gmres_cs[j] = c_new
    gmres_sn[j] = s_new
    gmres_H[j * R + j]       = r_diag
    gmres_H[(j + 1) * R + j] = float64(0.0)

    g_j            = gmres_g[j]
    gmres_g[j]     =  c_new * g_j
    gmres_g[j + 1] = -s_new * g_j

    return math.fabs(gmres_g[j + 1])


# ─────────────────────────────────────────────────────────────────────────────
#  Back-substitution  (shared by both GMRES variants)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _gmres_backsolve(gmres_H, gmres_g, gmres_y, m, R):
    for k in range(m - 1, -1, -1):
        s = gmres_g[k]
        for l in range(k + 1, m):
            s -= gmres_H[k * R + l] * gmres_y[l]
        d = gmres_H[k * R + k]
        gmres_y[k] = s / d if math.fabs(d) > 1e-300 else float64(0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Restarted GMRES — dense-matvec variant
#  Used when the explicit Jacobian J_flat is already assembled (LU path).
#  Interface identical to v4.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _gmres_givens_device(
    J_flat,        # (n*n,)   — Jacobian row-major, READ-ONLY
    rhs,           # (n,)     — right-hand side (-F)
    dx,            # (n,)     — solution, zeroed by caller
    gmres_V,       # ((R+1)*n,)
    gmres_H,       # ((R+1)*R,)
    gmres_cs,      # (R,)
    gmres_sn,      # (R,)
    gmres_g,       # (R+1,)
    gmres_w,       # (n,)
    gmres_y,       # (R,)
    n,
    restart,
    gmres_tol,
    gmres_maxiter,
):
    beta = float64(0.0)
    for i in range(n):
        beta += rhs[i] * rhs[i]
    beta = math.sqrt(beta)
    if beta < gmres_tol:
        return

    R = restart

    for outer in range(gmres_maxiter):
        for idx in range((R + 1) * R):
            gmres_H[idx] = float64(0.0)
        for idx in range(R):
            gmres_cs[idx] = float64(0.0)
            gmres_sn[idx] = float64(0.0)
        for idx in range(R + 1):
            gmres_g[idx] = float64(0.0)

        if outer == 0:
            inv_beta = float64(1.0) / beta
            for i in range(n):
                gmres_V[i] = rhs[i] * inv_beta
        else:
            _matvec_dense(J_flat, dx, gmres_w, n)
            beta = float64(0.0)
            for i in range(n):
                ri = rhs[i] - gmres_w[i]
                gmres_w[i] = ri
                beta += ri * ri
            beta = math.sqrt(beta)
            if beta < gmres_tol:
                return
            inv_beta = float64(1.0) / beta
            for i in range(n):
                gmres_V[i] = gmres_w[i] * inv_beta

        gmres_g[0] = beta
        j_converged = R

        for j in range(R):
            vj_base = j * n
            _matvec_dense(J_flat, gmres_V[vj_base:vj_base + n], gmres_w, n)

            res_est = _arnoldi_givens_step(
                gmres_V, gmres_H, gmres_cs, gmres_sn, gmres_g, gmres_w,
                j, n, R,
            )
            if res_est < gmres_tol:
                j_converged = j + 1
                break

        m = j_converged
        _gmres_backsolve(gmres_H, gmres_g, gmres_y, m, R)

        for i in range(n):
            acc = float64(0.0)
            for k in range(m):
                acc += gmres_V[k * n + i] * gmres_y[k]
            dx[i] += acc

        _matvec_dense(J_flat, dx, gmres_w, n)
        res_true = float64(0.0)
        for i in range(n):
            ri = rhs[i] - gmres_w[i]
            res_true += ri * ri
        res_true = math.sqrt(res_true)

        if res_true < gmres_tol or j_converged < R:
            return

        beta = res_true