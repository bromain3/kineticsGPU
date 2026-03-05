"""
Per-thread linear algebra device functions for implicit integration.

Two strategies:
  1. Dense LU decomposition (Doolittle, in-place) – recommended for N≤60.
     Operates entirely on local arrays (registers / local memory).
     No thread communication needed.

  2. Restarted GMRES with Givens rotations – provided as alternative for
     larger systems or when memory is tight (avoids N×N Jacobian storage).

All routines are @cuda.jit(device=True) and operate on local arrays passed
by reference (numba local arrays are on-thread stack / local memory).

Memory note (per Stone & Davis 2013):
  - For N=54 (GRI30 + T): LU needs N² = 2916 float64 = 23 KB local memory.
  - GMRES(m=20): V(21×54) + H(21×20) + cs,sn(20) + g(21) ≈ 12 KB local memory.
  - Both fit within per-thread local memory (several MB available, albeit slow).
  - Critical inner loops are register-bound: numba's register allocator will
    spill to local memory for large N.
"""

import math
from numba import cuda, float64, int32
import numba


# ─────────────────────────────────────────────────────────────────────────────
#  Dense LU (Doolittle, partial pivoting) – per-thread
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _lu_factor_inplace(A, piv, n):
    """
    In-place LU factorisation with partial pivoting (Doolittle).
    A: (n*n,) row-major flat array, modified in-place.
    piv: (n,) pivot index array, filled in-place.
    Returns: 0 on success, 1 if singular.
    """
    for k in range(n):
        # Find pivot row
        max_val = abs(A[k*n + k])
        max_row = k
        for i in range(k+1, n):
            v = abs(A[i*n + k])
            if v > max_val:
                max_val = v
                max_row = i
        piv[k] = max_row
        # Swap rows k and max_row
        if max_row != k:
            for j in range(n):
                tmp = A[k*n + j]
                A[k*n + j] = A[max_row*n + j]
                A[max_row*n + j] = tmp
        # Singular check
        if A[k*n + k] == 0.0:
            return 1
        inv_akk = 1.0 / A[k*n + k]
        # Elimination
        for i in range(k+1, n):
            factor = A[i*n + k] * inv_akk
            A[i*n + k] = factor          # store L factor
            for j in range(k+1, n):
                A[i*n + j] -= factor * A[k*n + j]
    return 0


@cuda.jit(device=True, inline=True)
def _lu_solve(LU, piv, b, x, n):
    """
    Solves LU·x = b using factored LU and pivot array.
    b: (n,) right-hand side (NOT modified).
    x: (n,) solution output.
    """
    # Forward substitution with row permutation (apply pivots to b)
    for i in range(n):
        x[i] = b[i]
    for k in range(n):
        p = piv[k]
        if p != k:
            tmp = x[k]; x[k] = x[p]; x[p] = tmp
        for j in range(k):
            x[k] -= LU[k*n + j] * x[j]
    # Back substitution
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            x[i] -= LU[i*n + j] * x[j]
        x[i] /= LU[i*n + i]


# ─────────────────────────────────────────────────────────────────────────────
#  GMRES with Givens rotations – per-thread, matrix-free
#  (used when dense Jacobian is too large or not available)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _dot(a, b, n):
    s = 0.0
    for i in range(n): s += a[i] * b[i]
    return s

@cuda.jit(device=True, inline=True)
def _norm(a, n):
    s = 0.0
    for i in range(n): s += a[i] * a[i]
    return math.sqrt(s)

@cuda.jit(device=True, inline=True)
def _axpy(alpha, x, y, n):
    """y += alpha * x"""
    for i in range(n): y[i] += alpha * x[i]

@cuda.jit(device=True, inline=True)
def _copy(src, dst, n):
    for i in range(n): dst[i] = src[i]

@cuda.jit(device=True, inline=True)
def _zero(a, n):
    for i in range(n): a[i] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Dense Jacobian assembly via finite differences (used once per Newton restart)
#  Builds J = (I - h * ∂f/∂TC) column by column using the JVP device function.
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: The actual Jacobian assembly function is in the integrator kernel
# because it needs access to all mechanism arrays. It is not defined here
# to avoid circular imports between device function files.
# The pattern is:
#   for col in range(N):
#       e_col = 0 everywhere, 1 at col
#       Jv_col = jvp(ctx, T, C, v_T=e_col[0], v_C=e_col[1:])
#       J_col  = e_col - h * Jv_col        # (I - h*∂f/∂TC) · e_col
#       store in J_flat[row*N + col]