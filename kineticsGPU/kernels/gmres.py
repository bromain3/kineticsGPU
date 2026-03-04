"""
gmres.py
========
Batched restarted GMRES with Givens rotations — pure CUDA/Numba.

Two strategies, matching the paper recommendations:

PER-THREAD  (pt_gmres_kernel)
  • Launch: <<<ceil(ncells/TPB), TPB>>>
  • Each thread solves ONE linear system (B, N) sequentially.
  • All temporary arrays (V, H, g, cs, sn, y) are stored in registers
    or local memory.  No shared memory, no synchronisation overhead.
  • Ideal when ncells >> nspecies and thread divergence in the Newton
    solver is limited.

PER-BLOCK   (pb_gmres_kernel)
  • Launch: <<<ncells, TPB>>>  with TPB = blockDim.x (power of 2, ≤ 1024)
  • All threads in block cooperate to solve ONE linear system.
  • Dot products and norms done via shared-memory parallel reduction.
  • matvec (J·v) is split across threads over the species dimension.
  • Recommended for large n_species where the per-thread local-memory
    footprint would exceed register limits.

Both kernels implement:
  Restarted GMRES with Givens rotations (no lstsq, pure back-sub).
  The linear system solved is:  (I + h * J) · dx = -F
  where J is provided via a callback pattern (device-function pointer
  is not directly supported by Numba, so we use a flag to switch between
  chemistry types at compile time — see NumbaSolver below).

Since Numba does not support function pointers in CUDA kernels, the
matvec is computed externally (one call to pt_jvp_* / pb_jvp_*) and
stored in a preallocated device buffer `Av`.  The GMRES kernels then
only perform the Krylov subspace construction on that buffer.

High-level flow (per Newton step):
  1. Compute F (residual)  — rhs kernel
  2. For each GMRES iteration:
       a. Call jvp kernel(s) to fill `Av`  (the matvec A·v)
       b. Orthogonalise via modified Gram-Schmidt
       c. Apply Givens rotations
  3. Back-solve triangular system
  4. Update solution xk += V · y

This split (jvp in one kernel, Krylov bookkeeping in another) keeps the
kernels short and compilable, and allows profiling each part separately.
"""

import math
import numpy as np
from numba import cuda, float64
from numba import float32

@cuda.jit(device=True)
def copysigne_GPU(a, b):
    return math.copysign(float32(a), float32(b))
# ══════════════════════════════════════════════════════════════════════════════
# §1  SHARED-MEMORY PARALLEL REDUCTION HELPER
# ══════════════════════════════════════════════════════════════════════════════
@cuda.jit(device=True)
def _block_dot(a, b, sdata, tid, TPB, N):
    """
    Compute dot(a[0:N], b[0:N]) using shared memory reduction.
    sdata must be a shared array of size TPB (float64).
    All threads in the block must call this together.
    Returns the result on ALL threads (broadcast).
    """
    acc = 0.0
    for k in range(tid, N, TPB):
        acc += a[k] * b[k]
    sdata[tid] = acc
    cuda.syncthreads()
    # parallel reduction (assumes TPB is power of 2)
    stride = TPB >> 1
    while stride > 0:
        if tid < stride:
            sdata[tid] += sdata[tid + stride]
        cuda.syncthreads()
        stride >>= 1
    return sdata[0]


@cuda.jit(device=True)
def _block_norm(a, sdata, tid, TPB, N):
    """‖a‖₂  via shared-memory reduction."""
    acc = 0.0
    for k in range(tid, N, TPB):
        acc += a[k] * a[k]
    sdata[tid] = acc
    cuda.syncthreads()
    stride = TPB >> 1
    while stride > 0:
        if tid < stride:
            sdata[tid] += sdata[tid + stride]
        cuda.syncthreads()
        stride >>= 1
    return math.sqrt(sdata[0])


# ══════════════════════════════════════════════════════════════════════════════
# §2  PER-THREAD GMRES  (all bookkeeping sequential per thread)
# ══════════════════════════════════════════════════════════════════════════════

@cuda.jit
def pt_gmres_step(
    # Current Newton state (input/output)
    xk,        # (ncells, N)  current iterate, updated in-place
    F,         # (ncells, N)  residual  = h*f(xk) - (xk - x0)  (precomputed)
    # Krylov basis buffers (preallocated, workspace only)
    V,         # (ncells, restart+1, N)
    H,         # (ncells, restart+1, restart)
    cs,        # (ncells, restart)
    sn,        # (ncells, restart)
    g_vec,     # (ncells, restart+1)
    # matvec buffer: caller must fill Av[:,j,:] = A * V[:,j,:] BEFORE calling
    # this kernel for each j; here we only read from it.
    Av,        # (ncells, N)  filled externally per Arnoldi step
    j_iter,    # scalar int: current Arnoldi column (0-based)
    # output: new basis vector (will be stored in V[:,j+1,:] by caller)
    beta,      # (ncells,)  current ‖r‖  (updated at restart)
    tol,       # scalar float
    N,         # system size
    do_backsolve,  # int: 1 = perform back-solve + update x, 0 = Arnoldi only
    restart,
):
    """
    Per-thread GMRES Arnoldi step + optional back-solve.

    Usage pattern from Python:
      1. Before restart loop: initialise V[:,0,:] = r/‖r‖, g_vec[:,0] = ‖r‖
      2. For j in range(restart):
           a. Fill Av = A * V[:,j,:]   (call jvp kernel externally)
           b. Call this kernel with j_iter=j, do_backsolve=0
              → orthogonalises w, applies Givens, stores V[:,j+1,:]
         c. Check convergence via g_vec[:,j+1]
      3. Call this kernel once more with do_backsolve=1
         → back-solves H·y = g and updates xk

    This design avoids re-launching a huge kernel and lets Python control
    the outer loop, which is fine because each call is short.
    """
    cell = cuda.grid(1)
    if cell >= xk.shape[0]:
        return

    j = j_iter
    m = restart

    # ── Arnoldi: orthogonalise Av against V[:,0:j+1,:] ──────────────────────
    if not do_backsolve:
        # w = Av (local copy via direct indexing — no local scratch needed)
        for i in range(j + 1):
            # h_ij = dot(Av, V[:,i,:])
            h_ij = 0.0
            for k in range(N):
                h_ij += Av[cell, k] * V[cell, i, k]
            H[cell, i, j] = h_ij
            # w -= h_ij * V[:,i,:]
            for k in range(N):
                Av[cell, k] -= h_ij * V[cell, i, k]

        # h_{j+1,j} = ‖w‖
        norm_w = 0.0
        for k in range(N):
            norm_w += Av[cell, k] * Av[cell, k]
        norm_w = math.sqrt(norm_w)
        H[cell, j + 1, j] = norm_w

        if norm_w > 0.0:
            inv_norm = 1.0 / norm_w
            for k in range(N):
                V[cell, j + 1, k] = Av[cell, k] * inv_norm
        else:
            for k in range(N):
                V[cell, j + 1, k] = 0.0

        # Apply previous Givens rotations to H[:,0:j,j]
        for i in range(j):
            h_ij_old = H[cell, i,   j]
            h_i1j    = H[cell, i+1, j]
            c_i = cs[cell, i]
            s_i = sn[cell, i]
            H[cell, i,   j] =  c_i * h_ij_old + s_i * h_i1j
            H[cell, i+1, j] = -s_i * h_ij_old + c_i * h_i1j

        # New Givens rotation for (j, j+1)
        h_jj  = H[cell, j,   j]
        h_j1j = H[cell, j+1, j]
        r_diag = math.sqrt(h_jj*h_jj + h_j1j*h_j1j)
        if r_diag > 0.0:
            c_new = h_jj  / r_diag
            s_new = h_j1j / r_diag
        else:
            c_new = 1.0; s_new = 0.0
        cs[cell, j] = c_new
        sn[cell, j] = s_new
        H[cell, j,   j] = r_diag
        H[cell, j+1, j] = 0.0
        # rotate g
        g_j        = g_vec[cell, j]
        g_vec[cell, j]   =  c_new * g_j
        g_vec[cell, j+1] = -s_new * g_j

    # ── Back-solve and solution update ───────────────────────────────────────
    else:
        # solve upper-triangular H[0:m,0:m] · y = g_vec[0:m]
        # back substitution stored in g_vec (in-place) — no extra alloc
        # We use g_vec temporarily; caller should not reuse it after.
        for k in range(m - 1, -1, -1):
            val = g_vec[cell, k]
            for l in range(k + 1, m):
                val -= H[cell, k, l] * g_vec[cell, l]   # g_vec[l] holds y[l] already
            diag = H[cell, k, k]
            if abs(diag) < 1e-30:
                diag = copysigne_GPU(1e-30, diag)
            g_vec[cell, k] = val / diag

        # xk += V[0:m,:] @ y  (y stored in g_vec[0:m])
        for k in range(N):
            delta_k = 0.0
            for l in range(m):
                delta_k += V[cell, l, k] * g_vec[cell, l]
            xk[cell, k] += delta_k


# ══════════════════════════════════════════════════════════════════════════════
# §3  PER-BLOCK GMRES  (threads cooperate per cell)
# ══════════════════════════════════════════════════════════════════════════════

@cuda.jit
def pb_gmres_step(
    xk,        # (ncells, N)
    F,         # (ncells, N)
    V,         # (ncells, restart+1, N)
    H,         # (ncells, restart+1, restart)
    cs,        # (ncells, restart)
    sn,        # (ncells, restart)
    g_vec,     # (ncells, restart+1)
    Av,        # (ncells, N)  filled externally
    j_iter,
    beta,      # (ncells,)
    tol,
    N,
    do_backsolve,
    restart,
):
    """
    Per-block GMRES Arnoldi step.
    Block = one cell.  Dot products use shared memory reduction.
    """
    cell = cuda.blockIdx.x
    tid  = cuda.threadIdx.x
    TPB  = cuda.blockDim.x

    if cell >= xk.shape[0]:
        return

    sdata = cuda.shared.array(shape=0, dtype=float64)   # dynamic allocation

    j = j_iter
    m = restart

    if not do_backsolve:
        # orthogonalise Av against V[:,0:j+1,:]
        for i in range(j + 1):
            h_ij = _block_dot(Av[cell], V[cell, i], sdata, tid, TPB, N)
            if tid == 0:
                H[cell, i, j] = h_ij
            cuda.syncthreads()
            h_ij_local = H[cell, i, j]   # broadcast via global mem (single value)
            for k in range(tid, N, TPB):
                Av[cell, k] -= h_ij_local * V[cell, i, k]
            cuda.syncthreads()

        norm_w = _block_norm(Av[cell], sdata, tid, TPB, N)
        if tid == 0:
            H[cell, j+1, j] = norm_w
        cuda.syncthreads()

        inv_norm = 1.0 / norm_w if norm_w > 0.0 else 0.0
        for k in range(tid, N, TPB):
            V[cell, j+1, k] = Av[cell, k] * inv_norm
        cuda.syncthreads()

        # Givens: only thread 0 touches the scalar H entries
        if tid == 0:
            for i in range(j):
                h_ij_old = H[cell, i,   j]
                h_i1j    = H[cell, i+1, j]
                c_i = cs[cell, i]; s_i = sn[cell, i]
                H[cell, i,   j] =  c_i*h_ij_old + s_i*h_i1j
                H[cell, i+1, j] = -s_i*h_ij_old + c_i*h_i1j
            h_jj  = H[cell, j,   j]
            h_j1j = H[cell, j+1, j]
            r_d = math.sqrt(h_jj*h_jj + h_j1j*h_j1j)
            if r_d > 0.0:
                c_new = h_jj/r_d; s_new = h_j1j/r_d
            else:
                c_new = 1.0; s_new = 0.0
            cs[cell, j] = c_new; sn[cell, j] = s_new
            H[cell, j, j]   = r_d
            H[cell, j+1, j] = 0.0
            g_j = g_vec[cell, j]
            g_vec[cell, j]   =  c_new * g_j
            g_vec[cell, j+1] = -s_new * g_j
        cuda.syncthreads()

    else:
        # Back-solve: thread 0 only (scalar operations)
        if tid == 0:
            for k in range(m - 1, -1, -1):
                val = g_vec[cell, k]
                for l in range(k+1, m):
                    val -= H[cell, k, l] * g_vec[cell, l]
                d = H[cell, k, k]
                if abs(d) < 1e-30: d = copysigne_GPU(np.float32(1e-30), np.float32(d))
                g_vec[cell, k] = val / d
        cuda.syncthreads()

        # parallel update of xk
        for k in range(tid, N, TPB):
            delta_k = 0.0
            for l in range(m):
                delta_k += V[cell, l, k] * g_vec[cell, l]
            xk[cell, k] += delta_k
        cuda.syncthreads()


# ══════════════════════════════════════════════════════════════════════════════
# §4  RESIDUAL KERNEL  F = xk - x0 - h * rhs
# ══════════════════════════════════════════════════════════════════════════════

@cuda.jit
def compute_residual_kernel(
    xk,       # (ncells, N) current iterate
    x0,       # (ncells, N) state at start of step
    rhs,      # (ncells, N) f(xk)  (precomputed by chemistry kernels)
    h,        # scalar float  time-step size
    F,        # (ncells, N) output residual
    res_norms,# (ncells,)   ‖F[cell,:]‖  (output)
):
    """
    Per-thread residual:  F[cell] = xk[cell] - x0[cell] - h * rhs[cell]
    Also fills res_norms[cell] = ‖F[cell]‖₂  for convergence test.
    """
    cell = cuda.grid(1)
    if cell >= xk.shape[0]:
        return
    N = xk.shape[1]
    acc = 0.0
    for k in range(N):
        fk = xk[cell, k] - x0[cell, k] - h * rhs[cell, k]
        F[cell, k] = fk
        acc += fk * fk
    res_norms[cell] = math.sqrt(acc)


@cuda.jit
def compute_matvec_preprocess(
    # (I + h*J)·v  where J·v is provided in Jv
    # result = v + h * Jv    stored in Av
    v,    # (ncells, N)
    Jv,   # (ncells, N)   = analytic JVP output
    h,    # scalar float
    Av,   # (ncells, N)  output
):
    """
    Finalise matvec for implicit Euler Jacobian:
        A·v  =  (I + h·J)·v  =  v + h·(J·v)
    J·v  is provided by the analytic jvp kernels.
    """
    cell, k = cuda.grid(2)
    if cell < v.shape[0] and k < v.shape[1]:
        Av[cell, k] = v[cell, k] + h * Jv[cell, k]


@cuda.jit
def initialise_gmres(
    r,          # (ncells, N) initial residual  = -F
    V,          # (ncells, restart+1, N) Krylov basis (preallocated)
    g_vec,      # (ncells, restart+1)
    beta,       # (ncells,)
    N,
):
    """
    Set V[:,0,:] = r / ‖r‖   and   g_vec[:,0] = ‖r‖.
    Per-thread version.
    """
    cell = cuda.grid(1)
    if cell >= r.shape[0]:
        return
    # compute norm
    norm_r = 0.0
    for k in range(N):
        norm_r += r[cell, k] * r[cell, k]
    norm_r = math.sqrt(norm_r)
    beta[cell] = norm_r
    inv_norm = 1.0 / norm_r if norm_r > 0.0 else 0.0
    for k in range(N):
        V[cell, 0, k] = r[cell, k] * inv_norm
    g_vec[cell, 0] = norm_r
    for k in range(1, g_vec.shape[1]):
        g_vec[cell, k] = 0.0