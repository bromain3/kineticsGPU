"""
linalg_device.py  — device-side linear algebra for per-thread CUDA kernels.

_gmres_givens_device     : restarted GMRES using a pre-assembled dense J (LU path).
_gmres_jvp_device        : restarted GMRES using only JVP calls — never builds J.
    The matvec is  w = v - h * J(xk)·v  computed analytically via _eval_jvp +
    _adiabatic_jvp_TC.  This is the correct GMRES formulation: one JVP per
    Krylov vector, no Jacobian assembly.

ALL working arrays (V, H, cs, sn, g, w, y) are 1-D slices of
pre-allocated 2D global memory buffers (shape (B, stride)).

Existing helpers (unchanged): _lu_factor_inplace, _lu_solve,
    _dot, _norm, _axpy, _copy, _zero, _matvec_dense, _gmres_givens_device
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
#  (kept for _gmres_givens_device which is still used by the LU path diagnostics
#   and any caller that has already built the explicit Jacobian)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _matvec_dense(J_flat, v, w, n):
    """
    w[i] = sum_j J_flat[i*n + j] * v[j]
    All arrays are 1-D slices of global memory — no local arrays.
    """
    for i in range(n):
        s = float64(0.0)
        base = i * n
        for j in range(n):
            s += J_flat[base + j] * v[j]
        w[i] = s


# ─────────────────────────────────────────────────────────────────────────────
#  Shared Arnoldi + Givens inner loop
#  Called by both GMRES variants; the only difference is how w = A·v is computed.
#  This helper receives w already filled by the caller.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _arnoldi_givens_step(
    gmres_V, gmres_H, gmres_cs, gmres_sn, gmres_g, gmres_w,
    j, n, R,
):
    """
    One Arnoldi step + incremental Givens rotation.
    On entry  : gmres_w  holds  A·V[j]  (the matvec result).
    On exit   : gmres_V[j+1] is the new basis vector,
                gmres_H, gmres_cs, gmres_sn, gmres_g are updated.
    Returns   : |g[j+1]|  (the Givens-rotated residual estimate).
    """
    vj_base = j * n

    # Gram-Schmidt: project w ⊥ V[0..j]
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
        h_i   = gmres_H[i * R + j]
        h_i1  = gmres_H[(i + 1) * R + j]
        c     = gmres_cs[i]
        s     = gmres_sn[i]
        gmres_H[i * R + j]       =  c * h_i + s * h_i1
        gmres_H[(i + 1) * R + j] = -s * h_i + c * h_i1

    # New Givens rotation
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
#  Back-substitution shared by both GMRES variants
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
#  Restarted GMRES — dense-matvec variant (used by LU-path callers that already
#  have the explicit Jacobian assembled in J_flat).
#  Interface unchanged from v4.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _gmres_givens_device(
    J_flat,        # (n*n,)      global — Jacobian row-major, READ-ONLY
    rhs,           # (n,)        local  — right-hand side  (-F)
    dx,            # (n,)        global — solution (zeroed by caller)
    gmres_V,       # ((R+1)*n,)  global
    gmres_H,       # ((R+1)*R,)  global
    gmres_cs,      # (R,)        global
    gmres_sn,      # (R,)        global
    gmres_g,       # (R+1,)      global
    gmres_w,       # (n,)        global
    gmres_y,       # (R,)        global
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


# ─────────────────────────────────────────────────────────────────────────────
#  Restarted GMRES — JVP variant  (true matrix-free GMRES)
#
#  The matvec is  w = v - h * J(xk)·v  computed via the analytical JVP
#  functions _eval_jvp + _adiabatic_jvp_TC.
#  No Jacobian matrix is ever assembled or read.
#
#  Caller contract (same buffer shapes as _gmres_givens_device):
#    • dx zeroed before call.
#    • All gmres_X are 1-D slices of shape (stride,) from global 2-D buffers.
#    • T_loc, C_loc reflect the current Newton iterate xk.
#    • The chemistry context buffers (ctx_*) are already filled for xk by
#      the preceding _eval_rhs_and_context call in the Newton loop.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _gmres_jvp_device(
    # Newton iterate state
    T_loc,         # float64 scalar — current temperature
    C_loc,         # (n_species,)  — current concentrations (local array)
    h,             # float64 scalar — time step
    # Chemistry context (already evaluated for xk)
    ctx_kf_e, ctx_Kc_e, ctx_conc_r_e, ctx_conc_p_e,
    ctx_deltaG0_e, ctx_s_r_e, ctx_s_p_e,
    ctx_kf_tb, ctx_Kc_tb, ctx_conc_r_tb, ctx_conc_p_tb,
    ctx_baseRR_tb, ctx_M_tb,
    ctx_deltaG0_tb, ctx_s_r_tb, ctx_s_p_tb,
    ctx_kf_fall, ctx_Kc_fall, ctx_conc_r_fall, ctx_conc_p_fall,
    ctx_k0_fall, ctx_kinf_fall, ctx_M_fall,
    ctx_Pr_fall, ctx_F_fall, ctx_kbase_fall,
    ctx_Fcent_fall, ctx_logFcent_fall,
    ctx_C_par_fall, ctx_N_par_fall, ctx_f1_fall, ctx_exp_fall,
    ctx_deltaG0_fall, ctx_s_r_fall, ctx_s_p_fall,
    omega_loc,     # (n_species,)  — omega already evaluated for xk
    # Mechanism arrays
    n_species,
    thermo_coef_low, thermo_coef_high, thermo_Tmid,
    n_e, e_A, e_n, e_Ea, e_oneWay, e_delta_n,
    e_react_idx, e_react_sto, e_react_offset,
    e_prod_idx, e_prod_sto, e_prod_offset,
    e_idx_r_scatter, e_idx_p_scatter,
    n_tb, tb_A, tb_n, tb_Ea, tb_oneWay, tb_delta_n,
    tb_react_idx, tb_react_sto, tb_react_offset,
    tb_prod_idx, tb_prod_sto, tb_prod_offset,
    tb_idx_r_scatter, tb_idx_p_scatter, tb_eff_matrix,
    n_fall, fall_A_k0, fall_b_k0, fall_Ea_k0,
    fall_A_kinf, fall_b_kinf, fall_Ea_kinf,
    fall_oneWay, fall_delta_n,
    fall_troe_A, fall_troe_T1, fall_troe_T2, fall_troe_T3, fall_is_troe,
    fall_react_idx, fall_react_sto, fall_react_offset,
    fall_prod_idx, fall_prod_sto, fall_prod_offset,
    fall_idx_r_scatter, fall_idx_p_scatter, fall_eff_matrix,
    # GMRES buffers (1-D slices of 2-D global arrays)
    rhs,           # (n_state,)    — right-hand side (-F), READ-ONLY
    dx,            # (n_state,)    — solution, zeroed by caller
    gmres_V,       # ((R+1)*n,)
    gmres_H,       # ((R+1)*R,)
    gmres_cs,      # (R,)
    gmres_sn,      # (R,)
    gmres_g,       # (R+1,)
    gmres_w,       # (n_state,)    — Arnoldi work vector (also used as Jv_TC)
    gmres_y,       # (R,)
    # Sizes and tolerances
    n_state,
    restart,
    gmres_tol,
    gmres_maxiter,
):
    """
    Matrix-free restarted GMRES.
    The implicit system is  (I - h·J)·dx = rhs.
    Each matvec computes  w = v - h·J·v  via one analytical JVP call,
    exactly as done in the PyTorch version with make_matvec_user_defined_context.
    """
    from .linalg_device import _arnoldi_givens_step, _gmres_backsolve
    from .chemistry_device import _eval_jvp
    from ..kernels.integrator_kernel import _adiabatic_jvp_TC

    # These imports resolve at JIT compile time (Numba inlines device functions).
    # They are declared here for readability; the actual references are at module
    # level in integrator_kernel.py which already imports both symbols.

    beta = float64(0.0)
    for i in range(n_state):
        beta += rhs[i] * rhs[i]
    beta = math.sqrt(beta)
    if beta < gmres_tol:
        return

    R = restart

    # Temporary storage for the Jv intermediate — we reuse gmres_w for Jv_TC
    # and need a separate slice for Jv_omega.  We use the first n_species
    # elements of gmres_y as scratch (gmres_y is (R,) with R >= n_state > n_species
    # in all practical cases; if R < n_species the caller must ensure a separate
    # buffer).  To be safe we use gmres_V[(R)*n_state : (R)*n_state + n_species]
    # as the Jv_omega scratch (it is the V[R] slot which is never read after
    # the Arnoldi loop ends).
    Jv_omega_base = R * n_state   # offset into gmres_V for the scratch slot

    for outer in range(gmres_maxiter):
        for idx in range((R + 1) * R):
            gmres_H[idx] = float64(0.0)
        for idx in range(R):
            gmres_cs[idx] = float64(0.0)
            gmres_sn[idx] = float64(0.0)
        for idx in range(R + 1):
            gmres_g[idx] = float64(0.0)

        # ── Compute initial residual r = rhs - (I - h·J)·dx ──────────────
        if outer == 0:
            # dx = 0 → r = rhs
            inv_beta = float64(1.0) / beta
            for i in range(n_state):
                gmres_V[i] = rhs[i] * inv_beta
        else:
            # r = rhs - (dx - h·J·dx) = rhs - dx + h·J·dx
            # Compute (I - h·J)·dx via JVP
            v_T   = dx[0]
            for s in range(n_state - 1):
                gmres_V[Jv_omega_base + s] = dx[s + 1]   # reuse as v_C scratch

            _eval_jvp(
                T_loc, C_loc, v_T, gmres_V[Jv_omega_base:Jv_omega_base + n_state - 1],
                ctx_kf_e, ctx_Kc_e, ctx_conc_r_e, ctx_conc_p_e,
                ctx_deltaG0_e, ctx_s_r_e, ctx_s_p_e,
                ctx_kf_tb, ctx_Kc_tb, ctx_conc_r_tb, ctx_conc_p_tb,
                ctx_baseRR_tb, ctx_M_tb,
                ctx_deltaG0_tb, ctx_s_r_tb, ctx_s_p_tb,
                ctx_kf_fall, ctx_Kc_fall, ctx_conc_r_fall, ctx_conc_p_fall,
                ctx_k0_fall, ctx_kinf_fall, ctx_M_fall,
                ctx_Pr_fall, ctx_F_fall, ctx_kbase_fall,
                ctx_Fcent_fall, ctx_logFcent_fall,
                ctx_C_par_fall, ctx_N_par_fall, ctx_f1_fall, ctx_exp_fall,
                ctx_deltaG0_fall, ctx_s_r_fall, ctx_s_p_fall,
                gmres_V[Jv_omega_base:Jv_omega_base + n_state],  # Jv_omega output
                n_species,
                thermo_coef_low, thermo_coef_high, thermo_Tmid,
                n_e, e_A, e_n, e_Ea, e_oneWay, e_delta_n,
                e_react_idx, e_react_sto, e_react_offset,
                e_prod_idx, e_prod_sto, e_prod_offset,
                e_idx_r_scatter, e_idx_p_scatter,
                n_tb, tb_A, tb_n, tb_Ea, tb_oneWay, tb_delta_n,
                tb_react_idx, tb_react_sto, tb_react_offset,
                tb_prod_idx, tb_prod_sto, tb_prod_offset,
                tb_idx_r_scatter, tb_idx_p_scatter, tb_eff_matrix,
                n_fall, fall_A_k0, fall_b_k0, fall_Ea_k0,
                fall_A_kinf, fall_b_kinf, fall_Ea_kinf,
                fall_oneWay, fall_delta_n,
                fall_troe_A, fall_troe_T1, fall_troe_T2, fall_troe_T3, fall_is_troe,
                fall_react_idx, fall_react_sto, fall_react_offset,
                fall_prod_idx, fall_prod_sto, fall_prod_offset,
                fall_idx_r_scatter, fall_idx_p_scatter, fall_eff_matrix,
            )
            # gmres_w = Jv_TC  (complete JVP including temperature equation)
            _adiabatic_jvp_TC(
                T_loc, C_loc, v_T, gmres_V[Jv_omega_base:Jv_omega_base + n_state - 1],
                omega_loc, gmres_V[Jv_omega_base:Jv_omega_base + n_state],
                n_species,
                thermo_coef_low, thermo_coef_high, thermo_Tmid,
                gmres_w,   # output: Jv_TC
            )
            # r = rhs - (dx - h*Jv_TC) = rhs - dx + h*gmres_w
            beta = float64(0.0)
            for i in range(n_state):
                ri = rhs[i] - dx[i] + h * gmres_w[i]
                gmres_w[i] = ri
                beta += ri * ri
            beta = math.sqrt(beta)
            if beta < gmres_tol:
                return
            inv_beta = float64(1.0) / beta
            for i in range(n_state):
                gmres_V[i] = gmres_w[i] * inv_beta

        gmres_g[0] = beta
        j_converged = R

        # ── Arnoldi loop ──────────────────────────────────────────────────
        for j in range(R):
            vj_base = j * n_state

            # ── matvec: w = v_j - h * J · v_j  (one analytical JVP) ──────
            v_T = gmres_V[vj_base]
            for s in range(n_state - 1):
                gmres_V[Jv_omega_base + s] = gmres_V[vj_base + s + 1]

            _eval_jvp(
                T_loc, C_loc,
                v_T, gmres_V[Jv_omega_base:Jv_omega_base + n_state - 1],
                ctx_kf_e, ctx_Kc_e, ctx_conc_r_e, ctx_conc_p_e,
                ctx_deltaG0_e, ctx_s_r_e, ctx_s_p_e,
                ctx_kf_tb, ctx_Kc_tb, ctx_conc_r_tb, ctx_conc_p_tb,
                ctx_baseRR_tb, ctx_M_tb,
                ctx_deltaG0_tb, ctx_s_r_tb, ctx_s_p_tb,
                ctx_kf_fall, ctx_Kc_fall, ctx_conc_r_fall, ctx_conc_p_fall,
                ctx_k0_fall, ctx_kinf_fall, ctx_M_fall,
                ctx_Pr_fall, ctx_F_fall, ctx_kbase_fall,
                ctx_Fcent_fall, ctx_logFcent_fall,
                ctx_C_par_fall, ctx_N_par_fall, ctx_f1_fall, ctx_exp_fall,
                ctx_deltaG0_fall, ctx_s_r_fall, ctx_s_p_fall,
                gmres_V[Jv_omega_base:Jv_omega_base + n_state],  # Jv_omega
                n_species,
                thermo_coef_low, thermo_coef_high, thermo_Tmid,
                n_e, e_A, e_n, e_Ea, e_oneWay, e_delta_n,
                e_react_idx, e_react_sto, e_react_offset,
                e_prod_idx, e_prod_sto, e_prod_offset,
                e_idx_r_scatter, e_idx_p_scatter,
                n_tb, tb_A, tb_n, tb_Ea, tb_oneWay, tb_delta_n,
                tb_react_idx, tb_react_sto, tb_react_offset,
                tb_prod_idx, tb_prod_sto, tb_prod_offset,
                tb_idx_r_scatter, tb_idx_p_scatter, tb_eff_matrix,
                n_fall, fall_A_k0, fall_b_k0, fall_Ea_k0,
                fall_A_kinf, fall_b_kinf, fall_Ea_kinf,
                fall_oneWay, fall_delta_n,
                fall_troe_A, fall_troe_T1, fall_troe_T2, fall_troe_T3, fall_is_troe,
                fall_react_idx, fall_react_sto, fall_react_offset,
                fall_prod_idx, fall_prod_sto, fall_prod_offset,
                fall_idx_r_scatter, fall_idx_p_scatter, fall_eff_matrix,
            )
            _adiabatic_jvp_TC(
                T_loc, C_loc,
                v_T, gmres_V[Jv_omega_base:Jv_omega_base + n_state - 1],
                omega_loc, gmres_V[Jv_omega_base:Jv_omega_base + n_state],
                n_species,
                thermo_coef_low, thermo_coef_high, thermo_Tmid,
                gmres_w,   # Jv_TC output → used immediately below
            )
            # w = v - h * Jv_TC  (the implicit system matvec (I - h·J)·v)
            for i in range(n_state):
                gmres_w[i] = gmres_V[vj_base + i] - h * gmres_w[i]

            # Arnoldi orthogonalisation + Givens rotation (shared helper)
            res_est = _arnoldi_givens_step(
                gmres_V, gmres_H, gmres_cs, gmres_sn, gmres_g, gmres_w,
                j, n_state, R,
            )
            if res_est < gmres_tol:
                j_converged = j + 1
                break

        # ── Back-substitution ─────────────────────────────────────────────
        m = j_converged
        _gmres_backsolve(gmres_H, gmres_g, gmres_y, m, R)

        # ── dx += V[:, 0:m] · y ───────────────────────────────────────────
        for i in range(n_state):
            acc = float64(0.0)
            for k in range(m):
                acc += gmres_V[k * n_state + i] * gmres_y[k]
            dx[i] += acc

        # ── True residual via JVP ─────────────────────────────────────────
        v_T = dx[0]
        for s in range(n_state - 1):
            gmres_V[Jv_omega_base + s] = dx[s + 1]

        _eval_jvp(
            T_loc, C_loc,
            v_T, gmres_V[Jv_omega_base:Jv_omega_base + n_state - 1],
            ctx_kf_e, ctx_Kc_e, ctx_conc_r_e, ctx_conc_p_e,
            ctx_deltaG0_e, ctx_s_r_e, ctx_s_p_e,
            ctx_kf_tb, ctx_Kc_tb, ctx_conc_r_tb, ctx_conc_p_tb,
            ctx_baseRR_tb, ctx_M_tb,
            ctx_deltaG0_tb, ctx_s_r_tb, ctx_s_p_tb,
            ctx_kf_fall, ctx_Kc_fall, ctx_conc_r_fall, ctx_conc_p_fall,
            ctx_k0_fall, ctx_kinf_fall, ctx_M_fall,
            ctx_Pr_fall, ctx_F_fall, ctx_kbase_fall,
            ctx_Fcent_fall, ctx_logFcent_fall,
            ctx_C_par_fall, ctx_N_par_fall, ctx_f1_fall, ctx_exp_fall,
            ctx_deltaG0_fall, ctx_s_r_fall, ctx_s_p_fall,
            gmres_V[Jv_omega_base:Jv_omega_base + n_state],
            n_species,
            thermo_coef_low, thermo_coef_high, thermo_Tmid,
            n_e, e_A, e_n, e_Ea, e_oneWay, e_delta_n,
            e_react_idx, e_react_sto, e_react_offset,
            e_prod_idx, e_prod_sto, e_prod_offset,
            e_idx_r_scatter, e_idx_p_scatter,
            n_tb, tb_A, tb_n, tb_Ea, tb_oneWay, tb_delta_n,
            tb_react_idx, tb_react_sto, tb_react_offset,
            tb_prod_idx, tb_prod_sto, tb_prod_offset,
            tb_idx_r_scatter, tb_idx_p_scatter, tb_eff_matrix,
            n_fall, fall_A_k0, fall_b_k0, fall_Ea_k0,
            fall_A_kinf, fall_b_kinf, fall_Ea_kinf,
            fall_oneWay, fall_delta_n,
            fall_troe_A, fall_troe_T1, fall_troe_T2, fall_troe_T3, fall_is_troe,
            fall_react_idx, fall_react_sto, fall_react_offset,
            fall_prod_idx, fall_prod_sto, fall_prod_offset,
            fall_idx_r_scatter, fall_idx_p_scatter, fall_eff_matrix,
        )
        _adiabatic_jvp_TC(
            T_loc, C_loc,
            v_T, gmres_V[Jv_omega_base:Jv_omega_base + n_state - 1],
            omega_loc, gmres_V[Jv_omega_base:Jv_omega_base + n_state],
            n_species,
            thermo_coef_low, thermo_coef_high, thermo_Tmid,
            gmres_w,
        )
        res_true = float64(0.0)
        for i in range(n_state):
            ri = rhs[i] - (dx[i] - h * gmres_w[i])
            res_true += ri * ri
        res_true = math.sqrt(res_true)

        if res_true < gmres_tol or j_converged < R:
            return

        beta = res_true