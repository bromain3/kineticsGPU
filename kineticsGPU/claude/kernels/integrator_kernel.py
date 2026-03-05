"""
Per-thread backward Euler integrator CUDA kernel.

One CUDA thread = one reactor (IVP).
Block size: 64 threads (8 blocks/SM per literature recommendation).

Integration algorithm per thread:
  1. Backward Euler predictor: xk = x0 + h * f(x0, t)
  2. Newton loop (maxiter=20):
     a. Evaluate RHS f(xk, t+h) + fill context arrays  [expensive, ~1x per Newton step]
     b. Compute residual F(xk) = xk - x0 - h*f(xk, t+h)
     c. Assemble Jacobian J = I - h * ∂f/∂x  [N+1 JVP calls, only on first Newton step
        per restart OR on step failure, per literature]
     d. LU factorize J  [once per Jacobian assembly]
     e. Solve J·dx = -F  [cheap back-substitution]
     f. Update xk += dx
  3. Accept step if ||F|| < newton_tol

FIX (v2): Numba CUDA device functions do not support 2D array slices (e.g.
buf_xk[b, 1:]) as arguments — Numba cannot infer the stride/contiguity and
raises a TypingError at the setitem for coef7[:] inside _eval_rhs_and_context.
Solution: always copy the concentration row into a local 1D array before
passing it to any device function.  This adds a small memcopy overhead
(n_species float64 words) but is necessary for correctness.

Context reuse strategy (key for performance):
  - Context is computed ONCE per RHS evaluation (step 2a).
  - JVP calls in Jacobian assembly (step 2c) READ context, never recompute it.
  - Jacobian is REUSED across Newton steps (only reassembled on step failure
    or first step, per Stone & Davis 2013).

Memory layout: all working arrays are global memory buffers indexed by
reactor index b = blockIdx.x * blockDim.x + threadIdx.x.

N_STATE = n_species + 1  (temperature + concentrations)
"""

import math
import numpy as np
from numba import cuda, float64, int32
import numba

from .chemistry_device import (
    _eval_rhs_and_context,
    _eval_jvp,
    R_CONST, PREF, EPS,
)
from .linalg_device import (
    _lu_factor_inplace, _lu_solve,
    _dot, _norm, _axpy, _copy, _zero,
)

# Maximum supported state dimension (GRI30: 53 species + T = 54).
# If your mechanism is larger, increase MAX_N_STATE here.
MAX_N_STATE = 54


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: compute adiabatic RHS  dTC/dt = [dT/dt, dC/dt]
#  using pre-evaluated omega and thermodynamic data.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _adiabatic_rhs_from_omega(
    T, C, omega, n_species,
    thermo_coef_low, thermo_coef_high, thermo_Tmid,
    dTCdt,   # output (n_species+1,)
):
    """
    Compute dT/dt for constant-volume adiabatic reactor:
      dT/dt = -Σ(u_i * ω_i) / (n_tot * cv_mix)
      dC/dt = ω
    Fills dTCdt[0]=dT/dt, dTCdt[1:]=omega.
    C and omega are 1-D local arrays of length n_species.
    """
    coef7 = cuda.local.array(7, dtype=float64)

    n_tot = 0.0
    for s in range(n_species):
        n_tot += C[s]

    cv_mix_mol  = 0.0
    sum_u_omega = 0.0
    for s in range(n_species):
        base = s * 7
        if T <= thermo_Tmid[s]:
            for c in range(7): coef7[c] = thermo_coef_low[base + c]
        else:
            for c in range(7): coef7[c] = thermo_coef_high[base + c]
        T2 = T*T; T3=T2*T; T4=T3*T
        cpR  = (coef7[0] + coef7[1]*T + coef7[2]*T2
                + coef7[3]*T3 + coef7[4]*T4)
        cv_s = R_CONST * (cpR - 1.0)
        hRT  = (coef7[0] + coef7[1]*T/2.0 + coef7[2]*T2/3.0
                + coef7[3]*T3/4.0 + coef7[4]*T4/5.0 + coef7[5]/T)
        u_s  = R_CONST * T * hRT - R_CONST * T

        X_s         = C[s] / n_tot if n_tot > EPS else 0.0
        cv_mix_mol  += X_s * cv_s
        sum_u_omega += u_s * omega[s]

    denom = n_tot * cv_mix_mol
    if abs(denom) > EPS:
        dTCdt[0] = -sum_u_omega / denom
    else:
        dTCdt[0] = 0.0

    for s in range(n_species):
        dTCdt[s+1] = omega[s]


@cuda.jit(device=True)
def _adiabatic_jvp_TC(
    T, C, v_T, v_C,
    omega, Jv_omega,
    n_species,
    thermo_coef_low, thermo_coef_high, thermo_Tmid,
    Jv_TC,
):
    """
    JVP of the full adiabatic RHS w.r.t perturbation (v_T, v_C).
    C, v_C, omega, Jv_omega are 1-D local arrays of length n_species.
    """
    coef7 = cuda.local.array(7, dtype=float64)

    n_tot  = 0.0
    for s in range(n_species): n_tot += C[s]
    dn_tot = 0.0
    for s in range(n_species): dn_tot += v_C[s]

    cv_mix_mol = 0.0
    dcv_mix    = 0.0
    u_prime    = 0.0
    du_prime   = 0.0

    for s in range(n_species):
        base = s * 7
        if T <= thermo_Tmid[s]:
            for c in range(7): coef7[c] = thermo_coef_low[base + c]
        else:
            for c in range(7): coef7[c] = thermo_coef_high[base + c]
        T2=T*T; T3=T2*T; T4=T3*T
        cpR      = coef7[0] + coef7[1]*T + coef7[2]*T2 + coef7[3]*T3 + coef7[4]*T4
        cv_s     = R_CONST * (cpR - 1.0)
        dcv_dT_s = R_CONST * (coef7[1] + 2.0*coef7[2]*T
                               + 3.0*coef7[3]*T2 + 4.0*coef7[4]*T3)
        hRT  = (coef7[0] + coef7[1]*T/2.0 + coef7[2]*T2/3.0
                + coef7[3]*T3/4.0 + coef7[4]*T4/5.0 + coef7[5]/T)
        u_s  = R_CONST * T * hRT - R_CONST * T

        X_s  = C[s] / n_tot if n_tot > EPS else 0.0
        dX_s = ((v_C[s]/n_tot - C[s]*dn_tot/(n_tot*n_tot))
                if n_tot > EPS else 0.0)

        cv_mix_mol += X_s * cv_s
        dcv_mix    += dX_s * cv_s + X_s * dcv_dT_s * v_T
        u_prime    += u_s * omega[s]
        du_prime   += u_s * Jv_omega[s] + cv_s * omega[s] * v_T

    denom = n_tot * cv_mix_mol
    if abs(denom) > EPS:
        dTdt_prime = (
            -du_prime / denom
            + u_prime * dn_tot / (denom * n_tot)
            + u_prime * dcv_mix / (n_tot * cv_mix_mol * cv_mix_mol)
        )
    else:
        dTdt_prime = 0.0

    Jv_TC[0] = dTdt_prime
    for s in range(n_species):
        Jv_TC[s+1] = Jv_omega[s]


# ─────────────────────────────────────────────────────────────────────────────
#  Dense Jacobian assembly: J = I - h * ∂f/∂[T,C]
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _assemble_jacobian(
    T, C_loc, h, n_state,
    omega_loc,
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
    J_flat,
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
):
    """
    Assembles J = I - h * ∂f/∂[T,C] column by column.
    C_loc, omega_loc: 1-D local arrays of length n_species.
    """
    e_vec    = cuda.local.array(MAX_N_STATE, dtype=float64)
    v_C_col  = cuda.local.array(MAX_N_STATE, dtype=float64)   # length n_species
    Jv_omega = cuda.local.array(MAX_N_STATE, dtype=float64)
    Jv_TC    = cuda.local.array(MAX_N_STATE, dtype=float64)

    for col in range(n_state):
        # Build unit vector
        for i in range(n_state): e_vec[i] = 0.0
        e_vec[col] = 1.0
        v_T_col = e_vec[0]
        for s in range(n_species):
            v_C_col[s] = e_vec[s+1]

        _eval_jvp(
            T, C_loc, v_T_col, v_C_col,
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
            Jv_omega,
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
            T, C_loc, v_T_col, v_C_col,
            omega_loc, Jv_omega, n_species,
            thermo_coef_low, thermo_coef_high, thermo_Tmid,
            Jv_TC,
        )

        for row in range(n_state):
            J_flat[row * n_state + col] = e_vec[row] - h * Jv_TC[row]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: copy global 2D row into a local 1D array
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _load_row(src2d, row, dst1d, n):
    """Copy src2d[row, 0:n] → dst1d[0:n]."""
    for i in range(n):
        dst1d[i] = src2d[row, i]


@cuda.jit(device=True, inline=True)
def _store_row(src1d, dst2d, row, n):
    """Copy src1d[0:n] → dst2d[row, 0:n]."""
    for i in range(n):
        dst2d[row, i] = src1d[i]


# ─────────────────────────────────────────────────────────────────────────────
#  Main integrator kernel: per-thread backward Euler
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit
def backward_euler_per_thread_kernel(
    TC_in,      # (B, n_state)
    TC_out,     # (B, n_state)
    h_arr,      # (B,)
    t_arr,      # (B,)
    # Working global buffers
    buf_J,      # (B, n_state*n_state)
    buf_LU,     # (B, n_state*n_state)
    buf_piv,    # (B, n_state)
    buf_F,      # (B, n_state)
    buf_dx,     # (B, n_state)
    buf_xk,     # (B, n_state)
    buf_omega,  # (B, n_species)
    # Context – elementary
    buf_ctx_kf_e, buf_ctx_Kc_e,
    buf_ctx_conc_r_e, buf_ctx_conc_p_e, buf_ctx_RR_e,
    buf_ctx_dG0_e, buf_ctx_sr_e, buf_ctx_sp_e,
    # Context – three-body
    buf_ctx_kf_tb,  buf_ctx_Kc_tb,
    buf_ctx_cr_tb,  buf_ctx_cp_tb,
    buf_ctx_RR_tb,  buf_ctx_bRR_tb, buf_ctx_M_tb,
    buf_ctx_dG0_tb, buf_ctx_sr_tb,  buf_ctx_sp_tb,
    # Context – falloff
    buf_ctx_kf_fall,  buf_ctx_Kc_fall,
    buf_ctx_cr_fall,  buf_ctx_cp_fall,  buf_ctx_RR_fall,
    buf_ctx_k0_fall,  buf_ctx_kinf_fall, buf_ctx_M_fall,
    buf_ctx_Pr_fall,  buf_ctx_F_fall,   buf_ctx_kb_fall,
    buf_ctx_Fc_fall,  buf_ctx_lFc_fall,
    buf_ctx_Cp_fall,  buf_ctx_Np_fall,
    buf_ctx_f1_fall,  buf_ctx_exp_fall,
    buf_ctx_dG0_fall, buf_ctx_sr_fall,  buf_ctx_sp_fall,
    # Convergence
    converged,
    # Parameters
    newton_tol,
    newton_maxiter,
    reuse_jacobian,
    # Mechanism (read-only)
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
):
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    B = TC_in.shape[0]
    if b >= B:
        return

    n_state = n_species + 1
    h = h_arr[b]

    # ── Local working arrays (thread stack) ───────────────────────────────
    # FIX: all device function arguments must be 1-D local arrays,
    # never 2-D global buffer slices like buf_xk[b, 1:].
    xk_loc      = cuda.local.array(MAX_N_STATE, dtype=float64)   # full state [T, C...]
    C_loc       = cuda.local.array(MAX_N_STATE, dtype=float64)   # concentrations only
    omega_loc   = cuda.local.array(MAX_N_STATE, dtype=float64)
    dTCdt_loc   = cuda.local.array(MAX_N_STATE, dtype=float64)
    neg_F_loc   = cuda.local.array(MAX_N_STATE, dtype=float64)
    dx_loc      = cuda.local.array(MAX_N_STATE, dtype=float64)
    F_loc       = cuda.local.array(MAX_N_STATE, dtype=float64)

    # ── Load x0 ───────────────────────────────────────────────────────────
    for i in range(n_state):
        xk_loc[i] = TC_in[b, i]

    # ── Predictor: xk = x0 + h * f(x0) ──────────────────────────────────
    T_loc = xk_loc[0]
    for s in range(n_species):
        C_loc[s] = xk_loc[s+1]

    _eval_rhs_and_context(
        T_loc, C_loc,
        buf_ctx_kf_e[b],  buf_ctx_Kc_e[b],
        buf_ctx_conc_r_e[b], buf_ctx_conc_p_e[b], buf_ctx_RR_e[b],
        buf_ctx_dG0_e[b], buf_ctx_sr_e[b], buf_ctx_sp_e[b],
        buf_ctx_kf_tb[b],  buf_ctx_Kc_tb[b],
        buf_ctx_cr_tb[b],  buf_ctx_cp_tb[b],
        buf_ctx_RR_tb[b],  buf_ctx_bRR_tb[b], buf_ctx_M_tb[b],
        buf_ctx_dG0_tb[b], buf_ctx_sr_tb[b],  buf_ctx_sp_tb[b],
        buf_ctx_kf_fall[b], buf_ctx_Kc_fall[b],
        buf_ctx_cr_fall[b], buf_ctx_cp_fall[b], buf_ctx_RR_fall[b],
        buf_ctx_k0_fall[b], buf_ctx_kinf_fall[b], buf_ctx_M_fall[b],
        buf_ctx_Pr_fall[b], buf_ctx_F_fall[b],  buf_ctx_kb_fall[b],
        buf_ctx_Fc_fall[b], buf_ctx_lFc_fall[b],
        buf_ctx_Cp_fall[b], buf_ctx_Np_fall[b],
        buf_ctx_f1_fall[b], buf_ctx_exp_fall[b],
        buf_ctx_dG0_fall[b], buf_ctx_sr_fall[b], buf_ctx_sp_fall[b],
        omega_loc,
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

    _adiabatic_rhs_from_omega(
        T_loc, C_loc, omega_loc, n_species,
        thermo_coef_low, thermo_coef_high, thermo_Tmid,
        dTCdt_loc,
    )

    # Predictor step
    for i in range(n_state):
        xk_loc[i] = TC_in[b, i] + h * dTCdt_loc[i]

    # ── Newton loop ───────────────────────────────────────────────────────
    jacobian_valid = int32(0)

    for nit in range(newton_maxiter):
        T_loc = xk_loc[0]
        for s in range(n_species):
            C_loc[s] = xk_loc[s+1]

        # 2a. RHS evaluation
        _eval_rhs_and_context(
            T_loc, C_loc,
            buf_ctx_kf_e[b],  buf_ctx_Kc_e[b],
            buf_ctx_conc_r_e[b], buf_ctx_conc_p_e[b], buf_ctx_RR_e[b],
            buf_ctx_dG0_e[b], buf_ctx_sr_e[b], buf_ctx_sp_e[b],
            buf_ctx_kf_tb[b],  buf_ctx_Kc_tb[b],
            buf_ctx_cr_tb[b],  buf_ctx_cp_tb[b],
            buf_ctx_RR_tb[b],  buf_ctx_bRR_tb[b], buf_ctx_M_tb[b],
            buf_ctx_dG0_tb[b], buf_ctx_sr_tb[b],  buf_ctx_sp_tb[b],
            buf_ctx_kf_fall[b], buf_ctx_Kc_fall[b],
            buf_ctx_cr_fall[b], buf_ctx_cp_fall[b], buf_ctx_RR_fall[b],
            buf_ctx_k0_fall[b], buf_ctx_kinf_fall[b], buf_ctx_M_fall[b],
            buf_ctx_Pr_fall[b], buf_ctx_F_fall[b],  buf_ctx_kb_fall[b],
            buf_ctx_Fc_fall[b], buf_ctx_lFc_fall[b],
            buf_ctx_Cp_fall[b], buf_ctx_Np_fall[b],
            buf_ctx_f1_fall[b], buf_ctx_exp_fall[b],
            buf_ctx_dG0_fall[b], buf_ctx_sr_fall[b], buf_ctx_sp_fall[b],
            omega_loc,
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

        # 2b. dTCdt from omega
        _adiabatic_rhs_from_omega(
            T_loc, C_loc, omega_loc, n_species,
            thermo_coef_low, thermo_coef_high, thermo_Tmid,
            dTCdt_loc,
        )

        # 2c. Residual F = xk - x0 - h*f(xk)
        res_norm = 0.0
        for i in range(n_state):
            fi = xk_loc[i] - TC_in[b, i] - h * dTCdt_loc[i]
            F_loc[i] = fi
            res_norm += fi * fi
        res_norm = math.sqrt(res_norm)

        if res_norm < newton_tol:
            converged[b] = int32(1)
            break

        # 2d. Jacobian assembly (when needed)
        if jacobian_valid == 0 or reuse_jacobian == 0:
            _assemble_jacobian(
                T_loc, C_loc, h, n_state,
                omega_loc,
                buf_ctx_kf_e[b],  buf_ctx_Kc_e[b],
                buf_ctx_conc_r_e[b], buf_ctx_conc_p_e[b],
                buf_ctx_dG0_e[b], buf_ctx_sr_e[b], buf_ctx_sp_e[b],
                buf_ctx_kf_tb[b],  buf_ctx_Kc_tb[b],
                buf_ctx_cr_tb[b],  buf_ctx_cp_tb[b],
                buf_ctx_bRR_tb[b], buf_ctx_M_tb[b],
                buf_ctx_dG0_tb[b], buf_ctx_sr_tb[b], buf_ctx_sp_tb[b],
                buf_ctx_kf_fall[b], buf_ctx_Kc_fall[b],
                buf_ctx_cr_fall[b], buf_ctx_cp_fall[b],
                buf_ctx_k0_fall[b], buf_ctx_kinf_fall[b], buf_ctx_M_fall[b],
                buf_ctx_Pr_fall[b], buf_ctx_F_fall[b],  buf_ctx_kb_fall[b],
                buf_ctx_Fc_fall[b], buf_ctx_lFc_fall[b],
                buf_ctx_Cp_fall[b], buf_ctx_Np_fall[b],
                buf_ctx_f1_fall[b], buf_ctx_exp_fall[b],
                buf_ctx_dG0_fall[b], buf_ctx_sr_fall[b], buf_ctx_sp_fall[b],
                buf_J[b],
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
            # LU factorization
            for i in range(n_state * n_state):
                buf_LU[b, i] = buf_J[b, i]
            _lu_factor_inplace(buf_LU[b], buf_piv[b], n_state)
            jacobian_valid = int32(1)

        # 2e. Solve J·dx = -F
        for i in range(n_state):
            neg_F_loc[i] = -F_loc[i]
        _lu_solve(buf_LU[b], buf_piv[b], neg_F_loc, dx_loc, n_state)

        # 2f. Update xk
        for i in range(n_state):
            xk_loc[i] += dx_loc[i]
        # Hard constraints
        if xk_loc[0] < 1.0:
            xk_loc[0] = 1.0
        for s in range(n_species):
            if xk_loc[s+1] < 0.0:
                xk_loc[s+1] = 0.0

    # ── Write output ──────────────────────────────────────────────────────
    for i in range(n_state):
        TC_out[b, i] = xk_loc[i]