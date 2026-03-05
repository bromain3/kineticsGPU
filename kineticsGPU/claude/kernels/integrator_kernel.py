"""
Per-thread backward Euler integrator CUDA kernel — v5 (LU + matrix-free GMRES + timing).

Linear solver selection
───────────────────────
  linear_solver == 0  →  LU factorisation
  linear_solver == 1  →  restarted GMRES, MATRIX-FREE.
                         Each Krylov matvec  w = v - h·J·v  is one analytical
                         JVP call (_eval_jvp + _adiabatic_jvp_TC).
                         The Jacobian matrix is NEVER assembled.
                         ALL intermediate storage uses pre-allocated global
                         memory buffers — no cuda.local.array, no shared memory.

Why _gmres_jvp_device lives here (not in linalg_device.py)
───────────────────────────────────────────────────────────
Numba does not support imports inside device functions.
_gmres_jvp_device calls _eval_jvp (chemistry_device) and _adiabatic_jvp_TC
(this file). Defining it here avoids the circular import.

Global buffers used by _gmres_jvp_device (allocated in integrator.py)
──────────────────────────────────────────────────────────────────────
  buf_gmres_V    (B, (R+1)*N)   Krylov basis
  buf_gmres_H    (B, (R+1)*R)   Hessenberg matrix
  buf_gmres_cs   (B, R)         Givens cosines
  buf_gmres_sn   (B, R)         Givens sines
  buf_gmres_g    (B, R+1)       rotated RHS
  buf_gmres_w    (B, N)         Arnoldi work / matvec output
  buf_gmres_y    (B, R)         back-substitution result
  buf_gmres_vC   (B, N)         v_C scratch for _eval_jvp input
  buf_gmres_Jvo  (B, N)         Jv_omega scratch for _eval_jvp output

Divergence counters (optional, allocated in integrator.py)
──────────────────────────────────────────────────────────
  buf_iters  (B, 2) int32
    col 0 : n_newton_iters  — Newton iterations performed this step
    col 1 : n_krylov_iters  — total Krylov vectors produced this step
                              (sum over all GMRES restarts)
  do_count   int32  0=off  1=on
  When do_count=0, buf_iters can be a (B,1) dummy.

Timing instrumentation
──────────────────────
Seven phases (buf_timers shape (B, 7)):
  0  PREDICTOR   : initial RHS eval + explicit Euler step
  1  RHS         : _eval_rhs_and_context inside Newton loop
  2  RESIDUAL    : residual norm
  3  JAC_ASSEM   : Jacobian assembly — LU path only (always 0 for GMRES)
  4  LU_FACTOR   : LU factorisation  — LU path only (always 0 for GMRES)
  5  LINEAR_SOLVE: back-substitution (LU) or full GMRES call
  6  UPDATE      : xk += dx + hard constraints
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
    _matvec_dense,
    _arnoldi_givens_step,
    _gmres_backsolve,
    _gmres_givens_device,
    _dot, _norm, _axpy, _copy, _zero,
)

MAX_N_STATE = 54

_T_PREDICTOR    = 0
_T_RHS          = 1
_T_RESIDUAL     = 2
_T_JAC_ASSEM    = 3
_T_LU_FACTOR    = 4
_T_LINEAR_SOLVE = 5
_T_UPDATE       = 6
N_TIMER_PHASES  = 7


from numba.cuda.extending import intrinsic
from llvmlite import ir
from numba import types

@intrinsic
def cuda_clock64(typingctx):
    sig = types.uint64()
    def codegen(context, builder, sig, args):
        function_type = ir.FunctionType(ir.IntType(64), [])
        instruction = "mov.u64 $0, %clock64;"
        clock64 = ir.InlineAsm(function_type, instruction, "=l",
                               side_effect=True)
        return builder.call(clock64, [])
    return sig, codegen


# ─────────────────────────────────────────────────────────────────────────────
#  Adiabatic RHS helpers
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _adiabatic_rhs_from_omega(
    T, C, omega, n_species,
    thermo_coef_low, thermo_coef_high, thermo_Tmid,
    dTCdt,
):
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
        T2 = T*T; T3 = T2*T; T4 = T3*T
        cpR = (coef7[0] + coef7[1]*T + coef7[2]*T2
               + coef7[3]*T3 + coef7[4]*T4)
        cv_s = R_CONST * (cpR - 1.0)
        hRT  = (coef7[0] + coef7[1]*T/2.0 + coef7[2]*T2/3.0
                + coef7[3]*T3/4.0 + coef7[4]*T4/5.0 + coef7[5]/T)
        u_s  = R_CONST * T * hRT - R_CONST * T
        X_s         = C[s] / n_tot if n_tot > EPS else 0.0
        cv_mix_mol  += X_s * cv_s
        sum_u_omega += u_s * omega[s]
    denom = n_tot * cv_mix_mol
    dTCdt[0] = -sum_u_omega / denom if abs(denom) > EPS else 0.0
    for s in range(n_species):
        dTCdt[s + 1] = omega[s]


@cuda.jit(device=True)
def _adiabatic_jvp_TC(
    T, C, v_T, v_C,
    omega, Jv_omega,
    n_species,
    thermo_coef_low, thermo_coef_high, thermo_Tmid,
    Jv_TC,
):
    coef7 = cuda.local.array(7, dtype=float64)
    n_tot  = 0.0
    for s in range(n_species): n_tot += C[s]
    dn_tot = 0.0
    for s in range(n_species): dn_tot += v_C[s]
    cv_mix_mol = 0.0; dcv_mix = 0.0; u_prime = 0.0; du_prime = 0.0
    for s in range(n_species):
        base = s * 7
        if T <= thermo_Tmid[s]:
            for c in range(7): coef7[c] = thermo_coef_low[base + c]
        else:
            for c in range(7): coef7[c] = thermo_coef_high[base + c]
        T2 = T*T; T3 = T2*T; T4 = T3*T
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
        Jv_TC[s + 1] = Jv_omega[s]


# ─────────────────────────────────────────────────────────────────────────────
#  Dense Jacobian assembly — LU path only
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
    e_vec    = cuda.local.array(MAX_N_STATE, dtype=float64)
    v_C_col  = cuda.local.array(MAX_N_STATE, dtype=float64)
    Jv_omega = cuda.local.array(MAX_N_STATE, dtype=float64)
    Jv_TC    = cuda.local.array(MAX_N_STATE, dtype=float64)

    for col in range(n_state):
        for i in range(n_state): e_vec[i] = 0.0
        e_vec[col] = 1.0
        v_T_col = e_vec[0]
        for s in range(n_species):
            v_C_col[s] = e_vec[s + 1]
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
#  Matrix-free restarted GMRES  (JVP variant)
#
#  Returns int32 : total number of Krylov vectors produced.
#  The caller accumulates this into n_krylov_count for divergence analysis.
#
#  Return value encoding:
#    Early exit (beta < tol before any Arnoldi) → 0
#    Normal exit after m Krylov steps and `outer` restart cycles
#      → (outer * R + j_converged)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _gmres_jvp_device(
    T_loc,
    C_loc,
    h,
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
    rhs,
    dx,
    gmres_V,
    gmres_H,
    gmres_cs,
    gmres_sn,
    gmres_g,
    gmres_w,
    gmres_y,
    gmres_vC,
    gmres_Jvo,
    n_state,
    restart,
    gmres_tol,
    gmres_maxiter,
):
    beta = float64(0.0)
    for i in range(n_state):
        beta += rhs[i] * rhs[i]
    beta = math.sqrt(beta)
    if beta < gmres_tol:
        return int32(0)

    R = restart
    total_krylov = int32(0)

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
            for i in range(n_state):
                gmres_V[i] = rhs[i] * inv_beta
        else:
            v_T = dx[0]
            for s in range(n_species):
                gmres_vC[s] = dx[s + 1]
            _eval_jvp(
                T_loc, C_loc,
                v_T, gmres_vC,
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
                gmres_Jvo,
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
                T_loc, C_loc, v_T, gmres_vC,
                omega_loc, gmres_Jvo,
                n_species,
                thermo_coef_low, thermo_coef_high, thermo_Tmid,
                gmres_w,
            )
            beta = float64(0.0)
            for i in range(n_state):
                ri = rhs[i] - dx[i] + h * gmres_w[i]
                gmres_w[i] = ri
                beta += ri * ri
            beta = math.sqrt(beta)
            if beta < gmres_tol:
                return total_krylov
            inv_beta = float64(1.0) / beta
            for i in range(n_state):
                gmres_V[i] = gmres_w[i] * inv_beta

        gmres_g[0] = beta
        j_converged = R

        for j in range(R):
            vj_base = j * n_state
            v_T = gmres_V[vj_base]
            for s in range(n_species):
                gmres_vC[s] = gmres_V[vj_base + s + 1]
            _eval_jvp(
                T_loc, C_loc,
                v_T, gmres_vC,
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
                gmres_Jvo,
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
                T_loc, C_loc, v_T, gmres_vC,
                omega_loc, gmres_Jvo,
                n_species,
                thermo_coef_low, thermo_coef_high, thermo_Tmid,
                gmres_w,
            )
            for i in range(n_state):
                gmres_w[i] = gmres_V[vj_base + i] - h * gmres_w[i]

            res_est = _arnoldi_givens_step(
                gmres_V, gmres_H, gmres_cs, gmres_sn, gmres_g, gmres_w,
                j, n_state, R,
            )
            if res_est < gmres_tol:
                j_converged = j + 1
                break

        total_krylov += int32(j_converged)

        m = j_converged
        _gmres_backsolve(gmres_H, gmres_g, gmres_y, m, R)

        for i in range(n_state):
            acc = float64(0.0)
            for k in range(m):
                acc += gmres_V[k * n_state + i] * gmres_y[k]
            dx[i] += acc

        v_T = dx[0]
        for s in range(n_species):
            gmres_vC[s] = dx[s + 1]
        _eval_jvp(
            T_loc, C_loc,
            v_T, gmres_vC,
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
            gmres_Jvo,
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
            T_loc, C_loc, v_T, gmres_vC,
            omega_loc, gmres_Jvo,
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
            return total_krylov

        beta = res_true

    return total_krylov


# ─────────────────────────────────────────────────────────────────────────────
#  Row load / store helpers
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _load_row(src2d, row, dst1d, n):
    for i in range(n):
        dst1d[i] = src2d[row, i]

@cuda.jit(device=True, inline=True)
def _store_row(src1d, dst2d, row, n):
    for i in range(n):
        dst2d[row, i] = src1d[i]


# ─────────────────────────────────────────────────────────────────────────────
#  Main kernel
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit
def backward_euler_per_thread_kernel(
    TC_in, TC_out,
    h_arr, t_arr,
    buf_J, buf_LU, buf_piv,
    buf_F, buf_dx, buf_xk, buf_omega,
    buf_gmres_V,
    buf_gmres_H,
    buf_gmres_cs,
    buf_gmres_sn,
    buf_gmres_g,
    buf_gmres_w,
    buf_gmres_y,
    buf_gmres_vC,
    buf_gmres_Jvo,
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
    converged,
    newton_tol,
    newton_maxiter,
    reuse_jacobian,
    linear_solver,
    gmres_restart,
    gmres_tol,
    gmres_maxiter,
    buf_timers,
    do_timing,
    buf_iters,     # (B, 2) int32 — divergence counters; dummy (B,1) if do_count=0
    do_count,      # int32  0=off  1=on
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

    xk_loc    = cuda.local.array(MAX_N_STATE, dtype=float64)
    C_loc     = cuda.local.array(MAX_N_STATE, dtype=float64)
    omega_loc = cuda.local.array(MAX_N_STATE, dtype=float64)
    dTCdt_loc = cuda.local.array(MAX_N_STATE, dtype=float64)
    neg_F_loc = cuda.local.array(MAX_N_STATE, dtype=float64)
    dx_loc    = cuda.local.array(MAX_N_STATE, dtype=float64)
    F_loc     = cuda.local.array(MAX_N_STATE, dtype=float64)

    t_pred      = numba.int64(0)
    t_rhs       = numba.int64(0)
    t_res       = numba.int64(0)
    t_jac       = numba.int64(0)
    t_lu_fac    = numba.int64(0)
    t_lin_solve = numba.int64(0)
    t_update    = numba.int64(0)

    # ── Divergence counters ───────────────────────────────────────────────
    n_newton_count = int32(0)
    n_krylov_count = int32(0)

    for i in range(n_state):
        xk_loc[i] = TC_in[b, i]

    # ────────────────────────────────────────────────────────────────────
    #  PREDICTOR
    # ────────────────────────────────────────────────────────────────────
    if do_timing:
        t0 = cuda_clock64()

    T_loc = xk_loc[0]
    for s in range(n_species):
        C_loc[s] = xk_loc[s + 1]

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
    for i in range(n_state):
        xk_loc[i] = TC_in[b, i] + h * dTCdt_loc[i]

    if do_timing:
        t_pred += cuda_clock64() - t0

    # ────────────────────────────────────────────────────────────────────
    #  NEWTON LOOP
    # ────────────────────────────────────────────────────────────────────
    jacobian_valid = int32(0)

    for nit in range(newton_maxiter):
        T_loc = xk_loc[0]
        for s in range(n_species):
            C_loc[s] = xk_loc[s + 1]

        if do_timing:
            t0 = cuda_clock64()

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

        if do_timing:
            t_rhs += cuda_clock64() - t0

        if do_timing:
            t0 = cuda_clock64()

        res_norm = 0.0
        for i in range(n_state):
            fi = xk_loc[i] - TC_in[b, i] - h * dTCdt_loc[i]
            F_loc[i] = fi
            res_norm += fi * fi
        res_norm = math.sqrt(res_norm)

        if do_timing:
            t_res += cuda_clock64() - t0

        if res_norm < newton_tol:
            converged[b] = int32(1)
            break

        # ── Count Newton iterations ───────────────────────────────────────
        n_newton_count += int32(1)

        for i in range(n_state):
            neg_F_loc[i] = -F_loc[i]

        if linear_solver == 0:
            if jacobian_valid == 0 or reuse_jacobian == 0:
                if do_timing:
                    t0 = cuda_clock64()

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

                if do_timing:
                    t_jac += cuda_clock64() - t0

                if do_timing:
                    t0 = cuda_clock64()
                for i in range(n_state * n_state):
                    buf_LU[b, i] = buf_J[b, i]
                _lu_factor_inplace(buf_LU[b], buf_piv[b], n_state)
                if do_timing:
                    t_lu_fac += cuda_clock64() - t0

                jacobian_valid = int32(1)

        if do_timing:
            t0 = cuda_clock64()

        if linear_solver == 0:
            _lu_solve(buf_LU[b], buf_piv[b], neg_F_loc, dx_loc, n_state)

        else:
            for i in range(n_state):
                dx_loc[i]    = float64(0.0)
                buf_dx[b, i] = float64(0.0)

            # ── Accumulate Krylov count from return value ─────────────────
            n_krylov_count += _gmres_jvp_device(
                T_loc, C_loc, h,
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
                neg_F_loc,
                buf_dx[b],
                buf_gmres_V[b],
                buf_gmres_H[b],
                buf_gmres_cs[b],
                buf_gmres_sn[b],
                buf_gmres_g[b],
                buf_gmres_w[b],
                buf_gmres_y[b],
                buf_gmres_vC[b],
                buf_gmres_Jvo[b],
                n_state,
                gmres_restart,
                gmres_tol,
                gmres_maxiter,
            )
            for i in range(n_state):
                dx_loc[i] = buf_dx[b, i]

        if do_timing:
            t_lin_solve += cuda_clock64() - t0

        if do_timing:
            t0 = cuda_clock64()

        for i in range(n_state):
            xk_loc[i] += dx_loc[i]
        if xk_loc[0] < 1.0:
            xk_loc[0] = 1.0
        for s in range(n_species):
            if xk_loc[s + 1] < 0.0:
                xk_loc[s + 1] = 0.0

        if do_timing:
            t_update += cuda_clock64() - t0

    # ── Write output ──────────────────────────────────────────────────────
    for i in range(n_state):
        TC_out[b, i] = xk_loc[i]

    # ── Flush divergence counters ─────────────────────────────────────────
    if do_count:
        buf_iters[b, 0] = n_newton_count
        buf_iters[b, 1] = n_krylov_count

    # ── Flush timer counts ────────────────────────────────────────────────
    if do_timing:
        buf_timers[b, 0] += t_pred
        buf_timers[b, 1] += t_rhs
        buf_timers[b, 2] += t_res
        buf_timers[b, 3] += t_jac
        buf_timers[b, 4] += t_lu_fac
        buf_timers[b, 5] += t_lin_solve
        buf_timers[b, 6] += t_update