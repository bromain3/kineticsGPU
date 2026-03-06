"""
integrator_kernel_profiled.py — v5 + ultra-detailed per-device-function timing.

Phases (N_TIMER_PHASES = 27) — chaque phase est chronométrée INDIVIDUELLEMENT :

 PREDICTOR block
  0  PRED_RHS          : _eval_rhs_and_context (predictor)
  1  PRED_ADIAB        : _adiabatic_rhs_from_omega (predictor)
  2  PRED_UPDATE       : explicit Euler xk += h*f (predictor)

 Per-Newton-iteration block  (accumulé sur tous les itérations)
  3  NEWT_RHS          : _eval_rhs_and_context (Newton loop)
  4  NEWT_ADIAB        : _adiabatic_rhs_from_omega (Newton loop)
  5  NEWT_RESIDUAL     : résidu F = xk - x0 - h*f  +  norm
  6  NEWT_CONV_CHECK   : comparaison res_norm < tol (inclus dans RESIDUAL, séparé ici si besoin)

 LU path (accumulé)
  7  LU_JAC_JVP        : _eval_jvp  ×  n_state  (dans _assemble_jacobian)
  8  LU_JAC_ADIAB      : _adiabatic_jvp_TC  ×  n_state
  9  LU_JAC_STORE      : boucle d'écriture J_flat[row*n+col]
 10  LU_COPY_TO_LU     : memcopy J→LU avant factorisation
 11  LU_FACTOR         : _lu_factor_inplace
 12  LU_SOLVE          : _lu_solve

 GMRES path (accumulé sur tous les restarts et Krylov vectors)
 13  GMRES_INIT        : init V[0] / résidu restart
 14  GMRES_JVP         : _eval_jvp  dans Arnoldi matvec
 15  GMRES_ADIAB       : _adiabatic_jvp_TC  dans Arnoldi matvec
 16  GMRES_MATVEC_POST : w[i] = v[i] - h*w[i]  après JVP+adiab
 17  GMRES_ARNOLDI     : _arnoldi_givens_step
 18  GMRES_BACKSOLVE   : _gmres_backsolve
 19  GMRES_UPDATE_DX   : dx += V*y
 20  GMRES_RESCHECK_JVP : JVP+adiab pour vérification résidu (fin restart)
 21  GMRES_RESCHECK    : calcul résidu true après update dx

 UPDATE block (après solve)
 22  NEWT_STEP_UPDATE  : xk += dx + clamp

 TOTAL aggregates  (pour validation vs somme des phases)
 23  TOT_PREDICTOR     : = 0+1+2
 24  TOT_NEWTON_BODY   : = 3+4+5  (par itération)
 25  TOT_LINEAR        : = 7..12 ou 13..21
 26  TOT_UPDATE        : = 22

USAGE
─────
Remplacer les imports dans cuda_integrator.py :
    from ..kernels.integrator_kernel_profiled import (
        backward_euler_per_thread_kernel,
        N_TIMER_PHASES, PHASE_NAMES,
        ...
    )
Tous les autres arguments du kernel sont identiques à v5.
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

# ── Timer phase indices ────────────────────────────────────────────────────
_TP_PRED_RHS          = 0
_TP_PRED_ADIAB        = 1
_TP_PRED_UPDATE       = 2

_TP_NEWT_RHS          = 3
_TP_NEWT_ADIAB        = 4
_TP_NEWT_RESIDUAL     = 5

_TP_LU_JAC_JVP        = 7
_TP_LU_JAC_ADIAB      = 8
_TP_LU_JAC_STORE      = 9
_TP_LU_COPY_TO_LU     = 10
_TP_LU_FACTOR         = 11
_TP_LU_SOLVE          = 12

_TP_GMRES_INIT        = 13
_TP_GMRES_JVP         = 14
_TP_GMRES_ADIAB       = 15
_TP_GMRES_MATVEC_POST = 16
_TP_GMRES_ARNOLDI     = 17
_TP_GMRES_BACKSOLVE   = 18
_TP_GMRES_UPDATE_DX   = 19
_TP_GMRES_RESCHECK_JVP= 20
_TP_GMRES_RESCHECK    = 21

_TP_NEWT_STEP_UPDATE  = 22

_TP_TOT_PREDICTOR     = 23
_TP_TOT_NEWTON_BODY   = 24
_TP_TOT_LINEAR        = 25
_TP_TOT_UPDATE        = 26

N_TIMER_PHASES = 27

PHASE_NAMES = [
    "PRED_RHS         ",   #  0
    "PRED_ADIAB       ",   #  1
    "PRED_UPDATE      ",   #  2
    "NEWT_RHS         ",   #  3
    "NEWT_ADIAB       ",   #  4
    "NEWT_RESIDUAL    ",   #  5
    "--- (unused)     ",   #  6
    "LU_JAC_JVP       ",   #  7
    "LU_JAC_ADIAB     ",   #  8
    "LU_JAC_STORE     ",   #  9
    "LU_COPY_TO_LU    ",   # 10
    "LU_FACTOR        ",   # 11
    "LU_SOLVE         ",   # 12
    "GMRES_INIT       ",   # 13
    "GMRES_JVP        ",   # 14
    "GMRES_ADIAB      ",   # 15
    "GMRES_MATVEC_POST",   # 16
    "GMRES_ARNOLDI    ",   # 17
    "GMRES_BACKSOLVE  ",   # 18
    "GMRES_UPDATE_DX  ",   # 19
    "GMRES_RESCK_JVP  ",   # 20
    "GMRES_RESCHECK   ",   # 21
    "NEWT_STEP_UPDATE ",   # 22
    "=TOT_PREDICTOR   ",   # 23
    "=TOT_NEWTON_BODY ",   # 24
    "=TOT_LINEAR      ",   # 25
    "=TOT_UPDATE      ",   # 26
]


# ── Inline clock ───────────────────────────────────────────────────────────
from numba.cuda.extending import intrinsic
from llvmlite import ir
from numba import types

@intrinsic
def cuda_clock64(typingctx):
    sig = types.uint64()
    def codegen(context, builder, sig, args):
        function_type = ir.FunctionType(ir.IntType(64), [])
        instruction   = "mov.u64 $0, %clock64;"
        clock64       = ir.InlineAsm(function_type, instruction, "=l", side_effect=True)
        return builder.call(clock64, [])
    return sig, codegen


# ─────────────────────────────────────────────────────────────────────────────
#  Adiabatic helpers  (identical to v5)
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
#  Dense Jacobian assembly — LU path, with per-sub-call timing
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _assemble_jacobian_timed(
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
    # timing accumulators (passed by ref via 1-element local arrays trick)
    t_jac_jvp,    # numba.int64[1]
    t_jac_adiab,  # numba.int64[1]
    t_jac_store,  # numba.int64[1]
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

        _t0 = cuda_clock64()
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
        t_jac_jvp[0] += cuda_clock64() - _t0

        _t0 = cuda_clock64()
        _adiabatic_jvp_TC(
            T, C_loc, v_T_col, v_C_col,
            omega_loc, Jv_omega, n_species,
            thermo_coef_low, thermo_coef_high, thermo_Tmid,
            Jv_TC,
        )
        t_jac_adiab[0] += cuda_clock64() - _t0

        _t0 = cuda_clock64()
        for row in range(n_state):
            J_flat[row * n_state + col] = e_vec[row] - h * Jv_TC[row]
        t_jac_store[0] += cuda_clock64() - _t0


# ─────────────────────────────────────────────────────────────────────────────
#  Matrix-free GMRES — with per-sub-call timing
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _gmres_jvp_timed(
    T_loc, C_loc, h,
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
    rhs, dx,
    gmres_V, gmres_H, gmres_cs, gmres_sn, gmres_g,
    gmres_w, gmres_y, gmres_vC, gmres_Jvo,
    n_state, restart, gmres_tol, gmres_maxiter,
    # timing accumulators (per-phase, int64)
    t_init,        # phase 13
    t_jvp,         # phase 14
    t_adiab,       # phase 15
    t_matvec_post, # phase 16
    t_arnoldi,     # phase 17
    t_backsolve,   # phase 18
    t_update_dx,   # phase 19
    t_resck_jvp,   # phase 20
    t_rescheck,    # phase 21
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

        # ── INIT / restart residual ───────────────────────────────────────
        _t0 = cuda_clock64()
        if outer == 0:
            inv_beta = float64(1.0) / beta
            for i in range(n_state):
                gmres_V[i] = rhs[i] * inv_beta
        else:
            v_T = dx[0]
            for s in range(n_species):
                gmres_vC[s] = dx[s + 1]
            t_init[0] += cuda_clock64() - _t0

            _t0 = cuda_clock64()
            _eval_jvp(
                T_loc, C_loc, v_T, gmres_vC,
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
                gmres_Jvo, n_species,
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
            t_jvp[0] += cuda_clock64() - _t0

            _t0 = cuda_clock64()
            _adiabatic_jvp_TC(
                T_loc, C_loc, v_T, gmres_vC,
                omega_loc, gmres_Jvo, n_species,
                thermo_coef_low, thermo_coef_high, thermo_Tmid,
                gmres_w,
            )
            t_adiab[0] += cuda_clock64() - _t0

            _t0 = cuda_clock64()
            beta = float64(0.0)
            for i in range(n_state):
                ri = rhs[i] - dx[i] + h * gmres_w[i]
                gmres_w[i] = ri
                beta += ri * ri
            beta = math.sqrt(beta)
            if beta < gmres_tol:
                t_matvec_post[0] += cuda_clock64() - _t0
                return total_krylov
            inv_beta = float64(1.0) / beta
            for i in range(n_state):
                gmres_V[i] = gmres_w[i] * inv_beta
            t_matvec_post[0] += cuda_clock64() - _t0

            _t0 = cuda_clock64()
            # (init timing falls through from first branch)
            t_init[0] += cuda_clock64() - _t0

        if outer == 0:
            t_init[0] += cuda_clock64() - _t0

        gmres_g[0] = beta
        j_converged = R

        # ── Arnoldi loop ──────────────────────────────────────────────────
        for j in range(R):
            vj_base = j * n_state
            v_T = gmres_V[vj_base]
            for s in range(n_species):
                gmres_vC[s] = gmres_V[vj_base + s + 1]

            _t0 = cuda_clock64()
            _eval_jvp(
                T_loc, C_loc, v_T, gmres_vC,
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
                gmres_Jvo, n_species,
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
            t_jvp[0] += cuda_clock64() - _t0

            _t0 = cuda_clock64()
            _adiabatic_jvp_TC(
                T_loc, C_loc, v_T, gmres_vC,
                omega_loc, gmres_Jvo, n_species,
                thermo_coef_low, thermo_coef_high, thermo_Tmid,
                gmres_w,
            )
            t_adiab[0] += cuda_clock64() - _t0

            _t0 = cuda_clock64()
            for i in range(n_state):
                gmres_w[i] = gmres_V[vj_base + i] - h * gmres_w[i]
            t_matvec_post[0] += cuda_clock64() - _t0

            _t0 = cuda_clock64()
            res_est = _arnoldi_givens_step(
                gmres_V, gmres_H, gmres_cs, gmres_sn, gmres_g, gmres_w,
                j, n_state, R,
            )
            t_arnoldi[0] += cuda_clock64() - _t0

            if res_est < gmres_tol:
                j_converged = j + 1
                break

        total_krylov += int32(j_converged)

        _t0 = cuda_clock64()
        m = j_converged
        _gmres_backsolve(gmres_H, gmres_g, gmres_y, m, R)
        t_backsolve[0] += cuda_clock64() - _t0

        _t0 = cuda_clock64()
        for i in range(n_state):
            acc = float64(0.0)
            for k in range(m):
                acc += gmres_V[k * n_state + i] * gmres_y[k]
            dx[i] += acc
        t_update_dx[0] += cuda_clock64() - _t0

        # ── True residual check ───────────────────────────────────────────
        v_T = dx[0]
        for s in range(n_species):
            gmres_vC[s] = dx[s + 1]

        _t0 = cuda_clock64()
        _eval_jvp(
            T_loc, C_loc, v_T, gmres_vC,
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
            gmres_Jvo, n_species,
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
        t_resck_jvp[0] += cuda_clock64() - _t0

        _t0 = cuda_clock64()
        _adiabatic_jvp_TC(
            T_loc, C_loc, v_T, gmres_vC,
            omega_loc, gmres_Jvo, n_species,
            thermo_coef_low, thermo_coef_high, thermo_Tmid,
            gmres_w,
        )
        res_true = float64(0.0)
        for i in range(n_state):
            ri = rhs[i] - (dx[i] - h * gmres_w[i])
            res_true += ri * ri
        res_true = math.sqrt(res_true)
        t_rescheck[0] += cuda_clock64() - _t0

        if res_true < gmres_tol or j_converged < R:
            return total_krylov

        beta = res_true

    return total_krylov


# ─────────────────────────────────────────────────────────────────────────────
#  Row helpers
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
#  Main kernel  (ultra-detailed timing version)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit
def backward_euler_per_thread_kernel(
    TC_in, TC_out,
    h_arr,
    buf_J, buf_LU, buf_piv,
    buf_F, buf_dx, buf_xk, buf_omega,
    buf_gmres_V, buf_gmres_H,
    buf_gmres_cs, buf_gmres_sn, buf_gmres_g,
    buf_gmres_w, buf_gmres_y, buf_gmres_vC, buf_gmres_Jvo,
    buf_ctx_kf_e, buf_ctx_Kc_e,
    buf_ctx_conc_r_e, buf_ctx_conc_p_e, buf_ctx_RR_e,
    buf_ctx_dG0_e, buf_ctx_sr_e, buf_ctx_sp_e,
    buf_ctx_kf_tb,  buf_ctx_Kc_tb,
    buf_ctx_cr_tb,  buf_ctx_cp_tb,
    buf_ctx_RR_tb,  buf_ctx_bRR_tb, buf_ctx_M_tb,
    buf_ctx_dG0_tb, buf_ctx_sr_tb,  buf_ctx_sp_tb,
    buf_ctx_kf_fall,  buf_ctx_Kc_fall,
    buf_ctx_cr_fall,  buf_ctx_cp_fall,  buf_ctx_RR_fall,
    buf_ctx_k0_fall,  buf_ctx_kinf_fall, buf_ctx_M_fall,
    buf_ctx_Pr_fall,  buf_ctx_F_fall,   buf_ctx_kb_fall,
    buf_ctx_Fc_fall,  buf_ctx_lFc_fall,
    buf_ctx_Cp_fall,  buf_ctx_Np_fall,
    buf_ctx_f1_fall,  buf_ctx_exp_fall,
    buf_ctx_dG0_fall, buf_ctx_sr_fall,  buf_ctx_sp_fall,
    converged,
    newton_tol, newton_maxiter,
    reuse_jacobian, linear_solver,
    gmres_restart, gmres_tol, gmres_maxiter,
    buf_timers,   # (B, N_TIMER_PHASES=27)  int64
    do_timing,
    buf_iters, do_count,
    n_species,
    thermo_coef_low, thermo_coef_high, thermo_Tmid,
    n_e, e_A, e_n, e_Ea, e_oneWay, e_delta_n,
    e_react_idx, e_react_sto, e_react_offset,
    e_prod_idx,  e_prod_sto,  e_prod_offset,
    e_idx_r_scatter, e_idx_p_scatter,
    n_tb, tb_A, tb_n, tb_Ea, tb_oneWay, tb_delta_n,
    tb_react_idx, tb_react_sto, tb_react_offset,
    tb_prod_idx,  tb_prod_sto,  tb_prod_offset,
    tb_idx_r_scatter, tb_idx_p_scatter, tb_eff_matrix,
    n_fall, fall_A_k0, fall_b_k0, fall_Ea_k0,
    fall_A_kinf, fall_b_kinf, fall_Ea_kinf,
    fall_oneWay, fall_delta_n,
    fall_troe_A, fall_troe_T1, fall_troe_T2, fall_troe_T3, fall_is_troe,
    fall_react_idx, fall_react_sto, fall_react_offset,
    fall_prod_idx,  fall_prod_sto,  fall_prod_offset,
    fall_idx_r_scatter, fall_idx_p_scatter, fall_eff_matrix,
):
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    B = TC_in.shape[0]
    if b >= B:
        return

    n_state = n_species + 1
    h = h_arr[b]

    # ── Local scratch ──────────────────────────────────────────────────────
    xk_loc    = cuda.local.array(MAX_N_STATE, dtype=float64)
    C_loc     = cuda.local.array(MAX_N_STATE, dtype=float64)
    omega_loc = cuda.local.array(MAX_N_STATE, dtype=float64)
    dTCdt_loc = cuda.local.array(MAX_N_STATE, dtype=float64)
    neg_F_loc = cuda.local.array(MAX_N_STATE, dtype=float64)
    dx_loc    = cuda.local.array(MAX_N_STATE, dtype=float64)
    F_loc     = cuda.local.array(MAX_N_STATE, dtype=float64)

    # ── Per-phase timing accumulators (local int64) ────────────────────────
    # Predictor
    t_pred_rhs   = numba.int64(0)
    t_pred_adiab = numba.int64(0)
    t_pred_upd   = numba.int64(0)
    # Newton loop
    t_newt_rhs   = numba.int64(0)
    t_newt_adiab = numba.int64(0)
    t_newt_res   = numba.int64(0)
    # LU path
    t_lu_jac_jvp   = numba.int64(0)
    t_lu_jac_adiab = numba.int64(0)
    t_lu_jac_store = numba.int64(0)
    t_lu_copy      = numba.int64(0)
    t_lu_factor    = numba.int64(0)
    t_lu_solve     = numba.int64(0)
    # GMRES path
    t_gm_init        = numba.int64(0)
    t_gm_jvp         = numba.int64(0)
    t_gm_adiab       = numba.int64(0)
    t_gm_mvpost      = numba.int64(0)
    t_gm_arnoldi     = numba.int64(0)
    t_gm_backsolve   = numba.int64(0)
    t_gm_update_dx   = numba.int64(0)
    t_gm_resck_jvp   = numba.int64(0)
    t_gm_rescheck    = numba.int64(0)
    # After-solve update
    t_step_upd       = numba.int64(0)

    n_newton_count = int32(0)
    n_krylov_count = int32(0)

    for i in range(n_state):
        xk_loc[i] = TC_in[b, i]

    # ══════════════════════════════════════════════════════════════════════
    #  PREDICTOR
    # ══════════════════════════════════════════════════════════════════════
    T_loc = xk_loc[0]
    for s in range(n_species):
        C_loc[s] = xk_loc[s + 1]

    _t0 = cuda_clock64()
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
        omega_loc, n_species,
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
    t_pred_rhs += cuda_clock64() - _t0

    _t0 = cuda_clock64()
    _adiabatic_rhs_from_omega(
        T_loc, C_loc, omega_loc, n_species,
        thermo_coef_low, thermo_coef_high, thermo_Tmid,
        dTCdt_loc,
    )
    t_pred_adiab += cuda_clock64() - _t0

    _t0 = cuda_clock64()
    for i in range(n_state):
        xk_loc[i] = TC_in[b, i] + h * dTCdt_loc[i]
    t_pred_upd += cuda_clock64() - _t0

    # ══════════════════════════════════════════════════════════════════════
    #  NEWTON LOOP
    # ══════════════════════════════════════════════════════════════════════
    jacobian_valid = int32(0)

    for nit in range(newton_maxiter):
        T_loc = xk_loc[0]
        for s in range(n_species):
            C_loc[s] = xk_loc[s + 1]

        # ── RHS ──────────────────────────────────────────────────────────
        _t0 = cuda_clock64()
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
            omega_loc, n_species,
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
        t_newt_rhs += cuda_clock64() - _t0

        _t0 = cuda_clock64()
        _adiabatic_rhs_from_omega(
            T_loc, C_loc, omega_loc, n_species,
            thermo_coef_low, thermo_coef_high, thermo_Tmid,
            dTCdt_loc,
        )
        t_newt_adiab += cuda_clock64() - _t0

        # ── Residual ──────────────────────────────────────────────────────
        _t0 = cuda_clock64()
        res_norm = 0.0
        for i in range(n_state):
            fi = xk_loc[i] - TC_in[b, i] - h * dTCdt_loc[i]
            F_loc[i] = fi
            res_norm += fi * fi
        res_norm = math.sqrt(res_norm)
        t_newt_res += cuda_clock64() - _t0

        if res_norm < newton_tol:
            converged[b] = int32(1)
            break

        n_newton_count += int32(1)
        for i in range(n_state):
            neg_F_loc[i] = -F_loc[i]

        # ── LU path ───────────────────────────────────────────────────────
        if linear_solver == 0:
            if jacobian_valid == 0 or reuse_jacobian == 0:
                # Timing accumulators passed as 1-element local arrays
                t_jjvp_acc   = cuda.local.array(1, dtype=numba.int64)
                t_jadiab_acc = cuda.local.array(1, dtype=numba.int64)
                t_jstore_acc = cuda.local.array(1, dtype=numba.int64)
                t_jjvp_acc[0]   = numba.int64(0)
                t_jadiab_acc[0] = numba.int64(0)
                t_jstore_acc[0] = numba.int64(0)

                _assemble_jacobian_timed(
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
                    t_jjvp_acc, t_jadiab_acc, t_jstore_acc,
                )
                t_lu_jac_jvp   += t_jjvp_acc[0]
                t_lu_jac_adiab += t_jadiab_acc[0]
                t_lu_jac_store += t_jstore_acc[0]

                _t0 = cuda_clock64()
                for i in range(n_state * n_state):
                    buf_LU[b, i] = buf_J[b, i]
                t_lu_copy += cuda_clock64() - _t0

                _t0 = cuda_clock64()
                _lu_factor_inplace(buf_LU[b], buf_piv[b], n_state)
                t_lu_factor += cuda_clock64() - _t0

                jacobian_valid = int32(1)

            _t0 = cuda_clock64()
            _lu_solve(buf_LU[b], buf_piv[b], neg_F_loc, dx_loc, n_state)
            t_lu_solve += cuda_clock64() - _t0

        # ── GMRES path ────────────────────────────────────────────────────
        else:
            for i in range(n_state):
                dx_loc[i]    = float64(0.0)
                buf_dx[b, i] = float64(0.0)

            # 1-element arrays for passing timing accumulators into device fn
            t_gminit_a    = cuda.local.array(1, dtype=numba.int64)
            t_gmjvp_a     = cuda.local.array(1, dtype=numba.int64)
            t_gmadiab_a   = cuda.local.array(1, dtype=numba.int64)
            t_gmmvpost_a  = cuda.local.array(1, dtype=numba.int64)
            t_gmarnoldi_a = cuda.local.array(1, dtype=numba.int64)
            t_gmbs_a      = cuda.local.array(1, dtype=numba.int64)
            t_gmupddx_a   = cuda.local.array(1, dtype=numba.int64)
            t_gmrckjvp_a  = cuda.local.array(1, dtype=numba.int64)
            t_gmrck_a     = cuda.local.array(1, dtype=numba.int64)
            t_gminit_a[0]    = numba.int64(0)
            t_gmjvp_a[0]     = numba.int64(0)
            t_gmadiab_a[0]   = numba.int64(0)
            t_gmmvpost_a[0]  = numba.int64(0)
            t_gmarnoldi_a[0] = numba.int64(0)
            t_gmbs_a[0]      = numba.int64(0)
            t_gmupddx_a[0]   = numba.int64(0)
            t_gmrckjvp_a[0]  = numba.int64(0)
            t_gmrck_a[0]     = numba.int64(0)

            n_krylov_count += _gmres_jvp_timed(
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
                omega_loc, n_species,
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
                neg_F_loc, buf_dx[b],
                buf_gmres_V[b], buf_gmres_H[b],
                buf_gmres_cs[b], buf_gmres_sn[b], buf_gmres_g[b],
                buf_gmres_w[b], buf_gmres_y[b], buf_gmres_vC[b], buf_gmres_Jvo[b],
                n_state, gmres_restart, gmres_tol, gmres_maxiter,
                t_gminit_a, t_gmjvp_a, t_gmadiab_a, t_gmmvpost_a,
                t_gmarnoldi_a, t_gmbs_a, t_gmupddx_a, t_gmrckjvp_a, t_gmrck_a,
            )

            t_gm_init      += t_gminit_a[0]
            t_gm_jvp       += t_gmjvp_a[0]
            t_gm_adiab     += t_gmadiab_a[0]
            t_gm_mvpost    += t_gmmvpost_a[0]
            t_gm_arnoldi   += t_gmarnoldi_a[0]
            t_gm_backsolve += t_gmbs_a[0]
            t_gm_update_dx += t_gmupddx_a[0]
            t_gm_resck_jvp += t_gmrckjvp_a[0]
            t_gm_rescheck  += t_gmrck_a[0]

            for i in range(n_state):
                dx_loc[i] = buf_dx[b, i]

        # ── Step update ───────────────────────────────────────────────────
        _t0 = cuda_clock64()
        for i in range(n_state):
            xk_loc[i] += dx_loc[i]
        if xk_loc[0] < 1.0:
            xk_loc[0] = 1.0
        for s in range(n_species):
            if xk_loc[s + 1] < 0.0:
                xk_loc[s + 1] = 0.0
        t_step_upd += cuda_clock64() - _t0

    # ══════════════════════════════════════════════════════════════════════
    #  Write output
    # ══════════════════════════════════════════════════════════════════════
    for i in range(n_state):
        TC_out[b, i] = xk_loc[i]

    if do_count:
        buf_iters[b, 0] = n_newton_count
        buf_iters[b, 1] = n_krylov_count

    # ── Flush all timers ──────────────────────────────────────────────────
    if do_timing:
        buf_timers[b, _TP_PRED_RHS]       += t_pred_rhs
        buf_timers[b, _TP_PRED_ADIAB]     += t_pred_adiab
        buf_timers[b, _TP_PRED_UPDATE]    += t_pred_upd
        buf_timers[b, _TP_NEWT_RHS]       += t_newt_rhs
        buf_timers[b, _TP_NEWT_ADIAB]     += t_newt_adiab
        buf_timers[b, _TP_NEWT_RESIDUAL]  += t_newt_res
        buf_timers[b, _TP_LU_JAC_JVP]    += t_lu_jac_jvp
        buf_timers[b, _TP_LU_JAC_ADIAB]  += t_lu_jac_adiab
        buf_timers[b, _TP_LU_JAC_STORE]  += t_lu_jac_store
        buf_timers[b, _TP_LU_COPY_TO_LU] += t_lu_copy
        buf_timers[b, _TP_LU_FACTOR]     += t_lu_factor
        buf_timers[b, _TP_LU_SOLVE]      += t_lu_solve
        buf_timers[b, _TP_GMRES_INIT]        += t_gm_init
        buf_timers[b, _TP_GMRES_JVP]         += t_gm_jvp
        buf_timers[b, _TP_GMRES_ADIAB]       += t_gm_adiab
        buf_timers[b, _TP_GMRES_MATVEC_POST] += t_gm_mvpost
        buf_timers[b, _TP_GMRES_ARNOLDI]     += t_gm_arnoldi
        buf_timers[b, _TP_GMRES_BACKSOLVE]   += t_gm_backsolve
        buf_timers[b, _TP_GMRES_UPDATE_DX]   += t_gm_update_dx
        buf_timers[b, _TP_GMRES_RESCHECK_JVP]+= t_gm_resck_jvp
        buf_timers[b, _TP_GMRES_RESCHECK]    += t_gm_rescheck
        buf_timers[b, _TP_NEWT_STEP_UPDATE]  += t_step_upd
        # Aggregate totals
        buf_timers[b, _TP_TOT_PREDICTOR]  += t_pred_rhs + t_pred_adiab + t_pred_upd
        buf_timers[b, _TP_TOT_NEWTON_BODY]+= t_newt_rhs + t_newt_adiab + t_newt_res
        buf_timers[b, _TP_TOT_LINEAR]     += (
            t_lu_jac_jvp + t_lu_jac_adiab + t_lu_jac_store +
            t_lu_copy + t_lu_factor + t_lu_solve +
            t_gm_init + t_gm_jvp + t_gm_adiab + t_gm_mvpost +
            t_gm_arnoldi + t_gm_backsolve + t_gm_update_dx +
            t_gm_resck_jvp + t_gm_rescheck
        )
        buf_timers[b, _TP_TOT_UPDATE]     += t_step_upd