"""
kernels/per_block/chemistry_kernels_pb.py — Kernel principal per-block.

Architecture : 1 bloc CUDA = 1 réacteur (THREADS_PER_BLOCK threads coopèrent).

Kernel exporté :
    backward_euler_kernel_pb — intégration backward-Euler per-block

Constantes exportées :
    MAX_N_STATE, MAX_R, THREADS_PER_BLOCK

Budget mémoire partagée par bloc (R=20, n_state=54) :
    10 vecteurs × 54 × 8B = 4 320 B
    H: 21×20×8B            = 3 360 B
    g,y,cs,sn: ~84×8B      =   672 B
    scratch: 64×8B          =   512 B
    Total ≈ 8 864 B  →  tient dans 12 KB/bloc @ 8 blocs/SM
"""

import math
import numpy as np
from numba import cuda, float64, int32

from .context_pb import _eval_rhs_and_context_pb
from .jvp_pb import _eval_jvp_pb
from .reactor_odes_pb import _adiabatic_rhs_pb, _adiabatic_jvp_pb
from .gmres_pb import _gmres_pb
from .reductions import _block_reduce_sum

MAX_N_STATE       = 54
MAX_R             = 20
MAX_R_p_1         = MAX_R + 1
MAX_R_H           = MAX_R * (MAX_R + 1)
THREADS_PER_BLOCK = 64


@cuda.jit(cache=True)
def backward_euler_kernel_pb(
    TC_in, TC_out,
    h_scalar,
    newton_tol, newton_maxiter,
    gmres_restart, gmres_tol, gmres_maxiter,
    converged,
    gm_V,           # mémoire globale (B, (R+1)*n_state) — base de Krylov
    # ── Contexte cinétique (global, une ligne par réacteur) ───────────
    ctx_kf_e,   ctx_Kc_e,  ctx_cr_e,  ctx_cp_e,  ctx_dG0_e,
    ctx_sRre,   ctx_sRpe,
    ctx_kf_tb,  ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb,  ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f,   ctx_Kc_f,  ctx_cr_f,  ctx_cp_f,  ctx_dG0_f,
    ctx_sRrf,   ctx_sRpf,
    ctx_k0_f,   ctx_kinf_f, ctx_M_f,  ctx_Pr_f,  ctx_F_f,   ctx_kb_f,
    ctx_Fc_f,   ctx_lFc_f, ctx_Cp_f,  ctx_Np_f,  ctx_f1_f,  ctx_exp_f,
    # ── Mécanisme ─────────────────────────────────────────────────────
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    """
    Lance avec : backward_euler_kernel_pb[n_reactors, THREADS_PER_BLOCK](...)
    blockIdx.x  = indice du réacteur b
    threadIdx.x = tid au sein du bloc
    """
    b         = cuda.blockIdx.x
    tid       = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1
    h       = h_scalar

    # ── Allocation de la mémoire partagée ─────────────────────────────
    sh_xk  = cuda.shared.array(MAX_N_STATE, float64)
    sh_C   = cuda.shared.array(MAX_N_STATE, float64)
    sh_om  = cuda.shared.array(MAX_N_STATE, float64)
    sh_dTC = cuda.shared.array(MAX_N_STATE, float64)
    sh_rhs = cuda.shared.array(MAX_N_STATE, float64)
    sh_dx  = cuda.shared.array(MAX_N_STATE, float64)
    sh_w   = cuda.shared.array(MAX_N_STATE, float64)
    sh_Jvo = cuda.shared.array(MAX_N_STATE, float64)
    sh_vC  = cuda.shared.array(MAX_N_STATE, float64)
    sh_g   = cuda.shared.array(MAX_R_p_1,   float64)
    sh_y   = cuda.shared.array(MAX_R,        float64)
    sh_cs  = cuda.shared.array(MAX_R,        float64)
    sh_sn  = cuda.shared.array(MAX_R,        float64)
    sh_H   = cuda.shared.array(MAX_R_H,      float64)
    sh_scratch = cuda.shared.array(THREADS_PER_BLOCK, float64)

    # ── Chargement de l'état initial en partagée ──────────────────────
    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()

    # ── Row-slices du contexte cinétique pour ce réacteur ─────────────
    e_kf   = ctx_kf_e[b];   e_Kc  = ctx_Kc_e[b]
    e_cr   = ctx_cr_e[b];   e_cp  = ctx_cp_e[b];  e_dG = ctx_dG0_e[b]
    e_sRr  = ctx_sRre[b];   e_sRp = ctx_sRpe[b]

    tb_kf  = ctx_kf_tb[b];  tb_Kc  = ctx_Kc_tb[b]
    tb_cr  = ctx_cr_tb[b];  tb_cp  = ctx_cp_tb[b]; tb_dG = ctx_dG0_tb[b]
    tb_sRr = ctx_sRrtb[b];  tb_sRp = ctx_sRptb[b]
    tb_bRR = ctx_bRR_tb[b]; tb_M   = ctx_M_tb[b]

    f_kf   = ctx_kf_f[b];   f_Kc   = ctx_Kc_f[b]
    f_cr   = ctx_cr_f[b];   f_cp   = ctx_cp_f[b];  f_dG  = ctx_dG0_f[b]
    f_sRr  = ctx_sRrf[b];   f_sRp  = ctx_sRpf[b]
    f_k0   = ctx_k0_f[b];   f_ki   = ctx_kinf_f[b]
    f_M    = ctx_M_f[b];    f_Pr   = ctx_Pr_f[b]
    f_F    = ctx_F_f[b];    f_kb   = ctx_kb_f[b]
    f_Fc   = ctx_Fc_f[b];   f_lFc  = ctx_lFc_f[b]
    f_Cp   = ctx_Cp_f[b];   f_Np   = ctx_Np_f[b]
    f_f1   = ctx_f1_f[b];   f_exp  = ctx_exp_f[b]

    # ── PRÉDICTEUR : Euler explicite depuis x0 ────────────────────────
    T_loc = sh_xk[0]
    i = tid
    while i < n_sp:
        sh_C[i] = sh_xk[i + 1]
        i += n_threads
    cuda.syncthreads()

    _eval_rhs_and_context_pb(
        tid, n_threads,
        T_loc, sh_C,
        e_kf, e_Kc, e_cr, e_cp, e_dG, e_sRr, e_sRp,
        tb_kf, tb_Kc, tb_cr, tb_cp, tb_dG, tb_sRr, tb_sRp, tb_bRR, tb_M,
        f_kf, f_Kc, f_cr, f_cp, f_dG, f_sRr, f_sRp,
        f_k0, f_ki, f_M, f_Pr, f_F, f_kb,
        f_Fc, f_lFc, f_Cp, f_Np, f_f1, f_exp,
        sh_om, n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )
    _adiabatic_rhs_pb(
        tid, n_threads,
        T_loc, sh_C, sh_om, n_sp, thL, thH, thM,
        sh_dTC, sh_scratch,
    )

    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i] + h * sh_dTC[i]
        i += n_threads
    cuda.syncthreads()
    if tid == 0:
        if sh_xk[0] < 1.0:
            sh_xk[0] = 1.0
    i = tid + 1
    while i < n_state:
        if sh_xk[i] < 0.0:
            sh_xk[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    # ── BOUCLE DE NEWTON ──────────────────────────────────────────────
    for nit in range(newton_maxiter):
        T_loc = sh_xk[0]
        i = tid
        while i < n_sp:
            sh_C[i] = sh_xk[i + 1]
            i += n_threads
        cuda.syncthreads()

        _eval_rhs_and_context_pb(
            tid, n_threads,
            T_loc, sh_C,
            e_kf, e_Kc, e_cr, e_cp, e_dG, e_sRr, e_sRp,
            tb_kf, tb_Kc, tb_cr, tb_cp, tb_dG, tb_sRr, tb_sRp, tb_bRR, tb_M,
            f_kf, f_Kc, f_cr, f_cp, f_dG, f_sRr, f_sRp,
            f_k0, f_ki, f_M, f_Pr, f_F, f_kb,
            f_Fc, f_lFc, f_Cp, f_Np, f_f1, f_exp,
            sh_om, n_sp, thL, thH, thM,
            n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
            n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
            n_f, fA0, fb0, fE0, fAi, fbi, fEi,
            fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
        )
        _adiabatic_rhs_pb(
            tid, n_threads,
            T_loc, sh_C, sh_om, n_sp, thL, thH, thM,
            sh_dTC, sh_scratch,
        )

        # Calcul du résidu et de sh_rhs = -F(xk)
        acc = float64(0.0)
        i   = tid
        while i < n_state:
            fi        = sh_xk[i] - TC_in[b, i] - h * sh_dTC[i]
            sh_rhs[i] = -fi
            acc      += fi * fi
            i += n_threads
        _block_reduce_sum(acc, sh_scratch, tid, n_threads)
        cuda.syncthreads()
        res_norm = math.sqrt(sh_scratch[0])

        if res_norm < newton_tol:
            if tid == 0:
                converged[b] = int32(1)
            cuda.syncthreads()
            break

        _gmres_pb(
            tid, n_threads,
            T_loc, sh_C, h, sh_om,
            sh_rhs, sh_dx,
            sh_w, sh_g, sh_y, sh_cs, sh_sn, sh_H,
            sh_scratch, sh_Jvo, sh_vC,
            gm_V[b],
            n_state, gmres_restart, gmres_tol, gmres_maxiter,
            e_kf, e_Kc, e_cr, e_cp, e_dG, e_sRr, e_sRp,
            tb_kf, tb_Kc, tb_cr, tb_cp, tb_dG, tb_sRr, tb_sRp, tb_bRR, tb_M,
            f_kf, f_Kc, f_cr, f_cp, f_dG, f_sRr, f_sRp,
            f_k0, f_ki, f_M, f_Pr, f_F, f_kb,
            f_Fc, f_lFc, f_Cp, f_Np, f_f1, f_exp,
            n_sp, thL, thH, thM,
            n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
            n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
            n_f, fA0, fb0, fE0, fAi, fbi, fEi,
            fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
        )

        i = tid
        while i < n_state:
            sh_xk[i] += sh_dx[i]
            i += n_threads
        cuda.syncthreads()
        if tid == 0:
            if sh_xk[0] < 1.0:
                sh_xk[0] = 1.0
        i = tid + 1
        while i < n_state:
            if sh_xk[i] < 0.0:
                sh_xk[i] = float64(0.0)
            i += n_threads
        cuda.syncthreads()

    # ── Écriture du résultat en mémoire globale ───────────────────────
    i = tid
    while i < n_state:
        TC_out[b, i] = sh_xk[i]
        i += n_threads
    cuda.syncthreads()