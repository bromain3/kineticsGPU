"""
kernels/per_thread/chemistry_kernels.py — Kernel principal per-thread + utilitaires.

Architecture: 1 thread CUDA = 1 réacteur.

Kernels exportés :
    backward_euler_kernel   — kernel principal d'intégration
    _fill_int_zero          — remise à zéro du tableau de convergence
    _store_snapshot         — sauvegarde d'un snapshot (TC, conv)

Constantes exportées :
    MAX_N_STATE, THREADS_PER_BLOCK

Ce module est le point d'entrée public pour l'intégrateur per-thread.
Les fonctions device internes sont dans context.py, jvp.py, gmres.py
et shared/.
"""

import math
from numba import cuda, float64, int32

from kineticsGPU.cuda.kernels.per_thread.reactor_odes import _adiabatic_rhs
from .context import _eval_rhs_and_context
from .gmres import _gmres

MAX_N_STATE       = 54   # GRI30: 53 espèces + T
THREADS_PER_BLOCK = 64


# =============================================================================
#  Kernel principal — un thread par réacteur
# =============================================================================

@cuda.jit(cache=True)
def backward_euler_kernel(
    TC_in, TC_out,
    h_scalar,
    newton_tol, newton_maxiter,
    gmres_restart, gmres_tol, gmres_maxiter,
    converged,
    # ── Scratch GMRES (global, shape (B, ...)) ────────────────────────
    gm_V, gm_H, gm_cs, gm_sn, gm_g, gm_w, gm_y,
    # ── Contexte cinétique (global, shape (B, n_reactions) ou (B, n_flat)) ──
    ctx_kf_e,   ctx_Kc_e,  ctx_cr_e,  ctx_cp_e,  ctx_dG0_e,
    ctx_sRre,   ctx_sRpe,
    ctx_kf_tb,  ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb,  ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f,   ctx_Kc_f,  ctx_cr_f,  ctx_cp_f,  ctx_dG0_f,
    ctx_sRrf,   ctx_sRpf,
    ctx_k0_f,   ctx_kinf_f, ctx_M_f,  ctx_Pr_f,  ctx_F_f,   ctx_kb_f,
    ctx_Fc_f,   ctx_lFc_f, ctx_Cp_f,  ctx_Np_f,  ctx_f1_f,  ctx_exp_f,
    # ── Mécanisme (lecture seule, partagé par tous les threads) ───────
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1
    h       = h_scalar

    # Scratch local (tient dans les registres pour GRI30)
    xk      = cuda.local.array(MAX_N_STATE, float64)
    C       = cuda.local.array(MAX_N_STATE, float64)
    om      = cuda.local.array(MAX_N_STATE, float64)
    dTC     = cuda.local.array(MAX_N_STATE, float64)
    rhs     = cuda.local.array(MAX_N_STATE, float64)
    dx      = cuda.local.array(MAX_N_STATE, float64)
    vC_tmp  = cuda.local.array(MAX_N_STATE, float64)
    Jvo_tmp = cuda.local.array(MAX_N_STATE, float64)

    for i in range(n_state):
        xk[i] = TC_in[b, i]

    # ── PRÉDICTEUR : Euler explicite depuis x0 ────────────────────────
    T_loc = xk[0]
    for s in range(n_sp): C[s] = xk[s + 1]
    _eval_rhs_and_context(
        T_loc, C,
        ctx_kf_e[b],   ctx_Kc_e[b],  ctx_cr_e[b],  ctx_cp_e[b],  ctx_dG0_e[b],
        ctx_sRre[b],   ctx_sRpe[b],
        ctx_kf_tb[b],  ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
        ctx_sRrtb[b],  ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
        ctx_kf_f[b],   ctx_Kc_f[b],  ctx_cr_f[b],  ctx_cp_f[b],  ctx_dG0_f[b],
        ctx_sRrf[b],   ctx_sRpf[b],
        ctx_k0_f[b],   ctx_kinf_f[b], ctx_M_f[b],  ctx_Pr_f[b],  ctx_F_f[b],
        ctx_kb_f[b],
        ctx_Fc_f[b],   ctx_lFc_f[b], ctx_Cp_f[b],  ctx_Np_f[b],
        ctx_f1_f[b],   ctx_exp_f[b],
        om, n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )
    _adiabatic_rhs(T_loc, C, om, n_sp, thL, thH, thM, dTC)
    for i in range(n_state):
        xk[i] = TC_in[b, i] + h * dTC[i]
    if xk[0] < 1.0: xk[0] = 1.0
    for s in range(n_sp):
        if xk[s + 1] < 0.0: xk[s + 1] = float64(0.0)

    # ── BOUCLE DE NEWTON ──────────────────────────────────────────────
    for nit in range(newton_maxiter):
        T_loc = xk[0]
        for s in range(n_sp): C[s] = xk[s + 1]

        _eval_rhs_and_context(
            T_loc, C,
            ctx_kf_e[b],   ctx_Kc_e[b],  ctx_cr_e[b],  ctx_cp_e[b],  ctx_dG0_e[b],
            ctx_sRre[b],   ctx_sRpe[b],
            ctx_kf_tb[b],  ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
            ctx_sRrtb[b],  ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
            ctx_kf_f[b],   ctx_Kc_f[b],  ctx_cr_f[b],  ctx_cp_f[b],  ctx_dG0_f[b],
            ctx_sRrf[b],   ctx_sRpf[b],
            ctx_k0_f[b],   ctx_kinf_f[b], ctx_M_f[b],  ctx_Pr_f[b],  ctx_F_f[b],
            ctx_kb_f[b],
            ctx_Fc_f[b],   ctx_lFc_f[b], ctx_Cp_f[b],  ctx_Np_f[b],
            ctx_f1_f[b],   ctx_exp_f[b],
            om, n_sp, thL, thH, thM,
            n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
            n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
            n_f, fA0, fb0, fE0, fAi, fbi, fEi,
            fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
        )
        _adiabatic_rhs(T_loc, C, om, n_sp, thL, thH, thM, dTC)

        res2 = float64(0.0)
        for i in range(n_state):
            fi     = xk[i] - TC_in[b, i] - h * dTC[i]
            rhs[i] = -fi
            res2  += fi * fi
        if math.sqrt(res2) < newton_tol:
            converged[b] = int32(1)
            break

        _gmres(
            T_loc, C, h, om,
            rhs, dx,
            gm_V[b], gm_H[b], gm_cs[b], gm_sn[b], gm_g[b],
            gm_w[b], gm_y[b],
            vC_tmp, Jvo_tmp,
            n_state, gmres_restart, gmres_tol, gmres_maxiter,
            ctx_kf_e[b],   ctx_Kc_e[b],  ctx_cr_e[b],  ctx_cp_e[b],  ctx_dG0_e[b],
            ctx_sRre[b],   ctx_sRpe[b],
            ctx_kf_tb[b],  ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
            ctx_sRrtb[b],  ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
            ctx_kf_f[b],   ctx_Kc_f[b],  ctx_cr_f[b],  ctx_cp_f[b],  ctx_dG0_f[b],
            ctx_sRrf[b],   ctx_sRpf[b],
            ctx_k0_f[b],   ctx_kinf_f[b], ctx_M_f[b],  ctx_Pr_f[b],  ctx_F_f[b],
            ctx_kb_f[b],
            ctx_Fc_f[b],   ctx_lFc_f[b], ctx_Cp_f[b],  ctx_Np_f[b],
            ctx_f1_f[b],   ctx_exp_f[b],
            n_sp, thL, thH, thM,
            n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
            n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
            n_f, fA0, fb0, fE0, fAi, fbi, fEi,
            fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
        )

        for i in range(n_state):
            xk[i] += dx[i]
        if xk[0] < 1.0: xk[0] = 1.0
        for s in range(n_sp):
            if xk[s + 1] < 0.0: xk[s + 1] = float64(0.0)

    for i in range(n_state):
        TC_out[b, i] = xk[i]


# =============================================================================
#  Kernels utilitaires
# =============================================================================

@cuda.jit
def _fill_int_zero(arr):
    """Remise à zéro d'un tableau int32 (1 thread = 1 élément)."""
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < arr.shape[0]:
        arr[i] = int32(0)


@cuda.jit
def _store_snapshot(TC_src, conv_src, TC_hist, conv_hist, snap_idx):
    """
    Copie la ligne b de TC_src / conv_src dans TC_hist[snap_idx] / conv_hist[snap_idx].
    Lance avec le même grid que backward_euler_kernel.
    """
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= TC_src.shape[0]:
        return
    for i in range(TC_src.shape[1]):
        TC_hist[snap_idx, b, i] = TC_src[b, i]
    conv_hist[snap_idx, b] = conv_src[b]