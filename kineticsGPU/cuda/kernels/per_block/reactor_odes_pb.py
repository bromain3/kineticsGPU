"""
kernels/per_block/reactor_odes_pb.py — Équations du réacteur adiabatique (per-block).

Version parallèle des équations du réacteur :
    - RHS adiabatique  : _adiabatic_rhs_pb
    - JVP adiabatique  : _adiabatic_jvp_pb

Différences vs shared/reactor_odes.py (version scalaire) :
    - Boucles sur les espèces parallélisées avec stride tid/n_threads
    - Réductions (cv_mix, su_om, ntot, etc.) via _block_reduce_sum
    - cuda.syncthreads() après chaque écriture partagée
    - Thread 0 écrit dTCdt[0] / Jv_TC[0]
"""

import math
from numba import cuda, float64

from ..shared.thermo import _load_coef7, _h_RT, _s_R, _cp_R, R_CONST, EPS
from .reductions import _block_reduce_sum
import numpy as np

from ..shared.thermo import EPS
from .context_pb import _eval_rhs_and_context_pb
from .jacobian_pb import _eval_jacobian_pb


@cuda.jit(device=True)
def _adiabatic_rhs_pb(
    tid, n_threads,
    T, C, omega, n_sp, thL, thH, thM,
    dTCdt,    # tableau partagé de sortie, longueur n_sp+1
    scratch,  # tableau partagé pour réductions, longueur >= n_threads
):
    """
    RHS adiabatique parallèle (per-block).

    dT/dt = -sum(u_s * omega_s) / (n_tot * cv_mix)
    dC_s/dt = omega_s

    scratch et dTCdt doivent être en mémoire partagée.
    """
    c7 = cuda.local.array(7, float64)

    # ntot = sum(C[s]) — réduction parallèle
    acc = float64(0.0)
    i   = tid
    while i < n_sp:
        acc += C[i]
        i   += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)
    cuda.syncthreads()
    ntot = scratch[0]

    # cv_mix et su_om — chaque thread accumule sa portion
    cv_loc = float64(0.0)
    su_loc = float64(0.0)
    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        cp_r = _cp_R(T, c7)
        cv_s = R_CONST * (cp_r - 1.0)
        hRT  = _h_RT(T, c7)
        u_s  = R_CONST * T * hRT - R_CONST * T
        Xs   = C[i] / ntot if ntot > EPS else float64(0.0)
        cv_loc += Xs * cv_s
        su_loc += u_s * omega[i]
        i += n_threads

    _block_reduce_sum(cv_loc, scratch, tid, n_threads)
    cuda.syncthreads()
    cv_mix = scratch[0]

    _block_reduce_sum(su_loc, scratch, tid, n_threads)
    cuda.syncthreads()
    su_om = scratch[0]

    if tid == 0:
        denom    = ntot * cv_mix
        dTCdt[0] = -su_om / denom if abs(denom) > EPS else float64(0.0)
    cuda.syncthreads()

    i = tid
    while i < n_sp:
        dTCdt[i + 1] = omega[i]
        i += n_threads
    cuda.syncthreads()


@cuda.jit(device=True)
def _adiabatic_jvp_pb(
    tid, n_threads,
    T, C, vT, vC, omega, Jv_om, n_sp, thL, thH, thM,
    Jv_TC,    # tableau partagé de sortie, longueur n_sp+1
    scratch,  # tableau partagé pour réductions, longueur >= n_threads
):
    """
    JVP du RHS adiabatique (per-block).

    Dérivée directionnelle de _adiabatic_rhs_pb dans la direction (vT, vC).
    Nécessite omega et Jv_om déjà calculés au point de base.
    """
    c7 = cuda.local.array(7, float64)

    # ntot, dntot — réductions parallèles
    acc_n  = float64(0.0); acc_dn = float64(0.0)
    i = tid
    while i < n_sp:
        acc_n  += C[i]
        acc_dn += vC[i]
        i += n_threads
    _block_reduce_sum(acc_n, scratch, tid, n_threads)
    cuda.syncthreads()
    ntot = scratch[0]
    _block_reduce_sum(acc_dn, scratch, tid, n_threads)
    cuda.syncthreads()
    dntot = scratch[0]

    # Accumulation parallèle de cv_mix, dcv, up, dup
    lcv  = float64(0.0); ldcv = float64(0.0)
    lup  = float64(0.0); ldup = float64(0.0)
    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        T2 = T * T; T3 = T2 * T; T4 = T3 * T
        cpR   = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        cv_s  = R_CONST * (cpR - 1.0)
        dcvdT = R_CONST * (c7[1] + 2.0*c7[2]*T + 3.0*c7[3]*T2 + 4.0*c7[4]*T3)
        hRT   = (c7[0] + c7[1]*T/2.0 + c7[2]*T2/3.0
                 + c7[3]*T3/4.0 + c7[4]*T4/5.0 + c7[5]/T)
        u_s   = R_CONST * T * hRT - R_CONST * T
        Xs    = C[i] / ntot if ntot > EPS else float64(0.0)
        dXs   = ((vC[i] / ntot - C[i] * dntot / (ntot * ntot))
                 if ntot > EPS else float64(0.0))
        lcv  += Xs  * cv_s
        ldcv += dXs * cv_s + Xs * dcvdT * vT
        lup  += u_s * omega[i]
        ldup += u_s * Jv_om[i] + cv_s * omega[i] * vT
        i += n_threads

    _block_reduce_sum(lcv, scratch, tid, n_threads)
    cuda.syncthreads()
    cv_mix = scratch[0]

    _block_reduce_sum(ldcv, scratch, tid, n_threads)
    cuda.syncthreads()
    dcv = scratch[0]

    _block_reduce_sum(lup, scratch, tid, n_threads)
    cuda.syncthreads()
    up = scratch[0]

    _block_reduce_sum(ldup, scratch, tid, n_threads)
    cuda.syncthreads()
    dup = scratch[0]

    if tid == 0:
        denom = ntot * cv_mix
        if abs(denom) > EPS:
            Jv_TC[0] = (-dup / denom
                        + up * dntot / (denom * ntot)
                        + up * dcv / (ntot * cv_mix * cv_mix))
        else:
            Jv_TC[0] = float64(0.0)
    cuda.syncthreads()

    i = tid
    while i < n_sp:
        Jv_TC[i + 1] = Jv_om[i]
        i += n_threads
    cuda.syncthreads()

@cuda.jit(device=True)
def _adiabatic_full_jacobian_pb(
    tid, n_threads,
    T, C, omega, J_chem,
    n_sp, thL, thH, thM,
    J_full,   # mémoire partagée ou globale, taille n_state²
    scratch,  # mémoire partagée pour réductions, taille >= n_threads
):
    """
    Jacobienne complète du réacteur adiabatique (per-block).

    J_full[i * n_state + j] = d(RHS_i) / d(xk_j)

    Ligne i=0   → d(dT/dt)/d(xk_j)   (calculée analytiquement)
    Ligne i=s+1 → d(dC_s/dt)/d(xk_j) = J_chem[s * n_state + j]

    Architecture
    ------------
    - Passe 1 : réductions parallèles pour ntot, cv_mix, U_num, dcv_dT
    - Passe 2 : boucle sur j_col — chaque thread calcule sa contribution
      à la ligne i=0 en utilisant les quantités réduites.
    - Passe 3 : copie des lignes espèces depuis J_chem.

    La boucle colonne (j_col) est parallélisée : thread tid traite les
    colonnes tid, tid+n_threads, … de la ligne d'énergie.
    """
    n_state = n_sp + 1
    c7      = cuda.local.array(7, float64)

    # ── Passe 1 : réductions parallèles ───────────────────────────────

    # ntot = Σ C_s
    acc = float64(0.0)
    i   = tid
    while i < n_sp:
        acc += C[i]
        i   += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)
    cuda.syncthreads()
    ntot = scratch[0]

    # cv_mix, U_num, dcv_dT — une seule passe espèces
    lcv  = float64(0.0)   # contribution locale à cv_mix
    lU   = float64(0.0)   # contribution locale à U_num = Σ u_s*omega_s
    ldcv = float64(0.0)   # contribution locale à d(cv_mix)/dT = Σ X_s*d(cv_s)/dT

    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        T2  = T * T; T3 = T2 * T; T4 = T3 * T
        cpR = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        cv_s    = R_CONST * (cpR - float64(1.0))
        dcvs_dT = R_CONST * (c7[1]
                             + float64(2.0)*c7[2]*T
                             + float64(3.0)*c7[3]*T2
                             + float64(4.0)*c7[4]*T3)
        hRT = (c7[0] + c7[1]*T/float64(2.0) + c7[2]*T2/float64(3.0)
               + c7[3]*T3/float64(4.0) + c7[4]*T4/float64(5.0) + c7[5]/T)
        u_s = R_CONST * T * hRT - R_CONST * T
        Xs  = C[i] / ntot if ntot > EPS else float64(0.0)

        lcv  += Xs  * cv_s
        lU   += u_s * omega[i]
        ldcv += Xs  * dcvs_dT
        i    += n_threads

    _block_reduce_sum(lcv, scratch, tid, n_threads)
    cuda.syncthreads()
    cv_mix = scratch[0]

    _block_reduce_sum(lU, scratch, tid, n_threads)
    cuda.syncthreads()
    U_num = scratch[0]

    _block_reduce_sum(ldcv, scratch, tid, n_threads)
    cuda.syncthreads()
    dcv_dT = scratch[0]

    # Dérivées du dénominateur D = ntot * cv_mix
    D      = ntot * cv_mix
    D_safe = D if abs(D) > EPS else (EPS if D >= float64(0.0) else -EPS)
    D_sq   = D_safe * D_safe

    # ── Passe 2 : ligne i=0 — parallélisation sur les colonnes j ──────
    #
    # Chaque thread calcule dU_j et dD_j pour ses colonnes assignées,
    # puis écrit J_full[0 * n_state + j] directement (pas de conflit
    # car chaque thread écrit dans une colonne distincte).

    j = tid
    while j < n_state:
        # d(U_num)/d(xk_j) = Σ_s u_s * J_chem[s,j]  +  correction_j
        dU_j = float64(0.0)
        for s in range(n_sp):
            _load_coef7(T, s, thM, thL, thH, c7)
            T2  = T * T; T3 = T2 * T; T4 = T3 * T
            hRT = (c7[0] + c7[1]*T/float64(2.0) + c7[2]*T2/float64(3.0)
                   + c7[3]*T3/float64(4.0) + c7[4]*T4/float64(5.0) + c7[5]/T)
            u_s  = R_CONST * T * hRT - R_CONST * T
            dU_j += u_s * J_chem[s * n_state + j]

        if j == 0:
            # terme (du_s/dT)*omega_s = cv_s*omega_s, colonne T uniquement
            for s in range(n_sp):
                _load_coef7(T, s, thM, thL, thH, c7)
                T2  = T * T; T3 = T2 * T; T4 = T3 * T
                cpR  = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
                cv_s = R_CONST * (cpR - float64(1.0))
                dU_j += cv_s * omega[s]
            # d(D)/dT = ntot * d(cv_mix)/dT
            dD_j = ntot * dcv_dT
        else:
            # d(D)/d(C_k) = cv_k
            s_k  = j - 1
            _load_coef7(T, s_k, thM, thL, thH, c7)
            T2   = T * T; T3 = T2 * T; T4 = T3 * T
            cpR  = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
            cv_k = R_CONST * (cpR - float64(1.0))
            dD_j = cv_k

        J_full[0 * n_state + j] = -(dU_j * D_safe - U_num * dD_j) / D_sq
        j += n_threads
    cuda.syncthreads()

    # ── Passe 3 : lignes espèces — copie depuis J_chem ─────────────────
    # J_full[(s+1)*n_state + j] = J_chem[s*n_state + j]
    # On itère sur l'index plat (s*n_state + j) en stride
    idx = tid
    while idx < n_sp * n_state:
        s = idx // n_state
        j = idx  % n_state
        J_full[(s + 1) * n_state + j] = J_chem[idx]
        idx += n_threads
    cuda.syncthreads()

"""
kernels/per_block/full_jacobian_pb.py — Jacobienne complète du réacteur adiabatique (per-block).

Architecture : 1 bloc = 1 réacteur. Les 64 threads coopèrent.

Calcule J de taille (n_state × n_state) :

    J[i, j] = d(RHS_i) / d(xk_j)

avec RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)
et   xk  = (T, C_0, …, C_{n_sp-1})

Convention row-major :
    J_out[i * n_state + j] = d(RHS_i) / d(xk_j)

Pipeline par réacteur (un bloc) :
    1. _eval_rhs_and_context_pb  — contexte cinétique + omega
    2. _eval_jacobian_pb         — jacobienne chimique (n_sp × n_state)
    3. _adiabatic_full_jacobian_pb — ligne énergie + copie lignes espèces
"""

_MAX_N_STATE       = 54
_MAX_N_STATE2      = _MAX_N_STATE * _MAX_N_STATE
_THREADS_PER_BLOCK = 64


# =============================================================================
#  Kernel CUDA global (un bloc = un réacteur)
# =============================================================================

@cuda.jit(cache=True)
def _full_jacobian_kernel_pb(
    TC_in,           # (B, n_state)    float64
    J_all,           # (B, n_state²)   float64  — sortie
    ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e,
    ctx_sRre, ctx_sRpe,
    ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f,
    ctx_sRrf, ctx_sRpf,
    ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
    ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b         = cuda.blockIdx.x
    tid       = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1

    # ── Mémoire partagée ──────────────────────────────────────────────
    sh_xk      = cuda.shared.array(_MAX_N_STATE,                   float64)
    sh_om      = cuda.shared.array(_MAX_N_STATE,                   float64)
    sh_col     = cuda.shared.array(_MAX_N_STATE,                   float64)
    sh_J_chem  = cuda.shared.array(_MAX_N_STATE2,    float64)
    sh_J_full  = cuda.shared.array(_MAX_N_STATE2,    float64)
    sh_scratch = cuda.shared.array(_THREADS_PER_BLOCK,             float64)

    # Chargement de l'état (T, C_0, …)
    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()

    T_loc = sh_xk[0]

    # ── Aliases contexte cinétique ─────────────────────────────────────
    e_kf  = ctx_kf_e[b];  e_Kc  = ctx_Kc_e[b]
    e_cr  = ctx_cr_e[b];  e_cp  = ctx_cp_e[b];  e_dG = ctx_dG0_e[b]
    e_sRr = ctx_sRre[b];  e_sRp = ctx_sRpe[b]

    tb_kf  = ctx_kf_tb[b];  tb_Kc  = ctx_Kc_tb[b]
    tb_cr  = ctx_cr_tb[b];  tb_cp  = ctx_cp_tb[b]; tb_dG = ctx_dG0_tb[b]
    tb_sRr = ctx_sRrtb[b];  tb_sRp = ctx_sRptb[b]
    tb_bRR = ctx_bRR_tb[b]; tb_M   = ctx_M_tb[b]

    f_kf  = ctx_kf_f[b];  f_Kc   = ctx_Kc_f[b]
    f_cr  = ctx_cr_f[b];  f_cp   = ctx_cp_f[b];  f_dG  = ctx_dG0_f[b]
    f_sRr = ctx_sRrf[b];  f_sRp  = ctx_sRpf[b]
    f_k0  = ctx_k0_f[b];  f_ki   = ctx_kinf_f[b]
    f_M   = ctx_M_f[b];   f_Pr   = ctx_Pr_f[b]
    f_F   = ctx_F_f[b];   f_kb   = ctx_kb_f[b]
    f_Fc  = ctx_Fc_f[b];  f_lFc  = ctx_lFc_f[b]
    f_Cp  = ctx_Cp_f[b];  f_Np   = ctx_Np_f[b]
    f_f1  = ctx_f1_f[b];  f_exp  = ctx_exp_f[b]

    # ── Étape 1 : contexte cinétique + omega ──────────────────────────
    _eval_rhs_and_context_pb(
        tid, n_threads,
        T_loc, sh_xk[1:],
        e_kf,  e_Kc,  e_cr,  e_cp,  e_dG,  e_sRr,  e_sRp,
        tb_kf, tb_Kc, tb_cr, tb_cp, tb_dG, tb_sRr, tb_sRp, tb_bRR, tb_M,
        f_kf,  f_Kc,  f_cr,  f_cp,  f_dG,  f_sRr,  f_sRp,
        f_k0,  f_ki,  f_M,   f_Pr,  f_F,   f_kb,
        f_Fc,  f_lFc, f_Cp,  f_Np,  f_f1,  f_exp,
        sh_om, n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # ── Étape 2 : jacobienne chimique pure → sh_J_chem (n_sp × n_state) ─
    _eval_jacobian_pb(
        tid, n_threads,
        T_loc, sh_xk[1:],
        sh_J_chem,
        e_kf,  e_Kc,  e_cr,  e_cp,  e_dG,  e_sRr,  e_sRp,
        tb_kf, tb_Kc, tb_cr, tb_cp, tb_dG, tb_sRr, tb_sRp, tb_bRR, tb_M,
        f_kf,  f_Kc,  f_cr,  f_cp,  f_dG,  f_sRr,  f_sRp,
        f_k0,  f_ki,  f_M,   f_Pr,  f_F,   f_kb,
        f_Fc,  f_lFc, f_Cp,  f_Np,  f_f1,  f_exp,
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
        sh_col,
    )

    # ── Étape 3 : jacobienne complète → sh_J_full (n_state × n_state) ─
    _adiabatic_full_jacobian_pb(
        tid, n_threads,
        T_loc, sh_xk[1:], sh_om, sh_J_chem,
        n_sp, thL, thH, thM,
        sh_J_full,
        sh_scratch,
    )

    # ── Copie sh_J_full → J_all[b] (mémoire globale) ─────────────────
    idx = tid
    while idx < n_state * n_state:
        J_all[b, idx] = sh_J_full[idx]
        idx += n_threads
    cuda.syncthreads()


# =============================================================================
#  Wrapper Python — CUDAFullJacobianEvaluatorPB
# =============================================================================

class CUDAFullJacobianEvaluatorPB:
    """
    Évalue la jacobienne complète du réacteur adiabatique (per-block).

    J[i, j] = d(RHS_i) / d(xk_j)

    avec RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)
    et   xk  = (T, C_0, …, C_{n_sp-1})

    Usage
    -----
    ev = CUDAFullJacobianEvaluatorPB(mech)
    J  = ev.evaluate(T_arr, C_arr)
    # T_arr : (B,)        float64
    # C_arr : (B, n_sp)   float64
    # J     : (B, n_state, n_state)  float64,  n_state = n_sp + 1

    Indexation de J[b]
    ------------------
    J[b, 0, j]   = d(dT/dt)   / d(xk_j)   ← ligne énergie
    J[b, s+1, j] = d(dC_s/dt) / d(xk_j)   ← ligne espèce s
    J[b, i, 0]   = d(RHS_i)   / dT         ← colonne T
    J[b, i, k+1] = d(RHS_i)   / d(C_k)     ← colonne concentration k

    Différences vs CUDAFullJacobianEvaluatorPT
    -------------------------------------------
    - 1 bloc CUDA par réacteur (64 threads coopèrent via shared memory)
    - sh_J_chem et sh_J_full vivent en shared memory (évite les accès globaux
      intermédiaires, réduit la latence pour les petits batches)
    - Adapté aux cas B petit mais n_sp grand, où la parallélisation
      intra-réacteur compense le faible parallélisme inter-réacteurs
    """

    THREADS_PER_BLOCK = _THREADS_PER_BLOCK

    def __init__(self, mech, dtype=np.float64):
        self._mech    = mech
        self._dtype   = dtype
        self._n_sp    = mech.n_species
        self._n_state = self._n_sp + 1
        self._ctx     = None
        self._B_alloc = -1

    # ------------------------------------------------------------------
    def _alloc_context(self, B):
        dt = self._dtype
        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff

        def z(shape):
            return cuda.device_array(shape, dtype=dt)

        n_e_flat  = e.react_offset[-1]  if e.n_reactions  else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1

        self._ctx = dict(
            kf_e   = z((B, e.n_reactions)),    Kc_e   = z((B, e.n_reactions)),
            cr_e   = z((B, e.n_reactions)),    cp_e   = z((B, e.n_reactions)),
            dG0_e  = z((B, e.n_reactions)),
            sRre   = z((B, n_e_flat)),         sRpe   = z((B, n_e_flat)),

            kf_tb  = z((B, tb.n_reactions)),   Kc_tb  = z((B, tb.n_reactions)),
            cr_tb  = z((B, tb.n_reactions)),   cp_tb  = z((B, tb.n_reactions)),
            dG0_tb = z((B, tb.n_reactions)),
            sRrtb  = z((B, n_tb_flat)),        sRptb  = z((B, n_tb_flat)),
            bRR_tb = z((B, tb.n_reactions)),   M_tb   = z((B, tb.n_reactions)),

            kf_f   = z((B, f.n_reactions)),    Kc_f   = z((B, f.n_reactions)),
            cr_f   = z((B, f.n_reactions)),    cp_f   = z((B, f.n_reactions)),
            dG0_f  = z((B, f.n_reactions)),
            sRrf   = z((B, f.react_offset[-1] if f.n_reactions else 1)),
            sRpf   = z((B, f.react_offset[-1] if f.n_reactions else 1)),
            k0_f   = z((B, f.n_reactions)),    kinf_f = z((B, f.n_reactions)),
            M_f    = z((B, f.n_reactions)),    Pr_f   = z((B, f.n_reactions)),
            F_f    = z((B, f.n_reactions)),    kb_f   = z((B, f.n_reactions)),
            Fc_f   = z((B, f.n_reactions)),    lFc_f  = z((B, f.n_reactions)),
            Cp_f   = z((B, f.n_reactions)),    Np_f   = z((B, f.n_reactions)),
            f1_f   = z((B, f.n_reactions)),    exp_f  = z((B, f.n_reactions)),
        )
        self._B_alloc = B

    # ------------------------------------------------------------------
    def evaluate(self, T_arr, C_arr):
        """
        Paramètres
        ----------
        T_arr : array-like (B,)       — températures (K)
        C_arr : array-like (B, n_sp)  — concentrations (mol/m³)

        Retourne
        --------
        J : np.ndarray (B, n_state, n_state)  float64
        """
        T_arr = np.asarray(T_arr, dtype=self._dtype)
        C_arr = np.asarray(C_arr, dtype=self._dtype)
        B     = T_arr.shape[0]
        nsp   = self._n_sp
        nst   = self._n_state

        if self._B_alloc != B:
            self._alloc_context(B)

        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0]  = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)

        # Sortie GPU : (B, n_state²) — stockage row-major du noyau
        d_J = cuda.device_array((B, nst * nst), dtype=self._dtype)

        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff
        cx = self._ctx

        _full_jacobian_kernel_pb[B, self.THREADS_PER_BLOCK](
            d_TC, d_J,
            cx['kf_e'],  cx['Kc_e'],  cx['cr_e'],  cx['cp_e'],  cx['dG0_e'],
            cx['sRre'],  cx['sRpe'],
            cx['kf_tb'], cx['Kc_tb'], cx['cr_tb'], cx['cp_tb'], cx['dG0_tb'],
            cx['sRrtb'], cx['sRptb'], cx['bRR_tb'], cx['M_tb'],
            cx['kf_f'],  cx['Kc_f'],  cx['cr_f'],  cx['cp_f'],  cx['dG0_f'],
            cx['sRrf'],  cx['sRpf'],
            cx['k0_f'],  cx['kinf_f'], cx['M_f'],  cx['Pr_f'],  cx['F_f'],
            cx['kb_f'],
            cx['Fc_f'],  cx['lFc_f'], cx['Cp_f'],  cx['Np_f'],
            cx['f1_f'],  cx['exp_f'],
            m.n_species,
            m.thermo.coef_low, m.thermo.coef_high, m.thermo.Tmid,
            e.n_reactions, e.A, e.n, e.Ea, e.oneWay,
            e.delta_n, e.react_idx, e.react_sto,
            e.react_offset, e.prod_idx, e.prod_sto, e.prod_offset,
            tb.n_reactions, tb.A, tb.n, tb.Ea, tb.oneWay,
            tb.delta_n, tb.react_idx, tb.react_sto,
            tb.react_offset, tb.prod_idx, tb.prod_sto, tb.prod_offset, tb.eff_matrix,
            f.n_reactions, f.A_k0, f.b_k0, f.Ea_k0, f.A_kinf, f.b_kinf, f.Ea_kinf,
            f.oneWay, f.delta_n,
            f.troe_A, f.troe_T1, f.troe_T2, f.troe_T3, f.is_troe,
            f.react_idx, f.react_sto, f.react_offset,
            f.prod_idx, f.prod_sto, f.prod_offset, f.eff_matrix,
        )
        cuda.synchronize()

        return d_J.copy_to_host().reshape(B, nst, nst)


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Jacobienne per-block du RHS isobare
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _constantP_full_jacobian_pb(
        tid, n_threads,
        T, C, omega, J_chem,
        n_sp, thL, thH, thM,
        J_full,  # mémoire partagée, taille n_state²
        scratch,  # mémoire partagée pour réductions, taille >= n_threads
):
    """
    Jacobienne complète du réacteur isobare (per-block).

    J_full[i * n_state + j] = d(RHS_i) / d(xk_j)

    Ligne i=0 :
        dT/dt = -H_num / D,    D = n_tot·cp_mix,  H_num = Σ h_s·ω_s
        → calcul analytique, parallélisation sur les colonnes j.

    Lignes i=s+1 :
        dC_s/dt = ω_s  - C_s·δ,    δ = Σω_k/n_tot + (dT/dt)/T
        → d(dC_s/dt)/d(xk_j) = J_chem[s,j] - C_s·(dδ/d(xk_j)) - δ·𝟙_{j=s+1}

    Architecture (identique au cas adiabatique) :
        Passe 1 — réductions parallèles (ntot, cp_mix, H_num, dcp_dT, omega_sum)
        Passe 2 — boucle colonnes j pour la ligne énergie
        Passe 3 — boucle plate sur (s, j) pour les lignes espèces
    """
    n_state = n_sp + 1
    c7 = cuda.local.array(7, float64)

    # ── Passe 1 : réductions parallèles ──────────────────────────────────────

    # ntot = Σ C_s
    acc = float64(0.0)
    i = tid
    while i < n_sp:
        acc += C[i]
        i += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)
    cuda.syncthreads()
    ntot = scratch[0]

    # cp_mix, H_num, dcp_dT, omega_sum — une seule passe espèces
    lcp = float64(0.0)  # Σ X_s·cp_s
    lH = float64(0.0)  # Σ h_s·ω_s
    ldcp = float64(0.0)  # Σ X_s·d(cp_s)/dT
    lom = float64(0.0)  # Σ ω_s

    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        T2 = T * T;
        T3 = T2 * T;
        T4 = T3 * T
        cpR = c7[0] + c7[1] * T + c7[2] * T2 + c7[3] * T3 + c7[4] * T4
        cp_s = R_CONST * cpR
        dcps_dT = R_CONST * (c7[1]
                             + float64(2.0) * c7[2] * T
                             + float64(3.0) * c7[3] * T2
                             + float64(4.0) * c7[4] * T3)
        hRT = (c7[0] + c7[1] * T / float64(2.0) + c7[2] * T2 / float64(3.0)
               + c7[3] * T3 / float64(4.0) + c7[4] * T4 / float64(5.0) + c7[5] / T)
        h_s = R_CONST * T * hRT
        Xs = C[i] / ntot if ntot > EPS else float64(0.0)

        lcp += Xs * cp_s
        lH += h_s * omega[i]
        ldcp += Xs * dcps_dT
        lom += omega[i]
        i += n_threads

    _block_reduce_sum(lcp, scratch, tid, n_threads)
    cuda.syncthreads()
    cp_mix = scratch[0]

    _block_reduce_sum(lH, scratch, tid, n_threads)
    cuda.syncthreads()
    H_num = scratch[0]

    _block_reduce_sum(ldcp, scratch, tid, n_threads)
    cuda.syncthreads()
    dcp_dT = scratch[0]

    _block_reduce_sum(lom, scratch, tid, n_threads)
    cuda.syncthreads()
    omega_sum = scratch[0]

    # Scalaires primaux dérivés
    D = ntot * cp_mix  # dénominateur de dT/dt
    D_safe = D if abs(D) > EPS else (EPS if D >= float64(0.0) else -EPS)
    D_sq = D_safe * D_safe
    dTdt = -H_num / D_safe  # = RHS[0]
    delta = omega_sum / ntot + dTdt / T  # dilatation isobare

    # ── Passe 2 : ligne énergie i=0, parallélisation sur les colonnes j ──────
    #
    # dT/dt = -H_num / D
    # d(dT/dt)/d(xk_j) = -(dH_j·D - H_num·dD_j) / D²
    #
    # dH_j = Σ_s h_s·J_chem[s,j]  +  𝟙_{j=0}·Σ_s cp_s·ω_s
    # dD_j :
    #   j=0  → ntot·dcp_dT
    #   j=k+1→ cp_k   (∂D/∂C_k = cp_k, car D=Σ C_s·cp_s indép. de T)

    j = tid
    while j < n_state:
        dH_j = float64(0.0)
        for s in range(n_sp):
            _load_coef7(T, s, thM, thL, thH, c7)
            T2 = T * T;
            T3 = T2 * T;
            T4 = T3 * T
            hRT = (c7[0] + c7[1] * T / float64(2.0) + c7[2] * T2 / float64(3.0)
                   + c7[3] * T3 / float64(4.0) + c7[4] * T4 / float64(5.0) + c7[5] / T)
            h_s = R_CONST * T * hRT
            dH_j += h_s * J_chem[s * n_state + j]

        if j == 0:
            # contribution d(h_s)/dT·ω_s = cp_s·ω_s
            for s in range(n_sp):
                _load_coef7(T, s, thM, thL, thH, c7)
                T2 = T * T;
                T3 = T2 * T;
                T4 = T3 * T
                cpR = c7[0] + c7[1] * T + c7[2] * T2 + c7[3] * T3 + c7[4] * T4
                cp_s = R_CONST * cpR
                dH_j += cp_s * omega[s]
            dD_j = ntot * dcp_dT
        else:
            s_k = j - 1
            _load_coef7(T, s_k, thM, thL, thH, c7)
            T2 = T * T;
            T3 = T2 * T;
            T4 = T3 * T
            cpR = c7[0] + c7[1] * T + c7[2] * T2 + c7[3] * T3 + c7[4] * T4
            cp_k = R_CONST * cpR
            dD_j = cp_k

        J_full[0 * n_state + j] = -(dH_j * D_safe - H_num * dD_j) / D_sq
        j += n_threads
    cuda.syncthreads()

    # ── Passe 3 : lignes espèces, parallélisation sur l'index plat (s, j) ────
    #
    # dC_s/dt = ω_s - C_s·δ
    # d(dC_s/dt)/d(xk_j) = J_chem[s,j]
    #                       - C_s · d(δ)/d(xk_j)
    #                       - δ · 𝟙_{j=s+1}
    #
    # d(δ)/d(xk_j) se décompose :
    #   δ = ω_sum/n_tot + dTdt/T
    #
    #   d(ω_sum)/d(xk_j) = Σ_k J_chem[k, j]          (= col sum of J_chem)
    #   d(1/n_tot)/d(xk_j) :
    #       j=0   → 0
    #       j=k+1 → -1/ntot²
    #   d(dTdt)/d(xk_j) = J_full[0, j]   (déjà calculé)
    #   d(1/T)/d(xk_j)  :
    #       j=0   → -1/T²
    #       j≥1   → 0
    #
    # → dδ_j =
    #     (Σ_k J_chem[k,j])/ntot  +  J_full[0,j]/T
    #     + [j=0]:   -1/T²·dTdt + 0               (terme d(1/T)/dT·ω_sum/n)
    #     + [j≥1]:   -ω_sum/ntot²                 (terme d(1/n_tot)/dC_k·ω_sum)
    #
    # On calcule d'abord la somme colonne de J_chem en scratch une fois par j,
    # mais ici les colonnes sont déjà traitées par threads différents, donc
    # chaque thread calcule sa propre colonne_sum localement (boucle interne s).

    idx = tid
    while idx < n_sp * n_state:
        s = idx // n_state
        j = idx % n_state

        # somme colonne j de J_chem
        col_sum = float64(0.0)
        for k in range(n_sp):
            col_sum += J_chem[k * n_state + j]

        # d(delta)/d(xk_j)
        ddelta_j = col_sum / ntot + J_full[0 * n_state + j] / T
        if j == 0:
            # d(omega_sum/ntot)/dT = col_sum/ntot (déjà inclus)
            # d(dTdt/T)/dT = J[0,0]/T + dTdt·(-1/T²)
            ddelta_j += dTdt * (-float64(1.0) / (T * T))
        else:
            # d(omega_sum/ntot)/dC_k = -omega_sum/ntot²
            ddelta_j += -omega_sum / (ntot * ntot)

        indicator = float64(1.0) if (j == s + 1) else float64(0.0)
        J_full[(s + 1) * n_state + j] = (J_chem[s * n_state + j]
                                         - C[s] * ddelta_j
                                         - delta * indicator)
        idx += n_threads
    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Kernel CUDA global
# ─────────────────────────────────────────────────────────────────────────────

_MAX_N_STATE = 54
_MAX_N_STATE2 = _MAX_N_STATE * _MAX_N_STATE
_THREADS_PER_BLOCK = 64


@cuda.jit(cache=True)
def _full_jacobian_constantP_kernel_pb(
        TC_in,  # (B, n_state)    float64
        J_all,  # (B, n_state²)   float64  — sortie
        ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e,
        ctx_sRre, ctx_sRpe,
        ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
        ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
        ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f,
        ctx_sRrf, ctx_sRpf,
        ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
        ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
        n_sp,
        thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1

    sh_xk = cuda.shared.array(_MAX_N_STATE, float64)
    sh_om = cuda.shared.array(_MAX_N_STATE, float64)
    sh_col = cuda.shared.array(_MAX_N_STATE, float64)
    sh_J_chem = cuda.shared.array(_MAX_N_STATE2, float64)
    sh_J_full = cuda.shared.array(_MAX_N_STATE2, float64)
    sh_scratch = cuda.shared.array(_THREADS_PER_BLOCK, float64)

    # Chargement de l'état
    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()
    T_loc = sh_xk[0]

    # Aliases contexte cinétique (identique au kernel adiabatique)
    e_kf = ctx_kf_e[b];
    e_Kc = ctx_Kc_e[b]
    e_cr = ctx_cr_e[b];
    e_cp = ctx_cp_e[b];
    e_dG = ctx_dG0_e[b]
    e_sRr = ctx_sRre[b];
    e_sRp = ctx_sRpe[b]
    tb_kf = ctx_kf_tb[b];
    tb_Kc = ctx_Kc_tb[b]
    tb_cr = ctx_cr_tb[b];
    tb_cp = ctx_cp_tb[b];
    tb_dG = ctx_dG0_tb[b]
    tb_sRr = ctx_sRrtb[b];
    tb_sRp = ctx_sRptb[b]
    tb_bRR = ctx_bRR_tb[b];
    tb_M = ctx_M_tb[b]
    f_kf = ctx_kf_f[b];
    f_Kc = ctx_Kc_f[b]
    f_cr = ctx_cr_f[b];
    f_cp = ctx_cp_f[b];
    f_dG = ctx_dG0_f[b]
    f_sRr = ctx_sRrf[b];
    f_sRp = ctx_sRpf[b]
    f_k0 = ctx_k0_f[b];
    f_ki = ctx_kinf_f[b]
    f_M = ctx_M_f[b];
    f_Pr = ctx_Pr_f[b]
    f_F = ctx_F_f[b];
    f_kb = ctx_kb_f[b]
    f_Fc = ctx_Fc_f[b];
    f_lFc = ctx_lFc_f[b]
    f_Cp = ctx_Cp_f[b];
    f_Np = ctx_Np_f[b]
    f_f1 = ctx_f1_f[b];
    f_exp = ctx_exp_f[b]

    # Étape 1 : omega
    _eval_rhs_and_context_pb(
        tid, n_threads,
        T_loc, sh_xk[1:],
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

    # Étape 2 : jacobienne chimique J_chem (n_sp × n_state)
    _eval_jacobian_pb(
        tid, n_threads,
        T_loc, sh_xk[1:],
        sh_J_chem,
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
        sh_col,
    )

    # Étape 3 : jacobienne complète isobare
    _constantP_full_jacobian_pb(
        tid, n_threads,
        T_loc, sh_xk[1:], sh_om, sh_J_chem,
        n_sp, thL, thH, thM,
        sh_J_full,
        sh_scratch,
    )

    # Copie → mémoire globale
    idx = tid
    while idx < n_state * n_state:
        J_all[b, idx] = sh_J_full[idx]
        idx += n_threads
    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Wrapper Python : entrées/sorties en TC
# ─────────────────────────────────────────────────────────────────────────────

class CUDAFullJacobianConstantPEvaluatorPB:
    """
    Jacobienne complète du réacteur isobare (per-block), coordonnées TC.

    J[i, j] = d(RHS_i) / d(xk_j)

    RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)  — réacteur P=cte
    xk  = (T, C_0, …, C_{n_sp-1})

    Usage
    -----
    ev = CUDAFullJacobianConstantPEvaluatorPB(mech)
    J  = ev.evaluate(T_arr, C_arr)
    # T_arr : (B,)       float64
    # C_arr : (B, n_sp)  float64
    # J     : (B, n_state, n_state) float64
    """

    THREADS_PER_BLOCK = _THREADS_PER_BLOCK

    def __init__(self, mech, dtype=np.float64):
        self._mech = mech
        self._dtype = dtype
        self._n_sp = mech.n_species
        self._n_state = self._n_sp + 1
        self._ctx = None
        self._B_alloc = -1

    def _alloc_context(self, B):
        dt = self._dtype
        m = self._mech
        e = m.elementary
        tb = m.threebody
        f = m.falloff

        def z(shape):
            return cuda.device_array(shape, dtype=dt)

        n_e_flat = e.react_offset[-1] if e.n_reactions else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1

        self._ctx = dict(
            kf_e=z((B, e.n_reactions)), Kc_e=z((B, e.n_reactions)),
            cr_e=z((B, e.n_reactions)), cp_e=z((B, e.n_reactions)),
            dG0_e=z((B, e.n_reactions)),
            sRre=z((B, n_e_flat)), sRpe=z((B, n_e_flat)),
            kf_tb=z((B, tb.n_reactions)), Kc_tb=z((B, tb.n_reactions)),
            cr_tb=z((B, tb.n_reactions)), cp_tb=z((B, tb.n_reactions)),
            dG0_tb=z((B, tb.n_reactions)),
            sRrtb=z((B, n_tb_flat)), sRptb=z((B, n_tb_flat)),
            bRR_tb=z((B, tb.n_reactions)), M_tb=z((B, tb.n_reactions)),
            kf_f=z((B, f.n_reactions)), Kc_f=z((B, f.n_reactions)),
            cr_f=z((B, f.n_reactions)), cp_f=z((B, f.n_reactions)),
            dG0_f=z((B, f.n_reactions)),
            sRrf=z((B, f.react_offset[-1] if f.n_reactions else 1)),
            sRpf=z((B, f.react_offset[-1] if f.n_reactions else 1)),
            k0_f=z((B, f.n_reactions)), kinf_f=z((B, f.n_reactions)),
            M_f=z((B, f.n_reactions)), Pr_f=z((B, f.n_reactions)),
            F_f=z((B, f.n_reactions)), kb_f=z((B, f.n_reactions)),
            Fc_f=z((B, f.n_reactions)), lFc_f=z((B, f.n_reactions)),
            Cp_f=z((B, f.n_reactions)), Np_f=z((B, f.n_reactions)),
            f1_f=z((B, f.n_reactions)), exp_f=z((B, f.n_reactions)),
        )
        self._B_alloc = B

    def evaluate(self, T_arr, C_arr):
        """
        Paramètres
        ----------
        T_arr : array-like (B,)
        C_arr : array-like (B, n_sp)

        Retourne
        --------
        J : np.ndarray (B, n_state, n_state)
        """
        T_arr = np.asarray(T_arr, dtype=self._dtype)
        C_arr = np.asarray(C_arr, dtype=self._dtype)
        B = T_arr.shape[0]
        nst = self._n_state

        if self._B_alloc != B:
            self._alloc_context(B)

        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0] = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)
        d_J = cuda.device_array((B, nst * nst), dtype=self._dtype)

        m = self._mech
        e = m.elementary
        tb = m.threebody
        f = m.falloff
        cx = self._ctx

        _full_jacobian_constantP_kernel_pb[B, self.THREADS_PER_BLOCK](
            d_TC, d_J,
            cx['kf_e'], cx['Kc_e'], cx['cr_e'], cx['cp_e'], cx['dG0_e'],
            cx['sRre'], cx['sRpe'],
            cx['kf_tb'], cx['Kc_tb'], cx['cr_tb'], cx['cp_tb'], cx['dG0_tb'],
            cx['sRrtb'], cx['sRptb'], cx['bRR_tb'], cx['M_tb'],
            cx['kf_f'], cx['Kc_f'], cx['cr_f'], cx['cp_f'], cx['dG0_f'],
            cx['sRrf'], cx['sRpf'],
            cx['k0_f'], cx['kinf_f'], cx['M_f'], cx['Pr_f'], cx['F_f'],
            cx['kb_f'],
            cx['Fc_f'], cx['lFc_f'], cx['Cp_f'], cx['Np_f'],
            cx['f1_f'], cx['exp_f'],
            m.n_species,
            m.thermo.coef_low, m.thermo.coef_high, m.thermo.Tmid,
            e.n_reactions, e.A, e.n, e.Ea, e.oneWay,
            e.delta_n, e.react_idx, e.react_sto,
            e.react_offset, e.prod_idx, e.prod_sto, e.prod_offset,
            tb.n_reactions, tb.A, tb.n, tb.Ea, tb.oneWay,
            tb.delta_n, tb.react_idx, tb.react_sto,
            tb.react_offset, tb.prod_idx, tb.prod_sto, tb.prod_offset, tb.eff_matrix,
            f.n_reactions, f.A_k0, f.b_k0, f.Ea_k0, f.A_kinf, f.b_kinf, f.Ea_kinf,
            f.oneWay, f.delta_n,
            f.troe_A, f.troe_T1, f.troe_T2, f.troe_T3, f.is_troe,
            f.react_idx, f.react_sto, f.react_offset,
            f.prod_idx, f.prod_sto, f.prod_offset, f.eff_matrix,
        )
        cuda.synchronize()
        return d_J.copy_to_host().reshape(B, nst, nst)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Wrapper TPY — push-forward analytique de la jacobienne TC
# ─────────────────────────────────────────────────────────────────────────────

# Constante universelle des gaz (J/mol/K)
_R = 8.314462618

import kineticsGPU.torch.thermo as thermo
import kineticsGPU.torch.reactionParser as reactionParser
from kineticsGPU.cuda.mechanism.mechanism_data import build_mechanism_data_from_torch

import torch

class CUDAFullJacobianConstantPTPY:
    """
    Jacobienne complète du réacteur isobare dans l'espace (T, P, Y).

    Calcule J_TPY tel que :

        J_TPY = (∂RHS_TPY/∂TPY) = (∂decode_rhs/∂TC) · J_TC · (∂encode/∂TPY)⁻¹

    ce qui revient à pousser J_TC à travers le changement de variables :

        ψ : (T, P, Y)  →  TC = (T, C_0, …, C_{ns-1})

    via la règle de la chaîne :

        J_TPY[α,β] = Σ_{i,j} (∂RHS_TPY_α/∂RHS_TC_i) · J_TC[i,j]
                              · (∂TC_j/∂TPY_β)

    La matrice (∂TC/∂TPY) est de taille (n_state, n_state+1), calculée
    analytiquement depuis encode() :

        C_k = ρ·Y_k/W_k,   ρ = P/(R·T·Σ Y_k/W_k)

    et la matrice (∂RHS_TPY/∂RHS_TC) est fournie par decode_rhs().

    Usage
    -----
    ev   = CUDAFullJacobianConstantPTPY(mech)
    Jtpy = ev.evaluate(T_arr, P_arr, Y_arr)
    # T_arr : (B,)       float64    températures (K)
    # P_arr : (B,)       float64    pressions (Pa)
    # Y_arr : (B, n_sp)  float64    fractions massiques (normalisées)
    # Jtpy  : (B, n_state+1, n_state+1) float64
    #         espace (T, P, Y_0, …, Y_{ns-1}), P inerte → ligne/colonne 1 = 0

    Notes
    -----
    • La colonne P et la ligne P de J_TPY sont nulles (contrainte P=cte).
    • Le calcul est entièrement vectorisé en NumPy (pas de kernel
      supplémentaire) : le coût est O(B·n_state²) flops CPU après
      rapatriement de J_TC depuis le GPU.
    • Pour de très grands B, un kernel de push-forward GPU serait plus
      efficace, mais la plupart des usages (solveurs implicites batch)
      font B ≤ quelques milliers.
    """

    def __init__(self, mech, torchComputationDtype=None, dataDevice=None, computationDevice=None, storageDtype=None):
        if torchComputationDtype == torch.float64:
            computationDtype = np.float64
        elif torchComputationDtype == torch.float32:
            computationDtype = np.float32
        else:
            raise NotImplementedError()
        self.torchComputationDtype = torchComputationDtype
        self.storageDtype = storageDtype
        self.dataDevice = dataDevice
        self.computationDevice = torch.device(computationDevice)
        y = thermo.yamlParser(mech)
        vec_th = y.populateVectorizedThermoObject(dtype=torchComputationDtype, device=dataDevice)
        reactions = reactionParser.parseReactions(mech)
        r_e, r_tb, r_f, system = reactionParser.build_reactions(
            vec_th.species, reactions, vec_th, dataDevice, torchComputationDtype
        )
        mech = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=computationDtype)

        self._ev = CUDAFullJacobianConstantPEvaluatorPB(mech, dtype=computationDtype)
        self._dtype = computationDtype
        W = vec_th.molar_masses.to(torchComputationDtype).to(computationDevice)  # (ns,) kg/mol
        self._W = W

    # ── Encodage ψ : (T, P, Y) → (T, C) ────────────────────────────────────

    def _encode(self, T, P, Y):
        """Retourne TC (B, n_state) et ρ (B,)."""
        sYW = (Y / self._W[None, :]).sum(axis=1)  # (B,)  = 1/W_mean
        rho = P / (_R * T * sYW)  # (B,)
        C = rho[:, None] * Y / self._W[None, :]  # (B, ns)
        TC = torch.cat([T[:, None], C], dim=1)
        return TC, rho

    # ── Jacobienne ∂TC/∂TPY : (B, n_state, n_state+1) ──────────────────────

    def _dTC_dTPY(self, T, P, Y, rho):
        """
        Matrice (∂TC_j/∂TPY_β) analytique.

        TPY indexé : 0=T, 1=P, 2..ns+1=Y_k
        TC  indexé : 0=T, 1..ns=C_k

        Lignes 0  (T→T) :   ∂T/∂T=1, reste=0
        Lignes k+1 (C_k) :
            ∂C_k/∂T  = -C_k/T
            ∂C_k/∂P  = C_k/P
            ∂C_k/∂Y_l = ρ/W_k·δ_{kl} - C_k·(1/W_l)/(Σ Y_m/W_m)
        """
        B = T.shape[0]
        ns = Y.shape[1]
        nst = ns + 1  # dim TC
        nst1 = ns + 2  # dim TPY

        dTC = torch.zeros((B, nst, nst1), dtype=self.torchComputationDtype, device=self.computationDevice)

        # Ligne 0 : T → T = 1
        dTC[:, 0, 0] = 1.0

        # Lignes k+1 : concentrations
        sYW = (Y / self._W[None, :]).sum(dim=1)  # (B,)
        C = rho[:, None] * Y / self._W[None, :]  # (B, ns)

        # ∂C_k/∂T = -C_k/T
        dTC[:, 1:, 0] = -C / T[:, None]

        # ∂C_k/∂P = C_k/P
        dTC[:, 1:, 1] = C / P[:, None]

        # ∂C_k/∂Y_l = ρ/W_k · δ_{kl} - C_k / (W_l · Σ Y_m/W_m)
        # Terme diagonal : (B, ns)
        diag_term = rho[:, None] / self._W[None, :]  # (B, ns)
        # Terme cross   : -(B, ns_k, 1) · (1, 1, ns_l)/(W_l · sYW)
        cross_term = (
                -C[:, :, None]  # (B, ns, 1)
                / (self._W[None, None, :] * sYW[:, None, None])  # (B, 1, ns)
        )  # (B, ns, ns)
        dTC[:, 1:, 2:] = cross_term
        for k in range(ns):
            dTC[:, k + 1, k + 2] += diag_term[:, k]  # diagonal

        return dTC

    # ── Jacobienne ∂RHS_TPY/∂RHS_TC : (B, n_state+1, n_state) ─────────────

    def _dRHSTPY_dRHSTC(self, T, C, rho):
        """
        Matrice (∂RHS_TPY_α/∂RHS_TC_i).

        RHS_TC  = (dT/dt, dC_0/dt, …)   [n_state]
        RHS_TPY = (dT/dt, dP/dt, dY_0/dt, …)   [n_state+1]

        dT/dt est identique → ligne 0, col 0 = 1.
        dP/dt = 0 → ligne 1 = 0.
        dY_k/dt = (W_k·dC_k/dt - Y_k·Σ W_l·dC_l/dt) / ρ
                = (W_k·δ_{ki} - Y_k·W_{i-1}) / ρ  pour i=k+1 (colonne C_k)
                  (0 pour i=0, colonne T)
        """
        B = T.shape[0]
        ns = C.shape[1]
        nst = ns + 1
        nst1 = ns + 2

        M = torch.zeros((B, nst1, nst), dtype=self.torchComputationDtype, device=self.computationDevice)

        # Ligne 0 : dT/dt → dT/dt
        M[:, 0, 0] = 1.0

        # Ligne 1 : dP/dt = 0 (déjà nul par init)

        # Lignes k+2 : dY_k/dt
        Y = C * self._W[None, :] / rho[:, None]  # (B, ns)

        # ∂(dY_k)/∂(dC_l) = (W_k·δ_{kl} - Y_k·W_l) / ρ
        # Terme cross : -Y_k·W_l/ρ  pour tous k, l
        M[:, 2:, 1:] = (
                -Y[:, :, None] * self._W[None, None, :]  # (B, ns_k, ns_l)
                / rho[:, None, None]
        )
        # Terme diagonal : +W_k/ρ
        for k in range(ns):
            M[:, k + 2, k + 1] += self._W[k] / rho  # (B,)

        return M

    # ── Interface publique ───────────────────────────────────────────────────

    def evaluate(self, T_arr, P_arr, Y_arr):
        """
        Paramètres
        ----------
        T_arr : array-like (B,)       températures (K)
        P_arr : array-like (B,)       pressions (Pa)
        Y_arr : array-like (B, n_sp)  fractions massiques (normalisées)

        Retourne
        --------
        J_TPY : np.ndarray (B, n_state+1, n_state+1)
                espace (T, P, Y_0, …, Y_{ns-1})
                La ligne 1 et la colonne 1 (pression) sont nulles.

        Relation
        --------
        J_TPY = M_out @ J_TC @ dTC_dTPY
        avec
            J_TC     : (B, n_state, n_state)  — GPU
            dTC_dTPY : (B, n_state, n_state+1)
            M_out    : (B, n_state+1, n_state)
        """
        _cuda_index= self.computationDevice.index if self.computationDevice.index is not None else 0
        T_arr = T_arr.to(self.torchComputationDtype).to(self.computationDevice)
        P_arr = P_arr.to(self.torchComputationDtype).to(self.computationDevice)
        Y_arr = Y_arr.to(self.torchComputationDtype).to(self.computationDevice)
        # Normalisation de sécurité
        Y_arr = Y_arr / Y_arr.sum(dim=1, keepdims=True)

        # Encodage → TC, ρ
        TC, rho = self._encode(T_arr, P_arr, Y_arr)

        # Jacobienne GPU dans l'espace TC
        J_TC = self._ev.evaluate(cuda.as_cuda_array(TC[:, 0]), cuda.as_cuda_array(TC[:, 1:]))  # (B, nst, nst)
        J_TC = torch.as_tensor(J_TC).to(self.computationDevice).to(self.torchComputationDtype)

        # Matrices de changement de variable (CPU, analytique)
        dTC_dTPY = self._dTC_dTPY(T_arr, P_arr, Y_arr, rho)  # (B, nst, nst+1)
        M_out = self._dRHSTPY_dRHSTC(T_arr, TC[:, 1:], rho)  # (B, nst+1, nst)

        # Push-forward : J_TPY = M_out @ J_TC @ dTC_dTPY
        # (B, nst+1, nst) @ (B, nst, nst) @ (B, nst, nst+1)
        J_TPY = M_out @ J_TC @ dTC_dTPY  # (B, nst+1, nst+1)

        return J_TPY.to(self.dataDevice).to(self.storageDtype)

# ─────────────────────────────────────────────────────────────────────────────
#  1.  Jacobienne per-block du RHS isochore
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _constantV_full_jacobian_pb(
        tid, n_threads,
        T, C, omega, J_chem,
        n_sp, thL, thH, thM,
        J_full,  # mémoire partagée, taille n_state²
        scratch,  # mémoire partagée pour réductions, taille >= n_threads
):
    """
    Jacobienne complète du réacteur isochore adiabatique (per-block).

    J_full[i * n_state + j] = d(RHS_i) / d(xk_j)

    Ligne i=0 :
        dT/dt = -U_num / D,    D = n_tot·cv_mix,    U_num = Σ u_s·ω_s
        → calcul analytique, parallélisation sur les colonnes j.

    Lignes i=s+1 :
        dC_s/dt = ω_s   (volume constant, pas de dilatation)
        → d(dC_s/dt)/d(xk_j) = J_chem[s, j]   (copie directe)

    Architecture (3 passes identiques à la version isobare)
    ───────────────────────────────────────────────────────
        Passe 1 — réductions parallèles (ntot, cv_mix, U_num, dcv_dT)
        Passe 2 — boucle colonnes j pour la ligne énergie
        Passe 3 — copie plate J_chem → lignes espèces

    Dérivation — ligne énergie
    --------------------------
    dT/dt = -U_num / D,    U_num = Σ u_s·ω_s,    D = n_tot·cv_mix

    d(dT/dt)/d(xk_j) = -(dU_j·D - U_num·dD_j) / D²

    dU_j :
        j = 0    → Σ_s cv_s·ω_s  +  Σ_s u_s·J_chem[s,0]
        j = k+1  → Σ_s u_s·J_chem[s,j]

    dD_j :
        j = 0    → n_tot · Σ_s X_s·(d(cv_s)/dT)
        j = k+1  → cv_k
    """
    n_state = n_sp + 1
    c7 = cuda.local.array(7, float64)

    # ── Passe 1 : réductions parallèles ──────────────────────────────────────

    # ntot = Σ C_s
    acc = float64(0.0)
    i = tid
    while i < n_sp:
        acc += C[i]
        i += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)
    cuda.syncthreads()
    ntot = scratch[0]

    # cv_mix, U_num, dcv_dT — une seule passe espèces
    lcv = float64(0.0)  # Σ X_s·cv_s
    lU = float64(0.0)  # Σ u_s·ω_s
    ldcv = float64(0.0)  # Σ X_s·d(cv_s)/dT

    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        T2 = T * T;
        T3 = T2 * T;
        T4 = T3 * T
        cpR = c7[0] + c7[1] * T + c7[2] * T2 + c7[3] * T3 + c7[4] * T4
        cv_s = R_CONST * (cpR - float64(1.0))
        dcvs_dT = R_CONST * (c7[1]
                             + float64(2.0) * c7[2] * T
                             + float64(3.0) * c7[3] * T2
                             + float64(4.0) * c7[4] * T3)
        hRT = (c7[0] + c7[1] * T / float64(2.0) + c7[2] * T2 / float64(3.0)
               + c7[3] * T3 / float64(4.0) + c7[4] * T4 / float64(5.0) + c7[5] / T)
        u_s = R_CONST * T * hRT - R_CONST * T
        Xs = C[i] / ntot if ntot > EPS else float64(0.0)

        lcv += Xs * cv_s
        lU += u_s * omega[i]
        ldcv += Xs * dcvs_dT
        i += n_threads

    _block_reduce_sum(lcv, scratch, tid, n_threads)
    cuda.syncthreads()
    cv_mix = scratch[0]

    _block_reduce_sum(lU, scratch, tid, n_threads)
    cuda.syncthreads()
    U_num = scratch[0]

    _block_reduce_sum(ldcv, scratch, tid, n_threads)
    cuda.syncthreads()
    dcv_dT = scratch[0]

    D = ntot * cv_mix
    D_safe = D if abs(D) > EPS else (EPS if D >= float64(0.0) else -EPS)
    D_sq = D_safe * D_safe

    # ── Passe 2 : ligne énergie i=0, parallélisation sur les colonnes j ──────

    j = tid
    while j < n_state:
        dU_j = float64(0.0)
        for s in range(n_sp):
            _load_coef7(T, s, thM, thL, thH, c7)
            T2 = T * T;
            T3 = T2 * T;
            T4 = T3 * T
            hRT = (c7[0] + c7[1] * T / float64(2.0) + c7[2] * T2 / float64(3.0)
                   + c7[3] * T3 / float64(4.0) + c7[4] * T4 / float64(5.0) + c7[5] / T)
            u_s = R_CONST * T * hRT - R_CONST * T
            dU_j += u_s * J_chem[s * n_state + j]

        if j == 0:
            # d(u_s)/dT · ω_s = cv_s · ω_s
            for s in range(n_sp):
                _load_coef7(T, s, thM, thL, thH, c7)
                T2 = T * T;
                T3 = T2 * T;
                T4 = T3 * T
                cpR = c7[0] + c7[1] * T + c7[2] * T2 + c7[3] * T3 + c7[4] * T4
                cv_s = R_CONST * (cpR - float64(1.0))
                dU_j += cv_s * omega[s]
            dD_j = ntot * dcv_dT
        else:
            s_k = j - 1
            _load_coef7(T, s_k, thM, thL, thH, c7)
            T2 = T * T;
            T3 = T2 * T;
            T4 = T3 * T
            cpR = c7[0] + c7[1] * T + c7[2] * T2 + c7[3] * T3 + c7[4] * T4
            cv_k = R_CONST * (cpR - float64(1.0))
            dD_j = cv_k

        J_full[0 * n_state + j] = -(dU_j * D_safe - U_num * dD_j) / D_sq
        j += n_threads
    cuda.syncthreads()

    # ── Passe 3 : lignes espèces — copie directe depuis J_chem ───────────────
    # Pas de terme de dilatation : dC_s/dt = ω_s → d/d(xk_j) = J_chem[s,j]
    idx = tid
    while idx < n_sp * n_state:
        s = idx // n_state
        j = idx % n_state
        J_full[(s + 1) * n_state + j] = J_chem[idx]
        idx += n_threads
    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Kernel CUDA global
# ─────────────────────────────────────────────────────────────────────────────

_MAX_N_STATE = 54
_MAX_N_STATE2 = _MAX_N_STATE * _MAX_N_STATE
_THREADS_PER_BLOCK = 64


@cuda.jit(cache=True)
def _full_jacobian_constantV_kernel_pb(
        TC_in,  # (B, n_state)    float64
        J_all,  # (B, n_state²)   float64  — sortie
        ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e,
        ctx_sRre, ctx_sRpe,
        ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
        ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
        ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f,
        ctx_sRrf, ctx_sRpf,
        ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
        ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
        n_sp,
        thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1

    sh_xk = cuda.shared.array(_MAX_N_STATE, float64)
    sh_om = cuda.shared.array(_MAX_N_STATE, float64)
    sh_col = cuda.shared.array(_MAX_N_STATE, float64)
    sh_J_chem = cuda.shared.array(_MAX_N_STATE2, float64)
    sh_J_full = cuda.shared.array(_MAX_N_STATE2, float64)
    sh_scratch = cuda.shared.array(_THREADS_PER_BLOCK, float64)

    # Chargement de l'état
    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()
    T_loc = sh_xk[0]

    # Aliases contexte cinétique
    e_kf = ctx_kf_e[b];
    e_Kc = ctx_Kc_e[b]
    e_cr = ctx_cr_e[b];
    e_cp = ctx_cp_e[b];
    e_dG = ctx_dG0_e[b]
    e_sRr = ctx_sRre[b];
    e_sRp = ctx_sRpe[b]
    tb_kf = ctx_kf_tb[b];
    tb_Kc = ctx_Kc_tb[b]
    tb_cr = ctx_cr_tb[b];
    tb_cp = ctx_cp_tb[b];
    tb_dG = ctx_dG0_tb[b]
    tb_sRr = ctx_sRrtb[b];
    tb_sRp = ctx_sRptb[b]
    tb_bRR = ctx_bRR_tb[b];
    tb_M = ctx_M_tb[b]
    f_kf = ctx_kf_f[b];
    f_Kc = ctx_Kc_f[b]
    f_cr = ctx_cr_f[b];
    f_cp = ctx_cp_f[b];
    f_dG = ctx_dG0_f[b]
    f_sRr = ctx_sRrf[b];
    f_sRp = ctx_sRpf[b]
    f_k0 = ctx_k0_f[b];
    f_ki = ctx_kinf_f[b]
    f_M = ctx_M_f[b];
    f_Pr = ctx_Pr_f[b]
    f_F = ctx_F_f[b];
    f_kb = ctx_kb_f[b]
    f_Fc = ctx_Fc_f[b];
    f_lFc = ctx_lFc_f[b]
    f_Cp = ctx_Cp_f[b];
    f_Np = ctx_Np_f[b]
    f_f1 = ctx_f1_f[b];
    f_exp = ctx_exp_f[b]

    # Étape 1 : omega
    _eval_rhs_and_context_pb(
        tid, n_threads,
        T_loc, sh_xk[1:],
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

    # Étape 2 : jacobienne chimique J_chem (n_sp × n_state)
    _eval_jacobian_pb(
        tid, n_threads,
        T_loc, sh_xk[1:],
        sh_J_chem,
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
        sh_col,
    )

    # Étape 3 : jacobienne complète isochore
    _constantV_full_jacobian_pb(
        tid, n_threads,
        T_loc, sh_xk[1:], sh_om, sh_J_chem,
        n_sp, thL, thH, thM,
        sh_J_full,
        sh_scratch,
    )

    # Copie → mémoire globale
    idx = tid
    while idx < n_state * n_state:
        J_all[b, idx] = sh_J_full[idx]
        idx += n_threads
    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Wrapper Python : entrées/sorties en TC
# ─────────────────────────────────────────────────────────────────────────────

class CUDAFullJacobianConstantVEvaluatorPB:
    """
    Jacobienne complète du réacteur isochore adiabatique (per-block),
    coordonnées TC.

    J[i, j] = d(RHS_i) / d(xk_j)

    RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)  — réacteur V=cte
    xk  = (T, C_0, …, C_{n_sp-1})

    Usage
    -----
    ev = CUDAFullJacobianConstantVEvaluatorPB(mech)
    J  = ev.evaluate(T_arr, C_arr)
    # T_arr : (B,)       float64
    # C_arr : (B, n_sp)  float64
    # J     : (B, n_state, n_state) float64,  n_state = n_sp + 1
    """

    THREADS_PER_BLOCK = _THREADS_PER_BLOCK

    def __init__(self, mech, dtype=np.float64):
        self._mech = mech
        self._dtype = dtype
        self._n_sp = mech.n_species
        self._n_state = self._n_sp + 1
        self._ctx = None
        self._B_alloc = -1

    def _alloc_context(self, B):
        dt = self._dtype
        m = self._mech
        e = m.elementary
        tb = m.threebody
        f = m.falloff

        def z(shape):
            return cuda.device_array(shape, dtype=dt)

        n_e_flat = e.react_offset[-1] if e.n_reactions else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1

        self._ctx = dict(
            kf_e=z((B, e.n_reactions)), Kc_e=z((B, e.n_reactions)),
            cr_e=z((B, e.n_reactions)), cp_e=z((B, e.n_reactions)),
            dG0_e=z((B, e.n_reactions)),
            sRre=z((B, n_e_flat)), sRpe=z((B, n_e_flat)),
            kf_tb=z((B, tb.n_reactions)), Kc_tb=z((B, tb.n_reactions)),
            cr_tb=z((B, tb.n_reactions)), cp_tb=z((B, tb.n_reactions)),
            dG0_tb=z((B, tb.n_reactions)),
            sRrtb=z((B, n_tb_flat)), sRptb=z((B, n_tb_flat)),
            bRR_tb=z((B, tb.n_reactions)), M_tb=z((B, tb.n_reactions)),
            kf_f=z((B, f.n_reactions)), Kc_f=z((B, f.n_reactions)),
            cr_f=z((B, f.n_reactions)), cp_f=z((B, f.n_reactions)),
            dG0_f=z((B, f.n_reactions)),
            sRrf=z((B, f.react_offset[-1] if f.n_reactions else 1)),
            sRpf=z((B, f.react_offset[-1] if f.n_reactions else 1)),
            k0_f=z((B, f.n_reactions)), kinf_f=z((B, f.n_reactions)),
            M_f=z((B, f.n_reactions)), Pr_f=z((B, f.n_reactions)),
            F_f=z((B, f.n_reactions)), kb_f=z((B, f.n_reactions)),
            Fc_f=z((B, f.n_reactions)), lFc_f=z((B, f.n_reactions)),
            Cp_f=z((B, f.n_reactions)), Np_f=z((B, f.n_reactions)),
            f1_f=z((B, f.n_reactions)), exp_f=z((B, f.n_reactions)),
        )
        self._B_alloc = B

    def evaluate(self, T_arr, C_arr):
        """
        Paramètres
        ----------
        T_arr : array-like (B,)
        C_arr : array-like (B, n_sp)

        Retourne
        --------
        J : np.ndarray (B, n_state, n_state)
        """
        T_arr = np.asarray(T_arr, dtype=self._dtype)
        C_arr = np.asarray(C_arr, dtype=self._dtype)
        B = T_arr.shape[0]
        nst = self._n_state

        if self._B_alloc != B:
            self._alloc_context(B)

        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0] = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)
        d_J = cuda.device_array((B, nst * nst), dtype=self._dtype)

        m = self._mech
        e = m.elementary
        tb = m.threebody
        f = m.falloff
        cx = self._ctx

        _full_jacobian_constantV_kernel_pb[B, self.THREADS_PER_BLOCK](
            d_TC, d_J,
            cx['kf_e'], cx['Kc_e'], cx['cr_e'], cx['cp_e'], cx['dG0_e'],
            cx['sRre'], cx['sRpe'],
            cx['kf_tb'], cx['Kc_tb'], cx['cr_tb'], cx['cp_tb'], cx['dG0_tb'],
            cx['sRrtb'], cx['sRptb'], cx['bRR_tb'], cx['M_tb'],
            cx['kf_f'], cx['Kc_f'], cx['cr_f'], cx['cp_f'], cx['dG0_f'],
            cx['sRrf'], cx['sRpf'],
            cx['k0_f'], cx['kinf_f'], cx['M_f'], cx['Pr_f'], cx['F_f'],
            cx['kb_f'],
            cx['Fc_f'], cx['lFc_f'], cx['Cp_f'], cx['Np_f'],
            cx['f1_f'], cx['exp_f'],
            m.n_species,
            m.thermo.coef_low, m.thermo.coef_high, m.thermo.Tmid,
            e.n_reactions, e.A, e.n, e.Ea, e.oneWay,
            e.delta_n, e.react_idx, e.react_sto,
            e.react_offset, e.prod_idx, e.prod_sto, e.prod_offset,
            tb.n_reactions, tb.A, tb.n, tb.Ea, tb.oneWay,
            tb.delta_n, tb.react_idx, tb.react_sto,
            tb.react_offset, tb.prod_idx, tb.prod_sto, tb.prod_offset, tb.eff_matrix,
            f.n_reactions, f.A_k0, f.b_k0, f.Ea_k0, f.A_kinf, f.b_kinf, f.Ea_kinf,
            f.oneWay, f.delta_n,
            f.troe_A, f.troe_T1, f.troe_T2, f.troe_T3, f.is_troe,
            f.react_idx, f.react_sto, f.react_offset,
            f.prod_idx, f.prod_sto, f.prod_offset, f.eff_matrix,
        )
        cuda.synchronize()
        return d_J.copy_to_host().reshape(B, nst, nst)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Wrapper TPY — push-forward analytique de la jacobienne TC
# ─────────────────────────────────────────────────────────────────────────────

class CUDAFullJacobianConstantVTPY:
    """
    Jacobienne complète du réacteur isochore dans l'espace (T, P, Y).

    J_TPY = M_out @ J_TC @ dTC_dTPY

    avec
        J_TC     : (B, n_state, n_state)   — GPU per-block
        dTC_dTPY : (B, n_state, n_state+1) — ∂encode/∂TPY, analytique
        M_out    : (B, n_state+1, n_state) — ∂decode_rhs/∂RHS_TC, analytique

    Différences vs CUDAFullJacobianConstantPTPY
    --------------------------------------------
    • dρ/dt = 0  →  dY_k/dt = W_k·dC_k/ρ  (matrice M_out diagonale sur le bloc Y)
    • dP/dt ≠ 0  →  ligne P non nulle :
          ∂(dP)/∂(dT/dt)   = R·n_tot
          ∂(dP)/∂(dC_k/dt) = R·T

    Usage
    -----
    ev   = CUDAFullJacobianConstantVTPY(
               mech,
               torchComputationDtype=torch.float64,
               dataDevice="cuda:0",
               computationDevice="cuda:0",
               storageDtype=torch.float64,
           )
    Jtpy = ev.evaluate(T_arr, P_arr, Y_arr)
    # T_arr : Tensor (B,)        températures (K)
    # P_arr : Tensor (B,)        pressions (Pa)
    # Y_arr : Tensor (B, n_sp)   fractions massiques
    # Jtpy  : Tensor (B, n_state+1, n_state+1)
    #         espace (T, P, Y_0, …, Y_{ns-1})
    #         Toutes les lignes/colonnes non nulles (dP/dt ≠ 0)
    """

    def __init__(self, mech, torchComputationDtype=None, dataDevice=None,
                 computationDevice=None, storageDtype=None):
        if torchComputationDtype == torch.float64:
            computationDtype = np.float64
        elif torchComputationDtype == torch.float32:
            computationDtype = np.float32
        else:
            raise NotImplementedError()

        self.torchComputationDtype = torchComputationDtype
        self.storageDtype = storageDtype
        self.dataDevice = dataDevice
        self.computationDevice = torch.device(computationDevice)

        y = thermo.yamlParser(mech)
        vec_th = y.populateVectorizedThermoObject(
            dtype=torchComputationDtype, device=dataDevice
        )
        reactions = reactionParser.parseReactions(mech)
        r_e, r_tb, r_f, system = reactionParser.build_reactions(
            vec_th.species, reactions, vec_th, dataDevice, torchComputationDtype
        )
        mech_data = build_mechanism_data_from_torch(
            r_e, r_tb, r_f, vec_th, dtype=computationDtype
        )

        self._ev = CUDAFullJacobianConstantVEvaluatorPB(mech_data, dtype=computationDtype)
        self._dtype = computationDtype
        self._W = vec_th.molar_masses.to(torchComputationDtype).to(self.computationDevice)

    # ── Encodage ψ : (T, P, Y) → (T, C) ────────────────────────────────────

    def _encode(self, T, P, Y):
        sYW = (Y / self._W[None, :]).sum(dim=1)  # (B,)
        rho = P / (_R * T * sYW)  # (B,)
        C = rho[:, None] * Y / self._W[None, :]  # (B, ns)
        TC = torch.cat([T[:, None], C], dim=1)  # (B, 1+ns)
        return TC, rho

    # ── ∂TC/∂TPY : (B, n_state, n_state+1) ─────────────────────────────────
    # Identique au cas isobare — même carte d'encodage ψ

    def _dTC_dTPY(self, T, P, Y, rho):
        B = T.shape[0]
        ns = Y.shape[1]
        nst = ns + 1
        nst1 = ns + 2

        dTC = torch.zeros(
            (B, nst, nst1),
            dtype=self.torchComputationDtype, device=self.computationDevice
        )
        dTC[:, 0, 0] = 1.0

        sYW = (Y / self._W[None, :]).sum(dim=1)  # (B,)
        C = rho[:, None] * Y / self._W[None, :]  # (B, ns)

        dTC[:, 1:, 0] = -C / T[:, None]  # ∂C_k/∂T
        dTC[:, 1:, 1] = C / P[:, None]  # ∂C_k/∂P

        cross = (
                -C[:, :, None]
                / (self._W[None, None, :] * sYW[:, None, None])
        )
        dTC[:, 1:, 2:] = cross
        diag = rho[:, None] / self._W[None, :]
        for k in range(ns):
            dTC[:, k + 1, k + 2] += diag[:, k]

        return dTC

    # ── ∂RHS_TPY/∂RHS_TC : (B, n_state+1, n_state) ─────────────────────────
    #
    # dT/dt  → identique,  M[:,0,0] = 1
    # dP/dt  = R·(T·Σ dC_k + n_tot·dT)   → ligne P non nulle
    # dY_k/dt = W_k·dC_k/ρ               → diagonal uniquement (dρ/dt = 0)

    def _dRHSTPY_dRHSTC(self, T, C, rho):
        B = T.shape[0]
        ns = C.shape[1]
        nst = ns + 1
        nst1 = ns + 2

        M = torch.zeros(
            (B, nst1, nst),
            dtype=self.torchComputationDtype, device=self.computationDevice
        )

        M[:, 0, 0] = 1.0  # dT/dt → dT/dt

        n_tot = C.sum(dim=1)  # (B,)
        M[:, 1, 0] = _R * n_tot  # ∂(dP)/∂(dT/dt)
        M[:, 1, 1:] = _R * T[:, None]  # ∂(dP)/∂(dC_k/dt)

        for k in range(ns):
            M[:, k + 2, k + 1] = self._W[k] / rho  # ∂(dY_k)/∂(dC_k/dt)

        return M

    # ── Interface publique ───────────────────────────────────────────────────

    def evaluate(self, T_arr, P_arr, Y_arr):
        """
        Paramètres
        ----------
        T_arr : Tensor (B,)       températures (K)
        P_arr : Tensor (B,)       pressions (Pa)
        Y_arr : Tensor (B, n_sp)  fractions massiques

        Retourne
        --------
        J_TPY : Tensor (B, n_state+1, n_state+1)
                espace (T, P, Y_0, …, Y_{ns-1})
        """
        T_arr = T_arr.to(self.torchComputationDtype).to(self.computationDevice)
        P_arr = P_arr.to(self.torchComputationDtype).to(self.computationDevice)
        Y_arr = Y_arr.to(self.torchComputationDtype).to(self.computationDevice)
        Y_arr = Y_arr / Y_arr.sum(dim=1, keepdim=True)

        TC, rho = self._encode(T_arr, P_arr, Y_arr)

        J_TC = self._ev.evaluate(
            cuda.as_cuda_array(TC[:, 0]),
            cuda.as_cuda_array(TC[:, 1:]),
        )
        J_TC = torch.as_tensor(J_TC).to(self.computationDevice).to(self.torchComputationDtype)

        dTC_dTPY = self._dTC_dTPY(T_arr, P_arr, Y_arr, rho)
        M_out = self._dRHSTPY_dRHSTC(T_arr, TC[:, 1:], rho)

        J_TPY = M_out @ J_TC @ dTC_dTPY

        return J_TPY.to(self.dataDevice).to(self.storageDtype)


_R = 8.314462618
_MAX_N_STATE = 54
_THREADS_PER_BLOCK = 64


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Kernel global : TC_in → dTCdt_out  (1 bloc = 1 réacteur)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(cache=True)
def _constantV_rhs_kernel_pb(
        TC_in,  # (B, n_state)  float64
        dTCdt_out,  # (B, n_state)  float64  — sortie
        ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e,
        ctx_sRre, ctx_sRpe,
        ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
        ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
        ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f,
        ctx_sRrf, ctx_sRpf,
        ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
        ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
        n_sp,
        thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1

    # Mémoire partagée
    sh_xk = cuda.shared.array(_MAX_N_STATE, float64)
    sh_om = cuda.shared.array(_MAX_N_STATE, float64)
    sh_dTCdt = cuda.shared.array(_MAX_N_STATE, float64)
    sh_scratch = cuda.shared.array(_THREADS_PER_BLOCK, float64)

    # Chargement de l'état
    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()

    T_loc = sh_xk[0]

    # Aliases contexte cinétique
    e_kf = ctx_kf_e[b];
    e_Kc = ctx_Kc_e[b]
    e_cr = ctx_cr_e[b];
    e_cp = ctx_cp_e[b];
    e_dG = ctx_dG0_e[b]
    e_sRr = ctx_sRre[b];
    e_sRp = ctx_sRpe[b]
    tb_kf = ctx_kf_tb[b];
    tb_Kc = ctx_Kc_tb[b]
    tb_cr = ctx_cr_tb[b];
    tb_cp = ctx_cp_tb[b];
    tb_dG = ctx_dG0_tb[b]
    tb_sRr = ctx_sRrtb[b];
    tb_sRp = ctx_sRptb[b]
    tb_bRR = ctx_bRR_tb[b];
    tb_M = ctx_M_tb[b]
    f_kf = ctx_kf_f[b];
    f_Kc = ctx_Kc_f[b]
    f_cr = ctx_cr_f[b];
    f_cp = ctx_cp_f[b];
    f_dG = ctx_dG0_f[b]
    f_sRr = ctx_sRrf[b];
    f_sRp = ctx_sRpf[b]
    f_k0 = ctx_k0_f[b];
    f_ki = ctx_kinf_f[b]
    f_M = ctx_M_f[b];
    f_Pr = ctx_Pr_f[b]
    f_F = ctx_F_f[b];
    f_kb = ctx_kb_f[b]
    f_Fc = ctx_Fc_f[b];
    f_lFc = ctx_lFc_f[b]
    f_Cp = ctx_Cp_f[b];
    f_Np = ctx_Np_f[b]
    f_f1 = ctx_f1_f[b];
    f_exp = ctx_exp_f[b]

    # Étape 1 : omega via _eval_rhs_and_context_pb
    _eval_rhs_and_context_pb(
        tid, n_threads,
        T_loc, sh_xk[1:],
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

    # Étape 2 : dTCdt = (dT/dt, omega) via _adiabatic_rhs_pb
    _adiabatic_rhs_pb(
        tid, n_threads,
        T_loc, sh_xk[1:], sh_om, n_sp, thL, thH, thM,
        sh_dTCdt,
        sh_scratch,
    )

    # Écriture en mémoire globale
    i = tid
    while i < n_state:
        dTCdt_out[b, i] = sh_dTCdt[i]
        i += n_threads
    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Évaluateur bas niveau TC → dTCdt
# ─────────────────────────────────────────────────────────────────────────────

class CUDAConstantVRHSEvaluatorPB:
    """
    Évalue le RHS isochore complet (dT/dt, dC/dt) dans l'espace TC — per-block.

    Usage
    -----
    ev    = CUDAConstantVRHSEvaluatorPB(mech)
    dTCdt = ev.evaluate(T_arr, C_arr)   # np.ndarray (B, n_state)
    """

    THREADS_PER_BLOCK = _THREADS_PER_BLOCK

    def __init__(self, mech, dtype=np.float64):
        self._mech = mech
        self._dtype = dtype
        self._n_sp = mech.n_species
        self._n_state = self._n_sp + 1
        self._ctx = None
        self._B_alloc = -1

    def _alloc_context(self, B):
        dt = self._dtype
        m = self._mech
        e = m.elementary
        tb = m.threebody
        f = m.falloff

        def z(shape):
            return cuda.device_array(shape, dtype=dt)

        n_e_flat = e.react_offset[-1] if e.n_reactions else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1

        self._ctx = dict(
            kf_e=z((B, e.n_reactions)), Kc_e=z((B, e.n_reactions)),
            cr_e=z((B, e.n_reactions)), cp_e=z((B, e.n_reactions)),
            dG0_e=z((B, e.n_reactions)),
            sRre=z((B, n_e_flat)), sRpe=z((B, n_e_flat)),
            kf_tb=z((B, tb.n_reactions)), Kc_tb=z((B, tb.n_reactions)),
            cr_tb=z((B, tb.n_reactions)), cp_tb=z((B, tb.n_reactions)),
            dG0_tb=z((B, tb.n_reactions)),
            sRrtb=z((B, n_tb_flat)), sRptb=z((B, n_tb_flat)),
            bRR_tb=z((B, tb.n_reactions)), M_tb=z((B, tb.n_reactions)),
            kf_f=z((B, f.n_reactions)), Kc_f=z((B, f.n_reactions)),
            cr_f=z((B, f.n_reactions)), cp_f=z((B, f.n_reactions)),
            dG0_f=z((B, f.n_reactions)),
            sRrf=z((B, f.react_offset[-1] if f.n_reactions else 1)),
            sRpf=z((B, f.react_offset[-1] if f.n_reactions else 1)),
            k0_f=z((B, f.n_reactions)), kinf_f=z((B, f.n_reactions)),
            M_f=z((B, f.n_reactions)), Pr_f=z((B, f.n_reactions)),
            F_f=z((B, f.n_reactions)), kb_f=z((B, f.n_reactions)),
            Fc_f=z((B, f.n_reactions)), lFc_f=z((B, f.n_reactions)),
            Cp_f=z((B, f.n_reactions)), Np_f=z((B, f.n_reactions)),
            f1_f=z((B, f.n_reactions)), exp_f=z((B, f.n_reactions)),
        )
        self._B_alloc = B

    def evaluate(self, T_arr, C_arr):
        T_arr = np.asarray(T_arr, dtype=self._dtype)
        C_arr = np.asarray(C_arr, dtype=self._dtype)
        B = T_arr.shape[0]
        nst = self._n_state

        if self._B_alloc != B:
            self._alloc_context(B)

        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0] = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)
        d_dTCdt = cuda.device_array((B, nst), dtype=self._dtype)

        m = self._mech;
        e = m.elementary;
        tb = m.threebody;
        f = m.falloff
        cx = self._ctx

        _constantV_rhs_kernel_pb[B, self.THREADS_PER_BLOCK](
            d_TC, d_dTCdt,
            cx['kf_e'], cx['Kc_e'], cx['cr_e'], cx['cp_e'], cx['dG0_e'],
            cx['sRre'], cx['sRpe'],
            cx['kf_tb'], cx['Kc_tb'], cx['cr_tb'], cx['cp_tb'], cx['dG0_tb'],
            cx['sRrtb'], cx['sRptb'], cx['bRR_tb'], cx['M_tb'],
            cx['kf_f'], cx['Kc_f'], cx['cr_f'], cx['cp_f'], cx['dG0_f'],
            cx['sRrf'], cx['sRpf'],
            cx['k0_f'], cx['kinf_f'], cx['M_f'], cx['Pr_f'], cx['F_f'],
            cx['kb_f'],
            cx['Fc_f'], cx['lFc_f'], cx['Cp_f'], cx['Np_f'],
            cx['f1_f'], cx['exp_f'],
            m.n_species,
            m.thermo.coef_low, m.thermo.coef_high, m.thermo.Tmid,
            e.n_reactions, e.A, e.n, e.Ea, e.oneWay,
            e.delta_n, e.react_idx, e.react_sto,
            e.react_offset, e.prod_idx, e.prod_sto, e.prod_offset,
            tb.n_reactions, tb.A, tb.n, tb.Ea, tb.oneWay,
            tb.delta_n, tb.react_idx, tb.react_sto,
            tb.react_offset, tb.prod_idx, tb.prod_sto, tb.prod_offset, tb.eff_matrix,
            f.n_reactions, f.A_k0, f.b_k0, f.Ea_k0, f.A_kinf, f.b_kinf, f.Ea_kinf,
            f.oneWay, f.delta_n,
            f.troe_A, f.troe_T1, f.troe_T2, f.troe_T3, f.is_troe,
            f.react_idx, f.react_sto, f.react_offset,
            f.prod_idx, f.prod_sto, f.prod_offset, f.eff_matrix,
        )
        cuda.synchronize()
        return d_dTCdt.copy_to_host()  # (B, n_state)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Wrapper TPY public
# ─────────────────────────────────────────────────────────────────────────────

class CUDAConstantVRHSTPY_PB:
    """
    RHS du réacteur isochore adiabatique dans l'espace (T, P, Y) — per-block.

    Usage
    -----
    ev   = CUDAConstantVRHSTPY_PB(
               mech,
               torchComputationDtype=torch.float64,
               dataDevice="cuda:0",
               computationDevice="cuda:0",
               storageDtype=torch.float64,
           )
    dTPY = ev.evaluate(T_arr, P_arr, Y_arr)
    # T_arr : Tensor (B,)       températures (K)
    # P_arr : Tensor (B,)       pressions (Pa)
    # Y_arr : Tensor (B, n_sp)  fractions massiques
    # dTPY  : Tensor (B, 2+ns)  → (dT/dt, dP/dt, dY_0/dt, …)
    """

    def __init__(self, mech, torchComputationDtype=None, dataDevice=None,
                 computationDevice=None, storageDtype=None):
        if torchComputationDtype == torch.float64:
            computationDtype = np.float64
        elif torchComputationDtype == torch.float32:
            computationDtype = np.float32
        else:
            raise NotImplementedError()

        self.torchComputationDtype = torchComputationDtype
        self.storageDtype = storageDtype
        self.dataDevice = dataDevice
        self.computationDevice = torch.device(computationDevice)

        y = thermo.yamlParser(mech)
        vec_th = y.populateVectorizedThermoObject(
            dtype=torchComputationDtype, device=dataDevice
        )
        reactions = reactionParser.parseReactions(mech)
        r_e, r_tb, r_f, system = reactionParser.build_reactions(
            vec_th.species, reactions, vec_th, dataDevice, torchComputationDtype
        )
        mech_data = build_mechanism_data_from_torch(
            r_e, r_tb, r_f, vec_th, dtype=computationDtype
        )

        self._ev = CUDAConstantVRHSEvaluatorPB(mech_data, dtype=computationDtype)
        self._dtype = computationDtype
        self._W = vec_th.molar_masses.to(torchComputationDtype).to(self.computationDevice)

    def _encode(self, T, P, Y):
        sYW = (Y / self._W[None, :]).sum(dim=1)
        rho = P / (_R * T * sYW)
        C = rho[:, None] * Y / self._W[None, :]
        return torch.cat([T[:, None], C], dim=1), rho

    def _decode_rhs(self, TC, dTC, rho):
        T = TC[:, 0];
        C = TC[:, 1:]
        dT = dTC[:, 0];
        dC = dTC[:, 1:]
        n_tot = C.sum(dim=1)
        dn_tot = dC.sum(dim=1)  # = Σ ω_k
        dP = _R * (T * dn_tot + n_tot * dT)
        dY = dC * self._W[None, :] / rho[:, None]  # dρ/dt = 0
        return torch.cat([dT[:, None], dP[:, None], dY], dim=1)

    def evaluate(self, T_arr, P_arr, Y_arr):
        T_arr = T_arr.to(self.torchComputationDtype).to(self.computationDevice)
        P_arr = P_arr.to(self.torchComputationDtype).to(self.computationDevice)
        Y_arr = Y_arr.to(self.torchComputationDtype).to(self.computationDevice)
        Y_arr = Y_arr / Y_arr.sum(dim=1, keepdim=True)

        TC, rho = self._encode(T_arr, P_arr, Y_arr)

        dTC_np = self._ev.evaluate(
            cuda.as_cuda_array(TC[:, 0]),
            cuda.as_cuda_array(TC[:, 1:]),
        )
        dTC = torch.as_tensor(dTC_np).to(self.computationDevice).to(self.torchComputationDtype)

        return self._decode_rhs(TC, dTC, rho).to(self.dataDevice).to(self.storageDtype)


_R = 8.314462618
_MAX_N_STATE = 54
_THREADS_PER_BLOCK = 64


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Kernel global : TC_in → dTCdt_out  (1 bloc = 1 réacteur)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(cache=True)
def _constantP_rhs_kernel_pb(
        TC_in,  # (B, n_state)  float64
        dTCdt_out,  # (B, n_state)  float64  — sortie
        ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e,
        ctx_sRre, ctx_sRpe,
        ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
        ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
        ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f,
        ctx_sRrf, ctx_sRpf,
        ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
        ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
        n_sp,
        thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1

    sh_xk = cuda.shared.array(_MAX_N_STATE, float64)
    sh_om = cuda.shared.array(_MAX_N_STATE, float64)
    sh_dTCdt = cuda.shared.array(_MAX_N_STATE, float64)
    sh_scratch = cuda.shared.array(_THREADS_PER_BLOCK, float64)

    # Chargement de l'état
    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()

    T_loc = sh_xk[0]

    # Aliases contexte cinétique
    e_kf = ctx_kf_e[b];
    e_Kc = ctx_Kc_e[b]
    e_cr = ctx_cr_e[b];
    e_cp = ctx_cp_e[b];
    e_dG = ctx_dG0_e[b]
    e_sRr = ctx_sRre[b];
    e_sRp = ctx_sRpe[b]
    tb_kf = ctx_kf_tb[b];
    tb_Kc = ctx_Kc_tb[b]
    tb_cr = ctx_cr_tb[b];
    tb_cp = ctx_cp_tb[b];
    tb_dG = ctx_dG0_tb[b]
    tb_sRr = ctx_sRrtb[b];
    tb_sRp = ctx_sRptb[b]
    tb_bRR = ctx_bRR_tb[b];
    tb_M = ctx_M_tb[b]
    f_kf = ctx_kf_f[b];
    f_Kc = ctx_Kc_f[b]
    f_cr = ctx_cr_f[b];
    f_cp = ctx_cp_f[b];
    f_dG = ctx_dG0_f[b]
    f_sRr = ctx_sRrf[b];
    f_sRp = ctx_sRpf[b]
    f_k0 = ctx_k0_f[b];
    f_ki = ctx_kinf_f[b]
    f_M = ctx_M_f[b];
    f_Pr = ctx_Pr_f[b]
    f_F = ctx_F_f[b];
    f_kb = ctx_kb_f[b]
    f_Fc = ctx_Fc_f[b];
    f_lFc = ctx_lFc_f[b]
    f_Cp = ctx_Cp_f[b];
    f_Np = ctx_Np_f[b]
    f_f1 = ctx_f1_f[b];
    f_exp = ctx_exp_f[b]

    # Étape 1 : omega via _eval_rhs_and_context_pb
    _eval_rhs_and_context_pb(
        tid, n_threads,
        T_loc, sh_xk[1:],
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

    # Étape 2 : RHS isobare complet via _constantP_rhs_pb
    _constantP_rhs_pb(
        tid, n_threads,
        T_loc, sh_xk[1:], sh_om, n_sp, thL, thH, thM,
        sh_dTCdt,
        sh_scratch,
    )

    # Écriture en mémoire globale
    i = tid
    while i < n_state:
        dTCdt_out[b, i] = sh_dTCdt[i]
        i += n_threads
    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Évaluateur bas niveau TC → dTCdt
# ─────────────────────────────────────────────────────────────────────────────

class CUDAConstantPRHSEvaluatorPB:
    """
    Évalue le RHS isobare complet (dT/dt, dC/dt) dans l'espace TC — per-block.

    Usage
    -----
    ev    = CUDAConstantPRHSEvaluatorPB(mech)
    dTCdt = ev.evaluate(T_arr, C_arr)   # np.ndarray (B, n_state)
    """

    THREADS_PER_BLOCK = _THREADS_PER_BLOCK

    def __init__(self, mech, dtype=np.float64):
        self._mech = mech
        self._dtype = dtype
        self._n_sp = mech.n_species
        self._n_state = self._n_sp + 1
        self._ctx = None
        self._B_alloc = -1

    def _alloc_context(self, B):
        dt = self._dtype
        m = self._mech
        e = m.elementary
        tb = m.threebody
        f = m.falloff

        def z(shape):
            return cuda.device_array(shape, dtype=dt)

        n_e_flat = e.react_offset[-1] if e.n_reactions else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1

        self._ctx = dict(
            kf_e=z((B, e.n_reactions)), Kc_e=z((B, e.n_reactions)),
            cr_e=z((B, e.n_reactions)), cp_e=z((B, e.n_reactions)),
            dG0_e=z((B, e.n_reactions)),
            sRre=z((B, n_e_flat)), sRpe=z((B, n_e_flat)),
            kf_tb=z((B, tb.n_reactions)), Kc_tb=z((B, tb.n_reactions)),
            cr_tb=z((B, tb.n_reactions)), cp_tb=z((B, tb.n_reactions)),
            dG0_tb=z((B, tb.n_reactions)),
            sRrtb=z((B, n_tb_flat)), sRptb=z((B, n_tb_flat)),
            bRR_tb=z((B, tb.n_reactions)), M_tb=z((B, tb.n_reactions)),
            kf_f=z((B, f.n_reactions)), Kc_f=z((B, f.n_reactions)),
            cr_f=z((B, f.n_reactions)), cp_f=z((B, f.n_reactions)),
            dG0_f=z((B, f.n_reactions)),
            sRrf=z((B, f.react_offset[-1] if f.n_reactions else 1)),
            sRpf=z((B, f.react_offset[-1] if f.n_reactions else 1)),
            k0_f=z((B, f.n_reactions)), kinf_f=z((B, f.n_reactions)),
            M_f=z((B, f.n_reactions)), Pr_f=z((B, f.n_reactions)),
            F_f=z((B, f.n_reactions)), kb_f=z((B, f.n_reactions)),
            Fc_f=z((B, f.n_reactions)), lFc_f=z((B, f.n_reactions)),
            Cp_f=z((B, f.n_reactions)), Np_f=z((B, f.n_reactions)),
            f1_f=z((B, f.n_reactions)), exp_f=z((B, f.n_reactions)),
        )
        self._B_alloc = B

    def evaluate(self, T_arr, C_arr):
        T_arr = np.asarray(T_arr, dtype=self._dtype)
        C_arr = np.asarray(C_arr, dtype=self._dtype)
        B = T_arr.shape[0]
        nst = self._n_state

        if self._B_alloc != B:
            self._alloc_context(B)

        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0] = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)
        d_dTCdt = cuda.device_array((B, nst), dtype=self._dtype)

        m = self._mech;
        e = m.elementary;
        tb = m.threebody;
        f = m.falloff
        cx = self._ctx

        _constantP_rhs_kernel_pb[B, self.THREADS_PER_BLOCK](
            d_TC, d_dTCdt,
            cx['kf_e'], cx['Kc_e'], cx['cr_e'], cx['cp_e'], cx['dG0_e'],
            cx['sRre'], cx['sRpe'],
            cx['kf_tb'], cx['Kc_tb'], cx['cr_tb'], cx['cp_tb'], cx['dG0_tb'],
            cx['sRrtb'], cx['sRptb'], cx['bRR_tb'], cx['M_tb'],
            cx['kf_f'], cx['Kc_f'], cx['cr_f'], cx['cp_f'], cx['dG0_f'],
            cx['sRrf'], cx['sRpf'],
            cx['k0_f'], cx['kinf_f'], cx['M_f'], cx['Pr_f'], cx['F_f'],
            cx['kb_f'],
            cx['Fc_f'], cx['lFc_f'], cx['Cp_f'], cx['Np_f'],
            cx['f1_f'], cx['exp_f'],
            m.n_species,
            m.thermo.coef_low, m.thermo.coef_high, m.thermo.Tmid,
            e.n_reactions, e.A, e.n, e.Ea, e.oneWay,
            e.delta_n, e.react_idx, e.react_sto,
            e.react_offset, e.prod_idx, e.prod_sto, e.prod_offset,
            tb.n_reactions, tb.A, tb.n, tb.Ea, tb.oneWay,
            tb.delta_n, tb.react_idx, tb.react_sto,
            tb.react_offset, tb.prod_idx, tb.prod_sto, tb.prod_offset, tb.eff_matrix,
            f.n_reactions, f.A_k0, f.b_k0, f.Ea_k0, f.A_kinf, f.b_kinf, f.Ea_kinf,
            f.oneWay, f.delta_n,
            f.troe_A, f.troe_T1, f.troe_T2, f.troe_T3, f.is_troe,
            f.react_idx, f.react_sto, f.react_offset,
            f.prod_idx, f.prod_sto, f.prod_offset, f.eff_matrix,
        )
        cuda.synchronize()
        return d_dTCdt.copy_to_host()  # (B, n_state)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Wrapper TPY public
# ─────────────────────────────────────────────────────────────────────────────

class CUDAConstantPRHSTPY_PB:
    """
    RHS du réacteur isobare dans l'espace (T, P, Y) — per-block.

    Usage
    -----
    ev   = CUDAConstantPRHSTPY_PB(
               mech,
               torchComputationDtype=torch.float64,
               dataDevice="cuda:0",
               computationDevice="cuda:0",
               storageDtype=torch.float64,
           )
    dTPY = ev.evaluate(T_arr, P_arr, Y_arr)
    # T_arr : Tensor (B,)       températures (K)
    # P_arr : Tensor (B,)       pressions (Pa)
    # Y_arr : Tensor (B, n_sp)  fractions massiques
    # dTPY  : Tensor (B, 2+ns)  → (dT/dt, 0, dY_0/dt, …, dY_{ns-1}/dt)
    """

    def __init__(self, mech, torchComputationDtype=None, dataDevice=None,
                 computationDevice=None, storageDtype=None):
        if torchComputationDtype == torch.float64:
            computationDtype = np.float64
        elif torchComputationDtype == torch.float32:
            computationDtype = np.float32
        else:
            raise NotImplementedError()

        self.torchComputationDtype = torchComputationDtype
        self.storageDtype = storageDtype
        self.dataDevice = dataDevice
        self.computationDevice = torch.device(computationDevice)

        y = thermo.yamlParser(mech)
        vec_th = y.populateVectorizedThermoObject(
            dtype=torchComputationDtype, device=dataDevice
        )
        reactions = reactionParser.parseReactions(mech)
        r_e, r_tb, r_f, system = reactionParser.build_reactions(
            vec_th.species, reactions, vec_th, dataDevice, torchComputationDtype
        )
        mech_data = build_mechanism_data_from_torch(
            r_e, r_tb, r_f, vec_th, dtype=computationDtype
        )

        self._ev = CUDAConstantPRHSEvaluatorPB(mech_data, dtype=computationDtype)
        self._dtype = computationDtype
        self._W = vec_th.molar_masses.to(torchComputationDtype).to(self.computationDevice)

    def _encode(self, T, P, Y):
        sYW = (Y / self._W[None, :]).sum(dim=1)
        rho = P / (_R * T * sYW)
        C = rho[:, None] * Y / self._W[None, :]
        return torch.cat([T[:, None], C], dim=1), rho

    def _decode_rhs(self, TC, dTC, rho):
        """
        dP/dt = 0 (contrainte isobare)
        dY_k/dt = (W_k·dC_k - Y_k·dρ) / ρ,   dρ = Σ W_l·dC_l
        """
        dT = dTC[:, 0]
        dC = dTC[:, 1:]
        dP = torch.zeros_like(dT)
        drho = (dC * self._W[None, :]).sum(dim=1, keepdim=True)  # (B, 1)
        Y = TC[:, 1:] * self._W[None, :] / rho[:, None]
        dY = (dC * self._W[None, :] - Y * drho) / rho[:, None]
        return torch.cat([dT[:, None], dP[:, None], dY], dim=1)

    def evaluate(self, T_arr, P_arr, Y_arr):
        T_arr = T_arr.to(self.torchComputationDtype).to(self.computationDevice)
        P_arr = P_arr.to(self.torchComputationDtype).to(self.computationDevice)
        Y_arr = Y_arr.to(self.torchComputationDtype).to(self.computationDevice)
        Y_arr = Y_arr / Y_arr.sum(dim=1, keepdim=True)

        TC, rho = self._encode(T_arr, P_arr, Y_arr)

        dTC_np = self._ev.evaluate(
            cuda.as_cuda_array(TC[:, 0]),
            cuda.as_cuda_array(TC[:, 1:]),
        )
        dTC = torch.as_tensor(dTC_np).to(self.computationDevice).to(self.torchComputationDtype)

        return self._decode_rhs(TC, dTC, rho).to(self.dataDevice).to(self.storageDtype)