"""
kernels/shared/reactor_odes.py — Équations du réacteur adiabatique (scalaire).

Contient les formulations mathématiques du réacteur :
    - RHS adiabatique (dT/dt, dC/dt)
    - JVP du RHS adiabatique

Ces fonctions sont la version SCALAIRE (per-thread). Les versions parallèles
(per-block) sont dans per_block/reactor_odes_pb.py.

Fonctions exportées :
    _adiabatic_rhs(T, C, omega, n_sp, thL, thH, thM, dTCdt)
    _adiabatic_jvp(T, C, vT, vC, omega, Jv_om, n_sp, thL, thH, thM, Jv_TC)
"""

import math
from numba import cuda, float64

from kineticsGPU.cuda.kernels.shared.thermo import _load_coef7, _h_RT, _s_R, _cp_R, _kf_arr

import numpy as np

from ..shared.thermo import EPS
from .context import _eval_rhs_and_context
from .jacobian import _eval_jacobian

R_CONST = 8.314_462_618_153_24
EPS     = 1e-300


@cuda.jit(device=True)
def _adiabatic_rhs(T, C, omega, n_sp, thL, thH, thM, dTCdt):
    """
    Calcule le RHS du réacteur adiabatique à volume constant.

    dT/dt = -sum(u_s * omega_s) / (n_tot * cv_mix)
    dC_s/dt = omega_s

    Paramètres
    ----------
    T     : float64     — température (K)
    C     : array(n_sp) — concentrations (mol/m³)
    omega : array(n_sp) — taux de production (mol/m³/s)
    dTCdt : array(n_sp+1) — sortie [dT/dt, dC_0/dt, ..., dC_{n-1}/dt]
    """
    c7   = cuda.local.array(7, float64)
    ntot = float64(0.0)
    for s in range(n_sp):
        ntot += C[s]
    cv_mix = float64(0.0)
    su_om  = float64(0.0)
    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        cp_r = _cp_R(T, c7)
        cv_s = R_CONST * (cp_r - 1.0)
        hRT  = _h_RT(T, c7)
        u_s  = R_CONST * T * hRT - R_CONST * T
        Xs   = C[s] / ntot if ntot > EPS else float64(0.0)
        cv_mix += Xs * cv_s
        su_om  += u_s * omega[s]
    denom     = ntot * cv_mix
    dTCdt[0]  = -su_om / denom if abs(denom) > EPS else float64(0.0)
    for s in range(n_sp):
        dTCdt[s + 1] = omega[s]


@cuda.jit(device=True)
def _adiabatic_jvp(T, C, vT, vC, omega, Jv_om, n_sp, thL, thH, thM, Jv_TC):
    """
    Calcule la JVP (produit Jacobien-vecteur) du RHS adiabatique.

    Dérivée directionnelle de _adiabatic_rhs dans la direction (vT, vC).

    Paramètres
    ----------
    T, C           : état de base
    vT, vC         : direction de la perturbation
    omega          : taux de production au point de base (déjà calculé)
    Jv_om          : JVP de omega (déjà calculé par _eval_jvp)
    Jv_TC          : sortie — JVP du RHS complet [JvdT/dt, JvdC/dt]
    """
    c7    = cuda.local.array(7, float64)
    ntot  = float64(0.0)
    for s in range(n_sp):
        ntot += C[s]
    dntot = float64(0.0)
    for s in range(n_sp):
        dntot += vC[s]
    cv_mix = float64(0.0); dcv = float64(0.0)
    up     = float64(0.0); dup = float64(0.0)
    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        T2 = T * T; T3 = T2 * T; T4 = T3 * T
        cpR   = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        cv_s  = R_CONST * (cpR - 1.0)
        dcvdT = R_CONST * (c7[1] + 2.0*c7[2]*T + 3.0*c7[3]*T2 + 4.0*c7[4]*T3)
        hRT   = (c7[0] + c7[1]*T/2.0 + c7[2]*T2/3.0
                 + c7[3]*T3/4.0 + c7[4]*T4/5.0 + c7[5]/T)
        u_s   = R_CONST * T * hRT - R_CONST * T
        Xs    = C[s] / ntot if ntot > EPS else float64(0.0)
        dXs   = ((vC[s] / ntot - C[s] * dntot / (ntot * ntot))
                 if ntot > EPS else float64(0.0))
        cv_mix += Xs * cv_s
        dcv    += dXs * cv_s + Xs * dcvdT * vT
        up     += u_s * omega[s]
        dup    += u_s * Jv_om[s] + cv_s * omega[s] * vT
    denom = ntot * cv_mix
    if abs(denom) > EPS:
        Jv_TC[0] = (-dup / denom
                    + up * dntot / (denom * ntot)
                    + up * dcv / (ntot * cv_mix * cv_mix))
    else:
        Jv_TC[0] = float64(0.0)
    for s in range(n_sp):
        Jv_TC[s + 1] = Jv_om[s]


# =============================================================================
#  Jacobienne complète du réacteur adiabatique : d(RHS)/d(xk)
#  xk = (T, C_0, ..., C_{n_sp-1})
#  RHS = (dT/dt, dC_0/dt, ..., dC_{n_sp-1}/dt)
#
#  La matrice J_full est de taille (n_state × n_state), row-major :
#      J_full[i * n_state + j] = d(RHS_i) / d(xk_j)
#
#  Ligne i=0        → d(dT/dt)   / d(xk_j)
#  Ligne i=s+1      → d(dCs/dt)  / d(xk_j) = d(omega_s) / d(xk_j)
#
#  Colonne j=0      → dérivée par rapport à T
#  Colonne j=s'+1   → dérivée par rapport à C_{s'}
# =============================================================================

@cuda.jit(device=True)
def _adiabatic_full_jacobian(
    T, C, omega, J_chem,
    n_sp, thL, thH, thM,
    J_full,
):
    """
    Jacobienne complète du réacteur adiabatique à volume constant.

    J_full[i, j] = d(RHS_i) / d(xk_j),  xk = (T, C_0, …, C_{n_sp-1})

    Paramètres
    ----------
    T       : float64           — température (K)
    C       : array(n_sp)       — concentrations (mol/m³)
    omega   : array(n_sp)       — taux de production (mol/m³/s) au point (T,C)
    J_chem  : array(n_sp*n_state) — jacobienne chimique d(omega)/d(xk), row-major,
                                    calculée par _eval_jacobian (jacobian_chem.py)
    n_sp    : int               — nombre d'espèces
    thL/H/M : tableaux NASA-7
    J_full  : array(n_state*n_state) — sortie, row-major
                J_full[i*n_state + j] = d(RHS_i)/d(xk_j)

    Notes
    -----
    Les lignes i >= 1 (espèces) sont directement les lignes de J_chem :
        d(dC_s/dt)/d(xk_j) = d(omega_s)/d(xk_j)

    La ligne i=0 (température) est calculée analytiquement :

        dT/dt = -( Σ_s u_s * omega_s ) / ( n_tot * cv_mix )
              = -U_num / D

    Dérivée par rapport à xk_j :

        d(dT/dt)/d(xk_j) = -[ d(U_num)/d(xk_j) * D
                               - U_num * d(D)/d(xk_j) ] / D²

    Termes nécessaires :
        d(U_num)/d(xk_j) = Σ_s [ u_s * J_chem[s,j]
                                  + (du_s/d(xk_j)) * omega_s ]
            où  du_s/dT      = cv_s  (dérivée de l'énergie interne molaire)
                du_s/d(C_k)  = 0     (u_s ne dépend pas des concentrations)

        d(D)/d(xk_j) = d(n_tot * cv_mix)/d(xk_j)
            avec n_tot = Σ C_s, cv_mix = Σ X_s cv_s, X_s = C_s/n_tot

            d(n_tot)/dT        = 0
            d(n_tot)/d(C_k)    = 1
            d(cv_mix)/dT       = Σ X_s * (d(cv_s)/dT)  (heat capacity derivative)
            d(cv_mix)/d(C_k)   = (cv_k - cv_mix) / n_tot  (changement de fraction molaire)

            d(D)/dT         = n_tot * d(cv_mix)/dT
            d(D)/d(C_k)     = cv_mix + n_tot * (cv_k - cv_mix) / n_tot
                             = cv_k
    """
    n_state = n_sp + 1
    c7      = cuda.local.array(7, float64)

    # ── Pré-calcul des quantités thermodynamiques ──────────────────────
    ntot   = float64(0.0)
    for s in range(n_sp):
        ntot += C[s]

    cv_mix  = float64(0.0)
    U_num   = float64(0.0)   # Σ u_s * omega_s
    D       = float64(0.0)   # n_tot * cv_mix  (calculé après la boucle)

    # Tableaux temporaires (en registres, taille fixe max)
    # On a besoin de cv_s[s] et u_s[s] pour chaque espèce
    # Pour CUDA on les recalcule à la volée dans la boucle colonne.
    # Ici on fait une première passe pour obtenir cv_mix et U_num.

    dcv_dT = float64(0.0)   # d(cv_mix)/dT = Σ X_s * d(cv_s)/dT

    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        T2  = T * T; T3 = T2 * T; T4 = T3 * T
        cpR = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        cv_s = R_CONST * (cpR - float64(1.0))
        hRT  = (c7[0] + c7[1]*T/float64(2.0) + c7[2]*T2/float64(3.0)
                + c7[3]*T3/float64(4.0) + c7[4]*T4/float64(5.0) + c7[5]/T)
        u_s  = R_CONST * T * hRT - R_CONST * T

        Xs   = C[s] / ntot if ntot > EPS else float64(0.0)

        dcvs_dT = R_CONST * (c7[1] + float64(2.0)*c7[2]*T
                              + float64(3.0)*c7[3]*T2
                              + float64(4.0)*c7[4]*T3)

        cv_mix  += Xs * cv_s
        U_num   += u_s * omega[s]
        dcv_dT  += Xs * dcvs_dT

    D = ntot * cv_mix
    D_safe = D if abs(D) > EPS else (EPS if D >= float64(0.0) else -EPS)
    D_sq   = D_safe * D_safe

    # ── Ligne i=0 : d(dT/dt)/d(xk_j) pour j=0..n_state-1 ─────────────
    for j in range(n_state):
        # d(U_num)/d(xk_j)
        dU_j = float64(0.0)
        # terme u_s * J_chem[s, j]
        for s in range(n_sp):
            _load_coef7(T, s, thM, thL, thH, c7)
            T2  = T * T; T3 = T2 * T; T4 = T3 * T
            hRT = (c7[0] + c7[1]*T/float64(2.0) + c7[2]*T2/float64(3.0)
                   + c7[3]*T3/float64(4.0) + c7[4]*T4/float64(5.0) + c7[5]/T)
            u_s = R_CONST * T * hRT - R_CONST * T
            dU_j += u_s * J_chem[s * n_state + j]

        if j == 0:
            # contribution de (du_s/dT) * omega_s  pour j=0
            for s in range(n_sp):
                _load_coef7(T, s, thM, thL, thH, c7)
                T2   = T * T; T3 = T2 * T; T4 = T3 * T
                cpR  = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
                cv_s = R_CONST * (cpR - float64(1.0))
                dU_j += cv_s * omega[s]

        # d(D)/d(xk_j)
        if j == 0:
            # d(D)/dT = n_tot * d(cv_mix)/dT
            dD_j = ntot * dcv_dT
        else:
            # d(D)/d(C_k) = cv_k   (démonstration dans la docstring)
            s_k  = j - 1
            _load_coef7(T, s_k, thM, thL, thH, c7)
            T2   = T * T; T3 = T2 * T; T4 = T3 * T
            cpR  = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
            cv_k = R_CONST * (cpR - float64(1.0))
            dD_j = cv_k

        # d(dT/dt)/d(xk_j) = -(dU_j * D - U_num * dD_j) / D²
        J_full[0 * n_state + j] = -(dU_j * D_safe - U_num * dD_j) / D_sq

    # ── Lignes i=1..n_sp : d(dC_s/dt)/d(xk_j) = J_chem[s, j] ─────────
    for s in range(n_sp):
        for j in range(n_state):
            J_full[(s + 1) * n_state + j] = J_chem[s * n_state + j]

"""
kernels/per_thread/full_jacobian.py — Jacobienne complète du réacteur adiabatique.

Architecture : 1 thread = 1 réacteur (scalaire).

Calcule la matrice jacobienne complète J de taille (n_state × n_state) :

    J[i, j] = d(RHS_i) / d(xk_j)

où RHS = (dT/dt, dC_0/dt, ..., dC_{n_sp-1}/dt)
et  xk  = (T, C_0, ..., C_{n_sp-1})

Convention :
    J_out est un tableau 1D de taille n_state², row-major.
    J_out[i * n_state + j] = d(RHS_i) / d(xk_j)

Ligne i=0  → équation d'énergie (dT/dt)
Ligne i≥1  → équation espèce (dC_{i-1}/dt = omega_{i-1})
"""

MAX_N_STATE = 54
MAX_N_STATE2 = MAX_N_STATE * MAX_N_STATE


# =============================================================================
#  Kernel CUDA global
# =============================================================================

@cuda.jit(cache=True)
def _full_jacobian_kernel_pt(
    TC_in,           # (B, n_state)  float64
    J_all,           # (B, n_state²) float64  — sortie
    # ── contexte cinétique (pré-alloué, rempli par _eval_rhs_and_context) ──
    ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e,
    ctx_sRre, ctx_sRpe,
    ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f,
    ctx_sRrf, ctx_sRpf,
    ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
    ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
    # ── données mécanisme ──
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
    T_loc   = TC_in[b, 0]

    C      = cuda.local.array(MAX_N_STATE, float64)
    om     = cuda.local.array(MAX_N_STATE, float64)
    J_chem = cuda.local.array(MAX_N_STATE2, float64)

    for s in range(n_sp):
        C[s] = TC_in[b, s + 1]

    # 1) Calcul du contexte cinétique + omega au point (T, C)
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

    # 2) Jacobienne chimique pure d(omega)/d(xk)  → J_chem (n_sp × n_state)
    _eval_jacobian(
        T_loc, C,
        J_chem,
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

    # 3) Jacobienne complète (ligne T + lignes espèces)
    _adiabatic_full_jacobian(
        T_loc, C, om, J_chem,
        n_sp, thL, thH, thM,
        J_all[b],
    )


# =============================================================================
#  Wrapper Python
# =============================================================================

class CUDAFullJacobianEvaluatorPT:
    """
    Évalue la jacobienne complète du réacteur adiabatique à volume constant.

    J[i, j] = d(RHS_i) / d(xk_j)

    avec RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)
    et   xk  = (T, C_0, …, C_{n_sp-1})

    Usage
    -----
    ev = CUDAFullJacobianEvaluatorPT(mech)
    J  = ev.evaluate(T_arr, C_arr)
    # T_arr : (B,)        float64
    # C_arr : (B, n_sp)   float64
    # J     : (B, n_state, n_state)  float64,  n_state = n_sp + 1

    Indexation de J[b]
    ------------------
    J[b, 0, j]   = d(dT/dt)    / d(xk_j)   ← ligne énergie
    J[b, s+1, j] = d(dC_s/dt)  / d(xk_j)   ← ligne espèce s
    J[b, i, 0]   = d(RHS_i)    / dT         ← colonne T
    J[b, i, k+1] = d(RHS_i)    / d(C_k)     ← colonne concentration k

    Notes
    -----
    - Aucune mémoire GPU n'est allouée entre deux appels à evaluate() si
      la taille de batch B ne change pas.
    - La jacobienne chimique (lignes i≥1) est calculée par _eval_jacobian
      (jacobian_chem.py, corrigée PyJac).
    - La ligne d'énergie (i=0) est calculée analytiquement dans
      _adiabatic_full_jacobian (reactor_odes.py).
    """

    THREADS = 64

    def __init__(self, mech, dtype=np.float64):
        self._mech    = mech
        self._dtype   = dtype
        self._n_sp    = mech.n_species
        self._n_state = self._n_sp + 1
        self._ctx     = None
        self._B_alloc = -1

    # ------------------------------------------------------------------
    def _alloc_context(self, B):
        """Alloue les buffers GPU du contexte cinétique pour un batch de taille B."""
        dt = self._dtype
        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff

        def z(shape):
            return cuda.device_array(shape, dtype=dt)

        n_e_flat  = e.react_offset[-1]  if e.n_reactions  else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1
        n_f_flat  = f.react_offset[-1]  if f.n_reactions  else 1

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
            sRrf   = z((B, n_f_flat)),         sRpf   = z((B, n_f_flat)),
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
        Calcule J[b] = d(RHS)/d(xk) pour chaque réacteur b du batch.

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

        # Assemblage de l'entrée (T, C_0, …, C_{n_sp-1})
        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0]  = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)

        # Sortie GPU : (B, n_state²)
        d_J = cuda.device_array((B, nst * nst), dtype=self._dtype)

        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff
        cx = self._ctx

        threads = self.THREADS
        blocks  = (B + threads - 1) // threads

        _full_jacobian_kernel_pt[blocks, threads](
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
#  1.  Fonction device scalaire
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _constantP_full_jacobian(
    T, C, omega, J_chem,
    n_sp, thL, thH, thM,
    J_full,
):
    """
    Jacobienne complète du réacteur isobare à pression constante (scalaire).

    J_full[i * n_state + j] = d(RHS_i) / d(xk_j)
    xk  = (T, C_0, …, C_{n_sp-1})
    RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)

    Paramètres
    ----------
    T       : float64              — température (K)
    C       : array(n_sp)          — concentrations (mol/m³)
    omega   : array(n_sp)          — taux de production au point (T, C)
    J_chem  : array(n_sp*n_state)  — d(omega_s)/d(xk_j), row-major
    n_sp    : int
    thL/H/M : tableaux NASA-7
    J_full  : array(n_state*n_state) — sortie row-major

    Dérivation — ligne énergie (i=0)
    ---------------------------------
    dT/dt = -H_num / D
        H_num = Σ_s h_s·ω_s
        D     = n_tot·cp_mix = Σ_s C_s · cp_s   (invariant en compositions)

    d(dT/dt)/d(xk_j) = -(dH_j·D - H_num·dD_j) / D²

    dH_j :
        j = 0       → Σ_s cp_s·ω_s  +  Σ_s h_s·J_chem[s,0]
        j = k+1     → Σ_s h_s·J_chem[s,j]

    dD_j :
        j = 0       → n_tot · Σ_s X_s·(d(cp_s)/dT)
        j = k+1     → cp_k

    Dérivation — lignes espèces (i=s+1)
    ------------------------------------
    dC_s/dt = ω_s - C_s·δ
        δ = Σω_k/n_tot + (dT/dt)/T

    d(dC_s/dt)/d(xk_j) = J_chem[s,j]
                          - C_s · (∂δ/∂xk_j)
                          - δ · 𝟙_{j=s+1}

    ∂δ/∂xk_j :
        = (Σ_k J_chem[k,j]) / n_tot
          + J_full[0,j] / T
          + [j=0]:   dTdt·(-1/T²)           ← d(1/T)/dT·dTdt  (déjà inclus via J[0,0])
          + [j≥1]:   -ω_sum / n_tot²         ← d(1/n_tot)/dC_k · ω_sum

    Remarque : J_full[0,j] est disponible dès qu'on a fini la ligne i=0,
    ce qui permet de calculer les lignes espèces dans la foulée.
    """
    n_state = n_sp + 1
    c7      = cuda.local.array(7, float64)

    # ── Passe 1 : scalaires thermodynamiques ─────────────────────────────────

    ntot = float64(0.0)
    for s in range(n_sp):
        ntot += C[s]

    cp_mix    = float64(0.0)
    H_num     = float64(0.0)
    dcp_dT    = float64(0.0)
    omega_sum = float64(0.0)

    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        T2  = T * T; T3 = T2 * T; T4 = T3 * T
        cpR = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        cp_s    = R_CONST * cpR
        dcps_dT = R_CONST * (c7[1]
                             + float64(2.0)*c7[2]*T
                             + float64(3.0)*c7[3]*T2
                             + float64(4.0)*c7[4]*T3)
        hRT = (c7[0] + c7[1]*T/float64(2.0) + c7[2]*T2/float64(3.0)
               + c7[3]*T3/float64(4.0) + c7[4]*T4/float64(5.0) + c7[5]/T)
        h_s = R_CONST * T * hRT
        Xs  = C[s] / ntot if ntot > EPS else float64(0.0)

        cp_mix    += Xs  * cp_s
        H_num     += h_s * omega[s]
        dcp_dT    += Xs  * dcps_dT
        omega_sum += omega[s]

    D      = ntot * cp_mix
    D_safe = D if abs(D) > EPS else (EPS if D >= float64(0.0) else -EPS)
    D_sq   = D_safe * D_safe
    dTdt   = -H_num / D_safe
    delta  = omega_sum / ntot + dTdt / T

    # ── Passe 2 : ligne i=0 — d(dT/dt)/d(xk_j) ──────────────────────────────

    for j in range(n_state):
        dH_j = float64(0.0)
        for s in range(n_sp):
            _load_coef7(T, s, thM, thL, thH, c7)
            T2  = T * T; T3 = T2 * T; T4 = T3 * T
            hRT = (c7[0] + c7[1]*T/float64(2.0) + c7[2]*T2/float64(3.0)
                   + c7[3]*T3/float64(4.0) + c7[4]*T4/float64(5.0) + c7[5]/T)
            h_s  = R_CONST * T * hRT
            dH_j += h_s * J_chem[s * n_state + j]

        if j == 0:
            # d(H_num)/dT += Σ_s cp_s·ω_s
            for s in range(n_sp):
                _load_coef7(T, s, thM, thL, thH, c7)
                T2  = T * T; T3 = T2 * T; T4 = T3 * T
                cpR  = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
                cp_s = R_CONST * cpR
                dH_j += cp_s * omega[s]
            dD_j = ntot * dcp_dT
        else:
            s_k  = j - 1
            _load_coef7(T, s_k, thM, thL, thH, c7)
            T2   = T * T; T3 = T2 * T; T4 = T3 * T
            cpR  = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
            cp_k = R_CONST * cpR
            dD_j = cp_k

        J_full[0 * n_state + j] = -(dH_j * D_safe - H_num * dD_j) / D_sq

    # ── Passe 3 : lignes i=s+1 — d(dC_s/dt)/d(xk_j) ─────────────────────────
    #
    # Pour chaque (s, j) on a besoin de :
    #   col_sum_j = Σ_k J_chem[k, j]    → boucle interne sur k
    #   J_full[0, j]                     → déjà calculé

    for s in range(n_sp):
        for j in range(n_state):
            # Somme colonne j de J_chem
            col_sum = float64(0.0)
            for k in range(n_sp):
                col_sum += J_chem[k * n_state + j]

            # ∂δ/∂xk_j
            ddelta_j = col_sum / ntot + J_full[0 * n_state + j] / T
            if j == 0:
                # d(dTdt/T)/dT — le terme J[0,0]/T est déjà dans ddelta_j,
                # il reste le terme dTdt·(-1/T²) issu de d(1/T)/dT
                ddelta_j += dTdt * (-float64(1.0) / (T * T))
            else:
                # d(ω_sum/n_tot)/dC_k = -ω_sum/n_tot²
                ddelta_j += -omega_sum / (ntot * ntot)

            indicator = float64(1.0) if (j == s + 1) else float64(0.0)

            J_full[(s + 1) * n_state + j] = (J_chem[s * n_state + j]
                                              - C[s] * ddelta_j
                                              - delta * indicator)


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Kernel CUDA global
# ─────────────────────────────────────────────────────────────────────────────

MAX_N_STATE  = 54
MAX_N_STATE2 = MAX_N_STATE * MAX_N_STATE


@cuda.jit(cache=True)
def _full_jacobian_constantP_kernel_pt(
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
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1
    T_loc   = TC_in[b, 0]

    C      = cuda.local.array(MAX_N_STATE,  float64)
    om     = cuda.local.array(MAX_N_STATE,  float64)
    J_chem = cuda.local.array(MAX_N_STATE2, float64)
    J_buf  = cuda.local.array(MAX_N_STATE2, float64)

    for s in range(n_sp):
        C[s] = TC_in[b, s + 1]

    # Étape 1 : contexte cinétique + omega
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

    # Étape 2 : jacobienne chimique d(omega)/d(xk) → J_chem (n_sp × n_state)
    _eval_jacobian(
        T_loc, C,
        J_chem,
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

    # Étape 3 : jacobienne complète isobare → J_buf (local)
    _constantP_full_jacobian(
        T_loc, C, om, J_chem,
        n_sp, thL, thH, thM,
        J_buf,
    )

    # Copie J_buf → J_all[b] (mémoire globale)
    for idx in range(n_state * n_state):
        J_all[b, idx] = J_buf[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Wrapper Python : entrées/sorties en TC
# ─────────────────────────────────────────────────────────────────────────────

class CUDAFullJacobianConstantPEvaluatorPT:
    """
    Jacobienne complète du réacteur isobare (per-thread), coordonnées TC.

    J[i, j] = d(RHS_i) / d(xk_j)

    RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)  — réacteur P=cte
    xk  = (T, C_0, …, C_{n_sp-1})

    Différences vs CUDAFullJacobianEvaluatorPT (adiabatique)
    ---------------------------------------------------------
    • Ligne énergie  : h/cp au lieu de u/cv
    • Lignes espèces : couplage via le terme de dilatation δ
    • Adapté aux grands batches B (fort parallélisme inter-réacteurs)

    Usage
    -----
    ev = CUDAFullJacobianConstantPEvaluatorPT(mech)
    J  = ev.evaluate(T_arr, C_arr)
    # T_arr : (B,)       float64
    # C_arr : (B, n_sp)  float64
    # J     : (B, n_state, n_state) float64,  n_state = n_sp + 1
    """

    THREADS = 64

    def __init__(self, mech, dtype=np.float64):
        self._mech    = mech
        self._dtype   = dtype
        self._n_sp    = mech.n_species
        self._n_state = self._n_sp + 1
        self._ctx     = None
        self._B_alloc = -1

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
        n_f_flat  = f.react_offset[-1]  if f.n_reactions  else 1

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
            sRrf   = z((B, n_f_flat)),         sRpf   = z((B, n_f_flat)),
            k0_f   = z((B, f.n_reactions)),    kinf_f = z((B, f.n_reactions)),
            M_f    = z((B, f.n_reactions)),    Pr_f   = z((B, f.n_reactions)),
            F_f    = z((B, f.n_reactions)),    kb_f   = z((B, f.n_reactions)),
            Fc_f   = z((B, f.n_reactions)),    lFc_f  = z((B, f.n_reactions)),
            Cp_f   = z((B, f.n_reactions)),    Np_f   = z((B, f.n_reactions)),
            f1_f   = z((B, f.n_reactions)),    exp_f  = z((B, f.n_reactions)),
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
        B     = T_arr.shape[0]
        nst   = self._n_state

        if self._B_alloc != B:
            self._alloc_context(B)

        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0]  = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)
        d_J  = cuda.device_array((B, nst * nst), dtype=self._dtype)

        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff
        cx = self._ctx

        threads = self.THREADS
        blocks  = (B + threads - 1) // threads

        _full_jacobian_constantP_kernel_pt[blocks, threads](
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
#  4.  Wrapper TPY — réutilise exactement CUDAFullJacobianConstantPTPY
# ─────────────────────────────────────────────────────────────────────────────

import torch
import kineticsGPU.torch.thermo as thermo
import kineticsGPU.torch.reactionParser as reactionParser
from kineticsGPU.cuda.mechanism.mechanism_data import build_mechanism_data_from_torch

# Constante universelle des gaz (J/mol/K)


class CUDAFullJacobianConstantPTPY_PT:
    """
    Jacobienne complète du réacteur isobare dans l'espace (T, P, Y)
    version per-thread (PT).

    Concordance totale avec la version PB :
    - Calculs intermédiaires via PyTorch (GPU/CPU selon config)
    - Entrées/Sorties identiques via Tensors
    """

    def __init__(self, mech, torchComputationDtype=None, dataDevice=None, computationDevice=None, storageDtype=None):
        if torchComputationDtype == torch.float64:
            computationDtype = np.float64
        elif torchComputationDtype == torch.float32:
            computationDtype = np.float32
        else:
            raise NotImplementedError("Seul float32 et float64 sont supportés.")

        self.torchComputationDtype = torchComputationDtype
        self.storageDtype = storageDtype
        self.dataDevice = dataDevice
        self.computationDevice = torch.device(computationDevice)

        # Initialisation du mécanisme identique à la version PB
        y = thermo.yamlParser(mech)
        vec_th = y.populateVectorizedThermoObject(dtype=torchComputationDtype, device=dataDevice)
        reactions = reactionParser.parseReactions(mech)
        r_e, r_tb, r_f, system = reactionParser.build_reactions(
            vec_th.species, reactions, vec_th, dataDevice, torchComputationDtype
        )
        mech_data = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=computationDtype)

        # Moteur Per-Thread
        self._ev = CUDAFullJacobianConstantPEvaluatorPT(mech_data, dtype=computationDtype)

        self._dtype = computationDtype
        self._W = vec_th.molar_masses.to(torchComputationDtype).to(self.computationDevice)

    def _encode(self, T, P, Y):
        # sYW = 1 / W_mean
        sYW = (Y / self._W[None, :]).sum(dim=1)
        rho = P / (_R * T * sYW)
        C = rho[:, None] * Y / self._W[None, :]
        TC = torch.cat([T[:, None], C], dim=1)
        return TC, rho

    def _dTC_dTPY(self, T, P, Y, rho):
        B = T.shape[0]
        ns = Y.shape[1]
        nst = ns + 1  # dim TC (T + ns espèces)
        nst1 = ns + 2  # dim TPY (T + P + ns espèces)

        dTC = torch.zeros((B, nst, nst1), dtype=self.torchComputationDtype, device=self.computationDevice)

        # dT/dT = 1
        dTC[:, 0, 0] = 1.0

        sYW = (Y / self._W[None, :]).sum(dim=1)
        C = rho[:, None] * Y / self._W[None, :]

        # ∂C_k/∂T = -C_k/T
        dTC[:, 1:, 0] = -C / T[:, None]
        # ∂C_k/∂P = C_k/P
        dTC[:, 1:, 1] = C / P[:, None]

        # ∂C_k/∂Y_l = ρ/W_k·δ_{kl} - C_k/(W_l · sYW)
        cross_term = -C[:, :, None] / (self._W[None, None, :] * sYW[:, None, None])
        dTC[:, 1:, 2:] = cross_term

        diag_term = rho[:, None] / self._W[None, :]
        for k in range(ns):
            dTC[:, k + 1, k + 2] += diag_term[:, k]

        return dTC

    def _dRHSTPY_dRHSTC(self, T, C, rho):
        B = T.shape[0]
        ns = C.shape[1]
        nst = ns + 1
        nst1 = ns + 2

        M = torch.zeros((B, nst1, nst), dtype=self.torchComputationDtype, device=self.computationDevice)

        # dT/dt -> dT/dt
        M[:, 0, 0] = 1.0

        # dY_k/dt via concentrations
        Y = C * self._W[None, :] / rho[:, None]

        # Terme : -Y_k * W_l / rho
        M[:, 2:, 1:] = -Y[:, :, None] * self._W[None, None, :] / rho[:, None, None]

        # Terme diagonal : +W_k / rho
        for k in range(ns):
            M[:, k + 2, k + 1] += self._W[k] / rho

        return M

    def evaluate(self, T_arr, P_arr, Y_arr):
        """
        Même interface que la version PB.
        """
        # Transfert et conversion
        T_arr = T_arr.to(self.torchComputationDtype).to(self.computationDevice)
        P_arr = P_arr.to(self.torchComputationDtype).to(self.computationDevice)
        Y_arr = Y_arr.to(self.torchComputationDtype).to(self.computationDevice)

        # Normalisation
        Y_arr = Y_arr / Y_arr.sum(dim=1, keepdims=True)

        # Encodage (T, P, Y) -> (T, C)
        TC, rho = self._encode(T_arr, P_arr, Y_arr)

        # Appel au kernel CUDA Per-Thread
        # On utilise cuda.as_cuda_array pour ne pas copier les données si elles sont déjà sur GPU
        J_TC_raw = self._ev.evaluate(
            cuda.as_cuda_array(TC[:, 0].detach()),
            cuda.as_cuda_array(TC[:, 1:].detach())
        )

        # Re-encapsulation dans torch
        J_TC = torch.as_tensor(J_TC_raw, device=self.computationDevice, dtype=self.torchComputationDtype)

        # Matrices de passage analytiques (Torch)
        dTC_dTPY = self._dTC_dTPY(T_arr, P_arr, Y_arr, rho)
        M_out = self._dRHSTPY_dRHSTC(T_arr, TC[:, 1:], rho)

        # Calcul final : J_TPY = M_out @ J_TC @ dTC_dTPY
        J_TPY = M_out @ J_TC @ dTC_dTPY

        return J_TPY.to(self.dataDevice).to(self.storageDtype)


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Fonction device scalaire
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _constantV_full_jacobian(
        T, C, omega, J_chem,
        n_sp, thL, thH, thM,
        J_full,
):
    """
    Jacobienne complète du réacteur isochore adiabatique (scalaire).

    J_full[i * n_state + j] = d(RHS_i) / d(xk_j)
    xk  = (T, C_0, …, C_{n_sp-1})
    RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)

    Paramètres
    ----------
    T       : float64              — température (K)
    C       : array(n_sp)          — concentrations (mol/m³)
    omega   : array(n_sp)          — taux de production au point (T, C)
    J_chem  : array(n_sp*n_state)  — d(omega_s)/d(xk_j), row-major
    n_sp    : int
    thL/H/M : tableaux NASA-7
    J_full  : array(n_state*n_state) — sortie row-major

    Dérivation — ligne énergie (i=0)
    ---------------------------------
    dT/dt = -U_num / D
        U_num = Σ_s u_s·ω_s
        D     = n_tot·cv_mix

    d(dT/dt)/d(xk_j) = -(dU_j·D - U_num·dD_j) / D²

    dU_j :
        j = 0    → Σ_s cv_s·ω_s  +  Σ_s u_s·J_chem[s,0]
        j = k+1  → Σ_s u_s·J_chem[s,j]

    dD_j :
        j = 0    → n_tot · Σ_s X_s·(d(cv_s)/dT)
        j = k+1  → cv_k

    Dérivation — lignes espèces (i=s+1)
    ------------------------------------
    dC_s/dt = ω_s   (volume constant, pas de dilatation)
    d(dC_s/dt)/d(xk_j) = J_chem[s, j]   (copie directe)
    """
    n_state = n_sp + 1
    c7 = cuda.local.array(7, float64)

    # ── Passe 1 : scalaires thermodynamiques ─────────────────────────────────

    ntot = float64(0.0)
    for s in range(n_sp):
        ntot += C[s]

    cv_mix = float64(0.0)
    U_num = float64(0.0)
    dcv_dT = float64(0.0)

    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
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
        Xs = C[s] / ntot if ntot > EPS else float64(0.0)

        cv_mix += Xs * cv_s
        U_num += u_s * omega[s]
        dcv_dT += Xs * dcvs_dT

    D = ntot * cv_mix
    D_safe = D if abs(D) > EPS else (EPS if D >= float64(0.0) else -EPS)
    D_sq = D_safe * D_safe

    # ── Passe 2 : ligne i=0 — d(dT/dt)/d(xk_j) ──────────────────────────────

    for j in range(n_state):
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

    # ── Passe 3 : lignes espèces — copie directe depuis J_chem ───────────────

    for s in range(n_sp):
        for j in range(n_state):
            J_full[(s + 1) * n_state + j] = J_chem[s * n_state + j]


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Kernel CUDA global
# ─────────────────────────────────────────────────────────────────────────────

MAX_N_STATE = 54
MAX_N_STATE2 = MAX_N_STATE * MAX_N_STATE


@cuda.jit(cache=True)
def _full_jacobian_constantV_kernel_pt(
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
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1
    T_loc = TC_in[b, 0]

    C = cuda.local.array(MAX_N_STATE, float64)
    om = cuda.local.array(MAX_N_STATE, float64)
    J_chem = cuda.local.array(MAX_N_STATE2, float64)
    J_buf = cuda.local.array(MAX_N_STATE2, float64)

    for s in range(n_sp):
        C[s] = TC_in[b, s + 1]

    # Étape 1 : contexte cinétique + omega
    _eval_rhs_and_context(
        T_loc, C,
        ctx_kf_e[b], ctx_Kc_e[b], ctx_cr_e[b], ctx_cp_e[b], ctx_dG0_e[b],
        ctx_sRre[b], ctx_sRpe[b],
        ctx_kf_tb[b], ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
        ctx_sRrtb[b], ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
        ctx_kf_f[b], ctx_Kc_f[b], ctx_cr_f[b], ctx_cp_f[b], ctx_dG0_f[b],
        ctx_sRrf[b], ctx_sRpf[b],
        ctx_k0_f[b], ctx_kinf_f[b], ctx_M_f[b], ctx_Pr_f[b], ctx_F_f[b],
        ctx_kb_f[b],
        ctx_Fc_f[b], ctx_lFc_f[b], ctx_Cp_f[b], ctx_Np_f[b],
        ctx_f1_f[b], ctx_exp_f[b],
        om, n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # Étape 2 : jacobienne chimique d(omega)/d(xk) → J_chem (n_sp × n_state)
    _eval_jacobian(
        T_loc, C,
        J_chem,
        ctx_kf_e[b], ctx_Kc_e[b], ctx_cr_e[b], ctx_cp_e[b], ctx_dG0_e[b],
        ctx_sRre[b], ctx_sRpe[b],
        ctx_kf_tb[b], ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
        ctx_sRrtb[b], ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
        ctx_kf_f[b], ctx_Kc_f[b], ctx_cr_f[b], ctx_cp_f[b], ctx_dG0_f[b],
        ctx_sRrf[b], ctx_sRpf[b],
        ctx_k0_f[b], ctx_kinf_f[b], ctx_M_f[b], ctx_Pr_f[b], ctx_F_f[b],
        ctx_kb_f[b],
        ctx_Fc_f[b], ctx_lFc_f[b], ctx_Cp_f[b], ctx_Np_f[b],
        ctx_f1_f[b], ctx_exp_f[b],
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # Étape 3 : jacobienne complète isochore → J_buf (local)
    _constantV_full_jacobian(
        T_loc, C, om, J_chem,
        n_sp, thL, thH, thM,
        J_buf,
    )

    # Copie J_buf → J_all[b] (mémoire globale)
    for idx in range(n_state * n_state):
        J_all[b, idx] = J_buf[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Wrapper Python : entrées/sorties en TC
# ─────────────────────────────────────────────────────────────────────────────

class CUDAFullJacobianConstantVEvaluatorPT:
    """
    Jacobienne complète du réacteur isochore adiabatique (per-thread),
    coordonnées TC.

    J[i, j] = d(RHS_i) / d(xk_j)

    RHS = (dT/dt, dC_0/dt, …, dC_{n_sp-1}/dt)  — réacteur V=cte
    xk  = (T, C_0, …, C_{n_sp-1})

    Usage
    -----
    ev = CUDAFullJacobianConstantVEvaluatorPT(mech)
    J  = ev.evaluate(T_arr, C_arr)
    # T_arr : (B,)       float64
    # C_arr : (B, n_sp)  float64
    # J     : (B, n_state, n_state) float64,  n_state = n_sp + 1
    """

    THREADS = 64

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
        n_f_flat = f.react_offset[-1] if f.n_reactions else 1

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
            sRrf=z((B, n_f_flat)), sRpf=z((B, n_f_flat)),
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

        threads = self.THREADS
        blocks = (B + threads - 1) // threads

        _full_jacobian_constantV_kernel_pt[blocks, threads](
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
#  4.  Wrapper TPY
# ─────────────────────────────────────────────────────────────────────────────

_R = 8.314462618  # J/mol/K


class CUDAFullJacobianConstantVTPY_PT:
    """
    Jacobienne complète du réacteur isochore dans l'espace (T, P, Y) — per-thread.

    Interface identique à CUDAFullJacobianConstantVTPY (per-block) ;
    seul le moteur GPU sous-jacent diffère.

    Préférer la version per-block pour les petits batches (B ≤ quelques centaines),
    et la version per-thread pour les grands batches.

    Usage
    -----
    ev   = CUDAFullJacobianConstantVTPY_PT(
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

        self._ev = CUDAFullJacobianConstantVEvaluatorPT(mech_data, dtype=computationDtype)
        self._dtype = computationDtype
        self._W = vec_th.molar_masses.to(torchComputationDtype).to(self.computationDevice)

    # ── Encodage ψ : (T, P, Y) → (T, C) ────────────────────────────────────

    def _encode(self, T, P, Y):
        sYW = (Y / self._W[None, :]).sum(dim=1)
        rho = P / (_R * T * sYW)
        C = rho[:, None] * Y / self._W[None, :]
        TC = torch.cat([T[:, None], C], dim=1)
        return TC, rho

    # ── ∂TC/∂TPY : (B, n_state, n_state+1) ─────────────────────────────────

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

        sYW = (Y / self._W[None, :]).sum(dim=1)
        C = rho[:, None] * Y / self._W[None, :]

        dTC[:, 1:, 0] = -C / T[:, None]
        dTC[:, 1:, 1] = C / P[:, None]

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

    def _dRHSTPY_dRHSTC(self, T, C, rho):
        B = T.shape[0]
        ns = C.shape[1]
        nst = ns + 1
        nst1 = ns + 2

        M = torch.zeros(
            (B, nst1, nst),
            dtype=self.torchComputationDtype, device=self.computationDevice
        )

        M[:, 0, 0] = 1.0

        n_tot = C.sum(dim=1)
        M[:, 1, 0] = _R * n_tot
        M[:, 1, 1:] = _R * T[:, None]

        for k in range(ns):
            M[:, k + 2, k + 1] = self._W[k] / rho

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


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Kernel global : TC_in → dTCdt_out
# ─────────────────────────────────────────────────────────────────────────────
_R            = 8.314462618
_MAX_N_STATE  = 54
_THREADS      = 64
@cuda.jit(cache=True)
def _constantV_rhs_kernel_pt(
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
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= TC_in.shape[0]:
        return

    T_loc = TC_in[b, 0]
    C = cuda.local.array(_MAX_N_STATE, float64)
    omega = cuda.local.array(_MAX_N_STATE, float64)
    dTCdt = cuda.local.array(_MAX_N_STATE, float64)

    for s in range(n_sp):
        C[s] = TC_in[b, s + 1]

    # Étape 1 : omega = dC/dt
    _eval_rhs_and_context(
        T_loc, C,
        ctx_kf_e[b], ctx_Kc_e[b], ctx_cr_e[b], ctx_cp_e[b], ctx_dG0_e[b],
        ctx_sRre[b], ctx_sRpe[b],
        ctx_kf_tb[b], ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
        ctx_sRrtb[b], ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
        ctx_kf_f[b], ctx_Kc_f[b], ctx_cr_f[b], ctx_cp_f[b], ctx_dG0_f[b],
        ctx_sRrf[b], ctx_sRpf[b],
        ctx_k0_f[b], ctx_kinf_f[b], ctx_M_f[b], ctx_Pr_f[b], ctx_F_f[b],
        ctx_kb_f[b],
        ctx_Fc_f[b], ctx_lFc_f[b], ctx_Cp_f[b], ctx_Np_f[b],
        ctx_f1_f[b], ctx_exp_f[b],
        omega, n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # Étape 2 : dTCdt = (dT/dt, omega)  via _adiabatic_rhs
    _adiabatic_rhs(T_loc, C, omega, n_sp, thL, thH, thM, dTCdt)

    # Écriture globale
    n_state = n_sp + 1
    for i in range(n_state):
        dTCdt_out[b, i] = dTCdt[i]


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Évaluateur bas niveau TC → dTCdt
# ─────────────────────────────────────────────────────────────────────────────

class CUDAConstantVRHSEvaluatorPT:
    """
    Évalue le RHS isochore complet (dT/dt, dC/dt) dans l'espace TC.

    Usage
    -----
    ev    = CUDAConstantVRHSEvaluatorPT(mech)
    dTCdt = ev.evaluate(T_arr, C_arr)   # np.ndarray (B, n_state)
    """

    THREADS = _THREADS

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
        n_f_flat = f.react_offset[-1] if f.n_reactions else 1

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
            sRrf=z((B, n_f_flat)), sRpf=z((B, n_f_flat)),
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

        threads = self.THREADS
        blocks = (B + threads - 1) // threads

        _constantV_rhs_kernel_pt[blocks, threads](
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

class CUDAConstantVRHSTPY_PT:
    """
    RHS du réacteur isochore adiabatique dans l'espace (T, P, Y) — per-thread.

    Usage
    -----
    ev   = CUDAConstantVRHSTPY_PT(
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

        self._ev = CUDAConstantVRHSEvaluatorPT(mech_data, dtype=computationDtype)
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


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Kernel global : TC_in → dTCdt_out
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(cache=True)
def _constantP_rhs_kernel_pt(
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
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= TC_in.shape[0]:
        return

    T_loc = TC_in[b, 0]
    C = cuda.local.array(_MAX_N_STATE, float64)
    omega = cuda.local.array(_MAX_N_STATE, float64)
    dTCdt = cuda.local.array(_MAX_N_STATE, float64)

    for s in range(n_sp):
        C[s] = TC_in[b, s + 1]

    # Étape 1 : omega = dC/dt chimique
    _eval_rhs_and_context(
        T_loc, C,
        ctx_kf_e[b], ctx_Kc_e[b], ctx_cr_e[b], ctx_cp_e[b], ctx_dG0_e[b],
        ctx_sRre[b], ctx_sRpe[b],
        ctx_kf_tb[b], ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
        ctx_sRrtb[b], ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
        ctx_kf_f[b], ctx_Kc_f[b], ctx_cr_f[b], ctx_cp_f[b], ctx_dG0_f[b],
        ctx_sRrf[b], ctx_sRpf[b],
        ctx_k0_f[b], ctx_kinf_f[b], ctx_M_f[b], ctx_Pr_f[b], ctx_F_f[b],
        ctx_kb_f[b],
        ctx_Fc_f[b], ctx_lFc_f[b], ctx_Cp_f[b], ctx_Np_f[b],
        ctx_f1_f[b], ctx_exp_f[b],
        omega, n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # Étape 2 : RHS isobare complet (dT/dt, dC/dt avec dilatation)
    _constantP_rhs(T_loc, C, omega, n_sp, thL, thH, thM, dTCdt)

    # Écriture globale
    n_state = n_sp + 1
    for i in range(n_state):
        dTCdt_out[b, i] = dTCdt[i]


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Évaluateur bas niveau TC → dTCdt
# ─────────────────────────────────────────────────────────────────────────────

class CUDAConstantPRHSEvaluatorPT:
    """
    Évalue le RHS isobare complet (dT/dt, dC/dt) dans l'espace TC — per-thread.

    Usage
    -----
    ev    = CUDAConstantPRHSEvaluatorPT(mech)
    dTCdt = ev.evaluate(T_arr, C_arr)   # np.ndarray (B, n_state)
    """

    THREADS = _THREADS

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
        n_f_flat = f.react_offset[-1] if f.n_reactions else 1

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
            sRrf=z((B, n_f_flat)), sRpf=z((B, n_f_flat)),
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

        threads = self.THREADS
        blocks = (B + threads - 1) // threads

        _constantP_rhs_kernel_pt[blocks, threads](
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

class CUDAConstantPRHSTPY_PT:
    """
    RHS du réacteur isobare dans l'espace (T, P, Y) — per-thread.

    Usage
    -----
    ev   = CUDAConstantPRHSTPY_PT(
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

        self._ev = CUDAConstantPRHSEvaluatorPT(mech_data, dtype=computationDtype)
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
        dY_k/dt = (W_k·dC_k - Y_k·dρ) / ρ
                  avec dρ = Σ W_l·dC_l  (ρ = Σ C_l·W_l)
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