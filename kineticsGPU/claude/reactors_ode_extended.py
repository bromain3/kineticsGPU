"""
kernels/shared/reactor_odes_extended.py — Dérivées temporelles de omega et Tdot.

Calcule d/dt[omega_s] et d²T/dt² le long de la trajectoire du réacteur
adiabatique à volume constant, en exploitant les JVP existantes.

Principe mathématique
---------------------
Le RHS du réacteur est :
    dT/dt  = f_T(T, C)  = -sum(u_s * omega_s) / (n_tot * cv_mix)
    dC_s/dt = f_s(T, C) = omega_s(T, C)

La dérivée temporelle de ces quantités le long de la trajectoire est
leur JVP dans la direction tangente (f_T, f_s) :

    d(omega_s)/dt = JVP_omega(T, C; vT=f_T, vC=f_s)
    d(Tdot)/dt    = JVP_{f_T}(T, C; vT=f_T, vC=f_s)

Ce module expose :

Scalaire (per-thread)
    _omega_tdot_rates(...)      — calcule (omega, dTCdt, Jv_om, Jv_TC)
    _domegadt_norm_ratio(...)   — retourne ||domega/dt|| / ||omega||

Per-block (parallèle)
    _omega_tdot_rates_pb(...)   — idem, avec stride tid/n_threads

Wrappers noyaux CUDA haut-niveau
    kernel_omega_rates          — lance un noyau sur un batch de réacteurs

Dépendances attendues dans le même package :
    kernels/shared/thermo.py
    kernels/shared/reactor_odes.py      (_adiabatic_rhs, _adiabatic_jvp)
    kernels/per_thread/context.py       (_eval_rhs_and_context)
    kernels/per_thread/jvp.py           (_eval_jvp)         [à adapter selon votre API]
    kernels/per_block/reactor_odes_pb.py (_adiabatic_rhs_pb, _adiabatic_jvp_pb)
    kernels/per_block/context_pb.py     (_eval_rhs_and_context_pb)
    kernels/per_block/jvp_pb.py         (_eval_jvp_pb)      [à adapter selon votre API]
"""

import math
import numpy as np
from numba import cuda, float64

from .thermo import R_CONST, EPS
from .reactor_odes import _adiabatic_rhs, _adiabatic_jvp


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION SCALAIRE (per-thread)
# ═══════════════════════════════════════════════════════════════════════════════

@cuda.jit(device=True)
def _omega_tdot_rates(
    T, C,
    # ── Contexte cinétique (vues 1D ligne b) ─────────────────────────
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    # ── Tampons locaux (tableaux locaux de longueur n_sp) ────────────
    omega,    # taux de production [mol/m³/s]      — longueur n_sp
    dTCdt,    # RHS du réacteur [dT/dt, dC/dt]     — longueur n_sp+1
    Jv_om,    # JVP de omega dans dir. tangente     — longueur n_sp
    Jv_TC,    # JVP du RHS = (d²T/dt², domega/dt)  — longueur n_sp+1
    # ── Données mécanisme ─────────────────────────────────────────────
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    # ── Fonction externe JVP de omega ────────────────────────────────
    # Note : _eval_jvp doit être importée et passée comme pointeur de
    # fonction device, ou appelée directement si dans le même module.
    # Ici on suppose qu'elle est inlinable depuis per_thread/jvp.py.
    _eval_jvp_fn,
):
    """
    Calcule en une passe :
        omega   = omega(T, C)                         [mol/m³/s]
        dTCdt   = (dT/dt, dC/dt)                      [K/s, mol/m³/s]
        Jv_om   = d(omega)/dt  (JVP omega, dir. tangente)
        Jv_TC   = (d²T/dt², d(omega)/dt)

    Tous les tampons (omega, dTCdt, Jv_om, Jv_TC) doivent être pré-alloués
    par l'appelant (cuda.local.array ou mémoire partagée).

    Après l'appel :
        Jv_TC[0]        → d²T/dt²                [K/s²]
        Jv_TC[1:n_sp+1] → domega_s/dt            [mol/m³/s²]
    """
    from .per_thread.context import _eval_rhs_and_context  # import device

    # ── Étape 1 : RHS cinétique → omega + cache cinétique ────────────
    _eval_rhs_and_context(
        T, C,
        kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
        kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
        kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
        k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
        Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
        omega,
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # ── Étape 2 : RHS du réacteur → direction tangente (vT, vC) ──────
    _adiabatic_rhs(T, C, omega, n_sp, thL, thH, thM, dTCdt)
    # dTCdt[0]   = vT  = dT/dt
    # dTCdt[1:] = vC_s = dC_s/dt = omega_s

    # ── Étape 3 : JVP de omega dans la direction (vT, vC) ─────────────
    # vT = dTCdt[0],  vC = omega (= dTCdt[1:])
    _eval_jvp_fn(
        T, C, dTCdt[0], omega,       # (T, C, vT, vC)
        kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
        kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
        kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
        k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
        Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
        Jv_om,
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # ── Étape 4 : JVP du RHS adiabatique → (d²T/dt², domega/dt) ──────
    _adiabatic_jvp(
        T, C,
        dTCdt[0], omega,    # direction tangente (vT, vC)
        omega, Jv_om,       # (omega au point de base, JVP de omega)
        n_sp, thL, thH, thM,
        Jv_TC,
    )
    # Résultats :
    #   Jv_TC[0]        = d²T/dt²        [K/s²]
    #   Jv_TC[s+1]      = domega_s/dt    [mol/m³/s²]


@cuda.jit(device=True)
def _domegadt_norm_ratio(Jv_TC, omega, n_sp):
    """
    Retourne ||domega/dt|| / ||omega|| (normes L2).

    Jv_TC[s+1] = domega_s/dt,  omega[s] = omega_s.
    Retourne 0.0 si ||omega|| < EPS.
    """
    num = float64(0.0)
    den = float64(0.0)
    for s in range(n_sp):
        dom = Jv_TC[s + 1]
        om  = omega[s]
        num += dom * dom
        den += om  * om
    if den < EPS:
        return float64(0.0)
    return math.sqrt(num / den)


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION PARALLÈLE (per-block)
# ═══════════════════════════════════════════════════════════════════════════════

@cuda.jit(device=True)
def _omega_tdot_rates_pb(
    tid, n_threads,
    T, C,
    # ── Contexte cinétique (vues 1D ligne b) ─────────────────────────
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    # ── Tampons en mémoire partagée ───────────────────────────────────
    omega,    # longueur n_sp   — mis à zéro par _eval_rhs_and_context_pb
    dTCdt,    # longueur n_sp+1
    Jv_om,    # longueur n_sp
    Jv_TC,    # longueur n_sp+1
    scratch,  # longueur >= n_threads (pour réductions)
    # ── Données mécanisme ─────────────────────────────────────────────
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    _eval_rhs_and_context_pb_fn,
    _eval_jvp_pb_fn,
):
    """
    Version per-block de _omega_tdot_rates.

    Tous les tampons (omega, dTCdt, Jv_om, Jv_TC, scratch) doivent être
    en mémoire partagée. Les syncthreads nécessaires sont effectués en interne.

    Après l'appel (visible par tous les threads du bloc) :
        Jv_TC[0]        → d²T/dt²
        Jv_TC[1:n_sp+1] → domega_s/dt
    """
    from .per_block.context_pb import _eval_rhs_and_context_pb  # device import

    # ── Étape 1 ───────────────────────────────────────────────────────
    _eval_rhs_and_context_pb_fn(
        tid, n_threads, T, C,
        kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
        kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
        kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
        k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
        Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
        omega,
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )
    # omega est prêt + syncthreads déjà fait en fin de _eval_rhs_and_context_pb

    # ── Étape 2 ───────────────────────────────────────────────────────
    from .per_block.reactor_odes_pb import _adiabatic_rhs_pb, _adiabatic_jvp_pb
    _adiabatic_rhs_pb(
        tid, n_threads,
        T, C, omega, n_sp, thL, thH, thM,
        dTCdt, scratch,
    )
    # dTCdt[0] = vT, dTCdt[1:] = vC ; syncthreads fait en interne

    # ── Étape 3 ───────────────────────────────────────────────────────
    # Mise à zéro de Jv_om en parallèle avant la JVP cinétique
    i = tid
    while i < n_sp:
        Jv_om[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    _eval_jvp_pb_fn(
        tid, n_threads,
        T, C, dTCdt[0], omega,   # vT = dTCdt[0], vC = omega
        kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
        kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
        kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
        k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
        Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
        Jv_om,
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )
    # Jv_om prêt, syncthreads fait en interne

    # ── Étape 4 ───────────────────────────────────────────────────────
    _adiabatic_jvp_pb(
        tid, n_threads,
        T, C,
        dTCdt[0], omega,   # direction tangente
        omega, Jv_om,      # (omega base, JVP omega)
        n_sp, thL, thH, thM,
        Jv_TC, scratch,
    )
    # Jv_TC[0]    = d²T/dt²
    # Jv_TC[s+1]  = domega_s/dt


# ═══════════════════════════════════════════════════════════════════════════════
# NOYAU CUDA HAUT-NIVEAU — wrapper host-side
# ═══════════════════════════════════════════════════════════════════════════════

@cuda.jit
def _kernel_omega_rates_pt(
    T_batch, C_batch,
    # Tableaux de contexte (B, ...)
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    # Sorties (B, ...)
    out_omega,    # (B, n_sp)
    out_dTCdt,    # (B, n_sp+1)
    out_Jv_om,    # (B, n_sp)
    out_Jv_TC,    # (B, n_sp+1)
    out_ratio,    # (B,)
    # Mécanisme
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    """
    Noyau per-thread : 1 thread = 1 réacteur.

    Grille recommandée : <<<B, 1>>> ou <<<ceil(B/TPB), TPB>>>.
    """
    b = cuda.grid(1)
    B = T_batch.shape[0]
    if b >= B:
        return

    T = T_batch[b]
    C = C_batch[b]

    omega  = cuda.local.array(256, float64)   # adapter à n_sp max
    dTCdt  = cuda.local.array(257, float64)
    Jv_om  = cuda.local.array(256, float64)
    Jv_TC  = cuda.local.array(257, float64)

    # ── Étape 1 : omega + contexte ────────────────────────────────────
    from kernels.per_thread.context import _eval_rhs_and_context
    _eval_rhs_and_context(
        T, C,
        kf_e[b], Kc_e[b], cr_e[b], cp_e[b], dG0_e[b], sR_re[b], sR_pe[b],
        kf_tb[b], Kc_tb[b], cr_tb[b], cp_tb[b], dG0_tb[b],
        sR_rtb[b], sR_ptb[b], bRR_tb[b], M_tb[b],
        kf_f[b], Kc_f[b], cr_f[b], cp_f[b], dG0_f[b], sR_rf[b], sR_pf[b],
        k0_f[b], kinf_f[b], M_f[b], Pr_f[b], F_f[b], kb_f[b],
        Fcent_f[b], lFcent_f[b], Cpar_f[b], Npar_f[b], f1_f[b], exp_f[b],
        omega,
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # ── Étape 2 : RHS du réacteur ─────────────────────────────────────
    _adiabatic_rhs(T, C, omega, n_sp, thL, thH, thM, dTCdt)

    # ── Étape 3 : JVP de omega dans la direction tangente ─────────────
    from kernels.per_thread.jvp import _eval_jvp
    _eval_jvp(
        T, C, dTCdt[0], omega,
        kf_e[b], Kc_e[b], cr_e[b], cp_e[b], dG0_e[b], sR_re[b], sR_pe[b],
        kf_tb[b], Kc_tb[b], cr_tb[b], cp_tb[b], dG0_tb[b],
        sR_rtb[b], sR_ptb[b], bRR_tb[b], M_tb[b],
        kf_f[b], Kc_f[b], cr_f[b], cp_f[b], dG0_f[b], sR_rf[b], sR_pf[b],
        k0_f[b], kinf_f[b], M_f[b], Pr_f[b], F_f[b], kb_f[b],
        Fcent_f[b], lFcent_f[b], Cpar_f[b], Npar_f[b], f1_f[b], exp_f[b],
        Jv_om,
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT,
        fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # ── Étape 4 : JVP du RHS complet ─────────────────────────────────
    _adiabatic_jvp(T, C, dTCdt[0], omega, omega, Jv_om,
                   n_sp, thL, thH, thM, Jv_TC)

    # ── Écriture des sorties ──────────────────────────────────────────
    num = float64(0.0)
    den = float64(0.0)
    for s in range(n_sp):
        out_omega[b, s]  = omega[s]
        out_dTCdt[b, s + 1] = dTCdt[s + 1]
        out_Jv_om[b, s]  = Jv_om[s]
        out_Jv_TC[b, s + 1] = Jv_TC[s + 1]
        num += Jv_TC[s + 1] ** 2
        den += omega[s] ** 2
    out_dTCdt[b, 0] = dTCdt[0]
    out_Jv_TC[b, 0] = Jv_TC[0]
    out_ratio[b] = math.sqrt(num / den) if den > EPS else float64(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE HOST (NumPy / Cantera-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class OmegaRatesResult:
    """
    Résultat retourné par compute_omega_rates_host.

    Attributs
    ---------
    omega      : (B, n_sp)   taux de production [mol/m³/s]
    dTdt       : (B,)        dT/dt              [K/s]
    domegadt   : (B, n_sp)   d(omega)/dt        [mol/m³/s²]
    d2Tdt2     : (B,)        d²T/dt²            [K/s²]
    ratio      : (B,)        ||domega/dt|| / ||omega||   [1/s]
    """
    def __init__(self, omega, dTCdt, domegadt, Jv_TC, ratio):
        self.omega    = omega
        self.dTdt     = dTCdt[:, 0]
        self.dCdt     = dTCdt[:, 1:]
        self.domegadt = domegadt
        self.d2Tdt2   = Jv_TC[:, 0]
        self.ratio    = ratio


def compute_omega_rates_host(
    T_np, C_np,
    ctx_arrays,   # dict contenant tous les tableaux de contexte (B, ...) sur GPU
    mech,         # objet mécanisme exposant les tableaux numpy nécessaires
    n_sp,
    threads_per_block=128,
):
    """
    Wrapper host-side pour _kernel_omega_rates_pt.

    Paramètres
    ----------
    T_np       : ndarray (B,)      températures [K]
    C_np       : ndarray (B, n_sp) concentrations [mol/m³]
    ctx_arrays : dict              tableaux de contexte cinétique sur device
    mech       : objet mécanisme   expose thL, thH, thM, eA, en, ... etc.
    n_sp       : int               nombre d'espèces

    Retourne
    --------
    OmegaRatesResult
    """
    B = T_np.shape[0]

    T_d  = cuda.to_device(T_np.astype(np.float64))
    C_d  = cuda.to_device(C_np.astype(np.float64))

    out_omega  = cuda.device_array((B, n_sp),   dtype=np.float64)
    out_dTCdt  = cuda.device_array((B, n_sp+1), dtype=np.float64)
    out_Jv_om  = cuda.device_array((B, n_sp),   dtype=np.float64)
    out_Jv_TC  = cuda.device_array((B, n_sp+1), dtype=np.float64)
    out_ratio  = cuda.device_array((B,),         dtype=np.float64)

    blocks = (B + threads_per_block - 1) // threads_per_block

    _kernel_omega_rates_pt[blocks, threads_per_block](
        T_d, C_d,
        ctx_arrays['kf_e'],   ctx_arrays['Kc_e'],   ctx_arrays['cr_e'],
        ctx_arrays['cp_e'],   ctx_arrays['dG0_e'],  ctx_arrays['sR_re'],
        ctx_arrays['sR_pe'],
        ctx_arrays['kf_tb'],  ctx_arrays['Kc_tb'],  ctx_arrays['cr_tb'],
        ctx_arrays['cp_tb'],  ctx_arrays['dG0_tb'], ctx_arrays['sR_rtb'],
        ctx_arrays['sR_ptb'], ctx_arrays['bRR_tb'], ctx_arrays['M_tb'],
        ctx_arrays['kf_f'],   ctx_arrays['Kc_f'],   ctx_arrays['cr_f'],
        ctx_arrays['cp_f'],   ctx_arrays['dG0_f'],  ctx_arrays['sR_rf'],
        ctx_arrays['sR_pf'],  ctx_arrays['k0_f'],   ctx_arrays['kinf_f'],
        ctx_arrays['M_f'],    ctx_arrays['Pr_f'],   ctx_arrays['F_f'],
        ctx_arrays['kb_f'],   ctx_arrays['Fcent_f'],ctx_arrays['lFcent_f'],
        ctx_arrays['Cpar_f'], ctx_arrays['Npar_f'], ctx_arrays['f1_f'],
        ctx_arrays['exp_f'],
        out_omega, out_dTCdt, out_Jv_om, out_Jv_TC, out_ratio,
        n_sp,
        mech.thL, mech.thH, mech.thM,
        mech.n_e,  mech.eA,  mech.en,  mech.eEa,  mech.eOW,  mech.eDn,
        mech.eRI,  mech.eRS, mech.eRO, mech.ePI,  mech.ePS,  mech.ePO,
        mech.n_tb, mech.tbA, mech.tbn, mech.tbEa, mech.tbOW, mech.tbDn,
        mech.tbRI, mech.tbRS,mech.tbRO,mech.tbPI, mech.tbPS, mech.tbPO,
        mech.tbEff,
        mech.n_f,  mech.fA0, mech.fb0, mech.fE0,  mech.fAi,  mech.fbi,
        mech.fEi,  mech.fOW, mech.fDn, mech.fTA,  mech.fT1,  mech.fT2,
        mech.fT3,  mech.fIT, mech.fRI, mech.fRS,  mech.fRO,  mech.fPI,
        mech.fPS,  mech.fPO, mech.fEff,
    )

    return OmegaRatesResult(
        omega    = out_omega.copy_to_host(),
        dTCdt    = out_dTCdt.copy_to_host(),
        domegadt = out_Jv_om.copy_to_host(),
        Jv_TC    = out_Jv_TC.copy_to_host(),
        ratio    = out_ratio.copy_to_host(),
    )