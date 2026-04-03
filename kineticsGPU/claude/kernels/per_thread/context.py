"""
kernels/per_thread/context.py — Calcul du RHS + mise en cache du contexte cinétique.

Architecture : 1 thread = 1 réacteur (scalaire).

Fonction exportée :
    _eval_rhs_and_context(T, C, <ctx_slices>, omega, n_sp, <mech_args>)

Le contexte cinétique (kf, Kc, cr, cp, dG0, sR_r, sR_p, M, etc.) est écrit
dans des row-slices de tableaux globaux pré-alloués (shape (B, n_reactions)).
Ces valeurs sont ensuite relues par _eval_jvp sans recalcul.

Signature des ctx_slices (tous sont des vues 1D de la ligne b) :
    Élémentaires : kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe
    Trois-corps   : kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb
    Falloff       : kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
                    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
                    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f
"""

import math
from numba import cuda, float64

from ..shared.thermo import (
    R_CONST, PREF, EPS, LN10, INV_LN10,
    _load_coef7, _h_RT, _s_R, _cp_R, _kf_arr, _log_conc_prod,
)


@cuda.jit(device=True)
def _eval_rhs_and_context(
    T, C,
    # ── Contexte élémentaires ─────────────────────────────────────────
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    # ── Contexte trois-corps ──────────────────────────────────────────
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    # ── Contexte falloff ─────────────────────────────────────────────
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    # ── Sortie ────────────────────────────────────────────────────────
    omega,
    # ── Données mécanisme ─────────────────────────────────────────────
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    """
    Calcule le vecteur omega (taux de production des espèces, mol/m³/s) et
    met en cache toutes les quantités cinétiques intermédiaires nécessaires à
    l'évaluation ultérieure de la JVP.

    Les contextes (kf_e, Kc_e, ...) sont des vues 1D de la ligne [b] des
    tableaux globaux (B, n_reactions).
    """
    c7 = cuda.local.array(7, float64)

    # Initialisation de omega
    for s in range(n_sp):
        omega[s] = float64(0.0)

    # ── Réactions élémentaires ────────────────────────────────────────
    for j in range(n_e):
        kf = _kf_arr(T, eA[j], en[j], eEa[j])
        kf_e[j] = kf
        dG = float64(0.0)
        for k in range(eRO[j], eRO[j + 1]):
            sp = eRI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_re[k] = s_s
            dG -= eRS[k] * (h_s - T * s_s)
        for k in range(ePO[j], ePO[j + 1]):
            sp = ePI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_pe[k] = s_s
            dG += ePS[k] * (h_s - T * s_s)
        dG0_e[j] = dG
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** eDn[j])
        Kc_e[j] = Kc
        lr = _log_conc_prod(C, eRO[j], eRO[j + 1], eRI, eRS)
        lp = _log_conc_prod(C, ePO[j], ePO[j + 1], ePI, ePS)
        cr = math.exp(lr); cp = math.exp(lp)
        cr_e[j] = cr; cp_e[j] = cp
        ow = eOW[j]
        RR = kf * (cr - (1.0 - ow) * cp / Kc)
        for k in range(eRO[j], eRO[j + 1]):
            omega[eRI[k]] -= eRS[k] * RR
        for k in range(ePO[j], ePO[j + 1]):
            omega[ePI[k]] += ePS[k] * RR

    # ── Réactions trois-corps ─────────────────────────────────────────
    for j in range(n_tb):
        kf = _kf_arr(T, tbA[j], tbn[j], tbEa[j])
        kf_tb[j] = kf
        dG = float64(0.0)
        for k in range(tbRO[j], tbRO[j + 1]):
            sp = tbRI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_rtb[k] = s_s
            dG -= tbRS[k] * (h_s - T * s_s)
        for k in range(tbPO[j], tbPO[j + 1]):
            sp = tbPI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_ptb[k] = s_s
            dG += tbPS[k] * (h_s - T * s_s)
        dG0_tb[j] = dG
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** tbDn[j])
        Kc_tb[j] = Kc
        lr = _log_conc_prod(C, tbRO[j], tbRO[j + 1], tbRI, tbRS)
        lp = _log_conc_prod(C, tbPO[j], tbPO[j + 1], tbPI, tbPS)
        cr = math.exp(lr); cp = math.exp(lp)
        cr_tb[j] = cr; cp_tb[j] = cp
        M = float64(0.0)
        for s in range(n_sp):
            M += C[s] * tbEff[j * n_sp + s]
        M_tb[j] = M
        ow    = tbOW[j]
        baseR = kf * (cr - (1.0 - ow) * cp / Kc)
        bRR_tb[j] = baseR
        RR = baseR * M
        for k in range(tbRO[j], tbRO[j + 1]):
            omega[tbRI[k]] -= tbRS[k] * RR
        for k in range(tbPO[j], tbPO[j + 1]):
            omega[tbPI[k]] += tbPS[k] * RR

    # ── Réactions falloff ─────────────────────────────────────────────
    for j in range(n_f):
        k0_v   = _kf_arr(T, fA0[j], fb0[j], fE0[j])
        kinf_v = _kf_arr(T, fAi[j], fbi[j], fEi[j])
        k0_f[j] = k0_v; kinf_f[j] = kinf_v
        M = float64(0.0)
        for s in range(n_sp):
            M += C[s] * fEff[j * n_sp + s]
        M_f[j] = M
        ki_s = kinf_v if kinf_v > EPS else EPS
        Pr   = (k0_v * M) / ki_s
        Pr_f[j] = Pr
        if fIT[j]:
            tA = fTA[j]; t1 = fT1[j]; t2 = fT2[j]; t3 = fT3[j]
            Fc = float64(0.0)
            if t1 > EPS: Fc += tA * math.exp(-T / t1)
            if t3 > EPS: Fc += (1.0 - tA) * math.exp(-T / t3)
            if t2 > EPS: Fc += math.exp(-t2 / T)
            if Fc < EPS: Fc = EPS
            lFc   = math.log10(Fc)
            Cp    = -0.4 - 0.67 * lFc
            Np    =  0.75 - 1.27 * lFc
            Prr   = Pr if Pr > EPS else EPS
            logPr = math.log10(Prr)
            num   = logPr + Cp
            den   = Np - 0.14 * (logPr + Cp)
            if abs(den) < EPS: den = EPS if den >= 0.0 else -EPS
            f1_v  = num / den
            ex    = lFc / (1.0 + f1_v * f1_v)
            F_v   = 10.0 ** ex
        else:
            Fc   = float64(1.0); lFc  = float64(0.0)
            Cp   = float64(0.0); Np   = float64(1.0)
            f1_v = float64(0.0); ex   = float64(0.0); F_v = float64(1.0)
        Fcent_f[j] = Fc;  lFcent_f[j] = lFc
        Cpar_f[j]  = Cp;  Npar_f[j]   = Np
        f1_f[j]    = f1_v; exp_f[j]   = ex; F_f[j] = F_v
        kbase = ki_s * (Pr / (1.0 + Pr))
        kf_v  = kbase * F_v
        kb_f[j] = kbase; kf_f[j] = kf_v
        dG = float64(0.0)
        for k in range(fRO[j], fRO[j + 1]):
            sp = fRI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_rf[k] = s_s
            dG -= fRS[k] * (h_s - T * s_s)
        for k in range(fPO[j], fPO[j + 1]):
            sp = fPI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_pf[k] = s_s
            dG += fPS[k] * (h_s - T * s_s)
        dG0_f[j] = dG
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** fDn[j])
        Kc_f[j] = Kc
        lr = _log_conc_prod(C, fRO[j], fRO[j + 1], fRI, fRS)
        lp = _log_conc_prod(C, fPO[j], fPO[j + 1], fPI, fPS)
        cr = math.exp(lr); cp = math.exp(lp)
        cr_f[j] = cr; cp_f[j] = cp
        ow = fOW[j]
        RR = kf_v * (cr - (1.0 - ow) * cp / Kc)
        for k in range(fRO[j], fRO[j + 1]):
            omega[fRI[k]] -= fRS[k] * RR
        for k in range(fPO[j], fPO[j + 1]):
            omega[fPI[k]] += fPS[k] * RR