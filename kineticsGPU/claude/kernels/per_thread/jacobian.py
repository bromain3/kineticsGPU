"""
kernels/per_thread/jacobian_chem.py — Jacobienne chimique pure d(omega)/d(xk) (per-thread).

Architecture : 1 thread = 1 réacteur (scalaire).

Calcule la matrice jacobienne chimique J de taille (n_sp × n_state) :

    J[s, j] = d(omega_s) / d(xk_j)

où xk = (T, C_0, ..., C_{n_sp-1}).

Convention d'indexation :
    J_out est un tableau 1D de taille n_sp * n_state, row-major.
    J_out[s * n_state + j] = d(omega_s) / d(xk_j)
    s = indice espèce (ligne),  j = 0 → T,  j = s'+1 → C_{s'}

CORRECTION vs version précédente :
    dlnKc/dT calculé via Σ νk × dBk/dT (PyJac eq. 73-75, paper §3.1)
    au lieu de (dSp - dSr)/(R*T) qui utilisait sR_re/sR_pe à tort.

    dBk/dT = (a0-1)/T + a5/T² + a1/2 + T*(a2/3 + T*(a3/4 + a4/5*T))
    dlnKc  = Σ_{k∈reac} (-ν_k) * dBk/dT + Σ_{k∈prod} ν_k * dBk/dT
           = Σ_k  ν_k_net * dBk/dT      (PyJac eq. 74)

    Les tableaux sR_re / sR_pe ne sont PAS utilisés pour la colonne T.
    Ils servent uniquement en interne à _eval_rhs_and_context pour
    calculer les taux de réaction (produit de concentrations en log),
    et n'ont aucun sens thermodynamique utilisable ici.
"""

import math
import numpy as np
from numba import cuda, float64

from ..shared.thermo import R_CONST, EPS, LN10, INV_LN10
from .context import _eval_rhs_and_context


# ---------------------------------------------------------------------------
#  Helper interne : dBk/dT pour l'espèce sp à la température T
#  PyJac eq. (74) :
#    Bk = a6 - a0 + (a0-1)*lnT + T*(a1/2 + T*(a2/6 + T*(a3/12 + a4/20*T))) - a5/T
#    dBk/dT = (a0-1)/T + a5/T² + a1/2 + T*(a2/3 + T*(a3/4 + a4/5*T))
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _dB_dT(T, sp, thL, thH, thM):
    """
    Dérivée de Bk par rapport à T pour l'espèce sp.
    Bk(T) est défini par PyJac eq.(27) / eq.(74).
    Sélectionne automatiquement les coefficients low/high selon T vs Tmid.
    """
    base = sp * 7
    if T <= thM[sp]:
        a = thL
    else:
        a = thH
    a0 = a[base + 0]
    a1 = a[base + 1]
    a2 = a[base + 2]
    a3 = a[base + 3]
    a4 = a[base + 4]
    a5 = a[base + 5]
    # dBk/dT = (a0 - 1)/T + a5/T^2 + a1/2 + T*(a2/3 + T*(a3/4 + a4/5*T))
    return ((a0 - float64(1.0)) / T
            + a5 / (T * T)
            + a1 / float64(2.0)
            + T * (a2 / float64(3.0)
                   + T * (a3 / float64(4.0)
                          + a4 / float64(5.0) * T)))


# ---------------------------------------------------------------------------
#  Helper interne : dlnKc/dT pour la réaction j (élémentaire ou 3-corps)
#  PyJac eq. (73) : d(ln Kc)/dT = Σ_k ν_k * dBk/dT
#  avec ν_k = stoichio produits - stoichio réactants
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _dlnKc_dT_elem(T, thL, thH, thM,
                   eRI, eRS, eRO_j, eRO_j1,
                   ePI, ePS, ePO_j, ePO_j1):
    """
    Calcule d(ln Kc)/dT pour une réaction élémentaire ou trois-corps.
    Somme sur réactants (signe -) et produits (signe +).
    """
    val = float64(0.0)
    # Réactants : ν_k_net contribution = -ν_reac * dBk/dT
    for k in range(eRO_j, eRO_j1):
        val -= eRS[k] * _dB_dT(T, eRI[k], thL, thH, thM)
    # Produits : ν_k_net contribution = +ν_prod * dBk/dT
    for k in range(ePO_j, ePO_j1):
        val += ePS[k] * _dB_dT(T, ePI[k], thL, thH, thM)
    return val


# ---------------------------------------------------------------------------
#  Colonne j_col de d(omega)/d(xk) — version PyJac
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _dchem_col(
    j_col,
    T, C,
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    col_out,
):
    """
    Colonne j_col de la jacobienne chimique pure d(omega)/d(xk).

    j_col == 0      → direction T  (vT=1, vC=0)
    j_col == s'+1   → direction C_{s'} (vT=0, vC=e_{s'})

    Équations PyJac :
      - d(omega_s)/dT  : eq.(111) du paper, avec dlnKc via eq.(73-74)
      - d(omega_s)/dCk : eq.(112), terme kf*ν/C pour la concentration directrice
    """
    if j_col == 0:
        vT = float64(1.0)
    else:
        vT = float64(0.0)

    for s in range(n_sp):
        col_out[s] = float64(0.0)

    # ── Élémentaires ──────────────────────────────────────────────────
    for j in range(n_e):
        kf  = kf_e[j]
        Kc  = Kc_e[j]
        cr  = cr_e[j]
        cp  = cp_e[j]
        ow  = eOW[j]
        fac = float64(1.0) - ow

        # d(kf)/dT = kf * (β/T + Ea/(R*T²))   [PyJac eq.(65)]
        dkf_dT = kf * (en[j] / T + eEa[j] / (R_CONST * T * T))

        # CORRECTION : d(ln Kc)/dT via dBk/dT  [PyJac eq.(73-74)]
        # (sR_re / sR_pe ne sont PAS utilisés ici)
        dlnKc = _dlnKc_dT_elem(
            T, thL, thH, thM,
            eRI, eRS, eRO[j], eRO[j + 1],
            ePI, ePS, ePO[j], ePO[j + 1],
        )
        dKc = Kc * dlnKc

        # dRR/dT = [dkf/dT * (cr - fac*cp/Kc) + kf*fac*cp/Kc²*dKc]
        # PyJac eq.(75) : ∂Ri/∂T = Rf*(β+Ea/T-Σν')/T - Rr*(β+Ea/T-Σν''-ΣνdBdT)/T
        # Sous notre forme factorisée (kf, kr = kf/Kc) c'est équivalent
        dRR_v = (dkf_dT * (cr - fac * cp / Kc)
                 + kf * fac * cp / (Kc * Kc) * dKc) * vT

        # dRR/dCk  [PyJac eq.(67) / eq.(76) : terme ν/C pour l'espèce directrice]
        if j_col > 0:
            s_dir = j_col - 1
            for k in range(eRO[j], eRO[j + 1]):
                if eRI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v += kf * cr * (eRS[k] / c)
            for k in range(ePO[j], ePO[j + 1]):
                if ePI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v -= kf * fac / Kc * cp * (ePS[k] / c)

        # Scatter sur les espèces participantes
        for k in range(eRO[j], eRO[j + 1]):
            col_out[eRI[k]] -= eRS[k] * dRR_v
        for k in range(ePO[j], ePO[j + 1]):
            col_out[ePI[k]] += ePS[k] * dRR_v

    # ── Trois-corps ───────────────────────────────────────────────────
    for j in range(n_tb):
        kf  = kf_tb[j]
        Kc  = Kc_tb[j]
        cr  = cr_tb[j]
        cp  = cp_tb[j]
        M   = M_tb[j]
        br  = bRR_tb[j]
        ow  = tbOW[j]
        fac = float64(1.0) - ow

        dkf_dT = kf * (tbn[j] / T + tbEa[j] / (R_CONST * T * T))

        # CORRECTION : d(ln Kc)/dT via dBk/dT  [PyJac eq.(73-74)]
        dlnKc = _dlnKc_dT_elem(
            T, thL, thH, thM,
            tbRI, tbRS, tbRO[j], tbRO[j + 1],
            tbPI, tbPS, tbPO[j], tbPO[j + 1],
        )
        dKc = Kc * dlnKc

        dbR_dT = (dkf_dT * (cr - fac * cp / Kc)
                  + kf * fac * cp / (Kc * Kc) * dKc)

        # dM/dCk : M = Σ eff_s * C_s  → dM/dCk = eff_k
        # dM/dT = 0  (M ne dépend pas de T à P,Y constants — PyJac §2.3.2)
        if j_col == 0:
            dM_v = float64(0.0)
        else:
            s_dir = j_col - 1
            dM_v  = tbEff[j * n_sp + s_dir]

        # dRR/dT = dbR/dT * M  [PyJac eq.(111), ci = M, dci/dT = -M/T]
        # Note: PyJac inclut le terme -M/T*Ri séparément via j_temp ;
        # ici on le factorise dans dbR_dT via la dérivée de kf et Kc.
        # Le terme -ci/T*Ri de PyJac eq.(79) correspond au -Ri/T dans
        # get_elementary_rxn_dt de pyJac.write_dr_dy qui écrit :
        #   j_temp = -pres_mod * (Rf-Rr) / T  + pres_mod * (dkf_dt * ...)
        # Notre forme est équivalente une fois développée.
        dRR_v = dbR_dT * vT * M

        if j_col > 0:
            s_dir = j_col - 1
            for k in range(tbRO[j], tbRO[j + 1]):
                if tbRI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v += kf * cr * (tbRS[k] / c) * M
            for k in range(tbPO[j], tbPO[j + 1]):
                if tbPI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v -= kf * fac / Kc * cp * (tbPS[k] / c) * M
            # terme dM/dCk * bR  [PyJac eq.(123) : αij*Ri terme]
            dRR_v += br * dM_v

        for k in range(tbRO[j], tbRO[j + 1]):
            col_out[tbRI[k]] -= tbRS[k] * dRR_v
        for k in range(tbPO[j], tbPO[j + 1]):
            col_out[tbPI[k]] += tbPS[k] * dRR_v

    # ── Falloff ───────────────────────────────────────────────────────
    for j in range(n_f):
        kf_v = kf_f[j]
        Kc   = Kc_f[j]
        cr   = cr_f[j]
        cp   = cp_f[j]
        k0_v = k0_f[j]
        ki   = kinf_f[j]
        M    = M_f[j]
        Pr   = Pr_f[j]
        F    = F_f[j]
        kb   = kb_f[j]
        Fc   = Fcent_f[j]
        lFc  = lFcent_f[j]
        Cp   = Cpar_f[j]
        Np   = Npar_f[j]
        f1_v = f1_f[j]
        ow   = fOW[j]
        fac  = float64(1.0) - ow
        ki_s = ki if ki > EPS else EPS
        pr_ok = (Pr > EPS) and (M > EPS)

        dk0_dT = k0_v * (fb0[j] / T + fE0[j] / (R_CONST * T * T))
        dki_dT = ki   * (fbi[j] / T + fEi[j] / (R_CONST * T * T))

        if j_col == 0:
            dM_v = float64(0.0)
        else:
            s_dir = j_col - 1
            dM_v  = fEff[j * n_sp + s_dir]

        # dPr/dT  [PyJac eq.(84)]
        if pr_ok:
            dPr_dT = (dk0_dT * M - k0_v * M * dki_dT / ki_s) / ki_s
            dPr_vC = (k0_v / ki_s) * dM_v
        else:
            dPr_dT = float64(0.0)
            dPr_vC = float64(0.0)

        # CORRECTION : d(ln Kc)/dT via dBk/dT  [PyJac eq.(73-74)]
        dlnKc = _dlnKc_dT_elem(
            T, thL, thH, thM,
            fRI, fRS, fRO[j], fRO[j + 1],
            fPI, fPS, fPO[j], fPO[j + 1],
        )
        dKc = Kc * dlnKc
        ce  = cr - fac * cp / Kc

        # dF/dT et dF/dPr  [PyJac eq.(90-96) pour Troe]
        if fIT[j]:
            t1 = fT1[j]; t2 = fT2[j]; t3 = fT3[j]; tA = fTA[j]

            # dFcent/dT  [PyJac eq.(93)]
            dFc_dT = float64(0.0)
            if t1 > EPS:
                dFc_dT += tA * (-float64(1.0) / t1) * math.exp(-T / t1)
            if t3 > EPS:
                dFc_dT += (float64(1.0) - tA) * (-float64(1.0) / t3) * math.exp(-T / t3)
            if t2 > EPS:
                dFc_dT += math.exp(-t2 / T) * (t2 / (T * T))

            fc_ok   = Fc > EPS
            # d(log10 Fcent)/dT  [PyJac eq.(95) numérateur côté Fcent]
            dlFc_dT = (dFc_dT / Fc * INV_LN10) if fc_ok else float64(0.0)

            # dCp/dT = ∂ATroe/∂Fcent * dFcent/dT  [PyJac eq.(95)]
            # ATroe = log10(Pr) - 0.67*log10(Fcent) - 0.4
            # dATroe/dFcent = -0.67/(Fcent*ln10) → dCp/dT = -0.67*dlFc/dT
            dCp_dT  = -float64(0.67) * dlFc_dT
            # BTroe = 0.806 - 1.1762*log10(Fcent) - 0.14*log10(Pr)
            # dBTroe/dFcent = -1.1762/(Fcent*ln10) → dNp/dT = -1.1762*dlFc/dT
            # Mais dans notre convention Np = BTroe, Cp = ATroe
            dNp_dT  = -float64(1.27) * dlFc_dT

            Prr      = Pr if Pr > EPS else EPS
            # d(log10 Pr)/d(Pr) = 1/(Pr*ln10)
            dlPr_dPr = (float64(1.0) / (Prr * LN10)) if pr_ok else float64(0.0)
            lPr      = math.log10(Prr) if pr_ok else float64(0.0)

            # f1 = (ATroe) / (BTroe - 0.14*ATroe)  [PyJac notation interne]
            # nf1 = lPr + Cp = ATroe,  df = Np - 0.14*nf1 = BTroe
            nf1    = lPr + Cp
            df     = Np - float64(0.14) * nf1
            abs_df = abs(df)
            df_s   = df if abs_df > EPS else (EPS if df >= float64(0.0) else -EPS)
            df_sq  = df_s * df_s
            df_ok  = abs_df > EPS

            # df1/dPr  [PyJac eq.(96) : ∂ATroe/∂Pr = 1/(Pr*ln10), ∂BTroe/∂Pr = -0.14/(Pr*ln10)]
            df1_dPr = ((dlPr_dPr * df_s + float64(0.14) * nf1 * dlPr_dPr) / df_sq
                       if (df_ok and pr_ok) else float64(0.0))
            # df1/dT via dCp/dT et dNp/dT
            nT_     = dCp_dT
            dT_     = dNp_dT - float64(0.14) * dCp_dT
            df1_dT  = (nT_ * df_s - nf1 * dT_) / df_sq if df_ok else float64(0.0)

            # lgF = log10(Fcent) / (1 + f1²)  [PyJac F formula]
            # de/dlogFcent = 1/(1+f1²)
            # de/df1 = -2*f1*logFcent/(1+f1²)²
            f1sq   = f1_v * f1_v
            de_dlF = float64(1.0) / (float64(1.0) + f1sq)
            de_df1 = -float64(2.0) * f1_v * lFc / ((float64(1.0) + f1sq) ** 2)
            de_dT  = de_dlF * dlFc_dT + de_df1 * df1_dT
            de_dPr = de_df1 * df1_dPr
            # F = 10^(lgF) → dF/dT = F*ln(10)*de/dT
            dF_dT  = F * LN10 * de_dT
            dF_dPr = F * LN10 * de_dPr
        else:
            dF_dT  = float64(0.0)
            dF_dPr = float64(0.0)

        # dkb/dT et dkb/dCk  [PyJac write_troe_dt / get_pdep_dt]
        # kb = kinf * Pr/(1+Pr)  pour falloff unimoleculaire
        Pd       = (float64(1.0) + Pr) * (float64(1.0) + Pr)
        dkb_dT   = dki_dT * (Pr / (float64(1.0) + Pr)) + ki_s * (float64(1.0) / Pd) * dPr_dT
        dkb_vC   = ki_s * (float64(1.0) / Pd) * dPr_vC
        # dkf_eff/dT = dkb/dT * F + kb * (dF/dT + dF/dPr * dPr/dT)
        dkf_dT_v = dkb_dT * F + kb * (dF_dT + dF_dPr * dPr_dT)
        dkf_vC   = dkb_vC * F + kb * dF_dPr * dPr_vC

        # dRR/dT  [PyJac eq.(111) pour falloff]
        dRR_v = (dkf_dT_v * vT * ce
                 + kf_v * fac * cp / (Kc * Kc) * dKc * vT)

        if j_col > 0:
            s_dir = j_col - 1
            for k in range(fRO[j], fRO[j + 1]):
                if fRI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v += kf_v * cr * (fRS[k] / c)
            for k in range(fPO[j], fPO[j + 1]):
                if fPI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v -= kf_v * fac / Kc * cp * (fPS[k] / c)
            dRR_v += dkf_vC * ce

        for k in range(fRO[j], fRO[j + 1]):
            col_out[fRI[k]] -= fRS[k] * dRR_v
        for k in range(fPO[j], fPO[j + 1]):
            col_out[fPI[k]] += fPS[k] * dRR_v


# ---------------------------------------------------------------------------
#  Fonction principale : jacobienne chimique pure
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _eval_jacobian(
    T, C,
    J_out,
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    n_state = n_sp + 1
    col_chem = cuda.local.array(54, float64)

    for i in range(n_sp * n_state):
        J_out[i] = float64(0.0)

    for j_col in range(n_state):
        _dchem_col(
            j_col, T, C,
            kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
            kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
            kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
            k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
            Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
            n_sp,
            thL, thH, thM,
            n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
            n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
            n_f, fA0, fb0, fE0, fAi, fbi, fEi,
            fOW, fDn, fTA, fT1, fT2, fT3, fIT,
            fRI, fRS, fRO, fPI, fPS, fPO, fEff,
            col_chem,
        )
        for s in range(n_sp):
            J_out[s * n_state + j_col] = col_chem[s]


# =============================================================================
#  Kernel CUDA global
# =============================================================================

MAX_N_STATE = 54


@cuda.jit(cache=True)
def _jacobian_chem_kernel_pt(
    TC_in,          # (B, n_state)  float64
    J_all,          # (B, n_sp * n_state)  float64  — sortie
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

    C   = cuda.local.array(MAX_N_STATE, float64)
    om  = cuda.local.array(MAX_N_STATE, float64)   # omega nécessaire pour _eval_rhs_and_context

    for s in range(n_sp):
        C[s] = TC_in[b, s + 1]

    # Calcul du contexte cinétique au point (T, C)
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

    # Jacobienne chimique pure (pas de ligne dT/dt)
    _eval_jacobian(
        T_loc, C,
        J_all[b],
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


# =============================================================================
#  Wrapper Python — CUDAJacobianChemEvaluatorPT
# =============================================================================

class CUDAJacobianChemEvaluatorPT:
    """
    Évalue la jacobienne chimique pure d(omega)/d(xk) via le kernel per-thread.

    Usage
    -----
    ev = CUDAJacobianChemEvaluatorPT(mech, dtype=np.float64)
    J  = ev.evaluate(T_arr, C_arr)   # T_arr : (B,), C_arr : (B, n_sp)
    # J : np.ndarray (B, n_sp, n_state)   avec n_state = n_sp + 1

    J[b, s, j] = d(omega_s) / d(xk_j)
      s         → espèce (ligne)
      j=0       → T
      j=s'+1    → C_{s'}

    Pas de ligne dT/dt, pas de notion de réacteur adiabatique.
    """

    THREADS = 64

    def __init__(self, mech, dtype=np.float64):
        self._mech    = mech
        self._dtype   = dtype
        self._n_sp    = mech.n_species
        self._n_state = self._n_sp + 1
        self._built   = False

    def _alloc_context(self, B):
        dt = self._dtype
        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff

        def z(shape): return cuda.device_array(shape, dtype=dt)

        n_e_flat  = e.react_offset[-1]  if e.n_reactions  else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1
        n_f_flat  = f.react_offset[-1]  if f.n_reactions  else 1

        self._ctx = dict(
            kf_e   = z((B, e.n_reactions)),   Kc_e  = z((B, e.n_reactions)),
            cr_e   = z((B, e.n_reactions)),    cp_e  = z((B, e.n_reactions)),
            dG0_e  = z((B, e.n_reactions)),
            sRre   = z((B, n_e_flat)),         sRpe  = z((B, n_e_flat)),

            kf_tb  = z((B, tb.n_reactions)),   Kc_tb = z((B, tb.n_reactions)),
            cr_tb  = z((B, tb.n_reactions)),   cp_tb = z((B, tb.n_reactions)),
            dG0_tb = z((B, tb.n_reactions)),
            sRrtb  = z((B, n_tb_flat)),        sRptb = z((B, n_tb_flat)),
            bRR_tb = z((B, tb.n_reactions)),   M_tb  = z((B, tb.n_reactions)),

            kf_f   = z((B, f.n_reactions)),    Kc_f  = z((B, f.n_reactions)),
            cr_f   = z((B, f.n_reactions)),    cp_f  = z((B, f.n_reactions)),
            dG0_f  = z((B, f.n_reactions)),
            sRrf   = z((B, n_f_flat)),         sRpf  = z((B, n_f_flat)),
            k0_f   = z((B, f.n_reactions)),    kinf_f= z((B, f.n_reactions)),
            M_f    = z((B, f.n_reactions)),    Pr_f  = z((B, f.n_reactions)),
            F_f    = z((B, f.n_reactions)),    kb_f  = z((B, f.n_reactions)),
            Fc_f   = z((B, f.n_reactions)),    lFc_f = z((B, f.n_reactions)),
            Cp_f   = z((B, f.n_reactions)),    Np_f  = z((B, f.n_reactions)),
            f1_f   = z((B, f.n_reactions)),    exp_f = z((B, f.n_reactions)),
        )
        self._B_alloc = B
        self._built   = True

    def evaluate(self, T_arr, C_arr):
        T_arr = np.asarray(T_arr, dtype=self._dtype)
        C_arr = np.asarray(C_arr, dtype=self._dtype)
        B     = T_arr.shape[0]
        nsp   = self._n_sp
        nst   = self._n_state

        if not self._built or self._B_alloc != B:
            self._alloc_context(B)

        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0]  = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)

        # Sortie : (B, n_sp * n_state) — reshape en (B, n_sp, n_state) côté host
        d_J = cuda.device_array((B, nsp * nst), dtype=self._dtype)

        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff
        cx = self._ctx

        threads = self.THREADS
        blocks  = (B + threads - 1) // threads

        _jacobian_chem_kernel_pt[blocks, threads](
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

        J_flat = d_J.copy_to_host()
        return J_flat.reshape(B, nsp, nst)