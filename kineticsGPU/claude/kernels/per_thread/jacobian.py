"""
kernels/per_thread/jacobian.py — Jacobienne analytique complète du RHS adiabatique (per-thread).

Architecture : 1 thread = 1 réacteur (scalaire).

Fonction exportée :
    _eval_jacobian(T, C, J_out, <ctx_slices>, n_sp, <mech_args>)

Calcule la matrice jacobienne complète J de taille (n_state × n_state) :

    J[i, j] = d(dTC_i/dt) / d(xk_j)

où xk = (T, C_0, ..., C_{n_sp-1}) et dTC = (dT/dt, dC_0/dt, ...).

Stratégie :
    1. Calcul de d(omega_i)/d(T)      → colonne 0 de J_chem  (n_sp lignes)
    2. Calcul de d(omega_i)/d(C_s)    → colonnes 1..n_sp      (n_sp × n_sp)
       Les trois types de réactions (élémentaires, trois-corps, falloff) sont
       traités avec la même algèbre que _eval_jvp mais en décomposant
       explicitement chaque direction de base e_j au lieu d'un vecteur v arbitraire.
    3. Propagation adiabatique :
       d(dT/dt)/d(xk_j) via les dérivées de (ntot, cv_mix, su_om) par rapport à xk_j.
       d(dC_s/dt)/d(xk_j) = d(omega_s)/d(xk_j)  (trivial).

Convention d'indexation :
    J_out est un tableau 2D de shape (n_state, n_state), row-major.
    J_out[i, j] = dérivée de la i-ième composante du RHS par rapport à xk[j].
    i=0 → dT/dt,  i=s+1 → dC_s/dt
    j=0 → T,      j=s+1 → C_s

Le contexte cinétique (kf_*, Kc_*, cr_*, cp_*, dG0_*, sR_*, M_*, bRR_*, Pr_*, F_*,
kb_*, k0_*, kinf_*, Fc_*, lFc_*, Cp_*, Np_*, f1_*, exp_*) doit avoir été
calculé au préalable par _eval_rhs_and_context (context.py) au point (T, C).
omega (vecteur des taux de production) doit également être disponible.

Aucune simplification physique n'est introduite : la physique est identique
à celle de _eval_jvp / _eval_rhs_and_context.
"""

import math
import numpy as np
from numba import cuda, float64

from ..shared.thermo import (
    R_CONST, PREF, EPS, LN10, INV_LN10,
    _load_coef7, _h_RT, _s_R, _cp_R,
)
from .context import _eval_rhs_and_context


# ---------------------------------------------------------------------------
#  Helpers scalaires internes (inlinés par Numba)
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _dchem_col(
    j_col,          # indice de la colonne xk (0=T, 1..n_sp=C_s)
    T, C,
    # contexte cinétique élémentaires
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    # contexte cinétique trois-corps
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    # contexte cinétique falloff
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    # données mécanisme
    n_sp,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    # sortie : colonne j_col de la jacobienne chimique (taille n_sp)
    col_out,
):
    """
    Calcule la colonne j_col de d(omega)/d(xk) et l'écrit dans col_out[0..n_sp-1].

    Si j_col == 0  : direction vT=1, vC=0  (dérivée par rapport à T)
    Si j_col == s+1: direction vT=0, vC=e_s (dérivée par rapport à C_s)

    C'est exactement la JVP évaluée sur le vecteur canonique correspondant,
    mais sans recours à _eval_jvp (on ne peut pas l'appeler ici par contrainte).
    """
    # Vecteur directionnel canonique
    if j_col == 0:
        vT = float64(1.0)
    else:
        vT = float64(0.0)
    # vC[s] = 1 si s == j_col-1 else 0  — on n'alloue pas le tableau,
    # on calcule directement les produits scalaires nécessaires.

    # Mise à zéro de col_out
    for s in range(n_sp):
        col_out[s] = float64(0.0)

    # ── Élémentaires ──────────────────────────────────────────────────
    for j in range(n_e):
        kf  = kf_e[j]
        Kc  = Kc_e[j]
        cr  = cr_e[j]
        cp  = cp_e[j]
        dG  = dG0_e[j]
        ow  = eOW[j]
        fac = float64(1.0) - ow

        dkf_dT = kf * (en[j] / T + eEa[j] / (R_CONST * T * T))

        dSr = float64(0.0)
        for k in range(eRO[j], eRO[j + 1]):
            dSr += eRS[k] * sR_re[k]
        dSp = float64(0.0)
        for k in range(ePO[j], ePO[j + 1]):
            dSp += ePS[k] * sR_pe[k]
        dlnKc = (dSp - dSr) / (R_CONST * T) + dG / (R_CONST * T * T) - eDn[j] / T
        dKc   = Kc * dlnKc

        # Partie T
        dRR_v = (dkf_dT * (cr - fac * cp / Kc)
                 + kf * fac * cp / (Kc * Kc) * dKc) * vT

        # Partie C_{j_col-1} (scalaire car vC est un vecteur de base)
        if j_col > 0:
            s_dir = j_col - 1  # espèce de la direction canonique
            # Contribution des réactifs
            for k in range(eRO[j], eRO[j + 1]):
                if eRI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v += kf * cr * (eRS[k] / c)
            # Contribution des produits
            for k in range(ePO[j], ePO[j + 1]):
                if ePI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v -= kf * fac / Kc * cp * (ePS[k] / c)

        # Scatter dans col_out
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
        dG  = dG0_tb[j]
        M   = M_tb[j]
        br  = bRR_tb[j]
        ow  = tbOW[j]
        fac = float64(1.0) - ow

        dkf_dT = kf * (tbn[j] / T + tbEa[j] / (R_CONST * T * T))

        dSr = float64(0.0)
        for k in range(tbRO[j], tbRO[j + 1]):
            dSr += tbRS[k] * sR_rtb[k]
        dSp = float64(0.0)
        for k in range(tbPO[j], tbPO[j + 1]):
            dSp += tbPS[k] * sR_ptb[k]
        dlnKc  = (dSp - dSr) / (R_CONST * T) + dG / (R_CONST * T * T) - tbDn[j] / T
        dKc    = Kc * dlnKc
        dbR_dT = (dkf_dT * (cr - fac * cp / Kc)
                  + kf * fac * cp / (Kc * Kc) * dKc)

        # dM dans la direction canonique
        if j_col == 0:
            dM_v = float64(0.0)   # M ne dépend pas de T
        else:
            s_dir = j_col - 1
            dM_v  = tbEff[j * n_sp + s_dir]

        dRR_v = dbR_dT * vT * M

        if j_col > 0:
            s_dir = j_col - 1
            # Contribution réactifs (terme kf * cr / c_s * M)
            for k in range(tbRO[j], tbRO[j + 1]):
                if tbRI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v += kf * cr * (tbRS[k] / c) * M
            # Contribution produits (terme -kf/Kc * cp / c_s * M)
            for k in range(tbPO[j], tbPO[j + 1]):
                if tbPI[k] == s_dir:
                    c = C[s_dir] if C[s_dir] > EPS else EPS
                    dRR_v -= kf * fac / Kc * cp * (tbPS[k] / c) * M
            # Contribution via dM (terme br * eff_s)
            dRR_v += br * dM_v

        # Scatter
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
        dG   = dG0_f[j]
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
        ex   = exp_f[j]
        ow   = fOW[j]
        fac  = float64(1.0) - ow
        ki_s = ki if ki > EPS else EPS
        pr_ok = (Pr > EPS) and (M > EPS)

        dk0_dT = k0_v * (fb0[j] / T + fE0[j] / (R_CONST * T * T))
        dki_dT = ki   * (fbi[j] / T + fEi[j] / (R_CONST * T * T))

        # dM dans la direction canonique
        if j_col == 0:
            dM_v = float64(0.0)
        else:
            s_dir = j_col - 1
            dM_v  = fEff[j * n_sp + s_dir]

        if pr_ok:
            dPr_dT = (dk0_dT * M - k0_v * M * dki_dT / ki_s) / ki_s
            dPr_vC = (k0_v / ki_s) * dM_v
        else:
            dPr_dT = float64(0.0)
            dPr_vC = float64(0.0)

        dSr = float64(0.0)
        for k in range(fRO[j], fRO[j + 1]):
            dSr += fRS[k] * sR_rf[k]
        dSp = float64(0.0)
        for k in range(fPO[j], fPO[j + 1]):
            dSp += fPS[k] * sR_pf[k]
        dlnKc = (dSp - dSr) / (R_CONST * T) + dG / (R_CONST * T * T) - fDn[j] / T
        dKc   = Kc * dlnKc
        ce    = cr - fac * cp / Kc

        # Dérivées de F par rapport à T et Pr
        if fIT[j]:
            t1 = fT1[j]; t2 = fT2[j]; t3 = fT3[j]; tA = fTA[j]
            dFc_dT = float64(0.0)
            if t1 > EPS: dFc_dT += tA * (-float64(1.0) / t1) * math.exp(-T / t1)
            if t3 > EPS: dFc_dT += (float64(1.0) - tA) * (-float64(1.0) / t3) * math.exp(-T / t3)
            if t2 > EPS: dFc_dT += math.exp(-t2 / T) * (t2 / (T * T))

            fc_ok   = Fc > EPS
            dlFc_dT = (dFc_dT / Fc * INV_LN10) if fc_ok else float64(0.0)
            dCp_dT  = -float64(0.67) * dlFc_dT
            dNp_dT  = -float64(1.27) * dlFc_dT

            Prr       = Pr if Pr > EPS else EPS
            dlPr_dPr  = (float64(1.0) / (Prr * LN10)) if pr_ok else float64(0.0)
            lPr       = math.log10(Prr) if pr_ok else float64(0.0)
            nf1       = lPr + Cp
            df        = Np - float64(0.14) * nf1
            abs_df    = abs(df)
            df_s      = df if abs_df > EPS else (EPS if df >= float64(0.0) else -EPS)
            df_sq     = df_s * df_s
            df_ok     = abs_df > EPS

            df1_dPr = ((dlPr_dPr * df_s + float64(0.14) * nf1 * dlPr_dPr) / df_sq
                       if (df_ok and pr_ok) else float64(0.0))
            nT      = dCp_dT
            dT_     = dNp_dT - float64(0.14) * dCp_dT
            df1_dT  = (nT * df_s - nf1 * dT_) / df_sq if df_ok else float64(0.0)

            f1sq    = f1_v * f1_v
            de_dlF  = float64(1.0) / (float64(1.0) + f1sq)
            de_df1  = -float64(2.0) * f1_v * lFc / ((float64(1.0) + f1sq) ** 2)
            de_dT   = de_dlF * dlFc_dT + de_df1 * df1_dT
            de_dPr  = de_df1 * df1_dPr
            dF_dT   = F * LN10 * de_dT
            dF_dPr  = F * LN10 * de_dPr
        else:
            dF_dT  = float64(0.0)
            dF_dPr = float64(0.0)

        Pd       = (float64(1.0) + Pr) * (float64(1.0) + Pr)
        dkb_dT   = dki_dT * (Pr / (float64(1.0) + Pr)) + ki_s * (float64(1.0) / Pd) * dPr_dT
        dkb_vC   = ki_s * (float64(1.0) / Pd) * dPr_vC
        dkf_dT_v = dkb_dT * F + kb * (dF_dT + dF_dPr * dPr_dT)
        dkf_vC   = dkb_vC * F + kb * dF_dPr * dPr_vC

        # Contribution T
        dRR_v = (dkf_dT_v * vT * ce
                 + kf_v * fac * cp / (Kc * Kc) * dKc * vT)

        # Contribution C_{s_dir}
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
            # Contribution via dM → dPr → dkf
            dRR_v += dkf_vC * ce

        # Scatter
        for k in range(fRO[j], fRO[j + 1]):
            col_out[fRI[k]] -= fRS[k] * dRR_v
        for k in range(fPO[j], fPO[j + 1]):
            col_out[fPI[k]] += fPS[k] * dRR_v


# ---------------------------------------------------------------------------
#  Fonction principale
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _eval_jacobian(
    T, C,
    omega,          # vecteur omega au point (T,C), taille n_sp — déjà calculé
    J_out,          # sortie : matrice (n_state, n_state), row-major, stockée en global
    # ── Contexte cinétique (row-slices 1D) ───────────────────────────
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    # ── Données mécanisme ─────────────────────────────────────────────
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

    c7 = cuda.local.array(7, float64)

    ntot = float64(0.0)
    for s in range(n_sp):
        ntot += C[s]

    cv_mix = float64(0.0)
    su_om  = float64(0.0)
    cv_s_arr  = cuda.local.array(54, float64)
    u_s_arr   = cuda.local.array(54, float64)
    dcvdT_arr = cuda.local.array(54, float64)

    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        T2  = T * T; T3 = T2 * T; T4 = T3 * T
        cpR = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        cv_s_arr[s]  = R_CONST * (cpR - float64(1.0))
        dcvdT_arr[s] = R_CONST * (c7[1]
                                  + float64(2.0)*c7[2]*T
                                  + float64(3.0)*c7[3]*T2
                                  + float64(4.0)*c7[4]*T3)
        hRT        = (c7[0] + c7[1]*T/float64(2.0) + c7[2]*T2/float64(3.0)
                      + c7[3]*T3/float64(4.0) + c7[4]*T4/float64(5.0)
                      + c7[5]/T)
        u_s_arr[s] = R_CONST * T * hRT - R_CONST * T

        Xs = C[s] / ntot if ntot > EPS else float64(0.0)
        cv_mix += Xs * cv_s_arr[s]
        su_om  += u_s_arr[s] * omega[s]

    denom = ntot * cv_mix

    # Initialisation de J_out à zéro
    for i in range(n_state):
        for jj in range(n_state):
            J_out[i * n_state + jj] = float64(0.0)

    # Boucle principale sur les colonnes
    for j_col in range(n_state):

        _dchem_col(
            j_col, T, C,
            kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
            kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
            kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
            k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
            Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
            n_sp,
            n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
            n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
            n_f, fA0, fb0, fE0, fAi, fbi, fEi,
            fOW, fDn, fTA, fT1, fT2, fT3, fIT,
            fRI, fRS, fRO, fPI, fPS, fPO, fEff,
            col_chem,
        )

        for s in range(n_sp):
            J_out[(s + 1) * n_state + j_col] = col_chem[s]

        if abs(denom) > EPS:
            dsuom = float64(0.0)
            for s in range(n_sp):
                dsuom += u_s_arr[s] * col_chem[s]

            if j_col == 0:
                for s in range(n_sp):
                    dsuom += cv_s_arr[s] * omega[s]
                dcvmix_dT = float64(0.0)
                for s in range(n_sp):
                    Xs = C[s] / ntot if ntot > EPS else float64(0.0)
                    dcvmix_dT += Xs * dcvdT_arr[s]
                ddenom = ntot * dcvmix_dT
            else:
                s_dir = j_col - 1
                if ntot > EPS:
                    dcvmix_dC = (cv_s_arr[s_dir] - cv_mix) / ntot
                else:
                    dcvmix_dC = float64(0.0)
                ddenom = cv_mix + ntot * dcvmix_dC

            J_out[j_col] = -(dsuom * denom - su_om * ddenom) / (denom * denom)
        else:
            J_out[j_col] = float64(0.0)


# =============================================================================
#  Kernel CUDA global (un thread = un réacteur)
# =============================================================================

MAX_N_STATE = 54


@cuda.jit(cache=True)
def _jacobian_kernel_pt(
    TC_in,          # (B, n_state)  float64
    J_all,          # (B, n_state*n_state)  float64  — sortie
    # ── Contexte cinétique (B, n_reactions) ──────────────────────────
    ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e,
    ctx_sRre, ctx_sRpe,
    ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f,
    ctx_sRrf, ctx_sRpf,
    ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
    ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
    # ── Mécanisme ─────────────────────────────────────────────────────
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
    om  = cuda.local.array(MAX_N_STATE, float64)

    for s in range(n_sp):
        C[s] = TC_in[b, s + 1]

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

    _eval_jacobian(
        T_loc, C, om,
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
#  Wrapper Python — CUDAJacobianEvaluatorPT
# =============================================================================

class CUDAJacobianEvaluatorPT:
    """
    Évalue la jacobienne complète du RHS adiabatique via le kernel per-thread.

    Usage
    -----
    ev = CUDAJacobianEvaluatorPT(mech, dtype=np.float64)
    J  = ev.evaluate(T_arr, C_arr)   # T_arr : (B,), C_arr : (B, n_sp)
    # J : np.ndarray (B, n_state, n_state)

    J[b, i, j] = d(dTC_i/dt) / d(xk_j)  au point (T_arr[b], C_arr[b])
      i=0     → dT/dt,     j=0     → T
      i=s+1   → dC_s/dt,   j=s+1   → C_s
    """

    THREADS = 64

    def __init__(self, mech, dtype=np.float64):
        self._mech  = mech
        self._dtype = dtype
        self._n_sp  = mech.n_species
        self._n_state = self._n_sp + 1
        self._built = False

    def _alloc_context(self, B):
        dt  = self._dtype
        m   = self._mech
        e   = m.elementary
        tb  = m.threebody
        f   = m.falloff

        def z(shape): return cuda.device_array(shape, dtype=dt)

        # Utilisation de react_offset (ElementaryData) et react_offset (ThreeBodyData/FalloffData)
        n_e_flat  = e.react_offset[-1] if e.n_reactions else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1
        n_f_flat  = f.react_offset[-1] if f.n_reactions else 1

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
        nst   = self._n_state

        if not self._built or self._B_alloc != B:
            self._alloc_context(B)

        TC_in = np.empty((B, nst), dtype=self._dtype)
        TC_in[:, 0]  = T_arr
        TC_in[:, 1:] = C_arr
        d_TC = cuda.to_device(TC_in)

        d_J = cuda.device_array((B, nst * nst), dtype=self._dtype)

        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff
        cx = self._ctx

        threads = self.THREADS
        blocks  = (B + threads - 1) // threads

        _jacobian_kernel_pt[blocks, threads](
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
            # ThermoData : coef_low → thL, coef_high → thH, Tmid → thM
            m.thermo.coef_low, m.thermo.coef_high, m.thermo.Tmid,
            # ElementaryData
            e.n_reactions, e.A, e.n, e.Ea, e.oneWay,
            e.delta_n, e.react_idx, e.react_sto,
            e.react_offset, e.prod_idx, e.prod_sto,
            e.prod_offset,
            # ThreeBodyData
            tb.n_reactions, tb.A, tb.n, tb.Ea, tb.oneWay,
            tb.delta_n, tb.react_idx, tb.react_sto,
            tb.react_offset, tb.prod_idx, tb.prod_sto,
            tb.prod_offset, tb.eff_matrix,
            # FalloffData
            f.n_reactions, f.A_k0, f.b_k0, f.Ea_k0, f.A_kinf, f.b_kinf, f.Ea_kinf,
            f.oneWay, f.delta_n,
            f.troe_A, f.troe_T1, f.troe_T2, f.troe_T3, f.is_troe,
            f.react_idx, f.react_sto, f.react_offset,
            f.prod_idx,  f.prod_sto,  f.prod_offset,
            f.eff_matrix,
        )
        cuda.synchronize()

        J_flat = d_J.copy_to_host()
        return J_flat.reshape(B, nst, nst)