"""
kernels/per_block/jvp_pb.py — JVP du vecteur source chimique (per-block).

Architecture : 1 bloc = 1 réacteur. Les threads coopèrent.

Fonction exportée :
    _eval_jvp_pb(tid, n_threads, T, C, vT, vC, Jv_om, <ctx_slices>, <mech_args>)

Parallélisme :
    - Chaque thread traite une réaction dans sa boucle
    - Scatter atomique dans Jv_om
    - cuda.syncthreads() en fin de chaque section
    - Jv_om doit être en mémoire partagée et sera mis à zéro au début
"""

import math
from numba import cuda, float64

from ..shared.thermo import R_CONST, EPS, LN10, INV_LN10


@cuda.jit(device=True)
def _eval_jvp_pb(
    tid, n_threads,
    T, C, vT, vC, Jv_om,
    # ── Contexte cinétique ────────────────────────────────────────────
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
    """
    Version parallèle de _eval_jvp.
    Jv_om est en mémoire partagée, mis à zéro au début.
    """
    # Mise à zéro de Jv_om en parallèle
    i = tid
    while i < n_sp:
        Jv_om[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    # ── Élémentaires ──────────────────────────────────────────────────
    j = tid
    while j < n_e:
        kf  = kf_e[j]; Kc = Kc_e[j]
        cr  = cr_e[j]; cp = cp_e[j]; dG = dG0_e[j]
        ow  = eOW[j];  fac = 1.0 - ow
        dkf_dT = kf * (en[j] / T + eEa[j] / (R_CONST * T * T))
        dSr = float64(0.0)
        for k in range(eRO[j], eRO[j + 1]):
            dSr += eRS[k] * sR_re[k]
        dSp = float64(0.0)
        for k in range(ePO[j], ePO[j + 1]):
            dSp += ePS[k] * sR_pe[k]
        dlnKc = (dSp - dSr) / (R_CONST * T) + dG / (R_CONST * T * T) - eDn[j] / T
        dKc   = Kc * dlnKc
        dRR_v = (dkf_dT * (cr - fac * cp / Kc) + kf * fac * cp / (Kc * Kc) * dKc) * vT
        for k in range(eRO[j], eRO[j + 1]):
            sp = eRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf * cr * (eRS[k] / c) * vC[sp]
        for k in range(ePO[j], ePO[j + 1]):
            sp = ePI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf * fac / Kc * cp * (ePS[k] / c) * vC[sp]
        for k in range(eRO[j], eRO[j + 1]):
            cuda.atomic.add(Jv_om, eRI[k], -eRS[k] * dRR_v)
        for k in range(ePO[j], ePO[j + 1]):
            cuda.atomic.add(Jv_om, ePI[k],  ePS[k] * dRR_v)
        j += n_threads
    cuda.syncthreads()

    # ── Trois-corps ───────────────────────────────────────────────────
    j = tid
    while j < n_tb:
        kf  = kf_tb[j]; Kc = Kc_tb[j]
        cr  = cr_tb[j]; cp = cp_tb[j]; dG = dG0_tb[j]
        M   = M_tb[j];  br = bRR_tb[j]; ow = tbOW[j]; fac = 1.0 - ow
        dkf_dT = kf * (tbn[j] / T + tbEa[j] / (R_CONST * T * T))
        dSr = float64(0.0)
        for k in range(tbRO[j], tbRO[j + 1]):
            dSr += tbRS[k] * sR_rtb[k]
        dSp = float64(0.0)
        for k in range(tbPO[j], tbPO[j + 1]):
            dSp += tbPS[k] * sR_ptb[k]
        dlnKc = (dSp - dSr) / (R_CONST * T) + dG / (R_CONST * T * T) - tbDn[j] / T
        dKc   = Kc * dlnKc
        dbR_dT = dkf_dT * (cr - fac * cp / Kc) + kf * fac * cp / (Kc * Kc) * dKc
        dM_v = float64(0.0)
        for s in range(n_sp):
            dM_v += vC[s] * tbEff[j * n_sp + s]
        dRR_v = dbR_dT * vT * M
        for k in range(tbRO[j], tbRO[j + 1]):
            sp = tbRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf * cr * (tbRS[k] / c) * vC[sp] * M
        for k in range(tbPO[j], tbPO[j + 1]):
            sp = tbPI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf * fac / Kc * cp * (tbPS[k] / c) * vC[sp] * M
        dRR_v += br * dM_v
        for k in range(tbRO[j], tbRO[j + 1]):
            cuda.atomic.add(Jv_om, tbRI[k], -tbRS[k] * dRR_v)
        for k in range(tbPO[j], tbPO[j + 1]):
            cuda.atomic.add(Jv_om, tbPI[k],  tbPS[k] * dRR_v)
        j += n_threads
    cuda.syncthreads()

    # ── Falloff ───────────────────────────────────────────────────────
    j = tid
    while j < n_f:
        kf_v = kf_f[j];  Kc  = Kc_f[j]
        cr   = cr_f[j];  cp  = cp_f[j];  dG  = dG0_f[j]
        k0_v = k0_f[j];  ki  = kinf_f[j]; M  = M_f[j]
        Pr   = Pr_f[j];  F   = F_f[j];   kb  = kb_f[j]
        Fc   = Fcent_f[j]; lFc = lFcent_f[j]
        Cp   = Cpar_f[j];  Np  = Npar_f[j]
        f1_v = f1_f[j];   ex  = exp_f[j]
        ow   = fOW[j]; fac = 1.0 - ow
        ki_s = ki if ki > EPS else EPS
        pr_ok = (Pr > EPS) and (M > EPS)
        dk0_dT = k0_v * (fb0[j] / T + fE0[j] / (R_CONST * T * T))
        dki_dT = ki   * (fbi[j] / T + fEi[j] / (R_CONST * T * T))
        dM_v = float64(0.0)
        for s in range(n_sp):
            dM_v += vC[s] * fEff[j * n_sp + s]
        if pr_ok:
            dPr_dT = (dk0_dT * M - k0_v * M * dki_dT / ki_s) / ki_s
            dPr_vC = (k0_v / ki_s) * dM_v
        else:
            dPr_dT = float64(0.0); dPr_vC = float64(0.0)
        dSr = float64(0.0)
        for k in range(fRO[j], fRO[j + 1]):
            dSr += fRS[k] * sR_rf[k]
        dSp = float64(0.0)
        for k in range(fPO[j], fPO[j + 1]):
            dSp += fPS[k] * sR_pf[k]
        dlnKc = (dSp - dSr) / (R_CONST * T) + dG / (R_CONST * T * T) - fDn[j] / T
        dKc   = Kc * dlnKc
        ce = cr - fac * cp / Kc
        if fIT[j]:
            t1 = fT1[j]; t2 = fT2[j]; t3 = fT3[j]; tA = fTA[j]
            dFc_dT = float64(0.0)
            if t1 > EPS: dFc_dT += tA * (-1.0 / t1) * math.exp(-T / t1)
            if t3 > EPS: dFc_dT += (1.0 - tA) * (-1.0 / t3) * math.exp(-T / t3)
            if t2 > EPS: dFc_dT += math.exp(-t2 / T) * (t2 / (T * T))
            fc_ok = Fc > EPS
            dlFc_dT = (dFc_dT / Fc * INV_LN10) if fc_ok else float64(0.0)
            dCp_dT = -0.67 * dlFc_dT; dNp_dT = -1.27 * dlFc_dT
            Prr   = Pr if Pr > EPS else EPS
            dlPr_dPr = (1.0 / (Prr * LN10)) if pr_ok else float64(0.0)
            lPr   = math.log10(Prr) if pr_ok else float64(0.0)
            nf1   = lPr + Cp
            df    = Np - 0.14 * nf1
            abs_df = abs(df)
            df_s  = df if abs_df > EPS else (EPS if df >= 0.0 else -EPS)
            df_sq = df_s * df_s; df_ok = abs_df > EPS
            df1_dPr = ((dlPr_dPr * df_s + 0.14 * nf1 * dlPr_dPr) / df_sq
                       if (df_ok and pr_ok) else float64(0.0))
            nT    = dCp_dT; dT_ = dNp_dT - 0.14 * dCp_dT
            df1_dT = (nT * df_s - nf1 * dT_) / df_sq if df_ok else float64(0.0)
            f1sq  = f1_v * f1_v
            de_dlF = 1.0 / (1.0 + f1sq)
            de_df1 = -2.0 * f1_v * lFc / ((1.0 + f1sq) ** 2)
            de_dT  = de_dlF * dlFc_dT + de_df1 * df1_dT
            de_dPr = de_df1 * df1_dPr
            dF_dT  = F * LN10 * de_dT
            dF_dPr = F * LN10 * de_dPr
        else:
            dF_dT = float64(0.0); dF_dPr = float64(0.0)
        Pd  = (1.0 + Pr) * (1.0 + Pr)
        dkb_dT = dki_dT * (Pr / (1.0 + Pr)) + ki_s * (1.0 / Pd) * dPr_dT
        dkb_vC = ki_s * (1.0 / Pd) * dPr_vC
        dkf_dT_v = dkb_dT * F + kb * (dF_dT + dF_dPr * dPr_dT)
        dkf_vC   = dkb_vC * F + kb * dF_dPr * dPr_vC
        dRR_v = dkf_dT_v * vT * ce + kf_v * fac * cp / (Kc * Kc) * dKc * vT
        for k in range(fRO[j], fRO[j + 1]):
            sp = fRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf_v * cr * (fRS[k] / c) * vC[sp]
        for k in range(fPO[j], fPO[j + 1]):
            sp = fPI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf_v * fac / Kc * cp * (fPS[k] / c) * vC[sp]
        dRR_v += dkf_vC * ce
        for k in range(fRO[j], fRO[j + 1]):
            cuda.atomic.add(Jv_om, fRI[k], -fRS[k] * dRR_v)
        for k in range(fPO[j], fPO[j + 1]):
            cuda.atomic.add(Jv_om, fPI[k],  fPS[k] * dRR_v)
        j += n_threads
    cuda.syncthreads()