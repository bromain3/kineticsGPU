"""
kernels/per_block/jacobian_pb.py — Jacobienne analytique complète du RHS adiabatique (per-block).

Architecture : 1 bloc = 1 réacteur. Les 64 threads coopèrent.

Fonction exportée :
    _eval_jacobian_pb(tid, n_threads, T, C, omega,
                      J_out,
                      <ctx_slices>, n_sp, <mech_args>)

Calcule la même matrice jacobienne J (n_state × n_state) que _eval_jacobian
(per_thread/jacobian.py), mais en parallèle :

    J[i, j] = d(dTC_i/dt) / d(xk_j)

Parallélisme :
    - La boucle sur les colonnes j_col = 0 .. n_state-1 est distribuée
      en stride tid/n_threads : chaque thread prend les colonnes
      j_col ≡ tid (mod n_threads).
    - À l'intérieur de _dchem_col_pb, les boucles sur les réactions sont
      également distribuées en stride, avec cuda.atomic.add dans J_out.
    - La ligne 0 (dérivée de dT/dt) est calculée par chaque thread pour
      ses colonnes propres, sans conflit d'écriture.
    - cuda.syncthreads() est placé aux points de synchronisation nécessaires.

J_out est en mémoire globale (allouée par l'appelant, shape (B, n_state*n_state)
ou équivalent 2D).  Accès : J_out[i * n_state + j].

Budget mémoire partagée supplémentaire :
    - sh_props : 3 × n_sp × 8B (cv_s, u_s, dcvdT_s) ≈ 3×54×8 = 1 296 B
    - sh_ntot, sh_cvmix, sh_suom : 3 × 8B (scalaires partagés)
    - sh_scratch : THREADS_PER_BLOCK × 8B = 512 B
    Total ≈ 1 814 B — tient aisément dans le budget du bloc.

Aucune simplification physique : même algèbre que context_pb.py / jvp_pb.py.
"""

import math
import numpy as np
from numba import cuda, float64

from ..shared.thermo import (
    R_CONST, PREF, EPS, LN10, INV_LN10,
    _load_coef7, _h_RT, _s_R, _cp_R,
)
from .reductions import _block_reduce_sum
from .context_pb import _eval_rhs_and_context_pb

_MAX_SP = 53


# ---------------------------------------------------------------------------
#  Helper interne : colonne j_col de d(omega)/d(xk), parallèle sur les réactions
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _dchem_col_pb(
    tid, n_threads,
    j_col,
    T, C,
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
    col_out,
):
    if j_col == 0:
        vT = float64(1.0)
    else:
        vT = float64(0.0)

    # ── Élémentaires ──────────────────────────────────────────────────
    j = tid
    while j < n_e:
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

        dRR_v = (dkf_dT * (cr - fac * cp / Kc)
                 + kf * fac * cp / (Kc * Kc) * dKc) * vT

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

        for k in range(eRO[j], eRO[j + 1]):
            cuda.atomic.add(col_out, eRI[k], -eRS[k] * dRR_v)
        for k in range(ePO[j], ePO[j + 1]):
            cuda.atomic.add(col_out, ePI[k],  ePS[k] * dRR_v)

        j += n_threads
    cuda.syncthreads()

    # ── Trois-corps ───────────────────────────────────────────────────
    j = tid
    while j < n_tb:
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

        if j_col == 0:
            dM_v = float64(0.0)
        else:
            s_dir = j_col - 1
            dM_v  = tbEff[j * n_sp + s_dir]

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
            dRR_v += br * dM_v

        for k in range(tbRO[j], tbRO[j + 1]):
            cuda.atomic.add(col_out, tbRI[k], -tbRS[k] * dRR_v)
        for k in range(tbPO[j], tbPO[j + 1]):
            cuda.atomic.add(col_out, tbPI[k],  tbPS[k] * dRR_v)

        j += n_threads
    cuda.syncthreads()

    # ── Falloff ───────────────────────────────────────────────────────
    j = tid
    while j < n_f:
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

            Prr      = Pr if Pr > EPS else EPS
            dlPr_dPr = (float64(1.0) / (Prr * LN10)) if pr_ok else float64(0.0)
            lPr      = math.log10(Prr) if pr_ok else float64(0.0)
            nf1      = lPr + Cp
            df       = Np - float64(0.14) * nf1
            abs_df   = abs(df)
            df_s     = df if abs_df > EPS else (EPS if df >= float64(0.0) else -EPS)
            df_sq    = df_s * df_s
            df_ok    = abs_df > EPS

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
            cuda.atomic.add(col_out, fRI[k], -fRS[k] * dRR_v)
        for k in range(fPO[j], fPO[j + 1]):
            cuda.atomic.add(col_out, fPI[k],  fPS[k] * dRR_v)

        j += n_threads
    cuda.syncthreads()


# ---------------------------------------------------------------------------
#  Fonction principale per-block
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _eval_jacobian_pb(
    tid, n_threads,
    T, C,
    omega,
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
    sh_col, sh_props, sh_scalars, sh_scratch,
):
    n_state = n_sp + 1
    c7      = cuda.local.array(7, float64)

    # Initialisation de J_out à zéro
    idx = tid
    while idx < n_state * n_state:
        J_out[idx] = float64(0.0)
        idx += n_threads
    cuda.syncthreads()

    # Calcul des propriétés thermo par espèce
    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        T2  = T * T; T3 = T2 * T; T4 = T3 * T
        cpR = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        sh_props[i]              = R_CONST * (cpR - float64(1.0))
        sh_props[2 * n_sp + i]   = R_CONST * (c7[1]
                                               + float64(2.0)*c7[2]*T
                                               + float64(3.0)*c7[3]*T2
                                               + float64(4.0)*c7[4]*T3)
        hRT = (c7[0] + c7[1]*T/float64(2.0) + c7[2]*T2/float64(3.0)
               + c7[3]*T3/float64(4.0) + c7[4]*T4/float64(5.0)
               + c7[5]/T)
        sh_props[n_sp + i] = R_CONST * T * hRT - R_CONST * T
        i += n_threads
    cuda.syncthreads()

    # Réductions : ntot, cv_mix, su_om
    acc_n = float64(0.0)
    i = tid
    while i < n_sp:
        acc_n += C[i]
        i += n_threads
    _block_reduce_sum(acc_n, sh_scratch, tid, n_threads)
    cuda.syncthreads()
    if tid == 0:
        sh_scalars[0] = sh_scratch[0]
    cuda.syncthreads()
    ntot = sh_scalars[0]

    acc_cv = float64(0.0); acc_su = float64(0.0)
    i = tid
    while i < n_sp:
        Xs = C[i] / ntot if ntot > EPS else float64(0.0)
        acc_cv += Xs * sh_props[i]
        acc_su += sh_props[n_sp + i] * omega[i]
        i += n_threads
    _block_reduce_sum(acc_cv, sh_scratch, tid, n_threads)
    cuda.syncthreads()
    if tid == 0:
        sh_scalars[1] = sh_scratch[0]
    cuda.syncthreads()
    _block_reduce_sum(acc_su, sh_scratch, tid, n_threads)
    cuda.syncthreads()
    if tid == 0:
        sh_scalars[2] = sh_scratch[0]
        sh_scalars[3] = sh_scalars[0] * sh_scalars[1]
    cuda.syncthreads()

    ntot   = sh_scalars[0]
    cv_mix = sh_scalars[1]
    su_om  = sh_scalars[2]
    denom  = sh_scalars[3]

    for j_col in range(n_state):

        i = tid
        while i < n_sp:
            sh_col[i] = float64(0.0)
            i += n_threads
        cuda.syncthreads()

        _dchem_col_pb(
            tid, n_threads,
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
            sh_col,
        )

        i = tid
        while i < n_sp:
            J_out[(i + 1) * n_state + j_col] = sh_col[i]
            i += n_threads
        cuda.syncthreads()

        if tid == 0:
            if abs(denom) > EPS:
                dsuom = float64(0.0)
                for s in range(n_sp):
                    dsuom += sh_props[n_sp + s] * sh_col[s]

                if j_col == 0:
                    for s in range(n_sp):
                        dsuom += sh_props[s] * omega[s]
                    dcvmix_dT = float64(0.0)
                    for s in range(n_sp):
                        Xs = C[s] / ntot if ntot > EPS else float64(0.0)
                        dcvmix_dT += Xs * sh_props[2 * n_sp + s]
                    ddenom = ntot * dcvmix_dT
                else:
                    s_dir = j_col - 1
                    if ntot > EPS:
                        dcvmix_dC = (sh_props[s_dir] - cv_mix) / ntot
                    else:
                        dcvmix_dC = float64(0.0)
                    ddenom = cv_mix + ntot * dcvmix_dC

                J_out[j_col] = -(dsuom * denom - su_om * ddenom) / (denom * denom)
            else:
                J_out[j_col] = float64(0.0)
        cuda.syncthreads()


# =============================================================================
#  Constantes locales
# =============================================================================

_MAX_N_STATE       = 54
_MAX_N_STATE_3     = _MAX_N_STATE * 3
_MAX_R_p_1         = 21
_THREADS_PER_BLOCK = 64


# =============================================================================
#  Kernel CUDA global (un bloc = un réacteur)
# =============================================================================

@cuda.jit(cache=True)
def _jacobian_kernel_pb(
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
    b         = cuda.blockIdx.x
    tid       = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1

    sh_xk      = cuda.shared.array(_MAX_N_STATE,       float64)
    sh_om      = cuda.shared.array(_MAX_N_STATE,       float64)
    sh_col     = cuda.shared.array(_MAX_N_STATE,       float64)
    sh_props   = cuda.shared.array(_MAX_N_STATE_3,  float64)
    sh_scalars = cuda.shared.array(4,                  float64)
    sh_scratch = cuda.shared.array(_THREADS_PER_BLOCK, float64)

    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()

    T_loc = sh_xk[0]

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

    _eval_jacobian_pb(
        tid, n_threads,
        T_loc, sh_xk[1:], sh_om,
        J_all[b],
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
        sh_col, sh_props, sh_scalars, sh_scratch,
    )


# =============================================================================
#  Wrapper Python — CUDAJacobianEvaluatorPB
# =============================================================================

class CUDAJacobianEvaluatorPB:
    """
    Évalue la jacobienne complète du RHS adiabatique via le kernel per-block.

    Usage
    -----
    ev = CUDAJacobianEvaluatorPB(mech, dtype=np.float64)
    J  = ev.evaluate(T_arr, C_arr)   # T_arr : (B,), C_arr : (B, n_sp)
    # J : np.ndarray (B, n_state, n_state)

    J[b, i, j] = d(dTC_i/dt) / d(xk_j)  au point (T_arr[b], C_arr[b])
      i=0     → dT/dt,     j=0     → T
      i=s+1   → dC_s/dt,   j=s+1   → C_s
    """

    def __init__(self, mech, dtype=np.float64):
        self._mech    = mech
        self._dtype   = dtype
        self._n_sp    = mech.n_species
        self._n_state = self._n_sp + 1
        self._built   = False

    def _alloc_context(self, B):
        dt  = self._dtype
        m   = self._mech
        e   = m.elementary
        tb  = m.threebody
        f   = m.falloff

        def z(shape): return cuda.device_array(shape, dtype=dt)

        # Utilisation de react_offset (présent dans les trois dataclasses)
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

        _jacobian_kernel_pb[B, _THREADS_PER_BLOCK](
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