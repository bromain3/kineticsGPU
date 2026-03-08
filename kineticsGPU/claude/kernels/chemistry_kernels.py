"""
chemistry_kernels.py — Production CUDA per-thread backward-Euler integrator.

Architecture: one CUDA thread = one reactor.

Memory layout (faithful to the original fast version):
  - Chemistry context (kf, Kc, conc products, M, sR, etc.) lives in
    PRE-ALLOCATED GLOBAL BUFFERS (B × n_reactions).  Although global memory
    has higher latency than registers, these arrays are too large to fit in
    registers without catastrophic spilling.  L2 cache reuse within a single
    step amortises the cost.
  - GMRES scratch (V, H, cs, sn, g, w, y) also in global memory.
  - Small per-step scalars (T, h, res_norm, …) stay in registers.

What changed vs the original v5:
  - LU path, timing/profiling, reuse_jacobian: entirely removed.
  - h passed as scalar (no _d_h_arr broadcast kernel).
  - Predictor (explicit Euler) + true-residual check in GMRES: kept.
  - Argument count reduced ~150 -> ~80 by removing dead parameters.
  - Scatter index arrays (idx_r/p_scatter) removed — unused in kernels.
  - Compilation is faster because no dead branches remain.
"""

import math
import numpy as np
from numba import cuda, float64, int32

R_CONST  = 8.314_462_618_153_24
PREF     = 101325.0
EPS      = 1e-300
LN10     = math.log(10.0)
INV_LN10 = 1.0 / LN10

MAX_N_STATE       = 54   # GRI30: 53 species + T
THREADS_PER_BLOCK = 64


# =============================================================================
#  Thermodynamics helpers
# =============================================================================

@cuda.jit(device=True, inline=True)
def _load_coef7(T, sp, Tmid, coef_low, coef_high, out7):
    base = sp * 7
    arr  = coef_low if T <= Tmid[sp] else coef_high
    for k in range(7):
        out7[k] = arr[base + k]


@cuda.jit(device=True, inline=True)
def _h_RT(T, c7):
    T2 = T*T; T3 = T2*T; T4 = T3*T
    return (c7[0] + c7[1]*T/2.0 + c7[2]*T2/3.0
            + c7[3]*T3/4.0 + c7[4]*T4/5.0 + c7[5]/T)


@cuda.jit(device=True, inline=True)
def _s_R(T, c7):
    T2 = T*T; T3 = T2*T; T4 = T3*T
    return (c7[0]*math.log(T) + c7[1]*T + c7[2]*T2/2.0
            + c7[3]*T3/3.0 + c7[4]*T4/4.0 + c7[6])


@cuda.jit(device=True, inline=True)
def _cp_R(T, c7):
    T2 = T*T; T3 = T2*T; T4 = T3*T
    return c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4


@cuda.jit(device=True, inline=True)
def _kf_arr(T, A, n, Ea):
    return A * (T**n) * math.exp(-Ea / (R_CONST * T))


@cuda.jit(device=True, inline=True)
def _log_conc_prod(C, i0, i1, sp_idx, sto):
    s = float64(0.0)
    for k in range(i0, i1):
        c = C[sp_idx[k]]
        s += sto[k] * math.log(c if c > EPS else EPS)
    return s


# =============================================================================
#  Full RHS — writes omega + all context into global row-slices
# =============================================================================

@cuda.jit(device=True)
def _eval_rhs_and_context(
    T, C,
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    omega,
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    c7 = cuda.local.array(7, float64)
    for s in range(n_sp):
        omega[s] = float64(0.0)

    # Elementary
    for j in range(n_e):
        kf = _kf_arr(T, eA[j], en[j], eEa[j])
        kf_e[j] = kf
        dG = float64(0.0)
        for k in range(eRO[j], eRO[j+1]):
            sp = eRI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_re[k] = s_s
            dG -= eRS[k] * (h_s - T*s_s)
        for k in range(ePO[j], ePO[j+1]):
            sp = ePI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_pe[k] = s_s
            dG += ePS[k] * (h_s - T*s_s)
        dG0_e[j] = dG
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** eDn[j])
        Kc_e[j] = Kc
        lr = _log_conc_prod(C, eRO[j], eRO[j+1], eRI, eRS)
        lp = _log_conc_prod(C, ePO[j], ePO[j+1], ePI, ePS)
        cr = math.exp(lr); cp = math.exp(lp)
        cr_e[j] = cr; cp_e[j] = cp
        ow = eOW[j]
        RR = kf * (cr - (1.0 - ow) * cp / Kc)
        for k in range(eRO[j], eRO[j+1]):
            omega[eRI[k]] -= eRS[k] * RR
        for k in range(ePO[j], ePO[j+1]):
            omega[ePI[k]] += ePS[k] * RR

    # Three-body
    for j in range(n_tb):
        kf = _kf_arr(T, tbA[j], tbn[j], tbEa[j])
        kf_tb[j] = kf
        dG = float64(0.0)
        for k in range(tbRO[j], tbRO[j+1]):
            sp = tbRI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_rtb[k] = s_s
            dG -= tbRS[k] * (h_s - T*s_s)
        for k in range(tbPO[j], tbPO[j+1]):
            sp = tbPI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_ptb[k] = s_s
            dG += tbPS[k] * (h_s - T*s_s)
        dG0_tb[j] = dG
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** tbDn[j])
        Kc_tb[j] = Kc
        lr = _log_conc_prod(C, tbRO[j], tbRO[j+1], tbRI, tbRS)
        lp = _log_conc_prod(C, tbPO[j], tbPO[j+1], tbPI, tbPS)
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
        for k in range(tbRO[j], tbRO[j+1]):
            omega[tbRI[k]] -= tbRS[k] * RR
        for k in range(tbPO[j], tbPO[j+1]):
            omega[tbPI[k]] += tbPS[k] * RR

    # Falloff
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
            if t1 > EPS: Fc += tA * math.exp(-T/t1)
            if t3 > EPS: Fc += (1.0 - tA) * math.exp(-T/t3)
            if t2 > EPS: Fc += math.exp(-t2/T)
            if Fc < EPS: Fc = EPS
            lFc   = math.log10(Fc)
            Cp    = -0.4 - 0.67*lFc
            Np    =  0.75 - 1.27*lFc
            Prr   = Pr if Pr > EPS else EPS
            logPr = math.log10(Prr)
            num   = logPr + Cp
            den   = Np - 0.14*(logPr + Cp)
            if abs(den) < EPS: den = EPS if den >= 0.0 else -EPS
            f1_v  = num / den
            ex    = lFc / (1.0 + f1_v*f1_v)
            F_v   = 10.0**ex
        else:
            Fc = float64(1.0); lFc = float64(0.0)
            Cp = float64(0.0); Np  = float64(1.0)
            f1_v = float64(0.0); ex = float64(0.0); F_v = float64(1.0)
        Fcent_f[j] = Fc; lFcent_f[j] = lFc
        Cpar_f[j]  = Cp; Npar_f[j]   = Np
        f1_f[j]    = f1_v; exp_f[j]  = ex; F_f[j] = F_v
        kbase = ki_s * (Pr / (1.0 + Pr))
        kf_v  = kbase * F_v
        kb_f[j] = kbase; kf_f[j] = kf_v
        dG = float64(0.0)
        for k in range(fRO[j], fRO[j+1]):
            sp = fRI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_rf[k] = s_s
            dG -= fRS[k] * (h_s - T*s_s)
        for k in range(fPO[j], fPO[j+1]):
            sp = fPI[k]
            _load_coef7(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT(T, c7)
            s_s = R_CONST * _s_R(T, c7)
            sR_pf[k] = s_s
            dG += fPS[k] * (h_s - T*s_s)
        dG0_f[j] = dG
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** fDn[j])
        Kc_f[j] = Kc
        lr = _log_conc_prod(C, fRO[j], fRO[j+1], fRI, fRS)
        lp = _log_conc_prod(C, fPO[j], fPO[j+1], fPI, fPS)
        cr = math.exp(lr); cp = math.exp(lp)
        cr_f[j] = cr; cp_f[j] = cp
        ow = fOW[j]
        RR = kf_v * (cr - (1.0 - ow) * cp / Kc)
        for k in range(fRO[j], fRO[j+1]):
            omega[fRI[k]] -= fRS[k] * RR
        for k in range(fPO[j], fPO[j+1]):
            omega[fPI[k]] += fPS[k] * RR


# =============================================================================
#  Adiabatic RHS
# =============================================================================

@cuda.jit(device=True)
def _adiabatic_rhs(T, C, omega, n_sp, thL, thH, thM, dTCdt):
    c7   = cuda.local.array(7, float64)
    ntot = float64(0.0)
    for s in range(n_sp): ntot += C[s]
    cv_mix = float64(0.0); su_om = float64(0.0)
    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        cp_r = _cp_R(T, c7)
        cv_s = R_CONST * (cp_r - 1.0)
        hRT  = _h_RT(T, c7)
        u_s  = R_CONST * T * hRT - R_CONST * T
        Xs   = C[s] / ntot if ntot > EPS else float64(0.0)
        cv_mix += Xs * cv_s
        su_om  += u_s * omega[s]
    denom    = ntot * cv_mix
    dTCdt[0] = -su_om / denom if abs(denom) > EPS else float64(0.0)
    for s in range(n_sp):
        dTCdt[s+1] = omega[s]


# =============================================================================
#  JVP — reads context from global row-slices
# =============================================================================

@cuda.jit(device=True)
def _eval_jvp(
    T, C, vT, vC, Jv_om,
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
    for s in range(n_sp):
        Jv_om[s] = float64(0.0)

    # Elementary
    for j in range(n_e):
        kf = kf_e[j]; Kc = Kc_e[j]
        cr = cr_e[j]; cp = cp_e[j]; dG = dG0_e[j]
        ow = eOW[j]; fac = 1.0 - ow
        dkf_dT = kf * (en[j]/T + eEa[j]/(R_CONST*T*T))
        dSr = float64(0.0)
        for k in range(eRO[j], eRO[j+1]):
            dSr += eRS[k] * sR_re[k]
        dSp = float64(0.0)
        for k in range(ePO[j], ePO[j+1]):
            dSp += ePS[k] * sR_pe[k]
        dlnKc = (dSp - dSr)/(R_CONST*T) + dG/(R_CONST*T*T) - eDn[j]/T
        dKc   = Kc * dlnKc
        dRR_v = (dkf_dT*(cr - fac*cp/Kc) + kf*fac*cp/(Kc*Kc)*dKc) * vT
        for k in range(eRO[j], eRO[j+1]):
            sp = eRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf * cr * (eRS[k]/c) * vC[sp]
        for k in range(ePO[j], ePO[j+1]):
            sp = ePI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf * fac / Kc * cp * (ePS[k]/c) * vC[sp]
        for k in range(eRO[j], eRO[j+1]):
            Jv_om[eRI[k]] -= eRS[k] * dRR_v
        for k in range(ePO[j], ePO[j+1]):
            Jv_om[ePI[k]] += ePS[k] * dRR_v

    # Three-body
    for j in range(n_tb):
        kf = kf_tb[j]; Kc = Kc_tb[j]
        cr = cr_tb[j]; cp = cp_tb[j]; dG = dG0_tb[j]
        M  = M_tb[j];  br = bRR_tb[j]; ow = tbOW[j]; fac = 1.0 - ow
        dkf_dT = kf * (tbn[j]/T + tbEa[j]/(R_CONST*T*T))
        dSr = float64(0.0)
        for k in range(tbRO[j], tbRO[j+1]):
            dSr += tbRS[k] * sR_rtb[k]
        dSp = float64(0.0)
        for k in range(tbPO[j], tbPO[j+1]):
            dSp += tbPS[k] * sR_ptb[k]
        dlnKc = (dSp - dSr)/(R_CONST*T) + dG/(R_CONST*T*T) - tbDn[j]/T
        dKc   = Kc * dlnKc
        dbR_dT = dkf_dT*(cr - fac*cp/Kc) + kf*fac*cp/(Kc*Kc)*dKc
        dM_v   = float64(0.0)
        for s in range(n_sp):
            dM_v += vC[s] * tbEff[j*n_sp + s]
        dRR_v = dbR_dT * vT * M
        for k in range(tbRO[j], tbRO[j+1]):
            sp = tbRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf * cr * (tbRS[k]/c) * vC[sp] * M
        for k in range(tbPO[j], tbPO[j+1]):
            sp = tbPI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf * fac / Kc * cp * (tbPS[k]/c) * vC[sp] * M
        dRR_v += br * dM_v
        for k in range(tbRO[j], tbRO[j+1]):
            Jv_om[tbRI[k]] -= tbRS[k] * dRR_v
        for k in range(tbPO[j], tbPO[j+1]):
            Jv_om[tbPI[k]] += tbPS[k] * dRR_v

    # Falloff
    for j in range(n_f):
        kf_v = kf_f[j]; Kc = Kc_f[j]
        cr   = cr_f[j]; cp = cp_f[j]; dG = dG0_f[j]
        k0_v = k0_f[j]; ki = kinf_f[j]; M  = M_f[j]
        Pr   = Pr_f[j]; F  = F_f[j];  kb  = kb_f[j]
        Fc   = Fcent_f[j]; lFc = lFcent_f[j]
        Cp   = Cpar_f[j];  Np  = Npar_f[j]
        f1_v = f1_f[j];    ex  = exp_f[j]
        ow   = fOW[j]; fac = 1.0 - ow
        ki_s = ki if ki > EPS else EPS
        pr_ok = (Pr > EPS) and (M > EPS)
        dk0_dT = k0_v * (fb0[j]/T + fE0[j]/(R_CONST*T*T))
        dki_dT = ki   * (fbi[j]/T + fEi[j]/(R_CONST*T*T))
        dM_v   = float64(0.0)
        for s in range(n_sp):
            dM_v += vC[s] * fEff[j*n_sp + s]
        if pr_ok:
            dPr_dT = (dk0_dT*M - k0_v*M*dki_dT/ki_s) / ki_s
            dPr_vC = (k0_v/ki_s) * dM_v
        else:
            dPr_dT = float64(0.0); dPr_vC = float64(0.0)
        dSr = float64(0.0)
        for k in range(fRO[j], fRO[j+1]):
            dSr += fRS[k] * sR_rf[k]
        dSp = float64(0.0)
        for k in range(fPO[j], fPO[j+1]):
            dSp += fPS[k] * sR_pf[k]
        dlnKc = (dSp - dSr)/(R_CONST*T) + dG/(R_CONST*T*T) - fDn[j]/T
        dKc   = Kc * dlnKc
        ce = cr - fac*cp/Kc
        if fIT[j]:
            t1 = fT1[j]; t2 = fT2[j]; t3 = fT3[j]; tA = fTA[j]
            dFc_dT = float64(0.0)
            if t1 > EPS: dFc_dT += tA*(-1.0/t1)*math.exp(-T/t1)
            if t3 > EPS: dFc_dT += (1.0-tA)*(-1.0/t3)*math.exp(-T/t3)
            if t2 > EPS: dFc_dT += math.exp(-t2/T)*(t2/(T*T))
            fc_ok = Fc > EPS
            dlFc_dT = (dFc_dT/Fc*INV_LN10) if fc_ok else float64(0.0)
            dCp_dT = -0.67*dlFc_dT; dNp_dT = -1.27*dlFc_dT
            Prr = Pr if Pr > EPS else EPS
            dlPr_dPr = (1.0/(Prr*LN10)) if pr_ok else float64(0.0)
            lPr = math.log10(Prr) if pr_ok else float64(0.0)
            nf1 = lPr + Cp
            df  = Np - 0.14*nf1
            abs_df = abs(df)
            df_s = df if abs_df > EPS else (EPS if df >= 0.0 else -EPS)
            df_sq = df_s*df_s; df_ok = abs_df > EPS
            df1_dPr = ((dlPr_dPr*df_s + 0.14*nf1*dlPr_dPr)/df_sq
                       if (df_ok and pr_ok) else float64(0.0))
            nT = dCp_dT; dT_ = dNp_dT - 0.14*dCp_dT
            df1_dT = (nT*df_s - nf1*dT_)/df_sq if df_ok else float64(0.0)
            f1sq = f1_v*f1_v
            de_dlF = 1.0/(1.0 + f1sq)
            de_df1 = -2.0*f1_v*lFc / ((1.0+f1sq)**2)
            de_dT  = de_dlF*dlFc_dT + de_df1*df1_dT
            de_dPr = de_df1*df1_dPr
            dF_dT  = F*LN10*de_dT
            dF_dPr = F*LN10*de_dPr
        else:
            dF_dT = float64(0.0); dF_dPr = float64(0.0)
        Pd  = (1.0+Pr)*(1.0+Pr)
        dkb_dT = dki_dT*(Pr/(1.0+Pr)) + ki_s*(1.0/Pd)*dPr_dT
        dkb_vC = ki_s*(1.0/Pd)*dPr_vC
        dkf_dT_v = dkb_dT*F + kb*(dF_dT + dF_dPr*dPr_dT)
        dkf_vC   = dkb_vC*F + kb*dF_dPr*dPr_vC
        dRR_v = dkf_dT_v*vT*ce + kf_v*fac*cp/(Kc*Kc)*dKc*vT
        for k in range(fRO[j], fRO[j+1]):
            sp = fRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf_v * cr * (fRS[k]/c) * vC[sp]
        for k in range(fPO[j], fPO[j+1]):
            sp = fPI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf_v * fac / Kc * cp * (fPS[k]/c) * vC[sp]
        dRR_v += dkf_vC * ce
        for k in range(fRO[j], fRO[j+1]):
            Jv_om[fRI[k]] -= fRS[k] * dRR_v
        for k in range(fPO[j], fPO[j+1]):
            Jv_om[fPI[k]] += fPS[k] * dRR_v


# =============================================================================
#  Adiabatic JVP
# =============================================================================

@cuda.jit(device=True)
def _adiabatic_jvp(T, C, vT, vC, omega, Jv_om, n_sp, thL, thH, thM, Jv_TC):
    c7   = cuda.local.array(7, float64)
    ntot = float64(0.0)
    for s in range(n_sp): ntot += C[s]
    dntot = float64(0.0)
    for s in range(n_sp): dntot += vC[s]
    cv_mix = float64(0.0); dcv = float64(0.0)
    up     = float64(0.0); dup = float64(0.0)
    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        T2 = T*T; T3 = T2*T; T4 = T3*T
        cpR   = c7[0]+c7[1]*T+c7[2]*T2+c7[3]*T3+c7[4]*T4
        cv_s  = R_CONST*(cpR - 1.0)
        dcvdT = R_CONST*(c7[1]+2.0*c7[2]*T+3.0*c7[3]*T2+4.0*c7[4]*T3)
        hRT   = (c7[0]+c7[1]*T/2.0+c7[2]*T2/3.0
                 +c7[3]*T3/4.0+c7[4]*T4/5.0+c7[5]/T)
        u_s   = R_CONST*T*hRT - R_CONST*T
        Xs    = C[s]/ntot  if ntot > EPS else float64(0.0)
        dXs   = ((vC[s]/ntot - C[s]*dntot/(ntot*ntot))
                 if ntot > EPS else float64(0.0))
        cv_mix += Xs*cv_s
        dcv    += dXs*cv_s + Xs*dcvdT*vT
        up     += u_s*omega[s]
        dup    += u_s*Jv_om[s] + cv_s*omega[s]*vT
    denom = ntot*cv_mix
    if abs(denom) > EPS:
        dTdt_p = (-dup/denom
                  + up*dntot/(denom*ntot)
                  + up*dcv/(ntot*cv_mix*cv_mix))
    else:
        dTdt_p = float64(0.0)
    Jv_TC[0] = dTdt_p
    for s in range(n_sp):
        Jv_TC[s+1] = Jv_om[s]


# =============================================================================
#  Restarted GMRES (matrix-free, JVP-based)
#  All scratch arrays are global row-slices (same strategy as original v5).
# =============================================================================

@cuda.jit(device=True)
def _gmres(
    T, C, h, omega,
    rhs, dx,
    V, H, cs, sn, g, w, y,
    vC_tmp, Jvo_tmp,
    n_state, R, tol, maxiter,
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    n_sp, thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    beta = float64(0.0)
    for i in range(n_state):
        dx[i] = float64(0.0)
        beta  += rhs[i]*rhs[i]
    beta = math.sqrt(beta)
    if beta < tol:
        return

    for outer in range(maxiter):
        for i in range((R+1)*R):  H[i]  = float64(0.0)
        for i in range(R):        cs[i] = float64(0.0); sn[i] = float64(0.0)
        for i in range(R+1):      g[i]  = float64(0.0)

        if outer == 0:
            inv = float64(1.0)/beta
            for i in range(n_state):
                V[i] = rhs[i]*inv
        else:
            vT_ = dx[0]
            for s in range(n_sp): vC_tmp[s] = dx[s+1]
            _eval_jvp(
                T, C, vT_, vC_tmp, Jvo_tmp,
                kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
                kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
                kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
                k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
                Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
                n_sp, thL, thH, thM,
                n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
                n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
                n_f, fA0, fb0, fE0, fAi, fbi, fEi,
                fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
            )
            _adiabatic_jvp(T, C, vT_, vC_tmp, omega, Jvo_tmp, n_sp, thL, thH, thM, w)
            beta = float64(0.0)
            for i in range(n_state):
                ri = rhs[i] - (dx[i] - h*w[i])
                w[i] = ri; beta += ri*ri
            beta = math.sqrt(beta)
            if beta < tol: return
            inv = float64(1.0)/beta
            for i in range(n_state):
                V[i] = w[i]*inv

        g[0] = beta
        j_conv = R

        # Arnoldi loop
        for j in range(R):
            vj = j*n_state
            vT_ = V[vj]
            for s in range(n_sp): vC_tmp[s] = V[vj+s+1]

            _eval_jvp(
                T, C, vT_, vC_tmp, Jvo_tmp,
                kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
                kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
                kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
                k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
                Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
                n_sp, thL, thH, thM,
                n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
                n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
                n_f, fA0, fb0, fE0, fAi, fbi, fEi,
                fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
            )
            _adiabatic_jvp(T, C, vT_, vC_tmp, omega, Jvo_tmp, n_sp, thL, thH, thM, w)
            for i in range(n_state):
                w[i] = V[vj+i] - h*w[i]

            for i in range(j+1):
                vi = i*n_state
                hij = float64(0.0)
                for k in range(n_state): hij += w[k]*V[vi+k]
                H[i*R+j] = hij
                for k in range(n_state): w[k] -= hij*V[vi+k]

            h_j1j = float64(0.0)
            for k in range(n_state): h_j1j += w[k]*w[k]
            h_j1j = math.sqrt(h_j1j)
            H[(j+1)*R+j] = h_j1j

            vj1 = (j+1)*n_state
            if h_j1j > 1e-300:
                inv2 = float64(1.0)/h_j1j
                for k in range(n_state): V[vj1+k] = w[k]*inv2
            else:
                for k in range(n_state): V[vj1+k] = float64(0.0)

            for i in range(j):
                hi  = H[i*R+j]; hi1 = H[(i+1)*R+j]
                c_  = cs[i];    s_  = sn[i]
                H[i*R+j]     =  c_*hi + s_*hi1
                H[(i+1)*R+j] = -s_*hi + c_*hi1

            hjj   = H[j*R+j]; hj1j = H[(j+1)*R+j]
            rdiag = math.sqrt(hjj*hjj + hj1j*hj1j)
            if rdiag > 1e-300:
                cn = hjj/rdiag; sn_ = hj1j/rdiag
            else:
                cn = float64(1.0); sn_ = float64(0.0)
            cs[j] = cn; sn[j] = sn_
            H[j*R+j]     = rdiag
            H[(j+1)*R+j] = float64(0.0)
            gj = g[j]
            g[j]   =  cn*gj
            g[j+1] = -sn_*gj

            if abs(g[j+1]) < tol:
                j_conv = j+1
                break

        # Back-substitution
        m = j_conv
        for k in range(m-1, -1, -1):
            s_ = g[k]
            for l in range(k+1, m): s_ -= H[k*R+l]*y[l]
            d = H[k*R+k]
            y[k] = s_/d if abs(d) > 1e-300 else float64(0.0)

        for i in range(n_state):
            acc = float64(0.0)
            for k in range(m): acc += V[k*n_state+i]*y[k]
            dx[i] += acc

        # True residual check
        vT_ = dx[0]
        for s in range(n_sp): vC_tmp[s] = dx[s+1]
        _eval_jvp(
            T, C, vT_, vC_tmp, Jvo_tmp,
            kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
            kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
            kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
            k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
            Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
            n_sp, thL, thH, thM,
            n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
            n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
            n_f, fA0, fb0, fE0, fAi, fbi, fEi,
            fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
        )
        _adiabatic_jvp(T, C, vT_, vC_tmp, omega, Jvo_tmp, n_sp, thL, thH, thM, w)
        res_true = float64(0.0)
        for i in range(n_state):
            ri = rhs[i] - (dx[i] - h*w[i])
            res_true += ri*ri
        res_true = math.sqrt(res_true)

        if res_true < tol or j_conv < R:
            return
        beta = res_true

    return


# =============================================================================
#  Main kernel — one thread per reactor
# =============================================================================

@cuda.jit(cache=True)
def backward_euler_kernel(
    TC_in, TC_out,
    h_scalar,
    newton_tol, newton_maxiter,
    gmres_restart, gmres_tol, gmres_maxiter,
    converged,
    # GMRES scratch — global, (B, ...)
    gm_V, gm_H, gm_cs, gm_sn, gm_g, gm_w, gm_y,
    # chemistry context — global, (B, n_reactions) or (B, n_flat_entries)
    ctx_kf_e,   ctx_Kc_e,  ctx_cr_e,  ctx_cp_e,  ctx_dG0_e,
    ctx_sRre,   ctx_sRpe,
    ctx_kf_tb,  ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb,  ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f,   ctx_Kc_f,  ctx_cr_f,  ctx_cp_f,  ctx_dG0_f,
    ctx_sRrf,   ctx_sRpf,
    ctx_k0_f,   ctx_kinf_f, ctx_M_f,  ctx_Pr_f,  ctx_F_f,   ctx_kb_f,
    ctx_Fc_f,   ctx_lFc_f, ctx_Cp_f,  ctx_Np_f,  ctx_f1_f,  ctx_exp_f,
    # mechanism (read-only, shared across all threads)
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
    h = h_scalar

    # Small local scratch — always fits in registers
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

    # ── PREDICTOR: explicit Euler from x0 ────────────────────────────────
    T_loc = xk[0]
    for s in range(n_sp): C[s] = xk[s+1]
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
        if xk[s+1] < 0.0: xk[s+1] = float64(0.0)

    # ── NEWTON LOOP ───────────────────────────────────────────────────────
    for nit in range(newton_maxiter):
        T_loc = xk[0]
        for s in range(n_sp): C[s] = xk[s+1]

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
            fi     = xk[i] - TC_in[b, i] - h*dTC[i]
            rhs[i] = -fi
            res2   += fi*fi
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
            if xk[s+1] < 0.0: xk[s+1] = float64(0.0)

    for i in range(n_state):
        TC_out[b, i] = xk[i]


# =============================================================================
#  Utility kernels
# =============================================================================

@cuda.jit
def _fill_int_zero(arr):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < arr.shape[0]:
        arr[i] = int32(0)


@cuda.jit
def _store_snapshot(TC_src, conv_src, TC_hist, conv_hist, snap_idx):
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= TC_src.shape[0]:
        return
    for i in range(TC_src.shape[1]):
        TC_hist[snap_idx, b, i] = TC_src[b, i]
    conv_hist[snap_idx, b] = conv_src[b]