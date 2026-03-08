"""
chemistry_kernels_pb.py — Per-block CUDA backward-Euler integrator.

Architecture: one CUDA BLOCK = one reactor.
All 64 threads within a block cooperate to integrate a single IVP.

Key differences vs per-thread version (chemistry_kernels.py):
  - Parallel loops over species/reactions using threadIdx.x
  - Atomic adds for scatter into omega[] (multiple reactions → same species)
  - Parallel reductions for dot-products and norms (GMRES)
  - Shared memory for hot per-reactor vectors:
      xk, C, om, dTC, rhs, dx, w, g, y, cs, sn, H
    (~8 KB per block — fits in 12 KB budget at 8 blocks/SM)
  - V (Krylov basis, ~9 KB) stays in global memory (too large for shared)
  - Context buffers (ctx_*) stay in global memory (variable size)
  - Thread 0 is the "leader" for scalar operations and back-substitution
  - cuda.syncthreads() after every cooperative write

Memory layout of shared arrays (declared via cuda.shared.array):
  All arrays are fixed-size MAX_N_STATE / MAX_R to allow static allocation.

Shared memory budget per block (R=20, n_state=54):
  10 vectors × 54 × 8B = 4 320 B   (xk,C,om,dTC,rhs,dx,w,Jvo,vC,acc_buf)
  H: (R+1)×R × 8B      = 3 360 B
  g,y,cs,sn: ~(2R+2)×8 =   336 B
  scalars (beta,res,...): ~  100 B
  Total ≈ 8 116 B  →  well within 12 KB/block at 8 blocks/SM
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
MAX_R             = 20   # max GMRES restart dimension
MAX_R_p_1 = MAX_R + 1
MAX_R_H = MAX_R * (MAX_R + 1)
THREADS_PER_BLOCK = 64


# =============================================================================
#  Thermodynamics helpers  (unchanged — pure device functions)
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
#  Parallel reduction helpers (warp + block level)
# =============================================================================

@cuda.jit(device=True, inline=True)
def _block_reduce_sum(val, scratch, tid, n_threads):
    """
    Block-wide sum reduction.
    scratch : shared array of length >= n_threads (float64)
    Returns the sum in scratch[0] (only thread 0 reads it safely).
    """
    scratch[tid] = val
    cuda.syncthreads()
    stride = n_threads >> 1
    while stride > 0:
        if tid < stride:
            scratch[tid] += scratch[tid + stride]
        cuda.syncthreads()
        stride >>= 1
    # result is in scratch[0]


@cuda.jit(device=True, inline=True)
def _block_dot(a, b, scratch, tid, n, n_threads):
    """
    Parallel dot product of vectors a[0:n] and b[0:n].
    scratch : shared float64 array of length >= n_threads.
    Returns result in scratch[0]; only thread 0 should read.
    """
    acc = float64(0.0)
    i = tid
    while i < n:
        acc += a[i] * b[i]
        i += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)


@cuda.jit(device=True, inline=True)
def _block_norm2(a, scratch, tid, n, n_threads):
    """
    Parallel squared-norm ||a[0:n]||^2.
    Returns result in scratch[0].
    """
    acc = float64(0.0)
    i = tid
    while i < n:
        acc += a[i] * a[i]
        i += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)


# =============================================================================
#  Full RHS — parallel over reactions, atomic scatter into omega
#  Each thread handles one reaction in each loop.
# =============================================================================

@cuda.jit(device=True)
def _eval_rhs_and_context_pb(
    tid, n_threads,
    T, C,
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    omega,          # shared memory, zeroed before call
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    # Zero omega in parallel
    i = tid
    while i < n_sp:
        omega[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    c7 = cuda.local.array(7, float64)

    # ── Elementary (each thread handles one reaction) ─────────────────
    j = tid
    while j < n_e:
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
            cuda.atomic.add(omega, eRI[k], -eRS[k] * RR)
        for k in range(ePO[j], ePO[j+1]):
            cuda.atomic.add(omega, ePI[k],  ePS[k] * RR)
        j += n_threads
    cuda.syncthreads()

    # ── Three-body ────────────────────────────────────────────────────
    j = tid
    while j < n_tb:
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
        # M: parallel reduction over species
        M = float64(0.0)
        for s in range(n_sp):
            M += C[s] * tbEff[j * n_sp + s]
        M_tb[j] = M
        ow     = tbOW[j]
        baseR  = kf * (cr - (1.0 - ow) * cp / Kc)
        bRR_tb[j] = baseR
        RR = baseR * M
        for k in range(tbRO[j], tbRO[j+1]):
            cuda.atomic.add(omega, tbRI[k], -tbRS[k] * RR)
        for k in range(tbPO[j], tbPO[j+1]):
            cuda.atomic.add(omega, tbPI[k],  tbPS[k] * RR)
        j += n_threads
    cuda.syncthreads()

    # ── Falloff ───────────────────────────────────────────────────────
    j = tid
    while j < n_f:
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
            cuda.atomic.add(omega, fRI[k], -fRS[k] * RR)
        for k in range(fPO[j], fPO[j+1]):
            cuda.atomic.add(omega, fPI[k],  fPS[k] * RR)
        j += n_threads
    cuda.syncthreads()


# =============================================================================
#  Adiabatic RHS — parallel over species, shared scalars via scratch
# =============================================================================

@cuda.jit(device=True)
def _adiabatic_rhs_pb(
    tid, n_threads,
    T, C, omega, n_sp, thL, thH, thM,
    dTCdt,       # shared output, length n_sp+1
    scratch,     # shared scratch for reductions, length >= n_threads
):
    c7 = cuda.local.array(7, float64)

    # ntot = sum(C)
    acc = float64(0.0)
    i = tid
    while i < n_sp:
        acc += C[i]
        i += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)
    cuda.syncthreads()
    ntot = scratch[0]

    # Each thread accumulates its portion of cv_mix and su_om
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

    # Reduce cv_mix
    _block_reduce_sum(cv_loc, scratch, tid, n_threads)
    cuda.syncthreads()
    cv_mix = scratch[0]

    # Reduce su_om
    _block_reduce_sum(su_loc, scratch, tid, n_threads)
    cuda.syncthreads()
    su_om = scratch[0]

    if tid == 0:
        denom     = ntot * cv_mix
        dTCdt[0]  = -su_om / denom if abs(denom) > EPS else float64(0.0)
    cuda.syncthreads()

    # species RHS in parallel
    i = tid
    while i < n_sp:
        dTCdt[i+1] = omega[i]
        i += n_threads
    cuda.syncthreads()


# =============================================================================
#  JVP — parallel over reactions, atomic scatter
# =============================================================================

@cuda.jit(device=True)
def _eval_jvp_pb(
    tid, n_threads,
    T, C, vT, vC, Jv_om,   # Jv_om is shared, zeroed before call
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
    # Zero Jv_om in parallel
    i = tid
    while i < n_sp:
        Jv_om[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    # ── Elementary ────────────────────────────────────────────────────
    j = tid
    while j < n_e:
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
            cuda.atomic.add(Jv_om, eRI[k], -eRS[k] * dRR_v)
        for k in range(ePO[j], ePO[j+1]):
            cuda.atomic.add(Jv_om, ePI[k],  ePS[k] * dRR_v)
        j += n_threads
    cuda.syncthreads()

    # ── Three-body ────────────────────────────────────────────────────
    j = tid
    while j < n_tb:
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
            cuda.atomic.add(Jv_om, tbRI[k], -tbRS[k] * dRR_v)
        for k in range(tbPO[j], tbPO[j+1]):
            cuda.atomic.add(Jv_om, tbPI[k],  tbPS[k] * dRR_v)
        j += n_threads
    cuda.syncthreads()

    # ── Falloff ───────────────────────────────────────────────────────
    j = tid
    while j < n_f:
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
            cuda.atomic.add(Jv_om, fRI[k], -fRS[k] * dRR_v)
        for k in range(fPO[j], fPO[j+1]):
            cuda.atomic.add(Jv_om, fPI[k],  fPS[k] * dRR_v)
        j += n_threads
    cuda.syncthreads()


# =============================================================================
#  Adiabatic JVP — parallel over species
# =============================================================================

@cuda.jit(device=True)
def _adiabatic_jvp_pb(
    tid, n_threads,
    T, C, vT, vC, omega, Jv_om, n_sp, thL, thH, thM,
    Jv_TC,    # shared output, length n_sp+1
    scratch,  # shared scratch for reductions, length >= n_threads
):
    c7 = cuda.local.array(7, float64)

    # ntot, dntot — parallel reductions
    acc_n = float64(0.0); acc_dn = float64(0.0)
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

    # Parallel accumulation of cv_mix, dcv, up, dup
    lcv = float64(0.0); ldcv = float64(0.0)
    lup = float64(0.0); ldup = float64(0.0)
    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        T2 = T*T; T3 = T2*T; T4 = T3*T
        cpR   = c7[0]+c7[1]*T+c7[2]*T2+c7[3]*T3+c7[4]*T4
        cv_s  = R_CONST*(cpR - 1.0)
        dcvdT = R_CONST*(c7[1]+2.0*c7[2]*T+3.0*c7[3]*T2+4.0*c7[4]*T3)
        hRT   = (c7[0]+c7[1]*T/2.0+c7[2]*T2/3.0
                 +c7[3]*T3/4.0+c7[4]*T4/5.0+c7[5]/T)
        u_s   = R_CONST*T*hRT - R_CONST*T
        Xs    = C[i]/ntot  if ntot > EPS else float64(0.0)
        dXs   = ((vC[i]/ntot - C[i]*dntot/(ntot*ntot))
                 if ntot > EPS else float64(0.0))
        lcv  += Xs*cv_s
        ldcv += dXs*cv_s + Xs*dcvdT*vT
        lup  += u_s*omega[i]
        ldup += u_s*Jv_om[i] + cv_s*omega[i]*vT
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
        denom = ntot*cv_mix
        if abs(denom) > EPS:
            Jv_TC[0] = (-dup/denom
                        + up*dntot/(denom*ntot)
                        + up*dcv/(ntot*cv_mix*cv_mix))
        else:
            Jv_TC[0] = float64(0.0)
    cuda.syncthreads()

    i = tid
    while i < n_sp:
        Jv_TC[i+1] = Jv_om[i]
        i += n_threads
    cuda.syncthreads()


# =============================================================================
#  Restarted GMRES (matrix-free, JVP-based) — per-block version
# =============================================================================

@cuda.jit(device=True)
def _gmres_pb(
    tid, n_threads,
    T, C, h, omega,
    sh_rhs, sh_dx,
    sh_w,
    sh_g, sh_y, sh_cs, sh_sn,
    sh_H,
    sh_scratch,
    sh_Jvo,
    sh_vC,
    gm_V,
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
    # Zero dx
    i = tid
    while i < n_state:
        sh_dx[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    # beta = ||rhs||
    _block_norm2(sh_rhs, sh_scratch, tid, n_state, n_threads)
    cuda.syncthreads()
    if tid == 0:
        sh_g[0] = math.sqrt(sh_scratch[0])
    cuda.syncthreads()
    beta = sh_g[0]
    if beta < tol:
        return

    for outer in range(maxiter):
        # Zero H, cs, sn, g
        i = tid
        while i < (R+1)*R:
            sh_H[i] = float64(0.0)
            i += n_threads
        i = tid
        while i < R:
            sh_cs[i] = float64(0.0)
            sh_sn[i] = float64(0.0)
            i += n_threads
        i = tid
        while i < R+1:
            sh_g[i] = float64(0.0)
            i += n_threads
        cuda.syncthreads()

        if outer == 0:
            inv = float64(1.0) / beta
            i = tid
            while i < n_state:
                gm_V[i] = sh_rhs[i] * inv
                i += n_threads
            cuda.syncthreads()
        else:
            vT_ = sh_dx[0]
            i = tid
            while i < n_sp:
                sh_vC[i] = sh_dx[i+1]
                i += n_threads
            cuda.syncthreads()
            _eval_jvp_pb(
                tid, n_threads,
                T, C, vT_, sh_vC, sh_Jvo,
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
            _adiabatic_jvp_pb(
                tid, n_threads,
                T, C, vT_, sh_vC, omega, sh_Jvo, n_sp, thL, thH, thM,
                sh_w, sh_scratch,
            )
            _block_norm2_inplace_sub(
                sh_rhs, sh_dx, sh_w, h, sh_scratch, sh_w,
                tid, n_state, n_threads,
            )
            cuda.syncthreads()
            beta = math.sqrt(sh_scratch[0])
            if beta < tol:
                return
            inv = float64(1.0) / beta
            i = tid
            while i < n_state:
                gm_V[i] = sh_w[i] * inv
                i += n_threads
            cuda.syncthreads()

        if tid == 0:
            sh_g[0] = beta
        cuda.syncthreads()

        j_conv = R

        # ── Arnoldi loop ──────────────────────────────────────────────
        for j in range(R):
            vj_off = j * n_state

            vT_ = gm_V[vj_off]
            i = tid
            while i < n_sp:
                sh_vC[i] = gm_V[vj_off + i + 1]
                i += n_threads
            cuda.syncthreads()

            _eval_jvp_pb(
                tid, n_threads,
                T, C, vT_, sh_vC, sh_Jvo,
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
            _adiabatic_jvp_pb(
                tid, n_threads,
                T, C, vT_, sh_vC, omega, sh_Jvo, n_sp, thL, thH, thM,
                sh_w, sh_scratch,
            )
            i = tid
            while i < n_state:
                sh_w[i] = gm_V[vj_off + i] - h * sh_w[i]
                i += n_threads
            cuda.syncthreads()

            # Modified Gram–Schmidt
            for ii in range(j+1):
                vi_off = ii * n_state
                _block_dot(sh_w, gm_V[vi_off:vi_off+n_state],
                           sh_scratch, tid, n_state, n_threads)
                cuda.syncthreads()
                hij = sh_scratch[0]
                if tid == 0:
                    sh_H[ii*R + j] = hij
                cuda.syncthreads()
                i = tid
                while i < n_state:
                    sh_w[i] -= hij * gm_V[vi_off + i]
                    i += n_threads
                cuda.syncthreads()

            _block_norm2(sh_w, sh_scratch, tid, n_state, n_threads)
            cuda.syncthreads()
            h_j1j = math.sqrt(sh_scratch[0])
            if tid == 0:
                sh_H[(j+1)*R + j] = h_j1j
            cuda.syncthreads()

            vj1_off = (j+1) * n_state
            if h_j1j > 1e-300:
                inv2 = float64(1.0) / h_j1j
                i = tid
                while i < n_state:
                    gm_V[vj1_off + i] = sh_w[i] * inv2
                    i += n_threads
            else:
                i = tid
                while i < n_state:
                    gm_V[vj1_off + i] = float64(0.0)
                    i += n_threads
            cuda.syncthreads()

            if tid == 0:
                for ii in range(j):
                    hi  = sh_H[ii*R+j]; hi1 = sh_H[(ii+1)*R+j]
                    c_  = sh_cs[ii];    s_  = sh_sn[ii]
                    sh_H[ii*R+j]     =  c_*hi + s_*hi1
                    sh_H[(ii+1)*R+j] = -s_*hi + c_*hi1
                hjj   = sh_H[j*R+j]; hj1j = sh_H[(j+1)*R+j]
                rdiag = math.sqrt(hjj*hjj + hj1j*hj1j)
                if rdiag > 1e-300:
                    cn = hjj/rdiag; sn_ = hj1j/rdiag
                else:
                    cn = float64(1.0); sn_ = float64(0.0)
                sh_cs[j] = cn; sh_sn[j] = sn_
                sh_H[j*R+j]     = rdiag
                sh_H[(j+1)*R+j] = float64(0.0)
                gj = sh_g[j]
                sh_g[j]   =  cn*gj
                sh_g[j+1] = -sn_*gj
            cuda.syncthreads()

            if abs(sh_g[j+1]) < tol:
                j_conv = j+1
                break

        # ── Back-substitution ─────────────────────────────────────────
        m = j_conv
        if tid == 0:
            for k in range(m-1, -1, -1):
                s_ = sh_g[k]
                for l in range(k+1, m):
                    s_ -= sh_H[k*R+l] * sh_y[l]
                d = sh_H[k*R+k]
                sh_y[k] = s_/d if abs(d) > 1e-300 else float64(0.0)
        cuda.syncthreads()

        i = tid
        while i < n_state:
            acc = float64(0.0)
            for k in range(m):
                acc += gm_V[k*n_state + i] * sh_y[k]
            sh_dx[i] += acc
            i += n_threads
        cuda.syncthreads()

        # True residual check
        vT_ = sh_dx[0]
        i = tid
        while i < n_sp:
            sh_vC[i] = sh_dx[i+1]
            i += n_threads
        cuda.syncthreads()
        _eval_jvp_pb(
            tid, n_threads,
            T, C, vT_, sh_vC, sh_Jvo,
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
        _adiabatic_jvp_pb(
            tid, n_threads,
            T, C, vT_, sh_vC, omega, sh_Jvo, n_sp, thL, thH, thM,
            sh_w, sh_scratch,
        )
        _block_norm2_inplace_sub(
            sh_rhs, sh_dx, sh_w, h, sh_scratch, sh_w,
            tid, n_state, n_threads,
        )
        cuda.syncthreads()
        res_true = math.sqrt(sh_scratch[0])

        if res_true < tol or j_conv < R:
            return
        beta = res_true

    return


# =============================================================================
#  Helper: r = rhs - (dx - h*Jv), return ||r||^2 in scratch[0]
# =============================================================================

@cuda.jit(device=True, inline=True)
def _block_norm2_inplace_sub(rhs, dx, Jv_TC, h, scratch, out, tid, n, n_threads):
    acc = float64(0.0)
    i = tid
    while i < n:
        ri    = rhs[i] - (dx[i] - h * Jv_TC[i])
        out[i] = ri
        acc   += ri * ri
        i += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)


# =============================================================================
#  Main kernel — one BLOCK per reactor
# =============================================================================

@cuda.jit(cache=True)
def backward_euler_kernel_pb(
    TC_in, TC_out,
    h_scalar,
    newton_tol, newton_maxiter,
    gmres_restart, gmres_tol, gmres_maxiter,
    converged,
    gm_V,
    ctx_kf_e,   ctx_Kc_e,  ctx_cr_e,  ctx_cp_e,  ctx_dG0_e,
    ctx_sRre,   ctx_sRpe,
    ctx_kf_tb,  ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb,  ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f,   ctx_Kc_f,  ctx_cr_f,  ctx_cp_f,  ctx_dG0_f,
    ctx_sRrf,   ctx_sRpf,
    ctx_k0_f,   ctx_kinf_f, ctx_M_f,  ctx_Pr_f,  ctx_F_f,   ctx_kb_f,
    ctx_Fc_f,   ctx_lFc_f, ctx_Cp_f,  ctx_Np_f,  ctx_f1_f,  ctx_exp_f,
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    """
    Launch with:  backward_euler_kernel_pb[n_reactors, THREADS_PER_BLOCK](...)
    blockIdx.x  = reactor index b
    threadIdx.x = tid within block
    """
    b   = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= TC_in.shape[0]:
        return

    n_state = n_sp + 1
    h = h_scalar

    # ── Shared memory allocation ──────────────────────────────────────
    sh_xk  = cuda.shared.array(MAX_N_STATE, float64)
    sh_C   = cuda.shared.array(MAX_N_STATE, float64)
    sh_om  = cuda.shared.array(MAX_N_STATE, float64)
    sh_dTC = cuda.shared.array(MAX_N_STATE, float64)
    sh_rhs = cuda.shared.array(MAX_N_STATE, float64)
    sh_dx  = cuda.shared.array(MAX_N_STATE, float64)
    sh_w   = cuda.shared.array(MAX_N_STATE, float64)
    sh_Jvo = cuda.shared.array(MAX_N_STATE, float64)
    sh_vC  = cuda.shared.array(MAX_N_STATE, float64)
    sh_g   = cuda.shared.array(MAX_R_p_1, float64)
    sh_y   = cuda.shared.array(MAX_R,     float64)
    sh_cs  = cuda.shared.array(MAX_R,     float64)
    sh_sn  = cuda.shared.array(MAX_R,     float64)
    sh_H   = cuda.shared.array(MAX_R_H,   float64)
    sh_scratch = cuda.shared.array(THREADS_PER_BLOCK, float64)

    # ── Load initial state into shared ───────────────────────────────
    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()

    # ── Context row-slices for this reactor ──────────────────────────
    e_kf   = ctx_kf_e[b];   e_Kc  = ctx_Kc_e[b]
    e_cr   = ctx_cr_e[b];   e_cp  = ctx_cp_e[b];  e_dG = ctx_dG0_e[b]
    e_sRr  = ctx_sRre[b];   e_sRp = ctx_sRpe[b]

    tb_kf  = ctx_kf_tb[b];  tb_Kc  = ctx_Kc_tb[b]
    tb_cr  = ctx_cr_tb[b];  tb_cp  = ctx_cp_tb[b]; tb_dG = ctx_dG0_tb[b]
    tb_sRr = ctx_sRrtb[b];  tb_sRp = ctx_sRptb[b]
    tb_bRR = ctx_bRR_tb[b]; tb_M   = ctx_M_tb[b]

    f_kf   = ctx_kf_f[b];   f_Kc   = ctx_Kc_f[b]
    f_cr   = ctx_cr_f[b];   f_cp   = ctx_cp_f[b];  f_dG  = ctx_dG0_f[b]
    f_sRr  = ctx_sRrf[b];   f_sRp  = ctx_sRpf[b]
    f_k0   = ctx_k0_f[b];   f_ki   = ctx_kinf_f[b]
    f_M    = ctx_M_f[b];    f_Pr   = ctx_Pr_f[b]
    f_F    = ctx_F_f[b];    f_kb   = ctx_kb_f[b]
    f_Fc   = ctx_Fc_f[b];   f_lFc  = ctx_lFc_f[b]
    f_Cp   = ctx_Cp_f[b];   f_Np   = ctx_Np_f[b]
    f_f1   = ctx_f1_f[b];   f_exp  = ctx_exp_f[b]

    # ── PREDICTOR: explicit Euler from x0 ────────────────────────────
    T_loc = sh_xk[0]
    i = tid
    while i < n_sp:
        sh_C[i] = sh_xk[i+1]
        i += n_threads
    cuda.syncthreads()

    _eval_rhs_and_context_pb(
        tid, n_threads,
        T_loc, sh_C,
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
    _adiabatic_rhs_pb(
        tid, n_threads,
        T_loc, sh_C, sh_om, n_sp, thL, thH, thM,
        sh_dTC, sh_scratch,
    )

    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i] + h * sh_dTC[i]
        i += n_threads
    cuda.syncthreads()
    if tid == 0:
        if sh_xk[0] < 1.0:
            sh_xk[0] = 1.0
    i = tid + 1
    while i < n_state:
        if sh_xk[i] < 0.0:
            sh_xk[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    # ── NEWTON LOOP ───────────────────────────────────────────────────
    for nit in range(newton_maxiter):
        T_loc = sh_xk[0]
        i = tid
        while i < n_sp:
            sh_C[i] = sh_xk[i+1]
            i += n_threads
        cuda.syncthreads()

        _eval_rhs_and_context_pb(
            tid, n_threads,
            T_loc, sh_C,
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
        _adiabatic_rhs_pb(
            tid, n_threads,
            T_loc, sh_C, sh_om, n_sp, thL, thH, thM,
            sh_dTC, sh_scratch,
        )

        acc = float64(0.0)
        i = tid
        while i < n_state:
            fi        = sh_xk[i] - TC_in[b, i] - h * sh_dTC[i]
            sh_rhs[i] = -fi
            acc      += fi * fi
            i += n_threads
        _block_reduce_sum(acc, sh_scratch, tid, n_threads)
        cuda.syncthreads()
        res_norm = math.sqrt(sh_scratch[0])

        if res_norm < newton_tol:
            if tid == 0:
                converged[b] = int32(1)
            cuda.syncthreads()
            break

        _gmres_pb(
            tid, n_threads,
            T_loc, sh_C, h, sh_om,
            sh_rhs, sh_dx,
            sh_w, sh_g, sh_y, sh_cs, sh_sn, sh_H,
            sh_scratch, sh_Jvo, sh_vC,
            gm_V[b],
            n_state, gmres_restart, gmres_tol, gmres_maxiter,
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
        )

        i = tid
        while i < n_state:
            sh_xk[i] += sh_dx[i]
            i += n_threads
        cuda.syncthreads()
        if tid == 0:
            if sh_xk[0] < 1.0:
                sh_xk[0] = 1.0
        i = tid + 1
        while i < n_state:
            if sh_xk[i] < 0.0:
                sh_xk[i] = float64(0.0)
            i += n_threads
        cuda.syncthreads()

    # ── Écriture du résultat en mémoire globale ───────────────────────
    i = tid
    while i < n_state:
        TC_out[b, i] = sh_xk[i]
        i += n_threads
    cuda.syncthreads()