"""
kernels/per_block/context_pb.py — Contexte cinétique + RHS (per-block).

Fonctions exportées :
    _eval_context_pb(tid, n_threads, T, C, <ctx_slices>, n_sp, <mech_args>)
    _eval_rhs_pb(tid, n_threads, omega, <ctx_slices>, n_sp, n_e, n_tb, n_f, ...)
    _eval_rhs_and_context_pb(...)   ← compatibilité
"""

import math
from numba import cuda, float64

from ..shared.thermo import (
    R_CONST, PREF, EPS,
    _load_coef7, _h_RT, _s_R, _kf_arr, _log_conc_prod,
)


@cuda.jit(device=True)
def _eval_context_pb(
    tid, n_threads,
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
):
    """
    Remplit les slices cinétiques sans écrire dans omega.
    Chaque thread traite ses réactions en stride.
    """
    c7 = cuda.local.array(7, float64)

    # ── Élémentaires ──────────────────────────────────────────────────
    j = tid
    while j < n_e:
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
        cr_e[j] = math.exp(lr)
        cp_e[j] = math.exp(lp)
        j += n_threads
    cuda.syncthreads()

    # ── Trois-corps ───────────────────────────────────────────────────
    j = tid
    while j < n_tb:
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
        cr_tb[j] = math.exp(lr)
        cp_tb[j] = math.exp(lp)
        M = float64(0.0)
        for s in range(n_sp):
            M += C[s] * tbEff[j * n_sp + s]
        M_tb[j] = M
        ow = tbOW[j]
        bRR_tb[j] = kf * (cr_tb[j] - (float64(1.0) - ow) * cp_tb[j] / Kc)
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
            if t1 > EPS: Fc += tA * math.exp(-T / t1)
            if t3 > EPS: Fc += (float64(1.0) - tA) * math.exp(-T / t3)
            if t2 > EPS: Fc += math.exp(-t2 / T)
            if Fc < EPS: Fc = EPS
            lFc   = math.log10(Fc)
            Cp    = float64(-0.4)  - float64(0.67) * lFc
            Np    = float64(0.75) - float64(1.27) * lFc
            Prr   = Pr if Pr > EPS else EPS
            logPr = math.log10(Prr)
            num   = logPr + Cp
            den   = Np - float64(0.14) * (logPr + Cp)
            if abs(den) < EPS: den = EPS if den >= float64(0.0) else -EPS
            f1_v  = num / den
            ex    = lFc / (float64(1.0) + f1_v * f1_v)
            F_v   = float64(10.0) ** ex
        else:
            Fc   = float64(1.0); lFc  = float64(0.0)
            Cp   = float64(0.0); Np   = float64(1.0)
            f1_v = float64(0.0); ex   = float64(0.0); F_v = float64(1.0)
        Fcent_f[j] = Fc;  lFcent_f[j] = lFc
        Cpar_f[j]  = Cp;  Npar_f[j]   = Np
        f1_f[j]    = f1_v; exp_f[j]   = ex; F_f[j] = F_v
        kbase = ki_s * (Pr / (float64(1.0) + Pr))
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
        cr_f[j] = math.exp(lr)
        cp_f[j] = math.exp(lp)
        j += n_threads
    cuda.syncthreads()


@cuda.jit(device=True)
def _eval_rhs_pb(
    tid, n_threads,
    omega,
    kf_e, Kc_e, cr_e, cp_e, eOW,
    kf_tb, Kc_tb, cr_tb, cp_tb, bRR_tb, M_tb, tbOW,
    kf_f, Kc_f, cr_f, cp_f, fOW,
    n_sp, n_e, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO,
    n_f, fRI, fRS, fRO, fPI, fPS, fPO,
):
    """
    Calcule omega depuis le contexte déjà rempli.
    omega doit être en mémoire partagée.
    """
    i = tid
    while i < n_sp:
        omega[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    # ── Élémentaires ──────────────────────────────────────────────────
    j = tid
    while j < n_e:
        kf  = kf_e[j]; Kc = Kc_e[j]
        cr  = cr_e[j]; cp = cp_e[j]; ow = eOW[j]
        RR  = kf * (cr - (float64(1.0) - ow) * cp / Kc)
        for k in range(eRO[j], eRO[j + 1]):
            cuda.atomic.add(omega, eRI[k], -eRS[k] * RR)
        for k in range(ePO[j], ePO[j + 1]):
            cuda.atomic.add(omega, ePI[k],  ePS[k] * RR)
        j += n_threads
    cuda.syncthreads()

    # ── Trois-corps ───────────────────────────────────────────────────
    j = tid
    while j < n_tb:
        RR = bRR_tb[j] * M_tb[j]
        for k in range(tbRO[j], tbRO[j + 1]):
            cuda.atomic.add(omega, tbRI[k], -tbRS[k] * RR)
        for k in range(tbPO[j], tbPO[j + 1]):
            cuda.atomic.add(omega, tbPI[k],  tbPS[k] * RR)
        j += n_threads
    cuda.syncthreads()

    # ── Falloff ───────────────────────────────────────────────────────
    j = tid
    while j < n_f:
        kf  = kf_f[j]; Kc = Kc_f[j]
        cr  = cr_f[j]; cp = cp_f[j]; ow = fOW[j]
        RR  = kf * (cr - (float64(1.0) - ow) * cp / Kc)
        for k in range(fRO[j], fRO[j + 1]):
            cuda.atomic.add(omega, fRI[k], -fRS[k] * RR)
        for k in range(fPO[j], fPO[j + 1]):
            cuda.atomic.add(omega, fPI[k],  fPS[k] * RR)
        j += n_threads
    cuda.syncthreads()


@cuda.jit(device=True)
def _eval_rhs_and_context_pb(
    tid, n_threads,
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
    """Compatibilité : _eval_context_pb puis _eval_rhs_pb."""
    _eval_context_pb(
        tid, n_threads, T, C,
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
    )
    _eval_rhs_pb(
        tid, n_threads, omega,
        kf_e, Kc_e, cr_e, cp_e, eOW,
        kf_tb, Kc_tb, cr_tb, cp_tb, bRR_tb, M_tb, tbOW,
        kf_f, Kc_f, cr_f, cp_f, fOW,
        n_sp, n_e, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO,
        n_f, fRI, fRS, fRO, fPI, fPS, fPO,
    )


"""
kernels/per_block/jvp_wrapper_pb.py — Wrappers Python pour JVP et RHS (per-block).

Classes exportées
-----------------
CUDARHSEvaluatorPB          → omega         (B, n_sp)
CUDAJVPEvaluatorPB          → Jv_om         (B, n_sp)
CUDAAdiabaticJVPEvaluatorPB → Jv_TC         (B, n_state)
"""

import numpy as np

_MAX_N_STATE       = 54
_THREADS_PER_BLOCK = 64


# =============================================================================
#  Kernels CUDA
# =============================================================================

@cuda.jit(cache=True)
def _rhs_kernel_pb(
    TC_in, omega_out,
    ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e, ctx_sRre, ctx_sRpe,
    ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f, ctx_sRrf, ctx_sRpf,
    ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
    ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
    n_sp, thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b         = cuda.blockIdx.x
    tid       = cuda.threadIdx.x
    n_threads = cuda.blockDim.x
    if b >= TC_in.shape[0]:
        return
    n_state = n_sp + 1
    sh_xk = cuda.shared.array(_MAX_N_STATE, float64)
    sh_om = cuda.shared.array(_MAX_N_STATE, float64)

    i = tid
    while i < n_state:
        sh_xk[i] = TC_in[b, i]
        i += n_threads
    cuda.syncthreads()

    T_loc = sh_xk[0]
    e_kf  = ctx_kf_e[b];  e_Kc  = ctx_Kc_e[b]
    e_cr  = ctx_cr_e[b];  e_cp  = ctx_cp_e[b];  e_dG = ctx_dG0_e[b]
    e_sRr = ctx_sRre[b];  e_sRp = ctx_sRpe[b]
    tb_kf = ctx_kf_tb[b]; tb_Kc = ctx_Kc_tb[b]
    tb_cr = ctx_cr_tb[b]; tb_cp = ctx_cp_tb[b]; tb_dG = ctx_dG0_tb[b]
    tb_sRr= ctx_sRrtb[b]; tb_sRp= ctx_sRptb[b]
    tb_bRR= ctx_bRR_tb[b];tb_M  = ctx_M_tb[b]
    f_kf  = ctx_kf_f[b];  f_Kc  = ctx_Kc_f[b]
    f_cr  = ctx_cr_f[b];  f_cp  = ctx_cp_f[b];  f_dG = ctx_dG0_f[b]
    f_sRr = ctx_sRrf[b];  f_sRp = ctx_sRpf[b]
    f_k0  = ctx_k0_f[b];  f_ki  = ctx_kinf_f[b]
    f_M   = ctx_M_f[b];   f_Pr  = ctx_Pr_f[b]
    f_F   = ctx_F_f[b];   f_kb  = ctx_kb_f[b]
    f_Fc  = ctx_Fc_f[b];  f_lFc = ctx_lFc_f[b]
    f_Cp  = ctx_Cp_f[b];  f_Np  = ctx_Np_f[b]
    f_f1  = ctx_f1_f[b];  f_exp = ctx_exp_f[b]

    _eval_context_pb(
        tid, n_threads, T_loc, sh_xk[1:],
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
    )
    _eval_rhs_pb(
        tid, n_threads, sh_om,
        e_kf,  e_Kc,  e_cr,  e_cp,  eOW,
        tb_kf, tb_Kc, tb_cr, tb_cp, tb_bRR, tb_M, tbOW,
        f_kf,  f_Kc,  f_cr,  f_cp,  fOW,
        n_sp, n_e, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO,
        n_f, fRI, fRS, fRO, fPI, fPS, fPO,
    )
    i = tid
    while i < n_sp:
        omega_out[b, i] = sh_om[i]
        i += n_threads
    cuda.syncthreads()


# =============================================================================
#  Classe de base — allocation partagée
# =============================================================================

class _BaseEvaluatorPB:
    THREADS_PER_BLOCK = _THREADS_PER_BLOCK

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

    def _prepare(self, T_arr, C_arr):
        T_arr = np.asarray(T_arr, dtype=self._dtype)
        C_arr = np.asarray(C_arr, dtype=self._dtype)
        B     = T_arr.shape[0]
        nst   = self._n_state
        if self._B_alloc != B:
            self._alloc_context(B)
        TC = np.empty((B, nst), dtype=self._dtype)
        TC[:, 0]  = T_arr
        TC[:, 1:] = C_arr
        return B, cuda.to_device(TC)

    def _mech_args(self):
        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff
        cx = self._ctx
        return (
            cx['kf_e'],  cx['Kc_e'],  cx['cr_e'],  cx['cp_e'],  cx['dG0_e'],
            cx['sRre'],  cx['sRpe'],
            cx['kf_tb'], cx['Kc_tb'], cx['cr_tb'], cx['cp_tb'], cx['dG0_tb'],
            cx['sRrtb'], cx['sRptb'], cx['bRR_tb'], cx['M_tb'],
            cx['kf_f'],  cx['Kc_f'],  cx['cr_f'],  cx['cp_f'],  cx['dG0_f'],
            cx['sRrf'],  cx['sRpf'],
            cx['k0_f'],  cx['kinf_f'], cx['M_f'],  cx['Pr_f'],  cx['F_f'],
            cx['kb_f'],  cx['Fc_f'],  cx['lFc_f'], cx['Cp_f'],  cx['Np_f'],
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

    def _launch(self, kernel, B, *args):
        kernel[B, self.THREADS_PER_BLOCK](*args)
        cuda.synchronize()


# =============================================================================
#  CUDARHSEvaluatorPB
# =============================================================================

class CUDARHSEvaluatorPB(_BaseEvaluatorPB):
    """
    Évalue omega = d(C)/dt (per-block).

    Usage
    -----
    ev    = CUDARHSEvaluatorPB(mech)
    omega = ev.evaluate(T_arr, C_arr)   # (B, n_sp)
    """

    def evaluate(self, T_arr, C_arr):
        B, d_TC = self._prepare(T_arr, C_arr)
        d_om    = cuda.device_array((B, self._n_sp), dtype=self._dtype)
        self._launch(_rhs_kernel_pb, B, d_TC, d_om, *self._mech_args())
        return d_om.copy_to_host()

