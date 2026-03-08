"""
chemistry_kernels_eval.py — Kernels d'évaluation standalone pour validation.

Un kernel par réacteur (version per-thread), utilisé uniquement dans les tests
pour calculer les taux de réaction et omega sur un batch de conditions (T, C).

Ces kernels ne sont PAS utilisés dans l'intégrateur — ils servent uniquement
à reproduire la section RR_and_source_terms du script de validation torch.

Architecture: 1 thread = 1 réacteur (pas de parallélisme intra-bloc).
"""

import math
import numpy as np
from numba import cuda, float64, int32

R_CONST  = 8.314_462_618_153_24
PREF     = 101325.0
EPS      = 1e-300

THREADS_EVAL = 128  # threads par bloc pour les kernels d'évaluation


# =============================================================================
#  Device helpers (copiés depuis chemistry_kernels_pb pour autonomie)
# =============================================================================

@cuda.jit(device=True, inline=True)
def _load_coef7_ev(T, sp, Tmid, coef_low, coef_high, out7):
    base = sp * 7
    arr  = coef_low if T <= Tmid[sp] else coef_high
    for k in range(7):
        out7[k] = arr[base + k]

@cuda.jit(device=True, inline=True)
def _h_RT_ev(T, c7):
    T2 = T*T; T3 = T2*T; T4 = T3*T
    return (c7[0] + c7[1]*T/2.0 + c7[2]*T2/3.0
            + c7[3]*T3/4.0 + c7[4]*T4/5.0 + c7[5]/T)

@cuda.jit(device=True, inline=True)
def _s_R_ev(T, c7):
    T2 = T*T; T3 = T2*T; T4 = T3*T
    return (c7[0]*math.log(T) + c7[1]*T + c7[2]*T2/2.0
            + c7[3]*T3/3.0 + c7[4]*T4/4.0 + c7[6])

@cuda.jit(device=True, inline=True)
def _cp_R_ev(T, c7):
    T2 = T*T; T3 = T2*T; T4 = T3*T
    return c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4

@cuda.jit(device=True, inline=True)
def _kf_arr_ev(T, A, n, Ea):
    return A * (T**n) * math.exp(-Ea / (R_CONST * T))

@cuda.jit(device=True, inline=True)
def _lcp_ev(C, i0, i1, sp_idx, sto):
    s = float64(0.0)
    for k in range(i0, i1):
        c = C[sp_idx[k]]
        s += sto[k] * math.log(c if c > EPS else EPS)
    return s


# =============================================================================
#  Kernel : évaluation des taux de réaction ET de omega
#
#  Sorties :
#    RR_e   (B, n_e)    — taux nets des réactions élémentaires  [mol/m³/s]
#    RR_tb  (B, n_tb)   — taux nets des réactions trois-corps
#    RR_f   (B, n_f)    — taux nets des réactions falloff
#    omega  (B, n_sp)   — taux de production des espèces        [mol/m³/s]
# =============================================================================

@cuda.jit
def eval_rr_and_omega_kernel(
    T_arr, C_arr,           # (B,) et (B, n_sp)
    RR_e, RR_tb, RR_f,      # sorties RR par type  (B, n_rxn)
    omega,                  # sortie omega          (B, n_sp)
    n_sp,
    thL, thH, thM,
    # Elementary
    n_e, eA, en, eEa, eOW, eDn,
    eRI, eRS, eRO, ePI, ePS, ePO,
    # Three-body
    n_tb, tbA, tbn, tbEa, tbOW, tbDn,
    tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    # Falloff
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= T_arr.shape[0]:
        return

    T = T_arr[b]
    c7 = cuda.local.array(7, float64)

    # ── Zéro omega ────────────────────────────────────────────────────
    for s in range(n_sp):
        omega[b, s] = float64(0.0)

    # ── Élémentaires ──────────────────────────────────────────────────
    for j in range(n_e):
        kf = _kf_arr_ev(T, eA[j], en[j], eEa[j])
        dG = float64(0.0)
        for k in range(eRO[j], eRO[j+1]):
            sp = eRI[k]
            _load_coef7_ev(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT_ev(T, c7)
            s_s = R_CONST * _s_R_ev(T, c7)
            dG -= eRS[k] * (h_s - T*s_s)
        for k in range(ePO[j], ePO[j+1]):
            sp = ePI[k]
            _load_coef7_ev(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT_ev(T, c7)
            s_s = R_CONST * _s_R_ev(T, c7)
            dG += ePS[k] * (h_s - T*s_s)
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** eDn[j])
        lr = _lcp_ev(C_arr[b], eRO[j], eRO[j+1], eRI, eRS)
        lp = _lcp_ev(C_arr[b], ePO[j], ePO[j+1], ePI, ePS)
        cr = math.exp(lr); cp = math.exp(lp)
        ow = eOW[j]
        RR = kf * (cr - (1.0 - ow) * cp / Kc)
        RR_e[b, j] = RR
        for k in range(eRO[j], eRO[j+1]):
            omega[b, eRI[k]] -= eRS[k] * RR
        for k in range(ePO[j], ePO[j+1]):
            omega[b, ePI[k]] += ePS[k] * RR

    # ── Trois-corps ───────────────────────────────────────────────────
    for j in range(n_tb):
        kf = _kf_arr_ev(T, tbA[j], tbn[j], tbEa[j])
        dG = float64(0.0)
        for k in range(tbRO[j], tbRO[j+1]):
            sp = tbRI[k]
            _load_coef7_ev(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT_ev(T, c7)
            s_s = R_CONST * _s_R_ev(T, c7)
            dG -= tbRS[k] * (h_s - T*s_s)
        for k in range(tbPO[j], tbPO[j+1]):
            sp = tbPI[k]
            _load_coef7_ev(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT_ev(T, c7)
            s_s = R_CONST * _s_R_ev(T, c7)
            dG += tbPS[k] * (h_s - T*s_s)
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** tbDn[j])
        lr = _lcp_ev(C_arr[b], tbRO[j], tbRO[j+1], tbRI, tbRS)
        lp = _lcp_ev(C_arr[b], tbPO[j], tbPO[j+1], tbPI, tbPS)
        cr = math.exp(lr); cp = math.exp(lp)
        M = float64(0.0)
        for s in range(n_sp):
            M += C_arr[b, s] * tbEff[j * n_sp + s]
        ow = tbOW[j]
        RR = kf * (cr - (1.0 - ow) * cp / Kc) * M
        RR_tb[b, j] = RR
        for k in range(tbRO[j], tbRO[j+1]):
            omega[b, tbRI[k]] -= tbRS[k] * RR
        for k in range(tbPO[j], tbPO[j+1]):
            omega[b, tbPI[k]] += tbPS[k] * RR

    # ── Falloff ───────────────────────────────────────────────────────
    for j in range(n_f):
        k0_v   = _kf_arr_ev(T, fA0[j], fb0[j], fE0[j])
        kinf_v = _kf_arr_ev(T, fAi[j], fbi[j], fEi[j])
        M = float64(0.0)
        for s in range(n_sp):
            M += C_arr[b, s] * fEff[j * n_sp + s]
        ki_s = kinf_v if kinf_v > EPS else EPS
        Pr   = (k0_v * M) / ki_s
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
            F_v = float64(1.0)
        kbase = ki_s * (Pr / (1.0 + Pr))
        kf_v  = kbase * F_v
        dG = float64(0.0)
        for k in range(fRO[j], fRO[j+1]):
            sp = fRI[k]
            _load_coef7_ev(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT_ev(T, c7)
            s_s = R_CONST * _s_R_ev(T, c7)
            dG -= fRS[k] * (h_s - T*s_s)
        for k in range(fPO[j], fPO[j+1]):
            sp = fPI[k]
            _load_coef7_ev(T, sp, thM, thL, thH, c7)
            h_s = R_CONST * T * _h_RT_ev(T, c7)
            s_s = R_CONST * _s_R_ev(T, c7)
            dG += fPS[k] * (h_s - T*s_s)
        Kp = math.exp(-dG / (R_CONST * T))
        Kc = Kp * ((PREF / (R_CONST * T)) ** fDn[j])
        lr = _lcp_ev(C_arr[b], fRO[j], fRO[j+1], fRI, fRS)
        lp = _lcp_ev(C_arr[b], fPO[j], fPO[j+1], fPI, fPS)
        cr = math.exp(lr); cp = math.exp(lp)
        ow = fOW[j]
        RR = kf_v * (cr - (1.0 - ow) * cp / Kc)
        RR_f[b, j] = RR
        for k in range(fRO[j], fRO[j+1]):
            omega[b, fRI[k]] -= fRS[k] * RR
        for k in range(fPO[j], fPO[j+1]):
            omega[b, fPI[k]] += fPS[k] * RR


# =============================================================================
#  Wrapper Python
# =============================================================================

class CUDAReactionEvaluator:
    """
    Évalue les taux de réaction et omega sur un batch de (T, C).

    Usage
    -----
    evaluator = CUDAReactionEvaluator(mech)
    RR_e, RR_tb, RR_f, omega = evaluator.evaluate(T_np, C_np)
    # T_np : (B,)  float64
    # C_np : (B, n_sp) float64  en mol/m³
    # RR_*  : (B, n_rxn) float64  en mol/m³/s
    # omega : (B, n_sp) float64  en mol/m³/s
    """

    def __init__(self, mech, dtype=np.float64):
        self.mech  = mech
        self.dtype = dtype
        self._d    = mech.to_cuda_arrays()
        self._args = self._pack_args()

    def _pack_args(self):
        d  = self._d
        m  = self.mech
        e  = m.elementary
        tb = m.threebody
        fa = m.falloff
        dt = self.dtype

        def _g(k):  return d.get(k, cuda.device_array((1,), dt))
        def _gi(k): return d.get(k, cuda.device_array((1,), np.int32))

        return [
            np.int32(m.n_species),
            d['thermo_coef_low'], d['thermo_coef_high'], d['thermo_Tmid'],
            # Elementary
            np.int32(e.n_reactions if e else 0),
            _g('e_A'), _g('e_n'), _g('e_Ea'), _g('e_oneWay'), _g('e_delta_n'),
            _gi('e_react_idx'), _g('e_react_sto'), _gi('e_react_offset'),
            _gi('e_prod_idx'),  _g('e_prod_sto'),  _gi('e_prod_offset'),
            # Three-body
            np.int32(tb.n_reactions if tb else 0),
            _g('tb_A'), _g('tb_n'), _g('tb_Ea'), _g('tb_oneWay'), _g('tb_delta_n'),
            _gi('tb_react_idx'), _g('tb_react_sto'), _gi('tb_react_offset'),
            _gi('tb_prod_idx'),  _g('tb_prod_sto'),  _gi('tb_prod_offset'),
            _g('tb_eff_matrix'),
            # Falloff
            np.int32(fa.n_reactions if fa else 0),
            _g('f_A_k0'),   _g('f_b_k0'),   _g('f_Ea_k0'),
            _g('f_A_kinf'), _g('f_b_kinf'), _g('f_Ea_kinf'),
            _g('f_oneWay'), _g('f_delta_n'),
            _g('f_troe_A'), _g('f_troe_T1'), _g('f_troe_T2'), _g('f_troe_T3'),
            _gi('f_is_troe'),
            _gi('f_react_idx'), _g('f_react_sto'), _gi('f_react_offset'),
            _gi('f_prod_idx'),  _g('f_prod_sto'),  _gi('f_prod_offset'),
            _g('f_eff_matrix'),
        ]

    def evaluate(self, T_np: np.ndarray, C_np: np.ndarray):
        """
        Parameters
        ----------
        T_np : (B,) float64
        C_np : (B, n_sp) float64 — concentrations en mol/m³

        Returns
        -------
        RR_e   : (B, n_e)   float64
        RR_tb  : (B, n_tb)  float64
        RR_f   : (B, n_f)   float64
        omega  : (B, n_sp)  float64
        """
        m   = self.mech
        B   = T_np.shape[0]
        dt  = self.dtype
        e   = m.elementary
        tb  = m.threebody
        fa  = m.falloff

        n_e  = e.n_reactions  if e  else 0
        n_tb = tb.n_reactions if tb else 0
        n_f  = fa.n_reactions if fa else 0

        d_T     = cuda.to_device(np.ascontiguousarray(T_np,  dtype=dt))
        d_C     = cuda.to_device(np.ascontiguousarray(C_np,  dtype=dt))
        d_RR_e  = cuda.device_array((B, max(n_e,  1)), dtype=dt)
        d_RR_tb = cuda.device_array((B, max(n_tb, 1)), dtype=dt)
        d_RR_f  = cuda.device_array((B, max(n_f,  1)), dtype=dt)
        d_omega = cuda.device_array((B, m.n_species),  dtype=dt)

        n_blocks = math.ceil(B / THREADS_EVAL)
        eval_rr_and_omega_kernel[n_blocks, THREADS_EVAL](
            d_T, d_C,
            d_RR_e, d_RR_tb, d_RR_f,
            d_omega,
            *self._args,
        )
        cuda.synchronize()

        RR_e  = d_RR_e.copy_to_host()[:, :n_e]
        RR_tb = d_RR_tb.copy_to_host()[:, :n_tb]
        RR_f  = d_RR_f.copy_to_host()[:, :n_f]
        omega = d_omega.copy_to_host()

        return RR_e, RR_tb, RR_f, omega


import math  # already imported at top, kept for clarity