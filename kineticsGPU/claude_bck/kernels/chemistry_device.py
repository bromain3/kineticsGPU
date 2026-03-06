"""
Per-thread CUDA kernels for chemical kinetics evaluation.

FIX (v2):
  thermo_coef_low / thermo_coef_high are stored as 2-D arrays (n_species, 7).
  Numba CUDA rejects flat indexing `coef_low[sp*7 + k]` when the array is 2-D
  because the resulting type is ambiguous for setitem into a local array.
  Two compatible approaches:
    (a) Store thermo arrays as 1-D (n_species*7,) and use flat index sp*7+k  ← chosen
    (b) Use 2-D indexing coef_low[sp, k] directly into coef7
  We choose (a) — 1-D flat storage — because it avoids any stride arithmetic
  and is unambiguous for Numba's type inference.
  The MechanismData.to_cuda_arrays() method must flatten these arrays before
  sending to device (see note at bottom of file).

Architecture: per-thread (one CUDA thread = one reactor/IVP).
"""

import math
import numpy as np
from numba import cuda, float64, int32
import numba


# ─── Physical constants ───────────────────────────────────────────────────────
R_CONST  = 8.314_462_618_153_24
PREF     = 101325.0
EPS      = 1e-300
LN10     = math.log(10.0)
INV_LN10 = 1.0 / LN10


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE FUNCTIONS: NASA7 thermodynamics
#  NOTE: coef_low / coef_high are FLAT 1-D arrays of shape (n_species * 7,)
#        indexed as coef[sp * 7 + k].  Pass them flattened from Python side.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _load_coef7(T, sp, Tmid_s, coef_low_flat, coef_high_flat, out7):
    """
    Load NASA7 coefficients for species sp into local array out7[7].
    coef_low_flat / coef_high_flat are 1-D, length n_species*7.
    """
    base = sp * 7
    if T <= Tmid_s[sp]:
        for k in range(7):
            out7[k] = coef_low_flat[base + k]
    else:
        for k in range(7):
            out7[k] = coef_high_flat[base + k]


@cuda.jit(device=True, inline=True)
def _h_species(T, coef7):
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    hRT = (coef7[0]
           + coef7[1] * T  / 2.0
           + coef7[2] * T2 / 3.0
           + coef7[3] * T3 / 4.0
           + coef7[4] * T4 / 5.0
           + coef7[5] / T)
    return R_CONST * T * hRT


@cuda.jit(device=True, inline=True)
def _s_species_Pref(T, coef7):
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    sR = (coef7[0] * math.log(T)
          + coef7[1] * T
          + coef7[2] * T2 / 2.0
          + coef7[3] * T3 / 3.0
          + coef7[4] * T4 / 4.0
          + coef7[6])
    return R_CONST * sR


@cuda.jit(device=True, inline=True)
def _cp_species(T, coef7):
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    cpR = (coef7[0] + coef7[1]*T + coef7[2]*T2 + coef7[3]*T3 + coef7[4]*T4)
    return R_CONST * cpR


@cuda.jit(device=True, inline=True)
def _kf_arrhenius(T, A, n, Ea):
    return A * (T ** n) * math.exp(-Ea / (R_CONST * T))


@cuda.jit(device=True, inline=True)
def _Kc_from_deltaG0(T, deltaG0, delta_n):
    Kp = math.exp(-deltaG0 / (R_CONST * T))
    return Kp * ((PREF / (R_CONST * T)) ** delta_n)


@cuda.jit(device=True, inline=True)
def _log_conc_product(C, offset_start, offset_end, sp_idx, sto):
    s = 0.0
    for k in range(offset_start, offset_end):
        c = C[sp_idx[k]]
        if c < EPS:
            c = EPS
        s += sto[k] * math.log(c)
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE FUNCTION: full RHS + context
#  C is a 1-D local array of length n_species (NOT a 2-D global slice).
#  thermo_coef_low / thermo_coef_high are 1-D flat arrays (n_species*7,).
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _eval_rhs_and_context(
    T, C,
    ctx_kf_e, ctx_Kc_e,
    ctx_conc_r_e, ctx_conc_p_e,
    ctx_RR_e,
    ctx_deltaG0_e, ctx_s_r_e, ctx_s_p_e,
    ctx_kf_tb, ctx_Kc_tb,
    ctx_conc_r_tb, ctx_conc_p_tb,
    ctx_RR_tb, ctx_baseRR_tb, ctx_M_tb,
    ctx_deltaG0_tb, ctx_s_r_tb, ctx_s_p_tb,
    ctx_kf_fall, ctx_Kc_fall,
    ctx_conc_r_fall, ctx_conc_p_fall,
    ctx_RR_fall,
    ctx_k0_fall, ctx_kinf_fall, ctx_M_fall,
    ctx_Pr_fall, ctx_F_fall, ctx_kbase_fall,
    ctx_Fcent_fall, ctx_logFcent_fall,
    ctx_C_par_fall, ctx_N_par_fall,
    ctx_f1_fall, ctx_exp_fall,
    ctx_deltaG0_fall, ctx_s_r_fall, ctx_s_p_fall,
    omega,
    n_species,
    thermo_coef_low,   # 1-D flat (n_species*7,)
    thermo_coef_high,  # 1-D flat (n_species*7,)
    thermo_Tmid,
    n_e, e_A, e_n, e_Ea, e_oneWay, e_delta_n,
    e_react_idx, e_react_sto, e_react_offset,
    e_prod_idx,  e_prod_sto,  e_prod_offset,
    e_idx_r_scatter, e_idx_p_scatter,
    n_tb, tb_A, tb_n, tb_Ea, tb_oneWay, tb_delta_n,
    tb_react_idx, tb_react_sto, tb_react_offset,
    tb_prod_idx,  tb_prod_sto,  tb_prod_offset,
    tb_idx_r_scatter, tb_idx_p_scatter,
    tb_eff_matrix,
    n_fall, fall_A_k0, fall_b_k0, fall_Ea_k0,
    fall_A_kinf, fall_b_kinf, fall_Ea_kinf,
    fall_oneWay, fall_delta_n,
    fall_troe_A, fall_troe_T1, fall_troe_T2, fall_troe_T3,
    fall_is_troe,
    fall_react_idx, fall_react_sto, fall_react_offset,
    fall_prod_idx,  fall_prod_sto,  fall_prod_offset,
    fall_idx_r_scatter, fall_idx_p_scatter,
    fall_eff_matrix,
):
    coef7 = cuda.local.array(7, dtype=float64)

    for s in range(n_species):
        omega[s] = 0.0

    # ── Elementary ────────────────────────────────────────────────────────
    for j in range(n_e):
        kf = _kf_arrhenius(T, e_A[j], e_n[j], e_Ea[j])
        ctx_kf_e[j] = kf

        dG0 = 0.0
        for k in range(e_react_offset[j], e_react_offset[j+1]):
            sp = e_react_idx[k]
            nu = e_react_sto[k]
            _load_coef7(T, sp, thermo_Tmid, thermo_coef_low, thermo_coef_high, coef7)
            h_sp = _h_species(T, coef7)
            s_sp = _s_species_Pref(T, coef7)
            ctx_s_r_e[k] = s_sp
            dG0 -= nu * (h_sp - T * s_sp)

        for k in range(e_prod_offset[j], e_prod_offset[j+1]):
            sp = e_prod_idx[k]
            nu = e_prod_sto[k]
            _load_coef7(T, sp, thermo_Tmid, thermo_coef_low, thermo_coef_high, coef7)
            h_sp = _h_species(T, coef7)
            s_sp = _s_species_Pref(T, coef7)
            ctx_s_p_e[k] = s_sp
            dG0 += nu * (h_sp - T * s_sp)

        ctx_deltaG0_e[j] = dG0
        Kc = _Kc_from_deltaG0(T, dG0, e_delta_n[j])
        ctx_Kc_e[j] = Kc

        log_r = _log_conc_product(C, e_react_offset[j], e_react_offset[j+1],
                                   e_react_idx, e_react_sto)
        log_p = _log_conc_product(C, e_prod_offset[j], e_prod_offset[j+1],
                                   e_prod_idx, e_prod_sto)
        conc_r = math.exp(log_r)
        conc_p = math.exp(log_p)
        ctx_conc_r_e[j] = conc_r
        ctx_conc_p_e[j] = conc_p

        oneWay = e_oneWay[j]
        RR = kf * (conc_r - (1.0 - oneWay) * conc_p / Kc)
        ctx_RR_e[j] = RR

        for k in range(e_react_offset[j], e_react_offset[j+1]):
            omega[e_react_idx[k]] -= e_react_sto[k] * RR
        for k in range(e_prod_offset[j], e_prod_offset[j+1]):
            omega[e_prod_idx[k]] += e_prod_sto[k] * RR

    # ── Three-body ────────────────────────────────────────────────────────
    for j in range(n_tb):
        kf = _kf_arrhenius(T, tb_A[j], tb_n[j], tb_Ea[j])
        ctx_kf_tb[j] = kf

        dG0 = 0.0
        for k in range(tb_react_offset[j], tb_react_offset[j+1]):
            sp = tb_react_idx[k]
            nu = tb_react_sto[k]
            _load_coef7(T, sp, thermo_Tmid, thermo_coef_low, thermo_coef_high, coef7)
            h_sp = _h_species(T, coef7)
            s_sp = _s_species_Pref(T, coef7)
            ctx_s_r_tb[k] = s_sp
            dG0 -= nu * (h_sp - T * s_sp)

        for k in range(tb_prod_offset[j], tb_prod_offset[j+1]):
            sp = tb_prod_idx[k]
            nu = tb_prod_sto[k]
            _load_coef7(T, sp, thermo_Tmid, thermo_coef_low, thermo_coef_high, coef7)
            h_sp = _h_species(T, coef7)
            s_sp = _s_species_Pref(T, coef7)
            ctx_s_p_tb[k] = s_sp
            dG0 += nu * (h_sp - T * s_sp)

        ctx_deltaG0_tb[j] = dG0
        Kc = _Kc_from_deltaG0(T, dG0, tb_delta_n[j])
        ctx_Kc_tb[j] = Kc

        log_r = _log_conc_product(C, tb_react_offset[j], tb_react_offset[j+1],
                                   tb_react_idx, tb_react_sto)
        log_p = _log_conc_product(C, tb_prod_offset[j], tb_prod_offset[j+1],
                                   tb_prod_idx, tb_prod_sto)
        conc_r = math.exp(log_r)
        conc_p = math.exp(log_p)
        ctx_conc_r_tb[j] = conc_r
        ctx_conc_p_tb[j] = conc_p

        M_val = 0.0
        for s in range(n_species):
            M_val += C[s] * tb_eff_matrix[j * n_species + s]
        ctx_M_tb[j] = M_val

        oneWay = tb_oneWay[j]
        baseRR = kf * (conc_r - (1.0 - oneWay) * conc_p / Kc)
        ctx_baseRR_tb[j] = baseRR
        RR = baseRR * M_val
        ctx_RR_tb[j] = RR

        for k in range(tb_react_offset[j], tb_react_offset[j+1]):
            omega[tb_react_idx[k]] -= tb_react_sto[k] * RR
        for k in range(tb_prod_offset[j], tb_prod_offset[j+1]):
            omega[tb_prod_idx[k]] += tb_prod_sto[k] * RR

    # ── Falloff ───────────────────────────────────────────────────────────
    for j in range(n_fall):
        k0   = _kf_arrhenius(T, fall_A_k0[j],   fall_b_k0[j],   fall_Ea_k0[j])
        kinf = _kf_arrhenius(T, fall_A_kinf[j],  fall_b_kinf[j], fall_Ea_kinf[j])
        ctx_k0_fall[j]   = k0
        ctx_kinf_fall[j] = kinf

        M_val = 0.0
        for s in range(n_species):
            M_val += C[s] * fall_eff_matrix[j * n_species + s]
        ctx_M_fall[j] = M_val

        kinf_safe = kinf if kinf > EPS else EPS
        Pr = (k0 * M_val) / kinf_safe
        ctx_Pr_fall[j] = Pr

        if fall_is_troe[j]:
            tA  = fall_troe_A[j]
            tT1 = fall_troe_T1[j]
            tT2 = fall_troe_T2[j]
            tT3 = fall_troe_T3[j]
            Fcent = 0.0
            if tT1 > EPS: Fcent += tA * math.exp(-T / tT1)
            if tT3 > EPS: Fcent += (1.0 - tA) * math.exp(-T / tT3)
            if tT2 > EPS: Fcent += math.exp(-tT2 / T)
            if Fcent < EPS: Fcent = EPS
            logFcent = math.log10(Fcent)
            C_par = -0.4 - 0.67 * logFcent
            N_par =  0.75 - 1.27 * logFcent
            Pr_safe = Pr if Pr > EPS else EPS
            logPr = math.log10(Pr_safe)
            num   = logPr + C_par
            denom = N_par - 0.14 * (logPr + C_par)
            if abs(denom) < EPS:
                denom = EPS if denom >= 0.0 else -EPS
            f1       = num / denom
            exponent = logFcent / (1.0 + f1 * f1)
            F_val    = 10.0 ** exponent
        else:
            Fcent    = 1.0; logFcent = 0.0
            C_par    = 0.0; N_par    = 1.0
            f1       = 0.0; exponent = 0.0
            F_val    = 1.0

        ctx_Fcent_fall[j]    = Fcent
        ctx_logFcent_fall[j] = logFcent
        ctx_C_par_fall[j]    = C_par
        ctx_N_par_fall[j]    = N_par
        ctx_f1_fall[j]       = f1
        ctx_exp_fall[j]      = exponent
        ctx_F_fall[j]        = F_val

        Pr_term  = Pr / (1.0 + Pr)
        kbase    = kinf_safe * Pr_term
        kf_val   = kbase * F_val
        ctx_kbase_fall[j] = kbase
        ctx_kf_fall[j]    = kf_val

        dG0 = 0.0
        for k in range(fall_react_offset[j], fall_react_offset[j+1]):
            sp = fall_react_idx[k]
            nu = fall_react_sto[k]
            _load_coef7(T, sp, thermo_Tmid, thermo_coef_low, thermo_coef_high, coef7)
            h_sp = _h_species(T, coef7)
            s_sp = _s_species_Pref(T, coef7)
            ctx_s_r_fall[k] = s_sp
            dG0 -= nu * (h_sp - T * s_sp)

        for k in range(fall_prod_offset[j], fall_prod_offset[j+1]):
            sp = fall_prod_idx[k]
            nu = fall_prod_sto[k]
            _load_coef7(T, sp, thermo_Tmid, thermo_coef_low, thermo_coef_high, coef7)
            h_sp = _h_species(T, coef7)
            s_sp = _s_species_Pref(T, coef7)
            ctx_s_p_fall[k] = s_sp
            dG0 += nu * (h_sp - T * s_sp)

        ctx_deltaG0_fall[j] = dG0
        Kc = _Kc_from_deltaG0(T, dG0, fall_delta_n[j])
        ctx_Kc_fall[j] = Kc

        log_r = _log_conc_product(C, fall_react_offset[j], fall_react_offset[j+1],
                                   fall_react_idx, fall_react_sto)
        log_p = _log_conc_product(C, fall_prod_offset[j], fall_prod_offset[j+1],
                                   fall_prod_idx, fall_prod_sto)
        conc_r = math.exp(log_r)
        conc_p = math.exp(log_p)
        ctx_conc_r_fall[j] = conc_r
        ctx_conc_p_fall[j] = conc_p

        oneWay = fall_oneWay[j]
        RR = kf_val * (conc_r - (1.0 - oneWay) * conc_p / Kc)
        ctx_RR_fall[j] = RR

        for k in range(fall_react_offset[j], fall_react_offset[j+1]):
            omega[fall_react_idx[k]] -= fall_react_sto[k] * RR
        for k in range(fall_prod_offset[j], fall_prod_offset[j+1]):
            omega[fall_prod_idx[k]] += fall_prod_sto[k] * RR


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE FUNCTION: JVP
#  Same thermo convention: flat 1-D arrays.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _eval_jvp(
    T, C,
    v_T, v_C,
    ctx_kf_e, ctx_Kc_e,
    ctx_conc_r_e, ctx_conc_p_e,
    ctx_deltaG0_e, ctx_s_r_e, ctx_s_p_e,
    ctx_kf_tb, ctx_Kc_tb,
    ctx_conc_r_tb, ctx_conc_p_tb,
    ctx_baseRR_tb, ctx_M_tb,
    ctx_deltaG0_tb, ctx_s_r_tb, ctx_s_p_tb,
    ctx_kf_fall, ctx_Kc_fall,
    ctx_conc_r_fall, ctx_conc_p_fall,
    ctx_k0_fall, ctx_kinf_fall, ctx_M_fall,
    ctx_Pr_fall, ctx_F_fall, ctx_kbase_fall,
    ctx_Fcent_fall, ctx_logFcent_fall,
    ctx_C_par_fall, ctx_N_par_fall,
    ctx_f1_fall, ctx_exp_fall,
    ctx_deltaG0_fall, ctx_s_r_fall, ctx_s_p_fall,
    Jv_omega,
    n_species,
    thermo_coef_low,   # 1-D flat
    thermo_coef_high,  # 1-D flat
    thermo_Tmid,
    n_e, e_A, e_n, e_Ea, e_oneWay, e_delta_n,
    e_react_idx, e_react_sto, e_react_offset,
    e_prod_idx,  e_prod_sto,  e_prod_offset,
    e_idx_r_scatter, e_idx_p_scatter,
    n_tb, tb_A, tb_n, tb_Ea, tb_oneWay, tb_delta_n,
    tb_react_idx, tb_react_sto, tb_react_offset,
    tb_prod_idx,  tb_prod_sto,  tb_prod_offset,
    tb_idx_r_scatter, tb_idx_p_scatter, tb_eff_matrix,
    n_fall, fall_A_k0, fall_b_k0, fall_Ea_k0,
    fall_A_kinf, fall_b_kinf, fall_Ea_kinf,
    fall_oneWay, fall_delta_n,
    fall_troe_A, fall_troe_T1, fall_troe_T2, fall_troe_T3,
    fall_is_troe,
    fall_react_idx, fall_react_sto, fall_react_offset,
    fall_prod_idx,  fall_prod_sto,  fall_prod_offset,
    fall_idx_r_scatter, fall_idx_p_scatter, fall_eff_matrix,
):
    for s in range(n_species):
        Jv_omega[s] = 0.0

    # ── Elementary ────────────────────────────────────────────────────────
    for j in range(n_e):
        kf     = ctx_kf_e[j]
        Kc     = ctx_Kc_e[j]
        conc_r = ctx_conc_r_e[j]
        conc_p = ctx_conc_p_e[j]
        dG0    = ctx_deltaG0_e[j]

        dkf_dT = kf * (e_n[j] / T + e_Ea[j] / (R_CONST * T * T))

        dS_r = 0.0
        for k in range(e_react_offset[j], e_react_offset[j+1]):
            dS_r += e_react_sto[k] * ctx_s_r_e[k]
        dS_p = 0.0
        for k in range(e_prod_offset[j], e_prod_offset[j+1]):
            dS_p += e_prod_sto[k] * ctx_s_p_e[k]
        dS = dS_p - dS_r
        dlnKc_dT = dS / (R_CONST * T) + dG0 / (R_CONST * T * T) - e_delta_n[j] / T
        dKc_dT = Kc * dlnKc_dT

        factor = 1.0 - e_oneWay[j]
        dRR_dT = (dkf_dT * (conc_r - factor * conc_p / Kc)
                  + kf * factor * conc_p / (Kc * Kc) * dKc_dT)
        dRR_from_T = dRR_dT * v_T

        dRR_from_C = 0.0
        for k in range(e_react_offset[j], e_react_offset[j+1]):
            sp = e_react_idx[k]
            c  = C[sp] if C[sp] > EPS else EPS
            dRR_from_C += kf * conc_r * (e_react_sto[k] / c) * v_C[sp]
        for k in range(e_prod_offset[j], e_prod_offset[j+1]):
            sp = e_prod_idx[k]
            c  = C[sp] if C[sp] > EPS else EPS
            dRR_from_C -= (kf * factor / Kc * conc_p
                           * (e_prod_sto[k] / c) * v_C[sp])

        dRR_v = dRR_from_T + dRR_from_C
        for k in range(e_react_offset[j], e_react_offset[j+1]):
            Jv_omega[e_react_idx[k]] -= e_react_sto[k] * dRR_v
        for k in range(e_prod_offset[j], e_prod_offset[j+1]):
            Jv_omega[e_prod_idx[k]] += e_prod_sto[k] * dRR_v

    # ── Three-body ────────────────────────────────────────────────────────
    for j in range(n_tb):
        kf     = ctx_kf_tb[j]
        Kc     = ctx_Kc_tb[j]
        conc_r = ctx_conc_r_tb[j]
        conc_p = ctx_conc_p_tb[j]
        M      = ctx_M_tb[j]
        baseRR = ctx_baseRR_tb[j]
        dG0    = ctx_deltaG0_tb[j]

        dkf_dT = kf * (tb_n[j] / T + tb_Ea[j] / (R_CONST * T * T))

        dS_r = 0.0
        for k in range(tb_react_offset[j], tb_react_offset[j+1]):
            dS_r += tb_react_sto[k] * ctx_s_r_tb[k]
        dS_p = 0.0
        for k in range(tb_prod_offset[j], tb_prod_offset[j+1]):
            dS_p += tb_prod_sto[k] * ctx_s_p_tb[k]
        dS = dS_p - dS_r
        dlnKc_dT = dS / (R_CONST * T) + dG0 / (R_CONST * T * T) - tb_delta_n[j] / T
        dKc_dT = Kc * dlnKc_dT

        factor = 1.0 - tb_oneWay[j]
        dbaseRR_dT = (dkf_dT * (conc_r - factor * conc_p / Kc)
                      + kf * factor * conc_p / (Kc * Kc) * dKc_dT)
        dRR_from_T = dbaseRR_dT * v_T * M

        dM_v = 0.0
        for s in range(n_species):
            dM_v += v_C[s] * tb_eff_matrix[j * n_species + s]

        dRR_from_C = 0.0
        for k in range(tb_react_offset[j], tb_react_offset[j+1]):
            sp = tb_react_idx[k]
            c  = C[sp] if C[sp] > EPS else EPS
            dRR_from_C += kf * conc_r * (tb_react_sto[k] / c) * v_C[sp] * M
        for k in range(tb_prod_offset[j], tb_prod_offset[j+1]):
            sp = tb_prod_idx[k]
            c  = C[sp] if C[sp] > EPS else EPS
            dRR_from_C -= (kf * factor / Kc * conc_p
                           * (tb_prod_sto[k] / c) * v_C[sp] * M)
        dRR_from_C += baseRR * dM_v
        dRR_v = dRR_from_T + dRR_from_C

        for k in range(tb_react_offset[j], tb_react_offset[j+1]):
            Jv_omega[tb_react_idx[k]] -= tb_react_sto[k] * dRR_v
        for k in range(tb_prod_offset[j], tb_prod_offset[j+1]):
            Jv_omega[tb_prod_idx[k]] += tb_prod_sto[k] * dRR_v

    # ── Falloff ───────────────────────────────────────────────────────────
    for j in range(n_fall):
        kf       = ctx_kf_fall[j]
        Kc       = ctx_Kc_fall[j]
        conc_r   = ctx_conc_r_fall[j]
        conc_p   = ctx_conc_p_fall[j]
        k0       = ctx_k0_fall[j]
        kinf     = ctx_kinf_fall[j]
        M        = ctx_M_fall[j]
        Pr       = ctx_Pr_fall[j]
        F        = ctx_F_fall[j]
        kbase    = ctx_kbase_fall[j]
        Fcent    = ctx_Fcent_fall[j]
        logFcent = ctx_logFcent_fall[j]
        C_par    = ctx_C_par_fall[j]
        N_par    = ctx_N_par_fall[j]
        f1       = ctx_f1_fall[j]
        exponent = ctx_exp_fall[j]
        dG0      = ctx_deltaG0_fall[j]

        factor    = 1.0 - fall_oneWay[j]
        kinf_safe = kinf if kinf > EPS else EPS
        pr_ok     = (Pr > EPS) and (M > EPS)

        dk0_dT   = k0   * (fall_b_k0[j]   / T + fall_Ea_k0[j]   / (R_CONST * T * T))
        dkinf_dT = kinf * (fall_b_kinf[j]  / T + fall_Ea_kinf[j] / (R_CONST * T * T))

        dM_v = 0.0
        for s in range(n_species):
            dM_v += v_C[s] * fall_eff_matrix[j * n_species + s]

        if pr_ok:
            dPr_dT  = (dk0_dT * M - k0 * M * dkinf_dT / kinf_safe) / kinf_safe
            dPr_v_C = (k0 / kinf_safe) * dM_v
        else:
            dPr_dT  = 0.0
            dPr_v_C = 0.0

        dS_r = 0.0
        for k in range(fall_react_offset[j], fall_react_offset[j+1]):
            dS_r += fall_react_sto[k] * ctx_s_r_fall[k]
        dS_p = 0.0
        for k in range(fall_prod_offset[j], fall_prod_offset[j+1]):
            dS_p += fall_prod_sto[k] * ctx_s_p_fall[k]
        dS = dS_p - dS_r
        dlnKc_dT = dS / (R_CONST * T) + dG0 / (R_CONST * T * T) - fall_delta_n[j] / T
        dKc_dT = Kc * dlnKc_dT

        conc_expr = conc_r - factor * conc_p / Kc

        if fall_is_troe[j]:
            tA  = fall_troe_A[j]
            tT1 = fall_troe_T1[j]
            tT2 = fall_troe_T2[j]
            tT3 = fall_troe_T3[j]
            dFcent_dT = 0.0
            if tT1 > EPS: dFcent_dT += tA * (-1.0/tT1) * math.exp(-T/tT1)
            if tT3 > EPS: dFcent_dT += (1.0-tA) * (-1.0/tT3) * math.exp(-T/tT3)
            if tT2 > EPS: dFcent_dT += math.exp(-tT2/T) * (tT2 / (T * T))

            fcent_ok = Fcent > EPS
            dlogFcent_dT = (dFcent_dT / Fcent * INV_LN10) if fcent_ok else 0.0
            dCpar_dT = -0.67 * dlogFcent_dT
            dNpar_dT = -1.27 * dlogFcent_dT

            Pr_safe      = Pr if Pr > EPS else EPS
            dlogPr_dPr   = (1.0 / (Pr_safe * LN10)) if pr_ok else 0.0
            logPr_val    = math.log10(Pr_safe) if pr_ok else 0.0
            numer_f1     = logPr_val + C_par
            denom_f      = N_par - 0.14 * numer_f1
            abs_df       = abs(denom_f)
            denom_safe   = denom_f if abs_df > EPS else (EPS if denom_f >= 0.0 else -EPS)
            denom_sq     = denom_safe * denom_safe
            denom_ok     = abs_df > EPS

            df1_dPr = ((dlogPr_dPr * denom_safe + 0.14 * numer_f1 * dlogPr_dPr)
                       / denom_sq) if (denom_ok and pr_ok) else 0.0
            numer_T = dCpar_dT
            denom_T = dNpar_dT - 0.14 * dCpar_dT
            df1_dT  = (numer_T * denom_safe - numer_f1 * denom_T) / denom_sq if denom_ok else 0.0

            f1sq           = f1 * f1
            dexp_dlogFcent = 1.0 / (1.0 + f1sq)
            dexp_df1       = -2.0 * f1 * logFcent / ((1.0 + f1sq) ** 2)
            dexp_dT        = dexp_dlogFcent * dlogFcent_dT + dexp_df1 * df1_dT
            dexp_dPr       = dexp_df1 * df1_dPr
            dF_dT          = F * LN10 * dexp_dT
            dF_dPr         = F * LN10 * dexp_dPr
        else:
            dF_dT  = 0.0
            dF_dPr = 0.0

        Pr_denom    = (1.0 + Pr) * (1.0 + Pr)
        dPrterm_dPr = 1.0 / Pr_denom
        dkbase_dT   = dkinf_dT * (Pr / (1.0 + Pr)) + kinf_safe * dPrterm_dPr * dPr_dT
        dkbase_vC   = kinf_safe * dPrterm_dPr * dPr_v_C

        dkf_dT  = dkbase_dT * F + kbase * (dF_dT + dF_dPr * dPr_dT)
        dkf_vC  = dkbase_vC * F + kbase * dF_dPr * dPr_v_C

        dRR_dT  = dkf_dT * conc_expr + kf * factor * conc_p / (Kc * Kc) * dKc_dT
        dRR_v_T = dRR_dT * v_T

        dRR_conc_v = 0.0
        for k in range(fall_react_offset[j], fall_react_offset[j+1]):
            sp = fall_react_idx[k]
            c  = C[sp] if C[sp] > EPS else EPS
            dRR_conc_v += kf * conc_r * (fall_react_sto[k] / c) * v_C[sp]
        for k in range(fall_prod_offset[j], fall_prod_offset[j+1]):
            sp = fall_prod_idx[k]
            c  = C[sp] if C[sp] > EPS else EPS
            dRR_conc_v -= kf * factor / Kc * conc_p * (fall_prod_sto[k] / c) * v_C[sp]

        dRR_v = dRR_v_T + dRR_conc_v + dkf_vC * conc_expr

        for k in range(fall_react_offset[j], fall_react_offset[j+1]):
            Jv_omega[fall_react_idx[k]] -= fall_react_sto[k] * dRR_v
        for k in range(fall_prod_offset[j], fall_prod_offset[j+1]):
            Jv_omega[fall_prod_idx[k]] += fall_prod_sto[k] * dRR_v


# ─────────────────────────────────────────────────────────────────────────────
#  NOTE FOR mechanism_data.py / to_cuda_arrays()
#
#  thermo_coef_low and thermo_coef_high must be sent to device as FLAT 1-D
#  arrays of shape (n_species * 7,), NOT as (n_species, 7).
#
#  In MechanismData.to_cuda_arrays(), replace:
#      d['thermo_coef_low']  = cuda.to_device(self.thermo.coef_low)   # (N,7)
#  with:
#      d['thermo_coef_low']  = cuda.to_device(
#          self.thermo.coef_low.reshape(-1).copy())                    # (N*7,)
#      d['thermo_coef_high'] = cuda.to_device(
#          self.thermo.coef_high.reshape(-1).copy())                   # (N*7,)
#
#  thermo_Tmid stays as (n_species,) — no change needed.
# ─────────────────────────────────────────────────────────────────────────────