"""
rhs_jvp.py
==========
CUDA kernels (Numba) for chemical kinetics RHS and analytic JVP.

Two parallelization strategies:
  - Per-thread : 1 CUDA thread  <-> 1 reactor cell.
                 All species / reaction loops are sequential inside the thread.
                 Launch config : <<<ceil(ncells/TPB), TPB>>>  (TPB ~ 64-256)

  - Per-block  : 1 CUDA block  <-> 1 reactor cell.
                 Threads in the block cooperate over the n_species / n_reactions
                 dimensions via shared memory and __syncthreads().
                 Launch config : <<<ncells, TPB>>>  with TPB == blockDim.x (~ 64-128)

Both strategies expose exactly the same Python-level API so they are
drop-in replaceable for benchmarking.

Physical conventions (identical to the Torch version):
  - Concentrations C  [mol/m^3]
  - Temperature    T  [K]
  - Arrhenius      kf = A * T^n * exp(-Ea / (R*T))
  - Equilibrium    Kc = Kp * (Pref/(R*T))^delta_n
  - conc_r / conc_p computed in log-space to avoid overflow
  - Three reaction types: elementary, three-body, falloff (Lindemann / Troe)
"""

import math
import numpy as np
from numba import cuda, float64, int32
import numba as nb

# ── Physical constants ─────────────────────────────────────────────────────────
R_GAS  = 8.31446261815324   # J/(mol·K)
PREF   = 101325.0           # Pa  (reference for Kc; note thermo uses 1e5 Pa for s°)
PREF_S = 1e5                # Pa  (reference pressure used in s° NASA polynomials)
EPS    = 1e-300

# ══════════════════════════════════════════════════════════════════════════════
# §1  LOW-LEVEL DEVICE HELPERS  (inlined into every kernel)
# ══════════════════════════════════════════════════════════════════════════════
from numba import float32

@cuda.jit(device=True)
def copysigne_GPU(a, b):
    return math.copysign(float32(a), float32(b))

@cuda.jit(device=True, inline=True)
def _arrhenius(A, n, Ea, T):
    """kf = A * T^n * exp(-Ea/(R*T))"""
    return A * (T ** n) * math.exp(-Ea / (R_GAS * T))

@cuda.jit(device=True, inline=True)
def _h_nasa(coef, T):
    """Molar enthalpy J/mol from 6-coefficient NASA poly (no integration constant)."""
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    return R_GAS * T * (coef[0] + coef[1]*T/2 + coef[2]*T2/3
                        + coef[3]*T3/4 + coef[4]*T4/5 + coef[5]/T)

@cuda.jit(device=True, inline=True)
def _s_nasa(coef, T):
    """Molar entropy J/(mol·K) at Pref_s from 7-coefficient NASA poly."""
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    return R_GAS * (coef[0]*math.log(T) + coef[1]*T + coef[2]*T2/2
                    + coef[3]*T3/3 + coef[4]*T4/4 + coef[6])

@cuda.jit(device=True, inline=True)
def _cv_nasa(coef, T):
    """Molar cv = cp - R  (J/(mol·K))."""
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    cp = R_GAS * (coef[0] + coef[1]*T + coef[2]*T2 + coef[3]*T3 + coef[4]*T4)
    return cp - R_GAS

@cuda.jit(device=True, inline=True)
def _u_nasa(coef, T):
    """Molar internal energy u = h - RT  (J/mol)."""
    return _h_nasa(coef, T) - R_GAS * T

@cuda.jit(device=True, inline=True)
def _select_coef(coef_low, coef_high, Tmid, sp, T):
    """Return pointer to the right coefficient block for species sp."""
    # Numba device functions cannot return arrays; we return an index flag.
    # Caller must test:   use_high = (T > Tmid[sp])
    pass  # inlined at call site for performance


# ══════════════════════════════════════════════════════════════════════════════
# §2  PER-THREAD KERNELS
# ══════════════════════════════════════════════════════════════════════════════
#
# Naming convention:  _pt_*  = per-thread
#
# Memory layout for packed reaction data (passed as 1-D device arrays):
#   react_idx[r_flat]   – species index of the r-th reactant entry
#   react_sto[r_flat]   – stoichiometric coefficient
#   r_start[i]          – first index in react_idx for reaction i
#   r_end[i]            – one-past-last index for reaction i
#   (same for products:  prod_idx / prod_sto / p_start / p_end)
# ──────────────────────────────────────────────────────────────────────────────

@cuda.jit
def pt_rhs_elementary(
    # state
    T, C,                    # (ncells,)  (ncells, nsp)
    # NASA thermo
    coef_low, coef_high,     # (nsp, 7)
    Tmid,                    # (nsp,)
    # reaction connectivity
    react_idx, react_sto, r_start, r_end,   # flattened reactant data
    prod_idx,  prod_sto,  p_start, p_end,   # flattened product  data
    # Arrhenius + flags
    A_arr, n_arr, Ea_arr,    # (nreac,)
    oneWay,                  # (nreac,)  1.0 = irreversible
    delta_n,                 # (nreac,)  Δν = Σν_prod - Σν_react
    # output
    omega,                   # (ncells, nsp)  species production rates
):
    """
    Per-thread kernel: each thread handles ONE cell.
    Elementary reactions only.
    """
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc = T[cell]
    nsp   = C.shape[1]
    nreac = A_arr.shape[0]

    # ── forward rate constants ────────────────────────────────────────────────
    # ── equilibrium Kc  ──────────────────────────────────────────────────────
    # ── reaction rates  ──────────────────────────────────────────────────────
    for i in range(nreac):
        kf_i = _arrhenius(A_arr[i], n_arr[i], Ea_arr[i], Tc)

        # conc_r = prod_j  C[react_idx[j]]^react_sto[j]   (log-space)
        log_cr = 0.0
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            Csp = C[cell, sp]
            if Csp < EPS: Csp = EPS
            log_cr += react_sto[jj] * math.log(Csp)
        conc_r = math.exp(log_cr)

        log_cp = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            Csp = C[cell, sp]
            if Csp < EPS: Csp = EPS
            log_cp += prod_sto[jj] * math.log(Csp)
        conc_p = math.exp(log_cp)

        # ΔG°  via NASA
        dG0 = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc)
            # s° reference is Pref_s; adjust to Pref for Kp
            s_sp -= R_GAS * math.log(PREF_S / PREF)   # s at Pref
            dG0 += prod_sto[jj] * (h_sp - Tc * s_sp)

        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc)
            s_sp -= R_GAS * math.log(PREF_S / PREF)
            dG0 -= react_sto[jj] * (h_sp - Tc * s_sp)

        Kp = math.exp(-dG0 / (R_GAS * Tc))
        Kc = Kp * (PREF / (R_GAS * Tc)) ** delta_n[i]

        RR = kf_i * (conc_r - (1.0 - oneWay[i]) * conc_p / Kc)

        # accumulate omega
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            cuda.atomic.add(omega, (cell, sp), -react_sto[jj] * RR)
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            cuda.atomic.add(omega, (cell, sp),  prod_sto[jj] * RR)


@cuda.jit
def pt_rhs_threebody(
    T, C,
    coef_low, coef_high, Tmid,
    react_idx, react_sto, r_start, r_end,
    prod_idx,  prod_sto,  p_start, p_end,
    A_arr, n_arr, Ea_arr, oneWay, delta_n,
    eff_matrix,            # (nreac, nsp) third-body efficiencies
    omega,
):
    """Per-thread kernel: three-body reactions."""
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc = T[cell]
    nsp   = C.shape[1]
    nreac = A_arr.shape[0]

    for i in range(nreac):
        kf_i = _arrhenius(A_arr[i], n_arr[i], Ea_arr[i], Tc)

        log_cr = 0.0
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            Csp = max(C[cell, sp], EPS)
            log_cr += react_sto[jj] * math.log(Csp)
        conc_r = math.exp(log_cr)

        log_cp = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            Csp = max(C[cell, sp], EPS)
            log_cp += prod_sto[jj] * math.log(Csp)
        conc_p = math.exp(log_cp)

        # [M]
        M_conc = 0.0
        for sp in range(nsp):
            M_conc += eff_matrix[i, sp] * C[cell, sp]

        # Kc
        dG0 = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 += prod_sto[jj] * (h_sp - Tc*s_sp)
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 -= react_sto[jj] * (h_sp - Tc*s_sp)

        Kp = math.exp(-dG0 / (R_GAS * Tc))
        Kc = Kp * (PREF / (R_GAS * Tc)) ** delta_n[i]

        RR = kf_i * M_conc * (conc_r - (1.0 - oneWay[i]) * conc_p / Kc)

        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            cuda.atomic.add(omega, (cell, sp), -react_sto[jj] * RR)
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            cuda.atomic.add(omega, (cell, sp),  prod_sto[jj] * RR)


@cuda.jit
def pt_rhs_falloff(
    T, C,
    coef_low, coef_high, Tmid,
    react_idx, react_sto, r_start, r_end,
    prod_idx,  prod_sto,  p_start, p_end,
    A_k0, b_k0, Ea_k0,
    A_kinf, b_kinf, Ea_kinf,
    oneWay, delta_n,
    eff_matrix,
    # Troe parameters  (set A=0 for Lindemann)
    troe_A, troe_T1, troe_T2, troe_T3,
    is_troe,           # (nreac,) int  1=Troe  0=Lindemann
    omega,
):
    """Per-thread kernel: falloff reactions (Lindemann + Troe)."""
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc = T[cell]
    nsp   = C.shape[1]
    nreac = A_k0.shape[0]

    for i in range(nreac):
        k0_i   = _arrhenius(A_k0[i],   b_k0[i],   Ea_k0[i],   Tc)
        kinf_i = _arrhenius(A_kinf[i], b_kinf[i], Ea_kinf[i], Tc)

        # [M]
        M_conc = 0.0
        for sp in range(nsp):
            M_conc += eff_matrix[i, sp] * C[cell, sp]

        kinf_safe = kinf_i if kinf_i > EPS else EPS
        Pr = (k0_i * M_conc) / kinf_safe
        Pr_term = Pr / (1.0 + Pr)

        # Troe / Lindemann blending factor F
        if is_troe[i]:
            A  = troe_A[i];  T1 = troe_T1[i]
            T2 = troe_T2[i]; T3 = troe_T3[i]
            Fc = 0.0
            if T1 > EPS: Fc += A  * math.exp(-Tc / T1)
            if T3 > EPS: Fc += (1.0 - A) * math.exp(-Tc / T3)
            if T2 > EPS: Fc += math.exp(-T2 / Tc)
            if Fc < EPS: Fc = EPS
            logFc = math.log10(Fc)
            C_p   = -0.4 - 0.67 * logFc
            N_p   =  0.75 - 1.27 * logFc
            Pr_s  = Pr if Pr > EPS else EPS
            logPr = math.log10(Pr_s)
            num   = logPr + C_p
            den   = N_p - 0.14 * num
            if abs(den) < EPS:
                den = copysigne_GPU(EPS, den)
            f1  = num / den
            exp_F = logFc / (1.0 + f1*f1)
            F = 10.0 ** exp_F
        else:
            F = 1.0

        kf_i = kinf_safe * Pr_term * F

        log_cr = 0.0
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            Csp = max(C[cell, sp], EPS)
            log_cr += react_sto[jj] * math.log(Csp)
        conc_r = math.exp(log_cr)

        log_cp = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            Csp = max(C[cell, sp], EPS)
            log_cp += prod_sto[jj] * math.log(Csp)
        conc_p = math.exp(log_cp)

        # Kc
        dG0 = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 += prod_sto[jj] * (h_sp - Tc*s_sp)
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 -= react_sto[jj] * (h_sp - Tc*s_sp)

        Kp = math.exp(-dG0 / (R_GAS * Tc))
        Kc = Kp * (PREF / (R_GAS * Tc)) ** delta_n[i]

        RR = kf_i * (conc_r - (1.0 - oneWay[i]) * conc_p / Kc)

        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            cuda.atomic.add(omega, (cell, sp), -react_sto[jj] * RR)
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            cuda.atomic.add(omega, (cell, sp),  prod_sto[jj] * RR)


# ──────────────────────────────────────────────────────────────────────────────
# Per-thread analytic JVP   (d omega / d[T,C]) · [vT, vC]
# for elementary reactions
# ──────────────────────────────────────────────────────────────────────────────
@cuda.jit
def pt_jvp_elementary(
    T, C,
    coef_low, coef_high, Tmid,
    react_idx, react_sto, r_start, r_end,
    prod_idx,  prod_sto,  p_start, p_end,
    A_arr, n_arr, Ea_arr, oneWay, delta_n,
    v_T, v_C,          # perturbation vectors
    Jv_omega,          # output (ncells, nsp)
):
    """
    Per-thread analytic JVP for elementary reactions.

    Formula (matching the Torch reference):
      dkf/dT  = kf * (n/T + Ea/(R T²))
      d ln Kc/dT = ΔS/(RT) + ΔG°/(RT²) - Δn/T
      dRR/dT  = dkf/dT*(cr - (1-ow)*cp/Kc) + kf*(1-ow)*cp*dKc/dT/Kc²
      dRR/dC  via per-species log-derivative:
              = kf*cr*(ν_r/C_r)*vC_r  -  kf*(1-ow)/Kc*cp*(ν_p/C_p)*vC_p
    """
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc   = T[cell]
    nreac = A_arr.shape[0]

    for i in range(nreac):
        kf_i = _arrhenius(A_arr[i], n_arr[i], Ea_arr[i], Tc)

        # concentrations
        log_cr = 0.0
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            Csp = max(C[cell, sp], EPS)
            log_cr += react_sto[jj] * math.log(Csp)
        conc_r = math.exp(log_cr)

        log_cp = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            Csp = max(C[cell, sp], EPS)
            log_cp += prod_sto[jj] * math.log(Csp)
        conc_p = math.exp(log_cp)

        # thermodynamics
        dG0 = 0.0; dS0 = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 += prod_sto[jj] * (h_sp - Tc*s_sp)
            dS0 += prod_sto[jj] * s_sp
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 -= react_sto[jj] * (h_sp - Tc*s_sp)
            dS0 -= react_sto[jj] * s_sp

        Kp = math.exp(-dG0 / (R_GAS * Tc))
        Kc = Kp * (PREF / (R_GAS * Tc)) ** delta_n[i]

        ow = oneWay[i]
        factor = 1.0 - ow

        # dkf/dT
        dkf_dT = kf_i * (n_arr[i] / Tc + Ea_arr[i] / (R_GAS * Tc * Tc))

        # d ln Kc/dT  =  ΔS/(RT) + ΔG°/(RT²) - Δn/T
        dlnKc_dT = dS0/(R_GAS*Tc) + dG0/(R_GAS*Tc*Tc) - delta_n[i]/Tc
        dKc_dT   = Kc * dlnKc_dT

        # dRR/dT contribution
        dRR_dT = (dkf_dT * (conc_r - factor*conc_p/Kc)
                  + kf_i * factor * conc_p * dKc_dT / (Kc*Kc))
        dRR_v  = dRR_dT * v_T[cell]

        # dRR/dC contribution  (reactants)
        for jj in range(r_start[i], r_end[i]):
            sp  = react_idx[jj]
            Csp = max(C[cell, sp], EPS)
            dRR_v += kf_i * conc_r * (react_sto[jj] / Csp) * v_C[cell, sp]

        # dRR/dC contribution  (products)
        for jj in range(p_start[i], p_end[i]):
            sp  = prod_idx[jj]
            Csp = max(C[cell, sp], EPS)
            dRR_v -= kf_i * factor / Kc * conc_p * (prod_sto[jj] / Csp) * v_C[cell, sp]

        # scatter to species
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            cuda.atomic.add(Jv_omega, (cell, sp), -react_sto[jj] * dRR_v)
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            cuda.atomic.add(Jv_omega, (cell, sp),  prod_sto[jj] * dRR_v)


@cuda.jit
def pt_jvp_threebody(
    T, C,
    coef_low, coef_high, Tmid,
    react_idx, react_sto, r_start, r_end,
    prod_idx,  prod_sto,  p_start, p_end,
    A_arr, n_arr, Ea_arr, oneWay, delta_n,
    eff_matrix,
    v_T, v_C,
    Jv_omega,
):
    """Per-thread analytic JVP for three-body reactions."""
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc   = T[cell]
    nsp  = C.shape[1]
    nreac = A_arr.shape[0]

    for i in range(nreac):
        kf_i = _arrhenius(A_arr[i], n_arr[i], Ea_arr[i], Tc)

        log_cr = 0.0
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]; Csp = max(C[cell, sp], EPS)
            log_cr += react_sto[jj] * math.log(Csp)
        conc_r = math.exp(log_cr)

        log_cp = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]; Csp = max(C[cell, sp], EPS)
            log_cp += prod_sto[jj] * math.log(Csp)
        conc_p = math.exp(log_cp)

        M_conc = 0.0
        for sp in range(nsp):
            M_conc += eff_matrix[i, sp] * C[cell, sp]

        dG0 = 0.0; dS0 = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 += prod_sto[jj]*(h_sp - Tc*s_sp); dS0 += prod_sto[jj]*s_sp
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 -= react_sto[jj]*(h_sp - Tc*s_sp); dS0 -= react_sto[jj]*s_sp

        Kp = math.exp(-dG0 / (R_GAS*Tc))
        Kc = Kp * (PREF / (R_GAS*Tc)) ** delta_n[i]
        ow = oneWay[i]; factor = 1.0 - ow

        baseRR = kf_i * (conc_r - factor*conc_p/Kc)

        dkf_dT   = kf_i * (n_arr[i]/Tc + Ea_arr[i]/(R_GAS*Tc*Tc))
        dlnKc_dT = dS0/(R_GAS*Tc) + dG0/(R_GAS*Tc*Tc) - delta_n[i]/Tc
        dKc_dT   = Kc * dlnKc_dT
        dbase_dT = (dkf_dT*(conc_r - factor*conc_p/Kc)
                    + kf_i*factor*conc_p*dKc_dT/(Kc*Kc))

        # dM/dC · vC
        dM_v = 0.0
        for sp in range(nsp):
            dM_v += eff_matrix[i, sp] * v_C[cell, sp]

        dRR_v = dbase_dT * v_T[cell] * M_conc + baseRR * dM_v

        # dbase/dC · vC (conc terms)
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]; Csp = max(C[cell, sp], EPS)
            dRR_v += kf_i * conc_r * (react_sto[jj]/Csp) * v_C[cell, sp] * M_conc
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]; Csp = max(C[cell, sp], EPS)
            dRR_v -= kf_i*factor/Kc * conc_p * (prod_sto[jj]/Csp) * v_C[cell, sp] * M_conc

        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            cuda.atomic.add(Jv_omega, (cell, sp), -react_sto[jj]*dRR_v)
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            cuda.atomic.add(Jv_omega, (cell, sp),  prod_sto[jj]*dRR_v)


@cuda.jit
def pt_jvp_falloff(
    T, C,
    coef_low, coef_high, Tmid,
    react_idx, react_sto, r_start, r_end,
    prod_idx,  prod_sto,  p_start, p_end,
    A_k0, b_k0, Ea_k0,
    A_kinf, b_kinf, Ea_kinf,
    oneWay, delta_n,
    eff_matrix,
    troe_A, troe_T1, troe_T2, troe_T3,
    is_troe,
    v_T, v_C,
    Jv_omega,
):
    """Per-thread analytic JVP for falloff reactions."""
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc   = T[cell]
    nsp  = C.shape[1]
    nreac = A_k0.shape[0]
    ln10  = math.log(10.0)
    inv_ln10 = 1.0 / ln10

    for i in range(nreac):
        k0_i   = _arrhenius(A_k0[i],   b_k0[i],   Ea_k0[i],   Tc)
        kinf_i = _arrhenius(A_kinf[i], b_kinf[i], Ea_kinf[i], Tc)
        kinf_safe = kinf_i if kinf_i > EPS else EPS

        # [M]
        M_conc = 0.0
        for sp in range(nsp):
            M_conc += eff_matrix[i, sp] * C[cell, sp]
        dM_v = 0.0
        for sp in range(nsp):
            dM_v += eff_matrix[i, sp] * v_C[cell, sp]

        Pr = (k0_i * M_conc) / kinf_safe
        Pr_term = Pr / (1.0 + Pr)
        k_base  = kinf_safe * Pr_term

        # dk0/dT, dkinf/dT
        dk0_dT   = k0_i   * (b_k0[i]  /Tc + Ea_k0[i]  /(R_GAS*Tc*Tc))
        dkinf_dT = kinf_i * (b_kinf[i]/Tc + Ea_kinf[i]/(R_GAS*Tc*Tc))

        dPr_dT   = (dk0_dT*M_conc - k0_i*M_conc*dkinf_dT/kinf_safe) / kinf_safe
        dPr_dM_v = k0_i / kinf_safe * dM_v
        dPr_term_dPr = 1.0 / ((1.0 + Pr)*(1.0 + Pr))

        dkbase_dT   = dkinf_dT*Pr_term + kinf_safe*dPr_term_dPr*dPr_dT
        dkbase_dM_v = kinf_safe*dPr_term_dPr*dPr_dM_v

        # Troe F and its derivatives
        if is_troe[i]:
            A_ = troe_A[i]; T1_ = troe_T1[i]; T2_ = troe_T2[i]; T3_ = troe_T3[i]
            Fc = 0.0; dFc_dT = 0.0
            if T1_ > EPS:
                e1 = math.exp(-Tc/T1_)
                Fc    += A_ * e1
                dFc_dT += A_ * (-1.0/T1_) * e1
            if T3_ > EPS:
                e3 = math.exp(-Tc/T3_)
                Fc    += (1.0 - A_) * e3
                dFc_dT += (1.0 - A_) * (-1.0/T3_) * e3
            if T2_ > EPS:
                e2 = math.exp(-T2_/Tc)
                Fc    += e2
                dFc_dT += e2 * T2_ / (Tc*Tc)
            if Fc < EPS: Fc = EPS

            logFc  = math.log10(Fc)
            dlogFc_dT = (dFc_dT / Fc) * inv_ln10
            C_p    = -0.4 - 0.67 * logFc
            N_p    =  0.75 - 1.27 * logFc
            dCp_dT = -0.67 * dlogFc_dT
            dNp_dT = -1.27 * dlogFc_dT

            Pr_s   = Pr if Pr > EPS else EPS
            logPr  = math.log10(Pr_s)
            dlogPr_dPr = inv_ln10 / Pr_s if Pr > EPS else 0.0

            num = logPr + C_p
            den = N_p - 0.14 * num
            if abs(den) < EPS:
                den = copysigne_GPU(EPS, den)
            f1 = num / den

            # df1/dPr
            df1_dPr = (dlogPr_dPr*den + 0.14*num*dlogPr_dPr) / (den*den)
            # df1/dT  (from logFc dependence)
            num_T = dCp_dT
            den_T = dNp_dT - 0.14*dCp_dT
            df1_dT = (num_T*den - num*den_T) / (den*den)

            f1sq = f1*f1
            denom_f = 1.0 + f1sq
            exp_F = logFc / denom_f

            # d(exp_F)/dT and d(exp_F)/dPr
            dexp_dlogFc = 1.0 / denom_f
            dexp_df1    = -2.0*f1*logFc / (denom_f*denom_f)
            dexp_dT  = dexp_dlogFc*dlogFc_dT + dexp_df1*df1_dT
            dexp_dPr = dexp_df1*df1_dPr

            F    = 10.0 ** exp_F
            dF_dT  = F * ln10 * dexp_dT
            dF_dPr = F * ln10 * dexp_dPr
        else:
            F = 1.0; dF_dT = 0.0; dF_dPr = 0.0

        # dkf/dT  and  dkf/d[M]·v[M]
        dkf_dT_v   = (dkbase_dT*F + k_base*dF_dT + k_base*dF_dPr*dPr_dT) * v_T[cell]
        dkf_dM_v   = (dkbase_dM_v*F + k_base*dF_dPr*dPr_dM_v)

        # kf
        kf_i = k_base * F

        # concentrations
        log_cr = 0.0
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]; Csp = max(C[cell, sp], EPS)
            log_cr += react_sto[jj]*math.log(Csp)
        conc_r = math.exp(log_cr)

        log_cp = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]; Csp = max(C[cell, sp], EPS)
            log_cp += prod_sto[jj]*math.log(Csp)
        conc_p = math.exp(log_cp)

        # Kc
        dG0 = 0.0; dS0 = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 += prod_sto[jj]*(h_sp - Tc*s_sp); dS0 += prod_sto[jj]*s_sp
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 -= react_sto[jj]*(h_sp - Tc*s_sp); dS0 -= react_sto[jj]*s_sp

        Kp = math.exp(-dG0 / (R_GAS*Tc))
        Kc = Kp * (PREF / (R_GAS*Tc)) ** delta_n[i]
        ow = oneWay[i]; factor = 1.0 - ow

        conc_expr = conc_r - factor*conc_p/Kc
        dlnKc_dT  = dS0/(R_GAS*Tc) + dG0/(R_GAS*Tc*Tc) - delta_n[i]/Tc
        dKc_dT    = Kc * dlnKc_dT

        # dRR/dT total
        dRR_v = (dkf_dT_v * conc_expr
                 + kf_i * factor * conc_p * dKc_dT / (Kc*Kc) * v_T[cell])

        # dkf/dM · vM contribution (kf depends on M through Pr)
        dRR_v += dkf_dM_v * conc_expr

        # dRR/dC (concentration terms)
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]; Csp = max(C[cell, sp], EPS)
            dRR_v += kf_i * conc_r * (react_sto[jj]/Csp) * v_C[cell, sp]
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]; Csp = max(C[cell, sp], EPS)
            dRR_v -= kf_i*factor/Kc * conc_p * (prod_sto[jj]/Csp) * v_C[cell, sp]

        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            cuda.atomic.add(Jv_omega, (cell, sp), -react_sto[jj]*dRR_v)
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            cuda.atomic.add(Jv_omega, (cell, sp),  prod_sto[jj]*dRR_v)


# ══════════════════════════════════════════════════════════════════════════════
# §3  PER-BLOCK KERNELS
# ══════════════════════════════════════════════════════════════════════════════
#
# Naming convention:  _pb_*  = per-block
#
# Launch config: <<<ncells, TPB>>>   with TPB = blockDim.x (must be power of 2)
#
# Each block owns ONE cell.  Threads cooperate over reactions (outer loop split
# among threads) and species (reduction via shared memory).
#
# Shared-memory layout (size = TPB * sizeof(float64)):
#   sdata[tid] used for parallel reductions (dot products in GMRES, etc.)
#
# We keep species output in global memory and use atomicAdd to accumulate from
# each thread's reaction sub-range.  This is safe because all threads in the
# block work on the same cell, and different reactions write to potentially
# overlapping species.
# ──────────────────────────────────────────────────────────────────────────────

@cuda.jit
def pb_rhs_elementary(
    T, C,
    coef_low, coef_high, Tmid,
    react_idx, react_sto, r_start, r_end,
    prod_idx,  prod_sto,  p_start, p_end,
    A_arr, n_arr, Ea_arr, oneWay, delta_n,
    omega,
):
    """
    Per-block kernel for elementary reactions.
    Block = one cell.  Threads are split across reactions.
    """
    cell = cuda.blockIdx.x
    tid  = cuda.threadIdx.x
    TPB  = cuda.blockDim.x

    if cell >= T.shape[0]:
        return

    Tc    = T[cell]
    nreac = A_arr.shape[0]

    # Each thread handles a contiguous sub-range of reactions
    for i in range(tid, nreac, TPB):
        kf_i = _arrhenius(A_arr[i], n_arr[i], Ea_arr[i], Tc)

        log_cr = 0.0
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]; Csp = max(C[cell, sp], EPS)
            log_cr += react_sto[jj]*math.log(Csp)
        conc_r = math.exp(log_cr)

        log_cp = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]; Csp = max(C[cell, sp], EPS)
            log_cp += prod_sto[jj]*math.log(Csp)
        conc_p = math.exp(log_cp)

        dG0 = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 += prod_sto[jj]*(h_sp - Tc*s_sp)
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 -= react_sto[jj]*(h_sp - Tc*s_sp)

        Kp = math.exp(-dG0 / (R_GAS*Tc))
        Kc = Kp * (PREF / (R_GAS*Tc)) ** delta_n[i]
        RR = kf_i * (conc_r - (1.0 - oneWay[i])*conc_p/Kc)

        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            cuda.atomic.add(omega, (cell, sp), -react_sto[jj]*RR)
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            cuda.atomic.add(omega, (cell, sp),  prod_sto[jj]*RR)
    # No syncthreads needed; atomicAdd guarantees correctness


@cuda.jit
def pb_jvp_elementary(
    T, C,
    coef_low, coef_high, Tmid,
    react_idx, react_sto, r_start, r_end,
    prod_idx,  prod_sto,  p_start, p_end,
    A_arr, n_arr, Ea_arr, oneWay, delta_n,
    v_T, v_C,
    Jv_omega,
):
    """Per-block analytic JVP for elementary reactions."""
    cell = cuda.blockIdx.x
    tid  = cuda.threadIdx.x
    TPB  = cuda.blockDim.x

    if cell >= T.shape[0]:
        return

    Tc    = T[cell]
    nreac = A_arr.shape[0]

    for i in range(tid, nreac, TPB):
        kf_i = _arrhenius(A_arr[i], n_arr[i], Ea_arr[i], Tc)

        log_cr = 0.0
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]; Csp = max(C[cell, sp], EPS)
            log_cr += react_sto[jj]*math.log(Csp)
        conc_r = math.exp(log_cr)

        log_cp = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]; Csp = max(C[cell, sp], EPS)
            log_cp += prod_sto[jj]*math.log(Csp)
        conc_p = math.exp(log_cp)

        dG0 = 0.0; dS0 = 0.0
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 += prod_sto[jj]*(h_sp - Tc*s_sp); dS0 += prod_sto[jj]*s_sp
        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            if Tc > Tmid[sp]:
                h_sp = _h_nasa(coef_high[sp], Tc)
                s_sp = _s_nasa(coef_high[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            else:
                h_sp = _h_nasa(coef_low[sp], Tc)
                s_sp = _s_nasa(coef_low[sp], Tc) - R_GAS*math.log(PREF_S/PREF)
            dG0 -= react_sto[jj]*(h_sp - Tc*s_sp); dS0 -= react_sto[jj]*s_sp

        Kp = math.exp(-dG0 / (R_GAS*Tc))
        Kc = Kp * (PREF / (R_GAS*Tc)) ** delta_n[i]
        ow = oneWay[i]; factor = 1.0 - ow

        dkf_dT   = kf_i * (n_arr[i]/Tc + Ea_arr[i]/(R_GAS*Tc*Tc))
        dlnKc_dT = dS0/(R_GAS*Tc) + dG0/(R_GAS*Tc*Tc) - delta_n[i]/Tc
        dKc_dT   = Kc * dlnKc_dT
        dRR_v    = (dkf_dT*(conc_r - factor*conc_p/Kc)
                    + kf_i*factor*conc_p*dKc_dT/(Kc*Kc)) * v_T[cell]

        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]; Csp = max(C[cell, sp], EPS)
            dRR_v += kf_i * conc_r * (react_sto[jj]/Csp) * v_C[cell, sp]
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]; Csp = max(C[cell, sp], EPS)
            dRR_v -= kf_i*factor/Kc * conc_p * (prod_sto[jj]/Csp) * v_C[cell, sp]

        for jj in range(r_start[i], r_end[i]):
            sp = react_idx[jj]
            cuda.atomic.add(Jv_omega, (cell, sp), -react_sto[jj]*dRR_v)
        for jj in range(p_start[i], p_end[i]):
            sp = prod_idx[jj]
            cuda.atomic.add(Jv_omega, (cell, sp),  prod_sto[jj]*dRR_v)


# ══════════════════════════════════════════════════════════════════════════════
# §4  UTILITY: zero-fill kernel (replaces torch.zeros_like on device arrays)
# ══════════════════════════════════════════════════════════════════════════════
@cuda.jit
def zero_2d(arr):
    """Zero a 2-D device array in parallel."""
    i, j = cuda.grid(2)
    if i < arr.shape[0] and j < arr.shape[1]:
        arr[i, j] = 0.0