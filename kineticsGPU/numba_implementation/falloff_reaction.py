from numba import cuda
import math
import kineticsGPU.numba_implementation.constants as constants
import kineticsGPU.numba_implementation.chemistry as chemistry

# =============================================================================
#  PIPELINE FALLOFF (Lindemann / Troe)
#
#  Étape 1 — pack_falloff_context_packaged(Tcell, Ccell, react_arrays,
#                                           thermo_arrays, ctx)
#  Étape 2 — fo_omega_packaged(react_arrays, ctx, omega_cell)
#  Étape 3 — fo_jvp_packaged(Ccell, vTcell, vCcell, react_arrays, ctx, jvp_cell)
#
#  Layout ctx (N_CTX_FO_FIELDS, nreac) :
#
#    ctx[CTX_FO_K0,       r]  k0(T)
#    ctx[CTX_FO_KINF,     r]  kinf(T)
#    ctx[CTX_FO_DK0,      r]  dk0/dT
#    ctx[CTX_FO_DKINF,    r]  dkinf/dT
#    ctx[CTX_FO_M,        r]  [M]
#    ctx[CTX_FO_PR,       r]  Pr = k0*M/kinf
#    ctx[CTX_FO_F,        r]  F  (Troe ou 1.0)
#    ctx[CTX_FO_KBASE,    r]  k_base = kinf * Pr/(1+Pr)
#    ctx[CTX_FO_KF,       r]  kf = k_base * F
#    ctx[CTX_FO_KC,       r]  Kc
#    ctx[CTX_FO_DKC,      r]  dKc/dT  (via deltaS + deltaG0)
#    ctx[CTX_FO_CR,       r]  C_react
#    ctx[CTX_FO_CP,       r]  C_prod
#    ctx[CTX_FO_RR,       r]  RR net
#    ctx[CTX_FO_DFDPR,    r]  dF/dPr   (0 pour Lindemann)
#    ctx[CTX_FO_DFDT,     r]  dF/dT direct (à Pr constant, 0 pour Lindemann)
# =============================================================================

CTX_FO_K0    =  0
CTX_FO_KINF  =  1
CTX_FO_DK0   =  2
CTX_FO_DKINF =  3
CTX_FO_M     =  4
CTX_FO_PR    =  5
CTX_FO_F     =  6
CTX_FO_KBASE =  7
CTX_FO_KF    =  8
CTX_FO_KC    =  9
CTX_FO_DKC   = 10
CTX_FO_CR    = 11
CTX_FO_CP    = 12
CTX_FO_RR    = 13
CTX_FO_DFDPR = 14
CTX_FO_DFDT  = 15

N_CTX_FO_FIELDS = 16

FALLOFF_LINDEMANN = 0
FALLOFF_TROE      = 1


# ── Accesseurs inline ────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _fo_k0(ctx, r):    return ctx[CTX_FO_K0,    r]

@cuda.jit(device=True, inline=True)
def _fo_kinf(ctx, r):  return ctx[CTX_FO_KINF,  r]

@cuda.jit(device=True, inline=True)
def _fo_dk0(ctx, r):   return ctx[CTX_FO_DK0,   r]

@cuda.jit(device=True, inline=True)
def _fo_dkinf(ctx, r): return ctx[CTX_FO_DKINF, r]

@cuda.jit(device=True, inline=True)
def _fo_M(ctx, r):     return ctx[CTX_FO_M,     r]

@cuda.jit(device=True, inline=True)
def _fo_Pr(ctx, r):    return ctx[CTX_FO_PR,    r]

@cuda.jit(device=True, inline=True)
def _fo_F(ctx, r):     return ctx[CTX_FO_F,     r]

@cuda.jit(device=True, inline=True)
def _fo_kbase(ctx, r): return ctx[CTX_FO_KBASE, r]

@cuda.jit(device=True, inline=True)
def _fo_kf(ctx, r):    return ctx[CTX_FO_KF,    r]

@cuda.jit(device=True, inline=True)
def _fo_Kc(ctx, r):    return ctx[CTX_FO_KC,    r]

@cuda.jit(device=True, inline=True)
def _fo_dKc(ctx, r):   return ctx[CTX_FO_DKC,   r]

@cuda.jit(device=True, inline=True)
def _fo_Cr(ctx, r):    return ctx[CTX_FO_CR,    r]

@cuda.jit(device=True, inline=True)
def _fo_Cp(ctx, r):    return ctx[CTX_FO_CP,    r]

@cuda.jit(device=True, inline=True)
def _fo_RR(ctx, r):    return ctx[CTX_FO_RR,    r]

@cuda.jit(device=True, inline=True)
def _fo_dFdPr(ctx, r): return ctx[CTX_FO_DFDPR, r]

@cuda.jit(device=True, inline=True)
def _fo_dFdT(ctx, r):  return ctx[CTX_FO_DFDT,  r]


# -----------------------------------------------------------------------------
#  ÉTAPE 1 — PACK
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def pack_falloff_context(
        Tcell, Ccell,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        A0_arr,  n0_arr,  Ea0_arr,
        Ainf_arr, ninf_arr, Eainf_arr,
        oneWay, delta_n,
        effmatrix,
        falloff_type_arr,
        troe_A, troe_T1, troe_T2, troe_T3,
        coef_low, coef_high, Tmid,
        ctx,                   # (N_CTX_FO_FIELDS, nreac)
):
    nreac = A0_arr.shape[0]

    for r in range(nreac):
        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        # ── k0 / kinf et leurs dérivées ────────────────────────────────────
        k0_r    = chemistry.arrhenius(A0_arr[r],   n0_arr[r],   Ea0_arr[r],   Tcell)
        kinf_r  = chemistry.arrhenius(Ainf_arr[r], ninf_arr[r], Eainf_arr[r], Tcell)
        dk0_r   = chemistry.d_arrhenius_dT(k0_r,   n0_arr[r],   Ea0_arr[r],   Tcell)
        dkinf_r = chemistry.d_arrhenius_dT(kinf_r, ninf_arr[r], Eainf_arr[r], Tcell)

        # ── [M] ────────────────────────────────────────────────────────────
        M_r = chemistry.M(Ccell, effmatrix[r])

        # ── Pr ────────────────────────────────────────────────────────────
        kinf_safe = max(kinf_r, 1e-300)
        Pr_r      = k0_r * M_r / kinf_safe

        # ── F et ses dérivées (stockées pour le jvp) ──────────────────────
        ftype = falloff_type_arr[r]
        if ftype == FALLOFF_TROE:
            F_r, dF_dT_r, dF_dPr_r = chemistry.F_troe_and_derivatives(
                Tcell, Pr_r,
                troe_A[r], troe_T1[r], troe_T2[r], troe_T3[r])
        else:
            F_r      = 1.0
            dF_dT_r  = 0.0
            dF_dPr_r = 0.0

        # ── k_base et kf ──────────────────────────────────────────────────
        Pr_term  = Pr_r / (1.0 + Pr_r)
        kbase_r  = kinf_safe * Pr_term
        kf_r     = kbase_r * F_r

        # ── Kc et dKc/dT (cohérent via deltaS + deltaG0) ──────────────────
        dG0_r = chemistry.delta_G0(
            react_start, react_nb, prod_start, prod_nb,
            react_idx, prod_idx, react_sto, prod_sto,
            coef_high, coef_low, Tmid, Tcell)
        Kc_r  = chemistry.Kc(delta_n[r], Tcell, chemistry.Kp(dG0_r, Tcell))

        dS0_r = chemistry.delta_S0(
            react_start, react_nb, prod_start, prod_nb,
            react_idx, prod_idx, react_sto, prod_sto,
            coef_high, coef_low, Tmid, Tcell)
        dKc_r = Kc_r * (
            dS0_r / (constants.R * Tcell)
            + dG0_r / (constants.R * Tcell * Tcell)
            - delta_n[r] / Tcell
        )

        # ── Concentrations ─────────────────────────────────────────────────
        C_react = 1.0
        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            C_react *= max(Ccell[idx], 1e-300) ** sto

        C_prod = 1.0
        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            C_prod *= max(Ccell[idx], 1e-300) ** sto

        # ── RR ────────────────────────────────────────────────────────────
        RR_r = kf_r * (C_react - (1.0 - oneWay[r]) / Kc_r * C_prod)

        # ── Écriture ──────────────────────────────────────────────────────
        ctx[CTX_FO_K0,    r] = k0_r
        ctx[CTX_FO_KINF,  r] = kinf_r
        ctx[CTX_FO_DK0,   r] = dk0_r
        ctx[CTX_FO_DKINF, r] = dkinf_r
        ctx[CTX_FO_M,     r] = M_r
        ctx[CTX_FO_PR,    r] = Pr_r
        ctx[CTX_FO_F,     r] = F_r
        ctx[CTX_FO_KBASE, r] = kbase_r
        ctx[CTX_FO_KF,    r] = kf_r
        ctx[CTX_FO_KC,    r] = Kc_r
        ctx[CTX_FO_DKC,   r] = dKc_r
        ctx[CTX_FO_CR,    r] = C_react
        ctx[CTX_FO_CP,    r] = C_prod
        ctx[CTX_FO_RR,    r] = RR_r
        ctx[CTX_FO_DFDPR, r] = dF_dPr_r
        ctx[CTX_FO_DFDT,  r] = dF_dT_r


# -----------------------------------------------------------------------------
#  ÉTAPE 2 — OMEGA
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def fo_omega(
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        ctx,
        omega_cell,
):
    nreac = reactant_number.shape[0]
    for r in range(nreac):
        RR_r        = _fo_RR(ctx, r)
        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        for i in range(react_nb):
            omega_cell[react_idx[i + react_start]] -= react_sto[i + react_start] * RR_r
        for i in range(prod_nb):
            omega_cell[prod_idx[i + prod_start]]   += prod_sto[i + prod_start]   * RR_r


# -----------------------------------------------------------------------------
#  ÉTAPE 3 — JVP
#
#  kf = k_base * F
#  RR = kf * (Cr - G)
#
#  dkf/dT = dk_base/dT * F  +  k_base * dF/dT_total
#         avec dF/dT_total = dF/dT_direct + dF/dPr * dPr/dT
#
#  dkf/dC · v = (dk_base/dC·v) * F  +  k_base * dF/dPr * dPr/dC·v
#
#  dPr/dT  = (dk0/dT * M - k0*M/kinf * dkinf/dT) / kinf
#  dPr/dC·v = (k0/kinf) * dM·v
#  dk_base/dT = dkinf/dT * Pr_term  +  kinf * dPr_term/dPr * dPr/dT
#  dk_base/dC·v = kinf * dPr_term/dPr * dPr/dC·v
#  dPr_term/dPr = 1/(1+Pr)²
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def fo_jvp(
        Ccell,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        oneWay,
        effmatrix,
        ctx,
        vTcell, vCcell,
        jvp_cell,
):
    nreac = reactant_number.shape[0]
    nsp   = Ccell.shape[0]

    for r in range(nreac):
        k0_r    = _fo_k0   (ctx, r)
        kinf_r  = _fo_kinf (ctx, r)
        dk0_r   = _fo_dk0  (ctx, r)
        dkinf_r = _fo_dkinf(ctx, r)
        M_r     = _fo_M    (ctx, r)
        Pr_r    = _fo_Pr   (ctx, r)
        F_r     = _fo_F    (ctx, r)
        kbase_r = _fo_kbase(ctx, r)
        kf_r    = _fo_kf   (ctx, r)
        Kc_r    = _fo_Kc   (ctx, r)
        dKc_r   = _fo_dKc  (ctx, r)
        Cr_r    = _fo_Cr   (ctx, r)
        Cp_r    = _fo_Cp   (ctx, r)
        dF_dPr  = _fo_dFdPr(ctx, r)
        dF_dT   = _fo_dFdT (ctx, r)

        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        ow        = oneWay[r]
        kinf_safe = max(kinf_r, 1e-300)

        # ── dPr/dT ────────────────────────────────────────────────────────
        dPr_dT = (dk0_r * M_r - k0_r * M_r / kinf_safe * dkinf_r) / kinf_safe

        # ── dM/dC · vC ────────────────────────────────────────────────────
        eff_r = effmatrix[r]
        dM_vC = 0.0
        for i in range(nsp):
            dM_vC += eff_r[i] * vCcell[i]

        # ── dPr/dC · vC ───────────────────────────────────────────────────
        dPr_vC = (k0_r / kinf_safe) * dM_vC

        # ── dPr_term/dPr = 1/(1+Pr)² ──────────────────────────────────────
        dPrterm_dPr = 1.0 / ((1.0 + Pr_r) * (1.0 + Pr_r))
        Pr_term     = Pr_r / (1.0 + Pr_r)

        # ── dk_base/dT et dk_base/dC·v ────────────────────────────────────
        dkbase_dT = dkinf_r * Pr_term + kinf_safe * dPrterm_dPr * dPr_dT
        dkbase_vC = kinf_safe * dPrterm_dPr * dPr_vC

        # ── dF/dT total et dF/dC·v ────────────────────────────────────────
        dF_dT_total = dF_dT + dF_dPr * dPr_dT
        dF_vC       = dF_dPr * dPr_vC

        # ── dkf · v ───────────────────────────────────────────────────────
        dkf_vT = (dkbase_dT * F_r + kbase_r * dF_dT_total) * vTcell
        dkf_vC = dkbase_vC * F_r + kbase_r * dF_vC

        # ── G et dG/dT ────────────────────────────────────────────────────
        G      = (1.0 - ow) / Kc_r * Cp_r
        dG_dT  = (1.0 - ow) * Cp_r * (-dKc_r / (Kc_r * Kc_r))
        conc_expr = Cr_r - G

        # ── dRR/dT · vT ───────────────────────────────────────────────────
        Jv_RR = dkf_vT * conc_expr + kf_r * (-dG_dT) * vTcell

        # ── dRR/dC · vC : contribution kf via concentrations ─────────────
        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            ci  = max(Ccell[idx], 1e-300)
            Jv_RR += kf_r * (sto * Cr_r / ci) * vCcell[idx]

        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            ci  = max(Ccell[idx], 1e-300)
            Jv_RR -= kf_r * ((1.0 - ow) / Kc_r) * (sto * Cp_r / ci) * vCcell[idx]

        # ── dRR/dC · vC : contribution dkf/dC (via Pr, M) ────────────────
        Jv_RR += dkf_vC * conc_expr

        # ── accumulation ──────────────────────────────────────────────────
        for i in range(react_nb):
            jvp_cell[react_idx[i + react_start]] -= react_sto[i + react_start] * Jv_RR
        for i in range(prod_nb):
            jvp_cell[prod_idx[i + prod_start]]   += prod_sto[i + prod_start]   * Jv_RR


# -----------------------------------------------------------------------------
#  VERSIONS PACKAGÉES
#  Layout react_arrays (falloff) :
#    [0] react_idx   [1] react_sto  [2] reactant_number  [3] reactant_offset
#    [4] prod_idx    [5] prod_sto   [6] prod_number       [7] prod_offset
#    [8]  A0_arr     [9]  n0_arr    [10] Ea0_arr
#    [11] Ainf_arr   [12] ninf_arr  [13] Eainf_arr
#    [14] oneWay     [15] delta_n
#    [16] effmatrix
#    [17] falloff_type_arr
#    [18] troe_A     [19] troe_T1   [20] troe_T2  [21] troe_T3
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def pack_falloff_context_packaged(
        Tcell, Ccell,
        react_arrays, thermo_arrays,
        ctx,
):
    pack_falloff_context(
        Tcell, Ccell,
        react_arrays[0],  react_arrays[1],  react_arrays[2],  react_arrays[3],
        react_arrays[4],  react_arrays[5],  react_arrays[6],  react_arrays[7],
        react_arrays[8],  react_arrays[9],  react_arrays[10],
        react_arrays[11], react_arrays[12], react_arrays[13],
        react_arrays[14], react_arrays[15],
        react_arrays[16],                  # effmatrix
        react_arrays[17],                  # falloff_type_arr
        react_arrays[18], react_arrays[19], react_arrays[20], react_arrays[21],
        thermo_arrays[1], thermo_arrays[0], thermo_arrays[2],
        ctx,
    )


@cuda.jit(device=True, inline=False)
def fo_omega_packaged(
        react_arrays,
        ctx,
        omega_cell,
):
    fo_omega(
        react_arrays[0],  react_arrays[1],  react_arrays[2],  react_arrays[3],
        react_arrays[4],  react_arrays[5],  react_arrays[6],  react_arrays[7],
        ctx, omega_cell,
    )


@cuda.jit(device=True, inline=False)
def fo_jvp_packaged(
        Ccell, vTcell, vCcell,
        react_arrays,
        ctx,
        jvp_cell,
):
    fo_jvp(
        Ccell,
        react_arrays[0],  react_arrays[1],  react_arrays[2],  react_arrays[3],
        react_arrays[4],  react_arrays[5],  react_arrays[6],  react_arrays[7],
        react_arrays[14],                  # oneWay
        react_arrays[16],                  # effmatrix
        ctx, vTcell, vCcell, jvp_cell,
    )