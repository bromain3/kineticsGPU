from numba import cuda
import math
import kineticsGPU.numba_implementation.constants as constants
import kineticsGPU.numba_implementation.chemistry as chemistry

# =============================================================================
#  PIPELINE THREE-BODY
#
#  Étape 1 — pack_threebody_context_packaged(Tcell, Ccell, react_arrays,
#                                             thermo_arrays, ctx)
#  Étape 2 — tb_omega_packaged(react_arrays, ctx, omega_cell)
#  Étape 3 — tb_jvp_packaged(Ccell, vTcell, vCcell, react_arrays, ctx, jvp_cell)
#
#  Layout ctx (N_CTX_TB_FIELDS, nreac) :
#
#    ctx[CTX_TB_KF,      r]  kf = A*T^n*exp(-Ea/RT)
#    ctx[CTX_TB_KC,      r]  Kc
#    ctx[CTX_TB_DKF,     r]  dkf/dT
#    ctx[CTX_TB_DKC,     r]  dKc/dT  (via deltaS + deltaG0, cohérent avec Kc)
#    ctx[CTX_TB_CR,      r]  C_react
#    ctx[CTX_TB_CP,      r]  C_prod
#    ctx[CTX_TB_M,       r]  [M]  concentration efficace du troisième corps
#    ctx[CTX_TB_BASERR,  r]  baseRR = kf*(C_react - (1-ow)/Kc*C_prod)
#    ctx[CTX_TB_RR,      r]  RR = baseRR * M
# =============================================================================

CTX_TB_KF     = 0
CTX_TB_KC     = 1
CTX_TB_DKF    = 2
CTX_TB_DKC    = 3
CTX_TB_CR     = 4
CTX_TB_CP     = 5
CTX_TB_M      = 6
CTX_TB_BASERR = 7
CTX_TB_RR     = 8

N_CTX_TB_FIELDS = 9


# ── Accesseurs inline ────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _tb_kf(ctx, r):     return ctx[CTX_TB_KF,     r]

@cuda.jit(device=True, inline=True)
def _tb_Kc(ctx, r):     return ctx[CTX_TB_KC,     r]

@cuda.jit(device=True, inline=True)
def _tb_dkf(ctx, r):    return ctx[CTX_TB_DKF,    r]

@cuda.jit(device=True, inline=True)
def _tb_dKc(ctx, r):    return ctx[CTX_TB_DKC,    r]

@cuda.jit(device=True, inline=True)
def _tb_Cr(ctx, r):     return ctx[CTX_TB_CR,     r]

@cuda.jit(device=True, inline=True)
def _tb_Cp(ctx, r):     return ctx[CTX_TB_CP,     r]

@cuda.jit(device=True, inline=True)
def _tb_M(ctx, r):      return ctx[CTX_TB_M,      r]

@cuda.jit(device=True, inline=True)
def _tb_baseRR(ctx, r): return ctx[CTX_TB_BASERR, r]

@cuda.jit(device=True, inline=True)
def _tb_RR(ctx, r):     return ctx[CTX_TB_RR,     r]


# -----------------------------------------------------------------------------
#  ÉTAPE 1 — PACK
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def pack_threebody_context(
        Tcell, Ccell,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        A_arr, n_arr, Ea_arr,
        oneWay, delta_n,
        effmatrix,             # (nreac, nsp)
        coef_low, coef_high, Tmid,
        ctx,                   # (N_CTX_TB_FIELDS, nreac)
):
    nreac = A_arr.shape[0]

    for r in range(nreac):
        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        # ── Arrhenius ──────────────────────────────────────────────────────
        kf_r  = chemistry.arrhenius(A_arr[r], n_arr[r], Ea_arr[r], Tcell)
        dkf_r = chemistry.d_arrhenius_dT(kf_r, n_arr[r], Ea_arr[r], Tcell)

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

        # ── [M] ────────────────────────────────────────────────────────────
        M_r = chemistry.M(Ccell, effmatrix[r])

        # ── baseRR et RR ───────────────────────────────────────────────────
        baseRR_r = kf_r * (C_react - (1.0 - oneWay[r]) / Kc_r * C_prod)
        RR_r     = baseRR_r * M_r

        ctx[CTX_TB_KF,     r] = kf_r
        ctx[CTX_TB_KC,     r] = Kc_r
        ctx[CTX_TB_DKF,    r] = dkf_r
        ctx[CTX_TB_DKC,    r] = dKc_r
        ctx[CTX_TB_CR,     r] = C_react
        ctx[CTX_TB_CP,     r] = C_prod
        ctx[CTX_TB_M,      r] = M_r
        ctx[CTX_TB_BASERR, r] = baseRR_r
        ctx[CTX_TB_RR,     r] = RR_r


# -----------------------------------------------------------------------------
#  ÉTAPE 2 — OMEGA
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def tb_omega(
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        ctx,
        omega_cell,
):
    nreac = reactant_number.shape[0]
    for r in range(nreac):
        RR_r        = _tb_RR(ctx, r)
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
#  RR_3b   = baseRR * M
#  dRR_3b  = dbaseRR * M  +  baseRR * dM
#
#  dbaseRR/dT  = dkf/dT * (Cr - G)  +  kf * (-dG/dT)
#  dbaseRR/dCi = kf * nu_i*Cr/Ci  (réactif)  ou  -kf*(1-ow)/Kc*nu_i*Cp/Ci (produit)
#  dM/dCi      = eff_i
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def tb_jvp(
        Ccell,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        oneWay,
        effmatrix,             # (nreac, nsp)
        ctx,
        vTcell, vCcell,
        jvp_cell,
):
    nreac = reactant_number.shape[0]
    nsp   = Ccell.shape[0]

    for r in range(nreac):
        kf_r     = _tb_kf    (ctx, r)
        Kc_r     = _tb_Kc    (ctx, r)
        dkf_r    = _tb_dkf   (ctx, r)
        dKc_r    = _tb_dKc   (ctx, r)
        Cr_r     = _tb_Cr    (ctx, r)
        Cp_r     = _tb_Cp    (ctx, r)
        M_r      = _tb_M     (ctx, r)
        baseRR_r = _tb_baseRR(ctx, r)

        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        ow = oneWay[r]
        G  = (1.0 - ow) / Kc_r * Cp_r

        # ── dbaseRR/dT · vT ───────────────────────────────────────────────
        dG_dT      = (1.0 - ow) * Cp_r * (-dKc_r / (Kc_r * Kc_r))
        dbaseRR_vT = (dkf_r * (Cr_r - G) + kf_r * (-dG_dT)) * vTcell

        # ── dbaseRR/dC · vC ───────────────────────────────────────────────
        dbaseRR_vC = 0.0
        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            ci  = max(Ccell[idx], 1e-300)
            dbaseRR_vC += kf_r * (sto * Cr_r / ci) * vCcell[idx]
        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            ci  = max(Ccell[idx], 1e-300)
            dbaseRR_vC -= kf_r * ((1.0 - ow) / Kc_r) * (sto * Cp_r / ci) * vCcell[idx]

        # ── dM/dC · vC ────────────────────────────────────────────────────
        eff_r = effmatrix[r]
        dM_vC = 0.0
        for i in range(nsp):
            dM_vC += eff_r[i] * vCcell[i]

        # ── Jv_RR_3b = dbaseRR·v * M  +  baseRR * dM·v ───────────────────
        Jv_RR = (dbaseRR_vT + dbaseRR_vC) * M_r + baseRR_r * dM_vC

        for i in range(react_nb):
            jvp_cell[react_idx[i + react_start]] -= react_sto[i + react_start] * Jv_RR
        for i in range(prod_nb):
            jvp_cell[prod_idx[i + prod_start]]   += prod_sto[i + prod_start]   * Jv_RR


# -----------------------------------------------------------------------------
#  VERSIONS PACKAGÉES
#  Layout react_arrays (three-body) :
#    [0] react_idx   [1] react_sto  [2] reactant_number  [3] reactant_offset
#    [4] prod_idx    [5] prod_sto   [6] prod_number       [7] prod_offset
#    [8] A_arr       [9] n_arr      [10] Ea_arr
#    [11] oneWay     [12] delta_n   [13] effmatrix
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def pack_threebody_context_packaged(
        Tcell, Ccell,
        react_arrays, thermo_arrays,
        ctx,
):
    pack_threebody_context(
        Tcell, Ccell,
        react_arrays[0],  react_arrays[1],  react_arrays[2],  react_arrays[3],
        react_arrays[4],  react_arrays[5],  react_arrays[6],  react_arrays[7],
        react_arrays[8],  react_arrays[9],  react_arrays[10],
        react_arrays[11], react_arrays[12],
        react_arrays[13],                  # effmatrix
        thermo_arrays[1], thermo_arrays[0], thermo_arrays[2],  # coef_low, coef_high, Tmid
        ctx,
    )


@cuda.jit(device=True, inline=False)
def tb_omega_packaged(
        react_arrays,
        ctx,
        omega_cell,
):
    tb_omega(
        react_arrays[0],  react_arrays[1],  react_arrays[2],  react_arrays[3],
        react_arrays[4],  react_arrays[5],  react_arrays[6],  react_arrays[7],
        ctx, omega_cell,
    )


@cuda.jit(device=True, inline=False)
def tb_jvp_packaged(
        Ccell, vTcell, vCcell,
        react_arrays,
        ctx,
        jvp_cell,
):
    tb_jvp(
        Ccell,
        react_arrays[0],  react_arrays[1],  react_arrays[2],  react_arrays[3],
        react_arrays[4],  react_arrays[5],  react_arrays[6],  react_arrays[7],
        react_arrays[11],                  # oneWay
        react_arrays[13],                  # effmatrix
        ctx, vTcell, vCcell, jvp_cell,
    )