from numba import cuda
import math
import kineticsGPU.numba_implementation.constants as constants
import kineticsGPU.numba_implementation.chemistry as chemistry
import kineticsGPU.numba_implementation.thermo as thermo

# =============================================================================
#  PIPELINE OBLIGATOIRE
#
#  Étape 1 — pack_elementary_context(Tcell, Ccell, ..., ctx)
#               → remplit ctx une seule fois
#
#  Étape 2 — elem_omega(react_arrays, ctx, omega_cell)
#               → lit ctx, accumule omega
#
#  Étape 3 — elem_jvp(Ccell, react_arrays, oneWay, ctx, vTcell, vCcell, jvp_cell)
#               → lit ctx, calcule le produit Jacobien-vecteur
#
#  Les fonctions packagées (étapes 2+3 avec unpack des react_arrays inclus)
#  NE recalculent JAMAIS le contexte. Elles supposent que pack a déjà été
#  appelé sur ctx avant elles.
#
#  Layout mémoire du contexte :
#    ctx : tableau 2D (N_CTX_FIELDS, nreac) — tranche ctx_buf[cell_idx]
#    ctx_buf alloué côté host : cuda.device_array((ncells, N_CTX_FIELDS, nreac))
# =============================================================================

# ── Indices des champs ────────────────────────────────────────────────────────
CTX_KF  = 0   # kf(T)
CTX_KC  = 1   # Kc(T)
CTX_DKF = 2   # dkf/dT
CTX_DKC = 3   # dKc/dT  calculé comme Kc*(deltaS/(R*T) + deltaG0/(R*T²) - delta_n/T)
CTX_CR  = 4   # C_react = Π Ci^νi  (réactifs)
CTX_CP  = 5   # C_prod  = Π Ci^νi  (produits)
CTX_RR  = 6   # RR net

N_CTX_FIELDS = 7


# -----------------------------------------------------------------------------
#  ÉTAPE 1 — PACK
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def pack_elementary_context(
        Tcell, Ccell,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        A_arr, n_arr, Ea_arr,
        oneWay, delta_n,
        coef_low, coef_high, Tmid,
        ctx,   # (N_CTX_FIELDS, nreac)
):
    """
    Calcule et empaquète tous les intermédiaires communs à omega et jvp.
    Doit être appelé une seule fois avant elem_omega et elem_jvp.
    """
    nreac = A_arr.shape[0]

    for r in range(nreac):
        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        kf_r  = chemistry.arrhenius(A_arr[r], n_arr[r], Ea_arr[r], Tcell)

        dG0_r = chemistry.delta_G0(
            react_start, react_nb, prod_start, prod_nb,
            react_idx, prod_idx, react_sto, prod_sto,
            coef_high, coef_low, Tmid, Tcell)
        Kp_r  = chemistry.Kp(dG0_r, Tcell)
        Kc_r  = chemistry.Kc(delta_n[r], Tcell, Kp_r)

        dkf_r = chemistry.d_arrhenius_dT(kf_r, n_arr[r], Ea_arr[r], Tcell)

        # dKc/dT cohérent avec la formulation torch :
        #   dlnKc/dT = deltaS/(R*T) + deltaG0/(R*T²) - delta_n/T
        # On recalcule deltaS = Σ sto*s - Σ sto*s via les mêmes polynômes NASA
        # que ceux utilisés pour deltaG0, garantissant la cohérence.
        deltaS_r = chemistry.delta_S0(
            react_start, react_nb, prod_start, prod_nb,
            react_idx, prod_idx, react_sto, prod_sto,
            coef_high, coef_low, Tmid, Tcell)
        dKc_r = Kc_r * (
            deltaS_r / (constants.R * Tcell)
            + dG0_r   / (constants.R * Tcell * Tcell)
            - delta_n[r] / Tcell
        )

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

        RR_r = kf_r * (C_react - (1.0 - oneWay[r]) / Kc_r * C_prod)

        ctx[CTX_KF,  r] = kf_r
        ctx[CTX_KC,  r] = Kc_r
        ctx[CTX_DKF, r] = dkf_r
        ctx[CTX_DKC, r] = dKc_r
        ctx[CTX_CR,  r] = C_react
        ctx[CTX_CP,  r] = C_prod
        ctx[CTX_RR,  r] = RR_r


# ── Accesseurs inline (éliminés à la compilation) ────────────────────────────

@cuda.jit(device=True, inline=True)
def unpack_kf(ctx, r):  return ctx[CTX_KF,  r]

@cuda.jit(device=True, inline=True)
def unpack_Kc(ctx, r):  return ctx[CTX_KC,  r]

@cuda.jit(device=True, inline=True)
def unpack_dkf(ctx, r): return ctx[CTX_DKF, r]

@cuda.jit(device=True, inline=True)
def unpack_dKc(ctx, r): return ctx[CTX_DKC, r]

@cuda.jit(device=True, inline=True)
def unpack_Cr(ctx, r):  return ctx[CTX_CR,  r]

@cuda.jit(device=True, inline=True)
def unpack_Cp(ctx, r):  return ctx[CTX_CP,  r]

@cuda.jit(device=True, inline=True)
def unpack_RR(ctx, r):  return ctx[CTX_RR,  r]


# -----------------------------------------------------------------------------
#  ÉTAPE 2 — OMEGA  (suppose ctx déjà rempli par pack)
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def elem_omega(
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        ctx,        # (N_CTX_FIELDS, nreac) — déjà rempli
        omega_cell, # (nsp,)
):
    """
    Accumule les taux de production d'espèces depuis le contexte.
    Prérequis : pack_elementary_context a été appelé sur ctx.
    """
    nreac = reactant_number.shape[0]

    for r in range(nreac):
        RR_r        = unpack_RR(ctx, r)
        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            omega_cell[idx] -= sto * RR_r

        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            omega_cell[idx] += sto * RR_r


# -----------------------------------------------------------------------------
#  ÉTAPE 3 — JVP  (suppose ctx déjà rempli par pack)
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def elem_jvp(
        Ccell,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        oneWay,
        ctx,       # (N_CTX_FIELDS, nreac) — déjà rempli
        vTcell,    # scalaire
        vCcell,    # (nsp,)
        jvp_cell,  # (nsp,)
):
    """
    Produit Jacobien-vecteur d(omega)/d(T,C)·(vT,vC).
    Prérequis : pack_elementary_context a été appelé sur ctx.
    Aucun recalcul de kf, Kc, concentrations.
    """
    nreac = reactant_number.shape[0]

    for r in range(nreac):
        kf_r  = unpack_kf (ctx, r)
        Kc_r  = unpack_Kc (ctx, r)
        dkf_r = unpack_dkf(ctx, r)
        dKc_r = unpack_dKc(ctx, r)
        Cr_r  = unpack_Cr (ctx, r)
        Cp_r  = unpack_Cp (ctx, r)

        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        ow    = oneWay[r]
        G     = (1.0 - ow) / Kc_r * Cp_r

        dG_dT = (1.0 - ow) * Cp_r * (-dKc_r / (Kc_r * Kc_r))
        Jv_RR = (dkf_r * (Cr_r - G) + kf_r * (-dG_dT)) * vTcell

        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            ci  = max(Ccell[idx], 1e-300)
            Jv_RR += kf_r * (sto * Cr_r / ci) * vCcell[idx]

        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            ci  = max(Ccell[idx], 1e-300)
            dG_dCi = (1.0 - ow) / Kc_r * (sto * Cp_r / ci)
            Jv_RR -= kf_r * dG_dCi * vCcell[idx]

        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            jvp_cell[idx] -= sto * Jv_RR

        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            jvp_cell[idx] += sto * Jv_RR


# -----------------------------------------------------------------------------
#  VERSIONS PACKAGÉES  (dépaquètent react_arrays, NE recalculent PAS le ctx)
#  Prérequis dans les deux cas : pack_elementary_context déjà appelé sur ctx.
# -----------------------------------------------------------------------------

@cuda.jit(device=True, inline=False)
def elem_omega_packaged(
        react_arrays,
        ctx,        # (N_CTX_FIELDS, nreac) — déjà rempli
        omega_cell, # (nsp,)
):
    """
    Accumule omega. Ne recalcule pas le contexte.
    Prérequis : pack_elementary_context(Tcell, Ccell, ..., ctx) déjà appelé.
    """
    elem_omega(
        react_arrays[0],  # react_idx
        react_arrays[1],  # react_sto
        react_arrays[2],  # reactant_number
        react_arrays[3],  # reactant_offset
        react_arrays[4],  # prod_idx
        react_arrays[5],  # prod_sto
        react_arrays[6],  # prod_number
        react_arrays[7],  # prod_offset
        ctx, omega_cell,
    )


@cuda.jit(device=True, inline=False)
def elem_jvp_packaged(
        Ccell,
        react_arrays,
        ctx,       # (N_CTX_FIELDS, nreac) — déjà rempli
        vTcell,    # scalaire
        vCcell,    # (nsp,)
        jvp_cell,  # (nsp,)
):
    """
    Calcule le JVP. Ne recalcule pas le contexte.
    Prérequis : pack_elementary_context(Tcell, Ccell, ..., ctx) déjà appelé.
    """
    elem_jvp(
        Ccell,
        react_arrays[0],  # react_idx
        react_arrays[1],  # react_sto
        react_arrays[2],  # reactant_number
        react_arrays[3],  # reactant_offset
        react_arrays[4],  # prod_idx
        react_arrays[5],  # prod_sto
        react_arrays[6],  # prod_number
        react_arrays[7],  # prod_offset
        react_arrays[11], # oneWay
        ctx, vTcell, vCcell, jvp_cell,
    )


@cuda.jit(device=True, inline=False)
def pack_elementary_context_packaged(
        Tcell, Ccell,
        react_arrays, thermo_arrays,
        ctx,   # (N_CTX_FIELDS, nreac)
):
    """
    Version packagée du pack : dépaquète react_arrays et thermo_arrays
    puis appelle pack_elementary_context.
    À appeler en étape 1 avant elem_omega_packaged / elem_jvp_packaged.
    """
    pack_elementary_context(
        Tcell, Ccell,
        react_arrays[0],  # react_idx
        react_arrays[1],  # react_sto
        react_arrays[2],  # reactant_number
        react_arrays[3],  # reactant_offset
        react_arrays[4],  # prod_idx
        react_arrays[5],  # prod_sto
        react_arrays[6],  # prod_number
        react_arrays[7],  # prod_offset
        react_arrays[8],  # A_arr
        react_arrays[9],  # n_arr
        react_arrays[10], # Ea_arr
        react_arrays[11], # oneWay
        react_arrays[12], # delta_n
        thermo_arrays[1], # coef_low
        thermo_arrays[0], # coef_high
        thermo_arrays[2], # Tmid
        ctx,
    )


# =============================================================================
#  EXEMPLE D'UTILISATION DANS UN KERNEL
#
#  ── Host ─────────────────────────────────────────────────────────────────────
#
#    import numpy as np
#    from numba import cuda
#
#    ctx_buf = cuda.device_array((ncells, N_CTX_FIELDS, nreac), dtype=np.float64)
#
#  ── Kernel ───────────────────────────────────────────────────────────────────
#
#    @cuda.jit
#    def implicit_step_kernel(T, C, vT, vC, omega, jvp_omega,
#                              react_arrays, thermo_arrays, ctx_buf):
#        cell_idx = cuda.grid(1)
#        if cell_idx >= T.shape[0]:
#            return
#
#        Tcell      = T        [cell_idx]
#        Ccell      = C        [cell_idx]
#        vTcell     = vT       [cell_idx]
#        vCcell     = vC       [cell_idx]
#        ctx        = ctx_buf  [cell_idx]   # (N_CTX_FIELDS, nreac)
#        omega_cell = omega    [cell_idx]
#        jvp_cell   = jvp_omega[cell_idx]
#
#        # Étape 1 — pack (un seul calcul)
#        pack_elementary_context_packaged(Tcell, Ccell,
#                                         react_arrays, thermo_arrays, ctx)
#
#        # Étape 2 — omega (lit ctx)
#        elem_omega_packaged(react_arrays, ctx, omega_cell)
#
#        # Étape 3 — jvp (lit ctx, peut être appelé plusieurs fois
#        #                dans la boucle Newton sans recalculer ctx)
#        elem_jvp_packaged(Ccell, react_arrays, ctx, vTcell, vCcell, jvp_cell)
# =============================================================================