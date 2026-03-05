from numba import cuda
import constants
import thermo
import math
from kineticsGPU.numba_implementation.elementary_reaction import (
    pack_elementary_context_packaged,
    elem_omega_packaged,
    elem_jvp_packaged,
    N_CTX_FIELDS as N_CTX_ELEM,
)
from kineticsGPU.numba_implementation.three_body_reaction import (
    pack_threebody_context_packaged,
    tb_omega_packaged,
    tb_jvp_packaged,
    N_CTX_TB_FIELDS as N_CTX_TB,
)
from kineticsGPU.numba_implementation.falloff_reaction import (
    pack_falloff_context_packaged,
    fo_omega_packaged,
    fo_jvp_packaged,
    N_CTX_FO_FIELDS as N_CTX_FO,
)

@cuda.jit(device=True, inline=False)
def constant_volume_reactor_rhs(
        Tcell,                    # scalaire : T de la cellule (K)
        Ccell,                    # (nsp,)   : concentrations mol/m³
        elem_reac_ctx,            # (N_CTX_ELEM, nreac_elem)
        elem_react_arrays,        # tuple des arrays élémentaires
        three_body_react_ctx,     # (N_CTX_TB,   nreac_tb)
        three_body_react_arrays,  # tuple des arrays three-body
        falloff_react_ctx,        # (N_CTX_FO,   nreac_fo)
        falloff_react_arrays,     # tuple des arrays falloff
        thermo_arrays,            # tuple : (coef_high, coef_low, Tmid)
        dXdt,                     # (nsp + 1,) : sortie
                                  #   dXdt[0..nsp-1] = dCdt  (mol/m³/s)
                                  #   dXdt[nsp]      = dTdt  (K/s)
):
    """
    Calcule le membre de droite du réacteur adiabatique à volume constant.

    Logique identique à rhs_adiabatic (torch) :
        dCdt         ← accumulation des omega des trois types
        n_tot        = Σ Ccell[i]
        X[i]         = Ccell[i] / n_tot
        cv_mol_mix   = (Σ_i X[i] * cp_mol_i) - R
        dUdt         = Σ_i u_mol_i(T) * dCdt[i]
        dTdt         = -dUdt / (n_tot * cv_mol_mix)
    """
    nsp = Ccell.shape[0]

    # ── 1. dCdt via omega (accumulé dans dXdt[0..nsp-1]) ─────────────────────
    #    Les fonctions omega_packaged font += sur omega_cell, donc on doit
    #    partir de zéro.
    for i in range(nsp):
        dXdt[i] = 0.0

    # dXdt[0..nsp-1] joue le rôle de omega_cell
    elem_omega_packaged(elem_react_arrays,        elem_reac_ctx,        dXdt)
    tb_omega_packaged  (three_body_react_arrays,  three_body_react_ctx, dXdt)
    fo_omega_packaged  (falloff_react_arrays,     falloff_react_ctx,    dXdt)

    # ── 2. n_tot = Σ Ccell[i] ────────────────────────────────────────────────
    n_tot = 0.0
    for i in range(nsp):
        n_tot += Ccell[i]

    # ── 3. cv_mol_mix = cp_mol_mix - R ───────────────────────────────────────
    #    cp_mol_mix = Σ_i X[i] * cp_mol_i  avec X[i] = Ccell[i] / n_tot
    coef_high = thermo_arrays[0]
    coef_low  = thermo_arrays[1]
    Tmid      = thermo_arrays[2]

    cp_mol_mix = 0.0
    for i in range(nsp):
        Xi = Ccell[i] / n_tot                           # fraction molaire
        cp_mol_mix += Xi * thermo.cp_mol_sp(coef_high, coef_low, Tmid, i, Tcell)

    cv_mol_mix = cp_mol_mix - constants.R                             # Cv = Cp - R (molaire)

    # ── 4. dUdt = Σ_i u_mol_i(T) * dCdt[i] ──────────────────────────────────
    dUdt = 0.0
    for i in range(nsp):
        u_i   = thermo.u_mol_sp(coef_high, coef_low, Tmid, i, Tcell)
        dUdt += u_i * dXdt[i]                                # dXdt[i] = dCdt[i]

    # ── 5. dTdt = -dUdt / (n_tot * cv_mol_mix) ───────────────────────────────
    denom = n_tot * cv_mol_mix

    dXdt[nsp] = -dUdt / denom



@cuda.jit(device=True, inline=False)
def constant_volume_reactor_jvp(
        Tcell,                    # scalaire T  (K)
        Ccell,                    # (nsp,)      concentrations mol/m³
        vTcell,                   # scalaire v_T
        vCcell,                   # (nsp,)      v_C
        # contextes déjà remplis par pack_*_context_packaged
        elem_reac_ctx,
        elem_react_arrays,
        three_body_react_ctx,
        three_body_react_arrays,
        falloff_react_ctx,
        falloff_react_arrays,
        thermo_arrays,
        dXdt,  # (nsp+1,)  dCdt  [0..nsp-1]  +  dTdt  [nsp]
        # sortie  (buffer remis à zéro)
        jvp_out, # (nsp+1,)  dCdt_jvp [0..nsp-1] + dT'  [nsp]
):
    """
    JVP du membre de droite du réacteur adiabatique à volume constant.
    Transcription exacte de jvp_rhs_adiabatic (torch).

    Pré-condition : les contextes doivent déjà être remplis (pack_*_context_packaged
    appelé avant cette fonction, typiquement dans constant_volume_reactor_rhs).
    jvp_out est mis à zéro.
    """
    nsp       = Ccell.shape[0]
    coef_high = thermo_arrays[0]
    coef_low  = thermo_arrays[1]
    Tmid      = thermo_arrays[2]

    # ── dTC'dt  (slice dXdt[0..nsp-1]) ─────────────────────────────────────────
    # Remise à zéro + accumulation jvp
    for i in range(nsp):
        jvp_out[i] = 0.0

    # ── dCdt_jvp  (slice jvp_out[0..nsp-1]) ──────────────────────────────────
    # Note : les jvp_packaged s'attendent à une slice (nsp,), pas (nsp+1,).
    # On utilise la slice directement via un tableau intermédiaire de vue.
    # Numba ne supporte pas les slices dynamiques comme sortie d'une device fn,
    # donc on accumule dans jvp_out[0..nsp-1] directement (même layout).
    elem_jvp_packaged(Ccell, elem_react_arrays,       elem_reac_ctx,
                      vTcell, vCcell, jvp_out)
    tb_jvp_packaged  (Ccell, vTcell, vCcell,
                      three_body_react_arrays, three_body_react_ctx, jvp_out)
    fo_jvp_packaged  (Ccell, vTcell, vCcell,
                      falloff_react_arrays,    falloff_react_ctx,    jvp_out)

    # ── scalaires scalaires scalaires ─────────────────────────────────────────
    # n_tot = Σ C[i]
    n_tot = 0.0
    for i in range(nsp):
        n_tot += Ccell[i]

    # dn_tot = Σ v_C[i]
    dn_tot = 0.0
    for i in range(nsp):
        dn_tot += vCcell[i]

    # ── boucle unique sur les espèces : accumule tous les scalaires ───────────
    #
    # Pour chaque espèce i on calcule une fois :
    #   X[i], dX[i], u_mol_i, cv_mol_i, dcv_mol_i
    #
    # et on accumule simultanément :
    #   cv_mix, dcv_mix_X (partie dX), dcv_mix_T (partie v_T),
    #   U_prime, dU_prime_C (partie dCdt_jvp), dU_prime_T (partie v_T)

    cv_mix      = 0.0   # Σ X[i] * cv_mol_i
    dcv_mix_X   = 0.0   # Σ dX[i] * cv_mol_i
    dcv_mix_T   = 0.0   # Σ X[i]  * dcv_mol_i   (multiplié par v_T ensuite)
    U_prime     = 0.0   # Σ u_mol_i * dCdt[i]
    dU_prime_C  = 0.0   # Σ u_mol_i * dCdt_jvp[i]
    dU_prime_T  = 0.0   # Σ cv_mol_i * dCdt[i]   (multiplié par v_T ensuite)

    for i in range(nsp):
        Xi    = Ccell[i] / n_tot
        # dX[i] = v_C[i]/n_tot  -  C[i]*dn_tot / n_tot²
        dXi   = vCcell[i] / n_tot  -  Ccell[i] * dn_tot / (n_tot * n_tot)

        cv_i   = thermo.cv_mol_sp   (coef_high, coef_low, Tmid, i, Tcell)
        dcv_i  = thermo.dcv_mol_sp_dT(coef_high, coef_low, Tmid, i, Tcell)
        u_i    = thermo.u_mol_sp    (coef_high, coef_low, Tmid, i, Tcell)

        dCdt_i     = dXdt   [i]   # déjà calculé ci-dessus
        dCdt_jvp_i = jvp_out[i]   # déjà calculé ci-dessus

        cv_mix     += Xi  * cv_i
        dcv_mix_X  += dXi * cv_i
        dcv_mix_T  += Xi  * dcv_i

        U_prime    += u_i  * dCdt_i
        dU_prime_C += u_i  * dCdt_jvp_i
        dU_prime_T += cv_i * dCdt_i

    # dcv_mix = dcv_mix_X + dcv_mix_T * v_T
    dcv_mix  = dcv_mix_X + dcv_mix_T * vTcell

    # dU' = dU'_C + dU'_T * v_T
    dU_prime = dU_prime_C + dU_prime_T * vTcell

    # ── dT' ───────────────────────────────────────────────────────────────────
    #
    # dT' = - dU' / (n_tot * cv_mix)
    #       + U' * dn_tot / (n_tot² * cv_mix)
    #       + U' * dcv_mix / (n_tot * cv_mix²)
    #

    denom1 = n_tot * cv_mix
    denom2 = n_tot * n_tot * cv_mix
    denom3 = n_tot * cv_mix * cv_mix

    dT_prime = (- dU_prime / denom1
                + U_prime * dn_tot   / denom2
                + U_prime * dcv_mix  / denom3)

    jvp_out[nsp] = dT_prime