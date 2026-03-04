from numba import cuda
import kineticsGPU.numba_implementation.chemistry as chemistry


# Sentinelles pour le type de falloff stocké dans falloff_type_arr
FALLOFF_LINDEMANN = 0
FALLOFF_TROE      = 1


@cuda.jit(device=True, inline=False)
def per_thread_falloff_react(
        # state
        T, C,                          # (ncells,)  (ncells, nsp)
        cell_idx,
        # NASA thermo
        coef_low, coef_high,           # (nsp, 7)
        Tmid,                          # (nsp,)
        # reaction connectivity
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto, prod_number,  prod_offset,
        # k0 / kinf Arrhenius
        A0_arr,  n0_arr,  Ea0_arr,     # (nreac,)  basse pression
        Ainf_arr, ninf_arr, Eainf_arr, # (nreac,)  haute pression
        # flags
        oneWay,                        # (nreac,)
        delta_n,                       # (nreac,)
        # third-body efficiencies
        effmatrix,                     # (nreac, nsp)
        # falloff type + Troe params
        falloff_type_arr,              # (nreac,)  int32 : 0=Lindemann, 1=Troe # TODO: passer un tableau bool par exemple tant qu'il n'y a que deux possibilités pour save de la mémoire
        troe_A,  troe_T1,             # (nreac,)  — valeur 0.0 si absent
        troe_T2, troe_T3,             # (nreac,)
        # output
        omega                          # (ncells, nsp)
):
    Tcell   = T[cell_idx]
    Ccell   = C[cell_idx]
    nreac   = A0_arr.shape[0]

    for r in range(nreac):
        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        # --- k0 et kinf ---
        k0   = chemistry.arrhenius(A0_arr[r],   n0_arr[r],   Ea0_arr[r],   Tcell)
        kinf = chemistry.arrhenius(Ainf_arr[r], ninf_arr[r], Eainf_arr[r], Tcell)

        # --- concentration efficace M ---
        M_val = chemistry.M(Ccell, effmatrix[r])

        # --- facteur de falloff F ---
        ftype = falloff_type_arr[r]
        if ftype == FALLOFF_TROE:
            Pr  = chemistry.reduced_pressure(k0, kinf, M_val)
            F   = chemistry.F_troe(Tcell, Pr,
                                   troe_A[r], troe_T1[r],
                                   troe_T2[r], troe_T3[r])
        else:  # Lindemann
            F = chemistry.F_lindemann()

        # --- constante de vitesse effective ---
        kf = chemistry.kf_falloff(k0, kinf, M_val, F)

        # --- Kc pour la réaction inverse ---
        dG0 = chemistry.delta_G0(react_start, react_nb,
                                  prod_start,  prod_nb,
                                  react_idx,   prod_idx,
                                  react_sto,   prod_sto,
                                  coef_high, coef_low, Tmid, Tcell)
        Kp_val = chemistry.Kp(dG0, Tcell)
        Kc_val = chemistry.Kc(delta_n[r], Tcell, Kp_val)

        # --- taux de réaction ---
        RR = chemistry.elem_RR(react_start, react_nb,
                                prod_start,  prod_nb,
                                react_idx,   prod_idx,
                                react_sto,   prod_sto,
                                Ccell, kf, oneWay[r], Kc_val)

        # --- accumulation dans omega ---
        for i in range(react_nb):
            idx = react_idx[i + react_start]
            omega[cell_idx, idx] -= react_sto[i + react_start] * RR
        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            omega[cell_idx, idx] += prod_sto[i + prod_start] * RR

@cuda.jit(device=True, inline=False)
def per_thread_falloff_react_jvp(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        A0_arr,  n0_arr,  Ea0_arr,
        Ainf_arr, ninf_arr, Eainf_arr,
        oneWay, delta_n,
        effmatrix,
        falloff_type_arr,
        troe_A, troe_T1, troe_T2, troe_T3,
        vT, vC,
        omega,
        jvp_omega):

    Tcell  = T[cell_idx]
    Ccell  = C[cell_idx]
    vTcell = vT[cell_idx]
    vCcell = vC[cell_idx]
    nreac  = A0_arr.shape[0]
    nsp    = Ccell.shape[0]

    for r in range(nreac):
        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]
        eff_r       = effmatrix[r]

        # ── Arrhenius k0 / kinf ───────────────────────────────────────────
        k0   = chemistry.arrhenius(A0_arr[r],   n0_arr[r],   Ea0_arr[r],   Tcell)
        kinf = chemistry.arrhenius(Ainf_arr[r], ninf_arr[r], Eainf_arr[r], Tcell)

        dk0_dT   = chemistry.d_arrhenius_dT(k0,   n0_arr[r],   Ea0_arr[r],   Tcell)
        dkinf_dT = chemistry.d_arrhenius_dT(kinf, ninf_arr[r], Eainf_arr[r], Tcell)

        # ── M et dM/dC · v ───────────────────────────────────────────────
        M_val = chemistry.M(Ccell, eff_r)

        dM_v = 0.0
        for i in range(nsp):
            dM_v += eff_r[i] * vCcell[i]

        # ── Pr et ses dérivées ────────────────────────────────────────────
        # Pr = k0 * M / kinf
        Pr = k0 * M_val / kinf

        # dPr/dT = (dk0/dT * M - k0*M/kinf * dkinf/dT) / kinf
        dPr_dT = (dk0_dT * M_val - k0 * M_val / kinf * dkinf_dT) / kinf

        # dPr/dC · v = (k0/kinf) * dM_v
        dPr_v = (k0 / kinf) * dM_v

        # ── k_base = kinf * Pr/(1+Pr) ─────────────────────────────────────
        Pr_term      = Pr / (1.0 + Pr)
        dPrterm_dPr  = 1.0 / ((1.0 + Pr) * (1.0 + Pr))

        k_base = kinf * Pr_term

        dk_base_dT = dkinf_dT * Pr_term + kinf * dPrterm_dPr * dPr_dT
        dk_base_v  = kinf * dPrterm_dPr * dPr_v   # contribution C seulement

        # ── facteur de falloff F et ses dérivées ──────────────────────────
        ftype = falloff_type_arr[r]

        if ftype == FALLOFF_TROE:
            # F_troe_and_derivatives renvoie (F, dF/dT_direct, dF/dPr)
            F, dF_dT_direct, dF_dPr = chemistry.F_troe_and_derivatives(
                Tcell, Pr,
                troe_A[r], troe_T1[r], troe_T2[r], troe_T3[r])

            # dF/dT total = dF/dT_direct + dF/dPr * dPr/dT
            dF_dT_total = dF_dT_direct + dF_dPr * dPr_dT
            # dF/dC · v = dF/dPr * dPr/dC · v
            dF_v = dF_dPr * dPr_v
        else:
            # Lindemann : F = 1, toutes les dérivées nulles
            F           = 1.0
            dF_dT_total = 0.0
            dF_v        = 0.0

        # ── kf = k_base * F ───────────────────────────────────────────────
        kf = k_base * F

        # dkf/dT = dk_base/dT * F + k_base * dF/dT_total
        dkf_dT_v_scalar = (dk_base_dT * F + k_base * dF_dT_total) * vTcell

        # dkf/dC · v = (dk_base/dC·v) * F + k_base * (dF/dC·v)
        dkf_C_v = dk_base_v * F + k_base * dF_v

        # dkf total · (vT, vC)
        dkf_total_v = dkf_dT_v_scalar + dkf_C_v

        # ── Kc ────────────────────────────────────────────────────────────
        dG0  = chemistry.delta_G0(react_start, react_nb, prod_start, prod_nb,
                                  react_idx, prod_idx, react_sto, prod_sto,
                                  coef_high, coef_low, Tmid, Tcell)
        Kp_v = chemistry.Kp(dG0, Tcell)
        Kc_v = chemistry.Kc(delta_n[r], Tcell, Kp_v)

        dKc_dT_v = chemistry.dKc_dT(react_start, react_nb, prod_start, prod_nb,
                                     react_idx, prod_idx, react_sto, prod_sto,
                                     coef_high, coef_low, Tmid,
                                     Tcell, Kc_v, delta_n[r])

        # ── C_react et C_prod ─────────────────────────────────────────────
        C_react = 1.0
        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            C_react *= Ccell[idx] ** sto

        C_prod = 1.0
        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            C_prod *= Ccell[idx] ** sto

        G  = (1.0 - oneWay[r]) / Kc_v * C_prod
        RR = kf * (C_react - G)

        # ── Jv_RR ─────────────────────────────────────────────────────────
        # Jv = dkf·v * (C_react - G)
        #    + kf * (dC_react/dC · vC  -  dG/dT·vT  -  dG/dC·vC)
        #
        # dG/dT = (1-ow) * C_prod * (-dKc/dT / Kc²)

        dG_dT  = (1.0 - oneWay[r]) * C_prod * (-dKc_dT_v / (Kc_v * Kc_v))
        Jv_RR  = dkf_total_v * (C_react - G) + kf * (-dG_dT * vTcell)

        # contribution dC_react/dCi · vCi (réactifs)
        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            d_Cr_dCi = sto * C_react / max(Ccell[idx], 1e-300)
            Jv_RR   += kf * d_Cr_dCi * vCcell[idx]

        # contribution -dG/dCi · vCi (produits)
        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            d_Cp_dCi = sto * C_prod / max(Ccell[idx], 1e-300)
            dG_dCi   = (1.0 - oneWay[r]) / Kc_v * d_Cp_dCi
            Jv_RR   -= kf * dG_dCi * vCcell[idx]

        # ── accumulation ──────────────────────────────────────────────────
        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            omega    [cell_idx, idx] -= sto * RR
            jvp_omega[cell_idx, idx] -= sto * Jv_RR
        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            omega    [cell_idx, idx] += sto * RR
            jvp_omega[cell_idx, idx] += sto * Jv_RR