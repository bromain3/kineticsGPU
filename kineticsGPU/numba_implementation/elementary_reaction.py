from numba import cuda
import kineticsGPU.numba_implementation.chemistry as chemistry


@cuda.jit(device=True, inline=False)
def per_thread_elem_react( # TODO: to improve computation speed, take cell variables as input tu prevent global memory use (and use local arrays in the kernel to have only one global memory read/write)
        # state
        T, C,  # (ncells,)  (ncells, nsp)
        cell_idx,
        # NASA thermo
        coef_low, coef_high,  # (nsp, 7)
        Tmid,  # (nsp,)
        # reaction connectivity
        react_idx, react_sto, reactant_number, reactant_offset,  # flattened reactant data
        prod_idx, prod_sto, prod_number, prod_offset,  # flattened product  data
        # Arrhenius + flags
        A_arr, n_arr, Ea_arr,  # (nreac,)
        oneWay,  # (nreac,)  1.0 = irreversible
        delta_n,  # (nreac,)  Δν = Σν_prod - Σν_react
        # output
        omega  # (ncells, nsp)  species production rates
):

    Tcell = T[cell_idx]
    Ccell = C[cell_idx]
    reaction_number = A_arr.shape[0]

    for r in range(reaction_number):
        Areact = A_arr[r]
        nreact = n_arr[r]
        Eareact = Ea_arr[r]
        oneWayreact = oneWay[r]
        delta_nreact = delta_n[r]
        react_start = reactant_offset[r]
        react_nb = reactant_number[r]
        prod_start = prod_offset[r]
        prod_nb = prod_number[r]


        kf = chemistry.arrhenius(Areact, nreact, Eareact, Tcell)
        dG0 = chemistry.delta_G0(react_start, react_nb,
                                 prod_start, prod_nb,
                                 react_idx, prod_idx,
                                 react_sto, prod_sto,
                                 coef_high, coef_low, Tmid, Tcell)

        Kp = chemistry.Kp(dG0, Tcell)
        Kc = chemistry.Kc(delta_nreact, Tcell, Kp)

        RR = chemistry.elem_RR(react_start, react_nb,
                               prod_start, prod_nb,
                               react_idx, prod_idx,
                               react_sto, prod_sto,
                               Ccell, kf, oneWayreact, Kc)

        for i in range(react_nb):
            idx = react_idx[i + react_start]
            sto = react_sto[i + react_start]
            omega[cell_idx, idx] -= sto * RR
        for i in range(prod_nb):
            idx = prod_idx[i + prod_start]
            sto = prod_sto[i + prod_start]
            omega[cell_idx, idx] += sto * RR


@cuda.jit(device=True, inline=False)
def per_thread_elem_react_jvp(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        A_arr, n_arr, Ea_arr,
        oneWay, delta_n,
        # perturbations (vT scalaire par cellule, vC vecteur par cellule)
        vT, vC,                    # (ncells,)  (ncells, nsp)
        # sorties
        omega,                     # (ncells, nsp)
        jvp_omega):                # (ncells, nsp)

    Tcell  = T[cell_idx]
    Ccell  = C[cell_idx]
    vTcell = vT[cell_idx]
    vCcell = vC[cell_idx]
    nreac  = A_arr.shape[0]

    for r in range(nreac):
        react_start = reactant_offset[r]
        react_nb    = reactant_number[r]
        prod_start  = prod_offset[r]
        prod_nb     = prod_number[r]

        # ── cinétique ────────────────────────────────────────────────────
        kf    = chemistry.arrhenius(A_arr[r], n_arr[r], Ea_arr[r], Tcell)
        dG0   = chemistry.delta_G0(react_start, react_nb, prod_start, prod_nb,
                                   react_idx, prod_idx, react_sto, prod_sto,
                                   coef_high, coef_low, Tmid, Tcell)
        Kp_v  = chemistry.Kp(dG0, Tcell)
        Kc_v  = chemistry.Kc(delta_n[r], Tcell, Kp_v)

        # ── dérivées scalaires ────────────────────────────────────────────
        dkf_dT_v = chemistry.d_arrhenius_dT(kf, n_arr[r], Ea_arr[r], Tcell)
        dKc_dT_v = chemistry.dKc_dT(react_start, react_nb, prod_start, prod_nb,
                                     react_idx, prod_idx, react_sto, prod_sto,
                                     coef_high, coef_low, Tmid,
                                     Tcell, Kc_v, delta_n[r])

        # ── RR et Jv_RR ─────────────────────────────────────────────────
        RR, Jv_RR = chemistry.elem_RR_jvp_full(
            react_start, react_nb, prod_start, prod_nb,
            react_idx, prod_idx, react_sto, prod_sto,
            Ccell, kf, dkf_dT_v, oneWay[r], Kc_v, dKc_dT_v,
            vTcell, vCcell)

        # ── accumulation ─────────────────────────────────────────────────
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