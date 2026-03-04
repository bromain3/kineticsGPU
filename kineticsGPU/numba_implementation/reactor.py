"""
kineticsGPU/numba_implementation/reactor.py

Device functions pour réacteur 0D adiabatique isochore.

État : TC = [T, C_0, ..., C_{nsp-1}]   longueur nsp+1

RHS :
  dC/dt = omega(T, C)
  dT/dt = -( Σ_k u_k * dCk/dt ) / ( n_tot * Cv_mix )

JVP du RHS (contraction avec perturbation v = [vT, vC]) :
  d(dC/dt)/dv = jvp_omega
  d(dT/dt)/dv = linearisation complète (voir commentaires inline)

Les fonctions ci-dessous sont des @cuda.jit(device=True) appelables
depuis un kernel global.  Elles écrivent leurs résultats dans des
tableaux locaux fournis par l'appelant (pas d'allocation dynamique).
"""

from numba import cuda
import kineticsGPU.numba_implementation.thermo_reactor as thermo_r

# Importation des per_thread de cinétique
import kineticsGPU.numba_implementation.elementary_reaction  as elem_mod
import kineticsGPU.numba_implementation.three_body_reaction  as tb_mod
import kineticsGPU.numba_implementation.falloff_reaction     as fo_mod


# ─────────────────────────────────────────────────────────────────────────────
#  RHS  —  dTC/dt
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def per_thread_rhs(
        # état
        T, C, cell_idx,          # (ncells,)  (ncells, nsp)
        # thermo NASA
        coef_low, coef_high, Tmid,   # (nsp, 7)  (nsp, 7)  (nsp,)
        nsp,
        # connectivité élémentaires
        e_ri, e_rs, e_rn, e_ro,
        e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        # connectivité three-body
        tb_ri, tb_rs, tb_rn, tb_ro,
        tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn,
        tb_eff,
        # connectivité falloff
        fo_ri, fo_rs, fo_rn, fo_ro,
        fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0,
        fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff,
        fo_ftype, fo_tA, fo_tT1, fo_tT2, fo_tT3,
        # sorties  (tableaux locaux de taille nsp+1, alloués par l'appelant)
        omega,       # (ncells, nsp)  — accumulateur espèces
        dTCdt_out):  # (nsp+1,)       — résultat final [dT/dt, dC/dt...]

    Tcell = T[cell_idx]
    Ccell = C[cell_idx]

    # ── remettre omega à zéro pour cette cellule ──────────────────────────
    for k in range(nsp):
        omega[cell_idx, k] = 0.0

    # ── accumulation des taux de production ──────────────────────────────
    elem_mod.per_thread_elem_react(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        e_ri, e_rs, e_rn, e_ro,
        e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        omega)

    tb_mod.per_thread_three_body_react(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        tb_ri, tb_rs, tb_rn, tb_ro,
        tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn,
        tb_eff,
        omega)

    fo_mod.per_thread_falloff_react(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        fo_ri, fo_rs, fo_rn, fo_ro,
        fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0,
        fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff,
        fo_ftype, fo_tA, fo_tT1, fo_tT2, fo_tT3,
        omega)

    # ── grandeurs thermo ──────────────────────────────────────────────────
    n_tot = 0.0
    for k in range(nsp):
        n_tot += Ccell[k]

    cv_mix = thermo_r.cv_mix_mol(Ccell, coef_high, coef_low, Tmid, Tcell, nsp)

    # dU/dt = Σ u_k * omega_k
    dUdt = 0.0
    for k in range(nsp):
        uk = thermo_r.u_species(coef_high[k], coef_low[k], Tmid[k], Tcell)
        dUdt += uk * omega[cell_idx, k]

    # dT/dt = -dU/dt / (n_tot * Cv_mix)
    dTdt = -dUdt / (n_tot * cv_mix)

    # ── écriture du résultat ──────────────────────────────────────────────
    dTCdt_out[0] = dTdt
    for k in range(nsp):
        dTCdt_out[k + 1] = omega[cell_idx, k]


# ─────────────────────────────────────────────────────────────────────────────
#  JVP du RHS  —  d(dTC/dt)/d(TC) · v
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def per_thread_jvp_rhs(
        # état
        T, C, cell_idx,
        # perturbation
        vT, vC,                  # (ncells,)  (ncells, nsp)
        # thermo
        coef_low, coef_high, Tmid,
        nsp,
        # connectivité — mêmes args que per_thread_rhs
        e_ri, e_rs, e_rn, e_ro,
        e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        tb_ri, tb_rs, tb_rn, tb_ro,
        tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn,
        tb_eff,
        fo_ri, fo_rs, fo_rn, fo_ro,
        fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0,
        fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff,
        fo_ftype, fo_tA, fo_tT1, fo_tT2, fo_tT3,
        # buffers locaux (ncells, nsp) — alloués par le kernel appelant
        omega,       # dC/dt (forward)
        jvp_omega,   # d(dC/dt)/dv (JVP cinétique)
        # sortie
        dTCdt_out,   # (nsp+1,)  forward [dT/dt, dC/dt...]
        jvp_out):    # (nsp+1,)  JVP     [d(dT/dt)/dv, d(dC/dt)/dv...]

    Tcell  = T[cell_idx]
    Ccell  = C[cell_idx]
    vTcell = vT[cell_idx]
    vCcell = vC[cell_idx]

    # ── remise à zéro ─────────────────────────────────────────────────────
    for k in range(nsp):
        omega    [cell_idx, k] = 0.0
        jvp_omega[cell_idx, k] = 0.0

    # ── cinétique : forward + JVP ─────────────────────────────────────────
    elem_mod.per_thread_elem_react_jvp(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        e_ri, e_rs, e_rn, e_ro,
        e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        vT, vC, omega, jvp_omega)

    tb_mod.per_thread_three_body_react_jvp(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        tb_ri, tb_rs, tb_rn, tb_ro,
        tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn,
        tb_eff,
        vT, vC, omega, jvp_omega)

    fo_mod.per_thread_falloff_react_jvp(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        fo_ri, fo_rs, fo_rn, fo_ro,
        fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0,
        fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff,
        fo_ftype, fo_tA, fo_tT1, fo_tT2, fo_tT3,
        vT, vC, omega, jvp_omega)

    # ── grandeurs thermo ──────────────────────────────────────────────────
    n_tot = 0.0
    for k in range(nsp):
        n_tot += Ccell[k]

    dn_tot = 0.0
    for k in range(nsp):
        dn_tot += vCcell[k]

    cv_mix = thermo_r.cv_mix_mol(Ccell, coef_high, coef_low, Tmid, Tcell, nsp)

    # ── forward dT/dt ─────────────────────────────────────────────────────
    dUdt = 0.0
    for k in range(nsp):
        uk = thermo_r.u_species(coef_high[k], coef_low[k], Tmid[k], Tcell)
        dUdt += uk * omega[cell_idx, k]

    dTdt = -dUdt / (n_tot * cv_mix)

    # ── JVP de dT/dt ──────────────────────────────────────────────────────
    #
    # dT' = d/dv [ -dU/dt / (n_tot * Cv_mix) ]
    #
    # Avec :
    #   U_prime    = Σ u_k * omega_k
    #   dU_prime   = Σ u_k * jvp_omega_k   +  Σ omega_k * cv_k * vT
    #   dcv_mix/dv = Σ_k (dcv_k/dT * Xk) * vT  +  Σ_k cv_k * dXk/dv
    #   dXk/dv     = vCk/n_tot - Ck*dn_tot/n_tot²
    #
    # dT' = -dU'/( n*Cv ) + U'*dn/( n²*Cv ) + U'*dCv/( n*Cv² )

    # Σ u_k * jvp_omega_k
    sum_u_jvp = 0.0
    for k in range(nsp):
        uk = thermo_r.u_species(coef_high[k], coef_low[k], Tmid[k], Tcell)
        sum_u_jvp += uk * jvp_omega[cell_idx, k]

    # Σ omega_k * cv_k * vT   (terme en vT de dU_prime)
    sum_om_cv = 0.0
    for k in range(nsp):
        cvk = thermo_r.cv_species(coef_high[k], coef_low[k], Tmid[k], Tcell)
        sum_om_cv += omega[cell_idx, k] * cvk

    dU_prime = sum_u_jvp + sum_om_cv * vTcell

    # dcv_mix/dv :
    #   terme en vT : Σ X_k * dcv_k/dT * vT
    #   terme en vC : Σ cv_k * dX_k/dv  avec dX_k = (vCk - Xk*dn_tot)/n_tot
    dcv_dT_term = 0.0
    for k in range(nsp):
        Xk = Ccell[k] / n_tot
        dcvk_dT = thermo_r.dcv_dT_species(coef_high[k], coef_low[k], Tmid[k], Tcell)
        dcv_dT_term += Xk * dcvk_dT

    dcv_dC_term = 0.0
    for k in range(nsp):
        cvk  = thermo_r.cv_species(coef_high[k], coef_low[k], Tmid[k], Tcell)
        dXk  = (vCcell[k] - Ccell[k] / n_tot * dn_tot) / n_tot
        dcv_dC_term += cvk * dXk

    dcv_mix_v = dcv_dT_term * vTcell + dcv_dC_term

    # Assemblage de dT'
    dT_prime = (- dU_prime / (n_tot * cv_mix)
                + dUdt * dn_tot / (n_tot * n_tot * cv_mix)
                + dUdt * dcv_mix_v / (n_tot * cv_mix * cv_mix))

    # ── écriture des sorties ──────────────────────────────────────────────
    dTCdt_out[0] = dTdt
    jvp_out  [0] = dT_prime
    for k in range(nsp):
        dTCdt_out[k + 1] = omega    [cell_idx, k]
        jvp_out  [k + 1] = jvp_omega[cell_idx, k]