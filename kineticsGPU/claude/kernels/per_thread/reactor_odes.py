"""
kernels/shared/reactor_odes.py — Équations du réacteur adiabatique (scalaire).

Contient les formulations mathématiques du réacteur :
    - RHS adiabatique (dT/dt, dC/dt)
    - JVP du RHS adiabatique

Ces fonctions sont la version SCALAIRE (per-thread). Les versions parallèles
(per-block) sont dans per_block/reactor_odes_pb.py.

Fonctions exportées :
    _adiabatic_rhs(T, C, omega, n_sp, thL, thH, thM, dTCdt)
    _adiabatic_jvp(T, C, vT, vC, omega, Jv_om, n_sp, thL, thH, thM, Jv_TC)
"""

import math
from numba import cuda, float64

from kineticsGPU.claude.kernels.shared.thermo import _load_coef7, _h_RT, _s_R, _cp_R, _kf_arr

R_CONST = 8.314_462_618_153_24
EPS     = 1e-300


@cuda.jit(device=True)
def _adiabatic_rhs(T, C, omega, n_sp, thL, thH, thM, dTCdt):
    """
    Calcule le RHS du réacteur adiabatique à volume constant.

    dT/dt = -sum(u_s * omega_s) / (n_tot * cv_mix)
    dC_s/dt = omega_s

    Paramètres
    ----------
    T     : float64     — température (K)
    C     : array(n_sp) — concentrations (mol/m³)
    omega : array(n_sp) — taux de production (mol/m³/s)
    dTCdt : array(n_sp+1) — sortie [dT/dt, dC_0/dt, ..., dC_{n-1}/dt]
    """
    c7   = cuda.local.array(7, float64)
    ntot = float64(0.0)
    for s in range(n_sp):
        ntot += C[s]
    cv_mix = float64(0.0)
    su_om  = float64(0.0)
    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        cp_r = _cp_R(T, c7)
        cv_s = R_CONST * (cp_r - 1.0)
        hRT  = _h_RT(T, c7)
        u_s  = R_CONST * T * hRT - R_CONST * T
        Xs   = C[s] / ntot if ntot > EPS else float64(0.0)
        cv_mix += Xs * cv_s
        su_om  += u_s * omega[s]
    denom     = ntot * cv_mix
    dTCdt[0]  = -su_om / denom if abs(denom) > EPS else float64(0.0)
    for s in range(n_sp):
        dTCdt[s + 1] = omega[s]


@cuda.jit(device=True)
def _adiabatic_jvp(T, C, vT, vC, omega, Jv_om, n_sp, thL, thH, thM, Jv_TC):
    """
    Calcule la JVP (produit Jacobien-vecteur) du RHS adiabatique.

    Dérivée directionnelle de _adiabatic_rhs dans la direction (vT, vC).

    Paramètres
    ----------
    T, C           : état de base
    vT, vC         : direction de la perturbation
    omega          : taux de production au point de base (déjà calculé)
    Jv_om          : JVP de omega (déjà calculé par _eval_jvp)
    Jv_TC          : sortie — JVP du RHS complet [JvdT/dt, JvdC/dt]
    """
    c7    = cuda.local.array(7, float64)
    ntot  = float64(0.0)
    for s in range(n_sp):
        ntot += C[s]
    dntot = float64(0.0)
    for s in range(n_sp):
        dntot += vC[s]
    cv_mix = float64(0.0); dcv = float64(0.0)
    up     = float64(0.0); dup = float64(0.0)
    for s in range(n_sp):
        _load_coef7(T, s, thM, thL, thH, c7)
        T2 = T * T; T3 = T2 * T; T4 = T3 * T
        cpR   = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        cv_s  = R_CONST * (cpR - 1.0)
        dcvdT = R_CONST * (c7[1] + 2.0*c7[2]*T + 3.0*c7[3]*T2 + 4.0*c7[4]*T3)
        hRT   = (c7[0] + c7[1]*T/2.0 + c7[2]*T2/3.0
                 + c7[3]*T3/4.0 + c7[4]*T4/5.0 + c7[5]/T)
        u_s   = R_CONST * T * hRT - R_CONST * T
        Xs    = C[s] / ntot if ntot > EPS else float64(0.0)
        dXs   = ((vC[s] / ntot - C[s] * dntot / (ntot * ntot))
                 if ntot > EPS else float64(0.0))
        cv_mix += Xs * cv_s
        dcv    += dXs * cv_s + Xs * dcvdT * vT
        up     += u_s * omega[s]
        dup    += u_s * Jv_om[s] + cv_s * omega[s] * vT
    denom = ntot * cv_mix
    if abs(denom) > EPS:
        Jv_TC[0] = (-dup / denom
                    + up * dntot / (denom * ntot)
                    + up * dcv / (ntot * cv_mix * cv_mix))
    else:
        Jv_TC[0] = float64(0.0)
    for s in range(n_sp):
        Jv_TC[s + 1] = Jv_om[s]