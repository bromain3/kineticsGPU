"""
kernels/per_block/reactor_odes_pb.py — Équations du réacteur adiabatique (per-block).

Version parallèle des équations du réacteur :
    - RHS adiabatique  : _adiabatic_rhs_pb
    - JVP adiabatique  : _adiabatic_jvp_pb

Différences vs shared/reactor_odes.py (version scalaire) :
    - Boucles sur les espèces parallélisées avec stride tid/n_threads
    - Réductions (cv_mix, su_om, ntot, etc.) via _block_reduce_sum
    - cuda.syncthreads() après chaque écriture partagée
    - Thread 0 écrit dTCdt[0] / Jv_TC[0]
"""

import math
from numba import cuda, float64

from ..shared.thermo import _load_coef7, _h_RT, _s_R, _cp_R, R_CONST, EPS
from .reductions import _block_reduce_sum


@cuda.jit(device=True)
def _adiabatic_rhs_pb(
    tid, n_threads,
    T, C, omega, n_sp, thL, thH, thM,
    dTCdt,    # tableau partagé de sortie, longueur n_sp+1
    scratch,  # tableau partagé pour réductions, longueur >= n_threads
):
    """
    RHS adiabatique parallèle (per-block).

    dT/dt = -sum(u_s * omega_s) / (n_tot * cv_mix)
    dC_s/dt = omega_s

    scratch et dTCdt doivent être en mémoire partagée.
    """
    c7 = cuda.local.array(7, float64)

    # ntot = sum(C[s]) — réduction parallèle
    acc = float64(0.0)
    i   = tid
    while i < n_sp:
        acc += C[i]
        i   += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)
    cuda.syncthreads()
    ntot = scratch[0]

    # cv_mix et su_om — chaque thread accumule sa portion
    cv_loc = float64(0.0)
    su_loc = float64(0.0)
    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        cp_r = _cp_R(T, c7)
        cv_s = R_CONST * (cp_r - 1.0)
        hRT  = _h_RT(T, c7)
        u_s  = R_CONST * T * hRT - R_CONST * T
        Xs   = C[i] / ntot if ntot > EPS else float64(0.0)
        cv_loc += Xs * cv_s
        su_loc += u_s * omega[i]
        i += n_threads

    _block_reduce_sum(cv_loc, scratch, tid, n_threads)
    cuda.syncthreads()
    cv_mix = scratch[0]

    _block_reduce_sum(su_loc, scratch, tid, n_threads)
    cuda.syncthreads()
    su_om = scratch[0]

    if tid == 0:
        denom    = ntot * cv_mix
        dTCdt[0] = -su_om / denom if abs(denom) > EPS else float64(0.0)
    cuda.syncthreads()

    i = tid
    while i < n_sp:
        dTCdt[i + 1] = omega[i]
        i += n_threads
    cuda.syncthreads()


@cuda.jit(device=True)
def _adiabatic_jvp_pb(
    tid, n_threads,
    T, C, vT, vC, omega, Jv_om, n_sp, thL, thH, thM,
    Jv_TC,    # tableau partagé de sortie, longueur n_sp+1
    scratch,  # tableau partagé pour réductions, longueur >= n_threads
):
    """
    JVP du RHS adiabatique (per-block).

    Dérivée directionnelle de _adiabatic_rhs_pb dans la direction (vT, vC).
    Nécessite omega et Jv_om déjà calculés au point de base.
    """
    c7 = cuda.local.array(7, float64)

    # ntot, dntot — réductions parallèles
    acc_n  = float64(0.0); acc_dn = float64(0.0)
    i = tid
    while i < n_sp:
        acc_n  += C[i]
        acc_dn += vC[i]
        i += n_threads
    _block_reduce_sum(acc_n, scratch, tid, n_threads)
    cuda.syncthreads()
    ntot = scratch[0]
    _block_reduce_sum(acc_dn, scratch, tid, n_threads)
    cuda.syncthreads()
    dntot = scratch[0]

    # Accumulation parallèle de cv_mix, dcv, up, dup
    lcv  = float64(0.0); ldcv = float64(0.0)
    lup  = float64(0.0); ldup = float64(0.0)
    i = tid
    while i < n_sp:
        _load_coef7(T, i, thM, thL, thH, c7)
        T2 = T * T; T3 = T2 * T; T4 = T3 * T
        cpR   = c7[0] + c7[1]*T + c7[2]*T2 + c7[3]*T3 + c7[4]*T4
        cv_s  = R_CONST * (cpR - 1.0)
        dcvdT = R_CONST * (c7[1] + 2.0*c7[2]*T + 3.0*c7[3]*T2 + 4.0*c7[4]*T3)
        hRT   = (c7[0] + c7[1]*T/2.0 + c7[2]*T2/3.0
                 + c7[3]*T3/4.0 + c7[4]*T4/5.0 + c7[5]/T)
        u_s   = R_CONST * T * hRT - R_CONST * T
        Xs    = C[i] / ntot if ntot > EPS else float64(0.0)
        dXs   = ((vC[i] / ntot - C[i] * dntot / (ntot * ntot))
                 if ntot > EPS else float64(0.0))
        lcv  += Xs  * cv_s
        ldcv += dXs * cv_s + Xs * dcvdT * vT
        lup  += u_s * omega[i]
        ldup += u_s * Jv_om[i] + cv_s * omega[i] * vT
        i += n_threads

    _block_reduce_sum(lcv, scratch, tid, n_threads)
    cuda.syncthreads()
    cv_mix = scratch[0]

    _block_reduce_sum(ldcv, scratch, tid, n_threads)
    cuda.syncthreads()
    dcv = scratch[0]

    _block_reduce_sum(lup, scratch, tid, n_threads)
    cuda.syncthreads()
    up = scratch[0]

    _block_reduce_sum(ldup, scratch, tid, n_threads)
    cuda.syncthreads()
    dup = scratch[0]

    if tid == 0:
        denom = ntot * cv_mix
        if abs(denom) > EPS:
            Jv_TC[0] = (-dup / denom
                        + up * dntot / (denom * ntot)
                        + up * dcv / (ntot * cv_mix * cv_mix))
        else:
            Jv_TC[0] = float64(0.0)
    cuda.syncthreads()

    i = tid
    while i < n_sp:
        Jv_TC[i + 1] = Jv_om[i]
        i += n_threads
    cuda.syncthreads()