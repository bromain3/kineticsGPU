"""
kernels/shared/thermo.py — Device helpers thermodynamiques partagés.

Fonctions pures (device=True) sans dépendance aux données réacteur.
Utilisées par per_thread et per_block indifféremment.

Fonctions exportées :
    _load_coef7(T, sp, Tmid, coef_low, coef_high, out7)
    _h_RT(T, c7)         → enthalpie réduite H/(RT)
    _s_R(T, c7)          → entropie réduite S/R
    _cp_R(T, c7)         → cp réduit Cp/R
    _kf_arr(T, A, n, Ea) → constante de vitesse Arrhenius
    _log_conc_prod(C, i0, i1, sp_idx, sto) → sum(sto * log(C))
"""

import math
from numba import cuda, float64

R_CONST  = 8.314_462_618_153_24
PREF     = 101325.0
EPS      = 1e-300
LN10     = math.log(10.0)
INV_LN10 = 1.0 / LN10


@cuda.jit(device=True, inline=True)
def _load_coef7(T, sp, Tmid, coef_low, coef_high, out7):
    """Charge les 7 coefficients NASA pour l'espèce sp à la température T."""
    base = sp * 7
    arr  = coef_low if T <= Tmid[sp] else coef_high
    for k in range(7):
        out7[k] = arr[base + k]


@cuda.jit(device=True, inline=True)
def _h_RT(T, c7):
    """Enthalpie réduite H/(RT) à partir des coefficients NASA."""
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    return (c7[0] + c7[1] * T / 2.0 + c7[2] * T2 / 3.0
            + c7[3] * T3 / 4.0 + c7[4] * T4 / 5.0 + c7[5] / T)


@cuda.jit(device=True, inline=True)
def _s_R(T, c7):
    """Entropie réduite S/R à partir des coefficients NASA."""
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    return (c7[0] * math.log(T) + c7[1] * T + c7[2] * T2 / 2.0
            + c7[3] * T3 / 3.0 + c7[4] * T4 / 4.0 + c7[6])


@cuda.jit(device=True, inline=True)
def _cp_R(T, c7):
    """Capacité calorifique réduite Cp/R à partir des coefficients NASA."""
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    return c7[0] + c7[1] * T + c7[2] * T2 + c7[3] * T3 + c7[4] * T4


@cuda.jit(device=True, inline=True)
def _kf_arr(T, A, n, Ea):
    """Constante de vitesse Arrhenius modifiée : k = A * T^n * exp(-Ea/(R*T))."""
    return A * (T ** n) * math.exp(-Ea / (R_CONST * T))


@cuda.jit(device=True, inline=True)
def _log_conc_prod(C, i0, i1, sp_idx, sto):
    """
    Calcule sum_{k=i0}^{i1-1} sto[k] * log(C[sp_idx[k]]).
    Utilisé pour le produit de concentrations dans les taux de réaction.
    C[i] clampé à EPS pour éviter log(0).
    """
    s = float64(0.0)
    for k in range(i0, i1):
        c  = C[sp_idx[k]]
        s += sto[k] * math.log(c if c > EPS else EPS)
    return s