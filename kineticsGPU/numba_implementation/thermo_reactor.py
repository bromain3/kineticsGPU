"""
Ajouts à kineticsGPU/numba_implementation/thermo.py
Fonctions thermodynamiques pour le réacteur 0D adiabatique.

Convention NASA-7 :
  Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
  H/RT = a1 + a2/2*T + a3/3*T^2 + a4/4*T^3 + a5/5*T^4 + a6/T
  U/RT = H/RT - 1   (gaz parfait: U = H - RT)
  Cv/R = Cp/R - 1

Toutes les fonctions retournent des grandeurs molaires [J/mol] ou [J/mol/K].
"""

from numba import cuda
import kineticsGPU.numba_implementation.constants as constants


# ─────────────────────────────────────────────────────────────────────────────
#  Sélection du jeu de coefficients (helper interne)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def _select_coefs_7(coefs_high, coefs_low, Tmid, T):
    if T < Tmid:
        return coefs_low
    else:
        return coefs_high


# ─────────────────────────────────────────────────────────────────────────────
#  Cp molaire  [J/mol/K]
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def cp_species(coefs_high, coefs_low, Tmid, T):
    """Cp°(T) [J/mol/K] pour une espèce via NASA-7."""
    a = _select_coefs_7(coefs_high, coefs_low, Tmid, T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    cp_R = (a[0]
            + a[1] * T
            + a[2] * T2
            + a[3] * T3
            + a[4] * T4)
    return constants.R * cp_R


# ─────────────────────────────────────────────────────────────────────────────
#  Cv molaire  [J/mol/K]   Cv = Cp - R  (gaz parfait)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def cv_species(coefs_high, coefs_low, Tmid, T):
    """Cv°(T) [J/mol/K] pour une espèce (gaz parfait)."""
    return cp_species(coefs_high, coefs_low, Tmid, T) - constants.R


# ─────────────────────────────────────────────────────────────────────────────
#  dCv/dT  [J/mol/K²]
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def dcv_dT_species(coefs_high, coefs_low, Tmid, T):
    """
    d(Cv°)/dT [J/mol/K²] = d(Cp°)/dT (la constante R disparaît).
    dCp/R / dT = a2 + 2*a3*T + 3*a4*T^2 + 4*a5*T^3
    """
    a = _select_coefs_7(coefs_high, coefs_low, Tmid, T)
    T2 = T * T
    T3 = T2 * T
    dcpR_dT = (a[1]
               + 2.0 * a[2] * T
               + 3.0 * a[3] * T * T
               + 4.0 * a[4] * T * T * T)
    return constants.R * dcpR_dT


# ─────────────────────────────────────────────────────────────────────────────
#  Énergie interne molaire  U°(T)  [J/mol]
#  U = H - RT  =>  U = R*(a1*T + a2/2*T^2 + a3/3*T^3 + a4/4*T^4
#                             + a5/5*T^5 + a6)  - R*T
#               = R*(( a1-1)*T + a2/2*T^2 + a3/3*T^3 + a4/4*T^4
#                              + a5/5*T^5 + a6)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def u_species(coefs_high, coefs_low, Tmid, T):
    """U°(T) [J/mol] pour une espèce (gaz parfait)."""
    a = _select_coefs_7(coefs_high, coefs_low, Tmid, T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    return constants.R * (
        (a[0] - 1.0) * T
        + a[1] * 0.5   * T2
        + a[2] / 3.0   * T3
        + a[3] * 0.25  * T4
        + a[4] * 0.2   * T5
        + a[5]
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Cv du mélange  [J/mol/K]
#  cv_mix = Σ_k X_k * Cv_k   (X = fractions molaires)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def cv_mix_mol(Ccell, coefs_high, coefs_low, Tmid, T, nsp):
    """
    Cv molaire du mélange [J/mol/K].
    Ccell : tableau de concentrations (nsp,) [mol/m³]
    Retourne cv_mix en J/mol/K (moles du mélange).
    """
    n_tot = 0.0
    for k in range(nsp):
        n_tot += Ccell[k]

    cv_mix = 0.0
    for k in range(nsp):
        Xk = Ccell[k] / n_tot
        cv_mix += Xk * cv_species(coefs_high[k], coefs_low[k], Tmid[k], T)
    return cv_mix


# ─────────────────────────────────────────────────────────────────────────────
#  dCv_mix/dT  [J/mol/K²]
#  d(cv_mix)/dT = Σ_k X_k * dCv_k/dT   (X constant par rapport à T)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def dcv_mix_dT_mol(Ccell, coefs_high, coefs_low, Tmid, T, nsp):
    """d(Cv_mix)/dT [J/mol/K²]."""
    n_tot = 0.0
    for k in range(nsp):
        n_tot += Ccell[k]

    dcv = 0.0
    for k in range(nsp):
        Xk = Ccell[k] / n_tot
        dcv += Xk * dcv_dT_species(coefs_high[k], coefs_low[k], Tmid[k], T)
    return dcv


# ─────────────────────────────────────────────────────────────────────────────
#  dCv_mix/dCi  [J/mol/K / (mol/m³)]
#  d(cv_mix)/dCi = (cv_i - cv_mix) / n_tot
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def dcv_mix_dCi(cv_i, cv_mix, n_tot):
    """Dérivée scalaire de cv_mix par rapport à Ci."""
    return (cv_i - cv_mix) / n_tot