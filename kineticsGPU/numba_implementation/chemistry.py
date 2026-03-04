from numba import cuda
import math
import kineticsGPU.numba_implementation.constants as constants
import kineticsGPU.numba_implementation.thermo as thermo

############
### Base ###
############


@cuda.jit(device=True, inline=False)
def arrhenius(A, n, Ea, T):
    """kf = A * T^n * exp(-Ea/(R*T))"""
    return A * (T ** n) * math.exp(-Ea / (constants.R * T))

@cuda.jit(device=True, inline=False)
def delta_S0(react_start, react_nb,
             prod_start,  prod_nb,
             react_idx,   prod_idx,
             react_sto,   prod_sto,
             coefs_high, coefs_low, Tmid, T):
    """
    ΔS°(T) = Σ ν_prod * s°_prod  -  Σ ν_react * s°_react
    Utilise les mêmes polynômes NASA7 que delta_G0, garantissant
    la cohérence : deltaG0 = deltaH0 - T*deltaS0.
    """
    dS = 0.0
    for i in range(react_nb):
        idx = react_idx[i + react_start]
        sto = react_sto[i + react_start]
        dS -= sto * thermo.s0_nasa7(coefs_high[idx], coefs_low[idx], Tmid[idx], T)
    for i in range(prod_nb):
        idx = prod_idx[i + prod_start]
        sto = prod_sto[i + prod_start]
        dS += sto * thermo.s0_nasa7(coefs_high[idx], coefs_low[idx], Tmid[idx], T)
    return dS


@cuda.jit(device=True, inline=False)
def delta_G0(react_start, react_nb,
             prod_start, prod_nb,
             react_idx, prod_idx,
             react_sto, prod_sto,
             coefs_high, coefs_low, Tmid, T):
    dG0 = 0.0
    for i in range(react_nb):
        idx = react_idx[i + react_start]
        sto = react_sto[i + react_start]
        dG0 -= sto * thermo.g0_nasa7(coefs_high[idx], coefs_low[idx], Tmid[idx], T)
    for i in range(prod_nb):
        idx = prod_idx[i + prod_start]
        sto = prod_sto[i + prod_start]
        dG0 += sto * thermo.g0_nasa7(coefs_high[idx], coefs_low[idx], Tmid[idx], T)
    return dG0

@cuda.jit(device=True, inline=False)
def delta_H0(react_start, react_nb,
             prod_start,  prod_nb,
             react_idx,   prod_idx,
             react_sto,   prod_sto,
             coefs_high, coefs_low, Tmid, T):

    dH = 0.0
    for i in range(react_nb):
        idx = react_idx[i + react_start]
        sto = react_sto[i + react_start]
        dH -= sto * thermo.h0_nasa7(coefs_high[idx], coefs_low[idx], Tmid[idx], T)
    for i in range(prod_nb):
        idx = prod_idx[i + prod_start]
        sto = prod_sto[i + prod_start]
        dH += sto * thermo.h0_nasa7(coefs_high[idx], coefs_low[idx], Tmid[idx], T)
    return dH

@cuda.jit(device=True, inline=False)
def Kp(deltaG0, T):
    return math.exp(-deltaG0 / (constants.R * T))

@cuda.jit(device=True, inline=False)
def Kc(delta_n, T, Kp):
    return Kp * (constants.Pref / (constants.R * T)) ** delta_n

@cuda.jit(device=True, inline=False)
def elem_RR(react_start, react_nb,
       prod_start, prod_nb,
       react_idx, prod_idx,
       react_sto, prod_sto,
       Ccell, kf, oneWay, Kc):

    C_react = 1.0
    for i in range(react_nb):
        idx = react_idx[i + react_start]
        sto = react_sto[i + react_start]
        conc = Ccell[idx]
        C_react *= conc ** sto
    C_prod = 1.0
    for i in range(prod_nb):
        idx = prod_idx[i + prod_start]
        sto = prod_sto[i + prod_start]
        conc = Ccell[idx]
        C_prod *= conc ** sto
    return kf * (C_react - (1 - oneWay) / Kc * C_prod)

@cuda.jit(device=True, inline=False)
def dKc_dT(react_start, react_nb,
           prod_start,  prod_nb,
           react_idx,   prod_idx,
           react_sto,   prod_sto,
           coefs_high, coefs_low, Tmid,
           T, Kc_val, delta_n):
    """
    Dérivée analytique de Kc par rapport à T.
    dKc/dT = Kc * ( ΔH°/(R T²)  −  Δn / T )
    """
    dH = delta_H0(react_start, react_nb,
                         prod_start,  prod_nb,
                         react_idx,   prod_idx,
                         react_sto,   prod_sto,
                         coefs_high, coefs_low, Tmid, T)

    dlnKp_dT = dH / (constants.R * T * T)
    return Kc_val * (dlnKp_dT - delta_n / T)

@cuda.jit(device=True, inline=False)
def d_arrhenius_dT(kf, n, Ea, T):
    """dkf/dT = kf * (n/T + Ea/(R*T²))"""
    return kf * (n / T + Ea / (constants.R * T * T))

@cuda.jit(device=True, inline=False)
def elem_RR_jvp_full(react_start, react_nb,
                     prod_start,  prod_nb,
                     react_idx,   prod_idx,
                     react_sto,   prod_sto,
                     Ccell, kf, dkf_dT_val, oneWay, Kc, dKcdt_val,
                     vT, vCcell):
    """
    Calcule RR et Jv_RR = dRR/dT·vT + Σ_i dRR/dCi·vCi.

    Paramètres précalculés par l'appelant :
      dkf_dT_val : scalaire  dkf/dT
      dKcdt_val  : scalaire  dKc/dT
    """
    # ── C_react et C_prod ──────────────────────────────────────────────────
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

    G  = (1.0 - oneWay) / Kc * C_prod
    RR = kf * (C_react - G)

    # ── dRR/dT · vT ────────────────────────────────────────────────────────
    dG_dT  = (1.0 - oneWay) * C_prod * (- dKcdt_val / (Kc * Kc))
    Jv     = (dkf_dT_val * (C_react - G) + kf * (- dG_dT)) * vT

    # ── dRR/dCi · vCi  (réactifs) ─────────────────────────────────────────
    for i in range(react_nb):
        idx = react_idx[i + react_start]
        sto = react_sto[i + react_start]
        ci  = max(Ccell[idx], 1e-300)
        # dC_react/dCi = sto * C_react / Ci
        d_C_react_dCi = sto * C_react / ci
        Jv += kf * d_C_react_dCi * vCcell[idx]

    # ── dRR/dCi · vCi  (produits) ─────────────────────────────────────────
    for i in range(prod_nb):
        idx = prod_idx[i + prod_start]
        sto = prod_sto[i + prod_start]
        ci  = max(Ccell[idx], 1e-300)
        d_C_prod_dCi = sto * C_prod / ci
        dG_dCi       = (1.0 - oneWay) / Kc * d_C_prod_dCi
        Jv          -= kf * dG_dCi * vCcell[idx]

    return RR, Jv

##################
### three body ###
##################

@cuda.jit(device=True, inline=False)
def M(Ccell, effsReac):
    #TODO: improve for memory by aving a reduced efficiency list with only non default efficiencies
    M = 0.0
    for i in range(len(Ccell)):
        M += Ccell[i] * effsReac[i]
    return(M)

###############
### falloff ###
###############

@cuda.jit(device=True, inline=False)
def reduced_pressure(k0, kinf, M):
    return k0 * M / kinf

@cuda.jit(device=True, inline=False)
def F_lindemann():
    return 1.0

@cuda.jit(device=True, inline=False)
def F_troe(T, Pr, A, T1, T2, T3):
    """
    Facteur de broadening de Troe.
    Passer T2=0.0 si le paramètre est absent dans le yaml
    (le terme exp(-T2/T) est alors ignoré).
    """
    Fcent = 0.0
    if T1 > 0.0:
        Fcent += A * math.exp(-T / T1)
    if T3 > 0.0:
        Fcent += (1.0 - A) * math.exp(-T / T3)
    if T2 > 0.0:
        Fcent += math.exp(-T2 / T)

    logFcent = math.log10(Fcent)
    C_p = -0.4 - 0.67 * logFcent
    N_p =  0.75 - 1.27 * logFcent
    f1_denom = N_p - 0.14 * (math.log10(Pr) + C_p)
    f1 = (math.log10(Pr) + C_p) / f1_denom
    return 10.0 ** (logFcent / (1.0 + f1 * f1))

@cuda.jit(device=True, inline=False)
def kf_falloff(k0, kinf, M, F):
    Pr = reduced_pressure(k0, kinf, M)
    return kinf * (Pr / (1.0 + Pr)) * F

@cuda.jit(device=True, inline=False)
def _troe_Fcent(T, A, T1, T2, T3):
    """Fcent = A*exp(-T/T1) + (1-A)*exp(-T/T3) + exp(-T2/T)"""
    fc = 0.0
    if T1 > 0.0:
        fc += A * math.exp(-T / T1)
    if T3 > 0.0:
        fc += (1.0 - A) * math.exp(-T / T3)
    if T2 > 0.0:
        fc += math.exp(-T2 / T)
    return fc


@cuda.jit(device=True, inline=False)
def _troe_dFcent_dT(T, A, T1, T2, T3):
    """dFcent/dT analytique."""
    dfc = 0.0
    if T1 > 0.0:
        dfc += A * (-1.0 / T1) * math.exp(-T / T1)
    if T3 > 0.0:
        dfc += (1.0 - A) * (-1.0 / T3) * math.exp(-T / T3)
    if T2 > 0.0:
        dfc += math.exp(-T2 / T) * (T2 / (T * T))
    return dfc


@cuda.jit(device=True, inline=False)
def _troe_f1_and_exponent(logPr, logFcent):
    """
    Calcule f1 et l'exposant e = logFcent/(1+f1²).
    Retourne (f1, e, C_par, N_par, D) pour réutilisation dans les dérivées.
    """
    C_par = -0.4  - 0.67  * logFcent
    N_par =  0.75 - 1.27  * logFcent
    S     = logPr + C_par
    D     = N_par - 0.14  * S
    # éviter division par zéro
    if D == 0.0:
        D = 1e-300
    f1    = S / D
    e     = logFcent / (1.0 + f1 * f1)
    return f1, e, C_par, N_par, D, S


@cuda.jit(device=True, inline=False)
def F_troe_and_derivatives(T, Pr, A, T1, T2, T3):
    """
    Calcule F_troe, dF/dT et dF/dPr en une passe.
    Retourne (F, dF_dT, dF_dPr).
    """
    ln10 = math.log(10.0)

    Fcent  = _troe_Fcent(T, A, T1, T2, T3)
    dFc_dT = _troe_dFcent_dT(T, A, T1, T2, T3)

    if Fcent <= 0.0:
        Fcent = 1e-300
    logFcent     = math.log10(Fcent)
    dlogFc_dT    = dFc_dT / (Fcent * ln10)

    # guard Pr
    if Pr <= 0.0:
        Pr = 1e-300
    logPr        = math.log10(Pr)
    dlogPr_dPr   = 1.0 / (Pr * ln10)

    f1, e, C_par, N_par, D, S = _troe_f1_and_exponent(logPr, logFcent)

    f1_sq  = f1 * f1
    denom2 = 1.0 + f1_sq

    # ── dF/dT (Pr constant) ──────────────────────────────────────────────
    dC_dT  = -0.67 * dlogFc_dT
    dN_dT  = -1.27 * dlogFc_dT
    dS_dT  =  dlogFc_dT * 0.0 + dC_dT   # logPr independent of T → dS/dT = dC/dT
    dD_dT  = dN_dT - 0.14 * dS_dT
    df1_dT = (dS_dT * D - S * dD_dT) / (D * D)

    de_dlogFc = 1.0 / denom2
    de_df1    = -2.0 * f1 * logFcent / (denom2 * denom2)
    de_dT     = de_dlogFc * dlogFc_dT + de_df1 * df1_dT

    F        = math.pow(10.0, e)
    dF_dT    = F * ln10 * de_dT

    # ── dF/dPr (T constant) ──────────────────────────────────────────────
    dS_dPr   =  dlogPr_dPr
    dD_dPr   = -0.14 * dlogPr_dPr
    df1_dPr  = (dS_dPr * D - S * dD_dPr) / (D * D)
    de_dPr   = de_df1 * df1_dPr
    dF_dPr   = F * ln10 * de_dPr

    return F, dF_dT, dF_dPr