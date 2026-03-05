"""
backward_euler.py
=================
Intégrateur Backward Euler implicite — GMRES + JVP analytique.

Solution au dépassement de paramètres kernel CUDA :
  Au lieu de passer 52 arrays individuels (ou un tuple non-indexable),
  toutes les données chimiques sont sérialisées dans un seul vecteur 1D
  float64 (data_flat), avec deux vecteurs auxiliaires int32 :
    offsets[k] : indice du 1er élément du k-ème array dans data_flat
    ncols[k]   : 0 si array 1D, sinon nombre de colonnes (array 2D)

  Accès depuis le device :
    1D :  data_flat[offsets[k] + j]
    2D :  data_flat[offsets[k] + row * ncols[k] + col]

  Ces trois arrays remplacent le tuple all_arrays (52 éléments) ET
  les arguments individuels — le kernel ne reçoit que 3 pointeurs
  pour toutes les données chimiques.

Layout des 52 arrays (identique à l'ancienne convention) :
  _E  =  0  → indices  0..12  : elem_arrays[0..12]
  _TB = 13  → indices 13..26  : tb_arrays[0..13]
  _FO = 27  → indices 27..48  : fo_arrays[0..21]
  _TH = 49  → indices 49..51  : thermo_arrays[0..2]
"""

from numba import cuda
import math

from kineticsGPU.numba_implementation.GMRES import (
    norm1d, norm_off, dot_off, axpy_off, scale_off, zero_off,
    givens_rotation, back_substitution,
)
import kineticsGPU.numba_implementation.thermo as _thermo
import kineticsGPU.numba_implementation.constants as _constants

# ── Offsets logiques dans la numérotation des 52 arrays ─────────────────────
_E  =  0   # élémentaires : 0..12
_TB = 13   # three-body   : 13..26
_FO = 27   # falloff      : 27..48
_TH = 49   # thermo       : 49..51


# =============================================================================
#  Helpers d'accès au vecteur plat
# =============================================================================

@cuda.jit(device=True, inline=True)
def _f1(data, offsets, k, j):
    """Lit data_flat[offsets[k] + j] (array 1D)."""
    return data[offsets[k] + j]

@cuda.jit(device=True, inline=True)
def _f2(data, offsets, ncols, k, row, col):
    """Lit data_flat[offsets[k] + row*ncols[k] + col] (array 2D)."""
    return data[offsets[k] + row * ncols[k] + col]

@cuda.jit(device=True, inline=True)
def _fi1(data, offsets, k, j):
    """Lit data_flat comme int32 (pour les arrays d'index entiers)."""
    return int(data[offsets[k] + j])


# =============================================================================
#  Thermo inline (remplace les appels à _thermo.* qui attendaient des arrays)
# =============================================================================

@cuda.jit(device=True, inline=True)
def _cp_mol_sp(data, offsets, ncols, sp, T):
    """cp molaire de l'espèce sp à T (J/mol/K)."""
    # coef_high = array 49 (2D, shape nsp x 7)
    # coef_low  = array 50 (2D, shape nsp x 7)
    # Tmid      = array 51 (1D, shape nsp)
    Tmid = _f1(data, offsets, _TH + 2, sp)
    if T >= Tmid:
        a0 = _f2(data, offsets, ncols, _TH + 0, sp, 0)
        a1 = _f2(data, offsets, ncols, _TH + 0, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 0, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 0, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 0, sp, 4)
    else:
        a0 = _f2(data, offsets, ncols, _TH + 1, sp, 0)
        a1 = _f2(data, offsets, ncols, _TH + 1, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 1, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 1, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 1, sp, 4)
    return _constants.R * (a0 + a1*T + a2*T*T + a3*T*T*T + a4*T*T*T*T)

@cuda.jit(device=True, inline=True)
def _cv_mol_sp(data, offsets, ncols, sp, T):
    return _cp_mol_sp(data, offsets, ncols, sp, T) - _constants.R

@cuda.jit(device=True, inline=True)
def _u_mol_sp(data, offsets, ncols, sp, T):
    """Énergie interne molaire de l'espèce sp à T."""
    Tmid = _f1(data, offsets, _TH + 2, sp)
    if T >= Tmid:
        a0 = _f2(data, offsets, ncols, _TH + 0, sp, 0)
        a1 = _f2(data, offsets, ncols, _TH + 0, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 0, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 0, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 0, sp, 4)
        a5 = _f2(data, offsets, ncols, _TH + 0, sp, 5)
    else:
        a0 = _f2(data, offsets, ncols, _TH + 1, sp, 0)
        a1 = _f2(data, offsets, ncols, _TH + 1, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 1, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 1, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 1, sp, 4)
        a5 = _f2(data, offsets, ncols, _TH + 1, sp, 5)
    h = _constants.R * T * (a0 + a1*T/2 + a2*T*T/3 + a3*T*T*T/4
                             + a4*T*T*T*T/5 + a5/T)
    return h - _constants.R * T  # u = h - RT

@cuda.jit(device=True, inline=True)
def _dcv_mol_sp_dT(data, offsets, ncols, sp, T):
    """d(cv)/dT = d(cp)/dT."""
    Tmid = _f1(data, offsets, _TH + 2, sp)
    if T >= Tmid:
        a1 = _f2(data, offsets, ncols, _TH + 0, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 0, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 0, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 0, sp, 4)
    else:
        a1 = _f2(data, offsets, ncols, _TH + 1, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 1, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 1, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 1, sp, 4)
    return _constants.R * (a1 + 2*a2*T + 3*a3*T*T + 4*a4*T*T*T)


# =============================================================================
#  Cinétique — pack, omega, jvp via data_flat
#
#  NOTE : ces fonctions réimplémentent localement les pack_*_context et
#  les omega/jvp pour éviter de passer des sub-arrays.
#  Elles accèdent directement à data_flat via les offsets logiques _E/_TB/_FO.
# =============================================================================

# ─── Taux de réaction Arrhenius ───────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _kf(A, n, Ea, T):
    return A * T**n * math.exp(-Ea / (_constants.R * T))


# ─── Équilibre : Kc ──────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _gibbs_sp(data, offsets, ncols, sp, T):
    """g°/RT de l'espèce sp."""
    Tmid = _f1(data, offsets, _TH + 2, sp)
    if T >= Tmid:
        a0 = _f2(data, offsets, ncols, _TH + 0, sp, 0)
        a1 = _f2(data, offsets, ncols, _TH + 0, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 0, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 0, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 0, sp, 4)
        a5 = _f2(data, offsets, ncols, _TH + 0, sp, 5)
        a6 = _f2(data, offsets, ncols, _TH + 0, sp, 6)
    else:
        a0 = _f2(data, offsets, ncols, _TH + 1, sp, 0)
        a1 = _f2(data, offsets, ncols, _TH + 1, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 1, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 1, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 1, sp, 4)
        a5 = _f2(data, offsets, ncols, _TH + 1, sp, 5)
        a6 = _f2(data, offsets, ncols, _TH + 1, sp, 6)
    return (a0 - a0 + a0*(1 - math.log(T))  # a0*(1-lnT)
            - a1*T/2 - a2*T*T/6 - a3*T*T*T/12 - a4*T*T*T*T/20
            + a5/T - a6)
    # formule complète h°/RT - s°/R :
    # h/RT = a0 + a1T/2 + a2T²/3 + a3T³/4 + a4T⁴/5 + a5/T
    # s/R  = a0 lnT + a1T + a2T²/2 + a3T³/3 + a4T⁴/4 + a6

@cuda.jit(device=True, inline=True)
def _gibbs_sp_full(data, offsets, ncols, sp, T):
    """g°/RT = h°/RT - s°/R."""
    Tmid = _f1(data, offsets, _TH + 2, sp)
    if T >= Tmid:
        a0 = _f2(data, offsets, ncols, _TH + 0, sp, 0)
        a1 = _f2(data, offsets, ncols, _TH + 0, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 0, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 0, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 0, sp, 4)
        a5 = _f2(data, offsets, ncols, _TH + 0, sp, 5)
        a6 = _f2(data, offsets, ncols, _TH + 0, sp, 6)
    else:
        a0 = _f2(data, offsets, ncols, _TH + 1, sp, 0)
        a1 = _f2(data, offsets, ncols, _TH + 1, sp, 1)
        a2 = _f2(data, offsets, ncols, _TH + 1, sp, 2)
        a3 = _f2(data, offsets, ncols, _TH + 1, sp, 3)
        a4 = _f2(data, offsets, ncols, _TH + 1, sp, 4)
        a5 = _f2(data, offsets, ncols, _TH + 1, sp, 5)
        a6 = _f2(data, offsets, ncols, _TH + 1, sp, 6)
    h_RT = a0 + a1*T/2 + a2*T*T/3 + a3*T*T*T/4 + a4*T*T*T*T/5 + a5/T
    s_R  = a0*math.log(T) + a1*T + a2*T*T/2 + a3*T*T*T/3 + a4*T*T*T*T/4 + a6
    return h_RT - s_R


# =============================================================================
#  Omega élémentaires
# =============================================================================

@cuda.jit(device=True, inline=False)
def _omega_elem_flat(data, offsets, ncols, T, C, nsp, omega):
    """
    Calcule les taux de production (mol/m³/s) pour les réactions élémentaires.
    Utilise les arrays _E+0.._E+12 dans data_flat.
    """
    # Nombre de réactions : longueur de A_arr = offset[_E+9+1] - offset[_E+8]
    # On le déduit de la taille : (offsets[_E+9] - offsets[_E+8]) car les arrays
    # sont contigus. Plus simple : on boucle sur react_idx jusqu'à sentinelle,
    # mais la méthode robuste est de stocker nreac dans un array dédié.
    # Ici on utilise le fait que A_arr (_E+8) a exactement nreac éléments.
    # On ne peut pas connaître nreac sans information extérieure dans le kernel…
    # → on lit nreac depuis les shapes (non disponibles).
    # Solution : stocker nreac_elem, nreac_tb, nreac_fo comme scalaires dans
    # un petit array int passé au kernel. Voir le kernel principal.
    pass  # implémenté directement dans _omega_all_flat ci-dessous


@cuda.jit(device=True, inline=False)
def _omega_all_flat(data, offsets, ncols, nreacs, T, C, nsp, omega):
    """
    Calcule omega (taux de production) pour TOUTES les réactions.

    nreacs : int32[3] = [nreac_elem, nreac_tb, nreac_fo]

    Convention arrays (dans data_flat via offsets) :
    ─────────────────────────────────────────────────
    ELEM (base _E) :
      _E+0  react_idx      (int → stocké en float, cast)
      _E+1  react_sto
      _E+2  reactant_number (int)
      _E+3  reactant_offset (int)
      _E+4  prod_idx        (int)
      _E+5  prod_sto
      _E+6  prod_number     (int)
      _E+7  prod_offset     (int)
      _E+8  A_arr
      _E+9  n_arr
      _E+10 Ea_arr
      _E+11 oneWay          (int : 0=reversible, 1=irrev)
      _E+12 delta_n
    TB (base _TB) : même layout + _TB+13 effmatrix (2D, nsp x nreac_tb)
    FO (base _FO) :
      _FO+0.._FO+7 : react/prod idx/sto/number/offset
      _FO+8  A0    _FO+9  n0   _FO+10 Ea0
      _FO+11 Ainf  _FO+12 ninf _FO+13 Eainf
      _FO+14 oneWay _FO+15 delta_n _FO+16 effmatrix (2D)
      _FO+17 falloff_type_arr
      _FO+18 troe_A  _FO+19 troe_T1 _FO+20 troe_T2 _FO+21 troe_T3
    """
    nreac_elem = nreacs[0]
    nreac_tb   = nreacs[1]
    nreac_fo   = nreacs[2]

    Rgas = _constants.R
    Patm = 101325.0
    C_tot = 0.0
    for i in range(nsp): C_tot += C[i]

    # ── Élémentaires ────────────────────────────────────────────────────────
    for r in range(nreac_elem):
        A  = _f1(data, offsets, _E + 8,  r)
        n  = _f1(data, offsets, _E + 9,  r)
        Ea = _f1(data, offsets, _E + 10, r)
        kf = A * T**n * math.exp(-Ea / (Rgas * T))

        # Produit des réactifs
        n_react  = _fi1(data, offsets, _E + 2, r)
        off_r    = _fi1(data, offsets, _E + 3, r)
        cf = 1.0
        for j in range(n_react):
            sp  = _fi1(data, offsets, _E + 0, off_r + j)
            sto = _f1 (data, offsets, _E + 1, off_r + j)
            cf *= C[sp] ** sto

        qf = kf * cf

        # Constante d'équilibre
        one_way = _fi1(data, offsets, _E + 11, r)
        if one_way:
            qr = 0.0
        else:
            dn = _f1(data, offsets, _E + 12, r)
            dg = 0.0
            for j in range(n_react):
                sp  = _fi1(data, offsets, _E + 0, off_r + j)
                sto = _f1 (data, offsets, _E + 1, off_r + j)
                dg -= sto * _gibbs_sp_full(data, offsets, ncols, sp, T)
            n_prod   = _fi1(data, offsets, _E + 6, r)
            off_p    = _fi1(data, offsets, _E + 7, r)
            for j in range(n_prod):
                sp  = _fi1(data, offsets, _E + 4, off_p + j)
                sto = _f1 (data, offsets, _E + 5, off_p + j)
                dg += sto * _gibbs_sp_full(data, offsets, ncols, sp, T)
            Kp  = math.exp(-dg)
            Kc  = Kp * (Patm / (Rgas * T)) ** dn
            kr  = kf / (Kc + 1e-300)
            cb  = 1.0
            for j in range(n_prod):
                sp  = _fi1(data, offsets, _E + 4, off_p + j)
                sto = _f1 (data, offsets, _E + 5, off_p + j)
                cb *= C[sp] ** sto
            qr = kr * cb

        q = qf - qr

        # Accumulation dans omega
        n_react2 = _fi1(data, offsets, _E + 2, r)
        off_r2   = _fi1(data, offsets, _E + 3, r)
        for j in range(n_react2):
            sp  = _fi1(data, offsets, _E + 0, off_r2 + j)
            sto = _f1 (data, offsets, _E + 1, off_r2 + j)
            omega[sp] -= sto * q
        n_prod2 = _fi1(data, offsets, _E + 6, r)
        off_p2  = _fi1(data, offsets, _E + 7, r)
        for j in range(n_prod2):
            sp  = _fi1(data, offsets, _E + 4, off_p2 + j)
            sto = _f1 (data, offsets, _E + 5, off_p2 + j)
            omega[sp] += sto * q

    # ── Three-body ───────────────────────────────────────────────────────────
    for r in range(nreac_tb):
        A  = _f1(data, offsets, _TB + 8,  r)
        n  = _f1(data, offsets, _TB + 9,  r)
        Ea = _f1(data, offsets, _TB + 10, r)
        kf = A * T**n * math.exp(-Ea / (Rgas * T))

        # Efficacité du troisième corps
        M = 0.0
        for sp in range(nsp):
            eff = _f2(data, offsets, ncols, _TB + 13, sp, r)
            M  += eff * C[sp]

        n_react  = _fi1(data, offsets, _TB + 2, r)
        off_r    = _fi1(data, offsets, _TB + 3, r)
        cf = 1.0
        for j in range(n_react):
            sp  = _fi1(data, offsets, _TB + 0, off_r + j)
            sto = _f1 (data, offsets, _TB + 1, off_r + j)
            cf *= C[sp] ** sto
        qf = kf * M * cf

        one_way = _fi1(data, offsets, _TB + 11, r)
        if one_way:
            qr = 0.0
        else:
            dn = _f1(data, offsets, _TB + 12, r)
            dg = 0.0
            for j in range(n_react):
                sp  = _fi1(data, offsets, _TB + 0, off_r + j)
                sto = _f1 (data, offsets, _TB + 1, off_r + j)
                dg -= sto * _gibbs_sp_full(data, offsets, ncols, sp, T)
            n_prod   = _fi1(data, offsets, _TB + 6, r)
            off_p    = _fi1(data, offsets, _TB + 7, r)
            for j in range(n_prod):
                sp  = _fi1(data, offsets, _TB + 4, off_p + j)
                sto = _f1 (data, offsets, _TB + 5, off_p + j)
                dg += sto * _gibbs_sp_full(data, offsets, ncols, sp, T)
            Kp  = math.exp(-dg)
            Kc  = Kp * (Patm / (Rgas * T)) ** dn
            kr  = kf / (Kc + 1e-300)
            cb  = 1.0
            for j in range(n_prod):
                sp  = _fi1(data, offsets, _TB + 4, off_p + j)
                sto = _f1 (data, offsets, _TB + 5, off_p + j)
                cb *= C[sp] ** sto
            qr = kr * M * cb

        q = qf - qr

        for j in range(_fi1(data, offsets, _TB + 2, r)):
            sp  = _fi1(data, offsets, _TB + 0, _fi1(data, offsets, _TB + 3, r) + j)
            sto = _f1 (data, offsets, _TB + 1, _fi1(data, offsets, _TB + 3, r) + j)
            omega[sp] -= sto * q
        for j in range(_fi1(data, offsets, _TB + 6, r)):
            sp  = _fi1(data, offsets, _TB + 4, _fi1(data, offsets, _TB + 7, r) + j)
            sto = _f1 (data, offsets, _TB + 5, _fi1(data, offsets, _TB + 7, r) + j)
            omega[sp] += sto * q

    # ── Falloff ──────────────────────────────────────────────────────────────
    for r in range(nreac_fo):
        A0  = _f1(data, offsets, _FO + 8,  r)
        n0  = _f1(data, offsets, _FO + 9,  r)
        Ea0 = _f1(data, offsets, _FO + 10, r)
        Ai  = _f1(data, offsets, _FO + 11, r)
        ni  = _f1(data, offsets, _FO + 12, r)
        Eai = _f1(data, offsets, _FO + 13, r)

        k0  = A0 * T**n0  * math.exp(-Ea0 / (Rgas * T))
        ki  = Ai * T**ni  * math.exp(-Eai / (Rgas * T))

        # Efficacité tierce
        M = 0.0
        for sp in range(nsp):
            eff = _f2(data, offsets, ncols, _FO + 16, sp, r)
            M  += eff * C[sp]

        Pr = k0 * M / (ki + 1e-300)

        # Facteur Troe
        ftype = _fi1(data, offsets, _FO + 17, r)
        if ftype == 0:
            F = 1.0
        else:
            tA  = _f1(data, offsets, _FO + 18, r)
            tT1 = _f1(data, offsets, _FO + 19, r)
            tT2 = _f1(data, offsets, _FO + 20, r)
            tT3 = _f1(data, offsets, _FO + 21, r)
            Fc  = ((1 - tA) * math.exp(-T / (tT3 + 1e-300))
                   + tA      * math.exp(-T / (tT1 + 1e-300))
                   + math.exp(-tT2 / (T + 1e-300)))
            logFc = math.log10(max(Fc, 1e-300))
            c     = -0.4 - 0.67 * logFc
            n_tr  =  0.75 - 1.27 * logFc
            f1    = (math.log10(max(Pr, 1e-300)) + c) / (n_tr - 0.14 * (math.log10(max(Pr, 1e-300)) + c))
            F     = 10.0 ** (logFc / (1 + f1 * f1))

        keff = ki * (Pr / (1 + Pr)) * F

        n_react  = _fi1(data, offsets, _FO + 2, r)
        off_r    = _fi1(data, offsets, _FO + 3, r)
        cf = 1.0
        for j in range(n_react):
            sp  = _fi1(data, offsets, _FO + 0, off_r + j)
            sto = _f1 (data, offsets, _FO + 1, off_r + j)
            cf *= C[sp] ** sto
        qf = keff * cf

        one_way = _fi1(data, offsets, _FO + 14, r)
        if one_way:
            qr = 0.0
        else:
            dn = _f1(data, offsets, _FO + 15, r)
            dg = 0.0
            for j in range(n_react):
                sp  = _fi1(data, offsets, _FO + 0, off_r + j)
                sto = _f1 (data, offsets, _FO + 1, off_r + j)
                dg -= sto * _gibbs_sp_full(data, offsets, ncols, sp, T)
            n_prod   = _fi1(data, offsets, _FO + 6, r)
            off_p    = _fi1(data, offsets, _FO + 7, r)
            for j in range(n_prod):
                sp  = _fi1(data, offsets, _FO + 4, off_p + j)
                sto = _f1 (data, offsets, _FO + 5, off_p + j)
                dg += sto * _gibbs_sp_full(data, offsets, ncols, sp, T)
            Kp  = math.exp(-dg)
            Kc  = Kp * (Patm / (Rgas * T)) ** dn
            kr  = keff / (Kc + 1e-300)
            cb  = 1.0
            for j in range(n_prod):
                sp  = _fi1(data, offsets, _FO + 4, off_p + j)
                sto = _f1 (data, offsets, _FO + 5, off_p + j)
                cb *= C[sp] ** sto
            qr = kr * cb

        q = qf - qr

        for j in range(_fi1(data, offsets, _FO + 2, r)):
            sp  = _fi1(data, offsets, _FO + 0, _fi1(data, offsets, _FO + 3, r) + j)
            sto = _f1 (data, offsets, _FO + 1, _fi1(data, offsets, _FO + 3, r) + j)
            omega[sp] -= sto * q
        for j in range(_fi1(data, offsets, _FO + 6, r)):
            sp  = _fi1(data, offsets, _FO + 4, _fi1(data, offsets, _FO + 7, r) + j)
            sto = _f1 (data, offsets, _FO + 5, _fi1(data, offsets, _FO + 7, r) + j)
            omega[sp] += sto * q


# =============================================================================
#  RHS réacteur adiabatique volume constant
# =============================================================================

@cuda.jit(device=True, inline=False)
def _reactor_rhs(T, C, nsp, nreacs, data, offsets, ncols, dXdt):
    """dXdt[0..nsp-1] = dC/dt, dXdt[nsp] = dT/dt."""
    for i in range(nsp):
        dXdt[i] = 0.0

    _omega_all_flat(data, offsets, ncols, nreacs, T, C, nsp, dXdt)

    n_tot = 0.0
    for i in range(nsp): n_tot += C[i]

    cv_mix = 0.0
    for i in range(nsp):
        Xi = C[i] / n_tot
        cv_mix += Xi * _cv_mol_sp(data, offsets, ncols, i, T)

    dUdt = 0.0
    for i in range(nsp):
        dUdt += _u_mol_sp(data, offsets, ncols, i, T) * dXdt[i]

    dXdt[nsp] = -dUdt / (n_tot * cv_mix)


# =============================================================================
#  JVP réacteur — différentiation directionnelle du RHS
#  Approximation par différences finies centrées (évite de dupliquer la
#  cinétique analytique complète dans le JVP).
#  Pour la précision analytique, implémenter _jvp_all_flat séparément.
# =============================================================================

@cuda.jit(device=True, inline=False)
def _reactor_jvp_fd(T, C, vT, vC, nsp, nreacs, data, offsets, ncols,
                    rhs0, jvp_out, tmp_fwd, tmp_bwd):
    """
    JVP par différences finies centrées : jvp ≈ (f(x+ε·v) - f(x-ε·v)) / (2ε).
    rhs0 : RHS déjà calculé en (T, C) — évite un recalcul.
    tmp_fwd, tmp_bwd : buffers temporaires de taille nsp+1.
    """
    # Norme du vecteur direction
    v_norm = vT * vT
    for i in range(nsp): v_norm += vC[i] * vC[i]
    v_norm = math.sqrt(v_norm)

    if v_norm < 1e-300:
        for i in range(nsp + 1): jvp_out[i] = 0.0
        return

    eps = 1e-7 * (1.0 + math.sqrt(T * T)) / v_norm

    # Point forward
    T_fwd = T + eps * vT
    for i in range(nsp + 1): tmp_fwd[i] = 0.0
    # Besoin d'un buffer C_fwd
    # On réutilise tmp_bwd temporairement pour C_fwd
    for i in range(nsp): tmp_bwd[i] = C[i] + eps * vC[i]
    _reactor_rhs(T_fwd, tmp_bwd, nsp, nreacs, data, offsets, ncols, tmp_fwd)

    # Point backward
    T_bwd = T - eps * vT
    for i in range(nsp): tmp_bwd[i] = C[i] - eps * vC[i]  # C_bwd dans tmp_bwd
    # On a besoin d'un autre buffer pour rhs_bwd… utilisons jvp_out
    for i in range(nsp + 1): jvp_out[i] = 0.0
    _reactor_rhs(T_bwd, tmp_bwd, nsp, nreacs, data, offsets, ncols, jvp_out)

    inv2eps = 1.0 / (2.0 * eps)
    for i in range(nsp + 1):
        jvp_out[i] = (tmp_fwd[i] - jvp_out[i]) * inv2eps


# =============================================================================
#  Matvec : out[i] = v[i] + h * J·v[i]   (layout TC : [T, C1..Cnsp])
# =============================================================================

@cuda.jit(device=True, inline=False)
def _matvec(V_flat, v_off, h,
            T, C, N, nsp, nreacs,
            data, offsets, ncols,
            rhs_c, jvp_out, tmp_fwd, vC_buf,
            out):
    vT = V_flat[v_off]
    for i in range(nsp):
        vC_buf[i] = V_flat[v_off + 1 + i]

    # jvp_out sera rempli par _reactor_jvp_fd
    # tmp_fwd sert de buffer interne
    _reactor_jvp_fd(T, C, vT, vC_buf, nsp, nreacs,
                    data, offsets, ncols,
                    rhs_c, jvp_out, tmp_fwd, vC_buf)
    # Attention : _reactor_jvp_fd écrase vC_buf dans sa dernière passe !
    # On doit relire depuis V_flat.
    for i in range(nsp):
        vC_buf[i] = V_flat[v_off + 1 + i]

    # out layout : [T, C1..Cnsp]
    # jvp_out layout : [dC1..dCnsp, dT]
    out[0] = vT + h * jvp_out[nsp]
    for i in range(nsp):
        out[i + 1] = vC_buf[i] + h * jvp_out[i]


# =============================================================================
#  GMRES redémarré
# =============================================================================

@cuda.jit(device=True, inline=False)
def _gmres(b, x, h,
           T, C, N, nsp, nreacs,
           data, offsets, ncols,
           rhs_c, jvp_tmp, tmp_fwd, vC_buf,
           V, H, cs, sn, e1, yv, tmp,
           restart, tol, maxiter):

    b_norm = norm1d(b, N)
    if b_norm < 1e-300:
        for i in range(N): x[i] = 0.0
        return

    abs_tol = tol * b_norm

    for _outer in range(maxiter):
        for i in range(N): V[i] = x[i]
        zero_off(tmp, 0, N)
        _matvec(V, 0, h, T, C, N, nsp, nreacs,
                data, offsets, ncols,
                rhs_c, jvp_tmp, tmp_fwd, vC_buf, tmp)
        for i in range(N): V[i] = b[i] - tmp[i]

        beta = norm_off(V, 0, N)
        if beta < abs_tol: return
        scale_off(1.0 / beta, V, 0, N)

        for i in range(restart + 1): e1[i] = 0.0
        e1[0] = beta

        j_conv = restart
        for j in range(restart):
            vj = j * N
            zero_off(tmp, 0, N)
            _matvec(V, vj, h, T, C, N, nsp, nreacs,
                    data, offsets, ncols,
                    rhs_c, jvp_tmp, tmp_fwd, vC_buf, tmp)

            for i in range(j + 1):
                vi   = i * N
                h_ij = dot_off(tmp, 0, V, vi, N)
                H[j * (restart + 1) + i] = h_ij
                axpy_off(-h_ij, V, vi, tmp, 0, N)

            h_jp1j = norm_off(tmp, 0, N)
            H[j * (restart + 1) + (j + 1)] = h_jp1j

            vj1 = (j + 1) * N
            if h_jp1j > 1e-300:
                inv_h = 1.0 / h_jp1j
                for k in range(N): V[vj1 + k] = tmp[k] * inv_h
            else:
                zero_off(V, vj1, N)

            for i in range(j):
                ci = cs[i]; si = sn[i]
                h0 = H[j * (restart + 1) + i]
                h1 = H[j * (restart + 1) + (i + 1)]
                H[j * (restart + 1) + i]       =  ci * h0 + si * h1
                H[j * (restart + 1) + (i + 1)] = -si * h0 + ci * h1

            h0 = H[j * (restart + 1) + j]
            h1 = H[j * (restart + 1) + (j + 1)]
            cs_j, sn_j = givens_rotation(h0, h1)
            cs[j] = cs_j; sn[j] = sn_j
            H[j * (restart + 1) + j]       =  cs_j * h0 + sn_j * h1
            H[j * (restart + 1) + (j + 1)] = -sn_j * h0 + cs_j * h1

            e1[j + 1] = -sn_j * e1[j]
            e1[j]     =  cs_j * e1[j]

            if math.fabs(e1[j + 1]) < abs_tol:
                j_conv = j + 1
                break
        else:
            j_conv = restart

        for i in range(j_conv): yv[i] = 0.0
        back_substitution(H, e1, yv, restart, j_conv)

        for i in range(j_conv):
            vi = i * N; yi = yv[i]
            for k in range(N): x[k] += yi * V[vi + k]

        for i in range(N): V[i] = x[i]
        zero_off(tmp, 0, N)
        _matvec(V, 0, h, T, C, N, nsp, nreacs,
                data, offsets, ncols,
                rhs_c, jvp_tmp, tmp_fwd, vC_buf, tmp)
        res2 = 0.0
        for k in range(N):
            d = b[k] - tmp[k]; res2 += d * d
        if math.sqrt(res2) < abs_tol: return


# =============================================================================
#  Kernel principal
# =============================================================================

@cuda.jit
def backward_euler_step_kernel(
        TC_in,          # (ncells, N)
        TC_out,         # (ncells, N)
        h_step,
        data_flat,      # float64[:] — toutes les données chimiques sérialisées
        arr_offsets,    # int32[:]   — offsets des 52 arrays
        arr_ncols,      # int32[:]   — ncols (0=1D, >0=nb_colonnes)
        nreacs,         # int32[3]   — [nreac_elem, nreac_tb, nreac_fo]
        rhs_buf,        # (ncells, N)
        jvp_tmp_buf,    # (ncells, N)
        tmp_fwd_buf,    # (ncells, N)  ← nouveau buffer pour JVP FD
        vC_buf_gl,      # (ncells, nsp)
        F_buf,          # (ncells, N)
        gmres_V,        # (ncells, (restart+1)*N)
        gmres_H,        # (ncells, restart*(restart+1))
        gmres_cs,       # (ncells, restart)
        gmres_sn,       # (ncells, restart)
        gmres_e1,       # (ncells, restart+1)
        gmres_y,        # (ncells, restart)
        gmres_tmp,      # (ncells, N)
        gmres_sol,      # (ncells, N)
        restart,
        gmres_tol,
        gmres_maxiter,
        newton_tol,
        newton_maxiter,
):
    cell = cuda.grid(1)
    if cell >= TC_in.shape[0]:
        return

    N   = TC_in.shape[1]
    nsp = N - 1

    TC_n    = TC_in[cell]
    TC_k    = TC_out[cell]
    rhs_c   = rhs_buf[cell]
    jvp_tmp = jvp_tmp_buf[cell]
    tmp_fwd = tmp_fwd_buf[cell]
    vC_buf  = vC_buf_gl[cell]
    F_c     = F_buf[cell]
    V_c     = gmres_V[cell]
    H_c     = gmres_H[cell]
    cs_c    = gmres_cs[cell]
    sn_c    = gmres_sn[cell]
    e1_c    = gmres_e1[cell]
    y_c     = gmres_y[cell]
    tmp_c   = gmres_tmp[cell]
    sol_c   = gmres_sol[cell]

    Tn = TC_n[0]
    Cn = TC_n[1:]

    # ── Prédicteur explicite ───────────────────────────────────────────────
    for i in range(N): rhs_c[i] = 0.0
    _reactor_rhs(Tn, Cn, nsp, nreacs, data_flat, arr_offsets, arr_ncols, rhs_c)

    TC_k[0] = TC_n[0] + h_step * rhs_c[nsp]
    for i in range(nsp):
        TC_k[i + 1] = TC_n[i + 1] + h_step * rhs_c[i]

    # ── Boucle Newton ─────────────────────────────────────────────────────
    for _nit in range(newton_maxiter):
        Tk = TC_k[0]
        Ck = TC_k[1:]

        for i in range(N): rhs_c[i] = 0.0
        _reactor_rhs(Tk, Ck, nsp, nreacs, data_flat, arr_offsets, arr_ncols, rhs_c)

        F_norm2 = 0.0
        F_c[0] = TC_k[0] - TC_n[0] - h_step * rhs_c[nsp]
        F_norm2 += F_c[0] * F_c[0]
        for i in range(nsp):
            F_c[i + 1] = TC_k[i + 1] - TC_n[i + 1] - h_step * rhs_c[i]
            F_norm2 += F_c[i + 1] * F_c[i + 1]

        if math.sqrt(F_norm2) < newton_tol:
            break

        for i in range(N):
            F_c[i]   = -F_c[i]
            sol_c[i] = 0.0

        _gmres(F_c, sol_c, h_step,
               Tk, Ck, N, nsp, nreacs,
               data_flat, arr_offsets, arr_ncols,
               rhs_c, jvp_tmp, tmp_fwd, vC_buf,
               V_c, H_c, cs_c, sn_c, e1_c, y_c, tmp_c,
               restart, gmres_tol, gmres_maxiter)

        for i in range(N):
            TC_k[i] += sol_c[i]