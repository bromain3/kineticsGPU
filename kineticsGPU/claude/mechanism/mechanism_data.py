"""
MechanismData: converts parsed reaction system (from existing torch code)
into numpy arrays suitable for Numba CUDA kernels.

Design principle: all data is packed into contiguous arrays indexed with
offset arrays (CSR-like), so each thread can access its reaction data
without Python objects.

Memory layout (per reaction type):
  - Elementary:   react_idx, prod_idx, react_sto, prod_sto (flat, offset-indexed)
  - ThreeBody:    same + eff_matrix (n_reactions x n_species)
  - Falloff:      same + k0/kinf params + Troe params + eff_matrix

NASA7 thermo:
  - coef_low  (n_species, 7)
  - coef_high (n_species, 7)
  - Tmid      (n_species,)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


R_CONST  = 8.314_462_618_153_24   # J/(mol·K)
PREF     = 101325.0               # Pa


@dataclass
class ElementaryData:
    """Flat CSR-like storage for all elementary reactions."""
    n_reactions: int
    # Kinetics
    A:   np.ndarray   # (n_reactions,)
    n:   np.ndarray   # (n_reactions,)  temperature exponent
    Ea:  np.ndarray   # (n_reactions,)  activation energy J/mol
    oneWay: np.ndarray  # (n_reactions,) bool as float (0/1)
    delta_n: np.ndarray # (n_reactions,) Δn = sum(prod_sto) - sum(react_sto)
    # Stoichiometry - flat CSR
    react_idx:    np.ndarray   # (n_total_react,) species indices
    react_sto:    np.ndarray   # (n_total_react,) stoich coefficients
    react_offset: np.ndarray   # (n_reactions+1,) CSR row pointers
    prod_idx:     np.ndarray   # (n_total_prod,)
    prod_sto:     np.ndarray   # (n_total_prod,)
    prod_offset:  np.ndarray   # (n_reactions+1,)
    # Scatter indices (precomputed: reaction index for each flat entry)
    idx_r_scatter: np.ndarray  # (n_total_react,)
    idx_p_scatter: np.ndarray  # (n_total_prod,)


@dataclass
class ThreeBodyData:
    """Flat CSR-like storage for all three-body reactions."""
    n_reactions: int
    A:   np.ndarray
    n:   np.ndarray
    Ea:  np.ndarray
    oneWay: np.ndarray
    delta_n: np.ndarray
    defaultEff: np.ndarray    # (n_reactions,)
    react_idx:    np.ndarray
    react_sto:    np.ndarray
    react_offset: np.ndarray
    prod_idx:     np.ndarray
    prod_sto:     np.ndarray
    prod_offset:  np.ndarray
    idx_r_scatter: np.ndarray
    idx_p_scatter: np.ndarray
    eff_matrix: np.ndarray    # (n_reactions, n_species)  third-body efficiencies


@dataclass
class FalloffData:
    """Flat CSR-like storage for all falloff reactions (Lindemann + Troe)."""
    n_reactions: int
    # k0 Arrhenius
    A_k0:  np.ndarray   # (n_reactions,)
    b_k0:  np.ndarray
    Ea_k0: np.ndarray
    # kinf Arrhenius
    A_kinf:  np.ndarray
    b_kinf:  np.ndarray
    Ea_kinf: np.ndarray
    oneWay:  np.ndarray
    delta_n: np.ndarray
    # Troe parameters (0 for Lindemann)
    troe_A:  np.ndarray   # (n_reactions,)
    troe_T1: np.ndarray
    troe_T2: np.ndarray
    troe_T3: np.ndarray
    is_troe: np.ndarray   # (n_reactions,) bool as int8
    # Stoichiometry CSR
    react_idx:    np.ndarray
    react_sto:    np.ndarray
    react_offset: np.ndarray
    prod_idx:     np.ndarray
    prod_sto:     np.ndarray
    prod_offset:  np.ndarray
    idx_r_scatter: np.ndarray
    idx_p_scatter: np.ndarray
    eff_matrix: np.ndarray    # (n_reactions, n_species)


@dataclass
class ThermoData:
    """NASA7 polynomial thermo data."""
    n_species: int
    species_names: list
    coef_low:      np.ndarray   # (n_species, 7)
    coef_high:     np.ndarray   # (n_species, 7)
    Tmid:          np.ndarray   # (n_species,)
    molar_masses:  np.ndarray   # (n_species,)


@dataclass
class MechanismData:
    """
    Complete mechanism data ready for CUDA kernels.
    All arrays are C-contiguous float64 (or int32/int64 for indices).
    """
    n_species: int
    n_reactions_elem: int
    n_reactions_tb:   int
    n_reactions_fall: int
    thermo: ThermoData
    elementary: Optional[ElementaryData] = None
    threebody:  Optional[ThreeBodyData]  = None
    falloff:    Optional[FalloffData]    = None

    @property
    def n_reactions_total(self):
        return self.n_reactions_elem + self.n_reactions_tb + self.n_reactions_fall

    def to_cuda_arrays(self):
        """
        Returns a flat dict of device arrays.
        IMPORTANT: thermo_coef_low / high are flattened to 1-D (n_species*7,)
        so that Numba CUDA can use flat indexing sp*7+k inside device functions.
        """
        import numba.cuda as cuda
        d = {}

        # --- Thermo: FLATTEN to 1-D ---
        d['thermo_coef_low']     = cuda.to_device(
            np.ascontiguousarray(self.thermo.coef_low.reshape(-1)))    # (n_species*7,)
        d['thermo_coef_high']    = cuda.to_device(
            np.ascontiguousarray(self.thermo.coef_high.reshape(-1)))   # (n_species*7,)
        d['thermo_Tmid']         = cuda.to_device(
            np.ascontiguousarray(self.thermo.Tmid))                    # (n_species,)
        d['thermo_molar_masses'] = cuda.to_device(
            np.ascontiguousarray(self.thermo.molar_masses))

        # --- Elementary ---
        if self.elementary is not None:
            e = self.elementary
            for k in ['A','n','Ea','oneWay','delta_n',
                      'react_idx','react_sto','react_offset',
                      'prod_idx','prod_sto','prod_offset',
                      'idx_r_scatter','idx_p_scatter']:
                d[f'elem_{k}'] = cuda.to_device(np.ascontiguousarray(getattr(e, k)))

        # --- ThreeBody ---
        if self.threebody is not None:
            tb = self.threebody
            for k in ['A','n','Ea','oneWay','delta_n','defaultEff',
                      'react_idx','react_sto','react_offset',
                      'prod_idx','prod_sto','prod_offset',
                      'idx_r_scatter','idx_p_scatter']:
                d[f'tb_{k}'] = cuda.to_device(np.ascontiguousarray(getattr(tb, k)))
            # eff_matrix: flatten (n_reactions, n_species) → (n_reactions * n_species,)
            d['tb_eff_matrix'] = cuda.to_device(
                np.ascontiguousarray(tb.eff_matrix.reshape(-1)))

        # --- Falloff ---
        if self.falloff is not None:
            f = self.falloff
            for k in ['A_k0','b_k0','Ea_k0','A_kinf','b_kinf','Ea_kinf',
                      'oneWay','delta_n',
                      'troe_A','troe_T1','troe_T2','troe_T3','is_troe',
                      'react_idx','react_sto','react_offset',
                      'prod_idx','prod_sto','prod_offset',
                      'idx_r_scatter','idx_p_scatter']:
                d[f'fall_{k}'] = cuda.to_device(np.ascontiguousarray(getattr(f, k)))
            # eff_matrix: flatten
            d['fall_eff_matrix'] = cuda.to_device(
                np.ascontiguousarray(f.eff_matrix.reshape(-1)))

        return d


# ─────────────────────────────────────────────────────────────────────────────
# Builder: converts existing torch-based reaction objects → MechanismData
# ─────────────────────────────────────────────────────────────────────────────

def build_mechanism_data_from_torch(
    r_e_list,    # list of elementary reaction objects (torch)
    r_tb_list,   # list of three-body reaction objects (torch)
    r_f_list,    # list of falloff reaction objects (torch)
    vec_th,      # ThermoData_vectorized (torch)
    dtype=np.float64
) -> MechanismData:
    """
    Converts the existing torch reaction objects into MechanismData.
    This is called once at setup time, not in the hot loop.
    """

    n_species = len(vec_th.species)

    # ── Thermo ──
    thermo = ThermoData(
        n_species    = n_species,
        species_names= list(vec_th.species),
        coef_low     = vec_th.coef_low.cpu().numpy().astype(dtype),
        coef_high    = vec_th.coef_high.cpu().numpy().astype(dtype),
        Tmid         = vec_th.Tmid.cpu().numpy().astype(dtype),
        molar_masses = vec_th.molar_masses.cpu().numpy().astype(dtype),
    )

    # ── Elementary ──
    elem_data = None
    if r_e_list:
        elem_data = _build_elementary(r_e_list, n_species, dtype)

    # ── ThreeBody ──
    tb_data = None
    if r_tb_list:
        tb_data = _build_threebody(r_tb_list, n_species, dtype)

    # ── Falloff ──
    fall_data = None
    if r_f_list:
        fall_data = _build_falloff(r_f_list, n_species, dtype)

    return MechanismData(
        n_species          = n_species,
        n_reactions_elem   = len(r_e_list),
        n_reactions_tb     = len(r_tb_list),
        n_reactions_fall   = len(r_f_list),
        thermo             = thermo,
        elementary         = elem_data,
        threebody          = tb_data,
        falloff            = fall_data,
    )


def _build_csr(reactions, dtype):
    """Builds flat CSR arrays for reactants and products from a reaction list."""
    react_idx_list, react_sto_list = [], []
    prod_idx_list,  prod_sto_list  = [], []
    react_offset = [0]
    prod_offset  = [0]

    for r in reactions:
        react_idx_list.extend([int(x) for x in r.reactants_idx])
        react_sto_list.extend([float(x) for x in r.reactants_sto])
        prod_idx_list.extend([int(x) for x in r.products_idx])
        prod_sto_list.extend([float(x) for x in r.products_sto])
        react_offset.append(react_offset[-1] + len(r.reactants_idx))
        prod_offset.append(prod_offset[-1] + len(r.products_idx))

    react_offset_arr = np.array(react_offset, dtype=np.int32)
    prod_offset_arr  = np.array(prod_offset,  dtype=np.int32)

    n_r = len(reactions)
    # scatter indices: reaction index for each flat entry
    idx_r_scatter = np.repeat(np.arange(n_r, dtype=np.int32),
                              np.diff(react_offset_arr))
    idx_p_scatter = np.repeat(np.arange(n_r, dtype=np.int32),
                              np.diff(prod_offset_arr))

    return (
        np.array(react_idx_list, dtype=np.int32),
        np.array(react_sto_list, dtype=dtype),
        react_offset_arr,
        np.array(prod_idx_list, dtype=np.int32),
        np.array(prod_sto_list, dtype=dtype),
        prod_offset_arr,
        idx_r_scatter,
        idx_p_scatter,
    )


def _build_elementary(reactions, n_species, dtype):
    n = len(reactions)
    A      = np.array([r.A.item()  for r in reactions], dtype=dtype)
    n_exp  = np.array([r.n.item()  for r in reactions], dtype=dtype)
    Ea     = np.array([r.Ea.item() for r in reactions], dtype=dtype)
    oneWay = np.array([float(r.oneWay) for r in reactions], dtype=dtype)
    delta_n = np.array(
        [float(r.products_sto.sum() - r.reactants_sto.sum()) for r in reactions],
        dtype=dtype
    )

    ri, rs, ro, pi, ps, po, idr, idp = _build_csr(reactions, dtype)

    return ElementaryData(
        n_reactions=n, A=A, n=n_exp, Ea=Ea, oneWay=oneWay, delta_n=delta_n,
        react_idx=ri, react_sto=rs, react_offset=ro,
        prod_idx=pi,  prod_sto=ps,  prod_offset=po,
        idx_r_scatter=idr, idx_p_scatter=idp,
    )


def _build_threebody(reactions, n_species, dtype):
    n = len(reactions)
    A       = np.array([r.A.item()  for r in reactions], dtype=dtype)
    n_exp   = np.array([r.n.item()  for r in reactions], dtype=dtype)
    Ea      = np.array([r.Ea.item() for r in reactions], dtype=dtype)
    oneWay  = np.array([float(r.oneWay) for r in reactions], dtype=dtype)
    defaultEff = np.array([float(r.defaultEff) for r in reactions], dtype=dtype)
    delta_n = np.array(
        [float(r.products_sto.sum() - r.reactants_sto.sum()) for r in reactions],
        dtype=dtype
    )
    eff_matrix = np.stack([r.eff.cpu().numpy() for r in reactions], axis=0).astype(dtype)
    # shape: (n_reactions, n_species)

    ri, rs, ro, pi, ps, po, idr, idp = _build_csr(reactions, dtype)

    return ThreeBodyData(
        n_reactions=n, A=A, n=n_exp, Ea=Ea, oneWay=oneWay,
        delta_n=delta_n, defaultEff=defaultEff,
        react_idx=ri, react_sto=rs, react_offset=ro,
        prod_idx=pi,  prod_sto=ps,  prod_offset=po,
        idx_r_scatter=idr, idx_p_scatter=idp,
        eff_matrix=eff_matrix,
    )


def _build_falloff(reactions, n_species, dtype):
    n = len(reactions)
    A_k0   = np.array([float(r.A_k0)   for r in reactions], dtype=dtype)
    b_k0   = np.array([float(r.b_k0)   for r in reactions], dtype=dtype)
    Ea_k0  = np.array([float(r.Ea_k0)  for r in reactions], dtype=dtype)
    A_kinf = np.array([float(r.A_kinf) for r in reactions], dtype=dtype)
    b_kinf = np.array([float(r.b_kinf) for r in reactions], dtype=dtype)
    Ea_kinf= np.array([float(r.Ea_kinf)for r in reactions], dtype=dtype)
    oneWay = np.array([float(r.oneWay) for r in reactions], dtype=dtype)
    delta_n = np.array([float(r.delta_n) for r in reactions], dtype=dtype)
    is_troe = np.array(
        [1 if str(getattr(r,'falloffType','lindemann')).lower()=='troe' else 0
         for r in reactions], dtype=np.int32
    )
    def _troe(r, k):
        p = getattr(r, 'falloffParams', {}) or {}
        return float(p.get(k, 0.0))
    troe_A  = np.array([_troe(r,'A')  for r in reactions], dtype=dtype)
    troe_T1 = np.array([_troe(r,'T1') for r in reactions], dtype=dtype)
    troe_T2 = np.array([_troe(r,'T2') for r in reactions], dtype=dtype)
    troe_T3 = np.array([_troe(r,'T3') for r in reactions], dtype=dtype)

    eff_matrix = np.stack([r.eff.cpu().numpy() for r in reactions], axis=0).astype(dtype)

    ri, rs, ro, pi, ps, po, idr, idp = _build_csr(reactions, dtype)

    return FalloffData(
        n_reactions=n,
        A_k0=A_k0, b_k0=b_k0, Ea_k0=Ea_k0,
        A_kinf=A_kinf, b_kinf=b_kinf, Ea_kinf=Ea_kinf,
        oneWay=oneWay, delta_n=delta_n,
        troe_A=troe_A, troe_T1=troe_T1, troe_T2=troe_T2, troe_T3=troe_T3,
        is_troe=is_troe,
        react_idx=ri, react_sto=rs, react_offset=ro,
        prod_idx=pi,  prod_sto=ps,  prod_offset=po,
        idx_r_scatter=idr, idx_p_scatter=idp,
        eff_matrix=eff_matrix,
    )