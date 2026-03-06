"""
mechanism_data.py — MechanismData for production CUDA kinetics integrator.

Stores all reaction/thermo data as flat numpy arrays ready for CUDA.
All floating-point arrays are float64 (C-contiguous).
Thermo coefficients are stored flat: shape (n_species*7,) for GPU indexing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


R_CONST = 8.314_462_618_153_24
PREF    = 101325.0


@dataclass
class ThermoData:
    n_species:     int
    species_names: list
    coef_low:      np.ndarray   # (n_species*7,)  — already flattened
    coef_high:     np.ndarray   # (n_species*7,)
    Tmid:          np.ndarray   # (n_species,)
    molar_masses:  np.ndarray   # (n_species,)


@dataclass
class ElementaryData:
    n_reactions:   int
    A:             np.ndarray   # (n,)
    n:             np.ndarray
    Ea:            np.ndarray
    oneWay:        np.ndarray
    delta_n:       np.ndarray
    react_idx:     np.ndarray   # int32, flat CSR
    react_sto:     np.ndarray
    react_offset:  np.ndarray   # int32 (n+1,)
    prod_idx:      np.ndarray
    prod_sto:      np.ndarray
    prod_offset:   np.ndarray


@dataclass
class ThreeBodyData:
    n_reactions:   int
    A:             np.ndarray
    n:             np.ndarray
    Ea:            np.ndarray
    oneWay:        np.ndarray
    delta_n:       np.ndarray
    react_idx:     np.ndarray
    react_sto:     np.ndarray
    react_offset:  np.ndarray
    prod_idx:      np.ndarray
    prod_sto:      np.ndarray
    prod_offset:   np.ndarray
    eff_matrix:    np.ndarray   # (n_reactions * n_species,)  — flat


@dataclass
class FalloffData:
    n_reactions:   int
    A_k0:          np.ndarray
    b_k0:          np.ndarray
    Ea_k0:         np.ndarray
    A_kinf:        np.ndarray
    b_kinf:        np.ndarray
    Ea_kinf:       np.ndarray
    oneWay:        np.ndarray
    delta_n:       np.ndarray
    troe_A:        np.ndarray
    troe_T1:       np.ndarray
    troe_T2:       np.ndarray
    troe_T3:       np.ndarray
    is_troe:       np.ndarray   # int32
    react_idx:     np.ndarray
    react_sto:     np.ndarray
    react_offset:  np.ndarray
    prod_idx:      np.ndarray
    prod_sto:      np.ndarray
    prod_offset:   np.ndarray
    eff_matrix:    np.ndarray   # (n_reactions * n_species,)  — flat


@dataclass
class MechanismData:
    n_species:         int
    n_reactions_elem:  int
    n_reactions_tb:    int
    n_reactions_fall:  int
    thermo:            ThermoData
    elementary:        Optional[ElementaryData] = None
    threebody:         Optional[ThreeBodyData]  = None
    falloff:           Optional[FalloffData]    = None

    @property
    def n_reactions_total(self):
        return self.n_reactions_elem + self.n_reactions_tb + self.n_reactions_fall

    def to_cuda_arrays(self) -> dict:
        """Upload all arrays to device. Returns flat dict of device arrays."""
        import numba.cuda as cuda

        def _d(arr):
            return cuda.to_device(np.ascontiguousarray(arr))
        def _di(arr):
            return cuda.to_device(np.ascontiguousarray(arr, dtype=np.int32))

        d = {}

        # Thermo — flat (n_species*7,)
        d['thermo_coef_low']  = _d(self.thermo.coef_low)
        d['thermo_coef_high'] = _d(self.thermo.coef_high)
        d['thermo_Tmid']      = _d(self.thermo.Tmid)

        if self.elementary is not None:
            e = self.elementary
            for k in ('A','n','Ea','oneWay','delta_n','react_sto','prod_sto'):
                d[f'e_{k}'] = _d(getattr(e, k))
            for k in ('react_idx','react_offset','prod_idx','prod_offset'):
                d[f'e_{k}'] = _di(getattr(e, k))

        if self.threebody is not None:
            tb = self.threebody
            for k in ('A','n','Ea','oneWay','delta_n','react_sto','prod_sto','eff_matrix'):
                d[f'tb_{k}'] = _d(getattr(tb, k))
            for k in ('react_idx','react_offset','prod_idx','prod_offset'):
                d[f'tb_{k}'] = _di(getattr(tb, k))

        if self.falloff is not None:
            f = self.falloff
            for k in ('A_k0','b_k0','Ea_k0','A_kinf','b_kinf','Ea_kinf',
                      'oneWay','delta_n','troe_A','troe_T1','troe_T2','troe_T3',
                      'react_sto','prod_sto','eff_matrix'):
                d[f'f_{k}'] = _d(getattr(f, k))
            for k in ('react_idx','react_offset','prod_idx','prod_offset'):
                d[f'f_{k}'] = _di(getattr(f, k))
            d['f_is_troe'] = cuda.to_device(
                np.ascontiguousarray(f.is_troe, dtype=np.int32))

        return d


# ─────────────────────────────────────────────────────────────────────────────
#  Builder: torch reaction objects → MechanismData
# ─────────────────────────────────────────────────────────────────────────────

def build_mechanism_data_from_torch(r_e_list, r_tb_list, r_f_list,
                                    vec_th, dtype=np.float64) -> MechanismData:
    n_species = len(vec_th.species)

    thermo = ThermoData(
        n_species     = n_species,
        species_names = list(vec_th.species),
        coef_low      = vec_th.coef_low.cpu().numpy().astype(dtype).reshape(-1),
        coef_high     = vec_th.coef_high.cpu().numpy().astype(dtype).reshape(-1),
        Tmid          = vec_th.Tmid.cpu().numpy().astype(dtype),
        molar_masses  = vec_th.molar_masses.cpu().numpy().astype(dtype),
    )

    return MechanismData(
        n_species         = n_species,
        n_reactions_elem  = len(r_e_list),
        n_reactions_tb    = len(r_tb_list),
        n_reactions_fall  = len(r_f_list),
        thermo            = thermo,
        elementary        = _build_elementary(r_e_list, dtype) if r_e_list else None,
        threebody         = _build_threebody(r_tb_list, n_species, dtype) if r_tb_list else None,
        falloff           = _build_falloff(r_f_list, n_species, dtype) if r_f_list else None,
    )


def _build_csr(reactions, dtype):
    ri, rs, ro, pi, ps, po = [], [], [0], [], [], [0]
    for r in reactions:
        ri.extend(int(x) for x in r.reactants_idx)
        rs.extend(float(x) for x in r.reactants_sto)
        pi.extend(int(x) for x in r.products_idx)
        ps.extend(float(x) for x in r.products_sto)
        ro.append(ro[-1] + len(r.reactants_idx))
        po.append(po[-1] + len(r.products_idx))
    return (np.array(ri, np.int32), np.array(rs, dtype),
            np.array(ro, np.int32),
            np.array(pi, np.int32), np.array(ps, dtype),
            np.array(po, np.int32))


def _build_elementary(reactions, dtype):
    ri, rs, ro, pi, ps, po = _build_csr(reactions, dtype)
    return ElementaryData(
        n_reactions  = len(reactions),
        A            = np.array([r.A.item()  for r in reactions], dtype),
        n            = np.array([r.n.item()  for r in reactions], dtype),
        Ea           = np.array([r.Ea.item() for r in reactions], dtype),
        oneWay       = np.array([float(r.oneWay) for r in reactions], dtype),
        delta_n      = np.array([float(r.products_sto.sum()-r.reactants_sto.sum())
                                 for r in reactions], dtype),
        react_idx=ri, react_sto=rs, react_offset=ro,
        prod_idx=pi,  prod_sto=ps,  prod_offset=po,
    )


def _build_threebody(reactions, n_species, dtype):
    ri, rs, ro, pi, ps, po = _build_csr(reactions, dtype)
    eff = np.stack([r.eff.cpu().numpy() for r in reactions]).astype(dtype).reshape(-1)
    return ThreeBodyData(
        n_reactions  = len(reactions),
        A            = np.array([r.A.item()  for r in reactions], dtype),
        n            = np.array([r.n.item()  for r in reactions], dtype),
        Ea           = np.array([r.Ea.item() for r in reactions], dtype),
        oneWay       = np.array([float(r.oneWay) for r in reactions], dtype),
        delta_n      = np.array([float(r.products_sto.sum()-r.reactants_sto.sum())
                                 for r in reactions], dtype),
        react_idx=ri, react_sto=rs, react_offset=ro,
        prod_idx=pi,  prod_sto=ps,  prod_offset=po,
        eff_matrix=eff,
    )


def _build_falloff(reactions, n_species, dtype):
    ri, rs, ro, pi, ps, po = _build_csr(reactions, dtype)
    eff = np.stack([r.eff.cpu().numpy() for r in reactions]).astype(dtype).reshape(-1)
    def _tp(r, k): return float((getattr(r,'falloffParams',{}) or {}).get(k, 0.0))
    return FalloffData(
        n_reactions = len(reactions),
        A_k0    = np.array([float(r.A_k0)    for r in reactions], dtype),
        b_k0    = np.array([float(r.b_k0)    for r in reactions], dtype),
        Ea_k0   = np.array([float(r.Ea_k0)   for r in reactions], dtype),
        A_kinf  = np.array([float(r.A_kinf)  for r in reactions], dtype),
        b_kinf  = np.array([float(r.b_kinf)  for r in reactions], dtype),
        Ea_kinf = np.array([float(r.Ea_kinf) for r in reactions], dtype),
        oneWay  = np.array([float(r.oneWay)  for r in reactions], dtype),
        delta_n = np.array([float(r.delta_n) for r in reactions], dtype),
        troe_A  = np.array([_tp(r,'A')  for r in reactions], dtype),
        troe_T1 = np.array([_tp(r,'T1') for r in reactions], dtype),
        troe_T2 = np.array([_tp(r,'T2') for r in reactions], dtype),
        troe_T3 = np.array([_tp(r,'T3') for r in reactions], dtype),
        is_troe = np.array([1 if str(getattr(r,'falloffType','lindemann')).lower()=='troe'
                            else 0 for r in reactions], np.int32),
        react_idx=ri, react_sto=rs, react_offset=ro,
        prod_idx=pi,  prod_sto=ps,  prod_offset=po,
        eff_matrix=eff,
    )