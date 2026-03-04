"""
packing.py
==========
Utilities to convert the Python-level reaction objects (from reactionTypesContext.py)
into flat device arrays suitable for the Numba CUDA kernels.

The Numba kernels use a Compressed Sparse Row (CSR) style layout for reactants
and products:
  react_idx[r_flat]  : species index of each reactant entry (across all reactions)
  react_sto[r_flat]  : stoichiometric coefficient
  r_start[i]         : index into react_idx/sto where reaction i starts
  r_end[i]           : one-past-last index for reaction i

This module also bundles all NASA polynomial data from a ThermoData_vectorized
object into flat C-contiguous float64 arrays so that Numba can index them
without Python overhead.
"""

import numpy as np
from numba import cuda


# ─────────────────────────────────────────────────────────────────────────────
# Low-level packing helpers
# ─────────────────────────────────────────────────────────────────────────────

def pack_thermo(thermo):
    """
    Convert a ThermoData_vectorized object to device arrays.

    Returns
    -------
    d_coef_low  : (nsp, 7)  float64 device array
    d_coef_high : (nsp, 7)  float64 device array
    d_Tmid      : (nsp,)    float64 device array
    """
    coef_low  = thermo.coef_low.cpu().numpy().astype(np.float64)   # (nsp, 7)
    coef_high = thermo.coef_high.cpu().numpy().astype(np.float64)  # (nsp, 7)
    Tmid      = thermo.Tmid.cpu().numpy().astype(np.float64)       # (nsp,)

    return (cuda.to_device(np.ascontiguousarray(coef_low)),
            cuda.to_device(np.ascontiguousarray(coef_high)),
            cuda.to_device(np.ascontiguousarray(Tmid)))


def pack_elementary(batch):
    """
    Pack an ElementaryReactionBatchContext into flat device arrays.

    Returns a dict with keys matching the pt/pb kernel signatures.
    """
    react_idx = batch.react_idx.cpu().numpy().astype(np.int32)
    react_sto = batch.react_sto.cpu().numpy().astype(np.float64)
    prod_idx  = batch.prod_idx.cpu().numpy().astype(np.int32)
    prod_sto  = batch.prod_sto.cpu().numpy().astype(np.float64)

    r_offsets = batch.reaction_offsets_r.cpu().numpy().astype(np.int32)
    p_offsets = batch.reaction_offsets_p.cpu().numpy().astype(np.int32)
    r_start   = r_offsets[:-1]
    r_end     = r_offsets[1:]
    p_start   = p_offsets[:-1]
    p_end     = p_offsets[1:]

    A_arr  = batch.A.cpu().numpy().astype(np.float64)
    n_arr  = batch.n.cpu().numpy().astype(np.float64)
    Ea_arr = batch.Ea.cpu().numpy().astype(np.float64)
    oneWay = batch.oneWay.cpu().numpy().astype(np.float64)
    dn     = batch.delta_n.cpu().numpy().astype(np.float64)

    return {
        'react_idx': cuda.to_device(np.ascontiguousarray(react_idx)),
        'react_sto': cuda.to_device(np.ascontiguousarray(react_sto)),
        'r_start':   cuda.to_device(np.ascontiguousarray(r_start)),
        'r_end':     cuda.to_device(np.ascontiguousarray(r_end)),
        'prod_idx':  cuda.to_device(np.ascontiguousarray(prod_idx)),
        'prod_sto':  cuda.to_device(np.ascontiguousarray(prod_sto)),
        'p_start':   cuda.to_device(np.ascontiguousarray(p_start)),
        'p_end':     cuda.to_device(np.ascontiguousarray(p_end)),
        'A_arr':     cuda.to_device(np.ascontiguousarray(A_arr)),
        'n_arr':     cuda.to_device(np.ascontiguousarray(n_arr)),
        'Ea_arr':    cuda.to_device(np.ascontiguousarray(Ea_arr)),
        'oneWay':    cuda.to_device(np.ascontiguousarray(oneWay)),
        'delta_n':   cuda.to_device(np.ascontiguousarray(dn)),
    }


def pack_threebody(batch):
    """Pack a ThreeBodyReactionBatchContext."""
    d = pack_elementary(batch)   # inherits all elementary fields
    eff = batch.eff_matrix.cpu().numpy().astype(np.float64)   # (nreac, nsp)
    d['eff_matrix'] = cuda.to_device(np.ascontiguousarray(eff))
    return d


def pack_falloff(batch):
    """Pack a FalloffReactionBatchContext."""
    react_idx = batch.react_idx.cpu().numpy().astype(np.int32)
    react_sto = batch.react_sto.cpu().numpy().astype(np.float64)
    prod_idx  = batch.prod_idx.cpu().numpy().astype(np.int32)
    prod_sto  = batch.prod_sto.cpu().numpy().astype(np.float64)

    r_offsets = batch.reaction_offsets_r.cpu().numpy().astype(np.int32)
    p_offsets = batch.reaction_offsets_p.cpu().numpy().astype(np.int32)
    r_start   = r_offsets[:-1]
    r_end     = r_offsets[1:]
    p_start   = p_offsets[:-1]
    p_end     = p_offsets[1:]

    eff = batch.eff_matrix.cpu().numpy().astype(np.float64)

    troe_A  = batch.troe_A.cpu().numpy().astype(np.float64)
    troe_T1 = batch.troe_T1.cpu().numpy().astype(np.float64)
    troe_T2 = batch.troe_T2.cpu().numpy().astype(np.float64)
    troe_T3 = batch.troe_T3.cpu().numpy().astype(np.float64)
    is_troe = np.array([1 if t == 'troe' else 0
                        for t in batch.falloff_types], dtype=np.int32)

    return {
        'react_idx': cuda.to_device(np.ascontiguousarray(react_idx)),
        'react_sto': cuda.to_device(np.ascontiguousarray(react_sto)),
        'r_start':   cuda.to_device(np.ascontiguousarray(r_start)),
        'r_end':     cuda.to_device(np.ascontiguousarray(r_end)),
        'prod_idx':  cuda.to_device(np.ascontiguousarray(prod_idx)),
        'prod_sto':  cuda.to_device(np.ascontiguousarray(prod_sto)),
        'p_start':   cuda.to_device(np.ascontiguousarray(p_start)),
        'p_end':     cuda.to_device(np.ascontiguousarray(p_end)),
        'A_k0':      cuda.to_device(np.ascontiguousarray(
                         batch.A_k0.cpu().numpy().astype(np.float64))),
        'b_k0':      cuda.to_device(np.ascontiguousarray(
                         batch.b_k0.cpu().numpy().astype(np.float64))),
        'Ea_k0':     cuda.to_device(np.ascontiguousarray(
                         batch.Ea_k0.cpu().numpy().astype(np.float64))),
        'A_kinf':    cuda.to_device(np.ascontiguousarray(
                         batch.A_kinf.cpu().numpy().astype(np.float64))),
        'b_kinf':    cuda.to_device(np.ascontiguousarray(
                         batch.b_kinf.cpu().numpy().astype(np.float64))),
        'Ea_kinf':   cuda.to_device(np.ascontiguousarray(
                         batch.Ea_kinf.cpu().numpy().astype(np.float64))),
        'oneWay':    cuda.to_device(np.ascontiguousarray(
                         batch.oneWay.cpu().numpy().astype(np.float64))),
        'delta_n':   cuda.to_device(np.ascontiguousarray(
                         batch.delta_n.cpu().numpy().astype(np.float64))),
        'eff_matrix':cuda.to_device(np.ascontiguousarray(eff)),
        'troe_A':    cuda.to_device(np.ascontiguousarray(troe_A)),
        'troe_T1':   cuda.to_device(np.ascontiguousarray(troe_T1)),
        'troe_T2':   cuda.to_device(np.ascontiguousarray(troe_T2)),
        'troe_T3':   cuda.to_device(np.ascontiguousarray(troe_T3)),
        'is_troe':   cuda.to_device(np.ascontiguousarray(is_troe)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# High-level packing: whole ReactionSystem
# ─────────────────────────────────────────────────────────────────────────────

class NumbaPacked:
    """
    Container for all device arrays needed by the Numba integrator.
    Created once at init; never transferred again during integration.
    """

    def __init__(self, thermo, batched_elem, batched_tb, batched_f):
        self.d_coef_low, self.d_coef_high, self.d_Tmid = pack_thermo(thermo)

        self.elem = pack_elementary(batched_elem) if batched_elem.n_reactions > 0 else None
        self.tb   = pack_threebody(batched_tb)   if batched_tb.n_reactions   > 0 else None
        self.fo   = pack_falloff(batched_f)      if batched_f.n_reactions    > 0 else None

        self.n_species   = len(thermo.species)
        self.has_elem    = (self.elem is not None)
        self.has_tb      = (self.tb   is not None)
        self.has_falloff = (self.fo   is not None)

    # Convenience: the full state vector is TC = [T | C]  shape (ncells, 1+nsp)
    @staticmethod
    def split_TC(TC_dev):
        """
        Split device array (ncells, 1+nsp) into T (ncells,) and C (ncells, nsp).
        Returns views — no copy.
        """
        import numba.cuda as _cuda
        # We cannot slice device arrays directly in Numba Python host code;
        # instead we use the sub-array trick via cupy interop or we pass
        # offset indices to the kernels.
        #
        # For simplicity we accept a torch Tensor on device and convert to
        # Numba device arrays via __cuda_array_interface__.
        raise NotImplementedError(
            "Use NumbaSolver.split_TC() which handles the torch↔numba bridge.")