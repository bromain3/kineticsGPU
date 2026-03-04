import time
import kineticsGPU.numba_implementation.thermo as thermo
from numba import cuda
import numpy as np
import torch
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.numba_implementation.yaml_to_GPU_array import (
    build_elementary_gpu_arrays,
    build_threebody_gpu_arrays,
    build_falloff_gpu_arrays,
)
from kineticsGPU.numba_implementation.elementary_reaction import (
    pack_elementary_context_packaged,
    elem_omega_packaged,
    elem_jvp_packaged,
    N_CTX_FIELDS as N_CTX_ELEM,
)
from kineticsGPU.numba_implementation.three_body_reaction import (
    pack_threebody_context_packaged,
    tb_omega_packaged,
    tb_jvp_packaged,
    N_CTX_TB_FIELDS as N_CTX_TB,
)
from kineticsGPU.numba_implementation.falloff_reaction import (
    pack_falloff_context_packaged,
    fo_omega_packaged,
    fo_jvp_packaged,
    N_CTX_FO_FIELDS as N_CTX_FO,
)
from validationReactions import ElementaryReactionVal, ThreeBodyReactionVal, FalloffReactionVal
import thermo_classique as th_clas
from reactionTypesContext import ElementaryReactionBatchContext, ThreeBodyReactionBatchContext, FalloffReactionBatchContext

# =============================================================================
#  SETUP
# =============================================================================

yaml_file = '../gri30.yaml'
ncells    = 1000

lecteur      = thermo.yamlParser(yaml_file)
lecteur_clas = th_clas.yamlParser(yaml_file)
th_object      = lecteur.populateVectorizedThermoObject()
clas_th_object = lecteur_clas.populateVectorizedThermoObject(device='cuda:0', dtype=torch.float64)

reactions = parseReactions(yaml_file)

elem_reacs     = []
threebody_reacs = []
falloff_reacs  = []

for r in reactions:
    if r['type'] == 'elementary':
        elem_reacs.append(ElementaryReactionVal(
            r['reactants'], r['products'], clas_th_object, clas_th_object.species,
            not r['reversible'], r['A_SI'], r['b'], r['Ea_SI']
        ))
    elif r['type'] == 'threebody':
        threebody_reacs.append(ThreeBodyReactionVal(
            r['reactants'], r['products'], clas_th_object, clas_th_object.species,
            not r['reversible'], r['A_SI'], r['b'], r['Ea_SI'],
            r['efficiencies'], defaultEfficiencies=r['defaultEff']
        ))
    elif 'falloff' in r['type']:
        falloff_reacs.append(FalloffReactionVal(
            r['reactants'], r['products'], clas_th_object, clas_th_object.species,
            not r['reversible'],
            (r['k0']['A_SI'], r['k0']['b'], r['k0']['Ea_SI']),
            (r['kinf']['A_SI'], r['kinf']['b'], r['kinf']['Ea_SI']),
            r['efficiencies'],
            falloff_type='troe' if 'troe' in r['type'] else 'lindemann',
            falloff_params=r.get('Troe_params', None)
        ))

r_e, r_tb, r_f, system = build_reactions(
    clas_th_object.species, reactions, clas_th_object, 'cuda:0', torch.float64)
batched_elem    = ElementaryReactionBatchContext(r_e,  device='cuda:0', dtype=torch.float64)
batched_tb      = ThreeBodyReactionBatchContext (r_tb, device='cuda:0', dtype=torch.float64)
batched_falloff = FalloffReactionBatchContext   (r_f,  device='cuda:0', dtype=torch.float64)

# ── arrays GPU pour les trois types ──────────────────────────────────────────
elem_arrays    = build_elementary_gpu_arrays(reactions, th_object)
tb_arrays      = build_threebody_gpu_arrays (reactions, th_object)
fo_arrays      = build_falloff_gpu_arrays   (reactions, th_object)
thermo_arrays  = th_object.package_values()

nspecies   = thermo_arrays[2].shape[0]
nreac_elem = elem_arrays[8].shape[0]    # A_arr élémentaires
nreac_tb   = tb_arrays  [8].shape[0]    # A_arr three-body
nreac_fo   = fo_arrays  [8].shape[0]    # A0_arr falloff

# ── données d'entrée ─────────────────────────────────────────────────────────
TC  = np.random.random((ncells, nspecies + 1))
TC[:, 0] = 2000.0
vTC = np.random.random((ncells, nspecies + 1))

T  = cuda.to_device(np.ascontiguousarray(TC[:,  0], dtype=np.float64))
C  = cuda.to_device(np.ascontiguousarray(TC[:, 1:], dtype=np.float64))
vT = cuda.to_device(np.ascontiguousarray(vTC[:,  0], dtype=np.float64))
vC = cuda.to_device(np.ascontiguousarray(vTC[:, 1:], dtype=np.float64))

T_torch  = torch.tensor(TC[:,  0], device='cuda:0', dtype=torch.float64)
C_torch  = torch.tensor(TC[:, 1:], device='cuda:0', dtype=torch.float64)
vT_torch = torch.tensor(vTC[:,  0], device='cuda:0', dtype=torch.float64)
vC_torch = torch.tensor(vTC[:, 1:], device='cuda:0', dtype=torch.float64)


# =============================================================================
#  KERNELS
# =============================================================================

@cuda.jit
def kernel_elementary(T, C, vT, vC, omega, jvp_out,
                       react_arrays, thermo_arrays, ctx_buf):
    cell_idx = cuda.grid(1)
    if cell_idx >= T.shape[0]:
        return
    Tcell      = T      [cell_idx]
    Ccell      = C      [cell_idx]
    vTcell     = vT     [cell_idx]
    vCcell     = vC     [cell_idx]
    ctx        = ctx_buf[cell_idx]
    omega_cell = omega  [cell_idx]
    jvp_cell   = jvp_out[cell_idx]

    pack_elementary_context_packaged(Tcell, Ccell, react_arrays, thermo_arrays, ctx)
    elem_omega_packaged(react_arrays, ctx, omega_cell)
    elem_jvp_packaged  (Ccell, react_arrays, ctx, vTcell, vCcell, jvp_cell)


@cuda.jit
def kernel_threebody(T, C, vT, vC, omega, jvp_out,
                     react_arrays, thermo_arrays, ctx_buf):
    cell_idx = cuda.grid(1)
    if cell_idx >= T.shape[0]:
        return
    Tcell      = T      [cell_idx]
    Ccell      = C      [cell_idx]
    vTcell     = vT     [cell_idx]
    vCcell     = vC     [cell_idx]
    ctx        = ctx_buf[cell_idx]
    omega_cell = omega  [cell_idx]
    jvp_cell   = jvp_out[cell_idx]

    pack_threebody_context_packaged(Tcell, Ccell, react_arrays, thermo_arrays, ctx)
    tb_omega_packaged(react_arrays, ctx, omega_cell)
    tb_jvp_packaged  (Ccell, vTcell, vCcell, react_arrays, ctx, jvp_cell)


@cuda.jit
def kernel_falloff(T, C, vT, vC, omega, jvp_out,
                   react_arrays, thermo_arrays, ctx_buf):
    cell_idx = cuda.grid(1)
    if cell_idx >= T.shape[0]:
        return
    Tcell      = T      [cell_idx]
    Ccell      = C      [cell_idx]
    vTcell     = vT     [cell_idx]
    vCcell     = vC     [cell_idx]
    ctx        = ctx_buf[cell_idx]
    omega_cell = omega  [cell_idx]
    jvp_cell   = jvp_out[cell_idx]

    pack_falloff_context_packaged(Tcell, Ccell, react_arrays, thermo_arrays, ctx)
    fo_omega_packaged(react_arrays, ctx, omega_cell)
    fo_jvp_packaged  (Ccell, vTcell, vCcell, react_arrays, ctx, jvp_cell)


# =============================================================================
#  HELPER : affiche les erreurs relatives omega et jvp
# =============================================================================

def print_errors(label, numba_arr, torch_ref):
    np_arr = numba_arr.copy_to_host()
    ref    = torch_ref.cpu().numpy()
    err    = np.abs(np_arr - ref) / (np.abs(ref) + 1e-300)
    print(f'  {label} — min: {err.min():.2e}  max: {err.max():.2e}'
          f'  mean: {err.mean():.2e}')


def run_validation(label, kernel, ctx_shape,
                   react_arrays, torch_ctx_fn, torch_omega_fn, torch_jvp_fn,
                   grid=(100000, 64)):

    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')

    omega_buf = cuda.to_device(np.zeros((ncells, nspecies), dtype=np.float64))
    jvp_buf   = cuda.to_device(np.zeros((ncells, nspecies), dtype=np.float64))
    ctx_buf   = cuda.to_device(np.zeros(ctx_shape,          dtype=np.float64))

    # ── compilation + premier run ─────────────────────────────────────────
    t0 = time.perf_counter()
    kernel[grid](T, C, vT, vC, omega_buf, jvp_buf,
                 react_arrays, thermo_arrays, ctx_buf)
    cuda.synchronize()
    print(f'  Compile + runtime : {time.perf_counter() - t0:.4f} s')

    # ── référence torch ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    ctx_torch   = torch_ctx_fn(T_torch, C_torch)
    omega_torch = torch_omega_fn(ctx_torch)
    jvp_torch   = torch_jvp_fn(ctx_torch, vT_torch, vC_torch)
    print(f'  Torch runtime     : {time.perf_counter() - t0:.4f} s')

    # ── erreurs ───────────────────────────────────────────────────────────
    print_errors('omega', omega_buf, omega_torch)
    print_errors('jvp  ', jvp_buf,   jvp_torch)

    # ── runtime seul ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    kernel[grid](T, C, vT, vC, omega_buf, jvp_buf,
                 react_arrays, thermo_arrays, ctx_buf)
    cuda.synchronize()
    print(f'  Runtime only      : {time.perf_counter() - t0:.4f} s')


# =============================================================================
#  VALIDATION — ELEMENTARY
# =============================================================================

run_validation(
    label         = 'Elementary reactions',
    kernel        = kernel_elementary,
    ctx_shape     = (ncells, N_CTX_ELEM, nreac_elem),
    react_arrays  = elem_arrays,
    torch_ctx_fn  = lambda T, C: batched_elem.evaluate_context(T, C),
    torch_omega_fn= lambda ctx: batched_elem.omega_from_RR(ctx.RR),
    torch_jvp_fn  = lambda ctx, vT, vC: batched_elem.jvp(ctx, vT, vC),
)

# =============================================================================
#  VALIDATION — THREE-BODY
# =============================================================================

run_validation(
    label         = 'Three-body reactions',
    kernel        = kernel_threebody,
    ctx_shape     = (ncells, N_CTX_TB, nreac_tb),
    react_arrays  = tb_arrays,
    torch_ctx_fn  = lambda T, C: batched_tb.evaluate_context(T, C),
    torch_omega_fn= lambda ctx: batched_tb.omega_from_RR(ctx.RR),
    torch_jvp_fn  = lambda ctx, vT, vC: batched_tb.jvp(ctx, vT, vC),
)

# =============================================================================
#  VALIDATION — FALLOFF
# =============================================================================

run_validation(
    label         = 'Falloff reactions',
    kernel        = kernel_falloff,
    ctx_shape     = (ncells, N_CTX_FO, nreac_fo),
    react_arrays  = fo_arrays,
    torch_ctx_fn  = lambda T, C: batched_falloff.evaluate_context(T, C),
    torch_omega_fn= lambda ctx: batched_falloff.omega_from_RR(ctx.RR),
    torch_jvp_fn  = lambda ctx, vT, vC: batched_falloff.jvp(ctx, vT, vC),
)