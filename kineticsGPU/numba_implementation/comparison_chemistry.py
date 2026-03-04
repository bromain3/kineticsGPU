import numba
import time
import kineticsGPU.numba_implementation.thermo as thermo
from numba import cuda
import numpy as np
import torch
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.numba_implementation.yaml_to_GPU_array import build_elementary_gpu_arrays, build_threebody_gpu_arrays, build_falloff_gpu_arrays
from kineticsGPU.numba_implementation.elementary_reaction import per_thread_elem_react, per_thread_elem_react_jvp
from kineticsGPU.numba_implementation.three_body_reaction import per_thread_three_body_react, per_thread_three_body_react_jvp
from kineticsGPU.numba_implementation.falloff_reaction import per_thread_falloff_react, per_thread_falloff_react_jvp
from kineticsGPU.reactionTypes import FalloffReactionBatch
from validationReactions import ElementaryReactionVal, ThreeBodyReactionVal, FalloffReactionVal
import thermo_classique as th_clas
from reactionTypes import ElementaryReactionBatch, ThreeBodyReactionBatch

yaml_file = '../gri30.yaml'
ncells = 1000

lecteur = thermo.yamlParser(yaml_file)
lecteur_clas = th_clas.yamlParser(yaml_file)
th_object = lecteur.populateVectorizedThermoObject()
clas_th_object = lecteur_clas.populateVectorizedThermoObject(device='cuda:0', dtype=torch.float64)

reactions = parseReactions(yaml_file)

elem_reacs = []
threebody_reacs = []
falloff_reacs = []


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

r_e, r_tb, r_f, system = build_reactions(clas_th_object.species, reactions, clas_th_object, 'cuda:0', torch.float64)
batched_elem = ElementaryReactionBatch(r_e, device='cuda:0', dtype=torch.float64)
batched_tb = ThreeBodyReactionBatch(r_tb, device='cuda:0', dtype=torch.float64)
batched_falloff = FalloffReactionBatch(r_f, device='cuda:0', dtype=torch.float64)


arrays = build_elementary_gpu_arrays(reactions, th_object)

react_idx_d = arrays[0]
react_sto_d = arrays[1]
reactant_number_d = arrays[2]
reactant_offset_d = arrays[3]

prod_idx_d = arrays[4]
prod_sto_d = arrays[5]
prod_number_d = arrays[6]
prod_offset_d = arrays[7]

A_d = arrays[8]
n_d = arrays[9]
Ea_d = arrays[10]
oneWay_d = arrays[11]
delta_n_d = arrays[12]


coef_low = th_object.coef_low
high_coef = th_object.coef_high
Tmid = th_object.Tmid


#omega = cuda.device_array((1, Tmid.shape[0]), dtype=np.float64)
omega = cuda.to_device(np.zeros((ncells, Tmid.shape[0]), dtype=np.float64))
jvp = cuda.to_device(np.zeros((ncells, Tmid.shape[0]), dtype=np.float64))
#T = cuda.device_array((1,), dtype=np.float64)
#C = cuda.device_array((1, Tmid.shape[0]), dtype=np.float64)

TC = np.random.random((ncells, len(clas_th_object.species) + 1))
TC[:, 0] = 2000.0
vTC = np.random.random((ncells, len(clas_th_object.species) + 1))
T = cuda.to_device(np.ascontiguousarray(TC[:, 0], dtype=np.float64))
C = cuda.to_device(np.ascontiguousarray(TC[:, 1:], dtype=np.float64))
vT = cuda.to_device(np.ascontiguousarray(vTC[:, 0], dtype=np.float64))
vC = cuda.to_device(np.ascontiguousarray(vTC[:, 1:], dtype=np.float64))

T_torch = torch.tensor(TC[:, 0], device='cuda:0', dtype=torch.float64)
C_torch = torch.tensor(TC[:, 1:], device='cuda:0', dtype=torch.float64)
vT_torch = torch.tensor(vTC[:, 0], device='cuda:0', dtype=torch.float64)
vC_torch = torch.tensor(vTC[:, 1:], device='cuda:0', dtype=torch.float64)

@cuda.jit # TODO: to improve computation speed, take cell variables as input tu prevent global memory use (and use local arrays in the kernel to have only one global memory read/write)
def kernel_elem(        # state
        T, C,  # (ncells,)  (ncells, nsp)
        # NASA thermo
        coef_low, coef_high,  # (nsp, 7)
        Tmid,  # (nsp,)
        # reaction connectivity
        react_idx, react_sto, reactant_number, reactant_offset,  # flattened reactant data
        prod_idx, prod_sto, prod_number, prod_offset,  # flattened product  data
        # Arrhenius + flags
        A_arr, n_arr, Ea_arr,  # (nreac,)
        oneWay,  # (nreac,)  1.0 = irreversible
        delta_n,  # (nreac,)  Δν = Σν_prod - Σν_react
        # perturbations
        vT, vC,
        # output
        omega,  # (ncells, nsp)  species production rates
        jvp
        ):
        cell_idx = cuda.grid(1)

        if cell_idx >= T.shape[0]:
                return


        per_thread_elem_react_jvp(T, C,  # (ncells,)  (ncells, nsp)
        cell_idx,
        # NASA thermo
        coef_low, coef_high,  # (nsp, 7)
        Tmid,  # (nsp,)
        # reaction connectivity
        react_idx, react_sto, reactant_number, reactant_offset,  # flattened reactant data
        prod_idx, prod_sto, prod_number, prod_offset,  # flattened product  data
        # Arrhenius + flags
        A_arr, n_arr, Ea_arr,  # (nreac,)
        oneWay,  # (nreac,)  1.0 = irreversible
        delta_n,  # (nreac,)  Δν = Σν_prod - Σν_react
        # perturbations
        vT, vC,
        # output
        omega, jvp)

print('Elementary reactions:')
t0 = time.perf_counter()
kernel_elem[100000, 64](
        # state
        T, C,  # (ncells,)  (ncells, nsp)
        # NASA thermo
        coef_low, high_coef,  # (nsp, 7)
        Tmid,  # (nsp,)
        # reaction connectivity
        react_idx_d, react_sto_d, reactant_number_d, reactant_offset_d,  # flattened reactant data
        prod_idx_d, prod_sto_d, prod_number_d, prod_offset_d,  # flattened product  data
        # Arrhenius + flags
        A_d, n_d, Ea_d,  # (nreac,)
        oneWay_d,  # (nreac,)  1.0 = irreversible
        delta_n_d,  # (nreac,)  Δν = Σν_prod - Σν_react
        # perturbations
        vT, vC,
        # output
        omega, jvp  # (ncells, nsp)  species production rates
)
cuda.synchronize()
print('Compile + runtime of kernel:', time.perf_counter() - t0)

t0 = time.perf_counter()
RR_torch = batched_elem.RR(T_torch, C_torch)
omega_torch = batched_elem.omega_from_RR(RR_torch)
jvp_torch = batched_elem.jvp(T_torch, C_torch, vT_torch, vC_torch)
print('runtime of torch:', time.perf_counter() - t0)

scale = np.abs(omega_torch.cpu().numpy()).max()
err_omega = np.abs(omega.copy_to_host() - omega_torch.cpu().numpy()) / (scale + 1e-300)

scale_jvp = np.abs(jvp_torch.cpu().numpy()).max()
err_jvp = np.abs(jvp.copy_to_host() - jvp_torch.cpu().numpy()) / (scale_jvp + 1e-300)

print('\tomega : min error', err_omega.min(), 'max error', err_omega.max(), 'mean error', err_omega.mean())
print('\tjvp : min error', err_jvp.min(), 'max error', err_jvp.max(), 'mean error', err_jvp.mean())













arrays = build_threebody_gpu_arrays(reactions, th_object)

(
    react_idx_d, react_sto_d, reactant_number_d, reactant_offset_d,
    prod_idx_d, prod_sto_d, prod_number_d, prod_offset_d,
    A_d, n_d, Ea_d, oneWay_d, delta_n_d,
    eff_array_d
) = arrays

omega = cuda.to_device(np.zeros((ncells, Tmid.shape[0]), dtype=np.float64))
jvp = cuda.to_device(np.zeros((ncells, Tmid.shape[0]), dtype=np.float64))


@cuda.jit # TODO: to improve computation speed, take cell variables as input tu prevent global memory use (and use local arrays in the kernel to have only one global memory read/write)
def kernel_tb(        # state
        T, C,  # (ncells,)  (ncells, nsp)
        # NASA thermo
        coef_low, coef_high,  # (nsp, 7)
        Tmid,  # (nsp,)
        # reaction connectivity
        react_idx, react_sto, reactant_number, reactant_offset,  # flattened reactant data
        prod_idx, prod_sto, prod_number, prod_offset,  # flattened product  data
        # Arrhenius + flags
        A_arr, n_arr, Ea_arr,  # (nreac,)
        oneWay,  # (nreac,)  1.0 = irreversible
        delta_n,  # (nreac,)  Δν = Σν_prod - Σν_react
        effmatrix,
        # perturbations
        vT, vC,
        # output
        omega,  # (ncells, nsp)  species production rates
        jvp
        ):
        cell_idx = cuda.grid(1)

        if cell_idx >= T.shape[0]:
                return


        per_thread_three_body_react_jvp(T, C,  # (ncells,)  (ncells, nsp)
        cell_idx,
        # NASA thermo
        coef_low, coef_high,  # (nsp, 7)
        Tmid,  # (nsp,)
        # reaction connectivity
        react_idx, react_sto, reactant_number, reactant_offset,  # flattened reactant data
        prod_idx, prod_sto, prod_number, prod_offset,  # flattened product  data
        # Arrhenius + flags
        A_arr, n_arr, Ea_arr,  # (nreac,)
        oneWay,  # (nreac,)  1.0 = irreversible
        delta_n,  # (nreac,)  Δν = Σν_prod - Σν_react
        effmatrix,
        # perturbation
        vT, vC,
        # output
        omega,
        jvp)


print('threebody reactions:')
t0 = time.perf_counter()
kernel_tb[100000, 64](
        # state
        T, C,  # (ncells,)  (ncells, nsp)
        # NASA thermo
        coef_low, high_coef,  # (nsp, 7)
        Tmid,  # (nsp,)
        # reaction connectivity
        react_idx_d, react_sto_d, reactant_number_d, reactant_offset_d,  # flattened reactant data
        prod_idx_d, prod_sto_d, prod_number_d, prod_offset_d,  # flattened product  data
        # Arrhenius + flags
        A_d, n_d, Ea_d,  # (nreac,)
        oneWay_d,  # (nreac,)  1.0 = irreversible
        delta_n_d,  # (nreac,)  Δν = Σν_prod - Σν_react
        eff_array_d,
        # perturbations
        vT, vC,
        # output
        omega,  # (ncells, nsp)  species production rates
        jvp
)
cuda.synchronize()
print('Compile + runtime of kernel:', time.perf_counter() - t0)

t0 = time.perf_counter()
RR_torch = batched_tb.RR(T_torch, C_torch)
omega_torch = batched_tb.omega_from_RR(RR_torch)
jvp_torch = batched_tb.jvp(T_torch, C_torch, vT_torch, vC_torch)
print('runtime of torch:', time.perf_counter() - t0)

scale = np.abs(omega_torch.cpu().numpy()).max()
err_omega = np.abs(omega.copy_to_host() - omega_torch.cpu().numpy()) / (scale + 1e-300)

scale_jvp = np.abs(jvp_torch.cpu().numpy()).max()
err_jvp = np.abs(jvp.copy_to_host() - jvp_torch.cpu().numpy()) / (scale_jvp + 1e-300)

print('\tomega : min error', err_omega.min(), 'max error', err_omega.max(), 'mean error', err_omega.mean())
print('\tjvp : min error', err_jvp.min(), 'max error', err_jvp.max(), 'mean error', err_jvp.mean())







@cuda.jit
def kernel_falloff(
        T, C,
        coef_low, coef_high, Tmid,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        A0, n0, Ea0, Ainf, ninf, Eainf,
        oneWay, delta_n,
        effmatrix,
        falloff_type,
        troe_A, troe_T1, troe_T2, troe_T3,
        vT, vC,
        omega, jvp):
    cell_idx = cuda.grid(1)
    if cell_idx >= T.shape[0]:
        return
    per_thread_falloff_react_jvp(
        T, C, cell_idx,
        coef_low, coef_high, Tmid,
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,  prod_offset,
        A0, n0, Ea0, Ainf, ninf, Eainf,
        oneWay, delta_n, effmatrix,
        falloff_type, troe_A, troe_T1, troe_T2, troe_T3,
        vT, vC,
        omega, jvp)


(react_idx_d, react_sto_d, reactant_number_d, reactant_offset_d,
 prod_idx_d,  prod_sto_d,  prod_number_d,  prod_offset_d,
 A0_d, n0_d, Ea0_d, Ainf_d, ninf_d, Eainf_d,
 oneWay_d, delta_n_d,
 eff_d,
 ftype_d,
 troe_A_d, troe_T1_d, troe_T2_d, troe_T3_d) = build_falloff_gpu_arrays(reactions, th_object)

omega = cuda.to_device(np.zeros((ncells, len(th_object.species)), dtype=np.float64))
jvp = cuda.to_device(np.zeros((ncells, len(th_object.species)), dtype=np.float64))

print('falloff reactions:')
t0 = time.perf_counter()
kernel_falloff[100000, 64](
    T, C,
    coef_low, high_coef, Tmid,
    react_idx_d, react_sto_d, reactant_number_d, reactant_offset_d,
    prod_idx_d,  prod_sto_d,  prod_number_d,  prod_offset_d,
    A0_d, n0_d, Ea0_d, Ainf_d, ninf_d, Eainf_d,
    oneWay_d, delta_n_d,
    eff_d, ftype_d,
    troe_A_d, troe_T1_d, troe_T2_d, troe_T3_d,
    vT, vC,
    omega, jvp)
print('Compile + runtime of kernel:', time.perf_counter() - t0)


t0 = time.perf_counter()
RR_torch = batched_falloff.RR(T_torch, C_torch)
omega_torch = batched_falloff.omega_from_RR(RR_torch)
jvp_torch = batched_falloff.jvp(T_torch, C_torch, vT_torch, vC_torch)
print('runtime of torch:', time.perf_counter() - t0)

scale = np.abs(omega_torch.cpu().numpy()).max()
err_omega = np.abs(omega.copy_to_host() - omega_torch.cpu().numpy()) / (scale + 1e-300)

scale_jvp = np.abs(jvp_torch.cpu().numpy()).max()
err_jvp = np.abs(jvp.copy_to_host() - jvp_torch.cpu().numpy()) / (scale_jvp + 1e-300)

print('falloff reactions:')
print('\tomega : min error', err_omega.min(), 'max error', err_omega.max(), 'mean error', err_omega.mean())
print('\tjvp : min error', err_jvp.min(), 'max error', err_jvp.max(), 'mean error', err_jvp.mean())
