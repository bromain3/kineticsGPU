"""
test_cuda_integrator_full.py — validation CUDA calquée sur le script torch.

Produit exactement les mêmes prints et figures que le script torch original :
  - Warmup       : force la compilation JIT du kernel principal avant les mesures
  - Section RR_and_source_terms : validation des taux de réaction et omega
    via CUDAReactionEvaluator (kernels annexes, un thread par réacteur)
  - Section Integrators : comparaison Cantera / CPU BDF / GPU CUDA
  - Section Scalability : temps de calcul GPU vs Cantera en fonction du nombre
    de réacteurs (N_reactors_list), produit scalability.png

Usage:
    python tests/test_cuda_integrator_full.py
"""

import numpy as np
import math
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.claude.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.claude.integrator.cuda_integrator_pb import CUDABackwardEulerIntegratorPB
from kineticsGPU.claude.kernels.chemistry_kernels_article import CUDAReactionEvaluator
import numba.cuda as cuda

import cantera as ct
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import torch


##############################
########## Analysis ##########
##############################
RR_and_source_terms = True
Integrators         = True
Scalability         = True

# Reactor counts to sweep for the scalability test.
# GPU side is always batched; Cantera side is sequential.
N_reactors_list     = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Number of time steps for the scalability test (shorter than the full run
# to keep wall time reasonable — same h, fewer steps).
N_STEPS_SCALABILITY = 3000   # = 3000 x h = 300 us of chemistry
size = 1000

###############################
########## Utilities ##########
###############################
custom_rc = {
    'text.usetex': True,
    'figure.figsize': (3.5, 2.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'svg',
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'axes.formatter.use_mathtext': True,
    'axes.formatter.useoffset': True,
    'axes.formatter.limits': (-4, 4),
    'axes.axisbelow': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'lines.linewidth': 0.8,
    'lines.markersize': 2.0,
    'legend.frameon': False,
    'legend.handlelength': 1.8,
    'image.cmap': 'coolwarm',
    'axes.prop_cycle': plt.cycler('color', plt.cm.tab10.colors),
}
matplotlib.rcParams.update(custom_rc)


def MAE(A, B):
    err = np.abs(A - B)
    return err.mean(), err.std(), err

def MSE(A, B):
    err = (A - B) ** 2
    return err.mean(), err.std(), err

def plot_vs_time(time, err, savingPath, show=False):
    fig, ax = plt.subplots()
    ax.plot(time, err)
    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show: plt.show()
    plt.close()

def plots_vs_time(time, errs, names, savingPath, show=False):
    fig, ax = plt.subplots()
    for err, name in zip(errs, names):
        ax.plot(time, err, label=name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show: plt.show()
    plt.close()

def scatter_vs_time(time, err, savingPath, show=False):
    fig, ax = plt.subplots()
    ax.scatter(time, err)
    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show: plt.show()
    plt.close()

def colored_scatter_vs_time(time, err, color, savingPath, show=False):
    fig, ax = plt.subplots()
    sc = ax.scatter(time, err, c=color)
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show: plt.show()
    plt.close()

def plot_hist(errs, names, savingPath, show=False):
    if len(errs) != len(names):
        raise ValueError("`errs` and `names` lists must have the same length.")
    data_to_plot = [e.ravel() for e in errs]
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.boxplot(data_to_plot,
               labels=names,
               patch_artist=True,
               showfliers=False,
               medianprops=dict(color="black"),
               boxprops=dict(facecolor="#1f77b4", alpha=0.6))
    from matplotlib import ticker as mticker
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(mticker.LogLocator(
        numticks=999, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9)))
    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show: plt.show()
    plt.close()

def interpolate_trajectory(times, T, P, X, t_ref):
    T_i = interp1d(times, T, kind='linear', fill_value="extrapolate")(t_ref)
    P_i = interp1d(times, P, kind='linear', fill_value="extrapolate")(t_ref)
    X_i = np.zeros((len(t_ref), X.shape[1]))
    for k in range(X.shape[1]):
        X_i[:, k] = interp1d(
            times, X[:, k], kind='linear', fill_value="extrapolate")(t_ref)
    return T_i, P_i, X_i


###################################
########## Files reading ##########
###################################

yaml_file = "../gri30.yaml"
dtype_np  = np.float64
dtype_th  = torch.float64
device    = 'cuda'

y      = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=dtype_th, device=device)
rxns   = parseReactions(yaml_file)

gas = ct.Solution(yaml_file)

r_e, r_tb, r_f, system = build_reactions(
    vec_th.species, rxns, vec_th, device, dtype_th)

print()
print(80 * '=')
print('General model stats:')
print(f'\t{len(r_e)} elementary reactions, '
      f'{len(r_tb)} third body reactions, '
      f'{len(r_f)} falloff reactions')

# MechanismData partagé par les deux sections
mech = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype_np)

# Ordered reaction-type list — needed to split Cantera RRs by type
reactions_index_to_type = [r['type'] for r in rxns]


##########################################################
########## Warmup — force JIT before any timing ##########
##########################################################
# The first call to backward_euler_kernel_pb triggers Numba JIT compilation.
# We run a single step on a tiny batch (1 reactor) so all subsequent calls
# (RR eval, integration, scalability) use the already-compiled kernel.

print()
print(80 * '=')
print('Warmup (JIT compilation)')

_wm_n_sp   = mech.n_species
_wm_TC0    = np.zeros((1, _wm_n_sp + 1), dtype=dtype_np)
_wm_TC0[0, 0] = 1500.0
gas.set_equivalence_ratio(1.0, "CH4", "O2:1.0, N2:3.76")
gas.TP = 1500.0, 101325.0
_wm_TC0[0, 1:] = gas.concentrations * 1e3   # mol/m³

_wm_integrator = CUDABackwardEulerIntegratorPB(
    mechanism      = mech,
    n_reactors     = 1,
    newton_tol     = 1e-6,
    newton_maxiter = 20,
    gmres_restart  = 20,
    gmres_tol      = 1e-6,
    gmres_maxiter  = 50,
    dtype          = dtype_np,
)
_wm_d_TC = cuda.to_device(_wm_TC0)

_t_wm = time.perf_counter()
_wm_d_TC = _wm_integrator.step(_wm_d_TC, h=1e-7)
cuda.synchronize()
_t_wm = time.perf_counter() - _t_wm

print(f'	Warmup completed in {_t_wm:.3f} s  '
      f'(JIT + 1 step on 1 reactor)')
del _wm_integrator, _wm_d_TC, _wm_TC0

# Also warm up the eval kernel
_wm_eval = CUDAReactionEvaluator(mech, dtype=dtype_np)
_wm_T = np.array([1500.0], dtype=dtype_np)
_wm_C = np.zeros((1, mech.n_species), dtype=dtype_np)
gas.TP = 1500.0, 101325.0
_wm_C[0] = gas.concentrations * 1e3
_wm_eval.evaluate(_wm_T, _wm_C)
del _wm_eval, _wm_T, _wm_C
print(f'	Eval kernel warmed up.')


###############################################
########## reaction rates comparison ##########
###############################################
if RR_and_source_terms:

    TC_rand = np.random.random((size, len(vec_th.species) + 1))
    TC_rand[:, 0] = np.random.uniform(low=500, high=5000, size=size)

    T_val = TC_rand[:, 0].astype(dtype_np)      # (size,)
    C_val = TC_rand[:, 1:].astype(dtype_np)      # (size, n_sp) — random, not physical

    # ── Cantera reference ──────────────────────────────────────────────
    # Convention : le script torch passe C en mol/m³ × 1e-3 à Cantera
    # (Cantera attend kmol/m³).  On reproduit exactement ce comportement.
    dC_cantera   = np.zeros((size, len(vec_th.species)))
    RRs_can_elem = []
    RRs_can_tb   = []
    RRs_can_f    = []

    for it in range(size):
        gas.TP = float(T_val[it]), 101325.0
        gas.concentrations = C_val[it] * 1e-3   # mol/m³ → kmol/m³
        RRs_can_elem.append([])
        RRs_can_tb.append([])
        RRs_can_f.append([])
        for i, rtype in enumerate(reactions_index_to_type):
            RR_can = gas.net_rates_of_progress[i] * 1e3   # kmol/m³/s → mol/m³/s
            if rtype == 'elementary':
                RRs_can_elem[it].append(RR_can)
            elif rtype == 'threebody':
                RRs_can_tb[it].append(RR_can)
            elif 'falloff' in rtype:
                RRs_can_f[it].append(RR_can)
        dC_cantera[it] = gas.net_production_rates * 1e3

    RRs_can_elem = np.array(RRs_can_elem)   # (size, n_e)
    RRs_can_tb   = np.array(RRs_can_tb)     # (size, n_tb)
    RRs_can_f    = np.array(RRs_can_f)      # (size, n_f)

    # ── CUDA evaluation ────────────────────────────────────────────────
    evaluator = CUDAReactionEvaluator(mech, dtype=dtype_np)
    # C_val est en mol/m³ — même convention que les kernels CUDA
    RR_e_gpu, RR_tb_gpu, RR_f_gpu, omega_gpu = evaluator.evaluate(T_val, C_val)

    # ── Metrics (identiques au script torch) ──────────────────────────
    elem_MAE, elem_MAE_std, elem_MAE_err = MAE(RRs_can_elem, RR_e_gpu)
    elem_MSE, elem_MSE_std, elem_MSE_err = MSE(RRs_can_elem, RR_e_gpu)

    tb_MAE, tb_MAE_std, tb_MAE_err = MAE(RRs_can_tb, RR_tb_gpu)
    tb_MSE, tb_MSE_std, tb_MSE_err = MSE(RRs_can_tb, RR_tb_gpu)

    f_MAE, f_MAE_std, f_MAE_err = MAE(RRs_can_f, RR_f_gpu)
    f_MSE, f_MSE_std, f_MSE_err = MSE(RRs_can_f, RR_f_gpu)

    omega_MAE, omega_MAE_std, omega_MAE_err = MAE(dC_cantera, omega_gpu)
    omega_MSE, omega_MSE_std, omega_MSE_err = MSE(dC_cantera, omega_gpu)

    # ── Figures ───────────────────────────────────────────────────────
    plot_hist(
        [elem_MAE_err, tb_MAE_err, f_MAE_err, omega_MAE_err],
        ["elementary\nreactions\nMAE", "third body\nreactions\nMAE",
         "falloff\nreactions\nMAE", "species\nproduction\nrate\nMAE"],
        'article_figures/RR_MAE.png')
    plot_hist(
        [elem_MSE_err, tb_MSE_err, f_MSE_err, omega_MSE_err],
        ["elementary\nreactions\nMSE", "third body\nreactions\nMSE",
         "falloff\nreactions\nMSE", "species\nproduction\nrate\nMSE"],
        'article_figures/RR_MSE.png')

    # ── Prints (identiques au script torch) ───────────────────────────
    print()
    print(80 * '=')
    print('GPU model validation')
    print()
    print('\tElementary reactions Reaction Rate:')
    print(f'\t\tMean absolute error: {elem_MAE:.6e} +- {elem_MAE_std:.6e} mol/m^3')
    print(f'\t\tMean squared error: {elem_MSE:.6e} +- {elem_MSE_std:.6e} mol/m^3')
    print()
    print('\tThird body reactions Reaction Rate:')
    print(f'\t\tMean absolute error: {tb_MAE:.6e} +- {tb_MAE_std:.6e} mol/m^3')
    print(f'\t\tMean squared error: {tb_MSE:.6e} +- {tb_MSE_std:.6e} mol/m^3')
    print()
    print('\tFalloff reactions Reaction Rate:')
    print(f'\t\tMean absolute error: {f_MAE:.6e} +- {f_MAE_std:.6e} mol/m^3')
    print(f'\t\tMean squared error: {f_MSE:.6e} +- {f_MSE_std:.6e} mol/m^3')
    print()
    print('\tCumulative species production rate (omega):')
    print(f'\t\tMean absolute error: {omega_MAE:.6e} +- {omega_MAE_std:.6e} mol/m^3')
    print(f'\t\tMean squared error: {omega_MSE:.6e} +- {omega_MSE_std:.6e} mol/m^3')


############################################
########## Integration comparison ##########
############################################
if Integrators:

    # ── Reference models ──────────────────────────────────────────────

    def cantera_constant_volume(T0, C0_mol, t_ref):
        """T0: float, C0_mol: (n_sp,) mol/m³."""
        gas_tmp = ct.Solution(yaml_file)
        gas_tmp.TP = float(T0), 101325.0
        gas_tmp.concentrations = C0_mol * 1e-3   # mol→kmol
        reactor = ct.IdealGasReactor(gas_tmp, energy='on')
        sim = ct.ReactorNet([reactor])
        times, T_can, P_can, X_can = [], [], [], []
        while sim.time < t_ref[-1]:
            sim.step()
            times.append(sim.time)
            T_can.append(reactor.T)
            P_can.append(reactor.thermo.P)
            X_can.append(reactor.thermo.X.copy())
        return interpolate_trajectory(
            np.array(times), np.array(T_can),
            np.array(P_can), np.array(X_can), t_ref)

    def CPU_model_constant_volume(T0, C0_mol, t_ref):
        """T0: float, C0_mol: (n_sp,) mol/m³."""
        gas_tmp = ct.Solution(yaml_file)

        def rhs_adiabatic(t, TC):
            T = TC[0]; C = TC[1:]
            gas_tmp.TP = T, 101325.0
            gas_tmp.concentrations = C * 1e-3          # mol→kmol
            dCdt = gas_tmp.net_production_rates * 1e3  # kmol→mol
            dTdt = (-np.dot(gas_tmp.partial_molar_int_energies,
                            gas_tmp.net_production_rates * 1e3)
                    / (gas_tmp.density_mass * gas_tmp.cv_mass))
            return np.concatenate(([dTdt], dCdt))

        Y0  = np.concatenate(([float(T0)], C0_mol))
        sol = solve_ivp(rhs_adiabatic, [0, t_ref[-1]], Y0,
                        method='BDF', atol=1e-12, rtol=1e-6)
        C_sol = sol.y[1:, :].T
        X_sol = C_sol / np.maximum(C_sol.sum(axis=1, keepdims=True), 1e-300)
        T_sol = sol.y[0, :]
        C_tot = C_sol.sum(axis=1)
        P_sol = 8.314_462_618_153_24 * T_sol * C_tot
        return interpolate_trajectory(sol.t, T_sol, P_sol, X_sol, t_ref)

    def GPU_model_constant_volume(integrator, d_TC0, t_ref, h):
        """
        integrator : CUDABackwardEulerIntegratorPB
        d_TC0      : numba device array (B, n_state), C en mol/m³
        """
        n_steps = len(t_ref)
        B       = d_TC0.shape[0]
        n_sp    = d_TC0.shape[1] - 1
        R_C     = 8.314_462_618_153_24

        d_TC   = d_TC0
        T_hist = np.zeros((B, n_steps))
        C_hist = np.zeros((B, n_steps, n_sp))

        for i in range(n_steps):
            d_TC  = integrator.step(d_TC, h)
            TC_np = d_TC.copy_to_host()
            T_hist[:, i]    = TC_np[:, 0]
            C_hist[:, i, :] = TC_np[:, 1:]

        C_tot = C_hist.sum(axis=2)
        X     = C_hist / np.maximum(C_tot[:, :, None], 1e-300)
        P     = R_C * T_hist * C_tot
        return T_hist, P, X

    def run_sequential_sim(model_fn, T_arr, C_arr, t_ref):
        res = {'T': [], 'P': [], 'X': []}
        for T0, C0 in zip(T_arr, C_arr):
            T, P, X = model_fn(float(T0), C0, t_ref)
            res['T'].append(T); res['P'].append(P); res['X'].append(X)
        res['T'] = np.stack(res['T'])
        res['P'] = np.stack(res['P'])
        res['X'] = np.stack(res['X'])
        return res

    def run_batched_sim(model_fn, *args):
        T, P, X = model_fn(*args)
        return {'T': T, 'P': P, 'X': X}

    # ── Time grid ─────────────────────────────────────────────────────
    t_end = 3e-4
    h     = 1e-7
    t_ref = np.arange(0.0, t_end, h)

    T0s  = np.random.uniform(low=1800, high=2200, size=size - 1)
    C0s_kmol = []
    phis = np.random.uniform(low=0.5, high=1.5, size=size - 1)
    T0_ref  = 1700.0
    phi_ref = 1.0

    for phi, T0 in zip(phis, T0s):
        gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
        gas.TP = T0, 101325.0
        C0s_kmol.append(gas.concentrations.copy())   # kmol/m³

    C0s_kmol = np.array(C0s_kmol)

    gas.set_equivalence_ratio(phi_ref, "CH4", "O2:1.0, N2:3.76")
    gas.TP = T0_ref, 101325.0
    C0_ref_kmol = gas.concentrations.copy()

    # All reactors, reference at index 0 — keep kmol/m³ for Cantera/CPU,
    # convert to mol/m³ only for CUDA (mirrors the torch script's C * 1e3)
    T_all     = np.concatenate([[T0_ref], T0s])                        # K
    C_all_mol = np.vstack([C0_ref_kmol[None, :], C0s_kmol]) * 1e3     # mol/m³

    # ── Cantera ───────────────────────────────────────────────────────
    print()
    print(80 * '=')
    print('Numerical integration validation')

    print()
    print('\tCantera model:')
    t0 = time.perf_counter()
    cant_res = run_sequential_sim(
        cantera_constant_volume, T_all, C_all_mol, t_ref)
    t_comp = time.perf_counter() - t0
    res_T, res_P, res_X = cant_res['T'], cant_res['P'], cant_res['X']
    print(f'\t\tcomputation time: {t_comp:.3f} seconds')
    print(f'\t\tobject shapes: T.shape = {res_T.shape}, '
          f'P.shape = {res_P.shape}, X.shape = {res_X.shape}')

    # ── CPU BDF ───────────────────────────────────────────────────────
    print()
    print('\tCPU model:')
    t0 = time.perf_counter()
    CPU_res = run_sequential_sim(
        CPU_model_constant_volume, T_all, C_all_mol, t_ref)
    t_comp = time.perf_counter() - t0
    res_T, res_P, res_X = CPU_res['T'], CPU_res['P'], CPU_res['X']
    print(f'\t\tcomputation time: {t_comp:.3f} seconds')
    print(f'\t\tobject shapes: T.shape = {res_T.shape}, '
          f'P.shape = {res_P.shape}, X.shape = {res_X.shape}')

    # ── GPU CUDA ──────────────────────────────────────────────────────
    print()
    print('\tGPU model:')

    integrator = CUDABackwardEulerIntegratorPB(
        mechanism      = mech,
        n_reactors     = size,
        newton_tol     = 1e-6,
        newton_maxiter = 20,
        gmres_restart  = 20,
        gmres_tol      = 1e-6,
        gmres_maxiter  = 50,
        dtype          = dtype_np,
    )

    TC0 = np.zeros((size, mech.n_species + 1), dtype=dtype_np)
    TC0[:, 0]  = T_all
    TC0[:, 1:] = C_all_mol                 # mol/m³
    d_TC0 = cuda.to_device(TC0)

    t0 = time.perf_counter()
    GPU_res = run_batched_sim(
        GPU_model_constant_volume, integrator, d_TC0, t_ref, h)
    t_comp = time.perf_counter() - t0
    res_T, res_P, res_X = GPU_res['T'], GPU_res['P'], GPU_res['X']
    print(f'\t\tcomputation time: {t_comp:.3f} seconds')
    print(f'\t\tobject shapes: T.shape = {res_T.shape}, '
          f'P.shape = {res_P.shape}, X.shape = {res_X.shape}')

    # ── Metrics — reactor 0 only ──────────────────────────────────────
    CPU_MAE, CPU_MAE_std, CPU_MAE_err = MAE(cant_res['T'][0], CPU_res['T'][0])
    CPU_MSE, CPU_MSE_std, CPU_MSE_err = MSE(cant_res['T'][0], CPU_res['T'][0])

    GPU_MAE, GPU_MAE_std, GPU_MAE_err = MAE(cant_res['T'][0], GPU_res['T'][0])
    GPU_MSE, GPU_MSE_std, GPU_MSE_err = MSE(cant_res['T'][0], GPU_res['T'][0])

    # ── Figures ───────────────────────────────────────────────────────
    plots_vs_time(
        t_ref,
        [cant_res['T'][0], CPU_res['T'][0], GPU_res['T'][0]],
        ['Cantera', 'CPU model', 'GPU model'],
        'article_figures/T_trajectory.png')

    dT_cant = np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0])

    plot_vs_time(t_ref, CPU_MAE_err, 'article_figures/CPU_MAE.png')
    plot_vs_time(t_ref, CPU_MSE_err, 'article_figures/CPU_MSE.png')
    scatter_vs_time(dT_cant, CPU_MSE_err, 'article_figures/CPU_MSE_vs_dT.png')
    scatter_vs_time(dT_cant, CPU_MAE_err, 'article_figures/CPU_MAE_vs_dT.png')
    colored_scatter_vs_time(t_ref, CPU_MSE_err, dT_cant,
                            'article_figures/colored_CPU_MSE_vs_time.png')
    colored_scatter_vs_time(t_ref, CPU_MAE_err, dT_cant,
                            'article_figures/colored_CPU_MAE_vs_time.png')

    plot_vs_time(t_ref, GPU_MAE_err, 'article_figures/GPU_MAE.png')
    plot_vs_time(t_ref, GPU_MSE_err, 'article_figures/GPU_MSE.png')
    scatter_vs_time(dT_cant, GPU_MSE_err, 'article_figures/GPU_MSE_vs_dT.png')
    scatter_vs_time(dT_cant, GPU_MAE_err, 'article_figures/GPU_MAE_vs_dT.png')
    colored_scatter_vs_time(t_ref, GPU_MSE_err, dT_cant,
                            'article_figures/colored_GPU_MSE_vs_time.png')
    colored_scatter_vs_time(t_ref, GPU_MAE_err, dT_cant,
                            'article_figures/colored_GPU_MAE_vs_time.png')



######################################
########## Scalability test ##########
######################################
if Scalability:

    print()
    print(80 * '=')
    print('Scalability test')
    print(f'\tSweeping N_reactors = {N_reactors_list}')
    print(f'\tN_steps per run     = {N_STEPS_SCALABILITY}  (h = 1e-7 s)')
    print()

    # Fixed reference conditions — same phi/T0 replicated across all reactors
    # so we don't penalise Cantera with harder initial conditions.
    gas.set_equivalence_ratio(1.0, "CH4", "O2:1.0, N2:3.76")
    gas.TP = 1700.0, 101325.0
    _C_ref_mol = gas.concentrations * 1e3   # mol/m³
    _T_ref     = 1700.0

    times_cantera_sc = []
    times_gpu_sc     = []
    valid_N          = []

    for N in N_reactors_list:

        # ── Build identical initial conditions for all N reactors ─────────
        TC0_sc = np.zeros((N, mech.n_species + 1), dtype=dtype_np)
        TC0_sc[:, 0]  = _T_ref
        TC0_sc[:, 1:] = _C_ref_mol[None, :]

        # ── Cantera (sequential over N reactors, N_STEPS_SCALABILITY steps) ─
        # We advance each reactor by N_STEPS_SCALABILITY * h seconds.
        _t_end_sc = N_STEPS_SCALABILITY * 1e-7

        t0 = time.perf_counter()
        for _ in range(N):
            gas_sc = ct.Solution(yaml_file)
            gas_sc.TP = _T_ref, 101325.0
            gas_sc.concentrations = _C_ref_mol * 1e-3   # kmol/m³
            reactor_sc = ct.IdealGasReactor(gas_sc, energy='on')
            sim_sc     = ct.ReactorNet([reactor_sc])
            sim_sc.advance(_t_end_sc)
        t_cantera = time.perf_counter() - t0
        times_cantera_sc.append(t_cantera)

        # ── GPU CUDA (batched, N_STEPS_SCALABILITY steps) ─────────────────
        intgr_sc = CUDABackwardEulerIntegratorPB(
            mechanism      = mech,
            n_reactors     = N,
            newton_tol     = 1e-6,
            newton_maxiter = 20,
            gmres_restart  = 20,
            gmres_tol      = 1e-6,
            gmres_maxiter  = 50,
            dtype          = dtype_np,
        )
        d_TC_sc = cuda.to_device(TC0_sc)

        t0 = time.perf_counter()
        for _ in range(N_STEPS_SCALABILITY):
            d_TC_sc = intgr_sc.step(d_TC_sc, 1e-7)
        cuda.synchronize()   # make sure all GPU work is done before stopping clock
        t_gpu = time.perf_counter() - t0
        times_gpu_sc.append(t_gpu)

        del intgr_sc, d_TC_sc

        valid_N.append(N)
        print(f'\tN = {N:>5d}  |  Cantera: {t_cantera:8.3f} s  |  GPU: {t_gpu:8.4f} s  |  '
              f'speedup: {t_cantera / t_gpu if t_gpu > 0 else float("inf"):7.1f}x')

    # ── Figure ────────────────────────────────────────────────────────────
    valid_N          = np.array(valid_N)
    times_cantera_sc = np.array(times_cantera_sc)
    times_gpu_sc     = np.array(times_gpu_sc)
    speedup_sc       = times_cantera_sc / np.maximum(times_gpu_sc, 1e-12)

    # Dual-axis plot: wall time (left) + speedup (right)
    fig, ax1 = plt.subplots()

    color_can = plt.cm.tab10.colors[0]   # blue
    color_gpu = plt.cm.tab10.colors[1]   # orange
    color_spd = plt.cm.tab10.colors[2]   # green

    ax1.loglog(valid_N, times_cantera_sc, 'o-',
               color=color_can, label='Cantera (sequential)')
    ax1.loglog(valid_N, times_gpu_sc,     's-',
               color=color_gpu, label='GPU CUDA (batched)')
    ax1.set_xlabel('Number of reactors')
    ax1.set_ylabel('Wall time (s)')

    ax2 = ax1.twinx()
    ax2.loglog(valid_N, speedup_sc, 'D--',
               color=color_spd, linewidth=0.6, label='Speedup')
    ax2.set_ylabel('Speedup (Cantera / GPU)')
    ax2.tick_params(axis='y', colors=color_spd)
    ax2.yaxis.label.set_color(color_spd)

    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    fig.savefig('article_figures/scalability.png', bbox_inches='tight')
    plt.close(fig)

    print()
    print('\tFigure saved to article_figures/scalability.png')


if __name__ == '__main__':
    os.makedirs('article_figures', exist_ok=True)