"""
test_cuda_integrator_full.py — validation et benchmarks GPU pour l'article Space Propulsion.

Sections:
  1. Warmup          — force la compilation JIT avant toute mesure
  2. RR_and_source   — validation taux de réaction vs Cantera (MAE/MSE par type)
  3. Integrators     — comparaison Cantera / GPU per-block sur 1000 réacteurs
  4. Architecture    — comparaison per-thread vs per-block (speedup vs Cantera)
  5. Scalability     — speedup GPU per-block vs Cantera en fonction de N_reactors
  6. Error_analysis  — analyse détaillée des erreurs (corrélation avec dT/dt, stats)
"""

import numpy as np
import math
import time
import sys, os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.claude.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.claude.integrator.cuda_integrator_pb  import CUDABackwardEulerIntegratorPB
from kineticsGPU.claude.integrator.cuda_integrator     import CUDABackwardEulerIntegrator
from kineticsGPU.claude.kernels.chemistry_kernels_article import CUDAReactionEvaluator
import numba.cuda as cuda

import cantera as ct
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import interp1d
import torch
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Flags
# ─────────────────────────────────────────────────────────────────────────────
RR_and_source_terms  = False
Integrators          = False
Architecture_compare = False
Scalability          = False
Error_analysis       = False
JVP_validation       = False
Jacobian_validation  = True



N_reactors_list      = [10, 100, 500]#[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
N_STEPS_SCALABILITY  = 3000

SIZE        = 1000
T_END       = 3e-4
H_DEFAULT   = 1e-7
H_LIST      = [5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 2e-8, 1e-8, 5e-9, 2e-9]

device = torch.device('cuda:0')
cuda.select_device(device.index)

os.makedirs('article_figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib style
# ─────────────────────────────────────────────────────────────────────────────
custom_rc = {
    'text.usetex': True,
    'figure.figsize': (3.5, 2.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
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
    'axes.formatter.limits': (-4, 4),
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'lines.linewidth': 0.9,
    'lines.markersize': 2.5,
    'legend.frameon': False,
    'axes.prop_cycle': plt.cycler('color', plt.cm.tab10.colors),
}
matplotlib.rcParams.update(custom_rc)
COLORS = plt.cm.tab10.colors

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def mae(A, B):
    e = np.abs(A - B)
    return e.mean(), e.std(), e

def mse(A, B):
    e = (A - B)**2
    return e.mean(), e.std(), e

def interp_traj(times, T, P, X, t_ref):
    T_i = interp1d(times, T, fill_value='extrapolate')(t_ref)
    P_i = interp1d(times, P, fill_value='extrapolate')(t_ref)
    X_i = np.column_stack([
        interp1d(times, X[:, k], fill_value='extrapolate')(t_ref)
        for k in range(X.shape[1])])
    return T_i, P_i, X_i

def plot_box(data_list, labels, path, ylabel='Error', log=True):
    fig, ax = plt.subplots()
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
        ax.yaxis.set_minor_locator(mticker.LogLocator(
            numticks=999, subs=(.2,.4,.6,.8)))
    ax.boxplot([d.ravel() for d in data_list], labels=labels,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='black'),
               boxprops=dict(facecolor=COLORS[0], alpha=0.6))
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def colored_scatter(x, y, c, path, xlabel='', ylabel='', clabel=''):
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=c, s=4, cmap='plasma', rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def ignition_delay(T_traj, t_ref, T_thresh=500):
    T0  = T_traj[0]
    idx = np.argmax(T_traj > T0 + T_thresh)
    return t_ref[idx] if idx > 0 else np.nan

# ─────────────────────────────────────────────────────────────────────────────
# Mechanism loading
# ─────────────────────────────────────────────────────────────────────────────
yaml_file = '../gri30.yaml'
dtype_np  = np.float64

y      = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=torch.float64, device='cuda')
rxns   = parseReactions(yaml_file)
gas    = ct.Solution(yaml_file)
r_e, r_tb, r_f, _ = build_reactions(vec_th.species, rxns, vec_th, device,
                                     torch.float64)
mech   = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype_np)
reactions_index_to_type = [r['type'] for r in rxns]

n_sp  = mech.n_species
R_GAS = 8.314_462_618_153_24

print(f'\nMechanism: {n_sp} species, '
      f'{mech.elementary.n_reactions} elementary, '
      f'{mech.threebody.n_reactions} three-body, '
      f'{mech.falloff.n_reactions} falloff reactions\n')

# ─────────────────────────────────────────────────────────────────────────────
# Warmup
# ─────────────────────────────────────────────────────────────────────────────
print('='*70)
print('Warmup (JIT compilation)')

gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')
gas.TP = 1700.0, 101325.0
_C_ref_mol = gas.concentrations * 1e3

_wm_intgr = CUDABackwardEulerIntegratorPB(mech, 1, dtype=dtype_np)
_wm_TC = cuda.to_device(
    np.concatenate([[1700.0], _C_ref_mol])[None, :].astype(dtype_np))
t0 = time.perf_counter()
_wm_intgr.step(_wm_TC, H_DEFAULT)
cuda.synchronize()
print(f'  per-block JIT: {time.perf_counter()-t0:.2f} s')
del _wm_intgr, _wm_TC

_wm_intgr2 = CUDABackwardEulerIntegrator(mech, 1, dtype=dtype_np)
_wm_TC2 = cuda.to_device(
    np.concatenate([[1700.0], _C_ref_mol])[None, :].astype(dtype_np))
_wm_intgr2.step(H_DEFAULT)
cuda.synchronize()
print(f'  per-thread JIT: done')
del _wm_intgr2, _wm_TC2

_wm_ev = CUDAReactionEvaluator(mech, dtype=dtype_np)
_wm_ev.evaluate(np.array([1700.0]), _C_ref_mol[None, :])
print('  eval kernel: done')
del _wm_ev

# ─────────────────────────────────────────────────────────────────────────────
# 1. Reaction rates and source terms validation
# ─────────────────────────────────────────────────────────────────────────────
if RR_and_source_terms:
    print('\n' + '='*70)
    print('Section 1 — Reaction rate and source term validation')

    np.random.seed(42)
    T_val = np.random.uniform(500, 5000, SIZE).astype(dtype_np)
    C_val = np.random.uniform(0, 1, (SIZE, n_sp)).astype(dtype_np)

    RRs_e_can, RRs_tb_can, RRs_f_can = [], [], []
    omega_can = np.zeros((SIZE, n_sp), dtype=dtype_np)
    for i in range(SIZE):
        gas.TP = float(T_val[i]), 101325.0
        gas.concentrations = C_val[i] * 1e-3
        row_e, row_tb, row_f = [], [], []
        for j, rtype in enumerate(reactions_index_to_type):
            RR = gas.net_rates_of_progress[j] * 1e3
            if   rtype == 'elementary': row_e.append(RR)
            elif rtype == 'threebody':  row_tb.append(RR)
            elif 'falloff' in rtype:    row_f.append(RR)
        RRs_e_can.append(row_e)
        RRs_tb_can.append(row_tb)
        RRs_f_can.append(row_f)
        omega_can[i] = gas.net_production_rates * 1e3
    RRs_e_can  = np.array(RRs_e_can)
    RRs_tb_can = np.array(RRs_tb_can)
    RRs_f_can  = np.array(RRs_f_can)

    evaluator = CUDAReactionEvaluator(mech, dtype=dtype_np)
    RR_e_gpu, RR_tb_gpu, RR_f_gpu, omega_gpu = evaluator.evaluate(T_val, C_val)

    e_MAE,  _, e_MAE_err  = mae(RRs_e_can,  RR_e_gpu)
    tb_MAE, _, tb_MAE_err = mae(RRs_tb_can, RR_tb_gpu)
    f_MAE,  _, f_MAE_err  = mae(RRs_f_can,  RR_f_gpu)
    om_MAE, _, om_MAE_err = mae(omega_can,  omega_gpu)

    e_MSE,  _, e_MSE_err  = mse(RRs_e_can,  RR_e_gpu)
    tb_MSE, _, tb_MSE_err = mse(RRs_tb_can, RR_tb_gpu)
    f_MSE,  _, f_MSE_err  = mse(RRs_f_can,  RR_f_gpu)
    om_MSE, _, om_MSE_err = mse(omega_can,  omega_gpu)

    print(f'  Elementary  MAE = {e_MAE:.3e}  MSE = {e_MSE:.3e}')
    print(f'  Three-body  MAE = {tb_MAE:.3e}  MSE = {tb_MSE:.3e}')
    print(f'  Falloff     MAE = {f_MAE:.3e}  MSE = {f_MSE:.3e}')
    print(f'  omega       MAE = {om_MAE:.3e}  MSE = {om_MSE:.3e}')

    plot_box(
        [e_MAE_err, tb_MAE_err, f_MAE_err, om_MAE_err],
        ['Elementary\nRR', 'Third body\nRR', 'Falloff\nRR', r'$\dot\omega$'],
        'article_figures/RR_MAE.png', ylabel='MAE (mol m$^{-3}$ s$^{-1}$)')
    plot_box(
        [e_MSE_err, tb_MSE_err, f_MSE_err, om_MSE_err],
        ['Elementary\nRR', 'Third body\nRR', 'Falloff\nRR', r'$\dot\omega$'],
        'article_figures/RR_MSE.png', ylabel='MSE (mol$^2$ m$^{-6}$ s$^{-2}$)')

    e_rel  = np.abs(RRs_e_can  - RR_e_gpu)  / (np.abs(RRs_e_can)  + 1e-100)
    tb_rel = np.abs(RRs_tb_can - RR_tb_gpu) / (np.abs(RRs_tb_can) + 1e-100)
    f_rel  = np.abs(RRs_f_can  - RR_f_gpu)  / (np.abs(RRs_f_can)  + 1e-100)
    om_rel = np.abs(omega_can  - omega_gpu)  / (np.abs(omega_can)  + 1e-100)

    plot_box(
        [e_rel, tb_rel, f_rel, om_rel],
        ['Elementary\nRR', 'Third body\nRR', 'Falloff\nRR', r'$\dot\omega$'],
        'article_figures/RR_relative_error.png',
        ylabel='Relative error (--)')
    print('  Figures: RR_MAE, RR_MSE, RR_relative_error')

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_cantera_CV(T0, C0_mol, t_ref, h):
    g = ct.Solution(yaml_file)
    g.TP = float(T0), 101325.0
    g.concentrations = C0_mol * 1e-3
    r = ct.IdealGasReactor(g, energy='on')
    sim = ct.ReactorNet([r])
    ts, Ts, Ps, Xs = [], [], [], []
    while sim.time < t_ref[-1]:
        sim.advance(sim.time + h)
        ts.append(sim.time); Ts.append(r.T)
        Ps.append(r.thermo.P); Xs.append(r.thermo.X.copy())
    return interp_traj(np.array(ts), np.array(Ts),
                       np.array(Ps), np.array(Xs), t_ref)

def run_gpu_CV(integrator, d_TC0, t_ref, h):
    n_steps = len(t_ref)
    B, nst  = d_TC0.shape[0], d_TC0.shape[1] - 1
    d_TC    = d_TC0
    T_hist  = np.zeros((B, n_steps))
    C_hist  = np.zeros((B, n_steps, nst))
    for i in range(n_steps):
        d_TC, _ = integrator.step(d_TC, h)
        TC = d_TC.copy_to_host()
        T_hist[:, i]    = TC[:, 0]
        C_hist[:, i, :] = TC[:, 1:]
    C_tot = C_hist.sum(2)
    X = C_hist / np.maximum(C_tot[:, :, None], 1e-300)
    P = R_GAS * T_hist * C_tot
    return T_hist, P, X

# ─────────────────────────────────────────────────────────────────────────────
# 2. Integration validation
# ─────────────────────────────────────────────────────────────────────────────
if Integrators:
    print('\n' + '='*70)
    print('Section 2 — Integration validation')

    np.random.seed(0)
    t_ref  = np.arange(H_DEFAULT, T_END + H_DEFAULT, H_DEFAULT)
    T0_ref = 1700.0; phi_ref = 1.0

    T0s  = np.random.uniform(1500, 2000, SIZE - 1)
    phis = np.random.uniform(0.5,  1.5,  SIZE - 1)
    C0s  = []
    for phi, T0 in zip(phis, T0s):
        gas.set_equivalence_ratio(phi, 'CH4', 'O2:1.0, N2:3.76')
        gas.TP = T0, 101325.0
        C0s.append(gas.concentrations * 1e3)
    gas.set_equivalence_ratio(phi_ref, 'CH4', 'O2:1.0, N2:3.76')
    gas.TP = T0_ref, 101325.0
    C0_ref = gas.concentrations * 1e3

    T_all = np.concatenate([[T0_ref], T0s])
    C_all = np.vstack([C0_ref[None, :], np.array(C0s)])

    # Cantera
    print('  Running Cantera ...', end=' ', flush=True)
    t0 = time.perf_counter()
    cant_T = []; cant_X = []; cant_P = []
    for i in range(SIZE):
        T, P, X = run_cantera_CV(T_all[i], C_all[i], t_ref, H_DEFAULT)
        cant_T.append(T); cant_P.append(P); cant_X.append(X)
    t_can = time.perf_counter() - t0
    cant_T = np.array(cant_T); cant_P = np.array(cant_P); cant_X = np.array(cant_X)
    print(f'{t_can:.1f} s ({1e6*t_can/SIZE:.0f} µs/reactor)')

    # GPU per-block
    print('  Running GPU per-block ...', end=' ', flush=True)
    intgr_pb = CUDABackwardEulerIntegratorPB(mech, SIZE, dtype=dtype_np)
    TC0 = np.column_stack([T_all, C_all]).astype(dtype_np)
    d_TC0 = cuda.to_device(TC0)
    t0 = time.perf_counter()
    gpu_T, gpu_P, gpu_X = run_gpu_CV(intgr_pb, d_TC0, t_ref, H_DEFAULT)
    t_gpu = time.perf_counter() - t0
    print(f'{t_gpu:.2f} s  |  speedup vs Cantera: {t_can/t_gpu:.1f}x')
    del intgr_pb

    # Figure 1 : trajectoire température réacteur référence
    fig, ax = plt.subplots()
    #ax.plot(t_ref*1e6, cant_T[0], label='Cantera', color=COLORS[0])
    #ax.plot(t_ref*1e6, gpu_T[0],  label='GPU (BE)', color=COLORS[2], linestyle=':', linewidth=0.7)
    #ax.set_xlabel(r'Time ($\mu$s)')
    #ax.set_ylabel('Temperature (K)')
    dT_dt_ref = np.diff(cant_T[0], prepend=cant_T[0][0]) / H_DEFAULT
    sc = ax.scatter(gpu_T[0], cant_T[0], c=np.abs(dT_dt_ref), s=4, marker='o', cmap='viridis', rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r'$|dT/dt|$ (K/s)')
    ax.set_xlabel(r'$T_{\rm GPU}$ (K)')
    ax.set_ylabel(r'$T_{\rm Cantera}$ (K)')
    ax.axline((gpu_T[0][0], cant_T[0][0]), slope=1, color='red', linewidth=1, linestyle='--', label='x=y')
    ax.legend()
    plt.tight_layout()
    plt.savefig('article_figures/T_trajectory.png', bbox_inches='tight')
    plt.close()

    # Figure 2 : MAE GPU vs time, colorée par dT/dt
    dT_dt_ref = np.diff(cant_T[0], prepend=cant_T[0][0]) / H_DEFAULT
    gpu_mae_t = np.abs(cant_T[0] - gpu_T[0])
    colored_scatter(t_ref*1e6, gpu_mae_t, np.abs(dT_dt_ref),
                    'article_figures/colored_GPU_MAE_vs_time.png',
                    xlabel=r'Time ($\mu$s)', ylabel='MAE$_T$ (K)',
                    clabel=r'$|dT/dt|$ (K/s)')

    # Figure 3 : distribution MAE sur les 1000 réacteurs
    gpu_peak_mae = np.abs(cant_T - gpu_T).max(axis=1)
    gpu_rms_mae  = np.sqrt(((cant_T - gpu_T)**2).mean(axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    for ax, data, title in zip(axes,
        [gpu_peak_mae, gpu_rms_mae],
        ['Peak MAE (K)', 'RMS MAE (K)']):
        ax.boxplot([data], labels=['GPU BE'],
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='black'),
                   boxprops=dict(facecolor=COLORS[0], alpha=0.6))
        ax.set_title(title)
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('article_figures/MAE_stats_all_reactors.png', bbox_inches='tight')
    plt.close()

    # Figure 4 : délai d'allumage
    tau_can = np.array([ignition_delay(cant_T[i], t_ref) for i in range(SIZE)])
    tau_gpu = np.array([ignition_delay(gpu_T[i],  t_ref) for i in range(SIZE)])
    valid   = ~np.isnan(tau_can) & ~np.isnan(tau_gpu)

    if valid.sum() > 1:
        rel_err_gpu = np.abs(tau_gpu[valid] - tau_can[valid]) / tau_can[valid] * 100
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
        lim = [tau_can[valid].min()*1e6*0.9, tau_can[valid].max()*1e6*1.1]
        axes[0].scatter(tau_can[valid]*1e6, tau_gpu[valid]*1e6,
                        s=4, alpha=0.5, color=COLORS[2], label='GPU BE')
        axes[0].plot(lim, lim, 'k--', linewidth=0.5)
        axes[0].set_xlabel(r'$\tau_{\rm ig}$ Cantera ($\mu$s)')
        axes[0].set_ylabel(r'$\tau_{\rm ig}$ GPU ($\mu$s)')
        axes[1].boxplot([rel_err_gpu], labels=['GPU BE'],
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black'),
                        boxprops=dict(facecolor=COLORS[0], alpha=0.6))
        axes[1].set_ylabel(r'Relative error on $\tau_{\rm ig}$ (\%)')
        plt.tight_layout()
        plt.savefig('article_figures/ignition_delay_comparison.png', bbox_inches='tight')
        plt.close()
        print(f'  GPU BE  τ_ig rel. err: {rel_err_gpu.mean():.2f} ± {rel_err_gpu.std():.2f} %')
    else:
        print(f'  Ignition delay: {valid.sum()}/{SIZE} réacteurs allumés — augmente T_END ou baisse T_thresh')

    print(f'  GPU BE  peak MAE: {gpu_peak_mae.mean():.2e} ± {gpu_peak_mae.std():.2e} K')
    print(f'  GPU BE  RMS  MAE: {gpu_rms_mae.mean():.2e} ± {gpu_rms_mae.std():.2e} K')
    print(f'  Timing — Cantera: {t_can:.1f}s  GPU PB: {t_gpu:.2f}s')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Architecture comparison: per-thread vs per-block
# ─────────────────────────────────────────────────────────────────────────────
if Architecture_compare:
    print('\n' + '='*70)
    print('Section 3 — Per-thread vs per-block architecture comparison')

    N_arch_list  = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    N_STEPS_ARCH = 500

    gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')
    gas.TP = 1700.0, 101325.0
    C_ref = gas.concentrations * 1e3

    times_can_arch = []
    times_pt_arch  = []
    times_pb_arch  = []

    for N in N_arch_list:
        TC0_a = np.zeros((N, n_sp + 1), dtype=dtype_np)
        TC0_a[:, 0]  = 1700.0
        TC0_a[:, 1:] = C_ref

        # Cantera
        t0 = time.perf_counter()
        for _ in range(N):
            g2 = ct.Solution(yaml_file)
            g2.TP = 1700.0, 101325.0
            g2.concentrations = C_ref * 1e-3
            r2 = ct.IdealGasReactor(g2, energy='on')
            s2 = ct.ReactorNet([r2])
            for __ in range(N_STEPS_ARCH):
                s2.step()
        times_can_arch.append(time.perf_counter() - t0)

        # Per-thread
        intgr_pt = CUDABackwardEulerIntegrator(mech, N, dtype=dtype_np)
        intgr_pt._d_TC_a.copy_to_device(TC0_a)
        t0 = time.perf_counter()
        for _ in range(N_STEPS_ARCH):
            intgr_pt.step(H_DEFAULT)
        cuda.synchronize()
        times_pt_arch.append(time.perf_counter() - t0)
        del intgr_pt

        # Per-block
        intgr_pb2 = CUDABackwardEulerIntegratorPB(mech, N, dtype=dtype_np)
        d_TC_pb = cuda.to_device(TC0_a)
        t0 = time.perf_counter()
        for _ in range(N_STEPS_ARCH):
            d_TC_pb, _ = intgr_pb2.step(d_TC_pb, H_DEFAULT)
        cuda.synchronize()
        times_pb_arch.append(time.perf_counter() - t0)
        del intgr_pb2

        print(f'  N={N:>5d} | Cantera: {times_can_arch[-1]:.3f}s '
              f'| PT: {times_pt_arch[-1]:.4f}s '
              f'| PB: {times_pb_arch[-1]:.4f}s '
              f'| speedup PB: {times_can_arch[-1]/times_pb_arch[-1]:.1f}x '
              f'| speedup PT: {times_can_arch[-1]/times_pt_arch[-1]:.1f}x')

    times_can_arch = np.array(times_can_arch)
    times_pt_arch  = np.array(times_pt_arch)
    times_pb_arch  = np.array(times_pb_arch)

    fig, ax1 = plt.subplots()
    ax1.loglog(N_arch_list, times_can_arch, 'o-', color=COLORS[0],
               label='Cantera (seq.)')
    ax1.loglog(N_arch_list, times_pt_arch,  's-', color=COLORS[1],
               label='GPU per-thread')
    ax1.loglog(N_arch_list, times_pb_arch,  'D-', color=COLORS[2],
               label='GPU per-block')
    ax1.set_xlabel('Number of reactors')
    ax1.set_ylabel('Wall time (s)')

    ax2 = ax1.twinx()
    ax2.loglog(N_arch_list, times_can_arch / times_pt_arch, 's--',
               color=COLORS[1], alpha=0.5, linewidth=0.6, label='Speedup PT')
    ax2.loglog(N_arch_list, times_can_arch / times_pb_arch, 'D--',
               color=COLORS[2], alpha=0.5, linewidth=0.6, label='Speedup PB')
    ax2.set_ylabel('Speedup over Cantera')
    ax2.tick_params(axis='y', colors=COLORS[3])

    lines,  labels  = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig('article_figures/architecture_comparison.png', bbox_inches='tight')
    plt.close()
    print('  Figure: architecture_comparison.png')

# ─────────────────────────────────────────────────────────────────────────────
# 4. Scalability — per-block vs Cantera
# ─────────────────────────────────────────────────────────────────────────────
if Scalability:
    print('\n' + '='*70)
    print('Section 4 — Scalability (per-block GPU vs Cantera)')

    gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')
    gas.TP = 1700.0, 101325.0
    C_ref = gas.concentrations * 1e3

    times_can_sc = []
    times_gpu_sc = []
    t_end_sc     = N_STEPS_SCALABILITY * H_DEFAULT

    for N in N_reactors_list:
        TC0_sc = np.zeros((N, n_sp + 1), dtype=dtype_np)
        TC0_sc[:, 0]  = 1700.0
        TC0_sc[:, 1:] = C_ref

        t0 = time.perf_counter()
        for _ in range(N):
            g3 = ct.Solution(yaml_file)
            g3.TP = 1700.0, 101325.0
            g3.concentrations = C_ref * 1e-3
            r3 = ct.IdealGasReactor(g3, energy='on')
            s3 = ct.ReactorNet([r3])
            s3.advance(t_end_sc)
        times_can_sc.append(time.perf_counter() - t0)

        intgr_sc = CUDABackwardEulerIntegratorPB(mech, N, dtype=dtype_np)
        d_TC_sc  = cuda.to_device(TC0_sc)
        t0 = time.perf_counter()
        for _ in range(N_STEPS_SCALABILITY):
            d_TC_sc, _ = intgr_sc.step(d_TC_sc, H_DEFAULT)
        cuda.synchronize()
        times_gpu_sc.append(time.perf_counter() - t0)
        del intgr_sc

        spd = times_can_sc[-1] / times_gpu_sc[-1]
        print(f'  N={N:>6d} | Cantera: {times_can_sc[-1]:8.3f}s '
              f'| GPU: {times_gpu_sc[-1]:7.4f}s | speedup: {spd:6.1f}x')

    times_can_sc = np.array(times_can_sc)
    times_gpu_sc = np.array(times_gpu_sc)
    speedup_sc   = times_can_sc / np.maximum(times_gpu_sc, 1e-12)

    fig, ax1 = plt.subplots()
    ax1.loglog(N_reactors_list, times_can_sc, 'o-', color=COLORS[0],
               label='Cantera (sequential)')
    ax1.loglog(N_reactors_list, times_gpu_sc, 's-', color=COLORS[2],
               label='GPU per-block')
    ax1.set_xlabel('Number of reactors')
    ax1.set_ylabel('Wall time (s)')

    ax2 = ax1.twinx()
    ax2.loglog(N_reactors_list, speedup_sc, 'D--', color=COLORS[3],
               linewidth=0.6, label='Speedup')
    ax2.set_ylabel('Speedup', color=COLORS[3])
    ax2.tick_params(axis='y', colors=COLORS[3])
    ax2.yaxis.label.set_color(COLORS[3])

    lines,  labels  = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig('article_figures/scalability.png', bbox_inches='tight')
    plt.close()

    throughput = np.array(N_reactors_list) / times_gpu_sc
    fig, ax = plt.subplots()
    ax.semilogx(N_reactors_list, throughput, 'D-', color=COLORS[2])
    ax.set_xlabel('Number of reactors')
    ax.set_ylabel('Throughput (reactor / s)')
    ax.axhline(throughput.max(), color='k', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('article_figures/gpu_throughput.png', bbox_inches='tight')
    plt.close()
    print('  Figures: scalability.png, gpu_throughput.png')

# ─────────────────────────────────────────────────────────────────────────────
# 5. Step-size sensitivity
# ─────────────────────────────────────────────────────────────────────────────
if Error_analysis and Integrators:
    print('\n' + '='*70)
    print('Section 5 — Step-size sensitivity (GPU accuracy vs h)')

    gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')
    gas.TP = 1700.0, 101325.0
    C0_ref_single = gas.concentrations * 1e3

    peak_mae_h = []
    rms_mae_h  = []
    tau_err_h  = []

    for h_test in H_LIST:
        t_ref_h  = np.arange(h_test, T_END + h_test, h_test)
        T_can_ref, _, _ = run_cantera_CV(1700.0, C0_ref_single, t_ref_h, H_DEFAULT)

        intgr_h = CUDABackwardEulerIntegratorPB(mech, 1, dtype=dtype_np)
        TC0_h   = np.concatenate([[1700.0], C0_ref_single])[None, :].astype(dtype_np)
        d_h     = cuda.to_device(TC0_h)

        T_gpu_h = np.zeros(len(t_ref_h))
        for i in range(len(t_ref_h)):
            d_h, _ = intgr_h.step(d_h, h_test)
            T_gpu_h[i] = d_h.copy_to_host()[0, 0]
        del intgr_h

        peak_mae_h.append(np.abs(T_can_ref - T_gpu_h).max())
        rms_mae_h.append(np.sqrt(((T_can_ref - T_gpu_h)**2).mean()))
        tau_c = ignition_delay(T_can_ref, t_ref_h)
        tau_g = ignition_delay(T_gpu_h,   t_ref_h)
        err   = abs(tau_c - tau_g) / tau_c * 100 if (tau_c > 0 and not np.isnan(tau_g)) else np.nan
        tau_err_h.append(err)
        print(f'  h={h_test:.0e}s | peak MAE={peak_mae_h[-1]:.2e} K '
              f'| RMS={rms_mae_h[-1]:.2e} K '
              f'| tau_ig err={tau_err_h[-1]:.2f}%' if not np.isnan(err)
              else f'  h={h_test:.0e}s | peak MAE={peak_mae_h[-1]:.2e} K '
                   f'| RMS={rms_mae_h[-1]:.2e} K | tau_ig: no ignition')

    h_us = np.array(H_LIST) * 1e6
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    axes[0].loglog(h_us, peak_mae_h, 'o-', color=COLORS[2], label='Peak MAE')
    axes[0].loglog(h_us, rms_mae_h,  's--', color=COLORS[2], alpha=0.6,
                   label='RMS MAE')
    h_ref = np.array([H_LIST[0]*1e6, H_LIST[-1]*1e6])
    axes[0].loglog(h_ref,
                   [peak_mae_h[0], peak_mae_h[0]*(H_LIST[-1]/H_LIST[0])],
                   'k:', linewidth=0.5, label='O(h)')
    axes[0].set_xlabel(r'Step size $h$ ($\mu$s)')
    axes[0].set_ylabel('Temperature MAE (K)')
    axes[0].legend()

    tau_plot = [v for v in tau_err_h if not np.isnan(v)]
    h_plot   = [H_LIST[i]*1e6 for i, v in enumerate(tau_err_h) if not np.isnan(v)]
    if tau_plot:
        axes[1].semilogx(h_plot, tau_plot, 'D-', color=COLORS[3])
    axes[1].set_xlabel(r'Step size $h$ ($\mu$s)')
    axes[1].set_ylabel(r'Rel. error on $\tau_{\rm ig}$ (\%)')

    plt.tight_layout()
    plt.savefig('article_figures/step_size_sensitivity.png', bbox_inches='tight')
    plt.close()
    print('  Figure: step_size_sensitivity.png')

"""
PATCH — Section à insérer dans test_cuda_integrator_full.py

1. Dans le bloc d'imports, ajouter :
   from kineticsGPU.claude.kernels.cuda_jvp_evaluator import CUDAJVPEvaluator

2. Dans le bloc Flags, ajouter :
   JVP_validation = True

3. Ajouter la section ci-dessous après la section Error_analysis (fin du fichier,
   avant le print "All sections complete").
"""

# ─────────────────────────────────────────────────────────────────────────────
# 6. JVP validation — Cantera exact vs Torch vs CUDA per-block
#
# On valide uniquement la JVP de omega = dC/dt (mol/m³/s).
# C'est ce que calculent _eval_jvp_pb (CUDA) et rxn_sys.jvp (Torch).
#
# Référence Cantera (100% analytique, zéro différences finies) :
#   Jv_om = net_production_rates_ddCi @ vC  +  net_production_rates_ddT * vT
#
# Conversions d'unités Cantera → mol :
#   net_production_rates_ddCi : s⁻¹  (∂(kmol/m³/s)/∂(kmol/m³), les 1e3 se
#                               compensent quand vC est en mol/m³)
#   net_production_rates_ddT  : kmol/m³/s/K  →  ×1e3  →  mol/m³/s/K
# ─────────────────────────────────────────────────────────────────────────────
if JVP_validation:
    print('\n' + '='*70)
    print('Section 6 — JVP validation (Cantera exact / Torch / CUDA per-block)')

    from kineticsGPU.claude.kernels.cuda_jvp_evaluator import CUDAJVPEvaluator
    from kineticsGPU.reactionTypes import (
        ReactionSystem, ElementaryReactionBatch,
        ThreeBodyReactionBatch, FalloffReactionBatch,
    )

    N_JVP = 200

    np.random.seed(99)
    T_jvp  = np.random.uniform(800, 4000, N_JVP).astype(dtype_np)
    C_jvp  = np.abs(np.random.randn(N_JVP, n_sp).astype(dtype_np)) * 5.0
    vT_raw = np.random.randn(N_JVP).astype(dtype_np)
    vC_raw = np.random.randn(N_JVP, n_sp).astype(dtype_np)
    norms  = np.sqrt(vT_raw**2 + (vC_raw**2).sum(axis=1))
    vT_jvp = vT_raw / norms
    vC_jvp = vC_raw / norms[:, None]

    # ── 1. Référence Cantera — JVP exacte de omega ────────────────────────────
    def _cantera_jvp_om(T, C, vT, vC):
        """
        Jv_om = (∂ω/∂C) · vC + (∂ω/∂T) · vT   [mol/m³/s]
        Uniquement des dérivées analytiques fournies par Cantera.
        """
        g = ct.Solution(yaml_file)
        g.TP = float(T), 101325.0
        g.concentrations = np.maximum(C, 0.0) * 1e-3      # mol/m³ → kmol/m³

        # ddCi en s⁻¹ : ∂(kmol/m³/s)/∂(kmol/m³)
        # vC en mol/m³, donc ddCi @ vC = [s⁻¹][mol/m³] = mol/m³/s  ✓
        ddCi = np.array(g.net_production_rates_ddCi)       # (n_sp, n_sp), s⁻¹
        ddT  = np.array(g.net_production_rates_ddT) * 1e3  # (n_sp,), mol/m³/s/K

        return ddCi @ vC + ddT * vT                        # (n_sp,), mol/m³/s

    print(f'  Computing Cantera exact JVP (N={N_JVP}) ...', end=' ', flush=True)
    t0 = time.perf_counter()
    ref_jvC = np.zeros((N_JVP, n_sp), dtype=dtype_np)
    for i in range(N_JVP):
        ref_jvC[i] = _cantera_jvp_om(T_jvp[i], C_jvp[i], vT_jvp[i], vC_jvp[i])
    print(f'{time.perf_counter()-t0:.1f}s')

    # ── 2. Torch JVP de omega ─────────────────────────────────────────────────
    batched_elem = ElementaryReactionBatch(r_e, device=str(device), dtype=torch.float64)
    batched_tb   = ThreeBodyReactionBatch(r_tb, device=str(device), dtype=torch.float64)
    batched_f    = FalloffReactionBatch(r_f,   device=str(device), dtype=torch.float64)
    rxn_sys      = ReactionSystem([batched_elem, batched_tb, batched_f])

    print('  Computing Torch JVP ...', end=' ', flush=True)
    t0 = time.perf_counter()
    T_t  = torch.tensor(T_jvp,  device=device, dtype=torch.float64)
    C_t  = torch.tensor(C_jvp,  device=device, dtype=torch.float64)
    vT_t = torch.tensor(vT_jvp, device=device, dtype=torch.float64)
    vC_t = torch.tensor(vC_jvp, device=device, dtype=torch.float64)
    torch_jvC = rxn_sys.jvp(T_t, C_t, vT_t, vC_t).cpu().numpy()  # (N, n_sp)
    torch.cuda.synchronize()
    print(f'{time.perf_counter()-t0:.4f}s')

    # ── 3. CUDA per-block JVP de omega ────────────────────────────────────────
    jvp_ev = CUDAJVPEvaluator(mech, dtype=dtype_np)
    jvp_ev.evaluate(T_jvp[:1], C_jvp[:1], vT_jvp[:1], vC_jvp[:1])  # warmup JIT
    print('  CUDA JVP evaluator JIT: done')

    print('  Computing CUDA per-block JVP ...', end=' ', flush=True)
    t0 = time.perf_counter()
    cuda_Jv_om, _ = jvp_ev.evaluate(T_jvp, C_jvp, vT_jvp, vC_jvp)
    t_cuda_jvp = time.perf_counter() - t0
    print(f'{t_cuda_jvp:.4f}s')

    # ── Métriques ─────────────────────────────────────────────────────────────
    def _stats(label, a, b_ref):
        diff = np.abs(a - b_ref)
        rel  = diff / (np.abs(b_ref) + 1e-100)
        print(f'    {label}')
        print(f'      MAE  = {diff.mean():.3e}  ± {diff.std():.3e}  max={diff.max():.3e}')
        print(f'      rMAE = {rel.mean():.3e}  ± {rel.std():.3e}  max={rel.max():.3e}')
        return diff, rel

    print('\n  ── Jv_om components (N×n_sp flattened) ──')
    d_torch, r_torch = _stats('Torch vs Cantera exact',
                               torch_jvC.ravel(), ref_jvC.ravel())
    d_cuda,  r_cuda  = _stats('CUDA  vs Cantera exact',
                               cuda_Jv_om.ravel(), ref_jvC.ravel())
    d_tc,    r_tc    = _stats('CUDA  vs Torch',
                               cuda_Jv_om.ravel(), torch_jvC.ravel())

    # ── Erreur par espèce ─────────────────────────────────────────────────────
    sp_mae_torch = np.abs(torch_jvC  - ref_jvC).mean(axis=0)
    sp_mae_cuda  = np.abs(cuda_Jv_om - ref_jvC).mean(axis=0)
    sp_names     = gas.species_names
    worst5_torch = np.argsort(sp_mae_torch)[-5:][::-1]
    worst5_cuda  = np.argsort(sp_mae_cuda )[-5:][::-1]

    print('\n  Top-5 worst species (Torch vs Cantera exact):')
    for k in worst5_torch:
        print(f'    {sp_names[k]:12s}  MAE={sp_mae_torch[k]:.3e}')
    print('  Top-5 worst species (CUDA vs Cantera exact):')
    for k in worst5_cuda:
        print(f'    {sp_names[k]:12s}  MAE={sp_mae_cuda[k]:.3e}')

    # ── Figures ───────────────────────────────────────────────────────────────

    # Fig 1 — Boxplots MAE absolue + relative, 3 comparaisons
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    pairs = [
        ('Torch vs\nCantera exact', d_torch, r_torch),
        ('CUDA vs\nCantera exact',  d_cuda,  r_cuda),
        ('CUDA vs\nTorch',          d_tc,    r_tc),
    ]
    for col, (title, d_abs, d_rel) in enumerate(pairs):
        for row, (data, ylabel) in enumerate([
            (d_abs, r'MAE (mol m$^{-3}$ s$^{-1}$)'),
            (d_rel, 'Relative error (--)'),
        ]):
            ax = axes[row, col]
            ax.boxplot([data], labels=[title], patch_artist=True, showfliers=False,
                       medianprops=dict(color='black'),
                       boxprops=dict(facecolor=COLORS[col], alpha=0.6))
            ax.set_yscale('log')
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle(r'JVP validation — $\dot{\omega}$ components', fontsize=10)
    plt.tight_layout()
    plt.savefig('article_figures/jvp_boxplots.png', bbox_inches='tight')
    plt.close()

    # Fig 2 — MAE par espèce
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    x = np.arange(n_sp)
    for ax, sp_mae, label, color in [
        (axes[0], sp_mae_torch, 'Torch vs Cantera exact', COLORS[1]),
        (axes[1], sp_mae_cuda,  'CUDA  vs Cantera exact', COLORS[2]),
    ]:
        ax.bar(x, sp_mae, color=color, alpha=0.7, width=1.0)
        ax.set_yscale('log')
        ax.set_xlabel('Species index')
        ax.set_ylabel(r'MAE (mol m$^{-3}$ s$^{-1}$)')
        ax.set_title(label, fontsize=8)
        for k in np.argsort(sp_mae)[-3:][::-1]:
            ax.text(k, sp_mae[k] * 1.5, sp_names[k],
                    ha='center', fontsize=6, rotation=90)
    plt.tight_layout()
    plt.savefig('article_figures/jvp_per_species.png', bbox_inches='tight')
    plt.close()

    # Fig 3 — Scatter : impl vs Cantera exact (sous-échantillonné)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    rng = np.random.default_rng(0)
    idx = rng.choice(N_JVP * n_sp, min(5000, N_JVP * n_sp), replace=False)
    v_ref = ref_jvC.ravel()[idx]
    for ax, jvC_impl, label, col in [
        (axes[0], torch_jvC,  'Torch', COLORS[1]),
        (axes[1], cuda_Jv_om, 'CUDA',  COLORS[2]),
    ]:
        v_im = jvC_impl.ravel()[idx]
        ax.scatter(v_ref, v_im, s=1, alpha=0.3, color=col, rasterized=True)
        lim = [min(v_ref.min(), v_im.min()), max(v_ref.max(), v_im.max())]
        ax.axline((lim[0], lim[0]), slope=1, color='red',
                  linewidth=0.4, linestyle=':', label='x=y')
        ax.set_xlabel(r'Cantera exact (mol m$^{-3}$ s$^{-1}$)')
        ax.set_ylabel(f'{label} (mol m$^{-3}$ s$^{-1}$)')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('article_figures/jvp_scatter.png', bbox_inches='tight')
    plt.close()

    print(f'\n  Figures: jvp_boxplots, jvp_per_species, jvp_scatter')
    print(f'  CUDA JVP timing: {t_cuda_jvp*1e3:.2f} ms  '
          f'({t_cuda_jvp/N_JVP*1e6:.2f} µs/reactor)')

if Jacobian_validation:
    print('\n' + '=' * 70)
    print('Section 7 — Jacobian validation (Cantera exact / CUDA PT / CUDA PB)')

    from kineticsGPU.claude.kernels.per_thread.jacobian import CUDAJacobianEvaluatorPT
    from kineticsGPU.claude.kernels.per_block.jacobian_pb import CUDAJacobianEvaluatorPB

    N_JAC = 50  # nombre de réacteurs — jacobienne (54×54) par réacteur, coûteux côté Cantera

    np.random.seed(17)
    T_jac = np.random.uniform(800, 4000, N_JAC).astype(dtype_np)
    C_jac = np.abs(np.random.randn(N_JAC, n_sp).astype(dtype_np)) * 5.0


    # ── Helpers thermo scalaires (cohérents avec reactor_odes_pb.py) ──────────

    def _dcvdT_analytic(gas_obj):
        """
        Retourne dcvdT_s pour toutes les espèces [J/mol/K²]
        basé sur les polynômes NASA7 (analytique).
        """
        T = gas_obj.T
        dcvdT = np.zeros(gas_obj.n_species)

        for k in range(gas_obj.n_species):
            sp = gas_obj.species(k)
            thermo = sp.thermo

            # coeffs NASA7
            # format Cantera :
            # [Tmid,
            #  a1..a7 (high T),
            #  a1..a7 (low T)]
            coeffs = thermo.coeffs

            Tmid = coeffs[0]

            if T >= Tmid:
                a = coeffs[1:8]  # high-T
            else:
                a = coeffs[8:15]  # low-T

            # cp/R = a1 + a2*T + a3*T² + ...
            # donc d(cp/R)/dT = a2 + 2a3*T + 3a4*T² + 4a5*T³
            dcpdT_R = (
                    a[1]
                    + 2.0 * a[2] * T
                    + 3.0 * a[3] * T ** 2
                    + 4.0 * a[4] * T ** 3
            )

            dcvdT[k] = dcpdT_R * R_GAS  # J/mol/K²

        return dcvdT

    def _thermo_scalar(gas_obj):
        """
        Retourne (ntot, cv_mix, su_om, cv_s, u_s, dcvdT_s)
        depuis un objet Cantera gas déjà initialisé (TP, concentrations fixés).
        Unités : concentrations en mol/m³, énergies en J/mol.
        """
        C_mol = gas_obj.concentrations * 1e3  # kmol/m³ → mol/m³
        T = gas_obj.T
        ntot = C_mol.sum()
        Xs = C_mol / ntot if ntot > 0 else np.zeros_like(C_mol)

        # cv_s  [J/mol/K] = cp_s - R
        cp_s = gas_obj.standard_cp_R * R_GAS  # cp_s = (cp/R) * R
        cv_s = cp_s - R_GAS

        # u_s   [J/mol]  = h_s - R*T
        h_s = gas_obj.standard_enthalpies_RT * R_GAS * T
        u_s = h_s - R_GAS * T

        # dcvdT_s : dérivée par rapport à T de cv_s = cp_s - R
        # On utilise la formule polynomiale NASA7 directement.
        # Cantera ne l'expose pas directement → différences finies 2nd ordre
        dcvdT = _dcvdT_analytic(gas_obj)

        cv_mix = (Xs * cv_s).sum()
        su_om = (u_s * gas_obj.net_production_rates * 1e3).sum()  # mol/m³/s

        return ntot, cv_mix, su_om, cv_s, u_s, dcvdT


    # ── 1. Référence Cantera complète ─────────────────────────────────────────

    def _cantera_jacobian(T, C):
        """
        Jacobienne complète (n_state × n_state) du RHS adiabatique.

        Bloc chimique [1:, :] :
          J[s+1, 0]   = dω_s/dT       (mol/m³/s/K)    ddT[s] * 1e3
          J[s+1, k+1] = dω_s/dC_k     (s⁻¹)           ddCi[s, k]

        Ligne thermique [0, :] :
          dT/dt = -su_om / denom,   denom = ntot * cv_mix

          d(dT/dt)/dT   = -(dsuom_T * denom - su_om * ddenom_T) / denom²
          d(dT/dt)/dC_s = -(dsuom_s * denom - su_om * ddenom_s) / denom²

          dsuom_T   = sum_k( u_k * J[k+1, 0] ) + sum_k( cv_k * ω_k )
          ddenom_T  = ntot * sum_k( X_k * dcvdT_k )

          dsuom_s   = sum_k( u_k * J[k+1, s+1] )
          ddenom_s  = cv_mix + ntot * (cv_s - cv_mix) / ntot
                    = cv_s    (pour ntot > 0)
        """
        g = ct.Solution(yaml_file)
        g.TP = float(T), 101325.0
        g.concentrations = np.maximum(C, 0.0) * 1e-3  # mol/m³ → kmol/m³

        n_state = n_sp + 1
        J = np.zeros((n_state, n_state), dtype=dtype_np)

        # ── Bloc chimique ─────────────────────────────────────────────
        # ddCi  s⁻¹  (n_sp × n_sp)
        ddCi = np.array(g.net_production_rates_ddCi)  # (n_sp, n_sp)
        # ddT   kmol/m³/s/K → mol/m³/s/K
        ddT = np.array(g.net_production_rates_ddT) * 1e3  # (n_sp,)

        J[1:, 0] = ddT  # colonne T
        J[1:, 1:] = ddCi  # colonnes C_s

        # ── Ligne thermique ───────────────────────────────────────────
        ntot, cv_mix, su_om, cv_s, u_s, dcvdT_s = _thermo_scalar(g)
        om = g.net_production_rates * 1e3  # mol/m³/s
        denom = ntot * cv_mix
        if abs(denom) < 1e-300:
            J[0, :] = 0.0
            return J

        C_mol = g.concentrations * 1e3
        Xs = C_mol / ntot if ntot > 0 else np.zeros(n_sp)

        # Colonne T
        dsuom_T = (u_s * ddT).sum() + (cv_s * om).sum()
        ddenom_T = ntot * (Xs * dcvdT_s).sum()
        J[0, 0] = -(dsuom_T * denom - su_om * ddenom_T) / (denom ** 2)

        # Colonnes C_s
        for s in range(n_sp):
            dsuom_s = (u_s * ddCi[:, s]).sum()
            ddenom_s = cv_s[s]  # = cv_mix + ntot*(cv_s-cv_mix)/ntot = cv_s
            J[0, s + 1] = -(dsuom_s * denom - su_om * ddenom_s) / (denom ** 2)

        return J


    print(f'  Computing Cantera exact Jacobian (N={N_JAC}) ...', end=' ', flush=True)
    t0 = time.perf_counter()
    ref_J = np.stack([_cantera_jacobian(T_jac[i], C_jac[i]) for i in range(N_JAC)])
    print(f'{time.perf_counter() - t0:.1f}s')

    # ── 2. CUDA per-thread ────────────────────────────────────────────────────
    jac_pt = CUDAJacobianEvaluatorPT(mech, dtype=dtype_np)
    jac_pt.evaluate(T_jac[:1], C_jac[:1])  # warmup JIT
    print('  CUDA PT Jacobian evaluator JIT: done')

    print('  Computing CUDA per-thread Jacobian ...', end=' ', flush=True)
    t0 = time.perf_counter()
    J_pt = jac_pt.evaluate(T_jac, C_jac)  # (N_JAC, n_state, n_state)
    t_pt = time.perf_counter() - t0
    print(f'{t_pt * 1e3:.2f}ms')

    # ── 3. CUDA per-block ─────────────────────────────────────────────────────
    jac_pb = CUDAJacobianEvaluatorPB(mech, dtype=dtype_np)
    jac_pb.evaluate(T_jac[:1], C_jac[:1])  # warmup JIT
    print('  CUDA PB Jacobian evaluator JIT: done')

    print('  Computing CUDA per-block Jacobian ...', end=' ', flush=True)
    t0 = time.perf_counter()
    J_pb = jac_pb.evaluate(T_jac, C_jac)  # (N_JAC, n_state, n_state)
    t_pb = time.perf_counter() - t0
    print(f'{t_pb * 1e3:.2f}ms')


    # ── Métriques ─────────────────────────────────────────────────────────────

    def _jac_stats(label, J_impl, J_ref):
        """Affiche MAE / rMAE sur les 3 blocs : total, chimique, thermique."""
        diff = np.abs(J_impl - J_ref)
        rel = diff / (np.abs(J_ref) + 1e-100)

        # Bloc total
        print(f'    {label} — total ({J_impl.shape[1]}×{J_impl.shape[2]})')
        print(f'      MAE  = {diff.mean():.3e}  max = {diff.max():.3e}')
        print(f'      rMAE = {rel.mean():.3e}   max = {rel.max():.3e}')

        # Bloc chimique J[1:, :]  (n_sp × n_state)
        dc = diff[:, 1:, :]
        rc = rel[:, 1:, :]
        print(f'    {label} — chimique [1:, :]')
        print(f'      MAE  = {dc.mean():.3e}  max = {dc.max():.3e}')
        print(f'      rMAE = {rc.mean():.3e}   max = {rc.max():.3e}')

        # Ligne thermique J[0, :]  (n_state,)
        dt = diff[:, 0, :]
        rt = rel[:, 0, :]
        print(f'    {label} — thermique [0, :]')
        print(f'      MAE  = {dt.mean():.3e}  max = {dt.max():.3e}')
        print(f'      rMAE = {rt.mean():.3e}   max = {rt.max():.3e}')

        return diff, rel


    print('\n  ── Métriques ──')
    d_pt, r_pt = _jac_stats('CUDA PT vs Cantera', J_pt, ref_J)
    d_pb, r_pb = _jac_stats('CUDA PB vs Cantera', J_pb, ref_J)
    d_pp, r_pp = _jac_stats('CUDA PB vs CUDA PT', J_pb, J_pt)

    # ── Erreur par colonne (par variable xk_j) ────────────────────────────────
    col_mae_pt = np.abs(J_pt - ref_J).mean(axis=(0, 1))  # (n_state,)
    col_mae_pb = np.abs(J_pb - ref_J).mean(axis=(0, 1))

    print(f'\n  Worst 5 columns (PT vs Cantera):')
    for k in np.argsort(col_mae_pt)[-5:][::-1]:
        name = 'T' if k == 0 else gas.species_names[k - 1]
        print(f'    xk[{k:2d}]={name:12s}  MAE={col_mae_pt[k]:.3e}')
    print(f'  Worst 5 columns (PB vs Cantera):')
    for k in np.argsort(col_mae_pb)[-5:][::-1]:
        name = 'T' if k == 0 else gas.species_names[k - 1]
        print(f'    xk[{k:2d}]={name:12s}  MAE={col_mae_pb[k]:.3e}')

    # ── Figures ───────────────────────────────────────────────────────────────

    # Fig 1 — Boxplots MAE absolue + relative  (total / chimique / thermique)
    labels_box = ['PT total', 'PT chem', 'PT therm',
                  'PB total', 'PB chem', 'PB therm']
    data_abs = [
        d_pt.ravel(),
        d_pt[:, 1:, :].ravel(),
        d_pt[:, 0, :].ravel(),
        d_pb.ravel(),
        d_pb[:, 1:, :].ravel(),
        d_pb[:, 0, :].ravel(),
    ]
    data_rel = [
        r_pt.ravel(),
        r_pt[:, 1:, :].ravel(),
        r_pt[:, 0, :].ravel(),
        r_pb.ravel(),
        r_pb[:, 1:, :].ravel(),
        r_pb[:, 0, :].ravel(),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, data, ylabel in [
        (axes[0], data_abs, r'MAE (SI)'),
        (axes[1], data_rel, 'Relative error (--)'),
    ]:
        ax.boxplot([d for d in data], labels=labels_box,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='black'),
                   boxprops=dict(facecolor=COLORS[2], alpha=0.6))
        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelrotation=30)
    fig.suptitle('Jacobian validation vs Cantera exact', fontsize=10)
    plt.tight_layout()
    plt.savefig('article_figures/jacobian_boxplots.png', bbox_inches='tight')
    plt.close()

    # Fig 2 — Scatter : CUDA vs Cantera (sous-échantillon)
    rng = np.random.default_rng(7)
    tot = N_JAC * (n_sp + 1) ** 2
    idx = rng.choice(tot, min(8000, tot), replace=False)
    vref = ref_J.ravel()[idx]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, J_impl, label, col in [
        (axes[0], J_pt, 'CUDA PT', COLORS[1]),
        (axes[1], J_pb, 'CUDA PB', COLORS[2]),
    ]:
        vim = J_impl.ravel()[idx]
        ax.scatter(vref, vim, s=1, alpha=0.3, color=col, rasterized=True)
        lo = min(vref.min(), vim.min())
        hi = max(vref.max(), vim.max())
        ax.axline((lo, lo), slope=1, color='red', linewidth=0.4,
                  linestyle=':', label='x=y')
        ax.set_xlabel('Cantera exact (SI)')
        ax.set_ylabel(f'{label} (SI)')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('article_figures/jacobian_scatter.png', bbox_inches='tight')
    plt.close()

    # Fig 3 — MAE par colonne xk_j
    x = np.arange(n_sp + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, col_mae, label, col in [
        (axes[0], col_mae_pt, 'CUDA PT vs Cantera', COLORS[1]),
        (axes[1], col_mae_pb, 'CUDA PB vs Cantera', COLORS[2]),
    ]:
        ax.bar(x, col_mae, color=col, alpha=0.7, width=1.0)
        ax.set_yscale('log')
        ax.set_xlabel('Column index $j$ in $x_k$  (0=T, j=C_{j-1})')
        ax.set_ylabel('MAE (SI)')
        ax.set_title(label, fontsize=8)
        # Annotation des 3 pires colonnes
        for k in np.argsort(col_mae)[-3:][::-1]:
            name = 'T' if k == 0 else gas.species_names[k - 1]
            ax.text(k, col_mae[k] * 1.5, name,
                    ha='center', fontsize=6, rotation=90)
    plt.tight_layout()
    plt.savefig('article_figures/jacobian_per_column.png', bbox_inches='tight')
    plt.close()

    # Fig 4 — Heatmap moyenne des erreurs relatives (n_state × n_state)
    mean_relerr_pt = r_pt.mean(axis=0)  # (n_state, n_state)
    mean_relerr_pb = r_pb.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in [
        (axes[0], mean_relerr_pt, 'CUDA PT — mean rel. error'),
        (axes[1], mean_relerr_pb, 'CUDA PB — mean rel. error'),
    ]:
        im = ax.imshow(np.log10(mat + 1e-20), aspect='auto',
                       origin='upper', cmap='plasma')
        plt.colorbar(im, ax=ax, label=r'$\log_{10}$ rel. error')
        ax.set_xlabel('Column $j$  (0=T)')
        ax.set_ylabel('Row $i$  (0=dT/dt)')
        ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig('article_figures/jacobian_heatmap.png', bbox_inches='tight')
    plt.close()

    print(f'\n  Figures: jacobian_boxplots, jacobian_scatter, '
          f'jacobian_per_column, jacobian_heatmap')
    print(f'  Timing — CUDA PT: {t_pt * 1e3:.2f}ms  '
          f'CUDA PB: {t_pb * 1e3:.2f}ms  '
          f'({N_JAC} réacteurs)')



print('\n' + '='*70)
print('All sections complete. Figures in article_figures/')