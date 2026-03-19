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
RR_and_source_terms  = True
Integrators          = True
Architecture_compare = True
Scalability          = True
Error_analysis       = True

N_reactors_list      = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
N_STEPS_SCALABILITY  = 3000

SIZE        = 1000
T_END       = 3e-4
H_DEFAULT   = 1e-7
H_LIST      = [5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 2e-8, 1e-8, 5e-9, 2e-9]

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
r_e, r_tb, r_f, _ = build_reactions(vec_th.species, rxns, vec_th, 'cuda',
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
        ['Elem.\nRR', 'TB\nRR', 'Falloff\nRR', r'$\dot\omega$'],
        'article_figures/RR_MAE.png', ylabel='MAE (mol m$^{-3}$ s$^{-1}$)')
    plot_box(
        [e_MSE_err, tb_MSE_err, f_MSE_err, om_MSE_err],
        ['Elem.\nRR', 'TB\nRR', 'Falloff\nRR', r'$\dot\omega$'],
        'article_figures/RR_MSE.png', ylabel='MSE (mol$^2$ m$^{-6}$ s$^{-2}$)')

    e_rel  = np.abs(RRs_e_can  - RR_e_gpu)  / (np.abs(RRs_e_can)  + 1e-100)
    tb_rel = np.abs(RRs_tb_can - RR_tb_gpu) / (np.abs(RRs_tb_can) + 1e-100)
    f_rel  = np.abs(RRs_f_can  - RR_f_gpu)  / (np.abs(RRs_f_can)  + 1e-100)
    om_rel = np.abs(omega_can  - omega_gpu)  / (np.abs(omega_can)  + 1e-100)

    plot_box(
        [e_rel, tb_rel, f_rel, om_rel],
        ['Elem.\nRR', 'TB\nRR', 'Falloff\nRR', r'$\dot\omega$'],
        'article_figures/RR_relative_error.png',
        ylabel='Relative error (--)')
    print('  Figures: RR_MAE, RR_MSE, RR_relative_error')

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_cantera_CV(T0, C0_mol, t_ref):
    g = ct.Solution(yaml_file)
    g.TP = float(T0), 101325.0
    g.concentrations = C0_mol * 1e-3
    r = ct.IdealGasReactor(g, energy='on')
    sim = ct.ReactorNet([r])
    ts, Ts, Ps, Xs = [], [], [], []
    while sim.time < t_ref[-1]:
        sim.step()
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
        T, P, X = run_cantera_CV(T_all[i], C_all[i], t_ref)
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
    ax.plot(t_ref*1e6, cant_T[0], label='Cantera', color=COLORS[0])
    ax.plot(t_ref*1e6, gpu_T[0],  label='GPU (BE)', color=COLORS[2],
            linestyle=':', linewidth=0.7)
    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel('Temperature (K)')
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
        T_can_ref, _, _ = run_cantera_CV(1700.0, C0_ref_single, t_ref_h)

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

print('\n' + '='*70)
print('All sections complete. Figures in article_figures/')