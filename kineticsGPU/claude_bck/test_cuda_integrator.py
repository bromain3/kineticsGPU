"""
test_cuda_integrator.py

Complete validation test for the Numba CUDA per-thread backward Euler integrator.
Mirrors the existing validation pipeline in the torch codebase (validation.py).

Steps:
  1. Load GRI30 mechanism and build MechanismData
  2. Initialize CUDABackwardEulerIntegrator
  3. Integrate a batch of CH4/air reactors
  4. Compare against Cantera reference solutions

Run:
    python tests/test_cuda_integrator.py
"""

import numpy as np
import time
import math

# ── Local imports ─────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.claude_bck.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.claude_bck.integrator.cuda_integrator import CUDABackwardEulerIntegrator


# ── Optional Cantera for reference ────────────────────────────────────────────
try:
    import cantera as ct
    HAS_CANTERA = True
except ImportError:
    HAS_CANTERA = False
    print("WARNING: Cantera not found, reference comparison skipped.")


# ─────────────────────────────────────────────────────────────────────────────
def run_cantera_batch(T0s, C0s, t_end):
    """Run Cantera constant-volume adiabatic reactors sequentially."""
    results = []
    for T0, C0 in zip(T0s, C0s):
        gas = ct.Solution('gri30.yaml')
        gas.TP = float(T0), 101325.0
        gas.concentrations = C0 / 1e3   # mol/m3 → mol/m3 (Cantera uses SI)
        r = ct.IdealGasReactor(gas, energy='on')
        sim = ct.ReactorNet([r])
        sim.advance(t_end)
        T_f = r.T
        C_f = np.array(r.thermo.concentrations) * 1e3  # → mol/m3
        results.append((T_f, C_f))
    return results


# ─────────────────────────────────────────────────────────────────────────────
def main():
    yaml_file = '../gri30.yaml'
    dtype = np.float64
    device_torch = 'cuda'

    print("=" * 70)
    print("Loading mechanism...")
    import torch
    y = yaml_thermo.yamlParser(yaml_file)
    vec_th = y.populateVectorizedThermoObject(
        dtype=torch.float64, device=device_torch
    )
    reactions = parseReactions(yaml_file)
    r_e, r_tb, r_f, system = build_reactions(
        vec_th.species, reactions, vec_th, device_torch, torch.float64
    )
    print(f"  {len(r_e)} elementary, {len(r_tb)} three-body, {len(r_f)} falloff")

    # Build MechanismData
    print("\nBuilding MechanismData for CUDA kernels...")
    mech = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype)
    print(f"  n_species = {mech.n_species}")
    print(f"  n_reactions = {mech.n_reactions_total}")

    # ── Initial conditions ────────────────────────────────────────────────
    n_reactors = 1
    print(f"\nSetting up {n_reactors} CH4/air reactors...")

    if HAS_CANTERA:
        gas_ref = ct.Solution(yaml_file)
    T0s = np.random.uniform(1800.0, 2000.0, n_reactors)
    phis = np.random.uniform(0.5, 1.5, n_reactors)
    C0s = np.zeros((n_reactors, mech.n_species), dtype=dtype)

    for i, (T0, phi) in enumerate(zip(T0s, phis)):
        if HAS_CANTERA:
            gas_ref.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
            gas_ref.TP = T0, 101325.0
            C0s[i] = np.array(gas_ref.concentrations) * 1e3  # mol/m3

    # Pack as TC array: [T, C1, ..., CN]
    TC0 = np.zeros((n_reactors, mech.n_species + 1), dtype=dtype)
    TC0[:, 0]  = T0s
    TC0[:, 1:] = C0s

    # ── Instantiate integrator ────────────────────────────────────────────
    print("\nInstantiating CUDA integrator...")
    integrator = CUDABackwardEulerIntegrator(
        mechanism=mech,
        n_reactors=n_reactors,
        newton_tol=1e-6,
        newton_maxiter=20,
        reuse_jacobian=False,
        dtype=dtype,
    )

    # ── Integration ───────────────────────────────────────────────────────
    t0      = 0.0
    t_end   = 3e-4      # 10 µs
    h       = 1e-7      # 100 ns step
    n_steps = int(t_end / h)

    print(f"\nIntegrating {n_reactors} reactors for {n_steps} steps (h={h:.1e} s)...")
    t_start = time.perf_counter()
    TC_hist, t_hist, conv_hist = integrator.integrate(TC0, t0, t_end, h, save_every=10, do_timing=True)
    integrator.print_timers(n_steps=int(t_end / h))
    t_elapsed = time.perf_counter() - t_start

    print(f"  Wall time: {t_elapsed:.3f} s")
    print(f"  Steps/s  : {n_steps * n_reactors / t_elapsed:.2e}")

    # Convergence check
    last_conv = conv_hist[-1]
    n_conv = np.sum(last_conv)
    print(f"  Converged reactors at last step: {n_conv}/{n_reactors}")

    TC_final = TC_hist[-1]   # (n_reactors, n_state)
    T_final  = TC_final[:, 0]
    print(f"  T range final: [{T_final.min():.1f}, {T_final.max():.1f}] K")

    # ── Cantera reference comparison (first reactor) ──────────────────────
    if HAS_CANTERA and n_reactors >= 1:
        print("\nRunning Cantera reference for first reactor...")
        ref_results = run_cantera_batch(T0s[:1], C0s[:1], t_end)
        T_ref, C_ref = ref_results[0]
        T_gpu = TC_final[0, 0]
        C_gpu = TC_final[0, 1:]

        dT = abs(T_gpu - T_ref)
        dC = np.abs(C_gpu - C_ref)

        print(f"\n  Temperature comparison (reactor 0):")
        print(f"    Cantera:  T = {T_ref:.4f} K")
        print(f"    CUDA GPU: T = {T_gpu:.4f} K")
        print(f"    |ΔT|    = {dT:.4e} K")
        print(f"\n  Concentrations (top-5 by magnitude):")
        top5 = np.argsort(np.abs(C_ref))[-5:][::-1]
        sp_names = mech.thermo.species_names
        for idx in top5:
            print(f"    {sp_names[idx]:10s}: Cantera={C_ref[idx]:.4e}  GPU={C_gpu[idx]:.4e}  "
                  f"err={abs(C_gpu[idx]-C_ref[idx])/(abs(C_ref[idx])+1e-30):.2e}")

        # ── Trajectory plot ───────────────────────────────────────────────
        print("\nCollecting Cantera trajectory for reactor 0...")
        t_ct, T_ct, C_ct = run_cantera_trajectory(T0s[0], C0s[0], t_end, h * 10)

        # GPU trajectory: TC_hist shape is (n_saved, n_reactors, n_state)
        t_gpu_traj = np.array(t_hist)          # (n_saved,)
        T_gpu_traj = TC_hist[:, 0, 0]          # (n_saved,)  reactor 0, temperature
        C_gpu_traj = TC_hist[:, 0, 1:]         # (n_saved, n_species)

        # Pick species to plot: top-4 by final Cantera concentration
        top4_sp = np.argsort(C_ref)[-4:][::-1]

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1 + len(top4_sp), 1,
                                 figsize=(9, 3 * (1 + len(top4_sp))),
                                 sharex=True)

        # Temperature
        ax = axes[0]
        ax.plot(t_ct  * 1e3, T_ct,       'k-',  lw=1.5, label='Cantera')
        ax.plot(t_gpu_traj * 1e3, T_gpu_traj, 'r--', lw=1.5, label='CUDA GPU')
        ax.set_ylabel('T (K)')
        ax.legend(loc='best')
        ax.set_title('Reactor 0 — temperature & species trajectories')

        # Species
        for ax, idx in zip(axes[1:], top4_sp):
            name = sp_names[idx]
            ax.plot(t_ct  * 1e3, C_ct[:, idx],       'k-',  lw=1.5, label='Cantera')
            ax.plot(t_gpu_traj * 1e3, C_gpu_traj[:, idx], 'r--', lw=1.5, label='CUDA GPU')
            ax.set_ylabel(f'[{name}] (mol/m³)')
            ax.legend(loc='best')

        axes[-1].set_xlabel('Time (ms)')
        fig.tight_layout()
        plt.savefig('trajectory_comparison.png', dpi=150)
        print("  Plot saved to trajectory_comparison.png")
        plt.show()

    print("\n" + "=" * 70)
    print("Test complete.")


def run_cantera_trajectory(T0, C0, t_end, dt):
    """Return full time trajectory for a single reactor."""
    gas = ct.Solution('gri30.yaml')
    gas.TP = float(T0), 101325.0
    gas.concentrations = C0 / 1e3

    r = ct.IdealGasReactor(gas, energy='on')
    sim = ct.ReactorNet([r])

    t = 0.0
    times = []
    T_hist = []
    C_hist = []

    while t < t_end:
        t += dt
        sim.advance(t)

        times.append(t)
        T_hist.append(r.T)
        C_hist.append(np.array(r.thermo.concentrations) * 1e3)

    return np.array(times), np.array(T_hist), np.array(C_hist)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()

    # try fastmath (slightly faster, from 190 to 168 µs of kernel and wall time from 150 to 134
    # 1 reactor (so no thread divergence) : mean step time of about 190 µs, meaning about 3000 * 190 = 0.57 s of pur kernel time vs 150 s of wall time. so ther is a huuuuuge problem somewhere else. maybe due to thread waiting for one another ?