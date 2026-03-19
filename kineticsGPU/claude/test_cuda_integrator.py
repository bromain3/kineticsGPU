"""
test_cuda_integrator.py — validation test for production CUDA integrator.

Usage:
    python tests/test_cuda_integrator.py
"""

import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.claude.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.claude.integrator.cuda_integrator_pb import CUDABackwardEulerIntegratorPB as CUDABackwardEulerIntegrator
import numba.cuda as cuda

try:
    import cantera as ct
    HAS_CANTERA = True
except ImportError:
    HAS_CANTERA = False
    print("WARNING: Cantera not found — reference comparison skipped.")


def run_cantera_batch(T0s, C0s, t_end):
    results = []
    for T0, C0 in zip(T0s, C0s):
        gas = ct.Solution('gri30.yaml')
        gas.TP = float(T0), 101325.0
        gas.concentrations = C0 / 1e3
        r   = ct.IdealGasReactor(gas, energy='on')
        sim = ct.ReactorNet([r])
        sim.advance(t_end)
        results.append((r.T, np.array(r.thermo.concentrations) * 1e3))
    return results


def run_cantera_trajectory(T0, C0, t_end, dt):
    gas = ct.Solution('gri30.yaml')
    gas.TP = float(T0), 101325.0
    gas.concentrations = C0 / 1e3
    r   = ct.IdealGasReactor(gas, energy='on')
    sim = ct.ReactorNet([r])
    t, Th, Ch = 0.0, [], []
    while t < t_end:
        t += dt; sim.advance(t)
        Th.append(r.T)
        Ch.append(np.array(r.thermo.concentrations) * 1e3)
    return np.array(Th), np.array(Ch)


def main():
    yaml_file = '../gri30.yaml'
    dtype     = np.float64

    print("=" * 70)
    print("Loading mechanism ...")
    import torch
    y      = yaml_thermo.yamlParser(yaml_file)
    vec_th = y.populateVectorizedThermoObject(dtype=torch.float64, device='cuda')
    rxns   = parseReactions(yaml_file)
    r_e, r_tb, r_f, system = build_reactions(
        vec_th.species, rxns, vec_th, 'cuda', torch.float64)
    print(f"  {len(r_e)} elementary, {len(r_tb)} three-body, {len(r_f)} falloff")

    print("\nBuilding MechanismData ...")
    mech = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype)
    print(f"  n_species={mech.n_species}, n_reactions={mech.n_reactions_total}")

    # ── Initial conditions ─────────────────────────────────────────────
    n_reactors = 100
    print(f"\nSetting up {n_reactors} CH4/air reactors ...")

    np.random.seed(0)
    T0s  = np.random.uniform(1800.0, 2000.0, n_reactors)
    phis = np.random.uniform(0.5, 1.5, n_reactors)
    C0s  = np.zeros((n_reactors, mech.n_species), dtype=dtype)

    if HAS_CANTERA:
        gas_ref = ct.Solution(yaml_file)
        for i, (T0, phi) in enumerate(zip(T0s, phis)):
            gas_ref.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
            gas_ref.TP = T0, 101325.0
            C0s[i]     = np.array(gas_ref.concentrations) * 1e3

    TC0       = np.zeros((n_reactors, mech.n_species + 1), dtype=dtype)
    TC0[:, 0] = T0s
    TC0[:, 1:]= C0s

    # ── Integrator ─────────────────────────────────────────────────────
    print("\nInstantiating integrator (triggers JIT compilation) ...")
    t0_compile = time.perf_counter()
    integrator = CUDABackwardEulerIntegrator(
        mechanism      = mech,
        n_reactors     = n_reactors,
        newton_tol     = 1e-6,
        newton_maxiter = 20,
        gmres_restart  = 20,
        gmres_tol      = 1e-6,
        gmres_maxiter  = 50,
        dtype          = dtype,
    )

    t_end    = 3e-4
    h        = 1e-7
    n_steps  = int(t_end / h)
    save_every = 10

    # ── Upload initial state to device once ────────────────────────────
    d_TC = cuda.to_device(np.ascontiguousarray(TC0, dtype=dtype))

    print(f"\nIntegrating {n_reactors} reactors, {n_steps} steps (h={h:.1e}s) ...")

    TC_snapshots = []
    t_snapshots  = []
    t = 0.0

    t_wall = 0.0
    for step_i in range(n_steps):
        h_eff = min(h, t_end - t)
        if h_eff <= 0.0:
            break

        t_wall -= time.perf_counter()
        d_TC, _ = integrator.step(d_TC, h_eff)
        t_wall += time.perf_counter()

        t += h_eff

        if step_i % save_every == 0:
            TC_snapshots.append(d_TC.copy_to_host())
            t_snapshots.append(t)

    print(f"  JIT + first run: {time.perf_counter()-t0_compile:.2f}s total")
    print(f"  Integration wall time: {t_wall:.4f}s "
          f"({1e6*t_wall/n_steps:.2f} µs/step, {n_reactors} reactors, "
          f"per-block launch {n_reactors}×64)")

    TC_hist  = np.stack(TC_snapshots, axis=0)   # (n_saves, B, n_state)
    t_hist   = np.array(t_snapshots)
    TC_final = d_TC.copy_to_host()

    T_final  = TC_final[:, 0]
    print(f"  T final range: [{T_final.min():.1f}, {T_final.max():.1f}] K")

    # ── Cantera comparison ─────────────────────────────────────────────
    if HAS_CANTERA:
        print("\nRunning Cantera reference ...")
        (T_ref, C_ref), = run_cantera_batch(T0s[:1], C0s[:1], t_end)
        T_gpu  = TC_final[0, 0]
        C_gpu  = TC_final[0, 1:]
        dT     = abs(T_gpu - T_ref)
        print(f"\n  Temperature comparison (reactor 0):")
        print(f"    Cantera : {T_ref:.4f} K")
        print(f"    GPU     : {T_gpu:.4f} K")
        print(f"    |ΔT|    : {dT:.4e} K")

        top5 = np.argsort(np.abs(C_ref))[-5:][::-1]
        sp   = mech.thermo.species_names
        print(f"\n  Top-5 species by Cantera magnitude:")
        for idx in top5:
            err = abs(C_gpu[idx]-C_ref[idx]) / (abs(C_ref[idx]) + 1e-30)
            print(f"    {sp[idx]:10s}: Cantera={C_ref[idx]:.4e}  "
                  f"GPU={C_gpu[idx]:.4e}  relErr={err:.2e}")

        # Trajectory plot
        print("\nPlotting trajectory ...")
        T_ct, C_ct = run_cantera_trajectory(T0s[0], C0s[0], t_end, h*10)
        t_gpu   = t_hist
        T_gpu_t = TC_hist[:, 0, 0]
        C_gpu_t = TC_hist[:, 0, 1:]
        top4    = np.argsort(C_ref)[-4:][::-1]

        import matplotlib.pyplot as plt
        t_ms_ct  = np.linspace(h*10, t_end, len(T_ct)) * 1e3
        t_ms_gpu = t_gpu * 1e3

        fig, axes = plt.subplots(1+len(top4), 1,
                                 figsize=(9, 3*(1+len(top4))), sharex=True)
        ax = axes[0]
        ax.plot(t_ms_ct,  T_ct,    'k-',  lw=1.5, label='Cantera')
        ax.plot(t_ms_gpu, T_gpu_t, 'r--', lw=1.5, label='CUDA GPU')
        ax.set_ylabel('T (K)'); ax.legend()
        ax.set_title('Reactor 0 — T and species trajectories')

        for ax, idx in zip(axes[1:], top4):
            ax.plot(t_ms_ct,  C_ct[:, idx],    'k-',  lw=1.5, label='Cantera')
            ax.plot(t_ms_gpu, C_gpu_t[:, idx], 'r--', lw=1.5, label='CUDA GPU')
            ax.set_ylabel(f'[{sp[idx]}] (mol/m³)'); ax.legend()

        axes[-1].set_xlabel('Time (ms)')
        fig.tight_layout()
        plt.savefig('trajectory_comparison.png', dpi=150)
        print("  Saved trajectory_comparison.png")
        plt.show()

    print("\n" + "=" * 70)
    print("Test complete.")


if __name__ == '__main__':
    main()