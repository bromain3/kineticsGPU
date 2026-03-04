"""
benchmark.py
============
Benchmark and correctness validation for the Numba CUDA kinetics integrator.

Compares:
  1. Torch baseline  (BDFTorchContext.BackwardEulerIntegrator)
  2. Numba per-thread strategy
  3. Numba per-block strategy

Metrics reported per strategy:
  - Wall-clock time for N_steps steps (excluding first-call JIT overhead)
  - Memory throughput estimate (GB/s)
  - Max absolute error vs Torch baseline (T and species)

Usage:
  python benchmark.py --ncells 1000 --nsteps 100 --yaml gri30.yaml
                      [--tpb 64]  [--restart 20]  [--no-torch-baseline]
"""

import argparse
import time
import numpy as np
import torch
from numba import cuda

# ── Your existing imports ─────────────────────────────────────────────────────
import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.reactionTypesContext import (
    ElementaryReactionBatchContext,
    ThreeBodyReactionBatchContext,
    FalloffReactionBatchContext,
    ReactionSystemContext,
)
from kineticsGPU.NumericalIntegrators.BDFTorchContext import BackwardEulerIntegrator

# ── Numba back-end ────────────────────────────────────────────────────────────
from kineticsGPU.packing import NumbaPacked
from kineticsGPU.integrator import NumbaSolver

import cantera as ct

R_GAS = 8.31446261815324


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_conditions(yaml_file, ncells, device, dtype, seed=42):
    """Generate random-ish but physically consistent initial conditions."""
    np.random.seed(seed)
    gas = ct.Solution(yaml_file)
    nsp = gas.n_species

    T0s = np.random.uniform(1500, 2000, ncells)
    phis = np.random.uniform(0.5, 1.5, ncells)
    C0s = np.zeros((ncells, nsp))

    for i, (phi, T0) in enumerate(zip(phis, T0s)):
        gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
        gas.TP = T0, 101325.0
        C0s[i] = gas.concentrations   # kmol/m³ → convert below

    T_torch = torch.tensor(T0s,  device=device, dtype=dtype)
    C_torch = torch.tensor(C0s,  device=device, dtype=dtype) * 1e3   # mol/m³
    TC = torch.cat([T_torch[:, None], C_torch], dim=1)               # (ncells, 1+nsp)
    return TC


def adiabatic_cv_rhs_torch(reactSys, t):
    """Adiabatic CV RHS for the Torch integrator (identical to validation script)."""
    T   = reactSys.T
    C   = reactSys.C
    dCdt = reactSys.omega_total()
    n_tot = torch.sum(C, dim=1)
    X     = C / n_tot[:, None]
    vec_th = reactSys.reactionsBatches[0].thermo
    cv_mol_mix = vec_th.cv_mix_mol(T, X)
    u_species  = vec_th.u_species_mol(T)
    dUdt = torch.sum(u_species * dCdt, dim=1)
    dTdt = -dUdt / (n_tot * cv_mol_mix)
    return torch.cat((dTdt[:, None], dCdt), dim=1)


def adiabatic_cv_jvp_torch(reactSys, t, dTC):
    """Analytic JVP for Torch integrator."""
    T   = reactSys.T
    C   = reactSys.C
    vT  = dTC[:, 0]
    vC  = dTC[:, 1:]
    vec_th = reactSys.reactionsBatches[0].thermo

    dCdt     = reactSys.omega_total()
    dCdt_jvp = reactSys.jvp(vT, vC)

    n_tot  = torch.sum(C, dim=1)
    dn_tot = torch.sum(vC, dim=1)
    X      = C / n_tot[:, None]
    dX     = vC / n_tot[:, None] - C * dn_tot[:, None] / (n_tot**2)[:, None]

    u_sp  = vec_th.u_species_mol(T)
    cv_sp = vec_th.cv_species_mol(T, torch.arange(len(vec_th.species), device=T.device))
    dcv   = vec_th.dcv_species_dT(T, torch.arange(len(vec_th.species), device=T.device))
    cv_m  = vec_th.cv_mix_mol(T, X)

    dcv_m   = torch.sum(dX * cv_sp, dim=1) + torch.sum(X * dcv, dim=1) * vT
    U_prime = torch.sum(u_sp * dCdt, dim=1)
    dU_p    = torch.sum(u_sp * dCdt_jvp, dim=1) + torch.sum(dCdt * cv_sp, dim=1) * vT
    denom   = n_tot * cv_m
    dT_jvp  = (-dU_p / denom
               + U_prime * dn_tot / (n_tot**2 * cv_m)
               + U_prime * dcv_m  / (n_tot   * cv_m**2))
    return torch.cat([dT_jvp[:, None], dCdt_jvp], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Timing utility
# ─────────────────────────────────────────────────────────────────────────────

class CUDATimer:
    def __init__(self):
        self.start_ev = cuda.event()
        self.stop_ev  = cuda.event()

    def __enter__(self):
        self.start_ev.record()
        return self

    def __exit__(self, *args):
        self.stop_ev.record()
        self.stop_ev.synchronize()
        self.elapsed_ms = cuda.event_elapsed_time(self.start_ev, self.stop_ev)
        self.elapsed_s  = self.elapsed_ms * 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    device = 'cuda'
    dtype  = torch.float64

    print(f"\n{'='*70}")
    print(f" Chemical Kinetics GPU Integrator Benchmark")
    print(f"{'='*70}")
    print(f"  YAML file  : {args.yaml}")
    print(f"  ncells     : {args.ncells}")
    print(f"  n_steps    : {args.nsteps}")
    print(f"  step size  : {args.h:.2e} s")
    print(f"  TPB        : {args.tpb}")
    print(f"  GMRES m    : {args.restart}")
    print(f"{'='*70}\n")

    # ── Build reaction system ─────────────────────────────────────────────────
    y       = yaml_thermo.yamlParser(args.yaml)
    vec_th  = y.populateVectorizedThermoObject(dtype=dtype, device=device)
    reactions = parseReactions(args.yaml)

    r_e, r_tb, r_f, _ = build_reactions(vec_th.species, reactions, vec_th, device, dtype)

    batched_elem = ElementaryReactionBatchContext(r_e,  device=device, dtype=dtype)
    batched_tb   = ThreeBodyReactionBatchContext(r_tb, device=device, dtype=dtype)
    batched_f    = FalloffReactionBatchContext(r_f,   device=device, dtype=dtype)
    reaction_sys = ReactionSystemContext([batched_elem, batched_tb, batched_f])

    n_e  = batched_elem.n_reactions
    n_tb = batched_tb.n_reactions
    n_f  = batched_f.n_reactions
    nsp  = len(vec_th.species)
    print(f"  {n_e} elementary + {n_tb} three-body + {n_f} falloff reactions")
    print(f"  {nsp} species  →  state size N = {nsp+1}\n")

    # ── Initial conditions ────────────────────────────────────────────────────
    TC0 = make_initial_conditions(args.yaml, args.ncells, device, dtype)

    # ── Pack Numba data (once, no more CPU↔GPU after this) ───────────────────
    packed = NumbaPacked(vec_th, batched_elem, batched_tb, batched_f)

    results = {}   # strategy → list of TC snapshots

    # ══════════════════════════════════════════════════════════════════════════
    # Torch baseline
    # ══════════════════════════════════════════════════════════════════════════
    if not args.no_torch:
        print("─── Torch baseline ───────────────────────────────────────────")
        torch_integrator = BackwardEulerIntegrator(
            reaction_sys,
            adiabatic_cv_rhs_torch,
            adiabatic_cv_jvp_torch,
            device=device, dtype=dtype,
            verbose=False,
            use_autograd_jvp=False,
            gmres_restart=args.restart,
        )
        TC_t = TC0.clone()

        # warm-up (1 step, triggers JIT for any torch.compile)
        _ = torch_integrator.step(TC_t, 0.0, args.h)

        TC_t = TC0.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(args.nsteps):
            TC_t = torch_integrator.step(TC_t, step * args.h, args.h)
        torch.cuda.synchronize()
        t_torch = time.perf_counter() - t0

        results['torch'] = TC_t.clone()
        print(f"  Wall time : {t_torch:.3f} s  ({t_torch/args.nsteps*1e3:.2f} ms/step)\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Numba per-thread
    # ══════════════════════════════════════════════════════════════════════════
    for strategy in ('per_thread', 'per_block'):
        print(f"─── Numba {strategy} {'─'*(40-len(strategy))}")
        solver = NumbaSolver(
            packed,
            strategy=strategy,
            tpb=args.tpb,
            newton_tol=1e-6,
            newton_maxiter=20,
            gmres_restart=args.restart,
            gmres_tol=1e-6,
            gmres_maxiter=args.restart * 3,
            verbose=False,
        )

        TC_n = TC0.clone()

        # warm-up: 1 step (triggers Numba JIT compilation)
        print("  Compiling kernels (first call)...", end=' ', flush=True)
        t_compile = time.perf_counter()
        _ = solver.step(TC_n, 0.0, args.h)
        torch.cuda.synchronize()
        cuda.synchronize()
        print(f"done in {time.perf_counter()-t_compile:.1f} s")

        TC_n = TC0.clone()
        torch.cuda.synchronize()
        cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(args.nsteps):
            TC_n = solver.step(TC_n, step * args.h, args.h)
        torch.cuda.synchronize()
        cuda.synchronize()
        t_numba = time.perf_counter() - t0

        results[strategy] = TC_n.clone()

        # Speedup vs Torch
        if not args.no_torch:
            speedup = t_torch / t_numba
        else:
            speedup = float('nan')

        print(f"  Wall time : {t_numba:.3f} s  ({t_numba/args.nsteps*1e3:.2f} ms/step)")
        print(f"  Speedup vs Torch : {speedup:.2f}x")

        if not args.no_torch and 'torch' in results:
            ref = results['torch']
            err_T  = (TC_n[:, 0]  - ref[:, 0]).abs().max().item()
            err_C  = (TC_n[:, 1:] - ref[:, 1:]).abs().max().item()
            print(f"  Max |ΔT|  vs Torch baseline : {err_T:.3e} K")
            print(f"  Max |ΔC|  vs Torch baseline : {err_C:.3e} mol/m³")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # Per-thread vs Per-block comparison
    # ══════════════════════════════════════════════════════════════════════════
    if 'per_thread' in results and 'per_block' in results:
        pt = results['per_thread']
        pb = results['per_block']
        err_T = (pt[:, 0]  - pb[:, 0]).abs().max().item()
        err_C = (pt[:, 1:] - pb[:, 1:]).abs().max().item()
        print("─── Per-thread vs Per-block consistency ──────────────────────")
        print(f"  Max |ΔT| : {err_T:.3e} K")
        print(f"  Max |ΔC| : {err_C:.3e} mol/m³")
        print()

    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Thread-divergence analysis helper
# ─────────────────────────────────────────────────────────────────────────────

def analyse_thread_divergence(yaml_file, ncells_list, tpb, h, nsteps,
                               strategy='per_thread'):
    """
    Measure wall time for different ncells values to estimate the impact of
    thread divergence (per-thread) or synchronisation overhead (per-block).
    Prints a small table.
    """
    device = 'cuda'; dtype = torch.float64
    y       = yaml_thermo.yamlParser(yaml_file)
    vec_th  = y.populateVectorizedThermoObject(dtype=dtype, device=device)
    reactions = parseReactions(yaml_file)
    r_e, r_tb, r_f, _ = build_reactions(vec_th.species, reactions, vec_th, device, dtype)
    be = ElementaryReactionBatchContext(r_e,  device=device, dtype=dtype)
    bt = ThreeBodyReactionBatchContext(r_tb, device=device, dtype=dtype)
    bf = FalloffReactionBatchContext(r_f,   device=device, dtype=dtype)
    packed = NumbaPacked(vec_th, be, bt, bf)

    print(f"\nThread-divergence analysis — strategy={strategy}  TPB={tpb}")
    print(f"{'ncells':>10}  {'time (s)':>10}  {'ms/step':>10}  {'cell/s':>12}")
    for nc in ncells_list:
        TC0 = make_initial_conditions(yaml_file, nc, device, dtype)
        solver = NumbaSolver(packed, strategy=strategy, tpb=tpb)
        _ = solver.step(TC0.clone(), 0.0, h)    # warm-up
        TC = TC0.clone()
        cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(nsteps):
            TC = solver.step(TC, s*h, h)
        cuda.synchronize()
        t = time.perf_counter() - t0
        print(f"{nc:>10d}  {t:>10.3f}  {t/nsteps*1e3:>10.2f}  {nc*nsteps/t:>12.0f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Numba CUDA kinetics benchmark')
    parser.add_argument('--yaml',    default='gri30.yaml')
    parser.add_argument('--ncells',  type=int,   default=1000)
    parser.add_argument('--nsteps',  type=int,   default=50)
    parser.add_argument('--h',       type=float, default=1e-7)
    parser.add_argument('--tpb',     type=int,   default=64)
    parser.add_argument('--restart', type=int,   default=20)
    parser.add_argument('--no-torch-baseline', dest='no_torch',
                        action='store_true', default=False)
    parser.add_argument('--divergence-analysis', action='store_true', default=False)
    args = parser.parse_args()

    run_benchmark(args)

    if args.divergence_analysis:
        ncells_sweep = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        for strat in ('per_thread', 'per_block'):
            analyse_thread_divergence(
                args.yaml, ncells_sweep,
                tpb=args.tpb, h=args.h, nsteps=10,
                strategy=strat,
            )