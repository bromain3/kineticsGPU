"""
test_reactor_cantera.py
=======================
Intégration d'un réacteur adiabatique à volume constant (GRI-3.0)
avec l'intégrateur Backward Euler GMRES Numba, comparaison à Cantera.

Conditions : CH4/air stœchiométrique, T0=1500 K, P0=101325 Pa.
"""

import time
import math
import numpy as np
from numba import cuda

import kineticsGPU.numba_implementation.thermo as thermo
from kineticsGPU.reactionParser import parseReactions
from kineticsGPU.numba_implementation.yaml_to_GPU_array import (
    build_elementary_gpu_arrays,
    build_threebody_gpu_arrays,
    build_falloff_gpu_arrays,
)
from kineticsGPU.numba_implementation.integrator_host import BackwardEulerNumba

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
#  Configuration
# =============================================================================

YAML_FILE      = '../gri30.yaml'
T0, P0, PHI   = 1700.0, 101325.0, 1.0
T_END          = 1e-3       # s
H_STEP         = 1e-7# s (pas fixe)
NCELLS         = 1
RESTART        = 20
GMRES_TOL      = 1e-6
GMRES_MAXITER  = 50
NEWTON_TOL     = 1e-6
NEWTON_MAXITER = 20


# =============================================================================
#  Setup
# =============================================================================

def setup():
    lecteur      = thermo.yamlParser(YAML_FILE)
    th_obj       = lecteur.populateVectorizedThermoObject()
    reactions    = parseReactions(YAML_FILE)

    elem_arrays   = build_elementary_gpu_arrays(reactions, th_obj)
    tb_arrays     = build_threebody_gpu_arrays (reactions, th_obj)
    fo_arrays     = build_falloff_gpu_arrays   (reactions, th_obj)
    thermo_arrays = th_obj.package_values()

    nspecies   = thermo_arrays[2].shape[0]
    nreac_elem = elem_arrays[8].shape[0]
    nreac_tb   = tb_arrays  [8].shape[0]
    nreac_fo   = fo_arrays  [8].shape[0]
    species    = th_obj.species

    return (elem_arrays, tb_arrays, fo_arrays, thermo_arrays,
            nspecies, nreac_elem, nreac_tb, nreac_fo, species)


def initial_conditions(species, nsp, T, P, phi=1.0):
    X = np.zeros(nsp)
    idx = {s: i for i, s in enumerate(species)}
    X[idx['CH4']] = phi
    X[idx['O2']]  = 2.0
    X[idx['N2']]  = 2.0 * 3.76
    X /= X.sum()
    C_tot = P / (8.314472 * T)
    return X * C_tot


# =============================================================================
#  Cantera
# =============================================================================

def run_cantera(species, T0, P0, phi, t_end):
    try:
        import cantera as ct
    except ImportError:
        print("  Cantera non disponible.")
        return None

    gas = ct.Solution(YAML_FILE)
    gas.TPX = T0, P0, f'CH4:{phi}, O2:2.0, N2:{2*3.76}'
    r   = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    sim.rtol = 1e-10; sim.atol = 1e-14

    T_hist = [r.T]; t_hist = [0.0]
    t0 = time.perf_counter()
    while sim.time < t_end:
        sim.advance(min(sim.time + t_end/100, t_end))
        t_hist.append(sim.time); T_hist.append(r.T)
    t_cantera = time.perf_counter() - t0

    print(f"  Cantera   : {t_cantera*1e3:.1f} ms, T_final = {r.T:.2f} K")
    return np.array(t_hist), np.array(T_hist)


# =============================================================================
#  Main
# =============================================================================

def main():
    print("=" * 55)
    print("Réacteur adiabatique CV — GRI-3.0  CH4/air phi=1")
    print(f"T0={T0} K, t_end={T_END*1e3:.1f} ms, h={H_STEP*1e6:.1f} µs")
    print("=" * 55)

    print("\n[1] Chargement mécanisme...")
    (elem_arrays, tb_arrays, fo_arrays, thermo_arrays,
     nsp, nreac_elem, nreac_tb, nreac_fo, species) = setup()
    N = nsp + 1
    print(f"    nsp={nsp}, elem={nreac_elem}, tb={nreac_tb}, fo={nreac_fo}")

    print("\n[2] Conditions initiales...")
    C0   = initial_conditions(species, nsp, T0, P0, PHI)
    TC0  = np.zeros((NCELLS, N), dtype=np.float64)
    TC0[:, 0]  = T0
    TC0[:, 1:] = C0

    TC_dev = cuda.to_device(TC0)
    TC_out = cuda.device_array_like(TC_dev)

    print("\n[3] Intégrateur...")
    integrator = BackwardEulerNumba(
        elem_arrays, tb_arrays, fo_arrays, thermo_arrays,
        nsp, NCELLS, nreac_elem, nreac_tb, nreac_fo,
        restart=RESTART, gmres_tol=GMRES_TOL, gmres_maxiter=GMRES_MAXITER,
        newton_tol=NEWTON_TOL, newton_maxiter=NEWTON_MAXITER)

    print("\n[4] JIT warm-up...")
    t0 = time.perf_counter()
    integrator.step(TC_dev, TC_out, H_STEP)
    print(f"    {(time.perf_counter()-t0):.2f} s (compile + 1 pas)")

    print(f"\n[5] Intégration sur {T_END*1e3:.1f} ms...")
    TC_dev = cuda.to_device(TC0)
    TC_ping = TC_dev; TC_pong = cuda.device_array_like(TC_dev)
    n_steps = int(math.ceil(T_END / H_STEP))

    # Enregistre T toutes les ~10 µs
    rec_ev  = max(1, int(1e-5 / H_STEP))
    t_hist  = [0.0]; T_hist = [T0]

    t  = 0.0; t0 = time.perf_counter()
    for si in range(n_steps):
        h_eff = min(H_STEP, T_END - t)
        if h_eff < 1e-15: break
        integrator.step(TC_ping, TC_pong, h_eff)
        TC_ping, TC_pong = TC_pong, TC_ping
        t += h_eff
        if si % rec_ev == 0:
            T_now = float(TC_ping[0, 0].copy_to_host())
            t_hist.append(t); T_hist.append(T_now)
    cuda.synchronize()
    t_gpu = time.perf_counter() - t0

    TC_final = TC_ping.copy_to_host()
    T_final  = TC_final[0, 0]
    print(f"    Runtime GPU : {t_gpu*1e3:.1f} ms ({n_steps} pas)")
    print(f"    T_final GPU : {T_final:.2f} K")

    print("\n[6] Référence Cantera...")
    cantera_result = run_cantera(species, T0, P0, PHI, T_END)

    if cantera_result is not None:
        T_c = cantera_result[1][-1]
        print(f"    Erreur rel T : {abs(T_final - T_c)/T_c:.2e}")

    # Tracé
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.array(t_hist)*1e3, T_hist, 'b-', lw=2, label='GPU BE Numba')
    if cantera_result is not None:
        ax.plot(cantera_result[0]*1e3, cantera_result[1], 'r--', lw=2, label='Cantera')
    ax.set(xlabel='t (ms)', ylabel='T (K)',
           title=f'Réacteur adiabatique CV — GRI-3.0, CH4/air φ={PHI}')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_reactor_cantera.png', dpi=150)
    print("\n    Figure : test_reactor_cantera.png")


if __name__ == '__main__':
    main()