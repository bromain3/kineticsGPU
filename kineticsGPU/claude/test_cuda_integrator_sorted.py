"""
test_sorted_integration.py
──────────────────────────
Intègre un batch de réacteurs CH4/air avec et sans tri par température
AVANT CHAQUE PAS, compare les performances et valide contre Cantera.

Tri actuel : température décroissante (proxy raideur), recalculé à chaque pas.
TODO : utiliser dT/dt comme proxy de raideur à la place de T.

Sorties
───────
  • Temps wall-clock avant/après tri
  • Timing breakdown par phase (print_timers)
  • Convergence stats Newton/Krylov (do_count=True)
  • Trajectoires T(t) réacteur 0 vs Cantera  →  sorted_trajectory.png

Run
───
    python test_sorted_integration.py
"""

import sys, os
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.claude.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.claude.integrator.cuda_integrator import CUDABackwardEulerIntegrator

try:
    import cantera as ct
    HAS_CANTERA = True
except ImportError:
    HAS_CANTERA = False
    print("WARNING: Cantera non trouvé — comparaison de référence désactivée.")


# ─────────────────────────────────────────────────────────────────────────────
#  Paramètres
# ─────────────────────────────────────────────────────────────────────────────

YAML_FILE  = '../gri30.yaml'
DEVICE     = 'cuda'
DTYPE      = np.float64

N_REACTORS = 1000
T_MIN      = 1800.0    # K
T_MAX      = 2000.0    # K
PHI_MIN    = 0.5
PHI_MAX    = 1.5
P          = 101325.0  # Pa

T0_INTEG   = 0.0
T_END      = 3e-4      # s
H_STEP     = 1e-7      # s
SAVE_EVERY = 10


# ─────────────────────────────────────────────────────────────────────────────
#  Cantera helpers
# ─────────────────────────────────────────────────────────────────────────────

def cantera_trajectory(T0, C0, t_end, dt, yaml_file=YAML_FILE):
    """Trajectoire complète d'un seul réacteur."""
    gas = ct.Solution(yaml_file)
    gas.TP = float(T0), P
    gas.concentrations = C0 / 1e3
    r   = ct.IdealGasReactor(gas, energy='on')
    sim = ct.ReactorNet([r])
    t, times, T_hist, C_hist = 0.0, [], [], []
    while t < t_end:
        t += dt
        sim.advance(t)
        times.append(t)
        T_hist.append(r.T)
        C_hist.append(np.array(r.thermo.concentrations) * 1e3)
    return np.array(times), np.array(T_hist), np.array(C_hist)


def cantera_final(T0, C0, t_end, yaml_file=YAML_FILE):
    """État final d'un seul réacteur."""
    gas = ct.Solution(yaml_file)
    gas.TP = float(T0), P
    gas.concentrations = C0 / 1e3
    r   = ct.IdealGasReactor(gas, energy='on')
    sim = ct.ReactorNet([r])
    sim.advance(t_end)
    return r.T, np.array(r.thermo.concentrations) * 1e3


# ─────────────────────────────────────────────────────────────────────────────
#  Tri par température — recalculé à chaque pas
# ─────────────────────────────────────────────────────────────────────────────

def sort_by_temperature(TC: np.ndarray):
    """
    Trie les réacteurs par température décroissante.

    Returns
    -------
    idx_sort   : (B,) int — permutation appliquée
    TC_sorted  : (B, N)  — batch réordonné
    idx_unsort : (B,) int — permutation inverse (pour restaurer l'ordre original)

    TODO : remplacer par un tri sur dT/dt (meilleur proxy de raideur).
    """
    idx_sort   = np.argsort(TC[:, 0])[::-1].copy()
    idx_unsort = np.empty_like(idx_sort)
    idx_unsort[idx_sort] = np.arange(len(idx_sort))
    return idx_sort, TC[idx_sort], idx_unsort


# ─────────────────────────────────────────────────────────────────────────────
#  Boucle d'intégration avec tri avant chaque pas
# ─────────────────────────────────────────────────────────────────────────────

def integrate_sorted(integrator, TC0, t0, t_end, h,
                     save_every=1, do_timing=False, do_count=False):
    """
    Intègre TC0 de t0 à t_end avec un tri par T° AVANT CHAQUE PAS.

    À chaque pas :
      1. Trier TC_current par T° décroissante  →  idx_sort, idx_unsort
      2. Charger TC_sorted dans integrator._d_TC_a
      3. Appeler integrator.step(t, h_eff)
      4. Lire TC_sorted_out depuis integrator._d_TC_a  (ping-pong déjà swappé)
      5. Désortir  →  TC_current = TC_sorted_out[idx_unsort]

    Les trajectoires sauvegardées sont dans l'ordre ORIGINAL (après désort),
    ce qui permet de suivre chaque réacteur de bout en bout sans bookkeeping.

    Returns
    -------
    TC_hist   : (n_saved, B, N_state)  ordre original
    t_hist    : (n_saved,)
    conv_hist : (n_saved, B)           ordre original
    """
    n_steps    = int(math.ceil((t_end - t0) / h))
    TC_current = np.ascontiguousarray(TC0, dtype=integrator.dtype)

    TC_hist, t_hist, conv_hist = [], [], []
    t = t0

    for step_i in range(n_steps):
        h_eff = min(h, t_end - t)
        if h_eff <= 0.0:
            break

        # 1. Tri
        _, TC_sorted, idx_unsort = sort_by_temperature(TC_current)

        # 2. Charger dans le buffer ping-pong de l'intégrateur
        integrator._d_TC_a.copy_to_device(TC_sorted)

        # 3. Un pas (step() swaps _d_TC_a ↔ _d_TC_b en interne)
        integrator.step(t, h_eff, do_timing=do_timing, do_count=do_count)

        t += h_eff

        # 4. Lire le résultat — après le swap, _d_TC_a contient le nouvel état
        TC_sorted_out = integrator._d_TC_a.copy_to_host()

        # 5. Désortir → ordre original
        TC_current = TC_sorted_out[idx_unsort]

        # Sauvegarde (conv aussi désortée pour cohérence)
        if step_i % save_every == 0:
            conv = integrator._d_conv.copy_to_host()[idx_unsort]
            TC_hist.append(TC_current.copy())
            t_hist.append(t)
            conv_hist.append(conv)

    from numba import cuda as _cuda
    _cuda.synchronize()

    return (
        np.stack(TC_hist,   axis=0),
        np.array(t_hist),
        np.stack(conv_hist, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Stats convergence
# ─────────────────────────────────────────────────────────────────────────────

def print_iter_stats(iters: np.ndarray, label: str):
    """Affiche un résumé Newton/Krylov depuis buf (B, 2)."""
    newton = iters[:, 0].astype(float)
    krylov = iters[:, 1].astype(float)
    W = 52
    print(f"\n  {'─'*W}")
    print(f"  Convergence stats — {label}")
    print(f"  {'─'*W}")
    print(f"  {'':20s}  {'Newton':>10}  {'Krylov':>10}")
    print(f"  {'mean':<20}  {newton.mean():>10.2f}  {krylov.mean():>10.2f}")
    print(f"  {'std':<20}  {newton.std():>10.2f}  {krylov.std():>10.2f}")
    print(f"  {'min':<20}  {newton.min():>10.0f}  {krylov.min():>10.0f}")
    print(f"  {'max':<20}  {newton.max():>10.0f}  {krylov.max():>10.0f}")
    print(f"  {'P95':<20}  {np.percentile(newton,95):>10.1f}  {np.percentile(krylov,95):>10.1f}")
    print(f"  {'─'*W}")


# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper mesure
# ─────────────────────────────────────────────────────────────────────────────

def integrate_and_measure(integrator, TC0, label, use_sort=False):
    """
    Lance l'intégration (avec ou sans tri par pas) et affiche les métriques.

    Returns
    -------
    TC_hist, t_hist, conv_hist, iters_last, wall_time
    """
    print(f"\n{'='*60}")
    print(f"  Intégration : {label}")
    print(f"{'='*60}")

    n_steps = int(math.ceil((T_END - T0_INTEG) / H_STEP))
    integrator.reset_timers()

    t_wall_start = time.perf_counter()

    if use_sort:
        TC_hist, t_hist, conv_hist = integrate_sorted(
            integrator, TC0,
            t0         = T0_INTEG,
            t_end      = T_END,
            h          = H_STEP,
            save_every = SAVE_EVERY,
            do_timing  = True,
            do_count   = True,
        )
    else:
        TC_hist, t_hist, conv_hist = integrator.integrate(
            TC0,
            t0         = T0_INTEG,
            t_end      = T_END,
            h          = H_STEP,
            save_every = SAVE_EVERY,
            do_timing  = True,
            do_count   = True,
        )

    t_wall = time.perf_counter() - t_wall_start

    print(f"\n  Wall time  : {t_wall:.3f} s")
    print(f"  Steps/s    : {n_steps * integrator.B / t_wall:.2e}")

    iters_last = integrator._d_iters.copy_to_host()
    n_conv     = int(conv_hist[-1].sum())
    print(f"  Réacteurs convergés (dernier pas) : {n_conv}/{integrator.B}")

    T_final = TC_hist[-1][:, 0]
    print(f"  T finale : [{T_final.min():.1f}, {T_final.max():.1f}] K")

    integrator.print_timers(n_steps=n_steps)
    print_iter_stats(iters_last, label)

    return TC_hist, t_hist, conv_hist, iters_last, t_wall


# ─────────────────────────────────────────────────────────────────────────────
#  Tracé trajectoires
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(
    t_gpu, T_gpu_unsorted, T_gpu_sorted,
    t_ct,  T_ct,
    sp_names, C_gpu_unsorted, C_gpu_sorted, C_ct,
    top4_idx,
    save_path="sorted_trajectory.png",
):
    n_panels = 1 + len(top4_idx)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(10, 3 * n_panels),
                             sharex=True)
    fig.suptitle("Réacteur 0 — Trajectoires GPU (trié/non trié) vs Cantera", fontsize=12)

    ax = axes[0]
    ax.plot(t_ct  * 1e3, T_ct,           'k-',  lw=1.8, label='Cantera')
    ax.plot(t_gpu * 1e3, T_gpu_unsorted, 'b--', lw=1.5, label='GPU sans tri')
    ax.plot(t_gpu * 1e3, T_gpu_sorted,   'r:',  lw=1.5, label='GPU tri/pas')
    ax.set_ylabel('T (K)')
    ax.legend(loc='best', fontsize=8)
    ax.set_title('Température')

    for ax, idx in zip(axes[1:], top4_idx):
        name = sp_names[idx]
        ax.plot(t_ct  * 1e3, C_ct[:, idx],          'k-',  lw=1.8, label='Cantera')
        ax.plot(t_gpu * 1e3, C_gpu_unsorted[:, idx], 'b--', lw=1.5, label='GPU sans tri')
        ax.plot(t_gpu * 1e3, C_gpu_sorted[:, idx],   'r:',  lw=1.5, label='GPU tri/pas')
        ax.set_ylabel(f'[{name}] (mol/m³)')
        ax.legend(loc='best', fontsize=8)

    axes[-1].set_xlabel('Temps (ms)')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot sauvegardé : {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Mécanisme ─────────────────────────────────────────────────────────
    print("="*60)
    print("  Chargement du mécanisme...")
    print("="*60)
    y      = yaml_thermo.yamlParser(YAML_FILE)
    vec_th = y.populateVectorizedThermoObject(dtype=torch.float64, device=DEVICE)
    reactions = parseReactions(YAML_FILE)
    r_e, r_tb, r_f, system = build_reactions(
        vec_th.species, reactions, vec_th, DEVICE, torch.float64
    )
    print(f"  {len(r_e)} élémentaires, {len(r_tb)} three-body, {len(r_f)} falloff")
    mech = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=DTYPE)
    print(f"  n_species={mech.n_species}, n_reactions={mech.n_reactions_total}")

    # ── Conditions initiales ──────────────────────────────────────────────
    print(f"\n  Setup de {N_REACTORS} réacteurs CH4/air...")
    rng  = np.random.default_rng(0)
    T0s  = rng.uniform(T_MIN, T_MAX, N_REACTORS)
    phis = rng.uniform(PHI_MIN, PHI_MAX, N_REACTORS)
    C0s  = np.zeros((N_REACTORS, mech.n_species), dtype=DTYPE)

    if HAS_CANTERA:
        gas_ref = ct.Solution(YAML_FILE)
        for i, (T0, phi) in enumerate(zip(T0s, phis)):
            gas_ref.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
            gas_ref.TP = T0, P
            C0s[i] = np.array(gas_ref.concentrations) * 1e3

    TC0 = np.zeros((N_REACTORS, mech.n_species + 1), dtype=DTYPE)
    TC0[:, 0]  = T0s
    TC0[:, 1:] = C0s

    # ── Intégrateur ───────────────────────────────────────────────────────
    print("\n  Instanciation de l'intégrateur...")
    integrator = CUDABackwardEulerIntegrator(
        mechanism      = mech,
        n_reactors     = N_REACTORS,
        newton_tol     = 1e-6,
        newton_maxiter = 20,
        reuse_jacobian = True,
        linear_solver  = 'gmres',
        gmres_restart  = 20,
        gmres_tol      = 1e-6,
        gmres_maxiter  = 50,
        dtype          = DTYPE,
    )

    # ── Run sans tri ──────────────────────────────────────────────────────
    TC_hist_u, t_hist_u, conv_hist_u, iters_u, t_wall_u = \
        integrate_and_measure(integrator, TC0, label="SANS TRI", use_sort=False)

    # ── Run avec tri avant chaque pas ─────────────────────────────────────
    TC_hist_s, t_hist_s, conv_hist_s, iters_s, t_wall_s = \
        integrate_and_measure(integrator, TC0, label="TRI PAR PAS (T°)", use_sort=True)

    # ── Résumé comparatif ─────────────────────────────────────────────────
    speedup = t_wall_u / t_wall_s if t_wall_s > 0 else 1.0
    print(f"\n{'='*60}")
    print(f"  Résumé comparatif")
    print(f"{'─'*60}")
    print(f"  Wall time sans tri     : {t_wall_u:.3f} s")
    print(f"  Wall time tri/pas      : {t_wall_s:.3f} s")
    print(f"  Speedup kernel net     : {speedup:.3f}×")
    print(f"  Δ Krylov mean          : "
          f"{iters_u[:,1].mean():.2f} → {iters_s[:,1].mean():.2f}")
    print(f"{'='*60}\n")

    # ── Comparaison Cantera (réacteur 0) ──────────────────────────────────
    if HAS_CANTERA:
        print("  Comparaison Cantera pour le réacteur 0...")

        T_ref, C_ref = cantera_final(T0s[0], C0s[0], T_END)

        # Les deux TC_hist sont dans l'ordre original (désort appliqué)
        # → réacteur 0 toujours à l'index 0
        T_gpu_u = TC_hist_u[-1][0, 0]
        C_gpu_u = TC_hist_u[-1][0, 1:]
        T_gpu_s = TC_hist_s[-1][0, 0]
        C_gpu_s = TC_hist_s[-1][0, 1:]

        print(f"\n  Température finale réacteur 0 :")
        print(f"    Cantera      : {T_ref:.4f} K")
        print(f"    GPU sans tri : {T_gpu_u:.4f} K  (|ΔT|={abs(T_gpu_u-T_ref):.2e})")
        print(f"    GPU tri/pas  : {T_gpu_s:.4f} K  (|ΔT|={abs(T_gpu_s-T_ref):.2e})")

        sp_names = mech.thermo.species_names
        top5     = np.argsort(np.abs(C_ref))[-5:][::-1]
        print(f"\n  Concentrations top-5 (réacteur 0) :")
        print(f"  {'Espèce':10s}  {'Cantera':>12}  {'GPU s/tri':>12}  "
              f"{'GPU tri/pas':>12}  {'err s/tri':>10}  {'err tri/pas':>10}")
        for idx in top5:
            denom = abs(C_ref[idx]) + 1e-30
            print(f"  {sp_names[idx]:10s}  {C_ref[idx]:>12.4e}  "
                  f"{C_gpu_u[idx]:>12.4e}  {C_gpu_s[idx]:>12.4e}  "
                  f"{abs(C_gpu_u[idx]-C_ref[idx])/denom:>10.2e}  "
                  f"{abs(C_gpu_s[idx]-C_ref[idx])/denom:>10.2e}")

        # Trajectoires Cantera
        print("\n  Collecte trajectoire Cantera réacteur 0...")
        t_ct, T_ct, C_ct = cantera_trajectory(
            T0s[0], C0s[0], T_END, dt=H_STEP * SAVE_EVERY
        )

        # Les deux trajectoires GPU ont réacteur 0 à l'index 0
        t_gpu_traj   = np.array(t_hist_u)
        T_gpu_u_traj = TC_hist_u[:, 0, 0]
        C_gpu_u_traj = TC_hist_u[:, 0, 1:]
        T_gpu_s_traj = TC_hist_s[:, 0, 0]
        C_gpu_s_traj = TC_hist_s[:, 0, 1:]

        top4 = np.argsort(C_ref)[-4:][::-1]

        plot_trajectories(
            t_gpu_traj, T_gpu_u_traj, T_gpu_s_traj,
            t_ct, T_ct,
            sp_names, C_gpu_u_traj, C_gpu_s_traj, C_ct,
            top4,
            save_path="sorted_trajectory.png",
        )

    print("\n" + "="*60)
    print("  Test terminé.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()