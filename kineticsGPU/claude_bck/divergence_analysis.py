"""
run_divergence_analysis.py
──────────────────────────
Script d'analyse de divergence de threads, prêt à lancer.

Usage
─────
    python run_divergence_analysis.py

Ce qu'il fait
─────────────
1. Lance le kernel GMRES sur ton batch de réacteurs avec do_count=True
2. Récupère _d_iters (n_newton, n_krylov) par thread depuis le buffer interne
3. Calcule et affiche les métriques de divergence par warp
4. Estime le speedup possible avec un tri par raideur
5. Re-lance avec les réacteurs triés et compare avant/après
6. Affiche 6 graphiques : distributions, efficacité par warp, spread max/min

Seule chose à adapter : la section "── Adapter ici ──" ci-dessous.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # pas d'affichage interactif sur serveur — changer en "TkAgg" en local
import matplotlib.pyplot as plt
import time
import sys, os
import torch

# ── Adapter ici ──────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.claude_bck.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.claude_bck.integrator.cuda_integrator import CUDABackwardEulerIntegrator
from numba import cuda

yaml_file      = '../gri30.yaml'
device_torch   = 'cuda'
dtype          = np.float64

# Paramètres du batch
B     = 8192      # nombre de réacteurs — doit être représentatif
h     = 1e-7      # pas de temps à analyser
T_min = 900.0     # K
T_max = 2200.0    # K
P     = 101325.0  # Pa

# ── Construction du mécanisme ─────────────────────────────────────────────────
print("Loading mechanism...")
y      = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=torch.float64, device=device_torch)
reactions = parseReactions(yaml_file)
r_e, r_tb, r_f, system = build_reactions(
    vec_th.species, reactions, vec_th, device_torch, torch.float64
)
mech = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype)
n_species = mech.n_species
print(f"  n_species={n_species}, n_reactions={mech.n_reactions_total}")

# ── Conditions initiales ──────────────────────────────────────────────────────
rng = np.random.default_rng(42)
T0  = rng.uniform(T_min, T_max, B).astype(dtype)

try:
    import cantera as ct
    gas_ref = ct.Solution(yaml_file)
    C0 = np.zeros((B, n_species), dtype=dtype)
    phis = rng.uniform(0.5, 1.5, B)
    for i, (Ti, phi) in enumerate(zip(T0, phis)):
        gas_ref.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
        gas_ref.TP = Ti, P
        C0[i] = np.array(gas_ref.concentrations) * 1e3
    print("  Conditions initiales CH4/air via Cantera")
except ImportError:
    C0 = rng.uniform(1e-6, 1e-2, (B, n_species)).astype(dtype)
    print("  WARNING: Cantera absent — concentrations aléatoires")

TC0 = np.empty((B, n_species + 1), dtype=dtype)
TC0[:, 0]  = T0
TC0[:, 1:] = C0

# ── Création de l'intégrateur ─────────────────────────────────────────────────
integ = CUDABackwardEulerIntegrator(
    mechanism      = mech,
    n_reactors     = B,
    linear_solver  = 'gmres',
    gmres_restart  = 10,
    gmres_tol      = 1e-6,
    gmres_maxiter  = 5,
    newton_tol     = 1e-8,
    newton_maxiter = 10,
    dtype          = dtype,
)

# ─────────────────────────────────────────────────────────────────────────────

WARP_SIZE         = cuda.get_current_device().WARP_SIZE
THREADS_PER_BLOCK = 64   # doit correspondre à integrator.py


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers d'analyse
# ─────────────────────────────────────────────────────────────────────────────

def _warp_stats(krylov: np.ndarray, warp_size: int = WARP_SIZE):
    """Calcule efficacité, max et min par warp."""
    n = len(krylov)
    n_warps = int(np.ceil(n / warp_size))
    eff  = np.zeros(n_warps)
    wmax = np.zeros(n_warps, dtype=int)
    wmin = np.zeros(n_warps, dtype=int)
    for w in range(n_warps):
        s = w * warp_size
        e = min(s + warp_size, n)
        k  = krylov[s:e]
        mx = k.max()
        wmax[w] = int(mx)
        wmin[w] = int(k.min())
        eff[w]  = k.mean() / mx if mx > 0 else 1.0
    return eff, wmax, wmin


def analyse_and_print(iters: np.ndarray, label: str = ""):
    newton = iters[:, 0].astype(float)
    krylov = iters[:, 1].astype(float)
    eff, wmax, wmin = _warp_stats(krylov)
    spread = wmax - wmin

    n       = len(krylov)
    n_warps = len(eff)
    total_paid   = sum(wmax[w] * min(WARP_SIZE, n - w*WARP_SIZE) for w in range(n_warps))
    total_useful = krylov.sum()
    wasted = 1.0 - total_useful / total_paid if total_paid > 0 else 0.0

    W = 64
    print(f"\n{'─'*W}")
    print(f"  {label}  [B={n}, warps={n_warps}]")
    print(f"{'─'*W}")
    print(f"  {'Metric':<36}  {'Newton':>8}  {'Krylov':>8}")
    print(f"  {'─'*36}  {'─'*8}  {'─'*8}")
    print(f"  {'mean':<36}  {newton.mean():>8.2f}  {krylov.mean():>8.2f}")
    print(f"  {'std':<36}  {newton.std():>8.2f}  {krylov.std():>8.2f}")
    print(f"  {'min / max':<36}  {int(newton.min()):>3}/{int(newton.max()):<4}  {int(krylov.min()):>3}/{int(krylov.max()):<4}")
    print(f"  {'P95 Krylov':<36}  {'':>8}  {np.percentile(krylov,95):>8.1f}")
    print()
    print(f"  {'Warp Krylov efficiency (mean)':<36}  {'':>8}  {eff.mean():>8.3f}")
    print(f"  {'Warp Krylov efficiency (min)':<36}  {'':>8}  {eff.min():>8.3f}")
    print(f"  {'Warp Krylov efficiency (median)':<36}  {'':>8}  {np.median(eff):>8.3f}")
    print(f"  {'Mean warp spread (max-min)':<36}  {'':>8}  {spread.mean():>8.1f}")
    print(f"  {'Global wasted Krylov work':<36}  {'':>8}  {wasted*100:>7.1f}%")
    print()
    ke = eff.mean()
    if   ke > 0.90: verdict = "Faible divergence ✓"
    elif ke > 0.70: verdict = "Divergence modérée — tri recommandé"
    elif ke > 0.50: verdict = "Divergence significative — tri important"
    else:           verdict = "Divergence sévère — tri indispensable ✗"
    print(f"  Diagnostic : {verdict}")
    print(f"{'─'*W}\n")

    return dict(
        newton=newton, krylov=krylov,
        eff=eff, wmax=wmax, wmin=wmin, spread=spread,
        wasted=wasted,
    )


def run_step_with_counters(integrator: CUDABackwardEulerIntegrator,
                           TC0: np.ndarray,
                           t: float = 0.0,
                           h: float = 1e-7) -> np.ndarray:
    """
    Copie TC0 dans le buffer interne, lance un pas avec do_count=True,
    et retourne buf_iters (B, 2) [newton_iters, krylov_iters] sur host.
    """
    # Charger l'état initial dans le buffer ping-pong de l'intégrateur
    integrator._d_TC_a.copy_to_device(
        np.ascontiguousarray(TC0, dtype=integrator.dtype)
    )
    # Remettre les compteurs à zéro
    import numpy as _np
    integrator._d_iters.copy_to_device(
        _np.zeros((integrator.B, 2), dtype=_np.int64)
    )

    # Un seul pas avec comptage activé
    integrator.step(t, h, do_timing=False, do_count=True)

    from numba import cuda as _cuda
    _cuda.synchronize()

    return integrator._d_iters.copy_to_host()   # (B, 2) int32


def sort_by_temperature(TC0: np.ndarray):
    """Tri par température décroissante — proxy de raideur."""
    idx = np.argsort(TC0[:, 0])[::-1].copy()
    return idx, TC0[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(r_before: dict, r_after: dict, save_path: str = "divergence.png"):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Thread Divergence Analysis — Before vs After Stiffness Sort", fontsize=13)

    configs = [
        (r_before, "Before sort"),
        (r_after,  "After sort"),
    ]

    for col, (r, label) in enumerate(configs):
        krylov  = r["krylov"]
        eff     = r["eff"]
        wmax    = r["wmax"]
        wmin    = r["wmin"]
        warp_ids = np.arange(len(eff))

        # ── Row 0 : distribution Krylov ───────────────────────────────────
        ax   = axes[0, col]
        bins = np.arange(max(0, krylov.min()-0.5), krylov.max()+1.5, 1)
        ax.hist(krylov, bins=bins, color="#4C72B0", edgecolor="white", lw=0.3)
        ax.axvline(krylov.mean(), color="red",    ls="--", lw=1.5,
                   label=f"μ={krylov.mean():.1f}")
        ax.axvline(np.percentile(krylov, 95), color="orange", ls=":", lw=1.5,
                   label=f"P95={np.percentile(krylov,95):.0f}")
        ax.set_xlabel("Krylov iters per reactor")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} — Krylov distribution")
        ax.legend(fontsize=8)

        # ── Row 1 : efficacité warp ───────────────────────────────────────
        ax = axes[1, col]
        colors = ["#55A868" if e >= 0.9 else "#DD8452" if e >= 0.7 else "#C44E52"
                  for e in eff]
        ax.bar(warp_ids, eff, color=colors, width=1.0, linewidth=0)
        ax.axhline(eff.mean(), color="red", ls="--", lw=1.5,
                   label=f"μ={eff.mean():.3f}")
        ax.axhline(0.90, color="green", ls=":", lw=1.0, alpha=0.7, label="0.90 target")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Warp index")
        ax.set_ylabel("Efficiency  (mean/max Krylov iters)")
        ax.set_title(f"{label} — Warp efficiency")
        ax.legend(fontsize=8)

        # ── Row 2 : spread max-min par warp ──────────────────────────────
        ax = axes[2, col]
        ax.bar(warp_ids, wmax, color="#C44E52", width=1.0, linewidth=0, label="max")
        ax.bar(warp_ids, wmin, color="#4C72B0", width=1.0, linewidth=0, label="min")
        spread_mean = (wmax - wmin).mean()
        ax.set_xlabel("Warp index")
        ax.set_ylabel("Krylov iterations")
        ax.set_title(f"{label} — Max/min per warp  (mean spread={spread_mean:.1f})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure sauvegardée : {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "═"*64)
    print("  Thread Divergence Analysis")
    print("═"*64)

    # ── 1. Run sans tri ───────────────────────────────────────────────────
    print("\n[1/4] Kernel sans tri (ordre original)...")
    t0_wall = time.perf_counter()
    iters_unsorted = run_step_with_counters(integ, TC0, t=0.0, h=h)
    dt_unsorted = time.perf_counter() - t0_wall
    print(f"      Temps kernel : {dt_unsorted*1e3:.2f} ms")

    r_before = analyse_and_print(iters_unsorted, "SANS TRI")

    # ── 2. Tri et estimation ──────────────────────────────────────────────
    print("[2/4] Estimation du gain par tri (simulation sur les compteurs)...")
    idx_sort, TC0_sorted = sort_by_temperature(TC0)

    krylov_sorted_sim = r_before["krylov"][idx_sort]
    eff_sim, _, _ = _warp_stats(krylov_sorted_sim)
    print(f"      Efficacité warp estimée après tri : {eff_sim.mean():.3f}")

    # ── 3. Run avec tri ───────────────────────────────────────────────────
    print("\n[3/4] Kernel avec tri par température (raideur)...")
    t0_wall = time.perf_counter()
    iters_sorted = run_step_with_counters(integ, TC0_sorted, t=0.0, h=h)
    dt_sorted = time.perf_counter() - t0_wall
    print(f"      Temps kernel : {dt_sorted*1e3:.2f} ms")

    r_after = analyse_and_print(iters_sorted, "AVEC TRI PAR T°")

    # ── 4. Résumé comparatif ──────────────────────────────────────────────
    print("═"*64)
    print("  Résumé comparatif")
    print("─"*64)
    print(f"  Efficacité warp Krylov  : {r_before['eff'].mean():.3f}  →  {r_after['eff'].mean():.3f}")
    print(f"  Travail Krylov gaspillé : {r_before['wasted']*100:.1f}%  →  {r_after['wasted']*100:.1f}%")
    print(f"  Temps kernel            : {dt_unsorted*1e3:.2f} ms  →  {dt_sorted*1e3:.2f} ms")
    speedup_real = dt_unsorted / dt_sorted if dt_sorted > 0 else 1.0
    print(f"  Speedup mesuré          : {speedup_real:.2f}×")
    print("═"*64 + "\n")

    # ── 5. Graphiques ─────────────────────────────────────────────────────
    print("[4/4] Génération des graphiques...")
    plot_comparison(r_before, r_after, save_path="divergence_analysis.png")

    print("Terminé.\n")