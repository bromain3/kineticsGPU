"""
validate_kinetics.py — Validation progressive de la cinétique chimique pure.

Pipeline en 4 étapes (sans réacteur, omega uniquement) :

  Étape 1 : Jacobienne chimique Cantera analytique vs Torch-autograd
            J_chem[s, j] = d(omega_s)/d(xk_j)  avec xk = (T, C_0, ..., C_{n-1})

  Étape 2 : JVP Torch-autograd vs JVP analytique Torch (rxn_sys.jvp)

  Étape 3 : Jacobienne CUDA PT vs Jacobienne Torch-autograd (bloc chimique J[1:,:])

  Étape 4 : JVP CUDA PT et PB vs JVP Torch-autograd

Références :
  - Cantera  : net_production_rates_ddCi (s⁻¹) et net_production_rates_ddT (kmol/m³/s/K)
  - Torch    : autograd via torch.autograd.functional.jacobian sur rxn_sys.omega_total()
  - CUDA PT  : bloc chimique J[1:, :] extrait de CUDAJacobianEvaluatorPT
  - CUDA PB  : bloc chimique J[1:, :] extrait de CUDAJacobianEvaluatorPB
"""

import sys, os
import numpy as np
import time
import warnings

import torch
import cantera as ct
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.autograd.functional import jacobian as torch_jacobian
import numba.cuda as cuda
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ── Imports kineticsGPU ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.claude.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.claude.kernels.per_thread.jacobian   import CUDAJacobianEvaluatorPT
from kineticsGPU.claude.kernels.per_block.jacobian_pb import CUDAJacobianEvaluatorPB
from kineticsGPU.reactionTypesContext import (
    ReactionSystemContext,
    ElementaryReactionBatchContext,
    ThreeBodyReactionBatchContext,
    FalloffReactionBatchContext,
)

# ── Configuration ──────────────────────────────────────────────────────────
yaml_file  = '../gri30.yaml'
dtype_np   = np.float64
dtype_th   = torch.float64
device     = torch.device('cuda:0')
cuda.select_device(device.index)

N_SAMPLES  = 200   # points (T, C) aléatoires
R_GAS      = 8.314_462_618_153_24

os.makedirs('validation_figures', exist_ok=True)

# ── Style matplotlib ───────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'text.usetex': False,          # passe à True si LaTeX disponible
    'figure.figsize': (5, 3.5),
    'figure.dpi': 150,
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 0.9,
    'legend.frameon': False,
})
COLORS = plt.cm.tab10.colors


# ═══════════════════════════════════════════════════════════════════════════
#  Utilitaires
# ═══════════════════════════════════════════════════════════════════════════

def stats(label, a, b, indent='    '):
    diff = np.abs(a - b)
    rel  = diff / (np.abs(b) + 1e-100)
    print(f'{indent}{label}')
    print(f'{indent}  MAE  = {diff.mean():.3e}  ± {diff.std():.3e}   max = {diff.max():.3e}')
    print(f'{indent}  rMAE = {rel.mean():.3e}  ± {rel.std():.3e}   max = {rel.max():.3e}')
    return diff, rel


def boxplot_panel(ax, data_list, labels, ylabel, log=True):
    if log:
        ax.set_yscale('log')
    ax.boxplot(
        [d.ravel() for d in data_list],
        labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.2),
        boxprops=dict(facecolor=COLORS[0], alpha=0.65),
    )
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=15)


def scatter_11(ax, x_flat, y_flat, label_x, label_y, color, n_max=6000):
    rng = np.random.default_rng(42)
    n   = min(n_max, len(x_flat))
    idx = rng.choice(len(x_flat), n, replace=False)
    xf, yf = x_flat[idx], y_flat[idx]
    ax.scatter(xf, yf, s=1, alpha=0.25, color=color, rasterized=True)
    lo, hi = min(xf.min(), yf.min()), max(xf.max(), yf.max())
    ax.axline((lo, lo), slope=1, color='red', linewidth=0.6, linestyle=':', label='x=y')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.legend(fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════
#  Chargement mécanisme
# ═══════════════════════════════════════════════════════════════════════════

print('=' * 70)
print('Chargement du mécanisme GRI30')

gas    = ct.Solution(yaml_file)
y      = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=dtype_th, device=str(device))
rxns   = parseReactions(yaml_file)

r_e, r_tb, r_f, _ = build_reactions(vec_th.species, rxns, vec_th, str(device), dtype_th)
mech               = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype_np)

n_sp     = mech.n_species
n_state  = n_sp + 1
sp_names = gas.species_names

reactions_index_to_type = [r['type'] for r in rxns]

# Système Torch
batched_elem = ElementaryReactionBatchContext(r_e, device=str(device), dtype=dtype_th)
batched_tb   = ThreeBodyReactionBatchContext(r_tb, device=str(device), dtype=dtype_th)
batched_f    = FalloffReactionBatchContext(r_f,    device=str(device), dtype=dtype_th)
rxn_sys      = ReactionSystemContext([batched_elem, batched_tb, batched_f])

print(f'  {n_sp} espèces  |  {mech.elementary.n_reactions} élém.  '
      f'|  {mech.threebody.n_reactions} 3-corps  '
      f'|  {mech.falloff.n_reactions} falloff\n')

# ═══════════════════════════════════════════════════════════════════════════
#  Données aléatoires
# ═══════════════════════════════════════════════════════════════════════════

np.random.seed(42)
T_np = np.random.uniform(800, 4000, N_SAMPLES).astype(dtype_np)
C_np = np.abs(np.random.randn(N_SAMPLES, n_sp).astype(dtype_np)) * 5.0   # mol/m³

T_th = torch.tensor(T_np, device=device, dtype=dtype_th)
C_th = torch.tensor(C_np, device=device, dtype=dtype_th)


# ═══════════════════════════════════════════════════════════════════════════
#  ÉTAPE 1 — Jacobienne chimique : Cantera analytique vs Torch-autograd
#
#  J_chem[i, s, j] = d(omega_s)/d(xk_j)
#    xk = (T, C_0, ..., C_{n-1})   →   j=0 : T,  j=k+1 : C_k
#    s  : indice espèce
#
#  Cantera fournit directement :
#    ddCi  (n_sp × n_sp)  en s⁻¹              →  J_chem[:, 1:]
#    ddT   (n_sp,)         en mol/m³/s/K        →  J_chem[:, 0]
#
#  Torch-autograd calcule la même chose via torch.autograd.functional.jacobian
#  appliqué à la fonction (T_scalar, C_1d) → omega_1d.
# ═══════════════════════════════════════════════════════════════════════════

print('=' * 70)
print('ÉTAPE 1 — J_chem : Cantera analytique  vs  Torch-autograd')

# ── 1a. Référence Cantera ─────────────────────────────────────────────────
print(f'\n  Calcul Cantera (N={N_SAMPLES}) ... ', end='', flush=True)
t0 = time.perf_counter()

# J_can_chem[i, s, j] avec s=ligne (espèce), j=colonne (variable)
J_can_chem = np.zeros((N_SAMPLES, n_sp, n_state), dtype=dtype_np)

for i in range(N_SAMPLES):
    g = ct.Solution(yaml_file)
    g.TP = float(T_np[i]), 101325.0
    g.concentrations = np.maximum(C_np[i], 0.0) * 1e-3   # mol/m³ → kmol/m³

    # ddCi : s⁻¹  (∂(omega_s)/∂(C_k), unités se compensent car vC en mol/m³)
    ddCi = np.array(g.net_production_rates_ddCi)           # (n_sp, n_sp)
    # ddT  : kmol/m³/s/K → ×1e3 → mol/m³/s/K
    ddT  = np.array(g.net_production_rates_ddT) * 1e3      # (n_sp,)

    J_can_chem[i, :, 0]  = ddT
    J_can_chem[i, :, 1:] = ddCi

print(f'{time.perf_counter()-t0:.1f}s')

# ── 1b. Torch-autograd ────────────────────────────────────────────────────
# Wrapper : (T_1d shape(1,), C_2d shape(1, n_sp)) → omega shape(n_sp,)
def _omega_fn_single(T_1, C_2):
    """Calcule omega pour UN seul réacteur (batch=1)."""
    rxn_sys.evaluate_context(T_1, C_2)
    return rxn_sys.omega_total().squeeze(0)   # (n_sp,)


print(f'\n  Calcul Torch-autograd (N={N_SAMPLES}) ... ', end='', flush=True)
t0 = time.perf_counter()

# J_auto_chem[i, s, j]  :  d(omega_s)/d(xk_j)
J_auto_chem = np.zeros((N_SAMPLES, n_sp, n_state), dtype=dtype_np)

for i in range(N_SAMPLES):
    T_i = T_th[i:i+1].detach().requires_grad_(True)    # (1,)
    C_i = C_th[i:i+1].detach().requires_grad_(True)    # (1, n_sp)

    # jacobian retourne tuple (d_omega/dT shape(n_sp,1), d_omega/dC shape(n_sp,1,n_sp))
    J_i = torch_jacobian(_omega_fn_single, (T_i, C_i))
    J_auto_chem[i, :, 0]  = J_i[0].reshape(n_sp).detach().cpu().numpy()
    J_auto_chem[i, :, 1:] = J_i[1].reshape(n_sp, n_sp).detach().cpu().numpy()

t_auto = time.perf_counter() - t0
print(f'{t_auto:.1f}s')

# ── 1c. Métriques ─────────────────────────────────────────────────────────
print('\n  ── Métriques (Torch-autograd vs Cantera) ──')
d1_tot, r1_tot = stats('Total  J_chem  (N_sp × n_state)',
                        J_auto_chem, J_can_chem)
d1_dT,  r1_dT  = stats('Colonne T  (d_omega/dT)',
                        J_auto_chem[:, :, 0], J_can_chem[:, :, 0])
d1_dC,  r1_dC  = stats('Colonnes C (d_omega/dC)',
                        J_auto_chem[:, :, 1:], J_can_chem[:, :, 1:])

# ── 1d. Figures ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
boxplot_panel(axes[0],
              [d1_tot, d1_dT, d1_dC],
              ['Total', 'd(om)/dT', 'd(om)/dC'],
              'MAE (mol m⁻³ s⁻¹)')
axes[0].set_title('Etape 1 — Abs. error')
boxplot_panel(axes[1],
              [r1_tot, r1_dT, r1_dC],
              ['Total', 'd(om)/dT', 'd(om)/dC'],
              'Relative error (–)')
axes[1].set_title('Etape 1 — Rel. error')
fig.suptitle('J_chem : Torch-autograd vs Cantera analytique', fontsize=9)
plt.tight_layout()
plt.savefig('validation_figures/step1_Jchem_auto_vs_cantera.png', bbox_inches='tight')
plt.close()

# Scatter par colonne (T et C séparément)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
scatter_11(axes[0],
           J_can_chem[:, :, 0].ravel(), J_auto_chem[:, :, 0].ravel(),
           'Cantera d(om)/dT', 'Torch-autograd d(om)/dT', COLORS[1])
axes[0].set_title('d(omega)/dT')
scatter_11(axes[1],
           J_can_chem[:, :, 1:].ravel(), J_auto_chem[:, :, 1:].ravel(),
           'Cantera d(om)/dC', 'Torch-autograd d(om)/dC', COLORS[2])
axes[1].set_title('d(omega)/dC')
fig.suptitle('Etape 1 — Scatter : Torch-autograd vs Cantera', fontsize=9)
plt.tight_layout()
plt.savefig('validation_figures/step1_Jchem_scatter.png', bbox_inches='tight')
plt.close()

# MAE par espèce
sp_mae1_dT = np.abs(J_auto_chem[:, :, 0]  - J_can_chem[:, :, 0]).mean(axis=0)   # (n_sp,)
sp_mae1_dC = np.abs(J_auto_chem[:, :, 1:] - J_can_chem[:, :, 1:]).mean(axis=(0, 2))

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
x_sp = np.arange(n_sp)
for ax, sp_mae, title, col in [
    (axes[0], sp_mae1_dT, 'd(omega)/dT  per species', COLORS[1]),
    (axes[1], sp_mae1_dC, 'd(omega)/dC  per species (avg over cols)', COLORS[2]),
]:
    ax.bar(x_sp, sp_mae, color=col, alpha=0.7, width=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('Species index')
    ax.set_ylabel('MAE (mol m⁻³ s⁻¹)')
    ax.set_title(title, fontsize=8)
    for k in np.argsort(sp_mae)[-3:][::-1]:
        ax.text(k, sp_mae[k] * 1.5, sp_names[k],
                ha='center', fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig('validation_figures/step1_Jchem_per_species.png', bbox_inches='tight')
plt.close()

print('  -> Figures : step1_Jchem_auto_vs_cantera.png  /  step1_Jchem_scatter.png  /  step1_Jchem_per_species.png')


# ═══════════════════════════════════════════════════════════════════════════
#  ÉTAPE 2 — JVP chimique : Torch-autograd  vs  analytique Torch (rxn_sys.jvp)
#
#  On calcule Jv_om = J_chem @ v  pour N vecteurs v aléatoires normalisés.
#    v = (vT, vC_0, ..., vC_{n-1})  de dimension n_state
#
#  Version autograd  : J_auto_chem[i] @ v[i]    (réutilise l'étape 1)
#  Version analytique: rxn_sys.jvp(vT, vC)
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ÉTAPE 2 — JVP chimique : Torch-autograd  vs  analytique Torch (rxn_sys.jvp)')

np.random.seed(17)
vT_raw = np.random.randn(N_SAMPLES).astype(dtype_np)
vC_raw = np.random.randn(N_SAMPLES, n_sp).astype(dtype_np)
norms  = np.sqrt(vT_raw**2 + (vC_raw**2).sum(axis=1))
vT_np  = vT_raw / norms
vC_np  = vC_raw / norms[:, None]

# ── 2a. Autograd : J_auto_chem[i] @ v[i] ─────────────────────────────────
v_flat_np   = np.concatenate([vT_np[:, None], vC_np], axis=1)   # (N, n_state)
Jv_auto_chem = np.einsum('ijk,ik->ij', J_auto_chem, v_flat_np)   # (N, n_sp)

# ── 2b. Analytique Torch ─────────────────────────────────────────────────
print('\n  Calcul JVP analytique Torch ... ', end='', flush=True)
t0 = time.perf_counter()

rxn_sys.evaluate_context(T_th, C_th)   # évalue le contexte sur tous les réacteurs

vT_th  = torch.tensor(vT_np, device=device, dtype=dtype_th)
vC_th  = torch.tensor(vC_np, device=device, dtype=dtype_th)
Jv_torch_chem = rxn_sys.jvp(vT_th, vC_th).cpu().numpy()   # (N, n_sp)
torch.cuda.synchronize()
print(f'{time.perf_counter()-t0:.4f}s')

# ── 2c. Métriques ─────────────────────────────────────────────────────────
print('\n  ── Métriques (JVP analytique Torch vs Torch-autograd) ──')
d2, r2 = stats('Jv_om  analytique vs autograd', Jv_torch_chem, Jv_auto_chem)

# ── 2d. Figures ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
boxplot_panel(axes[0], [d2], ['JVP om'], 'MAE (mol m⁻³ s⁻¹)')
axes[0].set_title('Etape 2 — Abs. error')
boxplot_panel(axes[1], [r2], ['JVP om'], 'Relative error (–)')
axes[1].set_title('Etape 2 — Rel. error')
fig.suptitle('JVP chimique : analytique Torch vs Torch-autograd', fontsize=9)
plt.tight_layout()
plt.savefig('validation_figures/step2_JVP_analytic_vs_auto.png', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(5, 4))
scatter_11(ax,
           Jv_auto_chem.ravel(), Jv_torch_chem.ravel(),
           'Autograd JVP_om', 'Analytique Torch JVP_om', COLORS[3])
ax.set_title('Etape 2 — Scatter JVP chimique')
plt.tight_layout()
plt.savefig('validation_figures/step2_JVP_scatter.png', bbox_inches='tight')
plt.close()

sp_mae2 = np.abs(Jv_torch_chem - Jv_auto_chem).mean(axis=0)
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.bar(np.arange(n_sp), sp_mae2, color=COLORS[3], alpha=0.7, width=1.0)
ax.set_yscale('log')
ax.set_xlabel('Species index')
ax.set_ylabel('MAE JVP_om')
ax.set_title('Etape 2 — MAE JVP chimique par espèce', fontsize=8)
for k in np.argsort(sp_mae2)[-3:][::-1]:
    ax.text(k, sp_mae2[k] * 1.5, sp_names[k], ha='center', fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig('validation_figures/step2_JVP_per_species.png', bbox_inches='tight')
plt.close()

print('  -> Figures : step2_JVP_analytic_vs_auto.png  /  step2_JVP_scatter.png  /  step2_JVP_per_species.png')


# ═══════════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — Jacobienne CUDA PT et PB  vs  Torch-autograd
#
#  Note : CUDAJacobianEvaluatorPT/PB calculent la Jacobienne du RHS COMPLET
#  (réacteur adiabatique, n_state × n_state).  Ici on ne valide que le
#  BLOC CHIMIQUE J[1:, :] qui correspond à d(omega)/d(xk).
#  La ligne J[0, :] (d(dT/dt)/d(xk)) n'est PAS validée dans ce script.
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ÉTAPE 3 — Bloc chimique CUDA (PT et PB) vs Torch-autograd')
print('         [J[1:, :] uniquement — d(omega)/d(xk)]')

# Warmup JIT
jac_pt = CUDAJacobianEvaluatorPT(mech, dtype=dtype_np)
jac_pb = CUDAJacobianEvaluatorPB(mech, dtype=dtype_np)

print('\n  Warmup JIT per-thread ... ', end='', flush=True)
jac_pt.evaluate(T_np[:1], C_np[:1])
print('done')
print('  Warmup JIT per-block  ... ', end='', flush=True)
jac_pb.evaluate(T_np[:1], C_np[:1])
print('done')

print(f'\n  CUDA per-thread ({N_SAMPLES} réacteurs) ... ', end='', flush=True)
t0 = time.perf_counter()
J_pt_full = jac_pt.evaluate(T_np, C_np)   # (N, n_state, n_state)
t_pt = time.perf_counter() - t0
print(f'{t_pt*1e3:.2f} ms  ({t_pt/N_SAMPLES*1e6:.1f} µs/réacteur)')

print(f'  CUDA per-block  ({N_SAMPLES} réacteurs) ... ', end='', flush=True)
t0 = time.perf_counter()
J_pb_full = jac_pb.evaluate(T_np, C_np)   # (N, n_state, n_state)
t_pb = time.perf_counter() - t0
print(f'{t_pb*1e3:.2f} ms  ({t_pb/N_SAMPLES*1e6:.1f} µs/réacteur)')

# Extraction du bloc chimique J[1:, :]  →  (N, n_sp, n_state)
J_pt_chem = J_pt_full[:, 1:, :]    # (N, n_sp, n_state)
J_pb_chem = J_pb_full[:, 1:, :]

# Référence autograd (même forme)
# J_auto_chem  (N, n_sp, n_state) — calculé à l'étape 1

print('\n  ── CUDA PT vs Torch-autograd (J_chem) ──')
d3_pt_tot, r3_pt_tot = stats('Total  J_chem',
                              J_pt_chem,         J_auto_chem)
d3_pt_dT,  r3_pt_dT  = stats('Colonne T',
                              J_pt_chem[:, :, 0], J_auto_chem[:, :, 0])
d3_pt_dC,  r3_pt_dC  = stats('Colonnes C',
                              J_pt_chem[:, :, 1:], J_auto_chem[:, :, 1:])

print('\n  ── CUDA PB vs Torch-autograd (J_chem) ──')
d3_pb_tot, r3_pb_tot = stats('Total  J_chem',
                              J_pb_chem,         J_auto_chem)
d3_pb_dT,  r3_pb_dT  = stats('Colonne T',
                              J_pb_chem[:, :, 0], J_auto_chem[:, :, 0])
d3_pb_dC,  r3_pb_dC  = stats('Colonnes C',
                              J_pb_chem[:, :, 1:], J_auto_chem[:, :, 1:])

print('\n  ── CUDA PB vs CUDA PT (J_chem) ──')
d3_pp, r3_pp = stats('Total  J_chem', J_pb_chem, J_pt_chem)

# ── Figures ───────────────────────────────────────────────────────────────
labels3 = ['PT\ntotal', 'PT\ndT', 'PT\ndC',
           'PB\ntotal', 'PB\ndT', 'PB\ndC']
data3_abs = [d3_pt_tot, d3_pt_dT, d3_pt_dC, d3_pb_tot, d3_pb_dT, d3_pb_dC]
data3_rel = [r3_pt_tot, r3_pt_dT, r3_pt_dC, r3_pb_tot, r3_pb_dT, r3_pb_dC]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
boxplot_panel(axes[0], data3_abs, labels3, 'MAE (mol m⁻³ s⁻¹)')
axes[0].set_title('Etape 3 — Abs. error')
boxplot_panel(axes[1], data3_rel, labels3, 'Relative error (–)')
axes[1].set_title('Etape 3 — Rel. error')
fig.suptitle('J_chem CUDA (PT & PB) vs Torch-autograd', fontsize=9)
plt.tight_layout()
plt.savefig('validation_figures/step3_Jchem_CUDA_vs_auto.png', bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
scatter_11(axes[0],
           J_auto_chem.ravel(), J_pt_chem.ravel(),
           'Autograd J_chem', 'CUDA PT J_chem', COLORS[1])
axes[0].set_title('CUDA per-thread')
scatter_11(axes[1],
           J_auto_chem.ravel(), J_pb_chem.ravel(),
           'Autograd J_chem', 'CUDA PB J_chem', COLORS[2])
axes[1].set_title('CUDA per-block')
fig.suptitle('Etape 3 — Scatter J_chem CUDA vs autograd', fontsize=9)
plt.tight_layout()
plt.savefig('validation_figures/step3_Jchem_scatter.png', bbox_inches='tight')
plt.close()

# MAE par espèce
sp_mae3_pt = np.abs(J_pt_chem - J_auto_chem).mean(axis=(0, 2))   # (n_sp,)
sp_mae3_pb = np.abs(J_pb_chem - J_auto_chem).mean(axis=(0, 2))

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for ax, sp_mae, label, col in [
    (axes[0], sp_mae3_pt, 'CUDA PT vs autograd', COLORS[1]),
    (axes[1], sp_mae3_pb, 'CUDA PB vs autograd', COLORS[2]),
]:
    ax.bar(np.arange(n_sp), sp_mae, color=col, alpha=0.7, width=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('Species index')
    ax.set_ylabel('MAE J_chem (avg cols)')
    ax.set_title(label, fontsize=8)
    for k in np.argsort(sp_mae)[-3:][::-1]:
        ax.text(k, sp_mae[k] * 1.5, sp_names[k],
                ha='center', fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig('validation_figures/step3_Jchem_per_species.png', bbox_inches='tight')
plt.close()

# Heatmap erreur relative moyenne
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, J_cuda, label in [
    (axes[0], J_pt_chem, 'CUDA PT vs autograd'),
    (axes[1], J_pb_chem, 'CUDA PB vs autograd'),
]:
    rel_mean = (np.abs(J_cuda - J_auto_chem) / (np.abs(J_auto_chem) + 1e-100)).mean(axis=0)
    im = ax.imshow(np.log10(rel_mean + 1e-20), aspect='auto', origin='upper', cmap='plasma')
    plt.colorbar(im, ax=ax, label='log10 rel. error')
    ax.set_xlabel('Column j  (0=T, j=C_{j-1})')
    ax.set_ylabel('Species row s')
    ax.set_title(label, fontsize=8)
plt.tight_layout()
plt.savefig('validation_figures/step3_Jchem_heatmap.png', bbox_inches='tight')
plt.close()

print('  -> Figures : step3_Jchem_CUDA_vs_auto.png  /  step3_Jchem_scatter.png')
print('               step3_Jchem_per_species.png  /  step3_Jchem_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════════
#  ÉTAPE 4 — JVP chimique : CUDA PT et PB  vs  Torch-autograd
#
#  On réutilise les mêmes vecteurs tangents v = (vT, vC) que l'étape 2.
#
#  Trois sources :
#    a) Autograd    : J_auto_chem[i] @ v[i]        (déjà calculé à l'étape 2)
#    b) CUDA PT     : J_pt_chem[i] @ v[i]
#    c) CUDA PB     : J_pb_chem[i] @ v[i]
#
#  Note : ici on utilise les Jacobiennes calculées à l'étape 3 pour construire
#  les JVP, ce qui isole proprement le bloc chimique.
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ÉTAPE 4 — JVP chimique : CUDA (PT & PB) vs Torch-autograd')
print('         [Jv = J_chem @ v,  v = (vT, vC) aléatoires normalisés]')

# JVP via Jacobienne (déjà calculée)
Jv_pt_chem = np.einsum('ijk,ik->ij', J_pt_chem, v_flat_np)   # (N, n_sp)
Jv_pb_chem = np.einsum('ijk,ik->ij', J_pb_chem, v_flat_np)

# Rappel : Jv_auto_chem et Jv_torch_chem calculés à l'étape 2

print('\n  ── CUDA PT vs Torch-autograd (JVP_om) ──')
d4_pt_auto, r4_pt_auto = stats('PT JVP vs autograd',  Jv_pt_chem, Jv_auto_chem)
d4_pt_anly, r4_pt_anly = stats('PT JVP vs analytique',Jv_pt_chem, Jv_torch_chem)

print('\n  ── CUDA PB vs Torch-autograd (JVP_om) ──')
d4_pb_auto, r4_pb_auto = stats('PB JVP vs autograd',  Jv_pb_chem, Jv_auto_chem)
d4_pb_anly, r4_pb_anly = stats('PB JVP vs analytique',Jv_pb_chem, Jv_torch_chem)

print('\n  ── CUDA PB vs CUDA PT (JVP_om) ──')
d4_pp, r4_pp = stats('PB JVP vs PT JVP', Jv_pb_chem, Jv_pt_chem)

# ── Figures ───────────────────────────────────────────────────────────────
labels4  = ['PT\nvs auto', 'PB\nvs auto', 'PT\nvs analy', 'PB\nvs analy', 'PB\nvs PT']
data4_abs = [d4_pt_auto, d4_pb_auto, d4_pt_anly, d4_pb_anly, d4_pp]
data4_rel = [r4_pt_auto, r4_pb_auto, r4_pt_anly, r4_pb_anly, r4_pp]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
boxplot_panel(axes[0], data4_abs, labels4, 'MAE JVP_om (mol m⁻³ s⁻¹)')
axes[0].set_title('Etape 4 — Abs. error')
boxplot_panel(axes[1], data4_rel, labels4, 'Relative error (–)')
axes[1].set_title('Etape 4 — Rel. error')
fig.suptitle('JVP chimique CUDA (PT & PB) vs Torch-autograd', fontsize=9)
plt.tight_layout()
plt.savefig('validation_figures/step4_JVP_CUDA_vs_auto.png', bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
scatter_11(axes[0],
           Jv_auto_chem.ravel(), Jv_pt_chem.ravel(),
           'Autograd JVP_om', 'CUDA PT JVP_om', COLORS[1])
axes[0].set_title('CUDA per-thread')
scatter_11(axes[1],
           Jv_auto_chem.ravel(), Jv_pb_chem.ravel(),
           'Autograd JVP_om', 'CUDA PB JVP_om', COLORS[2])
axes[1].set_title('CUDA per-block')
fig.suptitle('Etape 4 — Scatter JVP chimique CUDA vs autograd', fontsize=9)
plt.tight_layout()
plt.savefig('validation_figures/step4_JVP_scatter.png', bbox_inches='tight')
plt.close()

sp_mae4_pt = np.abs(Jv_pt_chem - Jv_auto_chem).mean(axis=0)
sp_mae4_pb = np.abs(Jv_pb_chem - Jv_auto_chem).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for ax, sp_mae, label, col in [
    (axes[0], sp_mae4_pt, 'CUDA PT JVP vs autograd', COLORS[1]),
    (axes[1], sp_mae4_pb, 'CUDA PB JVP vs autograd', COLORS[2]),
]:
    ax.bar(np.arange(n_sp), sp_mae, color=col, alpha=0.7, width=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('Species index')
    ax.set_ylabel('MAE JVP_om')
    ax.set_title(label, fontsize=8)
    for k in np.argsort(sp_mae)[-3:][::-1]:
        ax.text(k, sp_mae[k] * 1.5, sp_names[k],
                ha='center', fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig('validation_figures/step4_JVP_per_species.png', bbox_inches='tight')
plt.close()

print('  -> Figures : step4_JVP_CUDA_vs_auto.png  /  step4_JVP_scatter.png  /  step4_JVP_per_species.png')


# ═══════════════════════════════════════════════════════════════════════════
#  RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('RÉSUMÉ — cinétique chimique pure (sans réacteur adiabatique)')
print(f'  N = {N_SAMPLES} réacteurs  |  n_species = {n_sp}  |  n_state = {n_state}')

print(f'\n  Etape 1 — J_chem  (autograd vs Cantera)')
print(f'    Total     MAE = {d1_tot.mean():.3e}   rMAE = {r1_tot.mean():.3e}')
print(f'    dT col    MAE = {d1_dT.mean():.3e}   rMAE = {r1_dT.mean():.3e}')
print(f'    dC cols   MAE = {d1_dC.mean():.3e}   rMAE = {r1_dC.mean():.3e}')

print(f'\n  Etape 2 — JVP_om  (analytique Torch vs autograd)')
print(f'    MAE = {d2.mean():.3e}   rMAE = {r2.mean():.3e}')

print(f'\n  Etape 3 — J_chem  CUDA PT vs autograd')
print(f'    Total     MAE = {d3_pt_tot.mean():.3e}   rMAE = {r3_pt_tot.mean():.3e}')
print(f'  Etape 3 — J_chem  CUDA PB vs autograd')
print(f'    Total     MAE = {d3_pb_tot.mean():.3e}   rMAE = {r3_pb_tot.mean():.3e}')
print(f'  Etape 3 — J_chem  CUDA PB vs PT')
print(f'    Total     MAE = {d3_pp.mean():.3e}   rMAE = {r3_pp.mean():.3e}')

print(f'\n  Etape 4 — JVP_om  CUDA PT vs autograd')
print(f'    MAE = {d4_pt_auto.mean():.3e}   rMAE = {r4_pt_auto.mean():.3e}')
print(f'  Etape 4 — JVP_om  CUDA PB vs autograd')
print(f'    MAE = {d4_pb_auto.mean():.3e}   rMAE = {r4_pb_auto.mean():.3e}')
print(f'  Etape 4 — JVP_om  CUDA PB vs PT')
print(f'    MAE = {d4_pp.mean():.3e}   rMAE = {r4_pp.mean():.3e}')

print(f'\n  Timing — Torch-autograd J: {t_auto:.1f}s'
      f'  |  CUDA PT: {t_pt*1e3:.2f}ms  |  CUDA PB: {t_pb*1e3:.2f}ms')
print(f'  Figures dans : validation_figures/')
print('=' * 70)