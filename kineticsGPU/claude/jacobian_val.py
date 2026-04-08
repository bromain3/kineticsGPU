"""
validate_jacobian_chem_cuda.py — Validation directe : Jacobienne chimique CUDA vs Cantera.

Compare J_chem[s, j] = d(omega_s)/d(xk_j)  avec xk = (T, C_0, ..., C_{n-1})

Sources comparées :
  - Cantera  : net_production_rates_ddCi (s⁻¹) et net_production_rates_ddT (mol/m³/s/K)
  - CUDA PT  : CUDAJacobianChemEvaluatorPT  →  J[b, s, j]  (n_sp × n_state)
  - CUDA PB  : CUDAJacobianChemEvaluatorPB  →  J[b, s, j]  (n_sp × n_state)

Sorties :
  - Console  : MAE / rMAE par bloc (colonne T, colonnes C, total)
  - Figures  : heatmaps d'erreurs absolues et relatives moyennées sur les réacteurs
               + scatter plots + MAE par espèce et par colonne
               → dossier jac_chem_figures/
"""

import sys, os
import numpy as np
import time
import warnings

import cantera as ct
import matplotlib
import matplotlib.pyplot as plt
import numba.cuda as cuda
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.claude.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.claude.kernels.per_thread.jacobian import CUDAJacobianChemEvaluatorPT
from kineticsGPU.claude.kernels.per_block.jacobian_pb import CUDAJacobianChemEvaluatorPB

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

yaml_file = '../gri30.yaml'
dtype_np  = np.float64
dtype_th  = torch.float64
device    = torch.device('cuda:0')
cuda.select_device(device.index)

N_SAMPLES = 200   # nombre de points (T, C) aléatoires

os.makedirs('jac_chem_figures', exist_ok=True)

matplotlib.rcParams.update({
    'text.usetex': False,
    'figure.dpi': 150,
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
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
    print(f'{indent}  MAE  = {diff.mean():.3e}  ±{diff.std():.3e}   max={diff.max():.3e}')
    print(f'{indent}  rMAE = {rel.mean():.3e}  ±{rel.std():.3e}   max={rel.max():.3e}')
    return diff, rel


def heatmap(ax, mat, title, cbar_label, cmap='plasma'):
    im = ax.imshow(mat, aspect='auto', origin='upper', cmap=cmap)
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel('Column j  (0=T,  j=C_{j-1})')
    ax.set_ylabel('Species s')
    ax.set_title(title, fontsize=8)
    return im


def scatter_11(ax, x_flat, y_flat, label_x, label_y, color, n_max=8000):
    rng = np.random.default_rng(42)
    n   = min(n_max, len(x_flat))
    idx = rng.choice(len(x_flat), n, replace=False)
    xf, yf = x_flat[idx], y_flat[idx]
    ax.scatter(xf, yf, s=1, alpha=0.25, color=color, rasterized=True)
    lo = min(xf.min(), yf.min())
    hi = max(xf.max(), yf.max())
    ax.axline((lo, lo), slope=1, color='red', lw=0.6, linestyle=':', label='y=x')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.legend(fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════
#  Chargement mécanisme
# ═══════════════════════════════════════════════════════════════════════════

print('=' * 70)
print('Chargement du mécanisme')

gas    = ct.Solution(yaml_file)
y      = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=dtype_th, device=str(device))
rxns   = parseReactions(yaml_file)

r_e, r_tb, r_f, _ = build_reactions(vec_th.species, rxns, vec_th, str(device), dtype_th)
mech               = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype_np)

n_sp     = mech.n_species
n_state  = n_sp + 1
sp_names = gas.species_names

print(f'  {n_sp} espèces  |  {mech.elementary.n_reactions} élém.  '
      f'|  {mech.threebody.n_reactions} 3-corps  '
      f'|  {mech.falloff.n_reactions} falloff\n')


# ═══════════════════════════════════════════════════════════════════════════
#  Données aléatoires
# ═══════════════════════════════════════════════════════════════════════════

np.random.seed(42)
T_np = np.random.uniform(800, 4000, N_SAMPLES).astype(dtype_np)
C_np = np.abs(np.random.randn(N_SAMPLES, n_sp).astype(dtype_np)) * 5.0   # mol/m³


# ═══════════════════════════════════════════════════════════════════════════
#  Référence Cantera
#  J_can[i, s, j]  :  d(omega_s)/d(xk_j)
#    j=0       → d(omega_s)/dT         (mol/m³/s/K)
#    j=k+1     → d(omega_s)/d(C_k)     (s⁻¹)
# ═══════════════════════════════════════════════════════════════════════════

print('=' * 70)
print(f'Calcul référence Cantera (N={N_SAMPLES}) ...', end=' ', flush=True)
t0 = time.perf_counter()

J_can = np.zeros((N_SAMPLES, n_sp, n_state), dtype=dtype_np)

for i in range(N_SAMPLES):
    g = ct.Solution(yaml_file)
    g.TP = float(T_np[i]), 101325.0
    g.concentrations = np.maximum(C_np[i], 0.0) * 1e-3   # mol/m³ → kmol/m³

    # ddCi  : (n_sp, n_sp)  en s⁻¹
    J_can[i, :, 1:] = np.array(g.net_production_rates_ddCi)
    # ddT   : (n_sp,)  en kmol/m³/s/K  →  ×1e3  → mol/m³/s/K
    J_can[i, :, 0]  = np.array(g.net_production_rates_ddT) * 1e3

print(f'{time.perf_counter()-t0:.1f}s')


# ═══════════════════════════════════════════════════════════════════════════
#  Jacobiennes CUDA
# ═══════════════════════════════════════════════════════════════════════════

print('\nChargement et warmup JIT CUDA ...', end=' ', flush=True)
jac_pt = CUDAJacobianChemEvaluatorPT(mech, dtype=dtype_np)
jac_pb = CUDAJacobianChemEvaluatorPB(mech, dtype=dtype_np)
jac_pt.evaluate(T_np[:1], C_np[:1])
jac_pb.evaluate(T_np[:1], C_np[:1])
print('done')

print(f'\nCUDA per-thread ({N_SAMPLES} réacteurs) ...', end=' ', flush=True)
t0 = time.perf_counter()
J_pt = jac_pt.evaluate(T_np, C_np)   # (N, n_sp, n_state)
t_pt = time.perf_counter() - t0
print(f'{t_pt*1e3:.1f} ms  ({t_pt/N_SAMPLES*1e6:.1f} µs/réacteur)')

print(f'CUDA per-block  ({N_SAMPLES} réacteurs) ...', end=' ', flush=True)
t0 = time.perf_counter()
J_pb = jac_pb.evaluate(T_np, C_np)   # (N, n_sp, n_state)
t_pb = time.perf_counter() - t0
print(f'{t_pb*1e3:.1f} ms  ({t_pb/N_SAMPLES*1e6:.1f} µs/réacteur)')


# ═══════════════════════════════════════════════════════════════════════════
#  Métriques console
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('CUDA PT vs Cantera')
d_pt_tot, r_pt_tot = stats('Total  (n_sp × n_state)',   J_pt,            J_can)
d_pt_dT,  r_pt_dT  = stats('Colonne T  [d(om)/dT]',    J_pt[:, :, 0],   J_can[:, :, 0])
d_pt_dC,  r_pt_dC  = stats('Colonnes C [d(om)/dC]',    J_pt[:, :, 1:],  J_can[:, :, 1:])

print('\nCUDA PB vs Cantera')
d_pb_tot, r_pb_tot = stats('Total  (n_sp × n_state)',   J_pb,            J_can)
d_pb_dT,  r_pb_dT  = stats('Colonne T  [d(om)/dT]',    J_pb[:, :, 0],   J_can[:, :, 0])
d_pb_dC,  r_pb_dC  = stats('Colonnes C [d(om)/dC]',    J_pb[:, :, 1:],  J_can[:, :, 1:])

print('\nCUDA PB vs CUDA PT')
d_pp, r_pp = stats('Total', J_pb, J_pt)


# ═══════════════════════════════════════════════════════════════════════════
#  Heatmaps  — erreurs moyennées sur les N réacteurs
#  Axes : lignes = espèces s,  colonnes = variable xk_j  (0=T, 1..n_sp=C_k)
# ═══════════════════════════════════════════════════════════════════════════

print('\nGénération des figures ...')

# Erreurs absolues et relatives moyennées sur l'axe batch
mae_pt  = np.abs(J_pt - J_can).mean(axis=0)          # (n_sp, n_state)
rele_pt = (np.abs(J_pt - J_can)
            / (np.abs(J_can) + 1e-100)).mean(axis=0)

mae_pb  = np.abs(J_pb - J_can).mean(axis=0)
rele_pb = (np.abs(J_pb - J_can)
            / (np.abs(J_can) + 1e-100)).mean(axis=0)

# ── Figure 1 : heatmaps MAE ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
heatmap(axes[0], mae_pt,
        'CUDA PT vs Cantera — MAE',
        'MAE (mol m⁻³ s⁻¹  or  s⁻¹)')
heatmap(axes[1], mae_pb,
        'CUDA PB vs Cantera — MAE',
        'MAE (mol m⁻³ s⁻¹  or  s⁻¹)')
fig.suptitle(f'Jacobienne chimique — erreur absolue moyenne (N={N_SAMPLES})', fontsize=10)
plt.tight_layout()
plt.savefig('jac_chem_figures/heatmap_mae.png', bbox_inches='tight')
plt.close()

# ── Figure 2 : heatmaps erreur relative (log10) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
heatmap(axes[0], np.log10(rele_pt + 1e-20),
        'CUDA PT vs Cantera — log₁₀(rMAE)',
        'log₁₀ relative error', cmap='inferno')
heatmap(axes[1], np.log10(rele_pb + 1e-20),
        'CUDA PB vs Cantera — log₁₀(rMAE)',
        'log₁₀ relative error', cmap='inferno')
fig.suptitle(f'Jacobienne chimique — erreur relative (log) moyenne (N={N_SAMPLES})', fontsize=10)
plt.tight_layout()
plt.savefig('jac_chem_figures/heatmap_rele_log.png', bbox_inches='tight')
plt.close()

# ── Figure 3 : heatmaps colonne T et colonnes C séparément (MAE) ──────────
# Colonne T : (n_sp,) ; colonnes C : (n_sp, n_sp)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# MAE colonne T  (vecteur → on l'affiche comme une seule colonne)
for ax, J_cuda, label in [
    (axes[0, 0], J_pt, 'PT  d(om)/dT — MAE'),
    (axes[0, 1], J_pb, 'PB  d(om)/dT — MAE'),
]:
    col_T_mae = np.abs(J_cuda[:, :, 0] - J_can[:, :, 0]).mean(axis=0)   # (n_sp,)
    im = ax.imshow(col_T_mae[:, None], aspect='auto', origin='upper', cmap='plasma')
    plt.colorbar(im, ax=ax, label='MAE (mol m⁻³ s⁻¹ K⁻¹)')
    ax.set_xlabel('j=0  (T)')
    ax.set_ylabel('Species s')
    ax.set_title(label, fontsize=8)

# MAE colonnes C  (n_sp × n_sp)
for ax, J_cuda, label in [
    (axes[1, 0], J_pt, 'PT  d(om)/dC — MAE'),
    (axes[1, 1], J_pb, 'PB  d(om)/dC — MAE'),
]:
    col_C_mae = np.abs(J_cuda[:, :, 1:] - J_can[:, :, 1:]).mean(axis=0)   # (n_sp, n_sp)
    im = ax.imshow(col_C_mae, aspect='auto', origin='upper', cmap='plasma')
    plt.colorbar(im, ax=ax, label='MAE (s⁻¹)')
    ax.set_xlabel('Species column k  (d/dC_k)')
    ax.set_ylabel('Species row s  (d omega_s)')
    ax.set_title(label, fontsize=8)

fig.suptitle('Jacobienne chimique — MAE séparée : colonne T vs bloc C', fontsize=10)
plt.tight_layout()
plt.savefig('jac_chem_figures/heatmap_split_T_C.png', bbox_inches='tight')
plt.close()

# ── Figure 4 : scatter plots ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

scatter_11(axes[0, 0],
           J_can[:, :, 0].ravel(), J_pt[:, :, 0].ravel(),
           'Cantera d(om)/dT', 'CUDA PT d(om)/dT', COLORS[1])
axes[0, 0].set_title('PT — d(om)/dT')

scatter_11(axes[0, 1],
           J_can[:, :, 0].ravel(), J_pb[:, :, 0].ravel(),
           'Cantera d(om)/dT', 'CUDA PB d(om)/dT', COLORS[2])
axes[0, 1].set_title('PB — d(om)/dT')

scatter_11(axes[1, 0],
           J_can[:, :, 1:].ravel(), J_pt[:, :, 1:].ravel(),
           'Cantera d(om)/dC', 'CUDA PT d(om)/dC', COLORS[1])
axes[1, 0].set_title('PT — d(om)/dC')

scatter_11(axes[1, 1],
           J_can[:, :, 1:].ravel(), J_pb[:, :, 1:].ravel(),
           'Cantera d(om)/dC', 'CUDA PB d(om)/dC', COLORS[2])
axes[1, 1].set_title('PB — d(om)/dC')

fig.suptitle('Jacobienne chimique — scatter CUDA vs Cantera', fontsize=10)
plt.tight_layout()
plt.savefig('jac_chem_figures/scatter.png', bbox_inches='tight')
plt.close()

# ── Figure 5 : MAE par espèce (moyennée sur toutes les colonnes) ──────────
sp_mae_pt = np.abs(J_pt - J_can).mean(axis=(0, 2))   # (n_sp,)
sp_mae_pb = np.abs(J_pb - J_can).mean(axis=(0, 2))
x_sp = np.arange(n_sp)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
for ax, sp_mae, label, col in [
    (axes[0], sp_mae_pt, 'CUDA PT vs Cantera', COLORS[1]),
    (axes[1], sp_mae_pb, 'CUDA PB vs Cantera', COLORS[2]),
]:
    ax.bar(x_sp, sp_mae, color=col, alpha=0.75, width=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('Species index s')
    ax.set_ylabel('MAE (avg over j columns)')
    ax.set_title(label, fontsize=8)
    for k in np.argsort(sp_mae)[-5:][::-1]:
        ax.text(k, sp_mae[k] * 1.6, sp_names[k],
                ha='center', fontsize=6, rotation=90)
fig.suptitle('Jacobienne chimique — MAE par espèce', fontsize=10)
plt.tight_layout()
plt.savefig('jac_chem_figures/mae_per_species.png', bbox_inches='tight')
plt.close()

# ── Figure 6 : MAE par colonne xk_j ──────────────────────────────────────
col_mae_pt = np.abs(J_pt - J_can).mean(axis=(0, 1))   # (n_state,)
col_mae_pb = np.abs(J_pb - J_can).mean(axis=(0, 1))
x_col = np.arange(n_state)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
for ax, col_mae, label, col in [
    (axes[0], col_mae_pt, 'CUDA PT vs Cantera', COLORS[1]),
    (axes[1], col_mae_pb, 'CUDA PB vs Cantera', COLORS[2]),
]:
    ax.bar(x_col, col_mae, color=col, alpha=0.75, width=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('Column index j  (0=T,  j=C_{j-1})')
    ax.set_ylabel('MAE (avg over species s)')
    ax.set_title(label, fontsize=8)
    # Annotation colonne T
    ax.text(0, col_mae[0] * 1.6, 'T', ha='center', fontsize=7, color='red')
    for k in np.argsort(col_mae[1:])[-4:][::-1]:
        ax.text(k + 1, col_mae[k + 1] * 1.6, sp_names[k],
                ha='center', fontsize=6, rotation=90)
fig.suptitle('Jacobienne chimique — MAE par colonne xk_j', fontsize=10)
plt.tight_layout()
plt.savefig('jac_chem_figures/mae_per_column.png', bbox_inches='tight')
plt.close()

# ── Figure 7 : boxplots récapitulatifs ────────────────────────────────────
labels_b  = ['PT\ntotal', 'PT\nd(om)/dT', 'PT\nd(om)/dC',
             'PB\ntotal', 'PB\nd(om)/dT', 'PB\nd(om)/dC']
data_abs  = [d_pt_tot, d_pt_dT, d_pt_dC, d_pb_tot, d_pb_dT, d_pb_dC]
data_rel  = [r_pt_tot, r_pt_dT, r_pt_dC, r_pb_tot, r_pb_dT, r_pb_dC]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, ylabel in [
    (axes[0], data_abs, 'MAE'),
    (axes[1], data_rel, 'Relative error'),
]:
    ax.boxplot(
        [d.ravel() for d in data],
        labels=labels_b,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.2),
        boxprops=dict(facecolor=COLORS[0], alpha=0.65),
    )
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=15)
fig.suptitle('Jacobienne chimique CUDA vs Cantera — récapitulatif', fontsize=10)
plt.tight_layout()
plt.savefig('jac_chem_figures/boxplots_summary.png', bbox_inches='tight')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Résumé console
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('RÉSUMÉ')
print(f'  N={N_SAMPLES}  n_sp={n_sp}  n_state={n_state}')
print(f'\n  CUDA PT vs Cantera')
print(f'    total   MAE={d_pt_tot.mean():.3e}  rMAE={r_pt_tot.mean():.3e}')
print(f'    d(om)/dT MAE={d_pt_dT.mean():.3e}  rMAE={r_pt_dT.mean():.3e}')
print(f'    d(om)/dC MAE={d_pt_dC.mean():.3e}  rMAE={r_pt_dC.mean():.3e}')
print(f'\n  CUDA PB vs Cantera')
print(f'    total    MAE={d_pb_tot.mean():.3e}  rMAE={r_pb_tot.mean():.3e}')
print(f'    d(om)/dT MAE={d_pb_dT.mean():.3e}  rMAE={r_pb_dT.mean():.3e}')
print(f'    d(om)/dC MAE={d_pb_dC.mean():.3e}  rMAE={r_pb_dC.mean():.3e}')
print(f'\n  CUDA PB vs CUDA PT')
print(f'    total    MAE={d_pp.mean():.3e}   rMAE={r_pp.mean():.3e}')
print(f'\n  Timing  PT={t_pt*1e3:.1f}ms  PB={t_pb*1e3:.1f}ms  (Cantera={N_SAMPLES} appels séquentiels)')
print(f'\n  Figures dans : jac_chem_figures/')
print('=' * 70)