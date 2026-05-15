"""
compare_jacobian_trajectory.py — Comparaison Jacobienne Cantera vs CUDA sur trajectoire réacteur.

Pipeline :
  1. Intégration temporelle Cantera (réacteur isobare)
  2. Calcul de la Jacobienne chimique aux points de la trajectoire
     - Source A : Cantera analytique (net_production_rates_ddCi / ddT)
     - Source B : CUDA PT  (CUDAJacobianChemEvaluatorPT)
     - Source C : CUDA PB  (CUDAJacobianChemEvaluatorPB)
  3. Comparaison directe A vs B et A vs C sur les quantités dérivées :
       - J·Y                      (produit matrice-vecteur)
       - Rapport de raideur        (max|Re(λ)| / min|Re(λ)|) via valeurs propres
       - Valeurs propres (Re et Im)
       - J·ΔY                     (réponse à une perturbation)
       - Diagrammes de phase       (non affectés par J, montrés pour référence)
  4. Heatmaps d'erreur MAE et rMAE sur la Jacobienne le long de la trajectoire

Note : la Jacobienne chimique CUDA est d(omega)/d(xk) avec omega en mol/m³/s
       et xk = (T, C_0, ..., C_{n-1}).
       La Jacobienne de masse utilisée dans ce script est :
           J_mass[i, j] = (W_i / W_j) * d(omega_i)/d(C_j)
       Pour la colonne T on utilise directement d(omega_i)/dT.
"""

import sys, os
import numpy as np
import time
import warnings

import torch
import cantera as ct
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numba.cuda as cuda
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import kineticsGPU.torch.thermo as yaml_thermo
from kineticsGPU.torch.reactionParser import parseReactions, build_reactions
from kineticsGPU.cuda.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.cuda.kernels.per_thread.jacobian   import CUDAJacobianChemEvaluatorPT
from kineticsGPU.cuda.kernels.per_block.jacobian_pb import CUDAJacobianChemEvaluatorPB


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

yaml_file = '../gri30.yaml'
dtype_np  = np.float64
dtype_th  = torch.float64
device    = torch.device('cuda:0')
cuda.select_device(device.index)

T0    = 1000.0
P0    = ct.one_atm
phi   = 1.0
t_end = 10000e-3

os.makedirs('traj_figures', exist_ok=True)

matplotlib.rcParams.update({
    'text.usetex': False,
    'figure.dpi': 150,
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False,
})


# ═══════════════════════════════════════════════════════════════════════════
#  Chargement mécanisme CUDA
# ═══════════════════════════════════════════════════════════════════════════

print('=' * 70)
print('Chargement du mécanisme')

gas_ref = ct.Solution(yaml_file)
y_obj   = yaml_thermo.yamlParser(yaml_file)
vec_th  = y_obj.populateVectorizedThermoObject(dtype=dtype_th, device=str(device))
rxns    = parseReactions(yaml_file)

r_e, r_tb, r_f, _ = build_reactions(vec_th.species, rxns, vec_th, str(device), dtype_th)
mech               = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype_np)

n_sp     = mech.n_species
n_state  = n_sp + 1
sp_names = gas_ref.species_names

print(f'  {n_sp} espèces  |  {mech.elementary.n_reactions} élém.  '
      f'|  {mech.threebody.n_reactions} 3-corps  '
      f'|  {mech.falloff.n_reactions} falloff')


# ═══════════════════════════════════════════════════════════════════════════
#  Intégration temporelle Cantera
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('Intégration temporelle Cantera ...')

gas = ct.Solution(yaml_file)
gas.set_equivalence_ratio(phi, 'CH4', 'O2:1, N2:3.76')
gas.TP = T0, P0

reactor = ct.IdealGasReactor(gas)
sim     = ct.ReactorNet([reactor])

times_list = []
Y_list     = []
T_list     = []
C_list     = []   # concentrations mol/m³ (pour CUDA)
W_list     = []   # masses molaires

t = 0.0
while t < t_end:
    t = sim.step()
    g = reactor.thermo
    times_list.append(t)
    Y_list.append(g.Y.copy())
    T_list.append(g.T)
    # concentrations mol/m³ = kmol/m³ * 1e3
    C_list.append(g.concentrations * 1e3)
    W_list.append(g.molecular_weights.copy())

times = np.array(times_list)
Y_all = np.array(Y_list)       # (N, n_sp)
T_all = np.array(T_list)       # (N,)
C_all = np.array(C_list)       # (N, n_sp)  mol/m³
W_all = np.array(W_list)       # (N, n_sp)
N     = len(times)

print(f'  {N} pas de temps,  t_end = {times[-1]*1e3:.4f} ms')


# ═══════════════════════════════════════════════════════════════════════════
#  Jacobienne Cantera (référence)
#  J_can_chem[n, s, j] = d(omega_s)/d(xk_j)
#    j=0   → d(omega_s)/dT          (mol/m³/s/K)
#    j=k+1 → d(omega_s)/d(C_k)      (s⁻¹)
# ═══════════════════════════════════════════════════════════════════════════

print('\nJacobiennes Cantera ...', end=' ', flush=True)
t0 = time.perf_counter()

J_can_chem = np.zeros((N, n_sp, n_state), dtype=dtype_np)
omega_can  = np.zeros((N, n_sp), dtype=dtype_np)
fx_can = np.zeros((N, n_sp + 1), dtype=dtype_np)

for n in range(N):
    g = ct.Solution(yaml_file)
    g.TP = float(T_all[n]), P0
    g.concentrations = C_all[n] * 1e-3            # mol/m³ → kmol/m³
    J_can_chem[n, :, 1:] = g.net_production_rates_ddCi
    J_can_chem[n, :, 0]  = g.net_production_rates_ddT * 1e3
    omega_can[n]         = g.net_production_rates * 1e3    # mol/m³/s
    fx_can[n, 1:]        = omega_can[n]
    fx_can[n, 0]         = T_all[n]


print(f'{time.perf_counter()-t0:.1f}s')


# ═══════════════════════════════════════════════════════════════════════════
#  Jacobiennes CUDA
# ═══════════════════════════════════════════════════════════════════════════

print('Jacobiennes CUDA ...', end=' ', flush=True)
jac_pt = CUDAJacobianChemEvaluatorPT(mech, dtype=dtype_np)
jac_pb = CUDAJacobianChemEvaluatorPB(mech, dtype=dtype_np)
jac_pt.evaluate(T_all[:1], C_all[:1])   # warmup
jac_pb.evaluate(T_all[:1], C_all[:1])

t0 = time.perf_counter()
J_pt_chem = jac_pt.evaluate(T_all, C_all)   # (N, n_sp, n_state)
t_pt = time.perf_counter() - t0

t0 = time.perf_counter()
J_pb_chem = jac_pb.evaluate(T_all, C_all)   # (N, n_sp, n_state)
t_pb = time.perf_counter() - t0

print(f'PT={t_pt*1e3:.1f}ms  PB={t_pb*1e3:.1f}ms')


# ═══════════════════════════════════════════════════════════════════════════
#  Conversion en Jacobienne de masse pour les analyses dynamiques
#
#  J_mass[i, j] = (W_i / W_j) * d(omega_i)/d(C_j)   [s⁻¹]
#  On conserve aussi la colonne T séparément.
#
#  dY_k/dt = W_k * omega_k / rho
#  omega en mol/m³/s → dY/dt en s⁻¹ via le facteur W/rho
# ═══════════════════════════════════════════════════════════════════════════

def build_mass_jacobian(J_chem, W):
    """
    Construit J_mass (n_sp × n_sp) à partir de J_chem (n_sp × n_state).
    J_mass[i, j] = (W_i / W_j) * J_chem[i, j+1]   (colonnes C uniquement)
    """
    W_ratio = W[:, None] / W[None, :]          # (n_sp, n_sp)
    return W_ratio * J_chem[:, 1:]             # (n_sp, n_sp)


# Tableaux de Jacobiennes de masse (N, n_sp, n_sp)
J_can_mass = np.array([build_mass_jacobian(J_can_chem[n], W_all[n]) for n in range(N)])
J_pt_mass  = np.array([build_mass_jacobian(J_pt_chem[n],  W_all[n]) for n in range(N)])
J_pb_mass  = np.array([build_mass_jacobian(J_pb_chem[n],  W_all[n]) for n in range(N)])


# ═══════════════════════════════════════════════════════════════════════════
#  Quantités dérivées communes
# ═══════════════════════════════════════════════════════════════════════════

def derived_quantities(J_mass, Y):
    """Calcule J·Y, valeurs propres et rapport de raideur."""
    N_loc = J_mass.shape[0]

    JY          = np.einsum('nij,nj->ni', J_mass, Y)    # (N, n_sp)
    eigenvalues = np.array([np.linalg.eigvals(J_mass[n]) for n in range(N_loc)])
    eig_real    = np.real(eigenvalues)
    eig_imag    = np.imag(eigenvalues)

    stiffness = np.full(N_loc, np.nan)
    for n in range(N_loc):
        neg = np.abs(eig_real[n, eig_real[n] < 0])
        if len(neg) >= 2:
            stiffness[n] = np.max(neg) / (np.min(neg) + 1e-100)

    return JY, eig_real, eig_imag, stiffness


print('\nCalcul des quantités dérivées ...')
JY_can,  er_can,  ei_can,  stiff_can  = derived_quantities(J_can_mass, Y_all)
JY_pt,   er_pt,   ei_pt,   stiff_pt   = derived_quantities(J_pt_mass,  Y_all)
JY_pb,   er_pb,   ei_pb,   stiff_pb   = derived_quantities(J_pb_mass,  Y_all)

# J·ΔY  (ΔY = différence entre pas successifs)
DeltaY       = np.zeros_like(Y_all)
DeltaY[1:]   = Y_all[:-1] - Y_all[1:]
JDY_can = np.einsum('nij,nj->ni', J_can_mass, DeltaY)
JDY_pt  = np.einsum('nij,nj->ni', J_pt_mass,  DeltaY)
JDY_pb  = np.einsum('nij,nj->ni', J_pb_mass,  DeltaY)

# Sélection des espèces à tracer (max > seuil)
threshold    = 1e-10
mask         = np.max(np.abs(Y_all), axis=0) > threshold
sel_idx      = np.where(mask)[0]
sel_names    = [sp_names[i] for i in sel_idx]
n_sel        = len(sel_idx)
cmap_sp      = cm.get_cmap('tab20', n_sel)
sp_colors    = [cmap_sp(i) for i in range(n_sel)]

t_ms = times * 1e3


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers de tracé
# ═══════════════════════════════════════════════════════════════════════════

def _species_lines(ax, data_A, data_B, data_C, ylabel, title, logy=False):
    """
    Trace data_A (Cantera, trait plein), data_B (CUDA PT, tirets),
    data_C (CUDA PB, pointillés) pour toutes les espèces sélectionnées.
    """
    for j, idx in enumerate(sel_idx):
        col = sp_colors[j]
        lbl = sel_names[j]
        y_A = np.abs(data_A[:, idx]) + 1e-30
        y_B = np.abs(data_B[:, idx]) + 1e-30
        y_C = np.abs(data_C[:, idx]) + 1e-30
        if logy:
            ax.semilogy(t_ms, y_A, color=col, lw=1.4, ls='-',  label=f'{lbl} Can')
            ax.semilogy(t_ms, y_B, color=col, lw=1.0, ls='--', alpha=0.7)
            ax.semilogy(t_ms, y_C, color=col, lw=0.8, ls=':',  alpha=0.5)
        else:
            ax.plot(t_ms, data_A[:, idx], color=col, lw=1.4, ls='-',  label=f'{lbl} Can')
            ax.plot(t_ms, data_B[:, idx], color=col, lw=1.0, ls='--', alpha=0.7)
            ax.plot(t_ms, data_C[:, idx], color=col, lw=0.8, ls=':',  alpha=0.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(ncol=5, fontsize=5, loc='best')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.grid(True, ls='--', alpha=0.3)


def _diff_lines(ax, data_A, data_B, data_C, ylabel, title):
    """Trace |CUDA - Cantera| pour PT et PB, par espèce."""
    for j, idx in enumerate(sel_idx):
        col = sp_colors[j]
        lbl = sel_names[j]
        diff_pt = np.abs(data_B[:, idx] - data_A[:, idx]) + 1e-30
        diff_pb = np.abs(data_C[:, idx] - data_A[:, idx]) + 1e-30
        ax.semilogy(t_ms, diff_pt, color=col, lw=1.0, ls='--', label=f'{lbl} PT')
        ax.semilogy(t_ms, diff_pb, color=col, lw=0.7, ls=':')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(ncol=5, fontsize=5, loc='best')
    ax.grid(True, ls='--', alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1 — J·Y  (comparaison + différence)
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 1 — J·Y ...')
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
fig.suptitle(rf'$J \cdot Y$ — Cantera vs CUDA  ($T_0={T0:.0f}$ K, $\phi={phi}$)', fontsize=11)

_species_lines(axes[0], JY_can, JY_pt, JY_pb,
               r'$[J \cdot Y]_k$  (s$^{-1}$)',
               'Cantera (—)  CUDA PT (--)  CUDA PB (:)', logy=False)

_species_lines(axes[1], JY_can, JY_pt, JY_pb,
               r'$|[J \cdot Y]_k|$  (s$^{-1}$)',
               'Magnitude (log)', logy=True)

_diff_lines(axes[2], JY_can, JY_pt, JY_pb,
            r'$|[J \cdot Y]_k^{\rm CUDA} - [J \cdot Y]_k^{\rm Can}|$',
            'Erreur absolue CUDA PT (--)  CUDA PB (:)')

axes[2].set_xlabel('Time (ms)', fontsize=9)
fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2 — Rapport de raideur
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 2 — Stiffness ...')
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
fig.suptitle('Rapport de raideur — Cantera vs CUDA', fontsize=11)

ax = axes[0]
ax.semilogy(t_ms, stiff_can, 'k-',  lw=2,   label='Cantera')
ax.semilogy(t_ms, stiff_pt,  'b--', lw=1.5, label='CUDA PT', alpha=0.8)
ax.semilogy(t_ms, stiff_pb,  'r:',  lw=1.5, label='CUDA PB', alpha=0.8)
ax.set_ylabel(r'Stiffness ratio  $\max|\mathrm{Re}(\lambda^-)| / \min|\mathrm{Re}(\lambda^-)|$', fontsize=9)
ax.set_title('Rapport de raideur', fontsize=9)
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.3)

ax = axes[1]
ax.semilogy(t_ms, np.abs(stiff_pt - stiff_can) + 1e-30,
            'b--', lw=1.5, label='|PT - Can|')
ax.semilogy(t_ms, np.abs(stiff_pb - stiff_can) + 1e-30,
            'r:',  lw=1.5, label='|PB - Can|')
ax.set_ylabel('Erreur absolue (–)', fontsize=9)
ax.set_title('Erreur absolue sur le rapport de raideur', fontsize=9)
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.3)

# Erreur relative
ax = axes[2]
rele_pt_stiff = np.abs(stiff_pt - stiff_can) / (np.abs(stiff_can) + 1e-100)
rele_pb_stiff = np.abs(stiff_pb - stiff_can) / (np.abs(stiff_can) + 1e-100)
ax.semilogy(t_ms, rele_pt_stiff, 'b--', lw=1.5, label='PT rel. error')
ax.semilogy(t_ms, rele_pb_stiff, 'r:',  lw=1.5, label='PB rel. error')
ax.set_ylabel('Erreur relative (–)', fontsize=9)
ax.set_title('Erreur relative sur le rapport de raideur', fontsize=9)
ax.set_xlabel('Time (ms)', fontsize=9)
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.3)

fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3 — Valeurs propres (Re et Im)
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 3 — Eigenvalues ...')
cmap_eig = cm.get_cmap('coolwarm', n_sp)
eig_col  = [cmap_eig(i / n_sp) for i in range(n_sp)]

fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
fig.suptitle('Valeurs propres de J — Cantera vs CUDA', fontsize=11)

for panel_idx, (er_src, ei_src, src_label, ls) in enumerate([
    (er_can, ei_can, 'Cantera', '-'),
    (er_pt,  ei_pt,  'CUDA PT', '--'),
]):
    # Re(λ)
    ax = axes[panel_idx * 2]
    for i in range(n_sp):
        ax.plot(t_ms, er_src[:, i], color=eig_col[i], lw=0.7, alpha=0.5, ls=ls)
    ax.plot(t_ms, np.min(er_src, axis=1), 'navy',       lw=2, label=r'$\min\,\mathrm{Re}(\lambda)$')
    ax.plot(t_ms, np.max(er_src, axis=1), 'darkorange',  lw=2, label=r'$\max\,\mathrm{Re}(\lambda)$')
    ax.axhline(0, color='k', lw=0.7, ls=':')
    ax.set_ylabel(r'$\mathrm{Re}(\lambda_k)$  (s$^{-1}$)', fontsize=9)
    ax.set_title(f'Partie réelle — {src_label}', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, ls='--', alpha=0.3)

    # Im(λ)
    ax = axes[panel_idx * 2 + 1]
    for i in range(n_sp):
        ax.plot(t_ms, ei_src[:, i], color=eig_col[i], lw=0.7, alpha=0.5, ls=ls)
    ax.axhline(0, color='k', lw=0.7, ls=':')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda_k)$  (s$^{-1}$)', fontsize=9)
    ax.set_title(f'Partie imaginaire — {src_label}', fontsize=9)
    ax.grid(True, ls='--', alpha=0.3)

axes[-1].set_xlabel('Time (ms)', fontsize=9)
fig.tight_layout()
plt.show()

# ── Figure 3b — Erreur sur les valeurs propres triées ─────────────────────
print('Figure 3b — Eigenvalue errors ...')
# On trie les valeurs propres par Re croissant à chaque pas de temps pour comparer
er_can_s = np.sort(er_can, axis=1)
er_pt_s  = np.sort(er_pt,  axis=1)
er_pb_s  = np.sort(er_pb,  axis=1)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle('Erreur sur Re(λ) (valeurs propres triées) — CUDA vs Cantera', fontsize=11)

ax = axes[0]
for i in range(n_sp):
    ax.semilogy(t_ms, np.abs(er_pt_s[:, i] - er_can_s[:, i]) + 1e-30,
                color=eig_col[i], lw=0.7, ls='--', alpha=0.6)
ax.set_ylabel(r'$|\mathrm{Re}(\lambda)_{\rm PT} - \mathrm{Re}(\lambda)_{\rm Can}|$', fontsize=9)
ax.set_title('CUDA PT — erreur absolue sur chaque valeur propre triée', fontsize=9)
ax.grid(True, which='both', ls='--', alpha=0.3)

ax = axes[1]
for i in range(n_sp):
    ax.semilogy(t_ms, np.abs(er_pb_s[:, i] - er_can_s[:, i]) + 1e-30,
                color=eig_col[i], lw=0.7, ls=':', alpha=0.6)
ax.set_ylabel(r'$|\mathrm{Re}(\lambda)_{\rm PB} - \mathrm{Re}(\lambda)_{\rm Can}|$', fontsize=9)
ax.set_title('CUDA PB — erreur absolue sur chaque valeur propre triée', fontsize=9)
ax.set_xlabel('Time (ms)', fontsize=9)
ax.grid(True, which='both', ls='--', alpha=0.3)

fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 4 — J·ΔY
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 4 — J·ΔY ...')
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
fig.suptitle(r'$J \cdot \Delta Y$ — Cantera vs CUDA', fontsize=11)

_species_lines(axes[0], JDY_can, JDY_pt, JDY_pb,
               r'$[J \cdot \Delta Y]_k$',
               'Cantera (—)  CUDA PT (--)  CUDA PB (:)', logy=False)

_species_lines(axes[1], JDY_can, JDY_pt, JDY_pb,
               r'$|[J \cdot \Delta Y]_k|$',
               'Magnitude (log)', logy=True)

_diff_lines(axes[2], JDY_can, JDY_pt, JDY_pb,
            r'$|[J \cdot \Delta Y]^{\rm CUDA} - [J \cdot \Delta Y]^{\rm Can}|$',
            'Erreur absolue CUDA PT (--)  CUDA PB (:)')

axes[2].set_xlabel('Time (ms)', fontsize=9)
fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 5 — Heatmaps d'erreur sur la Jacobienne chimique le long de la trajectoire
#  Axes : x = pas de temps n,  y = indice espèce s  (pour une colonne j fixée)
#  On sépare : colonne T  et  bloc C (moyenné sur les colonnes C)
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 5 — Heatmaps erreur jacobienne ...')

# Erreur absolue moyennée sur les colonnes C (bloc C uniquement)
mae_pt_C  = np.abs(J_pt_chem[:, :, 1:] - J_can_chem[:, :, 1:])   # (N, n_sp, n_sp)
mae_pb_C  = np.abs(J_pb_chem[:, :, 1:] - J_can_chem[:, :, 1:])
rele_pt_C = mae_pt_C / (np.abs(J_can_chem[:, :, 1:]) + 1e-100)
rele_pb_C = mae_pb_C / (np.abs(J_can_chem[:, :, 1:]) + 1e-100)

# Moyenne sur les colonnes C → (N, n_sp)
mae_pt_C_avg  = mae_pt_C.mean(axis=2)
mae_pb_C_avg  = mae_pb_C.mean(axis=2)
rele_pt_C_avg = rele_pt_C.mean(axis=2)
rele_pb_C_avg = rele_pb_C.mean(axis=2)

# Colonne T
mae_pt_T  = np.abs(J_pt_chem[:, :, 0] - J_can_chem[:, :, 0])     # (N, n_sp)
mae_pb_T  = np.abs(J_pb_chem[:, :, 0] - J_can_chem[:, :, 0])
rele_pt_T = mae_pt_T / (np.abs(J_can_chem[:, :, 0]) + 1e-100)
rele_pb_T = mae_pb_T / (np.abs(J_can_chem[:, :, 0]) + 1e-100)

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle('Erreur Jacobienne chimique le long de la trajectoire\n'
             'Axes : x=temps, y=espèce', fontsize=11)

def _heatmap(ax, mat, title, cbar_label, cmap='plasma', log=False):
    data = np.log10(mat + 1e-20) if log else mat
    im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
                   extent=[t_ms[0], t_ms[-1], 0, n_sp])
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Species s')
    ax.set_title(title, fontsize=8)

_heatmap(axes[0, 0], mae_pt_T,         'PT  d(om)/dT — MAE',         'MAE (SI)')
_heatmap(axes[0, 1], mae_pb_T,         'PB  d(om)/dT — MAE',         'MAE (SI)')
_heatmap(axes[0, 2], rele_pt_T,        'PT  d(om)/dT — rMAE (log)',  'log10 rMAE', log=True)
_heatmap(axes[0, 3], rele_pb_T,        'PB  d(om)/dT — rMAE (log)',  'log10 rMAE', log=True)

_heatmap(axes[1, 0], mae_pt_C_avg,     'PT  d(om)/dC — MAE (avg C)',  'MAE (SI)')
_heatmap(axes[1, 1], mae_pb_C_avg,     'PB  d(om)/dC — MAE (avg C)',  'MAE (SI)')
_heatmap(axes[1, 2], rele_pt_C_avg,    'PT  d(om)/dC — rMAE (log)',   'log10 rMAE', log=True)
_heatmap(axes[1, 3], rele_pb_C_avg,    'PB  d(om)/dC — rMAE (log)',   'log10 rMAE', log=True)

fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 6 — Erreur globale sur J·Y et J·ΔY vs temps
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 6 — Erreurs globales ...')

norm_JY_can  = np.linalg.norm(JY_can,  axis=1) + 1e-30
norm_JDY_can = np.linalg.norm(JDY_can, axis=1) + 1e-30

err_JY_pt  = np.linalg.norm(JY_pt  - JY_can,  axis=1)
err_JY_pb  = np.linalg.norm(JY_pb  - JY_can,  axis=1)
err_JDY_pt = np.linalg.norm(JDY_pt - JDY_can, axis=1)
err_JDY_pb = np.linalg.norm(JDY_pb - JDY_can, axis=1)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('Erreurs globales (norme L2) — CUDA vs Cantera', fontsize=11)

ax = axes[0, 0]
ax.semilogy(t_ms, err_JY_pt, 'b--', lw=1.5, label='CUDA PT')
ax.semilogy(t_ms, err_JY_pb, 'r:',  lw=1.5, label='CUDA PB')
ax.set_ylabel(r'$\|J \cdot Y^{\rm CUDA} - J \cdot Y^{\rm Can}\|_2$', fontsize=9)
ax.set_title(r'Erreur absolue  $J \cdot Y$', fontsize=9)
ax.legend(); ax.grid(True, ls='--', alpha=0.3)

ax = axes[0, 1]
ax.semilogy(t_ms, err_JY_pt / norm_JY_can, 'b--', lw=1.5, label='CUDA PT')
ax.semilogy(t_ms, err_JY_pb / norm_JY_can, 'r:',  lw=1.5, label='CUDA PB')
ax.set_ylabel(r'Erreur relative  $J \cdot Y$', fontsize=9)
ax.set_title(r'Erreur relative  $J \cdot Y$', fontsize=9)
ax.legend(); ax.grid(True, ls='--', alpha=0.3)

ax = axes[1, 0]
ax.semilogy(t_ms, err_JDY_pt, 'b--', lw=1.5, label='CUDA PT')
ax.semilogy(t_ms, err_JDY_pb, 'r:',  lw=1.5, label='CUDA PB')
ax.set_ylabel(r'$\|J \cdot \Delta Y^{\rm CUDA} - J \cdot \Delta Y^{\rm Can}\|_2$', fontsize=9)
ax.set_title(r'Erreur absolue  $J \cdot \Delta Y$', fontsize=9)
ax.set_xlabel('Time (ms)', fontsize=9)
ax.legend(); ax.grid(True, ls='--', alpha=0.3)

ax = axes[1, 1]
ax.semilogy(t_ms, err_JDY_pt / norm_JDY_can, 'b--', lw=1.5, label='CUDA PT')
ax.semilogy(t_ms, err_JDY_pb / norm_JDY_can, 'r:',  lw=1.5, label='CUDA PB')
ax.set_ylabel(r'Erreur relative  $J \cdot \Delta Y$', fontsize=9)
ax.set_title(r'Erreur relative  $J \cdot \Delta Y$', fontsize=9)
ax.set_xlabel('Time (ms)', fontsize=9)
ax.legend(); ax.grid(True, ls='--', alpha=0.3)

fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 7 — Trajectoire : Y(t) + T(t)  (référence, identique pour tous)
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 7 — Trajectoire Y(t) + T(t) ...')
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(rf'Trajectoire — GRI-3.0  ($T_0={T0:.0f}$ K, $\phi={phi}$)', fontsize=11)

ax = axes[0]
for j, idx in enumerate(sel_idx):
    ax.semilogy(t_ms, np.abs(Y_all[:, idx]) + 1e-20,
                color=sp_colors[j], lw=1.2, label=sel_names[j])
ax.set_ylabel(r'$Y_k$  (mass fraction)', fontsize=9)
ax.set_title('Fractions massiques', fontsize=9)
ax.legend(ncol=5, fontsize=5, loc='lower left')
ax.set_ylim(bottom=1e-10)
ax.grid(True, which='both', ls='--', alpha=0.3)

ax = axes[1]
ax.plot(t_ms, T_all, 'firebrick', lw=2)
ax.set_ylabel('T (K)', fontsize=9)
ax.set_title('Température', fontsize=9)
ax.set_xlabel('Time (ms)', fontsize=9)
ax.grid(True, ls='--', alpha=0.3)

fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 8 — Diagrammes de phase (identiques, sans jacobienne)
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 8 — Phase diagrams ...')

phase_pairs = [
    ('CH4', 'O2'), ('CO', 'CO2'), ('OH', 'H2O'),
    ('H',  'H2'),  ('HO2', 'H2O2'), ('CH4', 'CO2'),
]

def get_idx(name):
    try: return sp_names.index(name)
    except ValueError: return None

valid_pairs = [(a, b, get_idx(a), get_idx(b))
               for a, b in phase_pairs
               if get_idx(a) is not None and get_idx(b) is not None]

t_norm = (times - times[0]) / (times[-1] - times[0])
ncols  = 2
nrows  = (len(valid_pairs) + 1) // 2

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
fig.suptitle('Diagrammes de phase', fontsize=11)
axes = axes.flatten()

for k, (na, nb, ia, ib) in enumerate(valid_pairs):
    ax = axes[k]
    sc = ax.scatter(Y_all[:, ia], Y_all[:, ib], c=t_norm, cmap='viridis',
                    s=4, alpha=0.8, linewidths=0)
    ax.plot(Y_all[0, ia],  Y_all[0, ib],  'go', ms=7, label='t=0')
    ax.plot(Y_all[-1, ia], Y_all[-1, ib], 'r*', ms=9, label=f't={times[-1]*1e3:.3f} ms')
    ax.set_xlabel(rf'$Y_{{\rm {na}}}$', fontsize=9)
    ax.set_ylabel(rf'$Y_{{\rm {nb}}}$', fontsize=9)
    ax.set_title(rf'$Y_{{\rm {na}}}$ vs $Y_{{\rm {nb}}}$', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, ls='--', alpha=0.3)
    plt.colorbar(sc, ax=ax, label='t normalisé', pad=0.02)

for k in range(len(valid_pairs), len(axes)):
    axes[k].set_visible(False)

fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Préparation : J·ω  pour les 3 sources
#
#  omega_can est déjà en mol/m³/s (calculé plus haut).
#  La Jacobienne chimique CUDA est d(omega_i)/d(x_k) avec :
#    colonne 0 → d(omega)/dT   [mol/m³/s/K]
#    colonnes 1..n → d(omega)/dC_k  [s⁻¹]
#
#  On calcule  (J·ω)_i = sum_k  J_chem[i, k+1] * omega_k   (colonnes C seulement)
#  c'est-à-dire le produit J_C · ω  où J_C = J_chem[:, :, 1:]
# ═══════════════════════════════════════════════════════════════════════════

print('\nCalcul J·ω pour les 3 sources ...')

# J_chem shape : (N, n_sp, n_state)  avec n_state = n_sp + 1
# omega_can shape : (N, n_sp)

Jomega_can = np.einsum('nij,nj->ni', J_can_chem[:, :, :], fx_can)   # (N, n_sp+1)
Jomega_pt  = np.einsum('nij,nj->ni', J_pt_chem[:, :, :],  fx_can)   # (N, n_sp+1)
Jomega_pb  = np.einsum('nij,nj->ni', J_pb_chem[:, :, :],  fx_can)   # (N, n_sp+1)

Jomega_can = Jomega_can[:, 1:]
Jomega_pt = Jomega_pt[:, 1:]
Jomega_pb = Jomega_pb[:, 1:]


norm_omega     = np.linalg.norm(omega_can, axis=1) + 1e-30               # (N,)
norm_Jomega_can = np.linalg.norm(Jomega_can, axis=1)
norm_Jomega_pt  = np.linalg.norm(Jomega_pt,  axis=1)
norm_Jomega_pb  = np.linalg.norm(Jomega_pb,  axis=1)

ratio_Jomega_can = norm_Jomega_can / norm_omega   # ‖J·ω‖ / ‖ω‖  Cantera
ratio_Jomega_pt  = norm_Jomega_pt  / norm_omega   # ‖J·ω‖ / ‖ω‖  CUDA PT
ratio_Jomega_pb  = norm_Jomega_pb  / norm_omega   # ‖J·ω‖ / ‖ω‖  CUDA PB


# ═══════════════════════════════════════════════════════════════════════════
#  Estimée numérique par différences finies :  ‖Δω/Δt‖ / ‖ω‖
#
#  Δω[n] = omega_can[n+1] - omega_can[n]
#  Δt[n] = times[n+1] - times[n]
#  → approximation forward de d(omega)/dt
# ═══════════════════════════════════════════════════════════════════════════

domega_dt_fd = np.zeros_like(omega_can)                       # (N, n_sp)
domega_dt_fd[:-1] = (omega_can[1:] - omega_can[:-1]) / (
    (times[1:] - times[:-1])[:, None])
domega_dt_fd[-1]  = domega_dt_fd[-2]                          # répéter le dernier

norm_domega_fd       = np.linalg.norm(domega_dt_fd, axis=1)   # (N,)
ratio_domega_fd      = norm_domega_fd / norm_omega             # ‖Δω/Δt‖ / ‖ω‖   [s⁻¹]


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 9 — ‖J·ω‖ / ‖ω‖  :  Cantera vs CUDA PT vs CUDA PB
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 9 — ‖J·ω‖/‖ω‖ comparaison ...')

fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
fig.suptitle(
    r'$\|J \cdot \omega\| / \|\omega\|$ — Cantera vs CUDA'
    rf'  ($T_0={T0:.0f}$ K, $\phi={phi}$)',
    fontsize=11)

# ── Panel 1 : les 3 sources superposées ──
ax = axes[0]
ax.semilogy(t_ms, ratio_Jomega_can, 'k-',  lw=2.5, label='Cantera')
ax.semilogy(t_ms, ratio_Jomega_pt,  'b--', lw=1.8, label='CUDA PT', alpha=0.85)
ax.semilogy(t_ms, ratio_Jomega_pb,  'r:',  lw=1.8, label='CUDA PB', alpha=0.85)
ax.set_ylabel(r'$\|J_C \cdot \omega\|_2 \;/\; \|\omega\|_2$  (s$^{-1}$)', fontsize=9)
ax.set_title('Comparaison globale', fontsize=9)
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.3)

# ── Panel 2 : erreur absolue PT et PB vs Cantera ──
ax = axes[1]
ax.semilogy(t_ms, np.abs(ratio_Jomega_pt - ratio_Jomega_can) + 1e-30,
            'b--', lw=1.5, label='|PT - Can|')
ax.semilogy(t_ms, np.abs(ratio_Jomega_pb - ratio_Jomega_can) + 1e-30,
            'r:',  lw=1.5, label='|PB - Can|')
ax.set_ylabel('Erreur absolue (s$^{-1}$)', fontsize=9)
ax.set_title('Erreur absolue CUDA vs Cantera', fontsize=9)
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.3)

# ── Panel 3 : erreur relative ──
ax = axes[2]
rele_pt_Jomega = np.abs(ratio_Jomega_pt - ratio_Jomega_can) / (ratio_Jomega_can + 1e-100)
rele_pb_Jomega = np.abs(ratio_Jomega_pb - ratio_Jomega_can) / (ratio_Jomega_can + 1e-100)
ax.semilogy(t_ms, rele_pt_Jomega, 'b--', lw=1.5, label='PT — erreur relative')
ax.semilogy(t_ms, rele_pb_Jomega, 'r:',  lw=1.5, label='PB — erreur relative')
ax.set_ylabel('Erreur relative (—)', fontsize=9)
ax.set_title('Erreur relative CUDA vs Cantera', fontsize=9)
ax.set_xlabel('Time (ms)', fontsize=9)
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.3)

fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 10 — ‖J·ω‖/‖ω‖  vs  ‖Δω/Δt‖/‖ω‖  (différences finies)
# ═══════════════════════════════════════════════════════════════════════════

print('Figure 10 — ‖J·ω‖/‖ω‖ vs différences finies ...')

fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
fig.suptitle(
    r'$\|J \cdot \omega\| / \|\omega\|$  vs  $\|\Delta\omega/\Delta t\| / \|\omega\|$  (différences finies)'
    rf'  —  $T_0={T0:.0f}$ K',
    fontsize=11)

# ── Panel 1 : les 4 courbes ──
ax = axes[0]
ax.semilogy(t_ms, ratio_domega_fd,   'g-',  lw=2.5,
            label=r'$\|\Delta\omega/\Delta t\|/\|\omega\|$ (FD, Cantera)')
ax.semilogy(t_ms, ratio_Jomega_can,  'k--', lw=2,
            label=r'$\|J \cdot \omega\|/\|\omega\|$ Cantera')
ax.semilogy(t_ms, ratio_Jomega_pt,   'b:',  lw=1.8,
            label=r'$\|J \cdot \omega\|/\|\omega\|$ CUDA PT', alpha=0.85)
ax.semilogy(t_ms, ratio_Jomega_pb,   color='darkorange', ls='-.', lw=1.5,
            label=r'$\|J \cdot \omega\|/\|\omega\|$ CUDA PB', alpha=0.85)
ax.set_ylabel(r'Taux de variation  (s$^{-1}$)', fontsize=9)
ax.set_title(
    r'Comparaison : estimée analytique $J\omega$ vs estimée numérique FD',
    fontsize=9)
ax.legend(fontsize=7)
ax.grid(True, which='both', ls='--', alpha=0.3)

# ── Panel 2 : rapport analytique / numérique ──
ax = axes[1]
ratio_can_vs_fd = ratio_Jomega_can / (ratio_domega_fd + 1e-100)
ratio_pt_vs_fd  = ratio_Jomega_pt  / (ratio_domega_fd + 1e-100)
ratio_pb_vs_fd  = ratio_Jomega_pb  / (ratio_domega_fd + 1e-100)
ax.semilogy(t_ms, ratio_can_vs_fd, 'k--', lw=2,   label='Cantera / FD')
ax.semilogy(t_ms, ratio_pt_vs_fd,  'b:',  lw=1.5, label='CUDA PT / FD')
ax.semilogy(t_ms, ratio_pb_vs_fd,  color='darkorange', ls='-.', lw=1.5,
            label='CUDA PB / FD')
ax.axhline(1, color='gray', lw=1, ls=':', alpha=0.7)
ax.set_ylabel(r'Rapport $\|J\omega\| / \|\Delta\omega/\Delta t\|$ (—)', fontsize=9)
ax.set_title(
    r'Rapport analytique / numérique  '
    r'(= 1 si $J\omega \approx \dot{\omega}$)', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, which='both', ls='--', alpha=0.3)

# ── Panel 3 : écart absolu Cantera vs FD (qualité de l'approximation FD) ──
ax = axes[2]
ax.semilogy(t_ms, np.abs(ratio_Jomega_can - ratio_domega_fd) + 1e-30,
            'k-', lw=2, label=r'$|\|J\omega\|_{\rm Can} - \|\Delta\omega/\Delta t\||$')
ax.semilogy(t_ms, np.abs(ratio_Jomega_pt  - ratio_domega_fd) + 1e-30,
            'b--', lw=1.5, label='idem CUDA PT')
ax.semilogy(t_ms, np.abs(ratio_Jomega_pb  - ratio_domega_fd) + 1e-30,
            'r:',  lw=1.5, label='idem CUDA PB')
ax.set_ylabel(r'Écart absolu (s$^{-1}$)', fontsize=9)
ax.set_title(
    r'Écart entre estimée analytique et FD',
    fontsize=9)
ax.set_xlabel('Time (ms)', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, which='both', ls='--', alpha=0.3)

fig.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Résumé console
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('RÉSUMÉ')
print(f'  N={N} pas de temps  |  n_sp={n_sp}')
print(f'\n  Jacobienne chimique (J_chem) — moyenne sur la trajectoire')

def _summary(tag, J_cuda, J_ref):
    mae  = np.abs(J_cuda - J_ref).mean()
    rele = (np.abs(J_cuda - J_ref) / (np.abs(J_ref) + 1e-100)).mean()
    print(f'    {tag}  MAE={mae:.3e}   rMAE={rele:.3e}')

_summary('CUDA PT vs Can  [total  ]', J_pt_chem,          J_can_chem)
_summary('CUDA PT vs Can  [col T  ]', J_pt_chem[:,:,0],   J_can_chem[:,:,0])
_summary('CUDA PT vs Can  [cols C ]', J_pt_chem[:,:,1:],  J_can_chem[:,:,1:])
_summary('CUDA PB vs Can  [total  ]', J_pb_chem,          J_can_chem)
_summary('CUDA PB vs Can  [col T  ]', J_pb_chem[:,:,0],   J_can_chem[:,:,0])
_summary('CUDA PB vs Can  [cols C ]', J_pb_chem[:,:,1:],  J_can_chem[:,:,1:])

print(f'\n  J·Y  erreur relative moyenne')
print(f'    CUDA PT  {(err_JY_pt / norm_JY_can).mean():.3e}')
print(f'    CUDA PB  {(err_JY_pb / norm_JY_can).mean():.3e}')

print(f'\n  Timing CUDA  PT={t_pt*1e3:.1f}ms  PB={t_pb*1e3:.1f}ms  ({N} points)')

print('\n  ‖J·ω‖/‖ω‖  — moyenne temporelle')
print(f'    Cantera  {ratio_Jomega_can.mean():.3e} s⁻¹')
print(f'    CUDA PT  {ratio_Jomega_pt.mean():.3e} s⁻¹  '
      f'(err rel moy = {rele_pt_Jomega.mean():.3e})')
print(f'    CUDA PB  {ratio_Jomega_pb.mean():.3e} s⁻¹  '
      f'(err rel moy = {rele_pb_Jomega.mean():.3e})')
print(f'\n  ‖Δω/Δt‖/‖ω‖ FD  moyenne temporelle : {ratio_domega_fd.mean():.3e} s⁻¹')
print(f'  Rapport Cantera/FD moyen : {ratio_can_vs_fd.mean():.3e}')

print('=' * 70)