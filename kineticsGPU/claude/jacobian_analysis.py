"""
test_jacobian_analysis.py — Analyse complète et progressive de la Jacobienne.

Pipeline de validation en 4 étapes :

  Étape 1 : Validation des taux de réaction Torch vs Cantera
            (omega par type : élémentaires, trois-corps, falloff)

  Étape 2 : Calcul de la Jacobienne de référence via autodiff Torch
            (torch.autograd.functional.jvp appliqué à omega(T,C))

  Étape 3 : Comparaison Jacobienne Torch-autodiff vs Cantera analytique
            (blocs chimique et thermique séparément)

  Étape 4 : Comparaison Jacobienne CUDA (per-thread et per-block) vs Torch-autodiff
            (mêmes blocs, + heatmaps et scatter plots)

Toutes les figures sont sauvegardées dans le dossier jacobian_figures/.
"""

import numpy as np
import math
import time
import sys
import os
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import cantera as ct
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.autograd.functional import jvp as torch_jvp
import numba.cuda as cuda

from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Imports kineticsGPU
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
yaml_file  = '../gri30.yaml'
dtype_np   = np.float64
dtype_th   = torch.float64
device     = torch.device('cuda:0')
cuda.select_device(device.index)

N_SAMPLES  = 200   # nombre de points (T, C) aléatoires pour les étapes 1-3
N_JAC      = 50    # nombre de réacteurs pour les étapes 3-4 (Jacobienne coûteuse côté Cantera)
R_GAS      = 8.314_462_618_153_24

os.makedirs('jacobian_figures', exist_ok=True)

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
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def stats(label, a, b):
    """Affiche et retourne les métriques MAE/rMAE entre a et b."""
    diff = np.abs(a - b)
    rel  = diff / (np.abs(b) + 1e-100)
    print(f'    {label}')
    print(f'      MAE  = {diff.mean():.3e}  ± {diff.std():.3e}   max = {diff.max():.3e}')
    print(f'      rMAE = {rel.mean():.3e}  ± {rel.std():.3e}   max = {rel.max():.3e}')
    return diff, rel


def boxplot(ax, data_list, labels, ylabel, log=True):
    if log:
        ax.set_yscale('log')
    ax.boxplot(
        [d.ravel() for d in data_list],
        labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black'),
        boxprops=dict(facecolor=COLORS[0], alpha=0.6),
    )
    ax.set_ylabel(ylabel)


def scatter_11(ax, x, y, label_x, label_y, color, n_max=6000, title=''):
    rng = np.random.default_rng(42)
    n   = min(n_max, len(x.ravel()))
    idx = rng.choice(len(x.ravel()), n, replace=False)
    xf  = x.ravel()[idx]
    yf  = y.ravel()[idx]
    ax.scatter(xf, yf, s=1, alpha=0.3, color=color, rasterized=True)
    lo  = min(xf.min(), yf.min())
    hi  = max(xf.max(), yf.max())
    ax.axline((lo, lo), slope=1, color='red', linewidth=0.4, linestyle=':', label='x=y')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title, fontsize=8)


def heatmap_log(ax, mat, title=''):
    im = ax.imshow(np.log10(mat + 1e-20), aspect='auto', origin='upper', cmap='plasma')
    plt.colorbar(im, ax=ax, label=r'$\log_{10}$ rel. error')
    ax.set_xlabel('Column $j$  (0=T)')
    ax.set_ylabel('Row $i$  (0=dT/dt)')
    if title:
        ax.set_title(title, fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Chargement du mécanisme
# ─────────────────────────────────────────────────────────────────────────────
print('='*70)
print('Chargement du mécanisme')

gas = ct.Solution(yaml_file)
y   = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=dtype_th, device=str(device))

rxns = parseReactions(yaml_file)
r_e, r_tb, r_f, _ = build_reactions(vec_th.species, rxns, vec_th, str(device), dtype_th)
mech = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th, dtype=dtype_np)

reactions_index_to_type = [r['type'] for r in rxns]
n_sp = mech.n_species
ALL_IDX = list(range(n_sp))   # à placer après la définition de n_sp


# Système Torch complet (pour omega et JVP)
batched_elem = ElementaryReactionBatchContext(r_e, device=str(device), dtype=dtype_th)
batched_tb   = ThreeBodyReactionBatchContext(r_tb, device=str(device), dtype=dtype_th)
batched_f    = FalloffReactionBatchContext(r_f,    device=str(device), dtype=dtype_th)
rxn_sys      = ReactionSystemContext([batched_elem, batched_tb, batched_f])

print(f'  {n_sp} espèces,  '
      f'{mech.elementary.n_reactions} élémentaires,  '
      f'{mech.threebody.n_reactions} trois-corps,  '
      f'{mech.falloff.n_reactions} falloff\n')

# ─────────────────────────────────────────────────────────────────────────────
# Données aléatoires communes
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
T_np  = np.random.uniform(800,  4000, N_SAMPLES).astype(dtype_np)
C_np  = np.abs(np.random.randn(N_SAMPLES, n_sp).astype(dtype_np)) * 5.0   # mol/m³

T_jac = T_np[:N_JAC]
C_jac = C_np[:N_JAC]

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — Validation des taux de réaction Torch vs Cantera
# ─────────────────────────────────────────────────────────────────────────────
print('='*70)
print('Étape 1 — Validation des taux de réaction Torch vs Cantera')

# ── Cantera ────────────────────────────────────────────────────────────────
omega_can = np.zeros((N_SAMPLES, n_sp), dtype=dtype_np)
RR_e_can  = []; RR_tb_can = []; RR_f_can = []

for i in range(N_SAMPLES):
    gas.TP = float(T_np[i]), 101325.0
    gas.concentrations = C_np[i] * 1e-3        # mol/m³ → kmol/m³
    omega_can[i] = gas.net_production_rates * 1e3   # kmol → mol

    row_e, row_tb, row_f = [], [], []
    for j, rtype in enumerate(reactions_index_to_type):
        RR = gas.net_rates_of_progress[j] * 1e3
        if   rtype == 'elementary': row_e.append(RR)
        elif rtype == 'threebody':  row_tb.append(RR)
        elif 'falloff' in rtype:    row_f.append(RR)
    RR_e_can.append(row_e); RR_tb_can.append(row_tb); RR_f_can.append(row_f)

RR_e_can  = np.array(RR_e_can)
RR_tb_can = np.array(RR_tb_can)
RR_f_can  = np.array(RR_f_can)

# ── Torch ──────────────────────────────────────────────────────────────────
T_t = torch.tensor(T_np, device=device, dtype=dtype_th)
C_t = torch.tensor(C_np, device=device, dtype=dtype_th)

rxn_sys.evaluate_context(T_t, C_t)
omega_torch = rxn_sys.omega_total().cpu().numpy()   # (N_SAMPLES, n_sp), mol/m³/s

# RR par type depuis le contexte
RR_e_torch  = batched_elem._ctx_RR if hasattr(batched_elem, '_ctx_RR') else None
# On recalcule proprement depuis les contextes
ctx_e  = batched_elem.evaluate_context(T_t, C_t)
ctx_tb = batched_tb.evaluate_context(T_t, C_t)
ctx_f  = batched_f.evaluate_context(T_t, C_t)

RR_e_torch  = ctx_e.RR.cpu().numpy()    # (N_SAMPLES, n_elem)
RR_tb_torch = ctx_tb.RR.cpu().numpy()   # (N_SAMPLES, n_tb)
RR_f_torch  = ctx_f.RR.cpu().numpy()    # (N_SAMPLES, n_falloff)

# Recalcul omega total correct depuis les contextes séparés
omega_torch_total = (
    batched_elem.omega_from_RR(ctx_e.RR)
    + batched_tb.omega_from_RR(ctx_tb.RR)
    + batched_f.omega_from_RR(ctx_f.RR)
).cpu().numpy()

# ── Métriques ─────────────────────────────────────────────────────────────
print('\n  ── Taux de réaction (RR) ──')
d_e,  r_e_  = stats('Élémentaires  Torch vs Cantera', RR_e_torch,       RR_e_can)
d_tb, r_tb_ = stats('Trois-corps   Torch vs Cantera', RR_tb_torch,      RR_tb_can)
d_f,  r_f_  = stats('Falloff       Torch vs Cantera', RR_f_torch,       RR_f_can)
d_om, r_om_ = stats('omega total   Torch vs Cantera', omega_torch_total, omega_can)

# ── Figure 1.1 — Boxplots RR ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
boxplot(axes[0],
        [d_e, d_tb, d_f, d_om],
        ['Elem.\nRR', '3-body\nRR', 'Falloff\nRR', r'$\dot\omega$'],
        r'MAE (mol m$^{-3}$ s$^{-1}$)')
axes[0].set_title('Absolute error', fontsize=8)

boxplot(axes[1],
        [r_e_, r_tb_, r_f_, r_om_],
        ['Elem.\nRR', '3-body\nRR', 'Falloff\nRR', r'$\dot\omega$'],
        'Relative error (-–)')
axes[1].set_title('Relative error', fontsize=8)
plt.suptitle('Step 1 — Torch vs Cantera : reaction rates', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step1_RR_validation.png', bbox_inches='tight')
plt.close()

# ── Figure 1.2 — Scatter omega Torch vs Cantera ────────────────────────────
fig, ax = plt.subplots()
scatter_11(ax, omega_can, omega_torch_total,
           r'Cantera $\dot\omega$ (mol m$^{-3}$ s$^{-1}$)',
           r'Torch $\dot\omega$ (mol m$^{-3}$ s$^{-1}$)',
           COLORS[0], title=r'$\dot\omega$ Torch vs Cantera')
plt.tight_layout()
plt.savefig('jacobian_figures/step1_omega_scatter.png', bbox_inches='tight')
plt.close()

print('\n  Figures : step1_RR_validation.png, step1_omega_scatter.png')

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — Jacobienne Torch via autodiff (référence numérique)
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('Étape 2 — Jacobienne Torch via autodiff (torch.autograd.functional.jvp)')

# On calcule la Jacobienne complète du RHS adiabatique :
#   xk = (T, C_0, ..., C_{n_sp-1})  →  dTC = (dT/dt, dC_0/dt, ...)
# via n_state applications de JVP sur les vecteurs de base canoniques.

def _thermo_scalar_np(gas_obj):
    """Propriétés thermo scalaires depuis un objet Cantera."""
    C_mol = gas_obj.concentrations * 1e3    # mol/m³
    T = gas_obj.T
    ntot = C_mol.sum()
    Xs   = C_mol / ntot if ntot > 0 else np.zeros_like(C_mol)
    cp_s = gas_obj.standard_cp_R * R_GAS
    cv_s = cp_s - R_GAS
    h_s  = gas_obj.standard_enthalpies_RT * R_GAS * T
    u_s  = h_s - R_GAS * T
    cv_mix = (Xs * cv_s).sum()
    su_om  = (u_s * gas_obj.net_production_rates * 1e3).sum()
    return ntot, cv_mix, su_om, cv_s, u_s



def _torch_rhs(T_1d, C_2d):
    """
    RHS adiabatique complet.
    T_1d  : (B,)        température
    C_2d  : (B, n_sp)   concentrations mol/m³
    Retourne (dT/dt, dC/dt) séparément.
    """
    rxn_sys.evaluate_context(T_1d, C_2d)
    omega = rxn_sys.omega_total()                           # (B, n_sp)
    ntot  = C_2d.sum(dim=1, keepdim=True)                  # (B, 1)

    # cv_s  = cp_s - R  (molaire, J/mol/K)
    cp_s  = vec_th.cp_species(T_1d, ALL_IDX)               # (B, n_sp)
    cv_s  = cp_s - R_GAS                                   # (B, n_sp)

    # u_s   molaire (J/mol)
    u_s   = vec_th.u_species_mol(T_1d)                     # (B, n_sp)

    Xs     = C_2d / (ntot + 1e-300)                        # (B, n_sp)
    cv_mix = (Xs * cv_s).sum(dim=1)                        # (B,)
    su_om  = (u_s * omega).sum(dim=1)                      # (B,)
    denom  = ntot.squeeze(1) * cv_mix                      # (B,)
    dT_dt  = -su_om / (denom + 1e-300)                     # (B,)
    return dT_dt, omega

def _omega_fn_flat(TC_flat, B, n_sp_):
    """Wrapper plat pour torch.autograd.functional.jvp."""
    T_ = TC_flat[:B]
    C_ = TC_flat[B:].reshape(B, n_sp_)
    rxn_sys.evaluate_context(T_, C_)
    w = rxn_sys.omega_total()
    return w.reshape(-1)


def _rhs_fn_flat(TC_flat, B, n_sp_):
    """RHS complet (dT/dt, dC/dt) en vecteur plat."""
    T_ = TC_flat[:B]
    C_ = TC_flat[B:].reshape(B, n_sp_)
    dT_dt, omega = _torch_rhs(T_, C_)
    return torch.cat([dT_dt, omega.reshape(-1)])


n_state = n_sp + 1

print(f'\n  Calcul de la Jacobienne Torch-autodiff ({N_JAC} réacteurs) ...',
      end=' ', flush=True)

T_jac_t = torch.tensor(T_jac, device=device, dtype=dtype_th)
C_jac_t = torch.tensor(C_jac, device=device, dtype=dtype_th)

from torch.autograd.functional import jacobian as torch_jacobian

J_torch_auto = np.zeros((N_JAC, n_state, n_state), dtype=dtype_np)

t0 = time.perf_counter()
for i in range(N_JAC):
    T_i = T_jac_t[i:i+1].detach().requires_grad_(True)   # (1,)
    C_i = C_jac_t[i:i+1].detach().requires_grad_(True)   # (1, n_sp)

    def rhs_i(T_, C_):
        dT_dt, omega = _torch_rhs(T_, C_)
        return torch.cat([dT_dt, omega.reshape(-1)])      # (n_state,)

    J_i = torch_jacobian(rhs_i, (T_i, C_i))
    # J_i est un tuple : (d_rhs/dT shape (n_state,1), d_rhs/dC shape (n_state,1,n_sp))
    dRHS_dT = J_i[0].reshape(n_state, 1)                 # (n_state, 1)
    dRHS_dC = J_i[1].reshape(n_state, n_sp)              # (n_state, n_sp)
    J_torch_auto[i, :, 0]  = dRHS_dT[:, 0].detach().cpu().numpy()
    J_torch_auto[i, :, 1:] = dRHS_dC.detach().cpu().numpy()

t_autodiff = time.perf_counter() - t0
print(f'{t_autodiff:.1f}s')
print(f'  Jacobienne Torch-autodiff calculée  shape={J_torch_auto.shape}')



# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — Jacobienne Cantera analytique vs Torch-autodiff
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('Étape 3 — Jacobienne Cantera analytique vs Torch-autodiff')


def _dcvdT_analytic(gas_obj):
    """dcvdT_s depuis les polynômes NASA7 (analytique)."""
    T = gas_obj.T
    dcvdT = np.zeros(gas_obj.n_species)
    for k in range(gas_obj.n_species):
        sp     = gas_obj.species(k)
        coeffs = sp.thermo.coeffs
        Tmid   = coeffs[0]
        a      = coeffs[1:8] if T >= Tmid else coeffs[8:15]
        dcpdT_R = (a[1] + 2.0*a[2]*T + 3.0*a[3]*T**2 + 4.0*a[4]*T**3)
        dcvdT[k] = dcpdT_R * R_GAS
    return dcvdT


def _cantera_jacobian_full(T, C):
    """
    Jacobienne analytique complète (n_state × n_state) via Cantera.
    Bloc chimique  J[1:, :] : d(omega)/d(T, C)
    Ligne thermo   J[0,  :] : d(dT/dt)/d(T, C)
    """
    g = ct.Solution(yaml_file)
    g.TP = float(T), 101325.0
    g.concentrations = np.maximum(C, 0.0) * 1e-3

    n_st = n_sp + 1
    J    = np.zeros((n_st, n_st), dtype=dtype_np)

    # ── Bloc chimique ─────────────────────────────────────────
    ddCi = np.array(g.net_production_rates_ddCi)       # (n_sp, n_sp), s⁻¹
    ddT  = np.array(g.net_production_rates_ddT) * 1e3  # (n_sp,), mol/m³/s/K
    J[1:, 0]  = ddT
    J[1:, 1:] = ddCi

    # ── Ligne thermique ───────────────────────────────────────
    ntot, cv_mix, su_om, cv_s, u_s = _thermo_scalar_np(g)
    om    = g.net_production_rates * 1e3
    denom = ntot * cv_mix
    if abs(denom) < 1e-300:
        return J

    C_mol = g.concentrations * 1e3
    Xs    = C_mol / ntot if ntot > 0 else np.zeros(n_sp)
    dcvdT = _dcvdT_analytic(g)

    # Colonne T
    dsuom_T  = (u_s * ddT).sum() + (cv_s * om).sum()
    ddenom_T = ntot * (Xs * dcvdT).sum()
    J[0, 0]  = -(dsuom_T * denom - su_om * ddenom_T) / (denom**2)

    # Colonnes C_s
    for s in range(n_sp):
        dsuom_s  = (u_s * ddCi[:, s]).sum()
        ddenom_s = cv_s[s]
        J[0, s+1] = -(dsuom_s * denom - su_om * ddenom_s) / (denom**2)

    return J


print(f'\n  Calcul Cantera (N={N_JAC}) ...', end=' ', flush=True)
t0 = time.perf_counter()
J_cantera = np.stack([_cantera_jacobian_full(T_jac[i], C_jac[i]) for i in range(N_JAC)])
t_can = time.perf_counter() - t0
print(f'{t_can:.1f}s')

# ── Métriques ─────────────────────────────────────────────────────────────
print('\n  ── Torch-autodiff vs Cantera ──')
d_total, r_total = stats('Total (n_state × n_state)',
                          J_torch_auto, J_cantera)
d_chem,  r_chem  = stats('Bloc chimique [1:, :]',
                          J_torch_auto[:, 1:, :], J_cantera[:, 1:, :])
d_therm, r_therm = stats('Ligne thermo  [0, :]',
                          J_torch_auto[:, 0, :], J_cantera[:, 0, :])

# ── Figure 3.1 — Boxplots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
boxplot(axes[0],
        [d_total, d_chem, d_therm],
        ['Total', 'Chimique\n[1:,:]', 'Therm.\n[0,:]'],
        'MAE (SI)')
axes[0].set_title('Absolute error', fontsize=8)
boxplot(axes[1],
        [r_total, r_chem, r_therm],
        ['Total', 'Chimique\n[1:,:]', 'Therm.\n[0,:]'],
        'Relative error (-–)')
axes[1].set_title('Relative error', fontsize=8)
plt.suptitle('Step 3 — Torch-autodiff vs Cantera : Jacobian', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step3_autodiff_vs_cantera_boxplots.png', bbox_inches='tight')
plt.close()

# ── Figure 3.2 — Scatter plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
scatter_11(axes[0],
           J_cantera[:, 1:, :].ravel(), J_torch_auto[:, 1:, :].ravel(),
           'Cantera (SI)', 'Torch autodiff (SI)', COLORS[1],
           title='Chimique $J[1:,:]$')
scatter_11(axes[1],
           J_cantera[:, 0, :].ravel(), J_torch_auto[:, 0, :].ravel(),
           'Cantera (SI)', 'Torch autodiff (SI)', COLORS[2],
           title='Therm. $J[0,:]$')
plt.suptitle('Step 3 — Torch-autodiff vs Cantera : scatter', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step3_autodiff_vs_cantera_scatter.png', bbox_inches='tight')
plt.close()

# ── Figure 3.3 — Heatmap erreur relative moyenne ─────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
heatmap_log(ax, r_total.mean(axis=0), 'Torch-autodiff vs Cantera — mean rel. error')
plt.tight_layout()
plt.savefig('jacobian_figures/step3_autodiff_vs_cantera_heatmap.png', bbox_inches='tight')
plt.close()

# ── Figure 3.4 — MAE par colonne xk_j ────────────────────────────────────
col_mae_autodiff = np.abs(J_torch_auto - J_cantera).mean(axis=(0, 1))    # (n_state,)
x = np.arange(n_sp + 1)
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.bar(x, col_mae_autodiff, color=COLORS[1], alpha=0.7, width=1.0)
ax.set_yscale('log')
ax.set_xlabel('Column index $j$  (0=T, j=C$_{j-1}$)')
ax.set_ylabel('MAE (SI)')
ax.set_title('Torch-autodiff vs Cantera — MAE per column', fontsize=8)
# Annotation des 5 pires colonnes
sp_names = gas.species_names
for k in np.argsort(col_mae_autodiff)[-5:][::-1]:
    name = 'T' if k == 0 else sp_names[k-1]
    ax.text(k, col_mae_autodiff[k] * 1.5, name, ha='center', fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig('jacobian_figures/step3_autodiff_vs_cantera_per_column.png', bbox_inches='tight')
plt.close()

print('\n  Figures : step3_autodiff_vs_cantera_*.png')


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — Jacobiennes CUDA vs Torch-autodiff
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('Étape 4 — Jacobiennes CUDA (PT et PB) vs Torch-autodiff')

# ── Warmup JIT ────────────────────────────────────────────────────────────
jac_pt = CUDAJacobianEvaluatorPT(mech, dtype=dtype_np)
jac_pb = CUDAJacobianEvaluatorPB(mech, dtype=dtype_np)

print('  Warmup JIT per-thread ...', end=' ', flush=True)
jac_pt.evaluate(T_jac[:1], C_jac[:1])
print('done')

print('  Warmup JIT per-block  ...', end=' ', flush=True)
jac_pb.evaluate(T_jac[:1], C_jac[:1])
print('done')

# ── Calcul ────────────────────────────────────────────────────────────────
print(f'\n  CUDA per-thread ({N_JAC} réacteurs) ...', end=' ', flush=True)
t0 = time.perf_counter()
J_pt = jac_pt.evaluate(T_jac, C_jac)     # (N_JAC, n_state, n_state)
t_pt = time.perf_counter() - t0
print(f'{t_pt*1e3:.2f} ms  ({t_pt/N_JAC*1e6:.1f} µs/réacteur)')

print(f'  CUDA per-block  ({N_JAC} réacteurs) ...', end=' ', flush=True)
t0 = time.perf_counter()
J_pb = jac_pb.evaluate(T_jac, C_jac)     # (N_JAC, n_state, n_state)
t_pb = time.perf_counter() - t0
print(f'{t_pb*1e3:.2f} ms  ({t_pb/N_JAC*1e6:.1f} µs/réacteur)')

ref = J_torch_auto     # référence = Torch-autodiff

# ── Métriques PT ──────────────────────────────────────────────────────────
print('\n  ── CUDA per-thread vs Torch-autodiff ──')
d_pt,    r_pt    = stats('Total',           J_pt,          ref)
d_pt_c,  r_pt_c  = stats('Chimique [1:,:]', J_pt[:, 1:, :], ref[:, 1:, :])
d_pt_t,  r_pt_t  = stats('Thermo   [0, :]', J_pt[:, 0,  :], ref[:, 0,  :])

print('\n  ── CUDA per-block vs Torch-autodiff ──')
d_pb,    r_pb    = stats('Total',           J_pb,          ref)
d_pb_c,  r_pb_c  = stats('Chimique [1:,:]', J_pb[:, 1:, :], ref[:, 1:, :])
d_pb_t,  r_pb_t  = stats('Thermo   [0, :]', J_pb[:, 0,  :], ref[:, 0,  :])

print('\n  ── CUDA per-block vs per-thread ──')
d_pp, r_pp = stats('Total', J_pb, J_pt)

# ── Figure 4.1 — Boxplots PT et PB côte à côte ───────────────────────────
labels_b = ['PT\ntotal', 'PT\nchem', 'PT\ntherm',
            'PB\ntotal', 'PB\nchem', 'PB\ntherm']
data_abs = [d_pt, d_pt_c, d_pt_t, d_pb, d_pb_c, d_pb_t]
data_rel = [r_pt, r_pt_c, r_pt_t, r_pb, r_pb_c, r_pb_t]

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, data, ylabel in [
    (axes[0], data_abs, 'MAE (SI)'),
    (axes[1], data_rel, 'Relative error (-–)'),
]:
    ax.boxplot(
        [d.ravel() for d in data],
        labels=labels_b,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black'),
        boxprops=dict(facecolor=COLORS[2], alpha=0.6),
    )
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=20)
plt.suptitle('Step 4 — CUDA PT \& PB vs Torch-autodiff', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step4_cuda_vs_autodiff_boxplots.png', bbox_inches='tight')
plt.close()

# ── Figure 4.2 — Scatter PT et PB ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
scatter_11(axes[0], ref.ravel(), J_pt.ravel(),
           'Torch-autodiff (SI)', 'CUDA PT (SI)', COLORS[1], title='CUDA per-thread')
scatter_11(axes[1], ref.ravel(), J_pb.ravel(),
           'Torch-autodiff (SI)', 'CUDA PB (SI)', COLORS[2], title='CUDA per-block')
plt.suptitle('Step 4 — CUDA vs Torch-autodiff : scatter', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step4_cuda_vs_autodiff_scatter.png', bbox_inches='tight')
plt.close()

# ── Figure 4.3 — MAE par colonne xk_j ────────────────────────────────────
col_mae_pt = np.abs(J_pt - ref).mean(axis=(0, 1))
col_mae_pb = np.abs(J_pb - ref).mean(axis=(0, 1))

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, col_mae, label, col in [
    (axes[0], col_mae_pt, 'CUDA PT vs Torch-autodiff', COLORS[1]),
    (axes[1], col_mae_pb, 'CUDA PB vs Torch-autodiff', COLORS[2]),
]:
    ax.bar(x, col_mae, color=col, alpha=0.7, width=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('Column index $j$  (0=T)')
    ax.set_ylabel('MAE (SI)')
    ax.set_title(label, fontsize=8)
    for k in np.argsort(col_mae)[-3:][::-1]:
        name = 'T' if k == 0 else sp_names[k-1]
        ax.text(k, col_mae[k] * 1.5, name, ha='center', fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig('jacobian_figures/step4_cuda_vs_autodiff_per_column.png', bbox_inches='tight')
plt.close()

# ── Figure 4.4 — Heatmaps erreur relative ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
heatmap_log(axes[0], r_pt.mean(axis=0),
            'CUDA PT vs Torch-autodiff')
heatmap_log(axes[1], r_pb.mean(axis=0),
            'CUDA PB vs Torch-autodiff')
plt.tight_layout()
plt.savefig('jacobian_figures/step4_cuda_vs_autodiff_heatmaps.png', bbox_inches='tight')
plt.close()

# ── Figure 4.5 — PB vs PT ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
scatter_11(axes[0], J_pt.ravel(), J_pb.ravel(),
           'CUDA PT (SI)', 'CUDA PB (SI)', COLORS[3], title='PB vs PT')
col_mae_pp = np.abs(J_pb - J_pt).mean(axis=(0, 1))
axes[1].bar(x, col_mae_pp, color=COLORS[3], alpha=0.7, width=1.0)
axes[1].set_yscale('log')
axes[1].set_xlabel('Column index $j$  (0=T)')
axes[1].set_ylabel('MAE (SI)')
axes[1].set_title('CUDA PB vs PT — MAE per column', fontsize=8)
plt.tight_layout()
plt.savefig('jacobian_figures/step4_pb_vs_pt.png', bbox_inches='tight')
plt.close()

print('\n  Figures : step4_*.png')

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3b — Analyse JVP : autograd vs analytique Cantera
# Objectif : isoler l'origine de l'écart CUDA vs autodiff
# On calcule Jv = J @ v pour des vecteurs v aléatoires, et on compare :
#   - JVP via autograd Torch  (dérivé de la Jacobienne autodiff)
#   - JVP analytique Cantera  (ddCi @ vC + ddT * vT)
#   - JVP CUDA PT et PB       (depuis les kernels)
# Séparément : bloc chimique (omega) et bloc thermique (dT/dt)
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('Étape 3b — Analyse JVP : autograd vs analytique (chimique + thermique)')

np.random.seed(77)
N_JVP = N_JAC  # on réutilise les mêmes points T_jac, C_jac

# Vecteurs tangents aléatoires normalisés
vT_raw = np.random.randn(N_JVP).astype(dtype_np)
vC_raw = np.random.randn(N_JVP, n_sp).astype(dtype_np)
norms  = np.sqrt(vT_raw**2 + (vC_raw**2).sum(axis=1))
vT_jvp = vT_raw / norms
vC_jvp = vC_raw / norms[:, None]

# ── 1. JVP autograd Torch = J_torch_auto @ v ─────────────────────────────
# On réutilise la Jacobienne déjà calculée
v_flat = np.concatenate([vT_jvp[:, None], vC_jvp], axis=1)  # (N_JVP, n_state)
# Jv[i] = J_torch_auto[i] @ v_flat[i]
Jv_auto_full  = np.einsum('ijk,ik->ij', J_torch_auto, v_flat)  # (N_JVP, n_state)
Jv_auto_therm = Jv_auto_full[:, 0]      # (N_JVP,)       ligne dT/dt
Jv_auto_chem  = Jv_auto_full[:, 1:]     # (N_JVP, n_sp)  bloc omega

# ── 2. JVP analytique Cantera ────────────────────────────────────────────
print(f'  Calcul JVP analytique Cantera (N={N_JVP}) ...', end=' ', flush=True)
t0 = time.perf_counter()

Jv_can_chem  = np.zeros((N_JVP, n_sp), dtype=dtype_np)
Jv_can_therm = np.zeros(N_JVP, dtype=dtype_np)

for i in range(N_JVP):
    g = ct.Solution(yaml_file)
    g.TP = float(T_jac[i]), 101325.0
    g.concentrations = np.maximum(C_jac[i], 0.0) * 1e-3

    # Bloc chimique
    ddCi = np.array(g.net_production_rates_ddCi)       # (n_sp, n_sp), s⁻¹
    ddT  = np.array(g.net_production_rates_ddT) * 1e3  # (n_sp,), mol/m³/s/K
    Jv_can_chem[i] = ddCi @ vC_jvp[i] + ddT * vT_jvp[i]

    # Bloc thermique : on réutilise la Jacobienne Cantera déjà calculée
    # J_cantera[i, 0, :] @ v_flat[i]
    Jv_can_therm[i] = J_cantera[i, 0, :] @ v_flat[i]

print(f'{time.perf_counter()-t0:.2f}s')

# ── 3. JVP CUDA PT = J_pt @ v ────────────────────────────────────────────
Jv_pt_full  = np.einsum('ijk,ik->ij', J_pt, v_flat)
Jv_pt_therm = Jv_pt_full[:, 0]
Jv_pt_chem  = Jv_pt_full[:, 1:]

Jv_pb_full  = np.einsum('ijk,ik->ij', J_pb, v_flat)
Jv_pb_therm = Jv_pb_full[:, 0]
Jv_pb_chem  = Jv_pb_full[:, 1:]

# ── Métriques ─────────────────────────────────────────────────────────────
print('\n  ── JVP bloc chimique (omega) ──')
d_ac_c,  r_ac_c  = stats('Autograd  vs Cantera', Jv_auto_chem,  Jv_can_chem)
d_ptc_c, r_ptc_c = stats('CUDA PT   vs Cantera', Jv_pt_chem,    Jv_can_chem)
d_pbc_c, r_pbc_c = stats('CUDA PB   vs Cantera', Jv_pb_chem,    Jv_can_chem)
d_ptc_a, r_ptc_a = stats('CUDA PT   vs Autograd', Jv_pt_chem,   Jv_auto_chem)
d_pbc_a, r_pbc_a = stats('CUDA PB   vs Autograd', Jv_pb_chem,   Jv_auto_chem)

print('\n  ── JVP bloc thermique (dT/dt) ──')
d_ac_t,  r_ac_t  = stats('Autograd  vs Cantera', Jv_auto_therm[:, None], Jv_can_therm[:, None])
d_ptc_t, r_ptc_t = stats('CUDA PT   vs Cantera', Jv_pt_therm[:, None],   Jv_can_therm[:, None])
d_pbc_t, r_pbc_t = stats('CUDA PB   vs Cantera', Jv_pb_therm[:, None],   Jv_can_therm[:, None])
d_ptc_at,r_ptc_at= stats('CUDA PT   vs Autograd', Jv_pt_therm[:, None],  Jv_auto_therm[:, None])
d_pbc_at,r_pbc_at= stats('CUDA PB   vs Autograd', Jv_pb_therm[:, None],  Jv_auto_therm[:, None])

# ── Figure 3b.1 — Boxplots JVP chimique ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
labels_jvp = ['Auto\nvs Can', 'PT\nvs Can', 'PB\nvs Can',
               'PT\nvs Auto', 'PB\nvs Auto']
for ax, data, ylabel in [
    (axes[0],
     [d_ac_c, d_ptc_c, d_pbc_c, d_ptc_a, d_pbc_a],
     r'MAE JVP$_\omega$ (mol m$^{-3}$ s$^{-1}$)'),
    (axes[1],
     [r_ac_c, r_ptc_c, r_pbc_c, r_ptc_a, r_pbc_a],
     r'rMAE JVP$_\omega$ (--)'),
]:
    ax.boxplot([d.ravel() for d in data], labels=labels_jvp,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='black'),
               boxprops=dict(facecolor=COLORS[4], alpha=0.6))
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
plt.suptitle('Step 3b — JVP chimique : autograd / CUDA / Cantera', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step3b_jvp_chem_boxplots.png', bbox_inches='tight')
plt.close()

# ── Figure 3b.2 — Boxplots JVP thermique ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, data, ylabel in [
    (axes[0],
     [d_ac_t, d_ptc_t, d_pbc_t, d_ptc_at, d_pbc_at],
     r'MAE JVP$_{dT/dt}$ (K s$^{-1}$)'),
    (axes[1],
     [r_ac_t, r_ptc_t, r_pbc_t, r_ptc_at, r_pbc_at],
     r'rMAE JVP$_{dT/dt}$ (--)'),
]:
    ax.boxplot([d.ravel() for d in data], labels=labels_jvp,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='black'),
               boxprops=dict(facecolor=COLORS[5], alpha=0.6))
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
plt.suptitle('Step 3b — JVP thermique : autograd / CUDA / Cantera', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step3b_jvp_therm_boxplots.png', bbox_inches='tight')
plt.close()

# ── Figure 3b.3 — Scatter JVP chimique : PT et PB vs Cantera ─────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
scatter_11(axes[0], Jv_can_chem.ravel(), Jv_pt_chem.ravel(),
           'Cantera (SI)', 'CUDA PT (SI)', COLORS[1], title=r'JVP$_\omega$ CUDA PT')
scatter_11(axes[1], Jv_can_chem.ravel(), Jv_pb_chem.ravel(),
           'Cantera (SI)', 'CUDA PB (SI)', COLORS[2], title=r'JVP$_\omega$ CUDA PB')
plt.suptitle('Step 3b — JVP chimique scatter vs Cantera', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step3b_jvp_chem_scatter.png', bbox_inches='tight')
plt.close()

# ── Figure 3b.4 — Scatter JVP thermique ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
scatter_11(axes[0], Jv_can_therm, Jv_pt_therm,
           'Cantera (SI)', 'CUDA PT (SI)', COLORS[1], title=r'JVP$_{dT/dt}$ CUDA PT')
scatter_11(axes[1], Jv_can_therm, Jv_pb_therm,
           'Cantera (SI)', 'CUDA PB (SI)', COLORS[2], title=r'JVP$_{dT/dt}$ CUDA PB')
plt.suptitle('Step 3b — JVP thermique scatter vs Cantera', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step3b_jvp_therm_scatter.png', bbox_inches='tight')
plt.close()

# ── Figure 3b.5 — MAE par espèce (JVP chimique) ──────────────────────────
sp_mae_auto = np.abs(Jv_auto_chem - Jv_can_chem).mean(axis=0)
sp_mae_pt   = np.abs(Jv_pt_chem   - Jv_can_chem).mean(axis=0)
sp_mae_pb   = np.abs(Jv_pb_chem   - Jv_can_chem).mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
x_sp = np.arange(n_sp)
for ax, sp_mae, label, col in [
    (axes[0], sp_mae_auto, 'Autograd vs Cantera', COLORS[0]),
    (axes[1], sp_mae_pt,   'CUDA PT  vs Cantera', COLORS[1]),
    (axes[2], sp_mae_pb,   'CUDA PB  vs Cantera', COLORS[2]),
]:
    ax.bar(x_sp, sp_mae, color=col, alpha=0.7, width=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('Species index')
    ax.set_ylabel(r'MAE JVP$_\omega$')
    ax.set_title(label, fontsize=8)
    for k in np.argsort(sp_mae)[-3:][::-1]:
        ax.text(k, sp_mae[k]*1.5, sp_names[k],
                ha='center', fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig('jacobian_figures/step3b_jvp_chem_per_species.png', bbox_inches='tight')
plt.close()

print('\n  Figures : step3b_jvp_*.png')
print('  Interprétation :')
print('    Si "Auto vs Can" ~ "PT vs Can" → le bug est dans le kernel CUDA')
print('    Si "PT vs Can"  >> "Auto vs Can" → le bug est dans le kernel CUDA')
print('    Si "PT vs Auto" ~ "PT vs Can"   → autodiff et Cantera sont cohérents,'
      ' kernel CUDA diverge')


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2b — Comparaison JVP : analytique Torch vs autograd vs Cantera
# Bloc chimique uniquement (omega), car rxn_sys.jvp ne couvre pas dT/dt
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('Étape 2b — JVP chimique : analytique Torch vs autograd vs Cantera')

np.random.seed(55)
N_JVP = N_JAC  # mêmes points T_jac, C_jac

# Vecteurs tangents aléatoires normalisés
vT_raw = np.random.randn(N_JVP).astype(dtype_np)
vC_raw = np.random.randn(N_JVP, n_sp).astype(dtype_np)
norms  = np.sqrt(vT_raw**2 + (vC_raw**2).sum(axis=1))
vT_np  = vT_raw / norms
vC_np  = vC_raw / norms[:, None]

vT_t = torch.tensor(vT_np, device=device, dtype=dtype_th)
vC_t = torch.tensor(vC_np, device=device, dtype=dtype_th)

# ── 1. JVP autograd = J_torch_auto[i, 1:, :] @ v ─────────────────────────
# Jacobienne autodiff déjà calculée, on extrait le bloc chimique [1:, :]
# v_flat = [vT, vC]  shape (n_state,)
v_flat_np = np.concatenate([vT_np[:, None], vC_np], axis=1)  # (N_JVP, n_state)
Jv_auto_chem = np.einsum('ijk,ik->ij',
                          J_torch_auto[:, 1:, :],
                          v_flat_np)                          # (N_JVP, n_sp)

# ── 2. JVP analytique Torch (rxn_sys.jvp) ────────────────────────────────
print('  Calcul JVP analytique Torch ...', end=' ', flush=True)
t0 = time.perf_counter()

# Il faut évaluer le contexte sur T_jac, C_jac d'abord
T_jac_t = torch.tensor(T_jac, device=device, dtype=dtype_th)
C_jac_t = torch.tensor(C_jac, device=device, dtype=dtype_th)
rxn_sys.evaluate_context(T_jac_t, C_jac_t)

Jv_torch_chem = rxn_sys.jvp(vT_t, vC_t).cpu().numpy()       # (N_JVP, n_sp)
torch.cuda.synchronize()
print(f'{time.perf_counter()-t0:.4f}s')

# ── 3. JVP analytique Cantera ─────────────────────────────────────────────
print(f'  Calcul JVP analytique Cantera (N={N_JVP}) ...', end=' ', flush=True)
t0 = time.perf_counter()

Jv_can_chem = np.zeros((N_JVP, n_sp), dtype=dtype_np)
for i in range(N_JVP):
    g = ct.Solution(yaml_file)
    g.TP = float(T_jac[i]), 101325.0
    g.concentrations = np.maximum(C_jac[i], 0.0) * 1e-3
    ddCi = np.array(g.net_production_rates_ddCi)        # (n_sp, n_sp), s⁻¹
    ddT  = np.array(g.net_production_rates_ddT) * 1e3   # (n_sp,), mol/m³/s/K
    Jv_can_chem[i] = ddCi @ vC_np[i] + ddT * vT_np[i]

print(f'{time.perf_counter()-t0:.2f}s')

# ── Métriques ─────────────────────────────────────────────────────────────
print('\n  ── JVP bloc chimique (omega) ──')
d_torch_can, r_torch_can = stats(
    'Analytique Torch vs Cantera', Jv_torch_chem, Jv_can_chem)
d_auto_can,  r_auto_can  = stats(
    'Autograd       vs Cantera',   Jv_auto_chem,  Jv_can_chem)
d_torch_auto,r_torch_auto= stats(
    'Analytique Torch vs Autograd',Jv_torch_chem, Jv_auto_chem)

# ── Figure 2b.1 — Boxplots ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
labels_jvp = ['Torch anal.\nvs Cantera',
               'Autograd\nvs Cantera',
               'Torch anal.\nvs Autograd']
for ax, data, ylabel in [
    (axes[0],
     [d_torch_can, d_auto_can, d_torch_auto],
     r'MAE JVP$_\omega$ (mol m$^{-3}$ s$^{-1}$)'),
    (axes[1],
     [r_torch_can, r_auto_can, r_torch_auto],
     r'rMAE JVP$_\omega$ (--)'),
]:
    ax.boxplot([d.ravel() for d in data], labels=labels_jvp,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='black'),
               boxprops=dict(facecolor=COLORS[4], alpha=0.6))
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
plt.suptitle('Step 2b — JVP chimique : Torch analytique / autograd / Cantera',
             fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step2b_jvp_chem_boxplots.png', bbox_inches='tight')
plt.close()

# ── Figure 2b.2 — Scatter ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
scatter_11(axes[0], Jv_can_chem.ravel(), Jv_torch_chem.ravel(),
           'Cantera (SI)', 'Torch analytique (SI)', COLORS[1],
           title=r'JVP$_\omega$ Torch analytique')
scatter_11(axes[1], Jv_can_chem.ravel(), Jv_auto_chem.ravel(),
           'Cantera (SI)', 'Autograd (SI)', COLORS[2],
           title=r'JVP$_\omega$ Autograd')
scatter_11(axes[2], Jv_auto_chem.ravel(), Jv_torch_chem.ravel(),
           'Autograd (SI)', 'Torch analytique (SI)', COLORS[3],
           title=r'JVP$_\omega$ Torch vs Autograd')
plt.suptitle('Step 2b — JVP chimique scatter', fontsize=9)
plt.tight_layout()
plt.savefig('jacobian_figures/step2b_jvp_chem_scatter.png', bbox_inches='tight')
plt.close()

# ── Figure 2b.3 — MAE par espèce ──────────────────────────────────────────
sp_mae_torch_can = np.abs(Jv_torch_chem - Jv_can_chem).mean(axis=0)
sp_mae_auto_can  = np.abs(Jv_auto_chem  - Jv_can_chem).mean(axis=0)
sp_mae_torch_auto= np.abs(Jv_torch_chem - Jv_auto_chem).mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(14, 3))
x_sp = np.arange(n_sp)
for ax, sp_mae, label, col in [
    (axes[0], sp_mae_torch_can,  'Torch anal. vs Cantera',  COLORS[1]),
    (axes[1], sp_mae_auto_can,   'Autograd vs Cantera',     COLORS[2]),
    (axes[2], sp_mae_torch_auto, 'Torch anal. vs Autograd', COLORS[3]),
]:
    ax.bar(x_sp, sp_mae, color=col, alpha=0.7, width=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('Species index')
    ax.set_ylabel(r'MAE JVP$_\omega$')
    ax.set_title(label, fontsize=8)
    for k in np.argsort(sp_mae)[-3:][::-1]:
        ax.text(k, sp_mae[k]*1.5, sp_names[k],
                ha='center', fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig('jacobian_figures/step2b_jvp_chem_per_species.png', bbox_inches='tight')
plt.close()

print('\n  Figures : step2b_jvp_*.png')
print('  Diagnostic :')
print(f'    Torch anal. vs Cantera  rMAE = {r_torch_can.mean():.3e}')
print(f'    Autograd    vs Cantera  rMAE = {r_auto_can.mean():.3e}')
print(f'    Torch anal. vs Autograd rMAE = {r_torch_auto.mean():.3e}')
print('  Si Torch anal. vs Autograd ~ 0 → JVP analytique cohérente avec autograd')
print('  Si Torch anal. vs Cantera  >> Autograd vs Cantera → bug dans jvp() Torch')



# ─────────────────────────────────────────────────────────────────────────────
# Résumé
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('RÉSUMÉ GÉNÉRAL')
print(f'  N_SAMPLES = {N_SAMPLES},  N_JAC = {N_JAC},  n_state = {n_state}')
print(f'\n  Étape 1 — omega Torch vs Cantera')
print(f'    MAE  = {d_om.mean():.3e}  rMAE = {r_om_.mean():.3e}')
print(f'\n  Étape 3 — Jacobienne Torch-autodiff vs Cantera')
print(f'    Total   MAE  = {d_total.mean():.3e}  rMAE = {r_total.mean():.3e}')
print(f'    Chimique MAE = {d_chem.mean():.3e}   rMAE = {r_chem.mean():.3e}')
print(f'    Thermo   MAE = {d_therm.mean():.3e}   rMAE = {r_therm.mean():.3e}')
print(f'\n  Étape 4 — CUDA PT vs Torch-autodiff')
print(f'    Total   MAE  = {d_pt.mean():.3e}  rMAE = {r_pt.mean():.3e}')
print(f'    Timing  PT   = {t_pt*1e3:.2f} ms  PB = {t_pb*1e3:.2f} ms  '
      f'Autodiff = {t_autodiff*1e3:.1f} ms')
print(f'\n  Étape 4 — CUDA PB vs Torch-autodiff')
print(f'    Total   MAE  = {d_pb.mean():.3e}  rMAE = {r_pb.mean():.3e}')
print(f'\n  Étape 4 — CUDA PB vs CUDA PT')
print(f'    Total   MAE  = {d_pp.mean():.3e}  rMAE = {r_pp.mean():.3e}')
print('\n  Toutes les figures dans jacobian_figures/')
print('='*70)