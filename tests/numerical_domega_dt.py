import matplotlib.pyplot as plt
import torch
import time
from torch.autograd.functional import jvp
import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.reactionTypes import ElementaryReactionBatch, ThreeBodyReactionBatch, FalloffReactionBatch, ReactionSystem
import numpy as np
import cantera as ct


tend = 10e-4
phi = 1.0
T0 = 1600.0
P0 = 101325.0


yaml_file = "../kineticsGPU/gri30.yaml"
dtype = torch.float64
device = 'cpu'
y = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device)
reactions = parseReactions(yaml_file)

elem_reacs = []
threebody_reacs = []
falloff_reacs = []


r_e, r_tb, r_f, system = build_reactions(vec_th.species, reactions, vec_th, device, dtype)

batched_elem = ElementaryReactionBatch(r_e, device=device, dtype=dtype)
batched_tb = ThreeBodyReactionBatch(r_tb, device=device, dtype=dtype)
batched_f = FalloffReactionBatch(r_f, device=device, dtype=dtype)
reaction_system = ReactionSystem([batched_elem, batched_tb, batched_f])


# === RHS vectorized ===
def rhs_adiabatic(TC, t):
    T = TC[:, 0]
    C = TC[:, 1:]  # mol/m3
    # cinétique
    dCdt = reaction_system.omega_total(T, C)
    # concentration totale
    n_tot = torch.sum(C, dim=1)  # (N,)
    # fractions molaires
    X = C / n_tot[:, None]
    # Cv (molaire)
    cv_mol_mix = vec_th.cv_mix_mol(T, X)  # (N,)
    # dU/dt = sum_k u_k * dC_k/dt
    # u_mix_mol renvoie la valeur du mélange pour DONNER Y, pas pour dCdt
    # → on doit calculer u_species_molar explicitement
    u_species = vec_th.u_species_mol(T)  # (N, nspecies)
    dUdt_molar = torch.sum(u_species * dCdt, dim=1)  # (N,)
    # dT/dt = - dU/dt / (n_tot * Cv)
    dTdt = - dUdt_molar / (n_tot * cv_mol_mix)
    # assemble
    dTCdt = torch.cat((dTdt[:, None], dCdt), dim=1)
    return dTCdt


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
omega_list = []
Tdot_list = []


t = 0.0
while t < tend:
    t = sim.step()
    g = reactor.thermo
    times_list.append(t)
    Y_list.append(g.Y.copy())
    T_list.append(g.T)
    # concentrations mol/m³ = kmol/m³ * 1e3
    C_list.append(g.concentrations * 1e3)
    W_list.append(g.molecular_weights.copy())
    omega_list.append(g.net_production_rates * 1e3)
    rho = g.density              # kg/m³
    cv  = g.cv_mass              # J/(kg·K)
    hrr = g.heat_release_rate    # W/m³

    Tdot = -hrr / (rho * cv)
    Tdot_list.append(Tdot)

times = torch.tensor(np.array(times_list), device=device, dtype=dtype, requires_grad=True)
Y_all = torch.tensor(np.array(Y_list), device=device, dtype=dtype, requires_grad=True)       # (N, n_sp)
T_all = torch.tensor(np.array(T_list), device=device, dtype=dtype, requires_grad=True)       # (N,)
C_all = torch.tensor(np.array(C_list), device=device, dtype=dtype, requires_grad=True)       # (N, n_sp)  mol/m³
W_all = torch.tensor(np.array(W_list), device=device, dtype=dtype, requires_grad=True)       # (N, n_sp)
omega_all = torch.tensor(np.array(omega_list), device=device, dtype=dtype, requires_grad=True)
Tdot_all = torch.tensor(np.array(Tdot_list), device=device, dtype=dtype, requires_grad=True)
N     = len(times)

print(f'  {N} pas de temps,  t_end = {times[-1]*1e3:.4f} ms')

# ═══════════════════════════════════════════════════════════════════════════
#  Calcul omega via batch torch
# ═══════════════════════════════════════════════════════════════════════════

omega_torch = reaction_system.omega_total(T_all, C_all)

print(torch.linalg.norm(omega_torch-omega_all, dim=0))

domega_dt_no_reactor_list = []
domega_dt_reactor_list = []
domega_dt_reactor_full_list = []



# concat état
def omega_flat(x):
    T_, C_ = x[0], x[1:]

    # reshape en batch de taille 1
    T_batch = T_.unsqueeze(0)  # (1,)
    C_batch = C_.unsqueeze(0)  # (1, n_sp)

    omega = reaction_system.omega_total(T_batch, C_batch)

    return omega.squeeze(0)  # (n_sp,)

def reactor_flat(x):

    xdot = rhs_adiabatic(x.unsqueeze(0), None)

    return xdot.squeeze(0)  # (n_sp,)

for i in range(N):
    # état à un instant donné
    T = T_all[i]
    C = C_all[i]

    # dérivées temporelles
    Tdot = Tdot_all[i]
    Cdot = omega_torch[i]  # ou autre si énergie couplée


    x = torch.cat([T.unsqueeze(0), C])

    xdot = torch.cat([Tdot.unsqueeze(0), Cdot])

    omega, domega_dt = jvp(omega_flat, (x,), (xdot,))
    domega_dt_no_reactor_list.append(domega_dt)

for i in range(N):

    T = T_all[i]
    C = C_all[i]

    # === état complet ===
    TC = torch.cat([T.unsqueeze(0), C]).unsqueeze(0)  # (1, 1+nsp)

    # === dérivée du système réacteur (IMPORTANT) ===
    dTCdt = rhs_adiabatic(TC, None).squeeze(0)

    Tdot = dTCdt[0]
    Cdot = dTCdt[1:]

    # === vecteurs pour JVP ===
    x = torch.cat([T.unsqueeze(0), C])
    xdot = torch.cat([Tdot.unsqueeze(0), Cdot])

    _, domega_dt = jvp(omega_flat, (x,), (xdot,))

    domega_dt_reactor_list.append(domega_dt)

for i in range(N):

    T = T_all[i]
    C = C_all[i]

    # === état complet ===
    TC = torch.cat([T.unsqueeze(0), C]).unsqueeze(0)  # (1, 1+nsp)

    # === dérivée du système réacteur (IMPORTANT) ===
    dTCdt = rhs_adiabatic(TC, None).squeeze(0)

    Tdot = dTCdt[0]
    Cdot = dTCdt[1:]

    # === vecteurs pour JVP ===
    x = torch.cat([T.unsqueeze(0), C])
    xdot = torch.cat([Tdot.unsqueeze(0), Cdot])

    _, domega_dt = jvp(reactor_flat, (x,), (xdot,))

    domega_dt_reactor_full_list.append(domega_dt[1:])

times_np = times.cpu().detach().numpy()
omega_np = omega_all.cpu().detach().numpy()
T_np = T_all.cpu().detach().numpy()

# === FD Cantera (baseline) ===
domega_dt_fd = np.gradient(omega_np, times_np, axis=0)
norm_fd = np.linalg.norm(domega_dt_fd, axis=1) / np.linalg.norm(omega_np, axis=1)

# === Torch results ===
norm_no_reactor = np.linalg.norm(domega_dt_no_reactor_list, axis=1) / np.linalg.norm(omega_np, axis=1)
norm_reactor = np.linalg.norm(domega_dt_reactor_list, axis=1) / np.linalg.norm(omega_np, axis=1)
norm_reactor_full = np.linalg.norm(domega_dt_reactor_full_list, axis=1) / np.linalg.norm(omega_np, axis=1)

# === Figure ===
fig, ax1 = plt.subplots(figsize=(9,5))

# === courbes principales (log) ===
ax1.plot(times_np * 1e3, norm_fd, 'k-', linewidth=2.5, label='Cantera FD (baseline)')
#ax1.plot(times_np * 1e3, norm_no_reactor, '--', linewidth=2, label='JVP (chimie seule)')
ax1.plot(times_np * 1e3, norm_reactor, '-.', linewidth=2, label='JVP (réacteur)')
ax1.plot(times_np * 1e3, norm_reactor_full, ':', linewidth=2.5, label='JVP (rhs complet)')

ax1.set_yscale('log')
ax1.set_xlabel('Temps [ms]')
ax1.set_ylabel(r'$||d\omega/dt|| / ||\omega||$')
ax1.grid(True, which='both', linestyle='--', alpha=0.6)

# === axe secondaire pour température ===
ax2 = ax1.twinx()
ax2.plot(times_np * 1e3, T_np, linewidth=2, alpha=0.75, label='Température', color='red')
ax2.set_ylabel('Température [K]')
ax2.set_yscale('linear')

# === fusion des légendes ===
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Comparaison des dérivées temporelles de ω\navec trajectoire thermique')
plt.tight_layout()
plt.show()