import numpy as np
import kineticsGPU.thermo as yaml_thermo
import torch

torch.set_grad_enabled(True)

from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.reactionTypes import ElementaryReactionBatch, ThreeBodyReactionBatch, FalloffReactionBatch, ReactionSystem
from validationReactions import ElementaryReactionVal, ThreeBodyReactionVal, FalloffReactionVal

yaml_file = "gri30.yaml"
dtype = torch.float64
device = 'cuda'
y = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device)
reactions = parseReactions(yaml_file)

from yaml_parser import yamlParser
import thermo_properties

y = yamlParser(yaml_file)
thgri = thermo_properties.ThermoData()
y.populateThermoObject(thgri)

elem_reacs = []
threebody_reacs = []
falloff_reacs = []


for r in reactions:
    if r['type'] == 'elementary':
        elem_reacs.append(ElementaryReactionVal(
            r['reactants'], r['products'], thgri, thgri.speciesNames,
            not r['reversible'], r['A_SI'], r['b'], r['Ea_SI']
        ))
    elif r['type'] == 'threebody':
        threebody_reacs.append(ThreeBodyReactionVal(
            r['reactants'], r['products'], thgri, thgri.speciesNames,
            not r['reversible'], r['A_SI'], r['b'], r['Ea_SI'],
            r['efficiencies'], defaultEfficiencies=r['defaultEff']
        ))
    elif 'falloff' in r['type']:
        falloff_reacs.append(FalloffReactionVal(
            r['reactants'], r['products'], thgri, thgri.speciesNames,
            not r['reversible'],
            (r['k0']['A_SI'], r['k0']['b'], r['k0']['Ea_SI']),
            (r['kinf']['A_SI'], r['kinf']['b'], r['kinf']['Ea_SI']),
            r['efficiencies'],
            falloff_type='troe' if 'troe' in r['type'] else 'lindemann',
            falloff_params=r.get('Troe_params', None)
        ))

r_e, r_tb, r_f, system = build_reactions(vec_th.species, reactions, vec_th, device, dtype)

batched_elem = ElementaryReactionBatch(r_e, device=device, dtype=dtype)
batched_tb = ThreeBodyReactionBatch(r_tb, device=device, dtype=dtype)
batched_f = FalloffReactionBatch(r_f, device=device, dtype=dtype)
reaction_system = ReactionSystem([batched_elem, batched_tb, batched_f])


TC = np.random.random((1000, len(vec_th.species) + 1))
TC[:, 0] = 2000.0

C = torch.tensor(TC[:, 1:], device=device, dtype=dtype)
T = torch.tensor(TC[:, 0], device=device, dtype=dtype)
"""
for n in range(len(r_e)):
    vec_RR = r_e[n].RR(T, C)
    for i in range(len(TC)):
        diff = abs(vec_RR[i] - elem_reacs[n].RR(TC[i]))
        isclose = math.isclose(vec_RR[i].item(), elem_reacs[n].RR(TC[i]), abs_tol=1e-12, rel_tol=1e-12)
        if not isclose:
            print('elem', vec_RR[i].item(), elem_reacs[n].RR(TC[i]))

for n in range(len(r_tb)):
    vec_RR = r_tb[n].RR(T, C)
    for i in range(len(TC)):
        diff = abs(vec_RR[i] - threebody_reacs[n].RR(TC[i]))
        isclose = math.isclose(vec_RR[i].item(), threebody_reacs[n].RR(TC[i]), abs_tol=1e-12, rel_tol=1e-12)
        if not isclose:
            print('tb', vec_RR[i].item(), threebody_reacs[n].RR(TC[i]))

for n in range(len(r_f)):
    vec_RR = r_f[n].RR(T, C)
    for i in range(len(TC)):
        diff = abs(vec_RR[i] - falloff_reacs[n].RR(TC[i]))
        isclose = math.isclose(vec_RR[i].item(), falloff_reacs[n].RR(TC[i]), abs_tol=1e-12, rel_tol=1e-12)
        if not isclose:
            print('falloff', vec_RR[i].item(), falloff_reacs[n].RR(TC[i]))
"""

import time
list = r_e

dCdt = torch.zeros((T.shape[0], len(list)), device=device, dtype=dtype)
t0 = time.perf_counter()
for i, r in enumerate(list):
    dCdt[:, i] = r.RR(T, C)
    """
    for spe, coef in zip(r.reactants_idx, r.reactants_sto):
        dCdt[:, spe] -= coef * RR
    for spe, coef in zip(r.products_idx, r.products_sto):
        dCdt[:, spe] += coef * RR
    """
print(time.perf_counter() - t0)
t0 = time.perf_counter()
omega = batched_elem.RR(T, C)
print(time.perf_counter() - t0)

print(omega,'\n', dCdt)

"""
t0 = time.perf_counter()
for tc in TC:
    for rval in elem_reacs+threebody_reacs+falloff_reacs:
        rval.RR(tc)
print(time.perf_counter() - t0)
"""

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import cantera as ct
import time

# ==============================================================
# Hypothèses :
#   - thgri : ton objet thermo
#   - reactions_array : liste de tes réactions
#   - thgri.speciesNames == gas.species_names
# ==============================================================
gas = ct.Solution(yaml_file)

# === CONDITIONS INITIALES ===
P0 = ct.one_atm
T0 = 1700.0  # K, par exemple une flamme chaude
phi = 1.0  # mélange stœchiométrique

# Mélange CH4-air stœchiométrique
gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
gas.TP = T0, P0

# gas.TPX = T0, P0, {'O2': 0.11, 'O': 0.1, 'N2': 0.78, 'AR': 0.01}

# concentrations initiales (mol/m3)
C0 = gas.concentrations * 1e3  # Cantera -> kmol/m3 → mol/m3

# === CONSTANTES PHYSIQUES ===
R_univ = 8.31446261815324  # J/mol/K


# === RHS not vectorized ===
def rhs_adiabatic(t, TC):
    T = TC[0]
    C = TC[1:]

    # dCdt = system.dCdt(TC)

    # cinétique chimique (ton modèle)
    dCdt = np.zeros_like(C)
    for reac in elem_reacs+threebody_reacs+falloff_reacs:
        RR = reac.RR(TC)  # mol/m3/s
        for spe, coef in reac.reactants.items():
            dCdt[thgri.speciesNames.index(spe)] -= coef * RR
        for spe, coef in reac.products.items():
            dCdt[thgri.speciesNames.index(spe)] += coef * RR

    # propriétés thermodynamiques (depuis Cantera)
    gas.TP = T, P0
    gas.concentrations = C * 1e-3
    rho = gas.density_mass  # kg/m3
    cv_mass = gas.cv_mass  # J/kg/K
    cv_mol = cv_mass * rho / np.sum(C)  # J/mol/K

    # dérivée de T
    dHdt = 0.0
    for i, sp in enumerate(thgri.speciesNames):
        hi = thgri.h(sp, T, P0)  # J/mol
        dHdt += hi * dCdt[i]
    dTdt = -dHdt / (np.sum(C) * cv_mol)

    return np.concatenate(([dTdt], dCdt))


# === INTÉGRATION ===
t_end = 3e-4
t_eval = np.linspace(0, t_end, 2)
Y0 = np.concatenate(([T0], C0))
t0 = time.perf_counter()
sol_my = solve_ivp(rhs_adiabatic,
                   [0, t_end],
                   Y0,
                   # t_eval=t_eval,
                   method='BDF',
                   # atol=1e-12, rtol=1e-10
                   )
print(time.perf_counter() - t0)


# === RHS vectorized ===
def rhs_adiabatic(TC, t):
    T = TC[:, 0]
    C = TC[:, 1:]

    # cinétique chimique (ton modèle)
    dCdt = reaction_system.omega_total(T, C)

    # total molar concentration (mol/m3)
    n_tot = torch.sum(C, dim=1)  # (N,)
    # mole fractions X = C / n_tot
    X = C / n_tot[:, None]  # (N, nspecies)
    # mixture molar Cv (J/mol/K)
    cv_mol_mix = vec_th.cv_mix_mol(T, X)  # (N,)
    # dH_molar/dt = Σ_i h_i(T) * dC_i/dt  (J/m3/s)
    dHdt_molar = vec_th.h_mix_mol(T, dCdt)  # (N,)
    # d(n_tot)/dt = Σ_i dC_i/dt
    #dn_dt = torch.sum(dCdt, dim=1)  # (N,)

    # final dT/dt (K/s) — full expression including R*T*dn/dt term
    #dTdt = - (dHdt_molar - R * T * dn_dt) / (n_tot * cv_mol_mix)
    dTdt = - (dHdt_molar) / (n_tot * cv_mol_mix)

    # assemble result
    dTCdt = torch.cat((dTdt[:, None], dCdt), dim=1)

    return dTCdt


# === INTÉGRATION ===
from kineticsGPU.NumericalIntegrators.BDFTorch import BackwardEulerIntegrator
from kineticsGPU.NumericalIntegrators.RKF45 import AdaptiveExplicitRKF45
from kineticsGPU.NumericalIntegrators.BDFAdaTsTorch import AdaptiveImplicitIntegrator
dt = 1e-6
t_eval_torch = torch.arange(0, t_end, dt, device=device, dtype=dtype)
print(len(t_eval_torch))
T0_t = torch.tensor([T0], device=device, dtype=dtype).unsqueeze(0)      # shape (1,1)
C0_t = torch.tensor(C0, device=device, dtype=dtype).unsqueeze(0)       # shape (1, n_species)
Y0 = torch.cat((T0_t, C0_t), dim=1).repeat(1000, 1)                     # shape (batch_size, state) => odeint will accept
integrator = BackwardEulerIntegrator(rhs_adiabatic, device=device, dtype=dtype, verbose=True, use_autograd_jvp=True)
integratorrkf45 = AdaptiveExplicitRKF45(rhs_adiabatic, device=device, dtype=dtype, verbose=False)
integratorAda = AdaptiveImplicitIntegrator(rhs_adiabatic, device=device, dtype=dtype, verbose=True)
t0 = time.perf_counter()

for i, t in enumerate(t_eval_torch):
    print(Y0)
    Y0 = integrator.step(Y0, t, dt, method='gmres', precond='identity') # TODO: investigate memory issue with chemistry preconditioning
    #Y0 = integratorAda.step_adaptive(Y0, t, dt, method='trapezoid', linear_solver='gmres', precond='identity')

#t = 0.0
#h = dt
#while t < t_eval_torch[-1].item():
    #Y0, h = integratorrkf45.step_adaptive(Y0, t, dt, atol=1e-8, rtol=1e-8, safety=0.9, controller='PI', verbose=False)
    #Y0, h = integratorAda.step_adaptive(Y0, t, h, method='backward_euler', linear_solver='gmres', precond='identity', jacobian_reuse=1)
    #print('\n\n\n\n')
    #print(Y0)
    #print('\n\n\n\n')
    #print(h)
    #print('\n\n\n\n')
    #t+=h
    #print(t)

#sol_torch = odeint(rhs_adiabatic, Y0, t_eval_torch, method='trbdf2')  # shape (n_points, state)
print(time.perf_counter() - t0)
# convert solution to numpy for comparisons & plotting
sol_torch_cpu = Y0.detach().cpu().numpy()   # shape (n_points, nbatches, 1 + n_species)
T_torch = sol_torch_cpu[:, 0, 0].flatten()
C_torch = sol_torch_cpu[:, 0, 1:]  # mol/m3
X_torch = C_torch / np.sum(C_torch, axis=1, keepdims=True)


# === INTÉGRATION CANTERA (adiabatique) ===
gas.TP = T0, P0
gas.concentrations = C0 * 1e-3
reactor = ct.IdealGasReactor(gas, energy='on')
sim = ct.ReactorNet([reactor])

times = []
T_can = []
X_can = []

while sim.time < t_end:
    sim.step()
    times.append(sim.time)
    T_can.append(reactor.T)
    X_can.append(reactor.thermo.X)

times = np.array(times)
T_can = np.array(T_can)
X_can = np.array(X_can)

# === Post-traitement ===
C_my = sol_my.y[1:, :].T
X_my = C_my / np.sum(C_my, axis=1, keepdims=True)
T_my = sol_my.y[0, :]

# ==============================================================
# === GRAPHIQUES ===
# ==============================================================

species_to_plot = ["CH4", "O2", "CO", "CO2", "H2O", "H", "OH"]

plt.figure(figsize=(10, 6))
for sp in species_to_plot:
    if sp in thgri.speciesNames:
        i = thgri.speciesNames.index(sp)
        plt.plot(sol_my.t, X_my[:, i], '--', label=f"{sp} (model)")
        plt.plot(times, X_can[:, i], '-', label=f"{sp} (Cantera)")
plt.xlabel("Time [s]")
plt.ylabel("Mole fraction X_i")
plt.legend()
plt.title("CH4/Air Adiabatic Combustion: My model vs Cantera")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Température ===
plt.figure(figsize=(8, 5))
plt.plot(sol_my.t, T_my, '--', label="My model")
plt.plot(times, T_can, '-', label="Cantera")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.title("Adiabatic Temperature Evolution (CH4/Air)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





plt.figure(figsize=(10, 6))
for sp in species_to_plot:
    if sp in thgri.speciesNames:
        i = thgri.speciesNames.index(sp)
        plt.plot(t_eval, X_torch[:, i], '--', label=f"{sp} (model)")
        plt.plot(times, X_can[:, i], '-', label=f"{sp} (Cantera)")
plt.xlabel("Time [s]")
plt.ylabel("Mole fraction X_i")
plt.legend()
plt.title("CH4/Air Adiabatic Combustion: My model vs Cantera")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Température ===
plt.figure(figsize=(8, 5))
plt.plot(t_eval, T_torch, '--', label="My model")
plt.plot(times, T_can, '-', label="Cantera")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.title("Adiabatic Temperature Evolution (CH4/Air)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.sh