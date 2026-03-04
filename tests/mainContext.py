import numpy as np
import kineticsGPU.thermo as yaml_thermo
import torch
import time

torch.set_grad_enabled(False)

from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.reactionTypesContext import ElementaryReactionBatchContext, ThreeBodyReactionBatchContext, FalloffReactionBatchContext, ReactionSystemContext
from validationReactions import ElementaryReactionVal, ThreeBodyReactionVal, FalloffReactionVal
from kineticsGPU.NumericalIntegrators.BDFTorchContext import BackwardEulerIntegrator

from yaml_parser import yamlParser
import thermo_properties
import cantera as ct

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time
import matplotlib


##############################
########## Analysis ##########
##############################
RR_and_source_terms = False
Integrators = True



###############################
########## Utilities ##########
###############################
custom_rc = {
    # --- Latex ---
    'text.usetex': True,

    # --- Figure ---
    'figure.figsize': (3.5, 2.5),  # en pouces, bon format pour une colonne (~90 mm)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    #'savefig.bbox': 'tight',
    'savefig.format': 'svg',

    # --- Police ---
    #'font.family': 'sans-serif',
    #'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.family': 'serif', # type cmr 10 : police latex
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,

    # --- Axes ---
    'axes.linewidth': 0.8,
    #'axes.spines.top': False,
    #'axes.spines.right': False,
    'axes.grid': False,
    #'axes.labelpad': 4,
    'axes.formatter.use_mathtext': True,    # offset en mathtext (affiche ×10^{...})
    'axes.formatter.useoffset': True,       # garde l'offset, mais on forcera la forme scientifique
    'axes.formatter.limits': (-4, 4),
    'axes.axisbelow': True,  # pour que les grilles/patchs soient en dessous des lignes

    # --- Ticks ---
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,

    # --- Lignes ---
    'lines.linewidth': 0.8,
    'lines.markersize': 2.0,

    # --- Légendes ---
    'legend.frameon': False,
    'legend.handlelength': 1.8,
    #'legend.handletextpad': 0.4,

    # --- Couleurs ---
    'image.cmap': 'coolwarm',
    'axes.prop_cycle': plt.cycler('color', plt.cm.tab10.colors),

}
matplotlib.rcParams.update(custom_rc)

def MAE(A, B):
    err = torch.abs(A - B)
    mae = torch.mean(err)
    mae_std = torch.std(err)
    return(mae, mae_std, err)

def MSE(A, B):
    err = (A - B) ** 2
    mse = torch.mean(err)
    mse_std = torch.std(err)
    return(mse, mse_std, err)

def plot_vs_time(time, err, savingPath, show=False):

    fig, ax = plt.subplots()
    ax.plot(time.cpu().numpy() if hasattr(time, "cpu") else time, err.cpu().numpy() if hasattr(err, "cpu") else err)

    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plots_vs_time(time, errs, names, savingPath, show=False):

    fig, ax = plt.subplots()
    for err, name in zip(errs, names):
        ax.plot(time.cpu().numpy() if hasattr(time, "cpu") else time, err.cpu().numpy() if hasattr(err, "cpu") else err, label=name)
    plt.legend()

    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def scatter_vs_time(time, err, savingPath, show=False):

    fig, ax = plt.subplots()
    ax.scatter(time.cpu().numpy() if hasattr(time, "cpu") else time, err.cpu().numpy() if hasattr(err, "cpu") else err)

    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def colored_scatter_vs_time(time, err, color, savingPath, show=False):

    fig, ax = plt.subplots()
    sc = ax.scatter(time.cpu().numpy() if hasattr(time, "cpu") else time, err.cpu().numpy() if hasattr(err, "cpu") else err, c=color)

    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_hist(errs, names, savingPath, show=False):
    if len(errs) != len(names):
        raise ValueError("`errs` and `names` lists must have the same length.")

    data_to_plot = [e.cpu().numpy().ravel() if hasattr(e, "cpu") else e.ravel()for e in errs]

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.boxplot(data_to_plot,
                labels=names,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black"),
                boxprops=dict(facecolor="#1f77b4", alpha=0.6))

    from matplotlib import ticker as mticker
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9)))

    plt.tight_layout()
    plt.savefig(savingPath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def interpolate_trajectory(times, T, P, X, t_ref):
    T_i = interp1d(times, T, kind='linear', fill_value="extrapolate")(t_ref)
    P_i = interp1d(times, P, kind='linear', fill_value="extrapolate")(t_ref)

    X_i = np.zeros((len(t_ref), X.shape[1]))
    for k in range(X.shape[1]):
        X_i[:, k] = interp1d(times, X[:, k], kind='linear', fill_value="extrapolate")(t_ref)

    return T_i, P_i, X_i

###################################
########## Files reading ##########
###################################

yaml_file = "../kineticsGPU/gri30.yaml"
dtype = torch.float64
device = 'cuda'
y = yaml_thermo.yamlParser(yaml_file)
vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device)
reactions = parseReactions(yaml_file)


gas = ct.Solution(yaml_file)

y = yamlParser(yaml_file)
thgri = thermo_properties.ThermoData()
y.populateThermoObject(thgri)

elem_reacs = []
threebody_reacs = []
falloff_reacs = []

reactions_index_to_type = []

for r in reactions:
    reactions_index_to_type.append(r['type'])
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
    #todo: add a else condition to stop if model not suported

reactions_array = elem_reacs + threebody_reacs + falloff_reacs
r_e, r_tb, r_f, system = build_reactions(vec_th.species, reactions, vec_th, device, dtype)

print()
print(80*'=')
print('General model stats:')
print(f'\t{len(r_e)} elementary reactions, {len(r_tb)} third body reactions, {len(r_f)} falloff reactions')


batched_elem = ElementaryReactionBatchContext(r_e, device=device, dtype=dtype)
batched_tb = ThreeBodyReactionBatchContext(r_tb, device=device, dtype=dtype)
batched_f = FalloffReactionBatchContext(r_f, device=device, dtype=dtype)
reaction_system = ReactionSystemContext([batched_elem, batched_tb, batched_f])



###############################################
########## reaction rates comparison ##########
###############################################
if RR_and_source_terms:
    size = 1000
    TC = np.random.random((size, len(vec_th.species) + 1))
    TC[:, 0] = np.random.uniform(low=500, high=5000, size=size)

    C = torch.tensor(TC[:, 1:], device=device, dtype=dtype)
    T = torch.tensor(TC[:, 0], device=device, dtype=dtype)

    dC = np.zeros((size, len(thgri.speciesNames)))

    RRs_cantera_elem = []
    RRs_cantera_tb = []
    RRs_cantera_f = []

    for it in range(size):
        gas.TP = T[it], 101325.0
        gas.concentrations = C[it].cpu() * 1e-3
        RRs_cantera_elem.append([])
        RRs_cantera_tb.append([])
        RRs_cantera_f.append([])

        for i, reac in enumerate(reactions_array):
            RR_cantera = gas.net_rates_of_progress[i] * 1000
            if reactions_index_to_type[i] == 'elementary':
                RRs_cantera_elem[it].append(RR_cantera)
            elif reactions_index_to_type[i] == 'threebody':
                RRs_cantera_tb[it].append(RR_cantera)
            elif 'falloff' in reactions_index_to_type[i]:
                RRs_cantera_f[it].append(RR_cantera)

        dC[it] = gas.net_production_rates * 1000


    #  elementary:
    ctx = batched_elem.evaluate_context(T, C)
    elem_RRs_GPU_model = ctx.RR

    elem_MAE, elem_MAE_std, elem_MAE_err = MAE(torch.tensor(RRs_cantera_elem, device=device, dtype=dtype), elem_RRs_GPU_model)
    elem_MSE, elem_MSE_std, elem_MSE_err = MSE(torch.tensor(RRs_cantera_elem, device=device, dtype=dtype), elem_RRs_GPU_model)

    #  third body:
    ctx = batched_tb.evaluate_context(T, C)
    tb_RRs_GPU_model = ctx.RR

    tb_MAE, tb_MAE_std, tb_MAE_err = MAE(torch.tensor(RRs_cantera_tb, device=device, dtype=dtype), tb_RRs_GPU_model)
    tb_MSE, tb_MSE_std, tb_MSE_err = MSE(torch.tensor(RRs_cantera_tb, device=device, dtype=dtype), tb_RRs_GPU_model)

    #  falloff:
    ctx = batched_f.evaluate_context(T, C)
    f_RRs_GPU_model = ctx.RR

    f_MAE, f_MAE_std, f_MAE_err = MAE(torch.tensor(RRs_cantera_f, device=device, dtype=dtype), f_RRs_GPU_model)
    f_MSE, f_MSE_std, f_MSE_err = MSE(torch.tensor(RRs_cantera_f, device=device, dtype=dtype), f_RRs_GPU_model)

    # omega:
    reaction_system.evaluate_context(T, C)
    omega_GPU_model = reaction_system.omega_total()

    omega_MAE, omega_MAE_std, omega_MAE_err = MAE(torch.tensor(dC, device=device, dtype=dtype), omega_GPU_model)
    omega_MSE, omega_MSE_std, omega_MSE_err = MSE(torch.tensor(dC, device=device, dtype=dtype), omega_GPU_model)

    plot_hist([elem_MAE_err, tb_MAE_err, f_MAE_err, omega_MAE_err],
              ["elementary\nreactions\nMAE", "third body\nreactions\nMAE", "falloff\nreactions\nMAE", "species\nproduction\nrate\nMAE"],
              'article_figures/RR_MAE.png')
    plot_hist([elem_MSE_err, tb_MSE_err, f_MSE_err, omega_MSE_err],
              ["elementary\nreactions\nMSE", "third body\nreactions\nMSE", "falloff\nreactions\nMSE", "species\nproduction\nrate\nMSE"],
              'article_figures/RR_MSE.png')

    print()
    print(80*'=')
    print('GPU model validation')
    print()
    print('\tElementary reactions Reaction Rate:')
    print(f'\t\tMean absolute error: {elem_MAE:.6e} +- {elem_MAE_std:.6e} mol/m^3')
    print(f'\t\tMean squared error: {elem_MSE:.6e} +- {elem_MSE_std:.6e} mol/m^3')
    print()
    print('\tThird body reactions Reaction Rate:')
    print(f'\t\tMean absolute error: {tb_MAE:.6e} +- {tb_MAE_std:.6e} mol/m^3')
    print(f'\t\tMean squared error: {tb_MSE:.6e} +- {tb_MSE_std:.6e} mol/m^3')
    print()
    print('\tFalloff reactions Reaction Rate:')
    print(f'\t\tMean absolute error: {f_MAE:.6e} +- {f_MAE_std:.6e} mol/m^3')
    print(f'\t\tMean squared error: {f_MSE:.6e} +- {f_MSE_std:.6e} mol/m^3')
    print()
    print('\tCumulative species production rate (omega):')
    print(f'\t\tMean absolute error: {omega_MAE:.6e} +- {omega_MAE_std:.6e} mol/m^3')
    print(f'\t\tMean squared error: {omega_MSE:.6e} +- {omega_MSE_std:.6e} mol/m^3')



############################################
########## Integration comparison ##########
############################################
if Integrators:
    def canter_constant_volume(T0, C0, t_ref):
        gas_tmp = ct.Solution('gri30.yaml')
        gas_tmp.TP = T0.cpu().numpy(), 101325.0
        gas_tmp.concentrations = C0.cpu().numpy()
        reactor = ct.IdealGasReactor(gas_tmp, energy='on')
        sim = ct.ReactorNet([reactor])

        times = []
        T_can = []
        P_can = []
        X_can = []

        while sim.time < t_ref[-1]:
            sim.step()
            times.append(sim.time)
            T_can.append(reactor.T)
            P_can.append(reactor.thermo.P)
            X_can.append(reactor.thermo.X)

        times = np.array(times)
        T_can = np.array(T_can)
        P_can = np.array(P_can)
        X_can = np.array(X_can)
        return interpolate_trajectory(times, T_can, P_can, X_can, t_ref)

    def canter_constant_presure(T0, C0, t_ref):
        gas_tmp = ct.Solution('gri30.yaml')
        gas_tmp.TP = T0.cpu().numpy(), 101325.0
        gas_tmp.concentrations = C0.cpu().numpy()
        reactor = ct.IdealGasConstPressureReactor(gas_tmp, energy='on')
        sim = ct.ReactorNet([reactor])

        times = []
        T_can = []
        P_can = []
        X_can = []

        while sim.time < t_ref[-1]:
            sim.step()
            times.append(sim.time)
            T_can.append(reactor.T)
            P_can.append(reactor.thermo.P)
            X_can.append(reactor.thermo.X)

        times = np.array(times)
        T_can = np.array(T_can)
        P_can = np.array(P_can)
        X_can = np.array(X_can)
        return interpolate_trajectory(times, T_can, P_can, X_can, t_ref)

    def CPU_model_constant_volume(T0, C0, t_ref):
        gas_tmp = ct.Solution('gri30.yaml')
        def rhs_adiabatic(t, TC):
            T = TC[0]
            C = TC[1:]
            gas_tmp.TP = T, 101325.0
            gas_tmp.concentrations = C
            dCdt = gas_tmp.net_production_rates
            # dT
            dTdt = - np.dot(gas_tmp.partial_molar_int_energies, gas_tmp.net_production_rates) / (gas_tmp.density_mass * gas_tmp.cv_mass)

            return np.concatenate(([dTdt], dCdt))

        # === INTÉGRATION ===
        Y0 = np.concatenate(([T0.cpu().numpy()], C0.cpu().numpy()))
        sol_CPU = solve_ivp(rhs_adiabatic,[0, t_ref[-1]], Y0, method='BDF', atol=1e-12, rtol=1e-6) # defalut CVODE values for atol and rtol
        C_CPU = sol_CPU.y[1:, :].T
        X_CPU = C_CPU / np.sum(C_CPU, axis=1, keepdims=True)
        T_CPU = sol_CPU.y[0, :]
        C_tot = np.sum(C_CPU, axis=1)  # mol/m^3
        P_CPU = 8.314_462_618_153_24 * T_CPU * C_tot  # Pa
        return interpolate_trajectory(sol_CPU.t, T_CPU, P_CPU, X_CPU, t_ref)

    def GPU_model_constant_volume(T0, C0, t_ref):
        def rhs_adiabatic(reactSys, t):
            T = reactSys.T
            C = reactSys.C  # mol/m3
            dCdt = reactSys.omega_total()
            n_tot = torch.sum(C, dim=1)  # (N,)
            X = C / n_tot[:, None]
            cv_mol_mix = vec_th.cv_mix_mol(T, X)  # (N,)
            u_species = vec_th.u_species_mol(T)  # (N, nspecies)
            dUdt_molar = torch.sum(u_species * dCdt, dim=1)  # (N,)
            # dT/dt = - dU/dt / (n_tot * Cv)
            dTdt = - dUdt_molar / (n_tot * cv_mol_mix)
            # assemble
            dTCdt = torch.cat((dTdt[:, None], dCdt), dim=1)
            return dTCdt

        def jvp_rhs_adiabatic(reactSys, t, dTC):
            T = reactSys.T
            C = reactSys.C
            v_T = dTC[:, 0]
            v_C = dTC[:, 1:]
            dCdt = reactSys.omega_total()
            dCdt_jvp = reactSys.jvp(v_T, v_C)
            n_tot = torch.sum(C, dim=1)
            dn_tot = torch.sum(v_C, dim=1)
            X = C / n_tot[:, None]
            dX = (v_C / n_tot[:, None]) - (C * dn_tot[:, None] / (n_tot ** 2)[:, None])
            u_species = vec_th.u_species_mol(T)  # (N, ns)
            cv_species = vec_th.cv_species_mol(T, torch.arange(0, len(vec_th.species)))  # (N, ns)
            dcv_dT = vec_th.dcv_species_dT(T, torch.arange(0, len(vec_th.species)))  # (N, ns)
            cv_mix = vec_th.cv_mix_mol(T, X)
            dcv_mix = torch.sum(dX * cv_species, dim=1) + torch.sum(X * dcv_dT, dim=1) * v_T
            U_prime = torch.sum(u_species * dCdt, dim=1)
            dU_prime = torch.sum(u_species * dCdt_jvp, dim=1) + torch.sum(dCdt * cv_species, dim=1) * v_T
            dT_prime = (- dU_prime / (n_tot * cv_mix) + U_prime * dn_tot / (n_tot ** 2 * cv_mix) + U_prime * dcv_mix / (n_tot * cv_mix ** 2))
            return torch.cat([dT_prime[:, None], dCdt_jvp], dim=1)

        t_eval_torch = torch.tensor(t_ref, device=device, dtype=dtype)
        Y0 = torch.cat((T0[:, None], C0), dim=1)  # shape (batch_size, state) => odeint will accept
        integrator = BackwardEulerIntegrator(reaction_system, rhs_adiabatic, jvp_rhs_adiabatic, device=device, dtype=dtype, verbose=False, use_autograd_jvp=False, gmres_restart=20)

        res = []
        for i, t in enumerate(t_eval_torch):
            Y0 = integrator.step(Y0, t, float(t_eval_torch[1] - t_eval_torch[0]), method='gmres',precond='identity')  # TODO: investigate memory issue with chemistry preconditioning
            res.append(Y0)
            print(Y0[0, 0])

        # convert solution to numpy for comparisons & plotting
        res = torch.stack(res, dim=0)
        # res.shape = (n_points, nbatches, 1 + n_species)
        T = res[:, :, 0].transpose(0, 1)  # (nbatches, n_points)
        C = res[:, :, 1:].permute(1, 0, 2)  # (nbatches, n_points, n_species)
        C_tot = torch.sum(C, dim=2)  # mol/m^3
        X = C / C_tot.unsqueeze(-1)
        R = 8.314_462_618_153_24
        P = R * T * C_tot  # Pa

        return T.cpu().numpy(), P.cpu().numpy(), X.cpu().numpy()

    def run_sequential_sim(model, T, C):
        res = {'T': [], 'P': [], 'X': []}
        for T0, C0 in zip(T, C):
            T, P, X = model(T0, C0, t_ref)
            res['T'].append(T)
            res['P'].append(P)
            res['X'].append(X)
        res['T'], res['P'], res['X'] = np.stack(res['T']), np.stack(res['P']), np.stack(res['X'])
        return(res)

    def run_batched_sim(model, T, C):
        res = {}
        T, P, X = model(T, C, t_ref)
        res['T'] = T
        res['P'] = P
        res['X'] = X
        return(res)

    t_end = 3e-4
    t_ref = np.arange(0.0, t_end, 1e-7)


    size = 1000

    T0s = np.random.uniform(low=1500, high=2000, size=size-1)
    C0s = []
    phis = np.random.uniform(low=0.5, high=1.5, size=size-1)
    T0_ref = 1700.0  # the conditions for the plot, other conditions are random
    phi_ref = 1.0  # the conditions for the plot, other conditions are random

    for phi, T0 in zip(phis, T0s):

        # Mélange CH4-air stœchiométrique
        gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
        gas.TP = T0, 101325.0

        # concentrations initiales (kmol/m3)
        C0 = gas.concentrations
        C0s.append(C0)

    T = torch.tensor(np.hstack([T0_ref, T0s]), device=device, dtype=dtype)
    gas.set_equivalence_ratio(phi_ref, "CH4", "O2:1.0, N2:3.76")
    C = torch.tensor(np.vstack([gas.concentrations, C0s]), device=device, dtype=dtype)

    print()
    print(80*'=')
    print('Numerical integration validation')


    print()
    print('\tCantera model:')
    t0 = time.perf_counter()
    cant_res = run_sequential_sim(canter_constant_volume, T, C)
    t_comp = time.perf_counter() - t0
    res_T, res_P, res_X = cant_res['T'], cant_res['P'], cant_res['X']
    print(f'\t\tcomputation time: {t_comp:.3f} seconds')
    print(f'\t\tobject shapes: T.shape = {res_T.shape}, P.shape = {res_P.shape}, X.shape = {res_X.shape}')

    print()
    print('\tCPU model:')

    t0 = time.perf_counter()
    CPU_res = run_sequential_sim(CPU_model_constant_volume, T, C)
    t_comp = time.perf_counter() - t0
    res_T, res_P, res_X = CPU_res['T'], CPU_res['P'], CPU_res['X']
    print(f'\t\tcomputation time: {t_comp:.3f} seconds')
    print(f'\t\tobject shapes: T.shape = {res_T.shape}, P.shape = {res_P.shape}, X.shape = {res_X.shape}')


    print()
    print('\tGPU model:')
    t0 = time.perf_counter()
    GPU_res = run_batched_sim(GPU_model_constant_volume, T, C * 1e3)
    t_comp = time.perf_counter() - t0
    res_T, res_P, res_X = GPU_res['T'], GPU_res['P'], GPU_res['X']
    print(f'\t\tcomputation time: {t_comp:.3f} seconds')
    print(f'\t\tobject shapes: T.shape = {res_T.shape}, P.shape = {res_P.shape}, X.shape = {res_X.shape}')


    # Comparing only first batch
    CPU_MAE, CPU_MAE_std, CPU_MAE_err = MAE(torch.tensor(cant_res['T'][0], device=device, dtype=dtype), torch.tensor(CPU_res['T'][0], device=device, dtype=dtype))
    CPU_MSE, CPU_MSE_std, CPU_MSE_err = MSE(torch.tensor(cant_res['T'][0], device=device, dtype=dtype), torch.tensor(CPU_res['T'][0], device=device, dtype=dtype))

    GPU_MAE, GPU_MAE_std, GPU_MAE_err = MAE(torch.tensor(cant_res['T'][0], device=device, dtype=dtype), torch.tensor(GPU_res['T'][0], device=device, dtype=dtype))
    GPU_MSE, GPU_MSE_std, GPU_MSE_err = MSE(torch.tensor(cant_res['T'][0], device=device, dtype=dtype), torch.tensor(GPU_res['T'][0], device=device, dtype=dtype))

    plots_vs_time(t_ref, [cant_res['T'][0], CPU_res['T'][0], GPU_res['T'][0]], ['Cantera', 'CPU model', 'GPU model'], 'article_figures/T_trajectory.png')

    plot_vs_time(t_ref, CPU_MAE_err, 'article_figures/CPU_MAE.png')
    plot_vs_time(t_ref, CPU_MSE_err, 'article_figures/CPU_MSE.png')
    scatter_vs_time(np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0]), CPU_MSE_err.cpu().numpy(), 'article_figures/CPU_MSE_vs_dT.png')
    scatter_vs_time(np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0]), CPU_MAE_err.cpu().numpy(), 'article_figures/CPU_MAE_vs_dT.png')
    colored_scatter_vs_time(t_ref, CPU_MSE_err.cpu().numpy(), np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0]), 'article_figures/colored_CPU_MSE_vs_time.png')
    colored_scatter_vs_time(t_ref, CPU_MAE_err.cpu().numpy(), np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0]), 'article_figures/colored_CPU_MAE_vs_time.png')



    plot_vs_time(t_ref, GPU_MAE_err, 'article_figures/GPU_MAE.png')
    plot_vs_time(t_ref, GPU_MSE_err, 'article_figures/GPU_MSE.png')
    scatter_vs_time(np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0]), GPU_MSE_err.cpu().numpy(), 'article_figures/GPU_MSE_vs_dT.png')
    scatter_vs_time(np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0]), GPU_MAE_err.cpu().numpy(), 'article_figures/GPU_MAE_vs_dT.png')
    colored_scatter_vs_time(t_ref, GPU_MSE_err.cpu().numpy(), np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0]), 'article_figures/colored_GPU_MSE_vs_time.png')
    colored_scatter_vs_time(t_ref, GPU_MAE_err.cpu().numpy(), np.diff(cant_res['T'][0], prepend=cant_res['T'][0][0]), 'article_figures/colored_GPU_MAE_vs_time.png')


1/0














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
        plt.plot(t_eval_torch.cpu(), X_torch[:, i], '--', label=f"{sp} (model)")
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
plt.plot(t_eval_torch.cpu(), T_torch, '--', label="My model")
plt.plot(times, T_can, '-', label="Cantera")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.title("Adiabatic Temperature Evolution (CH4/Air)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()