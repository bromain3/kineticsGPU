"""
reaction_derivatives.py
=======================
Patch pour ReactionSystemContext :
  - domega_dt(rhs_TC)   : dω/dt analytique (= JVP évalué sur le RHS)
  - d2TC_dt2(rhs_TC)    : d²[T,C]/dt² du réacteur adiabatique CV

Script de validation contre Cantera inclus en bas (__main__).

Principe mathématique
---------------------
Soit le réacteur à volume constant adiabatique :

    d/dt [T, C]  =  f(T, C)   avec f = rhs_adiabatic

Alors :
    d²/dt² [T, C]  =  df/dt  =  (∂f/∂T) Ṫ + (∂f/∂C) Ċ
                    =  Jf · [Ṫ, Ċ]
                    =  jvp_rhs_adiabatic(ctx, rhs_TC)

Et pour les seuls taux de production d'espèces :
    dω/dt  =  (∂ω/∂T) Ṫ + (∂ω/∂C) Ċ
           =  Jω · [Ṫ, Ċ]
           =  reactionSystem.jvp(v_T=Ṫ, v_C=Ċ)

→ Les deux expressions réutilisent le JVP déjà implémenté,
  sans nouvelle mécanique de dérivation.
"""

import torch
import time

# ──────────────────────────────────────────────────────────────
# 1.  Patch ReactionSystemContext
# ──────────────────────────────────────────────────────────────

def _domega_dt(self, v_T, v_C) -> torch.Tensor:
    """
    Dérivée temporelle analytique de ω (taux de production d'espèces).

    Paramètres
    ----------
    rhs_TC : (ncells, 1+n_species)
        Résultat de rhs_adiabatic : la colonne 0 est dT/dt,
        les colonnes 1: sont dC/dt.

    Retour
    ------
    domega_dt : (ncells, n_species)
        dω/dt = (∂ω/∂T)·Ṫ + (∂ω/∂C)·Ċ

    Notes
    -----
    Il faut avoir appelé evaluate_context(T, C) **avant**.
    Le contexte doit être cohérent avec l'état (T, C) courant.
    """
    #v_T = rhs_TC[:, 0]       # (ncells,)  tangent sur T
    #v_C = rhs_TC[:, 1:]      # (ncells, n_species)  tangent sur C
    return self.jvp(v_T, v_C)


def _d2TC_dt2(self, vec_th, v_T, v_C) -> torch.Tensor:
    """
    Dérivée temporelle seconde analytique de [T, C] pour le réacteur
    adiabatique à volume constant.

    d²[T,C]/dt²  =  d/dt [ rhs_adiabatic(T, C) ]
                 =  jvp_rhs_adiabatic( ctx, rhs_TC )

    Paramètres
    ----------
    vec_th  : ThermoData_vectorized
    rhs_TC  : (ncells, 1+n_species)   = rhs_adiabatic déjà calculé

    Retour
    ------
    d2TC : (ncells, 1+n_species)
    """
    rhs_TC = torch.stack([v_T, v_C])
    return jvp_rhs_adiabatic(self, vec_th, rhs_TC)


# Injection dans la classe (monkey-patch)
from kineticsGPU.reactionTypesContext import ReactionSystemContext, ElementaryReactionBatchContext, ThreeBodyReactionBatchContext, FalloffReactionBatchContext   # adapter l'import si besoin
ReactionSystemContext.domega_dt = _domega_dt
ReactionSystemContext.d2TC_dt2  = _d2TC_dt2


# ──────────────────────────────────────────────────────────────
# 2.  RHS et JVP du réacteur adiabatique CV  (repris de l'énoncé)
# ──────────────────────────────────────────────────────────────

def rhs_adiabatic(reactSys, vec_th) -> torch.Tensor:
    """
    Retourne d[T,C]/dt pour le réacteur adiabatique CV.

    Shape retour : (ncells, 1+n_species)
    """
    T   = reactSys.T
    C   = reactSys.C
    dCdt = reactSys.omega_total()                          # (N, ns)

    n_tot = torch.sum(C, dim=1)                            # (N,)
    X     = C / n_tot[:, None]
    cv_mol_mix = vec_th.cv_mix_mol(T, X)                   # (N,)
    u_species  = vec_th.u_species_mol(T)                   # (N, ns)
    dUdt_molar = torch.sum(u_species * dCdt, dim=1)        # (N,)
    dTdt = -dUdt_molar / (n_tot * cv_mol_mix)

    return torch.cat([dTdt[:, None], dCdt], dim=1)         # (N, 1+ns)


def jvp_rhs_adiabatic(reactSys, vec_th, dTC: torch.Tensor) -> torch.Tensor:
    """
    Produit Jacobien-vecteur du RHS adiabatique CV.

    d²[T,C]/dt²  =  (∂f/∂[T,C]) · [Ṫ, Ċ]

    Paramètres
    ----------
    dTC : (ncells, 1+n_species)   vecteur tangent = rhs_adiabatic(...)

    Retour
    ------
    d2TC : (ncells, 1+n_species)
    """
    T    = reactSys.T
    C    = reactSys.C
    v_T  = dTC[:, 0]
    v_C  = dTC[:, 1:]

    # ── ω et dω (depuis le contexte déjà évalué) ─────────────────────
    dCdt     = reactSys.omega_total()                      # (N, ns)
    dCdt_jvp = reactSys.jvp(v_T, v_C)                     # (N, ns)

    # ── scalaires thermodynamiques ────────────────────────────────────
    n_tot = torch.sum(C, dim=1)                            # (N,)
    dn_tot = torch.sum(v_C, dim=1)                         # (N,)

    X  = C / n_tot[:, None]
    dX = (v_C / n_tot[:, None]) - (C * dn_tot[:, None] / (n_tot**2)[:, None])

    idx_all    = torch.arange(len(vec_th.species), device=T.device)
    u_species  = vec_th.u_species_mol(T)                   # (N, ns)
    cv_species = vec_th.cv_species_mol(T, idx_all)         # (N, ns)
    dcv_dT     = vec_th.dcv_species_dT(T, idx_all)         # (N, ns)

    cv_mix = vec_th.cv_mix_mol(T, X)                       # (N,)

    # ── dérivée de cv_mix par rapport aux perturbations ──────────────
    # dcv_mix/dt = ∑_s (dX_s/dt · cv_s) + ∑_s (X_s · dcv_s/dT · Ṫ)
    dcv_mix = (torch.sum(dX * cv_species, dim=1)
               + torch.sum(X  * dcv_dT,   dim=1) * v_T)   # (N,)

    # ── U' = ∑_s u_s · ω_s  et sa dérivée ───────────────────────────
    U_prime  = torch.sum(u_species * dCdt, dim=1)          # (N,)

    # dU'/dt = ∑_s u_s · dω_s/dt  +  ∑_s ω_s · cv_s · Ṫ
    #        (car du_s/dT = cv_s pour un gaz parfait)
    dU_prime = (torch.sum(u_species  * dCdt_jvp, dim=1)
                + torch.sum(dCdt * cv_species, dim=1) * v_T)  # (N,)

    # ── dérivée de Ṫ = -U' / (n_tot · cv_mix) ───────────────────────
    # Règle du quotient :
    # d/dt[-U'/(n·cv)] = (-dU'·n·cv + U'·dn·cv + U'·n·dcv) / (n·cv)²
    denom    = n_tot * cv_mix
    dT_prime = (
        -dU_prime / denom
        + U_prime * dn_tot / (denom * n_tot)
        + U_prime * dcv_mix / (denom * cv_mix)
    )

    return torch.cat([dT_prime[:, None], dCdt_jvp], dim=1)  # (N, 1+ns)


# ──────────────────────────────────────────────────────────────
# 3.  Intégrateur Euler simple (pour générer une trajectoire)
# ──────────────────────────────────────────────────────────────

def euler_integrate(reactSys, vec_th, T0, C0, dt, n_steps):
    """
    Intègre le réacteur adiabatique CV avec Euler explicite.
    Retourne l'historique : list de (T, C, rhs, d2TC).
    """
    T = T0.clone()
    C = C0.clone()
    history = []

    for step in range(n_steps):
        reactSys.evaluate_context(T, C)
        rhs = rhs_adiabatic(reactSys, vec_th)

        # dérivées seconde analytiques
        d2TC = jvp_rhs_adiabatic(reactSys, vec_th, rhs)

        history.append({
            'T':    T.clone(),
            'C':    C.clone(),
            'rhs':  rhs.clone(),
            'd2TC': d2TC.clone(),
        })

        # avance d'un pas
        T = T + dt * rhs[:, 0]
        C = C + dt * rhs[:, 1:]
        C = torch.clamp(C, min=0.0)

    return history




if __name__ == '__main__':
    import numpy as np
    import cantera as ct
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import kineticsGPU.thermo as yaml_thermo
    from kineticsGPU.reactionParser import parseReactions, build_reactions
    from kineticsGPU.reactionTypesContext import (
        ReactionSystemContext,
        ElementaryReactionBatchContext,
        ThreeBodyReactionBatchContext,
        FalloffReactionBatchContext,
    )

    # ── config ────────────────────────────────────────────────
    yaml_file = "../kineticsGPU/gri30.yaml"
    dtype     = torch.float64
    device    = 'cpu'
    T_init_K  = 1700.0
    P_init_Pa = 101325.0
    phi       = 1.0
    fuel      = "CH4"
    oxidizer  = "O2:1,N2:3.76"

    # ── construction du modèle ────────────────────────────────
    y       = yaml_thermo.yamlParser(yaml_file)
    vec_th  = y.populateVectorizedThermoObject(dtype=dtype, device=device)
    species = vec_th.species
    n_species = len(species)

    reactions = parseReactions(yaml_file)
    r_e, r_tb, r_f, _ = build_reactions(
        vec_th.species, reactions, vec_th, device, dtype
    )
    batched_elem = ElementaryReactionBatchContext(r_e, device=device, dtype=dtype)
    batched_tb   = ThreeBodyReactionBatchContext(r_tb, device=device, dtype=dtype)
    batched_f    = FalloffReactionBatchContext(r_f, device=device, dtype=dtype)
    reactSys     = ReactionSystemContext([batched_elem, batched_tb, batched_f])

    print(f"Mécanisme : {yaml_file}  |  {n_species} espèces")

    # ── conditions initiales ──────────────────────────────────
    gas = ct.Solution(yaml_file)
    gas.TP = T_init_K, P_init_Pa
    gas.set_equivalence_ratio(phi, fuel, oxidizer)

    ct_idx    = {s: i for i, s in enumerate(gas.species_names)}
    our_to_ct = [ct_idx[s] for s in species]

    # ── trajectoire Cantera (états seulement) ─────────────────
    print("Simulation Cantera...")
    r_ct   = ct.IdealGasReactor(gas, energy='on')
    sim_ct = ct.ReactorNet([r_ct])

    dt_ct   = 1e-7
    n_steps = 3200

    times_list = []
    T_list     = []
    C_list     = []

    for i in range(n_steps):
        t_now = i * dt_ct
        times_list.append(t_now)
        T_list.append(r_ct.thermo.T)
        C_list.append(r_ct.thermo.concentrations[our_to_ct].copy())
        sim_ct.advance(t_now + dt_ct)

    times  = np.array(times_list)   # (N,)
    T_traj = np.array(T_list)       # (N,)
    C_traj = np.array(C_list)       # (N, ns)
    print(f"Trajectoire Cantera : {n_steps} pas, dt={dt_ct:.1e} s")

    # ── batch analytique sur toute la trajectoire ─────────────
    # evaluate_context sur tous les points d'un coup
    T_batch = torch.tensor(T_traj, dtype=dtype, device=device)   # (N,)
    C_batch = torch.tensor(C_traj, dtype=dtype, device=device)   # (N, ns)

    reactSys.evaluate_context(T_batch, C_batch)

    # RHS complet : dC/dt = ω  et  dT/dt depuis le modèle
    rhs = rhs_adiabatic(reactSys, vec_th)   # (N, 1+ns)
    v_T = rhs[:, 0]                          # (N,)   tangent T
    v_C = rhs[:, 1:]                         # (N, ns) tangent C = ω

    # ω calculé par ton modèle sur la trajectoire Cantera
    omega_model = v_C.cpu().numpy()          # (N, ns)

    # dω/dt analytique via JVP
    domega_dt_an = reactSys.jvp(v_T, v_C).cpu().numpy()   # (N, ns)

    # ── référence FD : gradient de ω_model (pas ω Cantera !) ──
    # On dérive ω tel que ton modèle le voit, pas Cantera
    # → cohérence garantie : même fonction, même états
    domega_dt_fd = np.gradient(omega_model, dt_ct, axis=0)  # (N, ns)

    print("Calcul analytique terminé.")

    # ── normes ||dω/dt|| / ||ω|| ──────────────────────────────
    norm_omega     = np.linalg.norm(omega_model,   axis=1)  # (N,)
    norm_domega_fd = np.linalg.norm(domega_dt_fd,  axis=1)  # (N,)
    norm_domega_an = np.linalg.norm(domega_dt_an,  axis=1)  # (N,)

    eps      = np.max(norm_omega) * 1e-10
    ratio_fd = norm_domega_fd / (norm_omega + eps)
    ratio_an = norm_domega_an / (norm_omega + eps)

    # ── sélection top espèces ─────────────────────────────────
    peak_omega = np.abs(omega_model).max(axis=0)
    top_idx    = np.argsort(peak_omega)[::-1][:6]
    top_names  = [species[i] for i in top_idx]
    colors     = plt.cm.tab10(np.linspace(0, 1, 6))

    # ── tracé ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Validation dω/dt | GRI30 | T₀={T_init_K} K, φ={phi}",
        fontsize=13
    )

    # (0,0) : T(t) — référence
    ax = axes[0, 0]
    ax.plot(times * 1e6, T_traj, 'k-', lw=2)
    ax.set_xlabel('Temps (µs)'); ax.set_ylabel('T (K)')
    ax.set_title('Température (trajectoire Cantera)'); ax.grid(True, alpha=0.3)

    # (0,1) : ||dω/dt|| / ||ω||
    ax = axes[0, 1]
    ax.semilogy(times * 1e6, ratio_fd, 'k-',  lw=2,   label='FD sur ω modèle')
    ax.semilogy(times * 1e6, ratio_an, 'r--', lw=1.5, label='Analytique (JVP)')
    ax.set_xlabel('Temps (µs)')
    ax.set_ylabel(r'$\|\mathrm{d}\omega/\mathrm{d}t\| \;/\; \|\omega\|$  (s⁻¹)')
    ax.set_title(r'$\|\dot\omega\| / \|\omega\|$')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (1,0) : dω/dt espèce par espèce
    ax = axes[1, 0]
    for k, (i, name) in enumerate(zip(top_idx, top_names)):
        ax.plot(times * 1e6, domega_dt_fd[:, i] * 1e-3,
                color=colors[k], lw=2,        label=f'{name} FD')
        ax.plot(times * 1e6, domega_dt_an[:, i] * 1e-3,
                color=colors[k], lw=1, ls='--', label=f'{name} an.')
    ax.set_xlabel('Temps (µs)'); ax.set_ylabel('dω/dt (kmol/m³/s²)')
    ax.set_title('dω/dt par espèce — FD vs analytique')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (1,1) : erreur relative par espèce
    ax = axes[1, 1]
    for k, (i, name) in enumerate(zip(top_idx, top_names)):
        ref = domega_dt_fd[:, i]
        rel = np.abs(domega_dt_an[:, i] - ref) / (np.abs(ref) + 1e-6)
        ax.semilogy(times * 1e6, rel, color=colors[k], lw=1.2, label=name)
    ax.set_xlabel('Temps (µs)'); ax.set_ylabel('Erreur relative')
    ax.set_title('Erreur relative dω/dt')
    ax.set_ylim([1e-5, 1e1])
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("validation_domega_dt.png", dpi=150)
    print("Figure sauvegardée : validation_domega_dt.png")

    # ── résumé numérique ──────────────────────────────────────
    mid = n_steps // 3
    print("\n── dω/dt (zone active, top 6 espèces) ──")
    for k, (i, name) in enumerate(zip(top_idx, top_names)):
        ref  = domega_dt_fd[mid:, i]
        pred = domega_dt_an[mid:, i]
        mae  = np.mean(np.abs(pred - ref))
        rel  = np.mean(np.abs(pred - ref) / (np.abs(ref) + 1e-8))
        print(f"  {name:8s}  MAE = {mae:.3e}  rel = {rel:.3e}")


    import numpy as np

    # ── un seul point de test ─────────────────────────────────
    T0 = torch.tensor([1700.0], dtype=dtype, device=device)   # (1,)
    C0 = torch.tensor([gas.concentrations * 1000],  dtype=dtype, device=device)   # (1, ns)

    # direction de perturbation arbitraire
    torch.manual_seed(42)
    v_T = torch.randn(1,          dtype=dtype, device=device)
    v_C = torch.randn(1, n_species, dtype=dtype, device=device)

    # ── JVP analytique ────────────────────────────────────────
    reactSys.evaluate_context(T0, C0)
    rhs0   = rhs_adiabatic(reactSys, vec_th)   # tangent physique
    # on teste avec le tangent physique (dT/dt, dC/dt)
    vT_phys = rhs0[:, 0]
    vC_phys = rhs0[:, 1:]
    domega_an = reactSys.jvp(vT_phys, vC_phys).cpu().numpy()   # (1, ns)

    # ── JVP par différences finies (même tangent physique) ────
    eps_fd = 1e-4   # perturbation finie

    T_p = T0 + eps_fd * vT_phys
    C_p = C0 + eps_fd * vC_phys
    reactSys.evaluate_context(T_p, C_p)
    omega_p = reactSys.omega_total().cpu().numpy()

    T_m = T0 - eps_fd * vT_phys
    C_m = C0 - eps_fd * vC_phys
    reactSys.evaluate_context(T_m, C_m)
    omega_m = reactSys.omega_total().cpu().numpy()

    domega_fd = (omega_p - omega_m) / (2 * eps_fd)   # (1, ns)

    # ── comparaison ───────────────────────────────────────────
    diff    = domega_an - domega_fd                          # (1, ns)
    rel_err = np.abs(diff) / (np.abs(domega_fd) + 1e-12)    # (1, ns)

    print("\n══ Test JVP ponctuel ══")
    print(f"Max erreur absolue : {np.abs(diff).max():.3e}")
    print(f"Max erreur relative : {rel_err.max():.3e}")
    print(f"Erreur relative médiane : {np.median(rel_err):.3e}")

    # top 5 pires espèces
    worst = np.argsort(rel_err[0])[::-1][:5]
    print("\nTop 5 pires espèces :")
    for i in worst:
        print(f"  {species[i]:10s}  FD={domega_fd[0,i]:.4e}  "
              f"AN={domega_an[0,i]:.4e}  rel={rel_err[0,i]:.3e}")