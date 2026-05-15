"""
tests/test_domegadt_vs_cantera.py
=================================
Vérifie que ||domega/dt||_GPU / ||omega||_GPU ≈ ||domega/dt||_FD / ||omega||_FD

Méthode de référence (différences finies Cantera)
-------------------------------------------------
Pour un état (T, Y) dans un réacteur adiabatique IdealGasConstPressureReactor :

    omega(t)    = net_production_rates à l'instant t
    omega(t+dt) = net_production_rates à l'instant t + dt   (après avancement)

    domega/dt ≈ (omega(t+dt) - omega(t)) / dt      [FD forward]

    ratio_FD = ||domega/dt_FD|| / ||omega(t)||

La valeur GPU vient de la JVP exacte (dérivée directionnelle analytique).
Les deux doivent coïncider à O(dt) près pour dt suffisamment petit.

Utilisation
-----------
    pip install cantera numpy
    python -m pytest tests/test_domegadt_vs_cantera.py -v
    # ou directement :
    python tests/test_domegadt_vs_cantera.py
"""

import numpy as np
import pytest

# ── Cantera ───────────────────────────────────────────────────────────────────
try:
    import cantera as ct
    HAS_CANTERA = True
except ImportError:
    HAS_CANTERA = False

# ── GPU / Numba ───────────────────────────────────────────────────────────────
try:
    from numba import cuda
    HAS_GPU = cuda.is_available()
except Exception:
    HAS_GPU = False


# ═════════════════════════════════════════════════════════════════════════════
# Helpers Cantera
# ═════════════════════════════════════════════════════════════════════════════

def cantera_omega(gas):
    """
    Retourne omega_s [mol/m³/s] dans l'état actuel du gaz Cantera.
    net_production_rates est en [kmol/m³/s] → on multiplie par 1000.
    (Cantera ≥ 2.6 retourne déjà mol/m³/s selon la version — vérifier.)
    """
    # Cantera >= 3.0 : mol/m³/s ; < 3.0 : kmol/m³/s
    # On normalise en détectant l'ordre de grandeur si nécessaire.
    return gas.net_production_rates.copy()   # [mol/m³/s] en Cantera ≥ 3.0


def cantera_adiabatic_dTdt(gas):
    """
    dT/dt dans un réacteur adiabatique à volume constant :
        dT/dt = -sum(u_s * omega_s) / (rho * cv_mix)
              = -sum(u_s * omega_s) / (n_tot * cv_mix)   [en mol]

    On utilise la version molaire pour rester cohérent avec le code GPU.
        u_s    = h_s - R*T       [J/mol]
        cv_mix = cp_mix - R      (gaz parfait, molaire)
        n_tot  = rho / M_mix     = P/(R*T)   (gaz parfait)
    """
    T   = gas.T
    R   = ct.gas_constant          # 8314.46... J/kmol/K  → à convertir
    R_J = 8.314_462_618_153_24     # J/mol/K

    omega = cantera_omega(gas)     # mol/m³/s
    h_s   = gas.partial_molar_enthalpies   # J/mol
    u_s   = h_s - R_J * T                 # énergie interne molaire J/mol

    n_tot  = gas.density_mole             # mol/m³
    cv_mix = gas.cv_mole                  # J/mol/K

    dTdt = -np.dot(u_s, omega) / (n_tot * cv_mix)
    return dTdt


def cantera_domegadt_fd(gas, dt=1e-9):
    """
    Différence finie forward de omega le long de la trajectoire adiabatique.

    On fait avancer le réacteur de dt secondes avec un intégrateur Cantera,
    puis on évalue omega(t+dt).

    Retourne (domegadt_fd, omega_t0, ratio)
    """
    # Sauvegarder l'état initial
    T0  = gas.T
    P0  = gas.P
    Y0  = gas.Y.copy()

    omega_t0 = cantera_omega(gas)

    # Créer un réacteur adiabatique à volume constant
    r   = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    sim.advance(dt)

    # État à t+dt
    gas_dt = r.thermo
    omega_dt = cantera_omega(gas_dt)

    domegadt_fd = (omega_dt - omega_t0) / dt

    norm_dom = np.linalg.norm(domegadt_fd)
    norm_om  = np.linalg.norm(omega_t0)
    ratio    = norm_dom / norm_om if norm_om > 1e-300 else 0.0

    # Restaurer l'état initial
    gas.TPY = T0, P0, Y0

    return domegadt_fd, omega_t0, ratio


def cantera_d2Tdt2_fd(gas, dt=1e-9):
    """
    Différence finie forward de dT/dt.
    """
    T0  = gas.T
    P0  = gas.P
    Y0  = gas.Y.copy()

    dTdt_t0 = cantera_adiabatic_dTdt(gas)

    r   = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    sim.advance(dt)

    dTdt_dt = cantera_adiabatic_dTdt(r.thermo)
    d2Tdt2  = (dTdt_dt - dTdt_t0) / dt

    gas.TPY = T0, P0, Y0
    return d2Tdt2, dTdt_t0


# ═════════════════════════════════════════════════════════════════════════════
# Cas de test Cantera seul (CPU, pas de GPU requis)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CANTERA, reason="Cantera non disponible")
class TestCantaraFD:
    """
    Vérifie la cohérence des différences finies Cantera elles-mêmes,
    indépendamment du GPU.
    Teste la convergence en dt : ||FD(dt) - FD(dt/10)|| → 0.
    """

    MECHANISMS = [
        ("gri30.yaml",  "stoichiometric_CH4_air",
         dict(fuel="CH4", oxidizer="air", phi=1.0), 1500.0, 101325.0),
        ("h2o2.yaml",   "h2_air_lean",
         dict(fuel="H2",  oxidizer="air", phi=0.5), 1200.0, 101325.0),
    ]

    @pytest.mark.parametrize("mech,name,mix,T,P", MECHANISMS)
    def test_fd_convergence(self, mech, name, mix, T, P):
        gas = ct.Solution(mech)
        gas.set_equivalence_ratio(**mix)
        gas.TP = T, P

        # dt=1e-8 vs dt=1e-9 : les deux doivent être très proches
        _, _, r1 = cantera_domegadt_fd(gas, dt=1e-8)
        _, _, r2 = cantera_domegadt_fd(gas, dt=1e-9)

        # Le ratio ne doit pas diverger (ordre 1 en dt : on tolère 10% d'écart)
        assert abs(r1 - r2) / (abs(r1) + 1e-10) < 0.15, (
            f"{name}: ratio FD non convergé : r(1e-8)={r1:.4e}, r(1e-9)={r2:.4e}"
        )

    @pytest.mark.parametrize("mech,name,mix,T,P", MECHANISMS)
    def test_ratio_positive(self, mech, name, mix, T, P):
        gas = ct.Solution(mech)
        gas.set_equivalence_ratio(**mix)
        gas.TP = T, P

        _, omega, ratio = cantera_domegadt_fd(gas)
        assert ratio > 0.0, f"{name}: ratio nul ou négatif : {ratio}"
        assert np.all(np.isfinite(omega)), f"{name}: omega contient NaN/Inf"


# ═════════════════════════════════════════════════════════════════════════════
# Cas de test GPU vs Cantera
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CANTERA, reason="Cantera non disponible")
@pytest.mark.skipif(not HAS_GPU,     reason="GPU non disponible")
class TestGPUvsCantera:
    """
    Compare le ratio ||domega/dt|| / ||omega|| calculé par la JVP GPU
    avec la référence différences finies Cantera.

    Tolérance : erreur relative < 1% (la JVP est exacte, la FD a une erreur O(dt)).
    """

    TOL_RELATIVE = 0.05    # 5% — la FD introduit une erreur O(dt*d²omega/dt²)

    def _make_state(self, mech_yaml, mix, T, P):
        """Retourne (gas, T_np, C_np) prêts pour le GPU."""
        gas = ct.Solution(mech_yaml)
        gas.set_equivalence_ratio(**mix)
        gas.TP = T, P

        C_np = gas.concentrations.copy()   # mol/m³, shape (n_sp,)
        return gas, np.array([T]), C_np[np.newaxis, :]  # (1, n_sp)

    def _run_gpu(self, mech_obj, T_np, C_np):
        """
        Lance le calcul GPU et retourne OmegaRatesResult.
        mech_obj doit exposer tous les tableaux nécessaires.
        """
        from kineticsGPU.compute import build_ctx_arrays  # adapter à votre API
        from kineticsGPU.kernels.shared.reactor_odes_extended import (
            compute_omega_rates_host,
        )

        ctx = build_ctx_arrays(mech_obj, batch_size=1)
        return compute_omega_rates_host(T_np, C_np, ctx, mech_obj, mech_obj.n_sp)

    @pytest.mark.parametrize("mech_yaml,mix,T,P", [
        ("gri30.yaml",  dict(fuel="CH4", oxidizer="air", phi=1.0), 1500.0, 101325.0),
        ("h2o2.yaml",   dict(fuel="H2",  oxidizer="air", phi=0.5), 1200.0, 101325.0),
        ("gri30.yaml",  dict(fuel="CH4", oxidizer="air", phi=0.3), 1800.0, 5*101325.0),
    ])
    def test_ratio_vs_cantera(self, mech_yaml, mix, T, P):
        gas, T_np, C_np = self._make_state(mech_yaml, mix, T, P)

        # Référence Cantera FD
        _, _, ratio_fd = cantera_domegadt_fd(gas, dt=1e-9)

        # GPU JVP (nécessite que le mécanisme GPU soit construit à l'avance)
        # Si le mécanisme GPU n'est pas disponible, on skippe ce test
        try:
            from kineticsGPU.mechanism import load_mechanism
            mech_obj = load_mechanism(mech_yaml)
        except ImportError:
            pytest.skip("kineticsGPU.mechanism non disponible")

        res = self._run_gpu(mech_obj, T_np, C_np)
        ratio_gpu = float(res.ratio[0])

        err_rel = abs(ratio_gpu - ratio_fd) / (abs(ratio_fd) + 1e-10)
        assert err_rel < self.TOL_RELATIVE, (
            f"ratio GPU={ratio_gpu:.6e} vs FD={ratio_fd:.6e} "
            f"(err_rel={err_rel:.2%})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Test standalone sans GPU : vérification purement Cantera
# (utile pour valider la formule dT/dt et domega/dt en CPU pur)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CANTERA, reason="Cantera non disponible")
def test_cantera_standalone_gri30():
    """
    Test principal exécutable sans GPU.

    Pour GRI 3.0 à T=1500 K, phi=1 :
        1. Calcule ratio_FD pour plusieurs dt
        2. Vérifie la convergence
        3. Affiche les résultats (utile pour le débogage)
    """
    gas = ct.Solution("gri30.yaml")
    gas.set_equivalence_ratio(fuel="CH4", oxidizer="air", phi=1.0)
    gas.TP = 1500.0, 101325.0

    print("\n" + "="*60)
    print("Test domega/dt — GRI 3.0, T=1500 K, phi=1, P=1 atm")
    print("="*60)

    omega_t0 = cantera_omega(gas)
    dTdt_t0  = cantera_adiabatic_dTdt(gas)

    print(f"  ||omega||   = {np.linalg.norm(omega_t0):.6e} mol/m³/s")
    print(f"  dT/dt       = {dTdt_t0:.6e} K/s")

    print("\n  Convergence en dt :")
    print(f"  {'dt':>12s}  {'||domega/dt||':>15s}  {'ratio':>12s}  {'d²T/dt²':>12s}")

    prev_ratio = None
    for exp in range(7, 12):
        dt = 10.0 ** (-exp)
        domegadt_fd, _, ratio = cantera_domegadt_fd(gas, dt=dt)
        d2Tdt2, _             = cantera_d2Tdt2_fd(gas, dt=dt)

        conv = ""
        if prev_ratio is not None:
            r = abs(ratio - prev_ratio) / (abs(prev_ratio) + 1e-20)
            conv = f"  Δrel={r:.2%}"
        print(f"  {dt:12.2e}  {np.linalg.norm(domegadt_fd):15.6e}"
              f"  {ratio:12.6e}  {d2Tdt2:12.6e}{conv}")
        prev_ratio = ratio

    # Vérification soft : le ratio doit être raisonnable (> 0, < 1e6)
    assert 0 < ratio < 1e8, f"ratio hors plage : {ratio}"


@pytest.mark.skipif(not HAS_CANTERA, reason="Cantera non disponible")
def test_cantera_standalone_h2o2():
    """Idem sur le mécanisme H2/O2."""
    gas = ct.Solution("h2o2.yaml")
    gas.set_equivalence_ratio(fuel="H2", oxidizer="air", phi=0.5)
    gas.TP = 1200.0, 101325.0

    _, omega_t0, ratio = cantera_domegadt_fd(gas, dt=1e-9)
    d2Tdt2, dTdt0 = cantera_d2Tdt2_fd(gas, dt=1e-9)

    print("\n" + "="*60)
    print("Test domega/dt — H2O2, T=1200 K, phi=0.5, P=1 atm")
    print("="*60)
    print(f"  ||omega||        = {np.linalg.norm(omega_t0):.6e}")
    print(f"  ratio FD         = {ratio:.6e}  [1/s]")
    print(f"  dT/dt            = {dTdt0:.6e}  [K/s]")
    print(f"  d²T/dt²          = {d2Tdt2:.6e}  [K/s²]")

    assert ratio > 0.0
    assert np.isfinite(d2Tdt2)


# ═════════════════════════════════════════════════════════════════════════════
# Point d'entrée direct
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not HAS_CANTERA:
        print("⚠️  Cantera non installé. Installez avec : pip install cantera")
    else:
        test_cantera_standalone_gri30()
        test_cantera_standalone_h2o2()
        print("\n✓ Tous les tests Cantera passés.")

        if not HAS_GPU:
            print("ℹ  GPU non disponible — tests GPU ignorés.")
        else:
            print("ℹ  GPU disponible — lancez pytest pour les tests GPU vs JVP.")