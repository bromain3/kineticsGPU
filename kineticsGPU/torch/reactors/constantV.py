import torch
import kineticsGPU.torch.thermo as thermo
import kineticsGPU.torch.reactionParser as reactionParser
from kineticsGPU.torch.reactionTypes import ElementaryReactionBatch, ThreeBodyReactionBatch, FalloffReactionBatch, ReactionSystem

class constantVReactor:
    def __init__(self, mechanism, device=None, dtype=None):
        y = thermo.yamlParser(mechanism)
        self.vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device)
        reactions = reactionParser.parseReactions(mechanism)

        r_e, r_tb, r_f, system = reactionParser.build_reactions(self.vec_th.species, reactions, self.vec_th, device, dtype)

        batched_elem = ElementaryReactionBatch(r_e, device=device, dtype=dtype)
        batched_tb = ThreeBodyReactionBatch(r_tb, device=device, dtype=dtype)
        batched_f = FalloffReactionBatch(r_f, device=device, dtype=dtype)
        self.reaction_system = ReactionSystem([batched_elem, batched_tb, batched_f])
        self.device = device
        self.dtype  = dtype

    def rhs_adiabatic(self, TC, t):
        T = TC[:, 0]
        C = TC[:, 1:]  # mol/m3
        # cinétique
        dCdt = self.reaction_system.omega_total(T, C)
        # concentration totale
        n_tot = torch.sum(C, dim=1)  # (N,)
        # fractions molaires
        X = C / n_tot[:, None]
        # Cv (molaire)
        cv_mol_mix = self.vec_th.cv_mix_mol(T, X)  # (N,)
        # dU/dt = sum_k u_k * dC_k/dt
        # u_mix_mol renvoie la valeur du mélange pour DONNER Y, pas pour dCdt
        # → on doit calculer u_species_molar explicitement
        u_species = self.vec_th.u_species_mol(T)  # (N, nspecies)
        dUdt_molar = torch.sum(u_species * dCdt, dim=1)  # (N,)
        # dT/dt = - dU/dt / (n_tot * Cv)
        dTdt = - dUdt_molar / (n_tot * cv_mol_mix)
        # assemble
        dTCdt = torch.cat((dTdt[:, None], dCdt), dim=1)
        return dTCdt

    def jvp_rhs_adiabatic(self, TC, t, dTC):
        T = TC[:, 0]
        C = TC[:, 1:]
        v_T = dTC[:, 0]
        v_C = dTC[:, 1:]
        # reaction rates and their JVP
        dCdt = self.reaction_system.omega_total(T, C)
        dCdt_jvp = self.reaction_system.jvp(T, C, v_T, v_C)
        # total moles
        n_tot = torch.sum(C, dim=1)
        dn_tot = torch.sum(v_C, dim=1)
        # mole fractions
        X = C / n_tot[:, None]
        dX = (v_C / n_tot[:, None]) - (C * dn_tot[:, None] / (n_tot**2)[:, None])
        # species thermodynamics
        u_species = self.vec_th.u_species_mol(T)                 # (N, ns)
        cv_species = self.vec_th.cv_species_mol(T, torch.arange(0, len(self.vec_th.species)))               # (N, ns)
        dcv_dT = self.vec_th.dcv_species_dT(T, torch.arange(0, len(self.vec_th.species)))                   # (N, ns)
        # mixture Cv
        cv_mix = self.vec_th.cv_mix_mol(T, X)
        dcv_mix = torch.sum(dX * cv_species, dim=1) + torch.sum(X * dcv_dT, dim=1) * v_T
        # U' = Σ u_k * dCk/dt
        U_prime = torch.sum(u_species * dCdt, dim=1)
        # dU' = Σ u_k * d(dCk/dt) + Σ dCk/dt * cv_k * vT
        dU_prime = torch.sum(u_species * dCdt_jvp, dim=1) \
                   + torch.sum(dCdt * cv_species, dim=1) * v_T
        # dT' = linearisation of -U'/(n * Cv)
        dT_prime = (
            - dU_prime / (n_tot * cv_mix)
            + U_prime * dn_tot / (n_tot**2 * cv_mix)
            + U_prime * dcv_mix / (n_tot * cv_mix**2)
        )

        return torch.cat([dT_prime[:, None], dCdt_jvp], dim=1)


class constantVReactorTPY:
    """
    Réacteur adiabatique isochore avec entrées/sorties en (T, P, Y).

    Usage
    -----
    reactor = constantVReactorTPY("mech.yaml", device="cuda", dtype=torch.float64)
    TPY0    = reactor.initial_state(T, P, Y)        # (N, 2+ns)
    dTPY    = reactor.rhs(TPY0, t=0.0)              # (N, 2+ns)
    dvTPY   = reactor.jvp(TPY0, t=0.0, vTPY)        # (N, 2+ns)
    """

    def __init__(self, mechanism, device=None, dtype=None):
        self.reactor = constantVReactor(mechanism, device=device, dtype=dtype)
        self.vec_th = self.reactor.vec_th
        self.device = device
        self.dtype = dtype
        # masses molaires (ns,) en kg/mol
        self.W = self.vec_th.molar_masses  # (ns,)

    # ─────────────────────────────────────────────────────────────────────────
    #  Changement de variables
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, TPY):
        """
        (T, P, Y) → (T, C)

        Paramètres
        ----------
        TPY : Tensor (N, 2+ns)

        Retourne
        --------
        TC  : Tensor (N, 1+ns)
        """
        T = TPY[:, 0]  # (N,)
        P = TPY[:, 1]  # (N,)
        Y = TPY[:, 2:]  # (N, ns)

        sum_Y_over_W = torch.sum(Y / self.W[None, :], dim=1)  # (N,)
        rho = P / (thermo.R * T * sum_Y_over_W)  # (N,)
        C = rho[:, None] * Y / self.W[None, :]  # (N, ns)
        return torch.cat([T[:, None], C], dim=1)  # (N, 1+ns)

    def decode(self, TC):
        """
        (T, C) → (T, P, Y)   — reconstruction complète de l'état.

        Paramètres
        ----------
        TC  : Tensor (N, 1+ns)

        Retourne
        --------
        TPY : Tensor (N, 2+ns)
        """
        T = TC[:, 0]  # (N,)
        C = TC[:, 1:]  # (N, ns)
        n_tot = torch.sum(C, dim=1)  # (N,)
        P = n_tot * thermo.R * T  # (N,)
        rho = torch.sum(C * self.W[None, :], dim=1, keepdim=True)  # (N, 1)
        Y = C * self.W[None, :] / rho  # (N, ns)
        return torch.cat([T[:, None], P[:, None], Y], dim=1)  # (N, 2+ns)

    # ─────────────────────────────────────────────────────────────────────────
    #  Conversion RHS : dTC/dt → dTPY/dt
    # ─────────────────────────────────────────────────────────────────────────

    def decode_rhs(self, TC, dTC):
        """
        Convertit le RHS interne (dT/dt, dC/dt) en (dT/dt, dP/dt, dY/dt).

        À volume constant ρ = cte → dρ/dt = 0, ce qui simplifie dY/dt :
            dY_k/dt = (W_k · dC_k/dt - Y_k · Σ_l W_l · dC_l/dt) / ρ
                    = (W_k · dC_k/dt - Y_k · dρ/dt) / ρ
                    =  W_k · dC_k/dt / ρ  -  Y_k · 0

        Paramètres
        ----------
        TC  : Tensor (N, 1+ns)  — état primal
        dTC : Tensor (N, 1+ns)  — RHS interne

        Retourne
        --------
        dTPY : Tensor (N, 2+ns)
        """
        T = TC[:, 0]  # (N,)
        C = TC[:, 1:]  # (N, ns)
        dT = dTC[:, 0]  # (N,)
        dC = dTC[:, 1:]  # (N, ns)

        n_tot = torch.sum(C, dim=1)  # (N,)
        dn_tot = torch.sum(dC, dim=1)  # (N,)

        # dP/dt = R · (T · Σ ω_k + n_tot · dT/dt)
        dP = thermo.R * (T * dn_tot + n_tot * dT)  # (N,)

        # ρ constant → dρ/dt = 0
        rho = torch.sum(C * self.W[None, :], dim=1, keepdim=True)  # (N, 1)
        # dY_k/dt = W_k · dC_k/dt / ρ
        dY = dC * self.W[None, :] / rho  # (N, ns)

        return torch.cat([dT[:, None], dP[:, None], dY], dim=1)  # (N, 2+ns)

    # ─────────────────────────────────────────────────────────────────────────
    #  JVP du décodage du RHS
    # ─────────────────────────────────────────────────────────────────────────

    def decode_rhs_jvp(self, TC, dTC, vTC, dvTC):
        """
        Tangent de decode_rhs par rapport à une perturbation (vTC, dvTC).

        On différentie decode_rhs(TC, dTC) dans la direction (vTC, dvTC) :

            d(dT/dt)/dε  = v_dT                            (trivial)

            d(dP/dt)/dε  = R · (vT · dn_tot  +  T · v_dntot
                               + v_ntot · dT  +  n_tot · v_dT)

            d(dY_k/dt)/dε = W_k · v_dC_k / ρ
                            − W_k · dC_k · v_ρ / ρ²
        avec
            v_ρ = Σ_l vC_l · W_l   (tangent de ρ)

        Paramètres
        ----------
        TC, dTC     : état primal et RHS interne
        vTC, dvTC   : tangents correspondants

        Retourne
        --------
        dvTPY : Tensor (N, 2+ns)
        """
        T = TC[:, 0];
        C = TC[:, 1:]  # primaux
        dT = dTC[:, 0];
        dC = dTC[:, 1:]

        vT = vTC[:, 0];
        vC = vTC[:, 1:]  # tangents
        v_dT = dvTC[:, 0];
        v_dC = dvTC[:, 1:]

        n_tot = torch.sum(C, dim=1)  # (N,)
        v_ntot = torch.sum(vC, dim=1)
        dn_tot = torch.sum(dC, dim=1)
        v_dntot = torch.sum(v_dC, dim=1)

        # ── tangent de dP ────────────────────────────────────────────────────
        # dP = R*(T*dn_tot + n_tot*dT)
        # d(dP)/dε = R*(vT*dn_tot + T*v_dntot + v_ntot*dT + n_tot*v_dT)
        v_dP = thermo.R * (vT * dn_tot + T * v_dntot
                           + v_ntot * dT + n_tot * v_dT)  # (N,)

        # ── tangent de dY ────────────────────────────────────────────────────
        # dY_k = W_k * dC_k / rho,  rho constant
        # d(dY_k)/dε = W_k * v_dC_k / rho - W_k * dC_k * v_rho / rho²
        rho = torch.sum(C * self.W[None, :], dim=1, keepdim=True)  # (N, 1)
        v_rho = torch.sum(vC * self.W[None, :], dim=1, keepdim=True)  # (N, 1)

        v_dY = (v_dC * self.W[None, :] / rho
                - dC * self.W[None, :] * v_rho / (rho * rho))  # (N, ns)

        return torch.cat([v_dT[:, None], v_dP[:, None], v_dY], dim=1)  # (N, 2+ns)

    # ─────────────────────────────────────────────────────────────────────────
    #  RHS principal
    # ─────────────────────────────────────────────────────────────────────────

    def rhs(self, TPY, t):
        """
        RHS du réacteur isochore dans l'espace (T, P, Y).

        Paramètres
        ----------
        TPY : Tensor (N, 2+ns)
        t   : float

        Retourne
        --------
        dTPY : Tensor (N, 2+ns)
        """
        dv = TPY.device
        dt = TPY.dtype
        TPY = TPY.to(self.reactor.dtype).to(self.reactor.device)

        TC = self.encode(TPY)
        dTC = self.reactor.rhs_adiabatic(TC, t)
        dTPY = self.decode_rhs(TC, dTC)

        return dTPY.to(dv).to(dt)

    # ─────────────────────────────────────────────────────────────────────────
    #  JVP du RHS
    # ─────────────────────────────────────────────────────────────────────────

    def jvp(self, TPY, t, dTPY_tangent):
        """
        JVP du RHS dans l'espace (T, P, Y).

        Paramètres
        ----------
        TPY          : Tensor (N, 2+ns)  — état primal
        t            : float
        dTPY_tangent : Tensor (N, 2+ns)  — direction de la perturbation

        Retourne
        --------
        dvTPY : Tensor (N, 2+ns)
        """
        dv_tpy = TPY.device
        dt_tpy = TPY.dtype
        TPY = TPY.to(self.reactor.dtype).to(self.reactor.device)
        dTPY_tangent = dTPY_tangent.to(self.reactor.dtype).to(self.reactor.device)

        T = TPY[:, 0];
        P = TPY[:, 1];
        Y = TPY[:, 2:]
        vT = dTPY_tangent[:, 0]
        vP = dTPY_tangent[:, 1]
        vY = dTPY_tangent[:, 2:]

        # ── 1. Encodage primal ───────────────────────────────────────────────
        TC = self.encode(TPY)
        C = TC[:, 1:]

        # ── 2. Tangent de l'encodage : (vT, vP, vY) → (vT, vC) ─────────────
        # Même formule que constantPReactorTPY.jvp :
        #   C_k = ρ·Y_k/W_k,  ρ = P/(R·T·Σ Y/W)
        #   vC_k = C_k·(vP/P − vT/T − v_sYW/sYW) + ρ·vY_k/W_k
        sum_Y_W = torch.sum(Y / self.W[None, :], dim=1, keepdim=True)  # (N,1)
        v_sum_Y_W = torch.sum(vY / self.W[None, :], dim=1, keepdim=True)  # (N,1)
        rho = P[:, None] / (thermo.R * T[:, None] * sum_Y_W)  # (N,1)

        term_global = (vP[:, None] / P[:, None]
                       - vT[:, None] / T[:, None]
                       - v_sum_Y_W / sum_Y_W)
        vC = C * term_global + rho * vY / self.W[None, :]  # (N, ns)
        vTC = torch.cat([vT[:, None], vC], dim=1)  # (N, 1+ns)

        # ── 3. RHS et JVP internes ───────────────────────────────────────────
        dTC = self.reactor.rhs_adiabatic(TC, t)
        dvTC = self.reactor.jvp_rhs_adiabatic(TC, t, vTC)

        # ── 4. Tangent du décodage RHS ───────────────────────────────────────
        dvTPY = self.decode_rhs_jvp(TC, dTC, vTC, dvTC)

        return dvTPY.to(dv_tpy).to(dt_tpy)

    # ─────────────────────────────────────────────────────────────────────────
    #  Utilitaire : état initial
    # ─────────────────────────────────────────────────────────────────────────

    def initial_state(self, T, P, Y):
        """
        Construit le tenseur TPY initial.

        Paramètres
        ----------
        T : scalar ou array-like (N,)
        P : scalar ou array-like (N,)
        Y : array-like (ns,) ou (N, ns)

        Retourne
        --------
        TPY : Tensor (N, 2+ns)
        """
        T = torch.as_tensor(T, dtype=self.W.dtype, device=self.W.device).reshape(-1)
        P = torch.as_tensor(P, dtype=self.W.dtype, device=self.W.device).reshape(-1)
        Y = torch.as_tensor(Y, dtype=self.W.dtype, device=self.W.device)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0).expand(T.shape[0], -1)
        Y = Y / Y.sum(dim=1, keepdim=True)  # normalisation de sécurité
        N = T.shape[0]
        P = P.expand(N)
        return torch.cat([T[:, None], P[:, None], Y], dim=1)  # (N, 2+ns)

    def condition_number(self, TPY, t):
        """
        Estime le nombre de conditionnement du système chimique :
            κ = ‖J_ω · f(TC)‖ / ‖ω‖

        où :
          - ω = dC/dt  (ns composantes du RHS chimique)
          - J_ω · f(TC) = JVP de ω dans la direction du RHS complet (dT/dt, dC/dt)

        Reproduit exactement le calcul `torch_jvp` du script de référence.

        Paramètres
        ----------
        TPY   : Tensor (N, 2+ns)
        t     : float

        Retourne
        --------
        kappa : Tensor (N,)
        """
        print('WARNING, condition number gives strange results, especially for near equilibrium regions')
        TPY = TPY.to(self.reactor.dtype).to(self.reactor.device)

        # ── 1. Encodage primal ───────────────────────────────────────────────
        TC = self.encode(TPY)

        # ── 2. RHS complet du réacteur : (dT/dt, dC/dt) ─────────────────────
        dTC = self.reactor.rhs_adiabatic(TC, t)
        omega = dTC[:, 1:]   # (N, ns)  — ω = dC/dt

        # ── 3. Tangent = RHS complet (vT = dT/dt, vC = dC/dt) ───────────────
        #    Reproduit : jvp(T, C, dTCdt[:,0], dTCdt[:,1:])
        vTC = dTC   # (N, 1+ns)
        Jomega_f = self.reactor.reaction_system.jvp(TC[:, 0], TC[:, 1:], vTC[:, 0], vTC[:, 1:])

        # ── 5. κ = ‖J_ω · f‖ / ‖ω‖ ─────────────────────────────────────────
        norm_num   = torch.linalg.norm(Jomega_f, dim=1)   # (N,)
        norm_denom = torch.linalg.norm(omega,    dim=1)   # (N,)
        print(norm_num, norm_denom)
        kappa = norm_num / norm_denom   # (N,)
        return kappa