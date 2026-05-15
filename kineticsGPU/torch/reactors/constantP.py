import torch
import kineticsGPU.torch.thermo as thermo
import kineticsGPU.torch.reactionParser as reactionParser
from kineticsGPU.torch.reactionTypes import (
    ElementaryReactionBatch, ThreeBodyReactionBatch,
    FalloffReactionBatch, ReactionSystem
)

class constantPReactor:
    def __init__(self, mechanism, device=None, dtype=None):
        y = thermo.yamlParser(mechanism)
        self.vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device)
        reactions = reactionParser.parseReactions(mechanism)
        r_e, r_tb, r_f, system = reactionParser.build_reactions(
            self.vec_th.species, reactions, self.vec_th, device, dtype
        )
        batched_elem = ElementaryReactionBatch(r_e, device=device, dtype=dtype)
        batched_tb  = ThreeBodyReactionBatch(r_tb, device=device, dtype=dtype)
        batched_f   = FalloffReactionBatch(r_f, device=device, dtype=dtype)
        self.reaction_system = ReactionSystem([batched_elem, batched_tb, batched_f])
        self.device = device
        self.dtype  = dtype

    def rhs(self, TC, t):
        T = TC[:, 0]
        C = TC[:, 1:]
        omega = self.reaction_system.omega_total(T, C)
        n_tot = torch.sum(C, dim=1)
        X = C / n_tot[:, None]
        h_species = self.vec_th.h_species_mol(T)
        cp_mix = self.vec_th.cp_mix_mol(T, X)
        # Équation de l'énergie
        dHdt = torch.sum(h_species * omega, dim=1)
        dTdt = -dHdt / (n_tot * cp_mix)
        # Dilatation TOTALE (thermique + chimique)
        # Indispensable pour que dP/dt = 0
        omega_sum = torch.sum(omega, dim=1)
        dilation = (omega_sum / n_tot) + (dTdt / T)
        dCdt = omega - C * dilation[:, None]
        return torch.cat([dTdt[:, None], dCdt], dim=1)

    def jvp(self, TC, t, dTC):
        T = TC[:, 0]
        C = TC[:, 1:]
        v_T = dTC[:, 0]
        v_C = dTC[:, 1:]
        # --- Primal ---
        omega = self.reaction_system.omega_total(T, C)
        n_tot = torch.sum(C, dim=1)
        X = C / n_tot[:, None]
        # Correction API : on utilise souvent cp_species_mol(T) sans ns_idx
        # Si l'erreur persiste, essaie self.vec_th.cp_mol(T)
        h_species = self.vec_th.h_species_mol(T)
        cp_species = self.vec_th.cp_species_mol(T)  # (N, ns)
        cp_mix = torch.sum(X * cp_species, dim=1)
        dTdt = -torch.sum(h_species * omega, dim=1) / (n_tot * cp_mix)
        omega_sum = torch.sum(omega, dim=1)
        # --- Tangent (JVP) ---
        omega_jvp = self.reaction_system.jvp(T, C, v_T, v_C)
        dn_tot = torch.sum(v_C, dim=1)
        # JVP de dTdt (simplifiée pour la lisibilité mais exacte)
        # cp_mix_jvp nécessite dcp/dT
        ns_idx = torch.arange(C.shape[1], device=TC.device)
        dcp_dT = self.vec_th.dcp_species_dT(T, ns_idx)
        v_cp_mix = torch.sum((v_C / n_tot[:, None] - C * dn_tot[:, None] / n_tot[:, None] ** 2) * cp_species, dim=1) + torch.sum(X * dcp_dT, dim=1) * v_T
        dH_prime = torch.sum(h_species * omega_jvp, dim=1) + torch.sum(cp_species * omega, dim=1) * v_T
        v_dTdt = -(dH_prime / (n_tot * cp_mix)) + (torch.sum(h_species * omega, dim=1) * (dn_tot * cp_mix + n_tot * v_cp_mix)) / (n_tot * cp_mix) ** 2
        # JVP de la dilatation
        v_omega_sum = torch.sum(omega_jvp, dim=1)
        v_dilation = (v_omega_sum / n_tot - omega_sum * dn_tot / n_tot ** 2) + (v_dTdt / T - dTdt * v_T / T ** 2)
        dilation = (omega_sum / n_tot) + (dTdt / T)
        v_dCdt = omega_jvp - (v_C * dilation[:, None] + C * v_dilation[:, None])
        return torch.cat([v_dTdt[:, None], v_dCdt], dim=1)

class constantPReactorTPY:
    def __init__(self, mechanism, device=None, dtype=None):
        self.reactor = constantPReactor(mechanism, device=device, dtype=dtype)
        self.vec_th = self.reactor.vec_th
        # masses molaires (ns,) en kg/mol
        W = self.vec_th.molar_masses  # (ns,)  [kg/mol]
        self.W = W

    def encode(self, TPY):
        T = TPY[:, 0]  # (N,)
        P = TPY[:, 1]  # (N,)
        Y = TPY[:, 2:]  # (N, ns)
        # 1. Calcul de sum(Y_k / W_k) qui équivaut à 1 / W_mean
        sum_Y_over_W = torch.sum(Y / self.W[None, :], dim=1)  # (N,)
        # 2. Masse volumique : rho = P / (R_spe * T)
        rho = P / (thermo.R * T * sum_Y_over_W)  # (N,)
        # 3. Concentrations molaires : C_k = (rho * Y_k) / W_k
        C = rho[:, None] * Y / self.W[None, :]  # (N, ns)
        return torch.cat([T[:, None], C], dim=1)  # (N, 1+ns)

    def decode(self, TC):
        T = TC[:, 0]  # (N,)
        C = TC[:, 1:]  # (N, ns)

        n_tot = torch.sum(C, dim=1)  # (N,)
        P = n_tot * thermo.R * T  # (N,)

        rho = torch.sum(C * self.W[None, :], dim=1, keepdim=True)  # (N, 1)
        Y = C * self.W[None, :] / rho  # (N, ns)

        return torch.cat([T[:, None], P[:, None], Y], dim=1)  # (N, 2+ns)



    # ------------------------------------------------------------------
    #  Conversion du RHS :  dTC/dt  →  dTPY/dt
    # ------------------------------------------------------------------

    def decode_rhs(self, TC, dTC):
        T = TC[:, 0]  # (N,)
        C = TC[:, 1:]  # (N, ns)
        dT = dTC[:, 0]  # (N,)
        dC = dTC[:, 1:]  # (N, ns)
        n_tot = torch.sum(C, dim=1)  # (N,)
        dn_tot = torch.sum(dC, dim=1)  # (N,)
        # dP/dt
        dP = (dn_tot * thermo.R * T) + (n_tot * thermo.R * dT)  # (N,)
        # dY/dt
        rho = torch.sum(C * self.W[None, :], dim=1, keepdim=True)  # (N,1)
        drho = torch.sum(dC * self.W[None, :], dim=1, keepdim=True)  # (N,1)
        Y = C * self.W[None, :] / rho  # (N, ns)
        dY = (dC * self.W[None, :] - Y * drho) / rho  # (N, ns)
        return torch.cat([dT[:, None], dP[:, None], dY], dim=1)  # (N, 2+ns)

    # ------------------------------------------------------------------
    #  RHS principal  (pour le solveur ODE)
    # ------------------------------------------------------------------

    def rhs(self, TPY, t):
        dv = TPY.device
        dt = TPY.dtype
        TPY = TPY.to(self.reactor.dtype).to(self.reactor.device)
        TC = self.encode(TPY)  # → [T, C]
        dTC = self.reactor.rhs(TC, t)  # RHS interne
        dTPY = self.decode_rhs(TC, dTC)  # → [T, P, Y]
        # À P constante : on force dP/dt = 0
        dTPY[:, 1] = 0.0

        return dTPY.to(dv).to(dt)

    # ------------------------------------------------------------------
    #  JVP du RHS (pour les solveurs différentiables / sensibilités)
    # ------------------------------------------------------------------
    def decode_rhs_jvp(self, TC, dTC, vTC, dvTC):
        T = TC[:, 0]
        C = TC[:, 1:]
        dT = dTC[:, 0]
        dC = dTC[:, 1:]

        vT = vTC[:, 0]
        vC = vTC[:, 1:]
        v_dT = dvTC[:, 0]
        v_dC = dvTC[:, 1:]

        n_tot = torch.sum(C, dim=1)  # (N,)
        v_ntot = torch.sum(vC, dim=1)  # tangent de n_tot
        dn_tot = torch.sum(dC, dim=1)  # = dTC version "speed"
        v_dntot = torch.sum(v_dC, dim=1)  # tangent de dn_tot
        # --- tangent de dP ---
        # dP = dn_tot*R*T + n_tot*R*dT
        # d(dP)/dε = v_dntot*R*T + dn_tot*R*vT + v_ntot*R*dT + n_tot*R*v_dT
        v_dP = (v_dntot * thermo.R * T
                + dn_tot * thermo.R * vT
                + v_ntot * thermo.R * dT
                + n_tot * thermo.R * v_dT)
        # --- quantités primales de decode_rhs ---
        rho = torch.sum(C * self.W[None, :], dim=1, keepdim=True)  # (N,1)
        drho = torch.sum(dC * self.W[None, :], dim=1, keepdim=True)  # (N,1)
        Y = C * self.W[None, :] / rho  # (N,ns)
        # tangents correspondants
        v_rho = torch.sum(vC * self.W[None, :], dim=1, keepdim=True)
        v_drho = torch.sum(v_dC * self.W[None, :], dim=1, keepdim=True)
        # --- tangent de dY ---
        # dY = (dC*W - Y*drho) / rho
        # d(dY)/dε = (v_dC*W - v_Y*drho - Y*v_drho) / rho
        #           - (dC*W - Y*drho) * v_rho / rho^2
        num = dC * self.W[None, :] - Y * drho  # (N,ns)
        v_Y = (vC * self.W[None, :] - Y * v_rho) / rho  # tangent de Y
        v_num = v_dC * self.W[None, :] - v_Y * drho - Y * v_drho
        v_dY = v_num / rho - num * v_rho / (rho * rho)
        # --- tangent de dT est directement v_dT ---
        return torch.cat([v_dT[:, None], v_dP[:, None], v_dY], dim=1)  # (N, 2+ns)

    def jvp(self, TPY, t, dTPY_tangent):
        dv_tpy = TPY.device
        dt_tpy = TPY.dtype
        TPY = TPY.to(self.reactor.dtype).to(self.reactor.device)
        dv_vtpy = dTPY_tangent.device
        dt_vtpy = dTPY_tangent.dtype
        dTPY_tangent = dTPY_tangent.to(self.reactor.dtype).to(self.reactor.device)
        T = TPY[:, 0]
        P = TPY[:, 1]
        Y = TPY[:, 2:]
        vT = dTPY_tangent[:, 0]
        vP = dTPY_tangent[:, 1]
        vY = dTPY_tangent[:, 2:]
        # --- 1. Encodage Primal ---
        TC = self.encode(TPY)
        C = TC[:, 1:]
        # --- 2. Tangent de l'encodage (vC) ---
        # Calcul de la somme Y/W et de son tangent
        sum_Y_W = torch.sum(Y / self.W[None, :], dim=1, keepdim=True)
        v_sum_Y_W = torch.sum(vY / self.W[None, :], dim=1, keepdim=True)
        # rho = P / (R * T * sum_Y_W)
        rho = P[:, None] / (thermo.R * T[:, None] * sum_Y_W)
        # vC_k = C_k * (vP/P - vT/T - v_sum_Y_W/sum_Y_W) + (rho * vY_k / W_k)
        term_global = (vP[:, None] / P[:, None] - vT[:, None] / T[:, None] - v_sum_Y_W / sum_Y_W)
        vC = C * term_global + (rho * vY / self.W[None, :])
        vTC = torch.cat([vT[:, None], vC], dim=1)
        # --- 3. RHS et JVP internes ---
        dTC = self.reactor.rhs(TC, t)
        dvTC = self.reactor.jvp(TC, t, vTC)
        # --- 4. Décodage du tangent ---
        dvTPY = self.decode_rhs_jvp(TC, dTC, vTC, dvTC)
        # Contrainte Pression constante : le tangent de dP/dt est nul
        dvTPY[:, 1] = 0.0
        return dvTPY.to(dv_tpy).to(dt_tpy)


    def initial_state(self, T, P, Y):
        T = torch.as_tensor(T, dtype=self.W.dtype, device=self.W.device).reshape(-1)
        P = torch.as_tensor(P, dtype=self.W.dtype, device=self.W.device).reshape(-1)
        Y = torch.as_tensor(Y, dtype=self.W.dtype, device=self.W.device)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0).expand(T.shape[0], -1)
        # normalisation de sécurité
        Y = Y / Y.sum(dim=1, keepdim=True)
        N = T.shape[0]
        P = P.expand(N)
        return torch.cat([T[:, None], P[:, None], Y], dim=1)  # (N, 2+ns)