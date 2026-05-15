from dataclasses import dataclass
import torch

#TODO: read Pref from chem file
#TODO: implement a full GPU parallelization with torch.scatter_add (not in place) for computation speed and scatter_add_ (in place) for memory
R = 8.31446261815324  # J/(mol*K)
Pref = 101325.0       # Pa

@dataclass
class ElementaryReactionsContext:
    T: torch.Tensor
    C: torch.Tensor

    # kinetics
    kf: torch.Tensor
    Kc: torch.Tensor
    conc_r: torch.Tensor
    conc_p: torch.Tensor
    RR: torch.Tensor

    h_r: torch.Tensor
    s_r: torch.Tensor
    h_p: torch.Tensor
    s_p: torch.Tensor
    deltaG0: torch.Tensor


class ElementaryReactionBatchContext:
    """
    Fully GPU-vectorized batch for multiple ElementaryReaction instances.
    Handles different-sized reactants/products without Python loops.
    """

    def __init__(self, reactions, device, dtype, eps=1e-300):
        self.device = device
        self.dtype = dtype
        self.eps = eps

        self.n_reactions = len(reactions)
        self.thermo = reactions[0].thermo
        self.species = reactions[0].species

        # ----- Flattened indices and stoichiometries -----
        self.react_idx_list = []
        self.prod_idx_list = []
        self.react_sto_list = []
        self.prod_sto_list = []
        self.reaction_offsets_r = [0]
        self.reaction_offsets_p = [0]

        oneWay = []
        A, n, Ea = [], [], []

        for r in reactions:
            self.react_idx_list.extend(r.reactants_idx)
            self.prod_idx_list.extend(r.products_idx)
            self.react_sto_list.extend(r.reactants_sto)
            self.prod_sto_list.extend(r.products_sto)

            self.reaction_offsets_r.append(self.reaction_offsets_r[-1] + len(r.reactants_idx))
            self.reaction_offsets_p.append(self.reaction_offsets_p[-1] + len(r.products_idx))

            oneWay.append(float(r.oneWay))
            A.append(r.A.item())
            n.append(r.n.item())
            Ea.append(r.Ea.item())

        # Convert to tensors
        self.react_idx = torch.tensor(self.react_idx_list, dtype=torch.long, device=device)
        self.prod_idx = torch.tensor(self.prod_idx_list, dtype=torch.long, device=device)
        self.react_sto = torch.tensor(self.react_sto_list, dtype=dtype, device=device)
        self.prod_sto = torch.tensor(self.prod_sto_list, dtype=dtype, device=device)
        self.reaction_offsets_r = torch.tensor(self.reaction_offsets_r, dtype=torch.long, device=device)
        self.reaction_offsets_p = torch.tensor(self.reaction_offsets_p, dtype=torch.long, device=device)
        self.oneWay = torch.tensor(oneWay, dtype=dtype, device=device)
        self.A = torch.tensor(A, dtype=dtype, device=device)
        self.n = torch.tensor(n, dtype=dtype, device=device)
        self.Ea = torch.tensor(Ea, dtype=dtype, device=device)

        self.idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_r)
        )
        self.idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_p)
        )

        self.delta_n = torch.tensor(
            [torch.sum(r.products_sto) - torch.sum(r.reactants_sto) for r in reactions],
            dtype=dtype, device=device
        )

    def evaluate_context(self, T, C):
        # ---- kinetics ----
        kf = self.kf(T)
        Kc, h_r, s_r, h_p, s_p, deltaG0  = self.Kc(T)

        conc_r, conc_p = self._compute_concentrations(T, C)
        RR = kf * (conc_r - (1 - self.oneWay[None, :]) * conc_p / Kc)




        return ElementaryReactionsContext(
            T=T, C=C,
            kf=kf, Kc=Kc,
            conc_r=conc_r,
            conc_p=conc_p,
            RR=RR,
            h_r=h_r,
            s_r=s_r,
            h_p=h_p,
            s_p=s_p,
            deltaG0=deltaG0
        )

    def _compute_concentrations(self, T, C):
        ncells = T.shape[0]

        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])  # (ncells, n_total_react)
        logC_p = torch.log(C_safe[:, self.prod_idx])

        conc_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        conc_p = torch.zeros_like(conc_r)
        #TODO: change scatter_add_ (in place) for scatter_add (not in place) for torch.compile compatibility
        conc_r = conc_r.scatter_add(1, self.idx_r_scatter[None, :].expand(ncells, -1), logC_r * self.react_sto[None, :]) # not in place fonction to make it compatible with vmap and torch.compile
        conc_p = conc_p.scatter_add(1, self.idx_p_scatter[None, :].expand(ncells, -1), logC_p * self.prod_sto[None, :]) # not in place fonction to make it compatible with vmap and torch.compile
        conc_r = torch.exp(conc_r)
        conc_p = torch.exp(conc_p)
        return(conc_r, conc_p)

    # ----- Arrhenius -----
    def kf(self, T):
        T_exp = T[:, None]  # (ncells,1)
        return self.A[None, :] * (T_exp ** self.n[None, :]) * torch.exp(-self.Ea[None, :] / (R * T_exp))

    # ----- Equilibrium Kc -----
    def Kc(self, T):
        ncells = T.shape[0]

        h_r = self.thermo.h_species(T, self.react_idx)  # (ncells, n_total_react)
        s_r = self.thermo.s_species_Pref(T, self.react_idx)
        h_p = self.thermo.h_species(T, self.prod_idx)
        s_p = self.thermo.s_species_Pref(T, self.prod_idx)

        G0_r = h_r - T[:, None] * s_r
        G0_p = h_p - T[:, None] * s_p

        # Sum contributions per reaction using scatter_add
        deltaG0_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        deltaG0_p = torch.zeros_like(deltaG0_r)

        # Compute indices for scatter
        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        # Expand stoichiometry to (ncells, n_total)
        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaG0_r = deltaG0_r.scatter_add(1, idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r) # not in place fonction to make it compatible with vmap and torch.compile
        deltaG0_p = deltaG0_p.scatter_add(1, idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p) # not in place fonction to make it compatible with vmap and torch.compile

        deltaG0 = deltaG0_p - deltaG0_r
        Kp = torch.exp(-deltaG0 / (R * T[:, None]))
        Kc = Kp * (Pref / (R * T[:, None])) ** self.delta_n[None, :]
        return (Kc, h_r, s_r, h_p, s_p, deltaG0)


    # ------------------------------------------------------------
    # Compute species production rates omega from RR
    # RR : (ncells, n_reactions)
    # Returns w : (ncells, n_species)
    # ------------------------------------------------------------
    def omega_from_RR(self, RR):
        ncells = RR.shape[0]
        n_species = len(self.species)

        w = torch.zeros((ncells, n_species), device=self.device, dtype=self.dtype)

        # ---------- Reactants: consumption ----------
        # flattened reactant species indices (size = n_total_react)
        idx_r = self.react_idx              # species indices
        sto_r = self.react_sto              # stoichiometric coefficients

        # map each flattened react entry to its reaction index
        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        # expand RR to match flattened react list
        RR_exp_r = RR[:, idx_r_scatter]     # (ncells, n_total_react)

        # contribution: -nu_r * RR
        contrib_r = -sto_r[None, :] * RR_exp_r

        # scatter add on species axis
        w = w.scatter_add(1, idx_r[None, :].expand(ncells, -1), contrib_r) # not in place fonction to make it compatible with vmap and torch.compile

        # ---------- Products: production ----------
        idx_p = self.prod_idx
        sto_p = self.prod_sto


        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p

        w = w.scatter_add(1, idx_p[None, :].expand(ncells, -1), contrib_p) # not in place fonction to make it compatible with vmap and torch.compile

        return w

    def jvp(self, ctx, v_T, v_C):
        #TODO: factorize even more as jvp is used a lot and many computations does not depend on v_T and v_C and can be done only once
        """
        Jacobian-vector product for species production rates w.r.t perturbations (v_T, v_C).
        Inputs:
            T:  (ncells,) temperature
            C:  (ncells, n_species) concentrations
            v_T:(ncells,) perturbation in T
            v_C:(ncells, n_species) perturbation in C
        Returns:
            Jv_w: (ncells, n_species) = (d omega / d[T,C]) · [v_T, v_C]
        Notes:
            - Uses same eps clamping as RR().
            - Fully vectorized (no Python loops).
        """
        #print('WARNING, NO PHYSICAL VALIDATION OF THIS CLASS')
        ncells = ctx.T.shape[0]
        device = self.device
        dtype = self.dtype

        # --- Precompute kf, Kc, conc_r, conc_p (re-using RR code patterns) ---
        kf = ctx.kf  # (ncells, n_reactions)
        Kc = ctx.Kc  # (ncells, n_reactions)
        C_safe = torch.clamp(ctx.C, min=self.eps)


        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        conc_r = ctx.conc_r
        conc_p = ctx.conc_p



        # --- d kf / dT ---
        # dkf/dT = kf * ( n/T + Ea/(R*T^2) )
        T_exp = ctx.T[:, None]  # (ncells,1)
        dkf_dT = kf * (self.n[None, :] / T_exp + self.Ea[None, :] / (R * T_exp * T_exp))

        # --- dKc/dT analytic ---
        # Recompute quantities used in Kc(): G0_r, G0_p and s_r, s_p to get deltaG0 and deltaS
        # Note: we re-use the same thermo calls as Kc()


        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]


        deltaG0 = ctx.deltaG0  # (ncells, n_reactions)

        # sum s per reaction
        deltaS_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        deltaS_p = torch.zeros_like(deltaS_r)

        deltaS_r = deltaS_r.scatter_add(1, idx_r_scatter[None, :].expand(ncells, -1),
                                         sto_r_exp * ctx.s_r)
        deltaS_p = deltaS_p.scatter_add(1, idx_p_scatter[None, :].expand(ncells, -1),
                                         sto_p_exp * ctx.s_p)

        deltaS = deltaS_p - deltaS_r  # Δs

        # d ln Kc / dT = Δs/(R*T) + ΔG0/(R*T^2) - Δn / T
        dlnKc_dT = deltaS / (R * T_exp) + deltaG0 / (R * T_exp * T_exp) - self.delta_n[None, :] / T_exp
        dKc_dT = Kc * dlnKc_dT  # (ncells, n_reactions)

        # --- dRR / dT  ---
        # dRR/dT = dkf/dT * (conc_r - (1-oneWay)*conc_p/Kc) + kf * (1-oneWay) * conc_p / Kc^2 * dKc/dT
        factor = (1.0 - self.oneWay[None, :])
        dRR_dT = dkf_dT * (conc_r - factor * conc_p / Kc) + kf * (factor * conc_p / (Kc * Kc) * dKc_dT)
        # shape: (ncells, n_reactions)

        # --- Contribution to RR from temperature perturbation ---
        # dRR_dT * v_T expands to per-reaction scalar
        dRR_from_T_v = dRR_dT * v_T[:, None]  # (ncells, n_reactions)

        # --- dRR / dC · v_C  (use flattened lists, then scatter_add to reactions) ---
        # For reactants: term = kf_j * conc_r_j * (nu / C_i) * vC_i   (sum over reactant species i in reaction j)
        # For products: term = - kf_j * (1-oneWay_j)/Kc_j * conc_p_j * (nu / C_i) * vC_i

        # flattened per-entry conc and kf
        conc_r_flat = conc_r[:, idx_r_scatter]  # (ncells, n_total_react)
        conc_p_flat = conc_p[:, idx_p_scatter]  # (ncells, n_total_prod)

        kf_r_flat = kf[:, idx_r_scatter]  # (ncells, n_total_react)
        kf_p_flat = kf[:, idx_p_scatter]  # (ncells, n_total_prod)

        # denominators and vC for the species indices in flattened lists
        denom_r = C_safe[:, self.react_idx]  # (ncells, n_total_react)
        denom_p = C_safe[:, self.prod_idx]  # (ncells, n_total_prod)

        vC_r = v_C[:, self.react_idx]  # (ncells, n_total_react)
        vC_p = v_C[:, self.prod_idx]  # (ncells, n_total_prod)

        # per-entry contributions
        contrib_r_flat = kf_r_flat * conc_r_flat * (self.react_sto[None, :] / denom_r) * vC_r
        contrib_p_flat = - kf_p_flat * ((1.0 - self.oneWay)[None, idx_p_scatter] / Kc[:, idx_p_scatter]) \
                         * conc_p_flat * (self.prod_sto[None, :] / denom_p) * vC_p

        # scatter-add per-reaction
        dRR_from_C_v = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)

        # reactants contribution
        dRR_from_C_v = dRR_from_C_v.scatter_add(1, idx_r_scatter[None, :].expand(ncells, -1), contrib_r_flat)

        # products contribution
        dRR_from_C_v = dRR_from_C_v.scatter_add(1, idx_p_scatter[None, :].expand(ncells, -1), contrib_p_flat)

        # --- total dRR · v (per reaction) ---
        dRR_v = dRR_from_T_v + dRR_from_C_v  # (ncells, n_reactions)

        # --- map per-reaction changes to species via same stoich mapping as omega_from_RR ---
        Jv_w = self.omega_from_RR(dRR_v)  # (ncells, n_species) - linear in RR so re-use

        return Jv_w


@dataclass
class ThreeBodyReactionsContext:
    T: torch.Tensor
    C: torch.Tensor

    # kinetics
    kf: torch.Tensor
    Kc: torch.Tensor
    conc_r: torch.Tensor
    conc_p: torch.Tensor
    baseRR: torch.Tensor
    M: torch.Tensor
    RR: torch.Tensor

    # thermo
    h_r: torch.Tensor
    s_r: torch.Tensor
    h_p: torch.Tensor
    s_p: torch.Tensor
    deltaG0: torch.Tensor


class ThreeBodyReactionBatchContext:
    """
    Batch GPU-vectorized for multiple ThreeBodyReaction instances.
    Handles different-sized reactants/products with context system.
    """

    def __init__(self, reactions, device, dtype, eps=1e-300):
        self.device = device
        self.dtype = dtype
        self.eps = eps

        self.n_reactions = len(reactions)
        self.thermo = reactions[0].thermo
        self.species = reactions[0].species
        self.n_species = len(self.species)

        # ----- Flattened indices and stoichiometries -----
        self.react_idx_list = []
        self.prod_idx_list = []
        self.react_sto_list = []
        self.prod_sto_list = []
        self.reaction_offsets_r = [0]
        self.reaction_offsets_p = [0]

        oneWay = []
        A, n, Ea = [], [], []

        self.defaultEff_r = torch.tensor([float(r.defaultEff) for r in reactions], dtype=dtype, device=device)
        for r in reactions:
            self.react_idx_list.extend(r.reactants_idx)
            self.prod_idx_list.extend(r.products_idx)
            self.react_sto_list.extend(r.reactants_sto)
            self.prod_sto_list.extend(r.products_sto)

            self.reaction_offsets_r.append(self.reaction_offsets_r[-1] + len(r.reactants_idx))
            self.reaction_offsets_p.append(self.reaction_offsets_p[-1] + len(r.products_idx))

            oneWay.append(float(r.oneWay))
            A.append(r.A.item())
            n.append(r.n.item())
            Ea.append(r.Ea.item())

        # Convert to tensors
        self.react_idx = torch.tensor(self.react_idx_list, dtype=torch.long, device=device)
        self.prod_idx = torch.tensor(self.prod_idx_list, dtype=torch.long, device=device)
        self.react_sto = torch.tensor(self.react_sto_list, dtype=dtype, device=device)
        self.prod_sto = torch.tensor(self.prod_sto_list, dtype=dtype, device=device)
        self.reaction_offsets_r = torch.tensor(self.reaction_offsets_r, dtype=torch.long, device=device)
        self.reaction_offsets_p = torch.tensor(self.reaction_offsets_p, dtype=torch.long, device=device)
        self.oneWay = torch.tensor(oneWay, dtype=dtype, device=device)
        self.A = torch.tensor(A, dtype=dtype, device=device)
        self.n = torch.tensor(n, dtype=dtype, device=device)
        self.Ea = torch.tensor(Ea, dtype=dtype, device=device)
        self.delta_n = torch.tensor(
            [torch.sum(r.products_sto) - torch.sum(r.reactants_sto) for r in reactions],
            dtype=dtype, device=device
        )
        self.eff_matrix = torch.stack([r.eff for r in reactions], dim=0).to(device=device, dtype=dtype)

        counts_r = torch.diff(self.reaction_offsets_r)
        counts_p = torch.diff(self.reaction_offsets_p)

        self.idx_r_scatter = torch.arange(self.n_reactions, device=device).repeat_interleave(counts_r)
        self.idx_p_scatter = torch.arange(self.n_reactions, device=device).repeat_interleave(counts_p)

    def evaluate_context(self, T, C):
        """
        Évalue et retourne le contexte complet de la réaction.
        """
        # Kinetics
        kf = self.kf(T)
        Kc, h_r, s_r, h_p, s_p, deltaG0 = self.Kc(T)

        # Concentrations
        conc_r, conc_p = self._compute_concentrations(T, C)

        # Base reaction rate (sans M)
        baseRR = kf * (conc_r - (1.0 - self.oneWay[None, :]) * conc_p / Kc)

        # M concentration et RR final
        M = self.M_concentration(C)
        RR = baseRR * M

        return ThreeBodyReactionsContext(
            T=T, C=C,
            kf=kf, Kc=Kc,
            conc_r=conc_r, conc_p=conc_p,
            baseRR=baseRR, M=M, RR=RR,
            h_r=h_r, s_r=s_r,
            h_p=h_p, s_p=s_p,
            deltaG0=deltaG0
        )

    def _compute_concentrations(self, T, C):
        """
        Calcule conc_r et conc_p à partir de C.
        IMPORTANT: retourne les valeurs exponentiées (pas les logs).
        """
        ncells = T.shape[0]

        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])
        logC_p = torch.log(C_safe[:, self.prod_idx])

        conc_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        conc_p = torch.zeros_like(conc_r)

        conc_r = conc_r.scatter_add(1, self.idx_r_scatter[None, :].expand(ncells, -1),
                                    logC_r * self.react_sto[None, :])
        conc_p = conc_p.scatter_add(1, self.idx_p_scatter[None, :].expand(ncells, -1),
                                    logC_p * self.prod_sto[None, :])

        # CRITICAL: appliquer l'exponentielle avant de retourner
        conc_r = torch.exp(conc_r)
        conc_p = torch.exp(conc_p)

        return (conc_r, conc_p)

    # ----- Arrhenius -----
    def kf(self, T):
        T_exp = T[:, None]
        return self.A[None, :] * (T_exp ** self.n[None, :]) * torch.exp(-self.Ea[None, :] / (R * T_exp))

    # ----- Equilibrium Kc -----
    def Kc(self, T):
        ncells = T.shape[0]

        h_r = self.thermo.h_species(T, self.react_idx)
        s_r = self.thermo.s_species_Pref(T, self.react_idx)
        h_p = self.thermo.h_species(T, self.prod_idx)
        s_p = self.thermo.s_species_Pref(T, self.prod_idx)

        G0_r = h_r - T[:, None] * s_r
        G0_p = h_p - T[:, None] * s_p

        deltaG0_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        deltaG0_p = torch.zeros_like(deltaG0_r)

        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaG0_r = deltaG0_r.scatter_add(1, idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r)
        deltaG0_p = deltaG0_p.scatter_add(1, idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p)

        deltaG0 = deltaG0_p - deltaG0_r
        Kp = torch.exp(-deltaG0 / (R * T[:, None]))
        Kc = Kp * (Pref / (R * T[:, None])) ** self.delta_n[None, :]

        return (Kc, h_r, s_r, h_p, s_p, deltaG0)

    # ----- [M] concentration -----
    def M_concentration(self, C):
        """
        C: (ncells, nspecies)
        Retour: M: (ncells, n_reactions)
        """
        return torch.matmul(C, self.eff_matrix.t())

    # ------------------------------------------------------------
    # Compute species production rates omega from RR
    # ------------------------------------------------------------
    def omega_from_RR(self, RR):
        ncells = RR.shape[0]
        n_species = len(self.species)

        w = torch.zeros((ncells, n_species), device=self.device, dtype=self.dtype)

        idx_r = self.react_idx
        sto_r = self.react_sto
        idx_r_scatter = self.idx_r_scatter

        RR_exp_r = RR[:, idx_r_scatter]
        contrib_r = -sto_r[None, :] * RR_exp_r

        w = w.scatter_add(1, idx_r[None, :].expand(ncells, -1), contrib_r)

        idx_p = self.prod_idx
        sto_p = self.prod_sto
        idx_p_scatter = self.idx_p_scatter

        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p

        w = w.scatter_add(1, idx_p[None, :].expand(ncells, -1), contrib_p)

        return w

    def jvp(self, ctx, v_T, v_C):
        #TODO: factorize even more as jvp is used a lot and many computations does not depend on v_T and v_C and can be done only once

        """
        Jacobian-vector product utilisant le contexte pré-calculé.
        """
        ncells = ctx.T.shape[0]
        device = self.device
        dtype = self.dtype

        # Récupérer les valeurs du contexte
        kf = ctx.kf
        Kc = ctx.Kc
        conc_r = ctx.conc_r  # déjà exponentié
        conc_p = ctx.conc_p  # déjà exponentié
        baseRR = ctx.baseRR
        M = ctx.M

        C_safe = torch.clamp(ctx.C, min=self.eps)

        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        # --- d kf / dT ---
        T_exp = ctx.T[:, None]
        dkf_dT = kf * (self.n[None, :] / T_exp + self.Ea[None, :] / (R * T_exp * T_exp))

        # --- dKc/dT analytic ---
        deltaG0 = ctx.deltaG0

        # sum s per reaction
        deltaS_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        deltaS_p = torch.zeros_like(deltaS_r)

        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaS_r = deltaS_r.scatter_add(1, idx_r_scatter[None, :].expand(ncells, -1),
                                        sto_r_exp * ctx.s_r)
        deltaS_p = deltaS_p.scatter_add(1, idx_p_scatter[None, :].expand(ncells, -1),
                                        sto_p_exp * ctx.s_p)

        deltaS = deltaS_p - deltaS_r

        dlnKc_dT = deltaS / (R * T_exp) + deltaG0 / (R * T_exp * T_exp) - self.delta_n[None, :] / T_exp
        dKc_dT = Kc * dlnKc_dT

        # --- d(baseRR) / dT  ---
        factor = (1.0 - self.oneWay[None, :])
        dbaseRR_dT = dkf_dT * (conc_r - factor * conc_p / Kc) + kf * (factor * conc_p / (Kc * Kc) * dKc_dT)

        # Contribution from temperature: d(baseRR)/dT * v_T * M
        dRR_from_T_v = dbaseRR_dT * v_T[:, None] * M

        # --- d(baseRR) / dC · v_C  ---
        conc_r_flat = conc_r[:, idx_r_scatter]
        conc_p_flat = conc_p[:, idx_p_scatter]

        kf_r_flat = kf[:, idx_r_scatter]
        kf_p_flat = kf[:, idx_p_scatter]

        denom_r = C_safe[:, self.react_idx]
        denom_p = C_safe[:, self.prod_idx]

        vC_r = v_C[:, self.react_idx]
        vC_p = v_C[:, self.prod_idx]

        contrib_r_flat = kf_r_flat * conc_r_flat * (self.react_sto[None, :] / denom_r) * vC_r
        contrib_p_flat = - kf_p_flat * ((1.0 - self.oneWay)[None, idx_p_scatter] / Kc[:, idx_p_scatter]) \
                         * conc_p_flat * (self.prod_sto[None, :] / denom_p) * vC_p

        dbaseRR_from_C_v = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)

        dbaseRR_from_C_v = dbaseRR_from_C_v.scatter_add(1, idx_r_scatter[None, :].expand(ncells, -1), contrib_r_flat)
        dbaseRR_from_C_v = dbaseRR_from_C_v.scatter_add(1, idx_p_scatter[None, :].expand(ncells, -1), contrib_p_flat)

        # --- dM/dC · v_C ---
        dM_from_C_v = torch.matmul(v_C, self.eff_matrix.t())

        # Total contribution from C: (d baseRR / dC · vC) * M  + baseRR * dM/dC · vC
        dRR_from_C_v = dbaseRR_from_C_v * M + baseRR * dM_from_C_v

        # --- total dRR · v ---
        dRR_v = dRR_from_T_v + dRR_from_C_v

        # --- map to species ---
        Jv_w = self.omega_from_RR(dRR_v)

        return Jv_w


@dataclass
class FalloffReactionContext:
    T: torch.Tensor
    C: torch.Tensor

    # kinetics
    k0: torch.Tensor
    kinf: torch.Tensor
    M: torch.Tensor
    Pr: torch.Tensor
    F: torch.Tensor
    k_base: torch.Tensor
    kf: torch.Tensor
    Kc: torch.Tensor

    # concentrations
    conc_r: torch.Tensor
    conc_p: torch.Tensor
    RR: torch.Tensor

    # thermo
    h_r: torch.Tensor
    s_r: torch.Tensor
    h_p: torch.Tensor
    s_p: torch.Tensor
    deltaG0: torch.Tensor

    # falloff-specific (pour Troe)
    Fcent: torch.Tensor
    logFcent: torch.Tensor
    C_par: torch.Tensor
    N_par: torch.Tensor
    f1: torch.Tensor
    exponent: torch.Tensor


class FalloffReactionBatchContext:
    """
    Batch vectorisé GPU pour plusieurs FalloffReaction (Lindemann / Troe).
    Version avec contexte pour optimiser les calculs de dérivées.
    """

    def __init__(self, reactions, device, dtype, eps=1e-300):
        self.device = device
        self.dtype = dtype
        self.eps = float(eps)

        self.n_reactions = len(reactions)
        self.thermo = reactions[0].thermo
        self.species = reactions[0].species
        self.n_species = len(self.species)

        # ----- Flattened indices and stoichiometries -----
        self.react_idx_list = []
        self.prod_idx_list = []
        self.react_sto_list = []
        self.prod_sto_list = []
        self.reaction_offsets_r = [0]
        self.reaction_offsets_p = [0]

        oneWay = []
        A_k0 = []
        b_k0 = []
        Ea_k0 = []
        A_kinf = []
        b_kinf = []
        Ea_kinf = []
        troe_A = []
        troe_T1 = []
        troe_T2 = []
        troe_T3 = []
        falloff_types = []
        eff_rows = []
        delta_n_list = []

        for r in reactions:
            self.react_idx_list.extend(r.reactants_idx)
            self.prod_idx_list.extend(r.products_idx)
            self.react_sto_list.extend(r.reactants_sto)
            self.prod_sto_list.extend(r.products_sto)

            self.reaction_offsets_r.append(self.reaction_offsets_r[-1] + len(r.reactants_idx))
            self.reaction_offsets_p.append(self.reaction_offsets_p[-1] + len(r.products_idx))

            oneWay.append(float(r.oneWay))
            A_k0.append(float(r.A_k0))
            b_k0.append(float(r.b_k0))
            Ea_k0.append(float(r.Ea_k0))
            A_kinf.append(float(r.A_kinf))
            b_kinf.append(float(r.b_kinf))
            Ea_kinf.append(float(r.Ea_kinf))

            ft = getattr(r, 'falloffType', 'lindemann')
            falloff_types.append(str(ft).lower())
            p = getattr(r, 'falloffParams', {}) or {}
            troe_A.append(float(p.get('A', 0.0)))
            troe_T1.append(float(p.get('T1', 0.0)))
            troe_T2.append(float(p.get('T2', 0.0)))
            troe_T3.append(float(p.get('T3', 0.0)))

            eff_rows.append(r.eff.to(device=device, dtype=dtype))
            delta_n_list.append(float(r.delta_n))

        # Convert to tensors
        self.react_idx = torch.tensor(self.react_idx_list, dtype=torch.long, device=device)
        self.prod_idx = torch.tensor(self.prod_idx_list, dtype=torch.long, device=device)
        self.react_sto = torch.tensor(self.react_sto_list, dtype=dtype, device=device)
        self.prod_sto = torch.tensor(self.prod_sto_list, dtype=dtype, device=device)
        self.reaction_offsets_r = torch.tensor(self.reaction_offsets_r, dtype=torch.long, device=device)
        self.reaction_offsets_p = torch.tensor(self.reaction_offsets_p, dtype=torch.long, device=device)
        self.oneWay = torch.tensor(oneWay, dtype=dtype, device=device)

        self.A_k0 = torch.tensor(A_k0, dtype=dtype, device=device)
        self.b_k0 = torch.tensor(b_k0, dtype=dtype, device=device)
        self.Ea_k0 = torch.tensor(Ea_k0, dtype=dtype, device=device)
        self.A_kinf = torch.tensor(A_kinf, dtype=dtype, device=device)
        self.b_kinf = torch.tensor(b_kinf, dtype=dtype, device=device)
        self.Ea_kinf = torch.tensor(Ea_kinf, dtype=dtype, device=device)

        self.troe_A = torch.tensor(troe_A, dtype=dtype, device=device)
        self.troe_T1 = torch.tensor(troe_T1, dtype=dtype, device=device)
        self.troe_T2 = torch.tensor(troe_T2, dtype=dtype, device=device)
        self.troe_T3 = torch.tensor(troe_T3, dtype=dtype, device=device)
        self.falloff_types = falloff_types

        self.eff_matrix = torch.stack(eff_rows, dim=0).to(device=device, dtype=dtype)
        self.delta_n = torch.tensor(delta_n_list, dtype=dtype, device=device)

        counts_r = torch.diff(self.reaction_offsets_r)
        counts_p = torch.diff(self.reaction_offsets_p)

        self.idx_r_scatter = torch.arange(self.n_reactions, device=device).repeat_interleave(counts_r)
        self.idx_p_scatter = torch.arange(self.n_reactions, device=device).repeat_interleave(counts_p)

    def evaluate_context(self, T, C):
        """
        Évalue et retourne le contexte complet de la réaction falloff.
        """
        ncells = T.shape[0]

        # Kinetics
        k0 = self._k0_T(T)
        kinf = self._kinf_T(T)
        M = self.M_concentration(C)

        # Pr and F
        kinf_safe = torch.clamp(kinf, min=self.eps)
        Pr = (k0 * M) / kinf_safe

        # Compute Fcent and related values for Troe
        Fcent, logFcent, C_par, N_par, f1, exponent = self._compute_troe_internals(T, Pr)

        # F matrix
        F = self._F_matrix_with_internals(Pr, Fcent, logFcent, C_par, N_par, f1, exponent)

        # k_base and kf
        Pr_term = Pr / (1.0 + Pr)
        k_base = kinf_safe * Pr_term
        kf = k_base * F

        # Kc
        Kc, h_r, s_r, h_p, s_p, deltaG0 = self.Kc(T)

        # Concentrations
        conc_r, conc_p = self._compute_concentrations(T, C)

        # RR
        factor = (1.0 - self.oneWay[None, :])
        conc_expr = (conc_r - factor * conc_p / Kc)
        RR = kf * conc_expr

        return FalloffReactionContext(
            T=T, C=C,
            k0=k0, kinf=kinf, M=M, Pr=Pr, F=F,
            k_base=k_base, kf=kf, Kc=Kc,
            conc_r=conc_r, conc_p=conc_p, RR=RR,
            h_r=h_r, s_r=s_r, h_p=h_p, s_p=s_p,
            deltaG0=deltaG0,
            Fcent=Fcent, logFcent=logFcent,
            C_par=C_par, N_par=N_par,
            f1=f1, exponent=exponent
        )

    def _compute_concentrations(self, T, C):
        """
        Calcule conc_r et conc_p à partir de C.
        IMPORTANT: retourne les valeurs exponentiées (pas les logs).
        """
        ncells = T.shape[0]

        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])
        logC_p = torch.log(C_safe[:, self.prod_idx])

        conc_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        conc_p = torch.zeros_like(conc_r)

        conc_r = conc_r.scatter_add(1, self.idx_r_scatter[None, :].expand(ncells, -1),
                                    logC_r * self.react_sto[None, :])
        conc_p = conc_p.scatter_add(1, self.idx_p_scatter[None, :].expand(ncells, -1),
                                    logC_p * self.prod_sto[None, :])

        # CRITICAL: appliquer l'exponentielle avant de retourner
        conc_r = torch.exp(conc_r)
        conc_p = torch.exp(conc_p)

        return (conc_r, conc_p)

    def _k0_T(self, T):
        """k0(T) = A_k0 * T^b_k0 * exp(-Ea_k0/(R*T))"""
        T_exp = T[:, None]
        return self.A_k0[None, :] * (T_exp ** self.b_k0[None, :]) * torch.exp(-self.Ea_k0[None, :] / (R * T_exp))

    def _kinf_T(self, T):
        """kinf(T) = A_kinf * T^b_kinf * exp(-Ea_kinf/(R*T))"""
        T_exp = T[:, None]
        return self.A_kinf[None, :] * (T_exp ** self.b_kinf[None, :]) * torch.exp(-self.Ea_kinf[None, :] / (R * T_exp))

    def M_concentration(self, C):
        """M = C @ eff_matrix.T"""
        return torch.matmul(C, self.eff_matrix.t())

    def _compute_troe_internals(self, T, Pr):
        """
        Calcule tous les intermédiaires Troe : Fcent, logFcent, C_par, N_par, f1, exponent
        Retourne des tensors (ncells, n_reactions)
        """
        ncells = T.shape[0]
        T_exp = T[:, None]

        # Expand parameters
        A = self.troe_A[None, :].expand(ncells, -1)
        T1 = self.troe_T1[None, :].expand(ncells, -1)
        T2 = self.troe_T2[None, :].expand(ncells, -1)
        T3 = self.troe_T3[None, :].expand(ncells, -1)

        # Masks
        mask_T1 = T1 > self.eps
        mask_T3 = T3 > self.eps
        mask_T2 = T2 > self.eps

        T1_safe = torch.where(mask_T1, T1, torch.ones_like(T1))
        T3_safe = torch.where(mask_T3, T3, torch.ones_like(T3))
        T2_safe = torch.where(mask_T2, T2, torch.ones_like(T2))

        # Fcent components
        comp1 = torch.zeros((ncells, self.n_reactions), dtype=self.dtype, device=self.device)
        comp2 = torch.zeros_like(comp1)
        comp3 = torch.zeros_like(comp1)

        if mask_T1.any():
            comp1 = (A * torch.exp(-T_exp / T1_safe)) * mask_T1.to(self.dtype)
        if mask_T3.any():
            comp2 = ((1.0 - A) * torch.exp(-T_exp / T3_safe)) * mask_T3.to(self.dtype)
        if mask_T2.any():
            comp3 = torch.exp(-T2_safe / T_exp) * mask_T2.to(self.dtype)

        Fcent = comp1 + comp2 + comp3
        Fcent = torch.clamp(Fcent, min=self.eps)

        logFcent = torch.log10(Fcent)
        C_par = -0.4 - 0.67 * logFcent
        N_par = 0.75 - 1.27 * logFcent

        Pr_safe = torch.clamp(Pr, min=self.eps)
        logPr = torch.log10(Pr_safe)

        numerator = (logPr + C_par)
        denominator = (N_par - 0.14 * (logPr + C_par))

        # Protect denominator
        denom_abs = torch.abs(denominator)
        denom_sign = torch.sign(denominator)
        denom_sign = torch.where(denom_sign == 0.0, torch.ones_like(denom_sign), denom_sign)
        denom_safe = torch.where(denom_abs < self.eps, denom_sign * self.eps, denominator)

        f1 = numerator / denom_safe
        f1_sq = f1 * f1
        exponent = logFcent / (1.0 + f1_sq)

        return Fcent, logFcent, C_par, N_par, f1, exponent

    def _F_matrix_with_internals(self, Pr, Fcent, logFcent, C_par, N_par, f1, exponent):
        """
        Construit la matrice F en utilisant les valeurs intermédiaires pré-calculées.
        """
        ncells = Pr.shape[0]
        F = torch.ones((ncells, self.n_reactions), dtype=self.dtype, device=self.device)

        troe_idx = [i for i, t in enumerate(self.falloff_types) if t == 'troe']
        if len(troe_idx) > 0:
            F_troe = torch.pow(10.0, exponent)
            mask = torch.zeros((self.n_reactions,), dtype=torch.bool, device=self.device)
            mask[torch.tensor(troe_idx, device=self.device, dtype=torch.long)] = True
            F[:, mask] = F_troe[:, mask]

        return F

    def Kc(self, T):
        """Calcule Kc et retourne aussi les valeurs thermodynamiques."""
        ncells = T.shape[0]

        h_r = self.thermo.h_species(T, self.react_idx)
        s_r = self.thermo.s_species_Pref(T, self.react_idx)
        h_p = self.thermo.h_species(T, self.prod_idx)
        s_p = self.thermo.s_species_Pref(T, self.prod_idx)

        G0_r = h_r - T[:, None] * s_r
        G0_p = h_p - T[:, None] * s_p

        deltaG0_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        deltaG0_p = torch.zeros_like(deltaG0_r)

        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaG0_r = deltaG0_r.scatter_add(1, self.idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r)
        deltaG0_p = deltaG0_p.scatter_add(1, self.idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p)

        deltaG0 = deltaG0_p - deltaG0_r
        Kp = torch.exp(-deltaG0 / (R * T[:, None]))
        Kc = Kp * (Pref / (R * T[:, None])) ** self.delta_n[None, :]

        return (Kc, h_r, s_r, h_p, s_p, deltaG0)

    def omega_from_RR(self, RR):
        """Convertit RR en taux de production d'espèces."""
        ncells = RR.shape[0]
        w = torch.zeros((ncells, self.n_species), device=self.device, dtype=self.dtype)

        idx_r = self.react_idx
        sto_r = self.react_sto
        idx_r_scatter = self.idx_r_scatter

        RR_exp_r = RR[:, idx_r_scatter]
        contrib_r = -sto_r[None, :] * RR_exp_r
        w = w.scatter_add(1, idx_r[None, :].expand(ncells, -1), contrib_r)

        idx_p = self.prod_idx
        sto_p = self.prod_sto
        idx_p_scatter = self.idx_p_scatter

        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p
        w = w.scatter_add(1, idx_p[None, :].expand(ncells, -1), contrib_p)

        return w

    def jvp(self, ctx, v_T, v_C):
        """
        Jacobian-vector product utilisant le contexte pré-calculé.
        """
        import math

        ncells = ctx.T.shape[0]
        device = self.device
        dtype = self.dtype

        # Récupérer les valeurs du contexte
        T_exp = ctx.T[:, None]
        kf = ctx.kf
        Kc = ctx.Kc
        conc_r = ctx.conc_r  # déjà exponentié
        conc_p = ctx.conc_p  # déjà exponentié
        k0 = ctx.k0
        kinf = ctx.kinf
        M = ctx.M
        Pr = ctx.Pr
        k_base = ctx.k_base
        F = ctx.F

        C_safe = torch.clamp(ctx.C, min=self.eps)

        factor = (1.0 - self.oneWay[None, :])
        conc_expr = (conc_r - factor * conc_p / Kc)

        # --- Derivatives of k0 and kinf w.r.t T ---
        dk0_dT = k0 * (self.b_k0[None, :] / T_exp + self.Ea_k0[None, :] / (R * T_exp * T_exp))
        kinf_mask = kinf > self.eps
        dkinf_dT = kinf * (self.b_kinf[None, :] / T_exp + self.Ea_kinf[None, :] / (R * T_exp * T_exp))
        dkinf_dT = torch.where(kinf_mask, dkinf_dT, torch.zeros_like(dkinf_dT))

        # --- M derivatives ---
        dM_from_C_v = torch.matmul(v_C, self.eff_matrix.t())

        # --- Pr derivatives ---
        kinf_safe = torch.clamp(kinf, min=self.eps)
        pr_mask = (Pr > self.eps) & (M > self.eps)

        dPr_dT = (dk0_dT * M - k0 * M * dkinf_dT / kinf_safe) / kinf_safe
        dPr_dT = torch.where(pr_mask, dPr_dT, torch.zeros_like(dPr_dT))

        dPr_dM = k0 / kinf_safe
        dPr_from_C_v = dPr_dM * dM_from_C_v
        dPr_from_C_v = torch.where(pr_mask, dPr_from_C_v, torch.zeros_like(dPr_from_C_v))

        # --- Fcent derivatives (from context) ---
        Fcent = ctx.Fcent
        logFcent = ctx.logFcent
        C_par = ctx.C_par
        N_par = ctx.N_par
        f1 = ctx.f1
        exponent = ctx.exponent

        # Compute dFcent/dT
        nc = ncells
        A = self.troe_A[None, :].expand(nc, -1)
        T1 = self.troe_T1[None, :].expand(nc, -1)
        T2 = self.troe_T2[None, :].expand(nc, -1)
        T3 = self.troe_T3[None, :].expand(nc, -1)

        mask_T1 = T1 > self.eps
        mask_T3 = T3 > self.eps
        mask_T2 = T2 > self.eps

        T1_safe = torch.where(mask_T1, T1, torch.ones_like(T1))
        T3_safe = torch.where(mask_T3, T3, torch.ones_like(T3))
        T2_safe = torch.where(mask_T2, T2, torch.ones_like(T2))

        dcomp1_dT = torch.zeros_like(Fcent)
        dcomp2_dT = torch.zeros_like(Fcent)
        dcomp3_dT = torch.zeros_like(Fcent)

        if mask_T1.any():
            dcomp1_dT = (A * (-1.0 / T1_safe) * torch.exp(-T_exp / T1_safe)) * mask_T1.to(dtype)
        if mask_T3.any():
            dcomp2_dT = ((1.0 - A) * (-1.0 / T3_safe) * torch.exp(-T_exp / T3_safe)) * mask_T3.to(dtype)
        if mask_T2.any():
            dcomp3_dT = (torch.exp(-T2_safe / T_exp) * (T2_safe / (T_exp * T_exp))) * mask_T2.to(dtype)

        dFcent_dT = dcomp1_dT + dcomp2_dT + dcomp3_dT

        inv_ln10 = 1.0 / math.log(10.0)
        fcent_mask = Fcent > self.eps
        dlogFcent_dT = torch.where(fcent_mask, (dFcent_dT / Fcent) * inv_ln10, torch.zeros_like(dFcent_dT))

        dCpar_dT = -0.67 * dlogFcent_dT
        dNpar_dT = -1.27 * dlogFcent_dT

        Pr_safe = torch.clamp(Pr, min=self.eps)
        logPr = torch.log10(Pr_safe)
        dlogPr_dPr = torch.where(pr_mask, 1.0 / (Pr_safe * math.log(10.0)), torch.zeros_like(Pr_safe))

        # f1 derivatives
        numerator = (logPr + C_par)
        denominator = (N_par - 0.14 * (logPr + C_par))

        denom_abs = torch.abs(denominator)
        denom_sign = torch.sign(denominator)
        denom_sign = torch.where(denom_sign == 0.0, torch.ones_like(denom_sign), denom_sign)
        denom_safe = torch.where(denom_abs < self.eps, denom_sign * self.eps, denominator)
        denom_sq = denom_safe * denom_safe
        denom_mask = (denom_abs > self.eps)

        df1_dPr = (dlogPr_dPr * denom_safe + 0.14 * numerator * dlogPr_dPr) / denom_sq
        df1_dPr = torch.where(denom_mask & pr_mask, df1_dPr, torch.zeros_like(df1_dPr))

        numerator_T = dCpar_dT
        denominator_T = dNpar_dT - 0.14 * dCpar_dT
        df1_dT = (numerator_T * denom_safe - numerator * denominator_T) / denom_sq
        df1_dT = torch.where(denom_mask, df1_dT, torch.zeros_like(df1_dT))

        # exponent derivatives
        f1_sq = f1 * f1
        denom_f = (1.0 + f1_sq)
        denom_f_sq = denom_f * denom_f

        dexponent_dlogFcent = 1.0 / denom_f
        dexponent_df1 = -2.0 * f1 * logFcent / denom_f_sq

        dexponent_dT_direct = dexponent_dlogFcent * dlogFcent_dT + dexponent_df1 * df1_dT
        dexponent_dPr = dexponent_df1 * df1_dPr

        # F derivatives
        F_troe = torch.pow(10.0, exponent)
        ln10 = math.log(10.0)

        dF_dT_direct = F_troe * ln10 * dexponent_dT_direct
        dF_dPr = F_troe * ln10 * dexponent_dPr

        # Build full matrices
        dF_dT_direct_full = torch.zeros((nc, self.n_reactions), dtype=dtype, device=device)
        dF_dPr_full = torch.zeros_like(dF_dT_direct_full)

        troe_idx = [i for i, t in enumerate(self.falloff_types) if t == 'troe']
        if len(troe_idx) > 0:
            mask = torch.zeros((self.n_reactions,), dtype=torch.bool, device=device)
            mask[torch.tensor(troe_idx, device=device, dtype=torch.long)] = True
            dF_dT_direct_full[:, mask] = dF_dT_direct[:, mask]
            dF_dPr_full[:, mask] = dF_dPr[:, mask]

        # --- k_base derivatives ---
        Pr_term = Pr / (1.0 + Pr)
        dPr_term_dPr = 1.0 / ((1.0 + Pr) * (1.0 + Pr))

        dkbase_dT = dkinf_dT * Pr_term + kinf_safe * dPr_term_dPr * dPr_dT
        dkbase_from_C_v = kinf_safe * dPr_term_dPr * dPr_from_C_v

        # --- kf derivatives ---
        dF_dT_total = dF_dT_direct_full + dF_dPr_full * dPr_dT
        dkf_dT = dkbase_dT * F + k_base * dF_dT_total

        dF_from_C_v = dF_dPr_full * dPr_from_C_v
        dkf_from_C_v = dkbase_from_C_v * F + k_base * dF_from_C_v

        # --- dKc/dT ---
        deltaG0 = ctx.deltaG0

        deltaS_r = torch.zeros((nc, self.n_reactions), device=device, dtype=dtype)
        deltaS_p = torch.zeros_like(deltaS_r)

        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter
        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaS_r = deltaS_r.scatter_add(1, idx_r_scatter[None, :].expand(nc, -1), sto_r_exp * ctx.s_r)
        deltaS_p = deltaS_p.scatter_add(1, idx_p_scatter[None, :].expand(nc, -1), sto_p_exp * ctx.s_p)

        deltaS = deltaS_p - deltaS_r

        dlnKc_dT = deltaS / (R * T_exp) + deltaG0 / (R * T_exp * T_exp) - self.delta_n[None, :] / T_exp
        dKc_dT = Kc * dlnKc_dT

        # --- dRR/dT ---
        dRR_dT = dkf_dT * conc_expr + kf * (factor * conc_p / (Kc * Kc) * dKc_dT)
        dRR_from_T_v = dRR_dT * v_T[:, None]

        # --- dRR/dC · v ---
        conc_r_flat = conc_r[:, idx_r_scatter]
        conc_p_flat = conc_p[:, idx_p_scatter]
        kf_r_flat = kf[:, idx_r_scatter]
        kf_p_flat = kf[:, idx_p_scatter]

        denom_r = C_safe[:, self.react_idx]
        denom_p = C_safe[:, self.prod_idx]
        vC_r = v_C[:, self.react_idx]
        vC_p = v_C[:, self.prod_idx]

        contrib_r_flat = kf_r_flat * conc_r_flat * (self.react_sto[None, :] / denom_r) * vC_r
        contrib_p_flat = - kf_p_flat * ((1.0 - self.oneWay)[None, idx_p_scatter] / Kc[:, idx_p_scatter]) \
                         * conc_p_flat * (self.prod_sto[None, :] / denom_p) * vC_p

        dbase_from_C_v = torch.zeros((nc, self.n_reactions), device=device, dtype=dtype)
        dbase_from_C_v = dbase_from_C_v.scatter_add(1, idx_r_scatter[None, :].expand(nc, -1), contrib_r_flat)
        dbase_from_C_v = dbase_from_C_v.scatter_add(1, idx_p_scatter[None, :].expand(nc, -1), contrib_p_flat)

        # Add contribution from dkf_from_C_v multiplying concentration expression
        dRR_from_C_v = dbase_from_C_v + dkf_from_C_v * conc_expr

        # --- total dRR·v ---
        dRR_v = dRR_from_T_v + dRR_from_C_v

        # --- map to species rates
        Jv_w = self.omega_from_RR(dRR_v)
        return Jv_w



class ReactionSystemContext:
    def __init__(self, reactionsBatches):
        self.reactionsBatches = reactionsBatches
        self.device = reactionsBatches[0].device
        self.dtype = reactionsBatches[0].dtype

        self.n_species = len(reactionsBatches[0].species)
        self.contexts = [None for _ in range(len(reactionsBatches))]



    def evaluate_context(self, T, C):
        for n, b in enumerate(self.reactionsBatches):
            self.contexts[n] = b.evaluate_context(T, C)

    #TODO: change so that T and C are centralized in the reaction system instead of copied in the reaction models
    @property
    def T(self):
        return self.contexts[0].T

    @property
    def C(self):
        return self.contexts[0].C

    def jvp(self, v_T, v_C):
        ncells = self.contexts[0].T.shape[0]
        jvp_total = torch.zeros((ncells, self.n_species), device=self.device, dtype=self.dtype)

        for n, b in enumerate(self.reactionsBatches):
            jvp_total += b.jvp(self.contexts[n], v_T, v_C)                  # sum over batches

        return jvp_total

    def omega_total(self):
        """
        Compute total species production rates omega.
        Returns tensor of shape (ncells, n_species)
        """
        ncells = self.contexts[0].T.shape[0]
        w_total = torch.zeros((ncells, self.n_species), device=self.device, dtype=self.dtype)

        for n, b in enumerate(self.reactionsBatches):
            w = b.omega_from_RR(self.contexts[n].RR)       # (ncells, n_species)
            w_total += w                  # sum over batches

        return w_total



if __name__ == '__main__':
    import torch
    import time
    from torch.autograd.functional import jvp
    import kineticsGPU.torch.thermo as yaml_thermo
    from kineticsGPU.torch.reactionParser import parseReactions, build_reactions

    yaml_file = "../gri30.yaml"
    dtype = torch.float64
    device = 'cpu'
    y = yaml_thermo.yamlParser(yaml_file)
    vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device)
    reactions = parseReactions(yaml_file)

    elem_reacs = []
    threebody_reacs = []
    falloff_reacs = []


    r_e, r_tb, r_f, system = build_reactions(vec_th.species, reactions, vec_th, device, dtype)



    # ============================================================
    # CONFIG
    # ============================================================
    ncells = 1000  # nombre de cellules pour le test
    #model = ElementaryReactionBatchContext(r_e, device=device, dtype=dtype)
    #model = ThreeBodyReactionBatchContext(r_tb, device=device, dtype=dtype)
    model = FalloffReactionBatchContext(r_f, device=device, dtype=dtype)

    nspecies = len(model.species) # OU mettre une valeur si tu veux tester sans initialiser un vrai modèle
    torch.set_default_dtype(dtype)


    T = torch.rand(ncells, device=device, dtype=dtype) * 1500 + 300
    C = torch.rand(ncells, nspecies, device=device, dtype=dtype) * 1e-2 + 1e-8

    v = torch.randn(ncells, nspecies+1, device=device, dtype=dtype)
    """
    print(80* '=')
    print('elem')
    print(80* '=')

    for reac in r_e:
        reac.compare_jvp(T, C, v[:,0], v[:,1:])
    print(80* '=')
    print('threebody')
    print(80* '=')
    for reac in r_tb:
        reac.compare_jvp(T, C, v[:,0], v[:,1:])
    print(80* '=')
    print('falloff')
    print(80* '=')
    for reac in r_f:
        reac.compare_jvp(T, C, v[:,0], v[:,1:])
    """
    # ============================================================
    # UTIL: fonction wrapper RR(T,C) → omega(T,C)
    # ============================================================
    def omega_fn(TC_flat):

        T = TC_flat[:ncells]  # (ncells,)
        C = TC_flat[ncells:].reshape(ncells, nspecies)  # (ncells, nspecies)

        # forward RR puis omega
        ctx = model.evaluate_context(T, C)
        w = model.omega_from_RR(ctx.RR)
        #RR = model.RR(T, C)
        #w = model.omega_from_RR(RR)
        return w.reshape(-1)


    # ============================================================
    # Génération des états tests
    # ============================================================
    #torch.manual_seed(0)

    T = torch.rand(ncells, device=device, dtype=dtype) * 1500 + 300
    C = torch.rand(ncells, nspecies, device=device, dtype=dtype) * 1e-2 + 1e-8

    vT = torch.randn(ncells, device=device, dtype=dtype)
    vC = torch.randn(ncells, nspecies, device=device, dtype=dtype)

    # ============================================================
    # APPEL JVP ANALYTIQUE
    # ============================================================

    t0 = time.time()
    ctx = model.evaluate_context(T, C)
    Jv_analytic = model.jvp(ctx, vT, vC)  # shape (ncells, nspecies)
    t_anneliseitic = time.time() - t0


    Jv_analytic_flat = Jv_analytic.reshape(-1)

    # ============================================================
    # APPEL AUTOGRAD (référence)
    # ============================================================
    # On prépare TC_flat et v_flat
    TC_flat = torch.cat([T, C.reshape(-1)], dim=0).requires_grad_(True)
    v_flat = torch.cat([vT, vC.reshape(-1)], dim=0)

    # jvp autograd
    t0 = time.time()
    _, Jv_auto = jvp(omega_fn, (TC_flat,), (v_flat,))
    t_auto = time.time() - t0
    Jv_auto = Jv_auto.detach()

    # ============================================================
    # COMPARAISON
    # ============================================================
    diff = Jv_analytic_flat - Jv_auto
    max_abs = diff.abs().max().item()
    rel_err = (diff.abs() / (Jv_auto.abs() + 1e-12)).max().item()

    print("\n================= JVP COMPARISON =================")
    print(f"Max absolute error: {max_abs:.3e}")
    print(f"Max relative error: {rel_err:.3e}")

    # Optionnel : si tu veux inspecter les plus grosses divergences
    idx = torch.argmax(diff.abs()).item()
    print(f"\nWorst entry analytic = {Jv_analytic_flat[idx].item():.6e}")
    print(f"Worst entry autograd = {Jv_auto[idx].item():.6e}")
    print(f"Difference           = {diff[idx].item():.6e}")

    print(f'\nAnalytic computation time: {t_anneliseitic:.6e} s')
    print(f'Autograd computation time {t_auto:.6e} s')
    print("===================================================\n")
