import torch
import math

#TODO: read Pref from chem file
#TODO: implement a full GPU parallelization with torch.scatter_add
R = 8.31446261815324  # J/(mol*K)
Pref = 101325.0       # Pa


class ElementaryReaction:
    """
    Torch-parallelized elementary reaction model.
    Allows per-cell temperatures and concentrations.

    Args:
        reactants_idx (Tensor): indices des espèces réactives (n_react,)
        products_idx (Tensor): indices des espèces produits (n_prod,)
        reactants_sto (Tensor): coefficients stoechiométriques des réactifs (n_react,)
        products_sto (Tensor): coefficients stoechiométriques des produits (n_prod,)
        thermo: objet contenant h(specie, T, P) et s(specie, T, P)
        species (list[str]): noms des espèces
        oneWay (bool): True si réaction irréversible
        A, n, Ea: paramètres d'Arrhenius
    """

    def __init__(self, reactants_idx, products_idx, reactants_sto, products_sto, thermo, species, oneWay, A, n, Ea, device, dtype):

        self.reactants_idx = torch.tensor(reactants_idx, device=device, dtype=torch.int)
        self.products_idx = torch.tensor(products_idx, device=device, dtype=torch.int)
        self.reactants_sto = torch.tensor(reactants_sto, device=device, dtype=dtype)
        self.products_sto = torch.tensor(products_sto, device=device, dtype=dtype)
        self.delta_n = torch.sum(self.products_sto) - torch.sum(self.reactants_sto) # should include phases to be exact

        self.thermo = thermo
        self.species = species
        self.oneWay = oneWay
        self.A = torch.tensor(A, device=device, dtype=dtype)
        self.n = torch.tensor(n, device=device, dtype=dtype)
        self.Ea = torch.tensor(Ea, device=device, dtype=dtype)

        # vecteur stoechiométrique global
        self.nu = torch.zeros(len(species), device=device, dtype=dtype)
        self.nu[self.products_idx] = self.products_sto
        self.nu[self.reactants_idx] -= self.reactants_sto

    # ---------- cinétique de base ----------
    def kf(self, T):
        """Facteur cinétique direct (Arrhenius)."""
        return self.A * (T ** self.n) * torch.exp(-self.Ea / (R * T))

    def Kp(self, T):
        """Constante d’équilibre en pression (torch compatible)."""
        # ici, on garde une approche scalaire car le modèle thermo n’est pas vectorisé
        # mais on peut vectoriser plus tard si thermo l’est aussi.
        # Suppose que thermo renvoie torch.Tensor
        h_r = self.thermo.h_species(T, self.reactants_idx)  # (ncells, n_react)
        s_r = self.thermo.s_species_Pref(T, self.reactants_idx)
        h_p = self.thermo.h_species(T, self.products_idx)  # (ncells, n_prod)
        s_p = self.thermo.s_species_Pref(T, self.products_idx)

        G0_r = h_r - T[:, None] * s_r
        G0_p = h_p - T[:, None] * s_p

        deltaG0 = torch.sum(self.products_sto * G0_p, dim=1) - torch.sum(self.reactants_sto * G0_r, dim=1)
        return torch.exp(-deltaG0 / (R * T))

    def Kc(self, T):
        """Conversion Kp -> Kc selon delta_n, toute espèce gazeuse. sinon il faut prendre en compte la pahse dans le delta_n."""
        return self.Kp(T) * (Pref / (R * T)) ** self.delta_n

    # ---------- réaction globale (vectorisée sur les cellules) ----------
    def RR(self, T, C):
        """
        Calcule le taux de réaction pour chaque cellule.
        Args:
            T: Tensor (ncells,)
            C: Tensor (ncells, nspecies)
        Returns:
            RR: Tensor (ncells,)
        """
        kf = self.kf(T)  # (ncells,)
        # calcul du produit des concentrations pour réactifs
        conc_react = torch.prod(torch.pow(C[:, self.reactants_idx], self.reactants_sto), dim=1)

        # et pour produits
        conc_prod = torch.prod(torch.pow(C[:, self.products_idx], self.products_sto), dim=1)

        # pour un modèle où Kc ne dépend pas des phases, on le pré-calculerait une fois.
        # ici on suppose delta_n basé sur les indices (ex. toutes les espèces gazeuses)
        Kc = self.Kc(T)

        return kf * (conc_react - (1.0 - self.oneWay) / Kc * conc_prod)


    def omega(self, T, C):
        """Production molaire ω_i pour chaque espèce (ncells, nspecies)."""
        RR = self.RR(T, C)  # (ncells,)
        return RR[:, None] * self.nu[None, :]   # (ncells, nspecies)


    def jvp(self, T, C, vT, vC):
        """
        Jacobian-vector product analytique.
        Args:
            T   : (ncells,)
            C   : (ncells, nspecies)
            vT  : (ncells,)
            vC  : (ncells, nspecies)
        Returns:
            Jv_RR : (ncells,)
            Jv_omega : (ncells, nspecies)
        """

        RR = self.RR(T, C)
        kf = self.kf(T)

        # --------- dérivées pour les concentrations ---------
        # réactifs :
        C_react = C[:, self.reactants_idx]  # (ncells, n_react)
        pow_react = self.reactants_sto
        F = torch.prod(C_react**pow_react, dim=1)  # (ncells,)

        # produits :
        C_prod = C[:, self.products_idx]
        pow_prod = self.products_sto
        P = torch.prod(C_prod**pow_prod, dim=1)    # (ncells,)

        Kc = self.Kc(T)
        G = (1.0 - self.oneWay) * P / Kc

        # dF/dC_j
        dF_dC = torch.zeros_like(C)
        for k, idx in enumerate(self.reactants_idx):
            dF_dC[:, idx] = F * (pow_react[k] / C[:, idx])

        # dG/dC_j
        dG_dC = torch.zeros_like(C)
        if not self.oneWay:
            for k, idx in enumerate(self.products_idx):
                dG_dC[:, idx] = G * (pow_prod[k] / C[:, idx])

        # dérivées en C
        dRR_dC = kf[:, None] * (dF_dC - dG_dC)   # (ncells, nspecies)

        # ---------- dérivée en T ----------

        # dkf/dT
        dkf_dT = kf * (self.n / T + self.Ea / (R * T**2))

        # dKc/dT : on utilise la définition log-diff
        Kc_val = Kc
        # dérivée log(Kc) = -(1/RT)*dG0/dT - delta_n/T  (approx)
        # Pour rester robuste, utiliser autograd interne pour Kc(T)
        T_req = T.detach().clone().requires_grad_()
        Kc_auto = self.Kc(T_req)
        dKc_dT = torch.autograd.grad(Kc_auto, T_req,
                                     grad_outputs=torch.ones_like(T),
                                     create_graph=False)[0]

        dG_dT = (1.0 - self.oneWay) * (P * (-dKc_dT / (Kc_val**2)))

        dRR_dT = dkf_dT * (F - G) + kf * (0 - dG_dT)

        # ---------- Jv ----------
        Jv_RR = dRR_dT * vT + torch.sum(dRR_dC * vC, dim=1)

        # ω = RR * nu
        Jv_omega = Jv_RR[:, None] * self.nu[None, :]

        return Jv_RR, Jv_omega


    # ---------- JVP via autograd ----------
    def jvp_autograd(self, T, C, vT, vC):
        T2 = T.detach().clone().requires_grad_(True)
        C2 = C.detach().clone().requires_grad_(True)

        RR = self.RR(T2, C2)

        Jv = torch.autograd.grad(
            RR,
            (T2, C2),
            grad_outputs=torch.ones_like(RR),
            create_graph=False,
            retain_graph=False
        )

        dRR_dT, dRR_dC = Jv
        Jv_RR = dRR_dT * vT + torch.sum(dRR_dC * vC, dim=1)
        Jv_omega = Jv_RR[:, None] * self.nu[None, :]
        return Jv_RR, Jv_omega


    # ---------- comparaison ----------
    def compare_jvp(self, T, C, vT, vC):
        Jv_ana = self.jvp(T, C, vT, vC)[0]
        Jv_auto = self.jvp_autograd(T, C, vT, vC)[0]
        err = torch.max(torch.abs(Jv_ana - Jv_auto))
        print("Max error:", err.item())
        return Jv_ana, Jv_auto, err


class ElementaryReactionBatch:
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

        #self.generate_padded_indices(reactions, device, dtype)

    def generate_padded_indices(self, reactions, device, dtype):
        n_reactions = len(reactions)

        # Déterminer le maximum de réactifs et produits
        max_n_react = max(len(r.reactants_idx) for r in reactions)
        max_n_prod = max(len(r.products_idx) for r in reactions)

        # Initialiser tensors padding avec 0
        react_idx_padded = torch.zeros((n_reactions, max_n_react), dtype=torch.long, device=device)
        prod_idx_padded = torch.zeros((n_reactions, max_n_prod), dtype=torch.long, device=device)
        react_sto_padded = torch.zeros((n_reactions, max_n_react), dtype=dtype, device=device)
        prod_sto_padded = torch.zeros((n_reactions, max_n_prod), dtype=dtype, device=device)

        # Masks pour ignorer le padding
        react_mask = torch.zeros((n_reactions, max_n_react), dtype=dtype, device=device)
        prod_mask = torch.zeros((n_reactions, max_n_prod), dtype=dtype, device=device)

        # Remplir tensors et masks
        for i, r in enumerate(reactions):
            nr, np = len(r.reactants_idx), len(r.products_idx)

            react_idx_padded[i, :nr] = torch.tensor(r.reactants_idx, dtype=torch.long, device=device)  #TODO: use r.reactants_idx.clone().detach()
            react_sto_padded[i, :nr] = torch.tensor(r.reactants_sto, dtype=dtype, device=device)       #TODO: same
            react_mask[i, :nr] = 1.0

            prod_idx_padded[i, :np] = torch.tensor(r.products_idx, dtype=torch.long, device=device)#TODO: same
            prod_sto_padded[i, :np] = torch.tensor(r.products_sto, dtype=dtype, device=device)#TODO: same
            prod_mask[i, :np] = 1.0
        self.react_idx_padded = react_idx_padded
        self.prod_idx_padded = prod_idx_padded
        self.react_sto_padded = react_sto_padded
        self.prod_sto_padded = prod_sto_padded
        self.react_mask = react_mask
        self.prod_mask = prod_mask

    # ----- Arrhenius -----
    def kf(self, T):
        T_exp = T[:, None]  # (ncells,1)
        return self.A[None, :] * (T_exp ** self.n[None, :]) * torch.exp(-self.Ea[None, :] / (R * T_exp))

    # ----- Equilibrium Kc -----
    def Kc(self, T):
        ncells = T.shape[0]

        # --- Reactants ---
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

        deltaG0_r = deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r) # not in place fonction to make it compatible with vmap and torch.compile
        deltaG0_p = deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p) # not in place fonction to make it compatible with vmap and torch.compile

        deltaG0 = deltaG0_p - deltaG0_r
        Kp = torch.exp(-deltaG0 / (R * T[:, None]))
        Kc = Kp * (Pref / (R * T[:, None])) ** self.delta_n[None, :]
        return Kc

    # ----- Reaction rates -----
    def RR_loop(self, T, C):
        ncells = T.shape[0]
        kf = self.kf(T)  # (ncells, n_reactions)
        Kc = self.Kc(T)  # (ncells, n_reactions)

        conc_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        conc_p = torch.zeros_like(conc_r)

        # loop over reactions, vectorized on cells
        for i in range(self.n_reactions):
            idx_r = self.react_idx[self.reaction_offsets_r[i]:self.reaction_offsets_r[i + 1]]
            sto_r = self.react_sto[self.reaction_offsets_r[i]:self.reaction_offsets_r[i + 1]]
            conc_r[:, i] = torch.prod(C[:, idx_r] ** sto_r, dim=1)

            idx_p = self.prod_idx[self.reaction_offsets_p[i]:self.reaction_offsets_p[i + 1]]
            sto_p = self.prod_sto[self.reaction_offsets_p[i]:self.reaction_offsets_p[i + 1]]
            conc_p[:, i] = torch.prod(C[:, idx_p] ** sto_p, dim=1)

        return kf * (conc_r - (1.0 - self.oneWay[None, :]) * conc_p / Kc)

    def RR_padding(self, T, C):
        ncells = T.shape[0]
        kf = self.kf(T)
        Kc = self.Kc(T)

        # C_r_padded: (ncells, n_reactions, max_n_react)
        C_r_padded = C[:, self.react_idx_padded]  # precomputed padded indices
        sto_r_padded = self.react_sto_padded[None, :, :]  # broadcast stoichiometries
        mask_r = self.react_mask[None, :, :]  # broadcast mask 1/0

        conc_r = torch.prod((C_r_padded ** sto_r_padded) * mask_r + (1 - mask_r), dim=2)

        # idem pour produits
        C_p_padded = C[:, self.prod_idx_padded]
        sto_p_padded = self.prod_sto_padded[None, :, :]
        mask_p = self.prod_mask[None, :, :]

        conc_p = torch.prod((C_p_padded ** sto_p_padded) * mask_p + (1 - mask_p), dim=2)

        return kf * (conc_r - (1.0 - self.oneWay[None, :]) * conc_p / Kc)

    ## previously RR_scatter(T, C)
    def RR(self, T, C):
        ncells = T.shape[0]
        kf = self.kf(T)
        Kc = self.Kc(T)

        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])  # (ncells, n_total_react)
        logC_p = torch.log(C_safe[:, self.prod_idx])

        conc_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        conc_p = torch.zeros_like(conc_r)

        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        conc_r = conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), logC_r * self.react_sto[None, :]) # not in place fonction to make it compatible with vmap and torch.compile
        conc_p = conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), logC_p * self.prod_sto[None, :]) # not in place fonction to make it compatible with vmap and torch.compile

        conc_r = torch.exp(conc_r)
        conc_p = torch.exp(conc_p)

        return kf * (conc_r - (1.0 - self.oneWay[None, :]) * conc_p / Kc)


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
        w = w.scatter_add_(1, idx_r[None, :].expand(ncells, -1), contrib_r) # not in place fonction to make it compatible with vmap and torch.compile

        # ---------- Products: production ----------
        idx_p = self.prod_idx
        sto_p = self.prod_sto


        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p

        w = w.scatter_add_(1, idx_p[None, :].expand(ncells, -1), contrib_p) # not in place fonction to make it compatible with vmap and torch.compile

        return w

    def jvp(self, T, C, v_T, v_C):
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
        ncells = T.shape[0]
        device = self.device
        dtype = self.dtype

        # --- Precompute kf, Kc, conc_r, conc_p (re-using RR code patterns) ---
        kf = self.kf(T)  # (ncells, n_reactions)
        Kc = self.Kc(T)  # (ncells, n_reactions)

        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])  # (ncells, n_total_react)
        logC_p = torch.log(C_safe[:, self.prod_idx])

        # Build per-reaction log-sum and exponentiate -> conc_r, conc_p
        conc_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        conc_p = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)

        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        conc_r = conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1),
                                     logC_r * self.react_sto[None, :])
        conc_p = conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1),
                                     logC_p * self.prod_sto[None, :])

        conc_r = torch.exp(conc_r)  # (ncells, n_reactions)
        conc_p = torch.exp(conc_p)

        # --- Reaction rate RR (for reuse in dRR/dT formula) ---
        RR = kf * (conc_r - (1.0 - self.oneWay[None, :]) * conc_p / Kc)

        # --- d kf / dT ---
        # dkf/dT = kf * ( n/T + Ea/(R*T^2) )
        T_exp = T[:, None]  # (ncells,1)
        dkf_dT = kf * (self.n[None, :] / T_exp + self.Ea[None, :] / (R * T_exp * T_exp))

        # --- dKc/dT analytic ---
        # Recompute quantities used in Kc(): G0_r, G0_p and s_r, s_p to get deltaG0 and deltaS
        # Note: we re-use the same thermo calls as Kc()
        h_r = self.thermo.h_species(T, self.react_idx)  # (ncells, n_total_react)
        s_r = self.thermo.s_species_Pref(T, self.react_idx)
        h_p = self.thermo.h_species(T, self.prod_idx)
        s_p = self.thermo.s_species_Pref(T, self.prod_idx)

        G0_r = h_r - T[:, None] * s_r
        G0_p = h_p - T[:, None] * s_p

        # Sum contributions per reaction (like in Kc)
        deltaG0_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        deltaG0_p = torch.zeros_like(deltaG0_r)

        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaG0_r = deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1),
                                           sto_r_exp * G0_r)
        deltaG0_p = deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1),
                                           sto_p_exp * G0_p)

        deltaG0 = deltaG0_p - deltaG0_r  # (ncells, n_reactions)

        # sum s per reaction
        deltaS_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        deltaS_p = torch.zeros_like(deltaS_r)

        deltaS_r = deltaS_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1),
                                         sto_r_exp * s_r)
        deltaS_p = deltaS_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1),
                                         sto_p_exp * s_p)

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
        dRR_from_C_v = dRR_from_C_v.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), contrib_r_flat)

        # products contribution
        dRR_from_C_v = dRR_from_C_v.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), contrib_p_flat)

        # --- total dRR · v (per reaction) ---
        dRR_v = dRR_from_T_v + dRR_from_C_v  # (ncells, n_reactions)

        # --- map per-reaction changes to species via same stoich mapping as omega_from_RR ---
        Jv_w = self.omega_from_RR(dRR_v)  # (ncells, n_species) - linear in RR so re-use

        return Jv_w





class ThreeBodyReaction(ElementaryReaction):
    """
    Réaction tri-moléculaire (three-body) torch-parallélisée.
    Identique à ElementaryReaction, mais le taux est multiplié
    par la concentration effective [M] tenant compte des efficacités.

    Args:
        reactants_idx, products_idx, reactants_sto, products_sto: mêmes formats que ElementaryReaction
        thermo: objet ThermoData_vectorized
        species (list[str]): noms des espèces
        oneWay (bool): True si réaction irréversible
        A, n, Ea: paramètres d’Arrhenius
        efficiencies (dict): efficacités des espèces (e.g. {'H2O': 12.0, 'O2': 1.0})
        defaultEff (float): efficacité par défaut
        device, dtype: torch.device et dtype
    """

    def __init__(self, reactants_idx, products_idx, reactants_sto, products_sto, thermo, species, oneWay, A, n, Ea, efficiencies, device, dtype, defaultEff=1.0):

        ElementaryReaction.__init__(self, reactants_idx, products_idx, reactants_sto, products_sto, thermo, species, oneWay, A, n, Ea, device, dtype)

        # --- préparation des efficacités en tenseur aligné avec l’ordre des espèces ---
        eff = torch.full((len(species),), float(defaultEff), device=device, dtype=dtype)
        for i, sp in enumerate(species):
            if sp in efficiencies:
                eff[i] = torch.tensor(efficiencies[sp], device=device, dtype=dtype)
        self.eff = eff  # (nspecies,)
        self.defaultEff = defaultEff

    def M_concentration(self, C):
        """
        Calcule la concentration effective [M] dans chaque cellule.
        Args:
            C: (ncells, nspecies)
        Returns:
            M: (ncells,)
        """
        # produit scalaire sur l’axe des espèces : somme_i (eff_i * C_i)
        return torch.sum(C * self.eff[None, :], dim=1)

    def RR(self, T, C):
        """
        Calcule le taux de réaction tri-moléculaire (par cellule).
        Args:
            T: (ncells,)
            C: (ncells, nspecies)
        Returns:
            RR: (ncells,)
        """
        baseRR = ElementaryReaction.RR(self, T, C)  # (ncells,)
        M = self.M_concentration(C)  # (ncells,)
        return baseRR * M

    def jvp(self, T, C, vT, vC):
        """
        JVP analytique pour réaction tri-moléculaire.
        Retourne (Jv_RR, Jv_omega)
        """

        # ----- RR_base et dérivées base -----
        # On appelle la reaction parent pour obtenir les dérivée de RR_base
        RR_base = ElementaryReaction.RR(self, T, C)
        Jv_base_RR, _ = ElementaryReaction.jvp(self, T, C, vT, vC)

        # ----- M et dérivée de M -----
        M = self.M_concentration(C)  # (ncells,)
        dM_dC = self.eff[None, :]    # (1, nspecies) broadcast

        # ----- dérivée en T -----
        # M ne dépend pas de T
        dRR_dT = M * (Jv_base_RR / vT) if torch.any(vT != 0) else M * 0.0

        # ----- dérivée en C -----
        # dRR/dC = eff_j * RR_base + M * dRR_base/dC_j
        # dRR_base/dC_j est obtenu via autograd local de ElementaryReaction :
        C2 = C.detach().clone().requires_grad_(True)
        RRb = ElementaryReaction.RR(self, T, C2)
        grads = torch.autograd.grad(
            RRb, C2,
            grad_outputs=torch.ones_like(RRb),
            retain_graph=False,
            create_graph=False
        )[0]  # (ncells, nspecies)
        dRR_dC = dM_dC * RR_base[:, None] + M[:, None] * grads  # (ncells, nspecies)

        # ----- JVP global -----
        Jv_RR = dRR_dT * vT + torch.sum(dRR_dC * vC, dim=1)
        Jv_omega = Jv_RR[:, None] * self.nu[None, :]

        return Jv_RR, Jv_omega


    # ---------- JVP via autograd ----------
    def jvp_autograd(self, T, C, vT, vC):
        T2 = T.detach().clone().requires_grad_(True)
        C2 = C.detach().clone().requires_grad_(True)

        RR = self.RR(T2, C2)

        dRRdT, dRRdC = torch.autograd.grad(
            RR, (T2, C2),
            grad_outputs=torch.ones_like(RR),
            retain_graph=False
        )

        Jv_RR = dRRdT * vT + torch.sum(dRRdC * vC, dim=1)
        Jv_omega = Jv_RR[:, None] * self.nu[None, :]

        return Jv_RR, Jv_omega


    # ---------- comparaison ----------
    def compare_jvp(self, T, C, vT, vC):
        Jv_ana, _ = self.jvp(T, C, vT, vC)
        Jv_auto, _ = self.jvp_autograd(T, C, vT, vC)
        err = torch.max(torch.abs(Jv_ana - Jv_auto))
        print("Max JVP error:", err.item())
        return Jv_ana, Jv_auto, err



class ThreeBodyReactionBatch:
    """
    Batch GPU-vectorized for multiple ThreeBodyReaction instances.
    Handles different-sized reactants/products, like ElementaryReactionBatch,
    and accounts for [M] concentration from species efficiencies.
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

        # efficacités alignées sur l'ordre des espèces
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
        # counts per reaction in flattened react/product lists
        counts_r = torch.diff(self.reaction_offsets_r)   # length = n_reactions
        counts_p = torch.diff(self.reaction_offsets_p)

        # idx_r_scatter: for each flattened react entry, the reaction index it belongs to
        # shape = (n_total_react_entries,)
        self.idx_r_scatter = torch.arange(self.n_reactions, device=device).repeat_interleave(counts_r)
        self.idx_p_scatter = torch.arange(self.n_reactions, device=device).repeat_interleave(counts_p)


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
        deltaG0_r = deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r) # not in place fonction to make it compatible with vmap and torch.compile
        deltaG0_p = deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p) # not in place fonction to make it compatible with vmap and torch.compile
        deltaG0 = deltaG0_p - deltaG0_r
        Kp = torch.exp(-deltaG0 / (R * T[:, None]))
        Kc = Kp * (Pref / (R * T[:, None])) ** self.delta_n[None, :]
        return Kc

    # ----- [M] concentration -----
    def M_concentration(self, C):
        """
        C: (ncells, nspecies)
        Retour: M: (ncells, n_reactions)
        """
        # produit matrice: (ncells, nspecies) @ (nspecies, n_reactions) -> (ncells, n_reactions)
        return torch.matmul(C, self.eff_matrix.t())

    # ----- Reaction rates -----
    # ----- RR scatter (full GPU vectorized, log + scatter_add) -----
    def RR(self, T, C):
        ncells = T.shape[0]
        kf = self.kf(T)
        Kc = self.Kc(T)

        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])
        logC_p = torch.log(C_safe[:, self.prod_idx])

        conc_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        conc_p = torch.zeros_like(conc_r)

        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        conc_r = conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), logC_r * self.react_sto[None, :]) # not in place fonction to make it compatible with vmap and torch.compile
        conc_p = conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), logC_p * self.prod_sto[None, :]) # not in place fonction to make it compatible with vmap and torch.compile

        conc_r = torch.exp(conc_r)
        conc_p = torch.exp(conc_p)

        baseRR = kf * (conc_r - (1.0 - self.oneWay[None, :]) * conc_p / Kc)
        M = self.M_concentration(C)
        return baseRR * M


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

        # expand RR to match flattened react list
        RR_exp_r = RR[:, idx_r_scatter]     # (ncells, n_total_react)

        # contribution: -nu_r * RR
        contrib_r = -sto_r[None, :] * RR_exp_r

        # scatter add on species axis
        w = w.scatter_add_(1, idx_r[None, :].expand(ncells, -1), contrib_r) # not in place fonction to make it compatible with vmap and torch.compile

        # ---------- Products: production ----------
        idx_p = self.prod_idx
        sto_p = self.prod_sto

        idx_p_scatter = self.idx_p_scatter

        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p

        w = w.scatter_add_(1, idx_p[None, :].expand(ncells, -1), contrib_p) # not in place fonction to make it compatible with vmap and torch.compile

        return w

    def jvp(self, T, C, v_T, v_C):
        """
        Jacobian-vector product for species production rates w.r.t perturbations (v_T, v_C)
        for three-body reaction batch.

        Inputs:
            T:   (ncells,) temperature
            C:   (ncells, n_species) concentrations
            v_T: (ncells,) perturbation in T
            v_C: (ncells, n_species) perturbation in C

        Returns:
            Jv_w: (ncells, n_species) = (d omega / d[T,C]) · [v_T, v_C]

        Notes:
            - Matches eps/clamping and vectorized style used in Elementary implementation.
            - Fully vectorized (no Python loops).
        """
        #print('WARNING, NO PHYSICAL VALIDATION OF THIS CLASS')
        ncells = T.shape[0]
        device = self.device
        dtype = self.dtype

        # --- Precompute kf, Kc, conc_r, conc_p (re-using RR code patterns) ---
        kf = self.kf(T)  # (ncells, n_reactions)
        Kc = self.Kc(T)  # (ncells, n_reactions)

        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])  # (ncells, n_total_react)
        logC_p = torch.log(C_safe[:, self.prod_idx])

        # Build per-reaction log-sum and exponentiate -> conc_r, conc_p
        conc_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        conc_p = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)

        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        conc_r = conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1),
                                     logC_r * self.react_sto[None, :])
        conc_p = conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1),
                                     logC_p * self.prod_sto[None, :])

        conc_r = torch.exp(conc_r)  # (ncells, n_reactions)
        conc_p = torch.exp(conc_p)

        # --- Reaction base rate (without [M]) ---
        baseRR = kf * (conc_r - (1.0 - self.oneWay[None, :]) * conc_p / Kc)  # (ncells, n_reactions)

        # --- M and its perturbation ---
        M = self.M_concentration(C)  # (ncells, n_reactions)
        # dM/dC · v_C = v_C @ eff_matrix.T
        dM_from_C_v = torch.matmul(v_C, self.eff_matrix.t())  # (ncells, n_reactions)

        # Full RR = baseRR * M
        RR = baseRR * M

        # --- d kf / dT ---
        T_exp = T[:, None]
        dkf_dT = kf * (self.n[None, :] / T_exp + self.Ea[None, :] / (R * T_exp * T_exp))

        # --- dKc/dT analytic ---
        # Recompute quantities used in Kc(): G0_r, G0_p and s_r, s_p to get deltaG0 and deltaS
        h_r = self.thermo.h_species(T, self.react_idx)
        s_r = self.thermo.s_species_Pref(T, self.react_idx)
        h_p = self.thermo.h_species(T, self.prod_idx)
        s_p = self.thermo.s_species_Pref(T, self.prod_idx)

        G0_r = h_r - T[:, None] * s_r
        G0_p = h_p - T[:, None] * s_p

        deltaG0_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        deltaG0_p = torch.zeros_like(deltaG0_r)

        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaG0_r = deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1),
                                           sto_r_exp * G0_r)
        deltaG0_p = deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1),
                                           sto_p_exp * G0_p)

        deltaG0 = deltaG0_p - deltaG0_r  # (ncells, n_reactions)

        # sum s per reaction
        deltaS_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        deltaS_p = torch.zeros_like(deltaS_r)

        deltaS_r = deltaS_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1),
                                         sto_r_exp * s_r)
        deltaS_p = deltaS_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1),
                                         sto_p_exp * s_p)

        deltaS = deltaS_p - deltaS_r  # Δs

        # d ln Kc / dT = Δs/(R*T) + ΔG0/(R*T^2) - Δn / T
        dlnKc_dT = deltaS / (R * T_exp) + deltaG0 / (R * T_exp * T_exp) - self.delta_n[None, :] / T_exp
        dKc_dT = Kc * dlnKc_dT  # (ncells, n_reactions)

        # --- d(baseRR) / dT  ---
        factor = (1.0 - self.oneWay[None, :])
        dbaseRR_dT = dkf_dT * (conc_r - factor * conc_p / Kc) + kf * (factor * conc_p / (Kc * Kc) * dKc_dT)
        # shape: (ncells, n_reactions)

        # Contribution to RR from temperature perturbation: d(baseRR)/dT * v_T * M
        dRR_from_T_v = dbaseRR_dT * v_T[:, None] * M

        # --- d(baseRR) / dC · v_C  (use flattened lists, then scatter_add to reactions) ---
        # For reactants: term = kf_j * conc_r_j * (nu / C_i) * vC_i   (sum over reactant species i in reaction j)
        # For products: term = - kf_j * (1-oneWay_j)/Kc_j * conc_p_j * (nu / C_i) * vC_i

        conc_r_flat = conc_r[:, idx_r_scatter]  # (ncells, n_total_react)
        conc_p_flat = conc_p[:, idx_p_scatter]  # (ncells, n_total_prod)

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

        # reactants contribution
        dbaseRR_from_C_v = dbaseRR_from_C_v.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), contrib_r_flat)

        # products contribution
        dbaseRR_from_C_v = dbaseRR_from_C_v.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), contrib_p_flat)

        # Now include effects of M: dRR_from_C_v = (d baseRR / dC · vC) * M  + baseRR * dM/dC · vC
        dRR_from_C_v = dbaseRR_from_C_v * M + baseRR * dM_from_C_v

        # --- total dRR · v (per reaction) ---
        dRR_v = dRR_from_T_v + dRR_from_C_v  # (ncells, n_reactions)

        # --- map per-reaction changes to species via same stoich mapping as omega_from_RR ---
        Jv_w = self.omega_from_RR(dRR_v)  # (ncells, n_species)

        return Jv_w


class FalloffReaction(ElementaryReaction):
    """
    Réaction de type Falloff (Lindemann / Troe), torch-vectorisée.
    Hérite de ElementaryReaction mais utilise k0(T) et kinf(T) et un
    facteur de broadening F(T,Pr).

    Args:
        reactants_idx, products_idx, reactants_sto, products_sto: mêmes formats que ElementaryReaction
        thermo: ThermoData_vectorized
        species: list of species names (order must match thermo)
        oneWay: bool
        k0_params: (A_k0, b_k0, Ea_k0)
        kinf_params: (A_kinf, b_kinf, Ea_kinf)
        efficiencies: dict {species_name: factor}
        defaultEff: float
        falloff_type: 'Lindemann' or 'Troe' (case-insensitive)
        falloff_params: dict for Troe: keys 'A', optional 'T1','T2','T3'
        device, dtype: torch device and dtype
    """

    def __init__(self, reactants_idx, products_idx, reactants_sto, products_sto,
                 thermo, species, oneWay,
                 k0_params, kinf_params, device, dtype,
                 efficiencies=None, defaultEff=1.0,
                 falloff_type='Lindemann', falloff_params=None,
                 eps=1e-300):
        # call parent with dummy Arrhenius (we override kf)
        ElementaryReaction.__init__(self, reactants_idx, products_idx, reactants_sto, products_sto,
                         thermo, species, oneWay, A=1.0, n=0.0, Ea=0.0,
                         device=device, dtype=dtype)


        self.eps = eps

        # store k0 and kinf parameters as tensors (scalars per reaction)
        A_k0, b_k0, Ea_k0 = k0_params
        A_kinf, b_kinf, Ea_kinf = kinf_params
        self.A_k0 = torch.tensor(A_k0, dtype=dtype, device=device)
        self.b_k0 = torch.tensor(b_k0, dtype=dtype, device=device)
        self.Ea_k0 = torch.tensor(Ea_k0, dtype=dtype, device=device)

        self.A_kinf = torch.tensor(A_kinf, dtype=dtype, device=device)
        self.b_kinf = torch.tensor(b_kinf, dtype=dtype, device=device)
        self.Ea_kinf = torch.tensor(Ea_kinf, dtype=dtype, device=device)

        # efficiencies -> tensor aligned with species order
        eff = torch.full((len(species),), float(defaultEff), dtype=dtype, device=device)
        if efficiencies:
            for i, sp in enumerate(species):
                if sp in efficiencies:
                    eff[i] = torch.tensor(efficiencies[sp], dtype=dtype, device=device)
        self.eff = eff  # (nspecies,)

        self.falloffType = falloff_type.lower()
        self.falloffParams = falloff_params

        # compute delta_n for conversion Kp -> Kc if needed (scalar)
        self.delta_n = torch.sum(self.products_sto) - torch.sum(self.reactants_sto)

    # ----- Arrhenius low/high (vectorized over T array) -----
    def _k0_T(self, T):
        """
        k0(T) vectorized: T shape (ncells,)
        returns shape (ncells,)
        """
        # compute with broadcasting: scalar params * T**b * exp(-Ea/(R*T))
        return self.A_k0 * (T ** self.b_k0) * torch.exp(-self.Ea_k0 / (R * T))

    def _kinf_T(self, T):
        return self.A_kinf * (T ** self.b_kinf) * torch.exp(-self.Ea_kinf / (R * T))

    # ----- [M] effective via efficiencies -----
    def M_concentration(self, C):
        """
        Calcule la concentration effective [M] dans chaque cellule.
        Args:
            C: (ncells, nspecies)
        Returns:
            M: (ncells,)
        """
        # produit scalaire sur l’axe des espèces : somme_i (eff_i * C_i)
        return torch.sum(C * self.eff[None, :], dim=1)

    # ----- Falloff factors -----
    def _F_lindemann(self, T, Pr):
        # Lindemann: F = 1
        return torch.ones_like(Pr, dtype=T.dtype, device=T.device)

    def _F_troe(self, T, Pr):
        """
        Vectorized Troe broadening factor.
        Fcent = (1-A)*exp(-T/T3) + A*exp(-T/T1) + exp(-T2/T)
        then F = 10**(logFcent / (1 + f1^2)) with f1 = (log10(Pr)+C) / (N - 0.14*(log10(Pr)+C))
        """
        params = self.falloffParams
        A = params.get('A', 0.0)
        T1 = params.get('T1', 0.0)
        T2 = params.get('T2', 0.0)
        T3 = params.get('T3', 0.0)


        # compute Fcent components with safe clamping
        Fcent = torch.zeros_like(T, dtype=T.dtype, device=T.device)

        if T1 > 0.0:
            Fcent = Fcent + (A * torch.exp(-T / T1))
        if T3 > 0.0:
            Fcent = Fcent + ((1.0 - A) * torch.exp(-T / T3))
        if T2 > 0.0:
            Fcent = Fcent + torch.exp(-T2 / T)

        # ensure positivity
        #Fcent = torch.clamp(Fcent, min=self.eps)

        logFcent = torch.log10(Fcent)  # (ncells,)
        C = -0.4 - 0.67 * logFcent
        N = 0.75 - 1.27 * logFcent

        # guard Pr for log10
        #Pr_safe = torch.clamp(Pr, min=self.eps)
        #logPr = torch.log10(Pr_safe)
        logPr = torch.log10(Pr)
        numerator = (logPr + C)
        denominator = (N - 0.14 * (logPr + C))
        # avoid division by zero -> clamp denominator
        #denominator = torch.where(denominator == 0.0, torch.full_like(denominator, self.eps), denominator)

        f1 = numerator / denominator
        f1_sq = f1 * f1

        exponent = logFcent / (1.0 + f1_sq)
        F = torch.pow(10.0, exponent)
        return F

    def _F(self, T, Pr):
        t = self.falloffType
        if t == 'troe':
            return self._F_troe(T, Pr)
        elif t == 'lindemann':
            return self._F_lindemann(T, Pr)
        else:
            raise NotImplementedError(f"Falloff type '{self.falloffType}' not implemented")

    # ----- override kf to include falloff correction -----
    def kf_falloff(self, T, C):
        """
        Return forward rate constant kf per cell: (ncells,)
        T: (ncells,)
        C: (ncells, nspecies) needed for [M]
        """
        # vectorized k0 and kinf
        k0 = self._k0_T(T)     # (ncells,)
        kinf = self._kinf_T(T) # (ncells,)
        M = self.M_concentration(C)  # (ncells,)

        # reduced pressure Pr = (k0 * [M]) / kinf
        # guard kinf > 0
        #kinf_safe = torch.clamp(kinf, min=self.eps)
        #Pr = (k0 * M) / kinf_safe
        Pr = (k0 * M) / kinf

        # base Lindemann expression: k = (k0 * M) / (1 + k0 * M / kinf) = kinf * Pr / (1+Pr)
        #k_base = kinf_safe * (Pr / (1.0 + Pr))
        k_base = kinf * (Pr / (1.0 + Pr))

        # apply falloff function F(T, Pr)
        F = self._F(T, Pr)
        kf = k_base * F
        return kf

    # ----- override RR to use vectorized kf(T,C) -----
    def RR(self, T, C):
        """
        Compute RR per cell: shape (ncells,)
        T: (ncells,)
        C: (ncells, nspecies)
        """

        # forward kf that depends on [M]
        kf_vec = self.kf_falloff(T, C)  # (ncells,)


        # calcul du produit des concentrations pour réactifs
        conc_react = torch.prod(torch.pow(C[:, self.reactants_idx], self.reactants_sto), dim=1)

        # et pour produits
        conc_prod = torch.prod(torch.pow(C[:, self.products_idx], self.products_sto), dim=1)

        # Kc conversion: use Kp computed from thermo (vectorized)
        # delta_n is scalar tensor
        Kc = self.Kc(T)  # returns (ncells,)

        # avoid division by zero
        #Kc = torch.clamp(Kc, min=self.eps)


        RR = kf_vec * (conc_react - (1.0 - self.oneWay) / Kc * conc_prod)
        return RR


class FalloffReactionBatch:
    """
    Batch vectorisé GPU pour plusieurs FalloffReaction (Lindemann / Troe).
    Gère différentes tailles de réactifs/produits via index aplatis et offsets
    (même approche que ThreeBodyReactionBatch / ElementaryReactionBatch).

    Sorties principales:
        - kf(T, C): constantes de vitesse avant (ncells, n_reactions)
        - Kc(T): constantes d'équilibre (ncells, n_reactions)
        - RR_scatter(T, C): taux de réaction (ncells, n_reactions)

    Hypothèses:
        - chaque objet FalloffReaction fourni dans `reactions` possède:
            .thermo, .species, .reactants_idx, .products_idx,
            .reactants_sto, .products_sto, .oneWay,
            .A_k0, .b_k0, .Ea_k0, .A_kinf, .b_kinf, .Ea_kinf,
            .eff (tensor aligned with species order),
            .falloffType ('troe' or 'lindemann'), .falloffParams (dict),
            .delta_n (scalar)
        - R et Pref sont disponibles globalement (même convention que le reste du code)
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

        # collect k0/kinf params per reaction (scalars)
        A_k0 = []
        b_k0 = []
        Ea_k0 = []
        A_kinf = []
        b_kinf = []
        Ea_kinf = []

        # falloff specifics (troe params): store per reaction, default zeros
        troe_A = []
        troe_T1 = []
        troe_T2 = []
        troe_T3 = []
        falloff_types = []

        # efficacités alignées sur l'ordre des espèces -> matrice (n_reactions, nspecies)
        eff_rows = []

        # delta_n per reaction
        delta_n_list = []

        for r in reactions:
            self.react_idx_list.extend(r.reactants_idx)
            self.prod_idx_list.extend(r.products_idx)
            self.react_sto_list.extend(r.reactants_sto)
            self.prod_sto_list.extend(r.products_sto)

            self.reaction_offsets_r.append(self.reaction_offsets_r[-1] + len(r.reactants_idx))
            self.reaction_offsets_p.append(self.reaction_offsets_p[-1] + len(r.products_idx))

            oneWay.append(float(r.oneWay))

            # parameters (cast to float then tensor later)
            A_k0.append(float(r.A_k0))
            b_k0.append(float(r.b_k0))
            Ea_k0.append(float(r.Ea_k0))
            A_kinf.append(float(r.A_kinf))
            b_kinf.append(float(r.b_kinf))
            Ea_kinf.append(float(r.Ea_kinf))

            # falloff params: may be None or dict
            ft = getattr(r, 'falloffType', 'lindemann')
            falloff_types.append(str(ft).lower())
            p = getattr(r, 'falloffParams', {}) or {}
            troe_A.append(float(p.get('A', 0.0)))
            troe_T1.append(float(p.get('T1', 0.0)))
            troe_T2.append(float(p.get('T2', 0.0)))
            troe_T3.append(float(p.get('T3', 0.0)))

            # efficiency row aligned with species
            eff_rows.append(r.eff.to(device=device, dtype=dtype))

            delta_n_list.append(float(r.delta_n))

        # convert to tensors
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
        # falloff_types kept as Python list of strings for branching in _F
        self.falloff_types = falloff_types

        self.eff_matrix = torch.stack(eff_rows, dim=0).to(device=device, dtype=dtype)  # (n_reactions, nspecies)

        self.delta_n = torch.tensor(delta_n_list, dtype=dtype, device=device)  # (n_reactions,)

        # counts per reaction in flattened react/product lists
        counts_r = torch.diff(self.reaction_offsets_r)   # length = n_reactions
        counts_p = torch.diff(self.reaction_offsets_p)

        # idx_r_scatter: for each flattened react entry, the reaction index it belongs to
        # shape = (n_total_react_entries,)
        self.idx_r_scatter = torch.arange(self.n_reactions, device=device).repeat_interleave(counts_r)
        self.idx_p_scatter = torch.arange(self.n_reactions, device=device).repeat_interleave(counts_p)

    # ----- k0 and kinf vectorized over T array -----
    def _k0_T(self, T):
        # T: (ncells,)
        # return (ncells, n_reactions)
        T_exp = T[:, None]  # (ncells,1)
        return self.A_k0[None, :] * (T_exp ** self.b_k0[None, :]) * torch.exp(-self.Ea_k0[None, :] / (R * T_exp))

    def _kinf_T(self, T):
        T_exp = T[:, None]
        return self.A_kinf[None, :] * (T_exp ** self.b_kinf[None, :]) * torch.exp(-self.Ea_kinf[None, :] / (R * T_exp))

    # ----- [M] concentration -----
    def M_concentration(self, C):
        """
        C: (ncells, nspecies)
        Retour: M: (ncells, n_reactions)
        """
        # produit matrice: (ncells, nspecies) @ (nspecies, n_reactions) -> (ncells, n_reactions)
        return torch.matmul(C, self.eff_matrix.t())

    # ----- Falloff factor F(T, Pr) vectorisé pour toutes les réactions -----
    def _F_troe_matrix(self, T, Pr):
        """
        T: (ncells, 1) or (ncells,)
        Pr: (ncells, n_reactions)
        Retour: F: (ncells, n_reactions)
        Les paramètres Troe peuvent être différents par réaction.
        """
        ncells = T.shape[0]
        # expand param tensors to shape (ncells, n_reactions)
        T_exp = T[:, None]
        A = self.troe_A[None, :].expand(ncells, -1)
        T1 = self.troe_T1[None, :].expand(ncells, -1)
        T2 = self.troe_T2[None, :].expand(ncells, -1)
        T3 = self.troe_T3[None, :].expand(ncells, -1)

        # compute Fcent per reaction per cell
        # avoid negative/zero by clamping if needed
        Fcent = torch.zeros_like(Pr, dtype=self.dtype, device=self.device)
        # components: (1-A)*exp(-T/T3) + A*exp(-T/T1) + exp(-T2/T)
        # handle T1/T3 zero by masking
        mask_T1 = T1 > 0.0
        mask_T3 = T3 > 0.0
        mask_T2 = T2 > 0.0

        if mask_T1.any():
            Fcent = Fcent + (A * torch.exp(-T_exp / torch.where(mask_T1, T1, torch.ones_like(T1)))) * mask_T1.to(self.dtype)
        if mask_T3.any():
            Fcent = Fcent + ((1.0 - A) * torch.exp(-T_exp / torch.where(mask_T3, T3, torch.ones_like(T3)))) * mask_T3.to(self.dtype)
        if mask_T2.any():
            Fcent = Fcent + torch.exp(-torch.where(mask_T2, T2, torch.ones_like(T2)) / T_exp) * mask_T2.to(self.dtype)

        # ensure Fcent positive
        Fcent = torch.clamp(Fcent, min=self.eps)

        logFcent = torch.log10(Fcent)
        C_par = -0.4 - 0.67 * logFcent
        N_par = 0.75 - 1.27 * logFcent

        # guard Pr for log10
        Pr_safe = torch.clamp(Pr, min=self.eps)
        logPr = torch.log10(Pr_safe)

        numerator = (logPr + C_par)
        denominator = (N_par - 0.14 * (logPr + C_par))
        denominator = torch.where(denominator == 0.0, torch.full_like(denominator, self.eps), denominator)
        f1 = numerator / denominator
        f1_sq = f1 * f1
        exponent = logFcent / (1.0 + f1_sq)
        F = torch.pow(10.0, exponent)
        return F

    def _F_matrix(self, T, Pr):
        """
        Build full F matrix depending on falloff type per reaction.
        Returns (ncells, n_reactions)
        """
        ncells = T.shape[0]
        # default ones
        F = torch.ones((ncells, self.n_reactions), dtype=self.dtype, device=self.device)

        # determine indices of troe reactions
        troe_idx = [i for i, t in enumerate(self.falloff_types) if t == 'troe']
        if len(troe_idx) > 0:
            # compute for all reactions then mask
            F_troe = self._F_troe_matrix(T, Pr)
            # mask columns corresponding to troe_idx
            mask = torch.zeros((self.n_reactions,), dtype=torch.bool, device=self.device)
            mask[torch.tensor(troe_idx, device=self.device, dtype=torch.long)] = True
            F[:, mask] = F_troe[:, mask]

        # lindemann (and defaults) remain ones
        return F

    # ----- forward kf for batch (ncells, n_reactions) -----
    def kf(self, T, C):
        """
        Compute forward rate constants for each cell and reaction.
        Returns kf: (ncells, n_reactions)
        """
        # vectorized k0 and kinf
        k0 = self._k0_T(T)      # (ncells, n_reactions)
        kinf = self._kinf_T(T)  # (ncells, n_reactions)

        M = self.M_concentration(C)  # (ncells, n_reactions)

        # reduced pressure Pr
        kinf_safe = torch.clamp(kinf, min=self.eps)
        Pr = (k0 * M) / kinf_safe

        # base Lindemann expression: kinf * Pr/(1+Pr)
        k_base = kinf * (Pr / (1.0 + Pr))

        # falloff factor
        F = self._F_matrix(T, Pr)

        # final kf
        kf = k_base * F
        return kf

    # ----- Equilibrium Kc (vectorized like in ThreeBodyReactionBatch) -----
    def Kc(self, T):
        ncells = T.shape[0]
        # species-level thermo for all flattened indices
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

        deltaG0_r = deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r) # not in place fonction to make it compatible with vmap and torch.compile
        deltaG0_p = deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p) # not in place fonction to make it compatible with vmap and torch.compile

        deltaG0 = deltaG0_p - deltaG0_r
        Kp = torch.exp(-deltaG0 / (R * T[:, None]))
        Kc = Kp * (Pref / (R * T[:, None])) ** self.delta_n[None, :]
        return Kc

    # ----- Reaction rates (scatter style) -----
    def RR(self, T, C):
        """
        Compute RR for all reactions and cells, returns (ncells, n_reactions)
        Implementation mirrors ThreeBodyReactionBatch.RR_scatter but uses kf(T,C)
        for falloff kinetics.
        """
        ncells = T.shape[0]
        kf = self.kf(T, C)  # (ncells, n_reactions)
        Kc = self.Kc(T)     # (ncells, n_reactions)

        # use log-space accumulation to support variable stoichiometries
        # guard concentrations
        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])
        logC_p = torch.log(C_safe[:, self.prod_idx])

        conc_r = torch.zeros((ncells, self.n_reactions), device=self.device, dtype=self.dtype)
        conc_p = torch.zeros_like(conc_r)

        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        conc_r = conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), logC_r * self.react_sto[None, :]) # not in place fonction to make it compatible with vmap and torch.compile
        conc_p = conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), logC_p * self.prod_sto[None, :]) # not in place fonction to make it compatible with vmap and torch.compile

        conc_r = torch.exp(conc_r)
        conc_p = torch.exp(conc_p)

        baseRR = kf * (conc_r - (1.0 - self.oneWay[None, :]) * conc_p / Kc)
        return baseRR



    # ----- Convenience: compute species production rates (omega) -----
    def omega_from_RR(self, RR):
        """
        Convert RR (ncells, n_reactions) into species rate of production
        w: (ncells, nspecies)
        using scatter_add with product stoichiometries.
        """
        ncells = RR.shape[0]
        w = torch.zeros((ncells, self.n_species), device=self.device, dtype=self.dtype)

        # species indices and stoich for reactants (consumption)
        idx_r = self.react_idx
        sto_r = self.react_sto
        # scatter add -nu_r * RR
        idx_r_scatter = self.idx_r_scatter
        # expand RR to match flattened list then accumulate per species
        # build matrix (ncells, n_flat) = RR[:, idx_r_scatter]
        RR_exp_r = RR[:, idx_r_scatter]
        # contribution per flattened entry: -sto_r[None,:] * RR_exp_r
        contrib_r = -sto_r[None, :] * RR_exp_r
        # scatter add along species axis
        w = w.scatter_add_(1, idx_r[None, :].expand(ncells, -1), contrib_r) # not in place fonction to make it compatible with vmap and torch.compile

        # products (production)
        idx_p = self.prod_idx
        sto_p = self.prod_sto
        idx_p_scatter = self.idx_p_scatter
        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p
        w = w.scatter_add_(1, idx_p[None, :].expand(ncells, -1), contrib_p) # not in place fonction to make it compatible with vmap and torch.compile

        return w

    def jvp(self, T, C, v_T, v_C): #TODO: investigate the error occuring compared to autograd for only few samples
        """
        Jacobian-vector product for species production rates w.r.t perturbations (v_T, v_C)
        for falloff (Lindemann + Troe) reaction batch.

        Signature matches ThreeBody/Elementary implementations.
        """
        #print('WARNING, NO PHYSICAL VALIDATION OF THIS CLASS')
        ncells = T.shape[0]
        device = self.device
        dtype = self.dtype

        # Precompute kf and Kc and concentrations as in RR()
        kf = self.kf(T, C)  # (ncells, n_reactions)
        Kc = self.Kc(T)  # (ncells, n_reactions)

        C_safe = torch.clamp(C, min=self.eps)
        logC_r = torch.log(C_safe[:, self.react_idx])
        logC_p = torch.log(C_safe[:, self.prod_idx])

        conc_r = torch.zeros((ncells, self.n_reactions), device=device, dtype=dtype)
        conc_p = torch.zeros_like(conc_r)
        idx_r_scatter = self.idx_r_scatter
        idx_p_scatter = self.idx_p_scatter

        conc_r = conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), logC_r * self.react_sto[None, :])
        conc_p = conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), logC_p * self.prod_sto[None, :])

        conc_r = torch.exp(conc_r)
        conc_p = torch.exp(conc_p)

        # base concentration expression used in RR
        factor = (1.0 - self.oneWay[None, :])
        conc_expr = (conc_r - factor * conc_p / Kc)  # (ncells, n_reactions)

        # --- derivatives of kf w.r.t T and C (analytical) ---
        # k0 and kinf and their T-derivatives
        k0 = self._k0_T(T)
        kinf = self._kinf_T(T)
        T_exp = T[:, None]
        # dk0/dT = k0 * (b_k0/T + Ea_k0/(R*T^2))
        dk0_dT = k0 * (self.b_k0[None, :] / T_exp + self.Ea_k0[None, :] / (R * T_exp * T_exp))
        dkinf_dT = kinf * (self.b_kinf[None, :] / T_exp + self.Ea_kinf[None, :] / (R * T_exp * T_exp))
        kinf_mask = kinf > self.eps
        dkinf_dT = torch.where(kinf_mask, dkinf_dT, torch.zeros_like(dkinf_dT))

        # M and its perturbation
        M = self.M_concentration(C)  # (ncells, n_reactions)
        dM_from_C_v = torch.matmul(v_C, self.eff_matrix.t())  # (ncells, n_reactions)

        # Pr and safe
        kinf_safe = torch.clamp(kinf, min=self.eps)
        Pr = (k0 * M) / kinf_safe
        pr_mask = Pr > self.eps
        Pr_safe = torch.clamp(Pr, min=self.eps)

        # derivatives of Pr
        # dPr/dT = (dk0/dT * M - k0 * dkinf/dT * M / kinf_safe) / kinf_safe
        # using quotient rules carefully
        dPr_dT = (dk0_dT * M - k0 * (dkinf_dT) * M / kinf_safe) / kinf_safe
        dPr_dT = torch.where(pr_mask, dPr_dT, torch.zeros_like(dPr_dT))
        # dPr/dC · v = (k0/kinf) * dM/dC · v
        k0_over_kinf = k0 / kinf_safe
        dPr_from_C_v = k0_over_kinf * dM_from_C_v
        dPr_from_C_v = torch.where(pr_mask, dPr_from_C_v, torch.zeros_like(dPr_from_C_v))

        # --- Falloff factor F and its derivatives ---
        # Build F like in _F_matrix but keep intermediate quantities for derivatives
        nc = ncells
        T_exp_col = T[:, None]
        # troe parameters expanded
        A = self.troe_A[None, :].expand(nc, -1)
        T1 = self.troe_T1[None, :].expand(nc, -1)
        T2 = self.troe_T2[None, :].expand(nc, -1)
        T3 = self.troe_T3[None, :].expand(nc, -1)

        # Compute Fcent components and derivative wrt T
        # masks
        mask_T1 = T1 > 0.0
        mask_T3 = T3 > 0.0
        mask_T2 = T2 > 0.0

        # components
        comp1 = torch.zeros_like(Pr, dtype=dtype, device=device)
        comp2 = torch.zeros_like(Pr, dtype=dtype, device=device)
        comp3 = torch.zeros_like(Pr, dtype=dtype, device=device)

        if mask_T1.any():
            comp1 = (A * torch.exp(-T_exp_col / torch.where(mask_T1, T1, torch.ones_like(T1)))) * mask_T1.to(dtype)
        if mask_T3.any():
            comp2 = ((1.0 - A) * torch.exp(-T_exp_col / torch.where(mask_T3, T3, torch.ones_like(T3)))) * mask_T3.to(
                dtype)
        if mask_T2.any():
            comp3 = (torch.exp(-torch.where(mask_T2, T2, torch.ones_like(T2)) / T_exp_col)) * mask_T2.to(dtype)

        Fcent = comp1 + comp2 + comp3
        Fcent = torch.clamp(Fcent, min=self.eps)

        # derivatives dFcent/dT
        dcomp1_dT = torch.zeros_like(comp1)
        dcomp2_dT = torch.zeros_like(comp2)
        dcomp3_dT = torch.zeros_like(comp3)

        if mask_T1.any():
            # d/dT [A * exp(-T/T1)] = A * (-1/T1) * exp(-T/T1)
            dcomp1_dT = (A * (-1.0 / torch.where(mask_T1, T1, torch.ones_like(T1))) *
                         torch.exp(-T_exp_col / torch.where(mask_T1, T1, torch.ones_like(T1)))) * mask_T1.to(dtype)
        if mask_T3.any():
            dcomp2_dT = ((1.0 - A) * (-1.0 / torch.where(mask_T3, T3, torch.ones_like(T3))) *
                         torch.exp(-T_exp_col / torch.where(mask_T3, T3, torch.ones_like(T3)))) * mask_T3.to(dtype)
        if mask_T2.any():
            # d/dT [exp(-T2/T)] = exp(-T2/T) * (T2 / T^2)
            dcomp3_dT = (torch.exp(-torch.where(mask_T2, T2, torch.ones_like(T2)) / T_exp_col) *
                         (torch.where(mask_T2, T2, torch.ones_like(T2)) / (T_exp_col * T_exp_col))) * mask_T2.to(dtype)

        dFcent_dT = dcomp1_dT + dcomp2_dT + dcomp3_dT

        # log10(Fcent) and derivative
        logFcent = torch.log10(Fcent)
        # d(log10 Fcent)/dT = (1 / (Fcent * ln(10))) * dFcent/dT
        inv_ln10 = 1.0 / math.log(10.0)
        fcent_mask = Fcent > self.eps
        dlogFcent_dT = (dFcent_dT / Fcent) * inv_ln10
        dFcent_dT = torch.where(fcent_mask, dFcent_dT, torch.zeros_like(dFcent_dT))

        C_par = -0.4 - 0.67 * logFcent
        N_par = 0.75 - 1.27 * logFcent
        dCpar_dT = -0.67 * dlogFcent_dT
        dNpar_dT = -1.27 * dlogFcent_dT

        # guard Pr for log10
        logPr = torch.log10(Pr_safe)
        dlogPr_dPr = torch.where(pr_mask, 1.0 / (Pr_safe * math.log(10.0)), torch.zeros_like(Pr_safe))



        # f1 numerator/denominator
        numerator = (logPr + C_par)
        denominator = (N_par - 0.14 * (logPr + C_par))
        denominator = torch.where(denominator == 0.0, torch.full_like(denominator, self.eps), denominator)
        f1 = numerator / denominator
        f1_sq = f1 * f1

        # derivatives of f1 w.r.t Pr and T
        # df1/dPr = (dlogPr_dPr * denominator - numerator * (-0.14 * dlogPr_dPr)) / denominator^2
        denom_sq = denominator * denominator
        df1_dPr = (dlogPr_dPr * denominator + 0.14 * numerator * dlogPr_dPr) / denom_sq
        denom_mask = (denominator != 0.0)
        df1_dPr = torch.where(denom_mask, df1_dPr, torch.zeros_like(df1_dPr))

        # df1/dT: numerator_T = dCpar_dT, denominator_T = dNpar_dT - 0.14 * dCpar_dT
        numerator_T = dCpar_dT
        denominator_T = dNpar_dT - 0.14 * dCpar_dT
        df1_dT = (numerator_T * denominator - numerator * denominator_T) / denom_sq
        df1_dT = torch.where(denom_mask, df1_dT, torch.zeros_like(df1_dT))

        # exponent = logFcent / (1 + f1^2)
        denom_f = (1.0 + f1_sq)
        denom_f_sq = denom_f * denom_f

        dexponent_dlogFcent = 1.0 / denom_f
        dexponent_df1 = -2.0 * f1 * logFcent / denom_f_sq

        # d exponent / dT = dexponent_dlogFcent * dlogFcent_dT + dexponent_df1 * df1_dT
        dexponent_dT = dexponent_dlogFcent * dlogFcent_dT + dexponent_df1 * df1_dT

        # d exponent / dPr = dexponent_df1 * df1_dPr  (logFcent independent of Pr)
        dexponent_dPr = dexponent_df1 * df1_dPr

        # F = 10^exponent -> dF = F * ln(10) * d exponent
        F_troe = torch.pow(10.0, logFcent / denom_f)
        ln10 = math.log(10.0)
        dF_dT = F_troe * ln10 * dexponent_dT
        dF_dPr = F_troe * ln10 * dexponent_dPr

        # Build full F matrix
        F = torch.ones((nc, self.n_reactions), dtype=dtype, device=device)
        troe_idx = [i for i, t in enumerate(self.falloff_types) if t == 'troe']
        if len(troe_idx) > 0:
            mask = torch.zeros((self.n_reactions,), dtype=torch.bool, device=device)
            mask[torch.tensor(troe_idx, device=device, dtype=torch.long)] = True
            F[:, mask] = F_troe[:, mask]
            # Only keep derivatives for troe reactions, others remain zero
            dF_dT = dF_dT * mask[None, :]
            dF_dPr = dF_dPr * mask[None, :]
        else:
            # no troe reactions
            dF_dT = torch.zeros_like(F)
            dF_dPr = torch.zeros_like(F)

        # For Lindemann, F = 1 -> derivatives zero

        # --- derivatives of k_base = kinf * Pr/(1+Pr) ---
        # f(Pr) = Pr/(1+Pr), f'(Pr) = 1/(1+Pr)^2
        f_prime = 1.0 / ((1.0 + Pr) * (1.0 + Pr))

        dkbase_dT = dkinf_dT * (Pr / (1.0 + Pr)) + kinf_safe * (f_prime * dPr_dT)
        # dkbase/dC · v = kinf * f'(Pr) * dPr_from_C_v
        dkbase_from_C_v = kinf_safe * (f_prime * dPr_from_C_v)

        # kf = k_base * F
        k_base = kinf_safe * (Pr / (1.0 + Pr))
        dkf_dT = dkbase_dT * F + k_base * dF_dT
        dkf_from_C_v = dkbase_from_C_v * F + k_base * (dF_dPr * dPr_from_C_v)

        # --- dKc/dT analytic (reuse pattern from Elementary) ---
        h_r = self.thermo.h_species(T, self.react_idx)
        s_r = self.thermo.s_species_Pref(T, self.react_idx)
        h_p = self.thermo.h_species(T, self.prod_idx)
        s_p = self.thermo.s_species_Pref(T, self.prod_idx)

        G0_r = h_r - T[:, None] * s_r
        G0_p = h_p - T[:, None] * s_p

        deltaG0_r = torch.zeros((nc, self.n_reactions), device=device, dtype=dtype)
        deltaG0_p = torch.zeros_like(deltaG0_r)

        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaG0_r = deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(nc, -1), sto_r_exp * G0_r)
        deltaG0_p = deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(nc, -1), sto_p_exp * G0_p)

        deltaG0 = deltaG0_p - deltaG0_r

        deltaS_r = torch.zeros((nc, self.n_reactions), device=device, dtype=dtype)
        deltaS_p = torch.zeros_like(deltaS_r)

        deltaS_r = deltaS_r.scatter_add_(1, idx_r_scatter[None, :].expand(nc, -1), sto_r_exp * s_r)
        deltaS_p = deltaS_p.scatter_add_(1, idx_p_scatter[None, :].expand(nc, -1), sto_p_exp * s_p)

        deltaS = deltaS_p - deltaS_r

        dlnKc_dT = deltaS / (R * T_exp) + deltaG0 / (R * T_exp * T_exp) - self.delta_n[None, :] / T_exp
        dKc_dT = Kc * dlnKc_dT

        # --- dRR/dT ---
        dRR_dT = dkf_dT * conc_expr + kf * (factor * conc_p / (Kc * Kc) * dKc_dT)
        dRR_from_T_v = dRR_dT * v_T[:, None]

        # --- dRR/dC · v ---
        # concentration contributions (reactants/products) same pattern as elemental
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
        dbase_from_C_v = dbase_from_C_v.scatter_add_(1, idx_r_scatter[None, :].expand(nc, -1), contrib_r_flat)
        dbase_from_C_v = dbase_from_C_v.scatter_add_(1, idx_p_scatter[None, :].expand(nc, -1), contrib_p_flat)

        # add contribution from dkf_from_C_v multiplying concentration expression
        dRR_from_C_v = dbase_from_C_v + dkf_from_C_v * conc_expr

        # --- total dRR·v ---
        dRR_v = dRR_from_T_v + dRR_from_C_v

        # --- map to species rates
        Jv_w = self.omega_from_RR(dRR_v)
        return Jv_w


class ReactionSystem:
    def __init__(self, reactionsBatches):
        self.reactionsBatches = reactionsBatches
        self.device = reactionsBatches[0].device
        self.dtype = reactionsBatches[0].dtype

        self.n_species = len(reactionsBatches[0].species)

    def jvp(self, T, C, v_T, v_C):
        ncells = T.shape[0]
        jvp_total = torch.zeros((ncells, self.n_species), device=self.device, dtype=self.dtype)

        for b in self.reactionsBatches:
            jvp_total += b.jvp(T, C, v_T, v_C)                  # sum over batches

        return jvp_total

    def omega_total(self, T, C):
        """
        Compute total species production rates omega.
        Returns tensor of shape (ncells, n_species)
        """
        ncells = T.shape[0]
        w_total = torch.zeros((ncells, self.n_species), device=self.device, dtype=self.dtype)

        for b in self.reactionsBatches:
            RR = b.RR(T, C)       # (ncells, n_reactions_in_batch)
            w = b.omega_from_RR(RR)       # (ncells, n_species)
            w_total += w                  # sum over batches

        return w_total

    def omega_total_no_log(self, T, C):
        """
        Compute total species production rates omega.
        Returns tensor of shape (ncells, n_species)
        """
        ncells = T.shape[0]
        w_total = torch.zeros((ncells, self.n_species), device=self.device, dtype=self.dtype)

        for b in self.reactionsBatches:
            RR = b.RR_no_log(T, C)       # (ncells, n_reactions_in_batch)
            w = b.omega_from_RR(RR)       # (ncells, n_species)
            w_total += w                  # sum over batches

        return w_total




if __name__ == '__main__':
    import torch
    import time
    from torch.autograd.functional import jvp
    import kineticsGPU.thermo as yaml_thermo
    from kineticsGPU.reactionParser import parseReactions, build_reactions

    yaml_file = "../tests/gri30.yaml"
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
    model = FalloffReactionBatch(r_f, device=device, dtype=dtype)
    nspecies = len(model.species) # OU mettre une valeur si tu veux tester sans initialiser un vrai modèle
    torch.set_default_dtype(dtype)


    T = torch.rand(ncells, device=device, dtype=dtype) * 1500 + 300
    C = torch.rand(ncells, nspecies, device=device, dtype=dtype) * 1e-2 + 1e-8

    v = torch.randn(ncells, nspecies+1, device=device, dtype=dtype)
    for reac in r_e:
        reac.compare_jvp(T, C, v[:,0], v[:,1:])

    for reac in r_tb:
        reac.compare_jvp(T, C, v[:,0], v[:,1:])


    """
    # ============================================================
    # UTIL: fonction wrapper RR(T,C) → omega(T,C)
    # ============================================================
    def omega_fn(TC_flat):

        T = TC_flat[:ncells]  # (ncells,)
        C = TC_flat[ncells:].reshape(ncells, nspecies)  # (ncells, nspecies)

        # forward RR puis omega
        RR = model.RR(T, C)
        w = model.omega_from_RR(RR)
        return w.reshape(-1)


    # ============================================================
    # Génération des états tests
    # ============================================================
    torch.manual_seed(0)

    T = torch.rand(ncells, device=device, dtype=dtype) * 1500 + 300
    C = torch.rand(ncells, nspecies, device=device, dtype=dtype) * 1e-2 + 1e-8

    vT = torch.randn(ncells, device=device, dtype=dtype)
    vC = torch.randn(ncells, nspecies, device=device, dtype=dtype)

    # ============================================================
    # APPEL JVP ANALYTIQUE
    # ============================================================

    t0 = time.time()
    Jv_analytic = model.jvp(T, C, vT, vC)  # shape (ncells, nspecies)
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
"""