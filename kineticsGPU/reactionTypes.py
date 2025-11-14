import torch

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
        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_r)
        )
        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_p)
        )

        # Expand stoichiometry to (ncells, n_total)
        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r)
        deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p)

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

        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_r))
        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_p))

        conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), logC_r * self.react_sto[None, :])
        conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), logC_p * self.prod_sto[None, :])

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
        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_r))

        # expand RR to match flattened react list
        RR_exp_r = RR[:, idx_r_scatter]     # (ncells, n_total_react)

        # contribution: -nu_r * RR
        contrib_r = -sto_r[None, :] * RR_exp_r

        # scatter add on species axis
        w.scatter_add_(1, idx_r[None, :].expand(ncells, -1), contrib_r)

        # ---------- Products: production ----------
        idx_p = self.prod_idx
        sto_p = self.prod_sto

        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_p))

        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p

        w.scatter_add_(1, idx_p[None, :].expand(ncells, -1), contrib_p)

        return w


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
        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_r))
        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_p))
        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]
        deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r)
        deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p)
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

        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_r))
        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_p))

        conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), logC_r * self.react_sto[None, :])
        conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), logC_p * self.prod_sto[None, :])

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
        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_r))

        # expand RR to match flattened react list
        RR_exp_r = RR[:, idx_r_scatter]     # (ncells, n_total_react)

        # contribution: -nu_r * RR
        contrib_r = -sto_r[None, :] * RR_exp_r

        # scatter add on species axis
        w.scatter_add_(1, idx_r[None, :].expand(ncells, -1), contrib_r)

        # ---------- Products: production ----------
        idx_p = self.prod_idx
        sto_p = self.prod_sto

        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(
            torch.diff(self.reaction_offsets_p))

        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p

        w.scatter_add_(1, idx_p[None, :].expand(ncells, -1), contrib_p)

        return w




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

        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_r))
        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_p))

        sto_r_exp = self.react_sto[None, :]
        sto_p_exp = self.prod_sto[None, :]

        deltaG0_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), sto_r_exp * G0_r)
        deltaG0_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), sto_p_exp * G0_p)

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

        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_r))
        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_p))

        conc_r.scatter_add_(1, idx_r_scatter[None, :].expand(ncells, -1), logC_r * self.react_sto[None, :])
        conc_p.scatter_add_(1, idx_p_scatter[None, :].expand(ncells, -1), logC_p * self.prod_sto[None, :])

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
        idx_r_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_r))
        # expand RR to match flattened list then accumulate per species
        # build matrix (ncells, n_flat) = RR[:, idx_r_scatter]
        RR_exp_r = RR[:, idx_r_scatter]
        # contribution per flattened entry: -sto_r[None,:] * RR_exp_r
        contrib_r = -sto_r[None, :] * RR_exp_r
        # scatter add along species axis
        w.scatter_add_(1, idx_r[None, :].expand(ncells, -1), contrib_r)

        # products (production)
        idx_p = self.prod_idx
        sto_p = self.prod_sto
        idx_p_scatter = torch.arange(self.n_reactions, device=self.device).repeat_interleave(torch.diff(self.reaction_offsets_p))
        RR_exp_p = RR[:, idx_p_scatter]
        contrib_p = sto_p[None, :] * RR_exp_p
        w.scatter_add_(1, idx_p[None, :].expand(ncells, -1), contrib_p)

        return w



class ReactionSystem:
    def __init__(self, reactionsBatches):
        self.reactionsBatches = reactionsBatches
        self.device = reactionsBatches[0].device
        self.dtype = reactionsBatches[0].dtype

        self.n_species = len(reactionsBatches[0].species)


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
