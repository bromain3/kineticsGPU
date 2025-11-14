import numpy as np
R = 8.31446261815324  # J/(mol*K)
Pref = 101325.0       # Pa

class ElementaryReactionVal:
    def __init__(self, reactants, products, thermo, species, oneWay, A, n, Ea):
        self.reactants = reactants
        self.products = products
        self.thermo = thermo
        self.oneWay = oneWay
        self.species = species  # for the moment, we store species list to ensure alignment and react/prods as dict but in the end, react and prod will be sparse arrays
        self.A = A
        self.n = n
        self.Ea = Ea

    def Kp(self, T):
        G0 = {}
        for specie in set(list(self.reactants.keys()) + list(self.products.keys())):
            h = self.thermo.h(specie, T, Pref)
            s = self.thermo.s(specie, T, Pref)
            G0[specie] = h - T * s

        deltaG0 = 0.0
        for s, coef in self.products.items():
            deltaG0 += coef * G0[s]
        for s, coef in self.reactants.items():
            deltaG0 -= coef * G0[s]

        # Calcul de Kp
        return (np.exp(-deltaG0 / (R * T)))

    def Kc(self, T, phases=None):
        # Δn ne prend en compte que les gaz
        delta_n = 0.0
        for s, coef in self.products.items():
            if not phases or phases.get(s, 'g') == 'g':
                delta_n += coef
        for s, coef in self.reactants.items():
            if not phases or phases.get(s, 'g') == 'g':
                delta_n -= coef

        # Conversion Kp -> Kc (gaz parfaits)
        Kc = self.Kp(T) * (Pref / (R * T)) ** delta_n
        return Kc

    def kf(self, T):
        return (self.A * (T ** self.n) * np.exp(-self.Ea / (R * T)))

    def kb(self, T):
        return (self.kf(T) / self.Kc(T))

    def RR(self, TC):
        T = TC[0]
        concentrations = TC[1:]

        ConcentrationReact = 1.0
        for spe, coef in self.reactants.items():
            conc = concentrations[self.species.index(spe)]
            ConcentrationReact *= conc ** coef

        ConcentrationProd = 1.0
        for spe, coef in self.products.items():
            conc = concentrations[self.species.index(spe)]
            ConcentrationProd *= conc ** coef

        return (self.kf(T) * (ConcentrationReact - (1 - self.oneWay) / self.Kc(T) * ConcentrationProd))

    def jacobian(self, TC):
        """
        Calcule les dérivées analytiques du taux de réaction (RR)
        par rapport à la température et aux concentrations.
        Retourne un tableau : [∂RR/∂T, ∂RR/∂C_1, ..., ∂RR/∂C_n]
        """
        T = TC[0]
        C = TC[1:]
        Ns = len(self.species)

        # --- Constantes et valeurs de base ---
        kf = self.kf(T)
        Kc = self.Kc(T)
        kb = kf / Kc

        # Concentrations des réactifs et produits
        conc_react = 1.0
        for s, nu in self.reactants.items():
            conc_react *= C[self.species.index(s)] ** nu

        conc_prod = 1.0
        for s, nu in self.products.items():
            conc_prod *= C[self.species.index(s)] ** nu

        # --- Dérivées par rapport à T ---
        dkf_dT = kf * (self.n / T + self.Ea / (R * T ** 2))
        # dKc/dT = Kc * d(ln Kc)/dT = Kc * (d(ln Kp)/dT - Δn/T)
        # Approximation : dérivée de ln(Kp) ~ (ΔH0 / (R T^2))
        # On simplifie si pas de données thermo : on ignore dKc/dT
        dKc_dT = 0.0

        dRR_dT = dkf_dT * (conc_react - (1 - self.oneWay) / Kc * conc_prod)
        if dKc_dT != 0:
            dRR_dT += kf * (1 - self.oneWay) * conc_prod * dKc_dT / (Kc ** 2)

        # --- Dérivées par rapport aux concentrations ---
        dRR_dC = np.zeros(Ns)
        for i, s in enumerate(self.species):
            # somme des contributions réactif + produit
            d_conc_react = 0.0
            d_conc_prod = 0.0

            if s in self.reactants:
                nu = self.reactants[s]
                d_conc_react = nu * conc_react / C[i]
            if s in self.products:
                nu = self.products[s]
                d_conc_prod = nu * conc_prod / C[i]

            dRR_dC[i] = kf * (d_conc_react - (1 - self.oneWay) / Kc * d_conc_prod)

        return np.concatenate(([dRR_dT], dRR_dC))


class ThreeBodyReactionVal(ElementaryReactionVal):
    def __init__(self, reactants, products, thermo, species, oneWay, A, n, Ea, efficiencies, defaultEfficiencies=1.0):
        ElementaryReactionVal.__init__(self, reactants, products, thermo, species, oneWay, A, n, Ea)
        self.eff = efficiencies
        self.defaultEff = defaultEfficiencies

    def MConcentration(self, TC):
        #TODO: construct the array only once
        eff = np.zeros(len(TC) - 1)
        for n, spe in enumerate(self.species):
            eff[n] = self.eff.get(spe, self.defaultEff)
        M = np.dot(TC[1:], eff)
        return(M)

    def RR(self, TC):
        baseRR = ElementaryReactionVal.RR(self, TC)
        return (baseRR * self.MConcentration(TC))

    def jacobian(self, TC):
        base_J = super().jacobian(TC)
        M = self.MConcentration(TC)
        eff = np.array([self.eff.get(s, self.defaultEff) for s in self.species])

        # dérivées par rapport à T et C (chaîne de produit avec M)
        dM_dC = eff
        dRR_dT = base_J[0] * M
        dRR_dC = base_J[1:] * M + ElementaryReactionVal.RR(self, TC) * dM_dC

        return np.concatenate(([dRR_dT], dRR_dC))


class FalloffReactionVal(ElementaryReactionVal):
    def __init__(self, reactants, products, thermo, species, oneWay, k0_params, kinf_params, efficiencies=None, falloff_type='Lindemann', falloff_params=None):
        super().__init__(reactants, products, thermo, species, oneWay, None, None, None)
        self.A_k0, self.b_k0, self.Ea_k0 = k0_params
        self.A_kinf, self.b_kinf, self.Ea_kinf = kinf_params
        self.efficiencies = efficiencies
        self.falloffType = falloff_type
        self.falloffParams = falloff_params

    def k0_T(self, T):
        return (self.A_k0 * (T ** self.b_k0) * np.exp(-self.Ea_k0 / (R * T)))

    def kinf_T(self, T):
        return (self.A_kinf * (T ** self.b_kinf) * np.exp(-self.Ea_kinf / (R * T)))

    def MConcentration(self, TC):
        #TODO: construct the array only once
        eff = np.zeros(len(TC) - 1)
        for n, spe in enumerate(self.species):
            eff[n] = self.efficiencies.get(spe, 1.0)
        M = np.dot(TC[1:], eff)
        return(M)

    def _F_troe(self, T, Pr):
        A = self.falloffParams['A']
        T1 = self.falloffParams.get('T1', 0.0)
        T2 = self.falloffParams.get('T2', 0.0)
        T3 = self.falloffParams.get('T3', 0.0)

        # F_cent selon Cantera
        Fcent = 0.0
        if T1 > 0.0:
            Fcent += A * np.exp(-T / T1)
        if T3 > 0.0:
            Fcent += (1.0 - A) * np.exp(-T / T3)
        if T2 > 0.0:
            Fcent += np.exp(-T2 / T)

        logFcent = np.log10(Fcent)
        C = -0.4 - 0.67 * logFcent
        N = 0.75 - 1.27 * logFcent
        f1 = (np.log10(Pr) + C) / (N - 0.14 * (np.log10(Pr) + C))
        F = 10 ** (logFcent / (1 + f1 ** 2))

        return F

    def _F_lindemann(self, T, Pr):
        # Lindemann: no broadening factor
        return 1.0

    def _F_sri(self, T, Pr):
        raise NotImplementedError('sri parametrisation not implemented')
        return F

    def _F(self, T, Pr):
        t = self.falloffType.lower()
        if t == 'troe':
            return self._F_troe(T, Pr)
        elif t == 'sri':
            return self._F_sri(T, Pr)
        elif t == 'lindemann':
            return self._F_lindemann(T, Pr)
        else:
            raise ValueError('Falloff type not recognised')

    def kf(self, TC):
        # TC is expected to be an array-like with T as first element and concentrations after
        M = self.MConcentration(TC)
        k0 = self.k0_T(TC[0])
        kinf = self.kinf_T(TC[0])
        #Pr = (k0 * M) / max(kinf, 1e-300)
        Pr = (k0 * M) / kinf
        # Lindemann base expression
        #kf_val = (k0 * M) / (1.0 + k0 * M / max(kinf, 1e-300))
        kf_val = (k0 * M) / (1.0 + k0 * M / kinf)
        # apply falloff correction F
        F = self._F(TC[0], Pr)
        return kf_val * F

    def kb(self, TC):
        return(self.kf(TC) / self.Kc(TC[0]))

    def RR(self, TC):
        # overwrite RR to compute rate using pressure-dependent kf(T,[M])
        T = TC[0]
        concentrations = TC[1:]

        ConcentrationReact = 1.0
        for spe, coef in self.reactants.items():
            conc = concentrations[self.species.index(spe)]
            ConcentrationReact *= conc ** coef

        ConcentrationProd = 1.0
        for spe, coef in self.products.items():
            conc = concentrations[self.species.index(spe)]
            ConcentrationProd *= conc ** coef

        # forward rate uses TC (so k accounts for [M])
        k_forward = self.kf(TC)

        return (k_forward * ConcentrationReact - (1.0 - self.oneWay) * k_forward / self.Kc(T) * ConcentrationProd)

    def jacobian(self, TC):
        """
        Calcule les dérivées analytiques du taux de réaction pour FalloffReactionVal
        par rapport à T et toutes les concentrations.
        Retourne un vecteur : [dRR/dT, dRR/dC1, ..., dRR/dCn]
        """
        T = TC[0]
        C = TC[1:]
        Ns = len(self.species)

        # --- Concentration efficace M ---
        eff = np.array([self.efficiencies.get(s, 1.0) for s in self.species])
        M = np.dot(C, eff)

        # --- Constantes ---
        k0 = self.k0_T(T)
        kinf = self.kinf_T(T)
        Pr = k0 * M / kinf

        # --- Facteur de falloff ---
        F = self._F(T, Pr)

        # --- Dérivées des constantes k0 et kinf ---
        dk0_dT = k0 * (self.b_k0 / T + self.Ea_k0 / (R * T ** 2))
        dkinf_dT = kinf * (self.b_kinf / T + self.Ea_kinf / (R * T ** 2))

        # --- Dérivée de M par rapport aux concentrations ---
        dM_dC = eff  # dM/dC_i = alpha_i

        # --- RR de base ---
        conc_react = np.prod([C[self.species.index(s)] ** nu for s, nu in self.reactants.items()])
        conc_prod = np.prod([C[self.species.index(s)] ** nu for s, nu in self.products.items()])

        kf_base = (k0 * M) / (1 + k0 * M / kinf)
        RR = kf_base * F * (conc_react - (1 - self.oneWay) / self.Kc(T) * conc_prod)

        # --- Dérivées ---
        # 1) dérivée par rapport à T
        dPr_dT = Pr * (dk0_dT / k0 - dkinf_dT / kinf)

        if self.falloffType.lower() == 'lindemann':
            dF_dT = 0.0
        elif self.falloffType.lower() == 'troe':
            # approximation de dF/dT via logFcent et f1
            A = self.falloffParams['A']
            T1 = self.falloffParams.get('T1', 0.0)
            T2 = self.falloffParams.get('T2', 0.0)
            T3 = self.falloffParams.get('T3', 0.0)
            Fcent = 0.0
            if T1 > 0.0: Fcent += A * np.exp(-T / T1)
            if T3 > 0.0: Fcent += (1.0 - A) * np.exp(-T / T3)
            if T2 > 0.0: Fcent += np.exp(-T2 / T)
            logFcent = np.log10(Fcent)
            C_param = -0.4 - 0.67 * logFcent
            N_param = 0.75 - 1.27 * logFcent
            f1 = (np.log10(Pr) + C_param) / (N_param - 0.14 * (np.log10(Pr) + C_param))
            F = 10 ** (logFcent / (1 + f1 ** 2))

            # dérivée approximative dF/dT (en supposant variation via logFcent)
            dlogFcent_dT = 0.0
            if T1 > 0.0: dlogFcent_dT += -A / (T1 * Fcent * np.log(10)) * np.exp(-T / T1)
            if T3 > 0.0: dlogFcent_dT += -(1 - A) / (T3 * Fcent * np.log(10)) * np.exp(-T / T3)
            if T2 > 0.0: dlogFcent_dT += -(T2 / T ** 2) / (Fcent * np.log(10)) * np.exp(-T2 / T)
            dF_dT = F * dlogFcent_dT
        elif self.falloffType.lower() == 'sri':
            raise NotImplementedError("Jacobian pour SRI non implémenté")
        else:
            raise ValueError('Falloff type non reconnu')

        # dérivée kf_base par rapport à T
        dkf_base_dT = (dk0_dT * M * kinf - k0 * M * dkinf_dT * k0 * M / kinf) / (kinf + k0 * M) ** 2

        dRR_dT = (dkf_base_dT * F + kf_base * dF_dT) * (conc_react - (1 - self.oneWay) / self.Kc(T) * conc_prod)

        # 2) dérivée par rapport aux concentrations
        dRR_dC = np.zeros(Ns)
        for i, s in enumerate(self.species):
            d_conc_react = 0.0
            d_conc_prod = 0.0
            if s in self.reactants:
                nu = self.reactants[s]
                d_conc_react = nu * conc_react / C[i]
            if s in self.products:
                nu = self.products[s]
                d_conc_prod = nu * conc_prod / C[i]

            # dRR/dC_i = contribution directe + contribution via M
            dRR_dC[i] = kf_base * F * (d_conc_react - (1 - self.oneWay) / self.Kc(T) * d_conc_prod)
            dRR_dC[i] += (F * (conc_react - (1 - self.oneWay) / self.Kc(T) * conc_prod)) * (dk0_dT) * dM_dC[i] / (
                        1 + k0 * M / kinf)

        return np.concatenate(([dRR_dT], dRR_dC))
