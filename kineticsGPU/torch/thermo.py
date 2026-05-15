import torch
import yaml
import numpy as np

#TODO: improve sanity checks and robustness for model selection
# divide in classes so that every model selection, ranges, coefs, etc is done at init time, so at runtime we juste have pur sheer computation done

R = 8.314_462_618_153_24  # SI
Pref = 1e5


class MM:
    O = 15.999e-3
    N = 14.0067e-3
    Ar = 39.948e-3
    H = 1.00784e-3
    C = 12.011e-3

class yamlParser:
    def __init__(self, YAML_file):
        self.Yaml_file = YAML_file
        with open(self.Yaml_file, 'r') as f:
            self.yaml_data = yaml.load(f, Loader=yaml.BaseLoader)
        self.thermo = self.getThermoData()

    def getThermoData(self):
        res = []
        for spe in self.yaml_data['species']:
            res.append({
                'name': spe['name'],
                'composition': [(int(v), k) for k,v in spe['composition'].items()],
                'modelType': spe['thermo']['model'],
                'trange': [float(k) for k in spe['thermo']['temperature-ranges']],
                'data': [[float(k) for k in spe['thermo']['data'][i]] for i in range(len(spe['thermo']['data']))]
            })
        return res

    def _find_species(self, name):
        """Retourne le dictionnaire thermo correspondant à une espèce donnée."""
        for s in self.thermo:
            if s['name'] == name:
                return s
        raise ValueError(f"Espèce '{name}' non trouvée dans le YAML")

    #TODO: make static
    def _make_fictive_species(self, fictive_name, species_list):
        #TODO: make a function to compute weighted mean
        """Construit une espèce fictive à partir de la moyenne des espèces_list."""
        selected = [self._find_species(n) for n in species_list]

        Tmin = np.mean([s['trange'][0] for s in selected])
        Tmid = np.mean([s['trange'][1] for s in selected])
        Tmax = np.mean([s['trange'][2] for s in selected])
        coef_low = np.mean(np.array([s['data'][0] for s in selected]), axis=0).tolist()
        coef_high = np.mean(np.array([s['data'][1] for s in selected]), axis=0).tolist()

        # masse molaire moyenne
        molar_mass = np.mean([
            sum([n * getattr(MM, atom) for n, atom in s['composition']])
            for s in selected
        ])

        return {
            'name': fictive_name,
            'trange': [Tmin, Tmid, Tmax],
            'data': [coef_low, coef_high],
            'molar_mass': molar_mass
        }

    #TODO: make static
    def compute_fictive_composition(self, base_species, new_species):
        """
        Calcule la composition de l'espèce fictive Yfict automatiquement.

        Args:
            base_species (list[str]): liste des espèces du schéma de base (ex: GRI30)
            new_species (list[str]): liste des espèces utilisées dans le modèle étendu, ex: [T, P, Y1, ..., Yn, Yfict]

        Returns:
            dict: ex {'Yfict': ['CO2', 'H2O', 'CH4', ...]}
        """
        # On retire T et P s'ils sont inclus dans la liste (cas de ton parser)
        filtered_new = [s for s in new_species if s not in ("T", "P")]

        # On identifie les nouvelles espèces (celles absentes du schéma de base)
        added_species = [s for s in filtered_new if s not in base_species]

        fictive_dict = {}

        for f in added_species:
            # toutes les espèces du schéma de base non présentes dans la nouvelle liste
            missing_from_new = [s for s in base_species if s not in filtered_new]
            fictive_dict[f] = missing_from_new

        return fictive_dict

    def populateVectorizedThermoObject(self, dtype, device, reducedSpeciesSet=None):
        #TODO: VERY IMPORTANT: CHECK THE AGGREGATION OF PROPERTIES TO BE SURE THE CODE IS VALID
        """
        species_order : liste des espèces dans l'ordre final (ex: ['O2', 'N2', 'Yfict'])
        fictive_species : dict { 'Yfict': ['CO2', 'H2O', 'N2'] }
        """
        thermo_full = {v['name']: v for v in self.thermo}
        species_order = [v['name'] for v in self.thermo]

        # si espèce fictive -> on la construit
        if reducedSpeciesSet is not None:
            fictive_species = self.compute_fictive_composition(species_order, reducedSpeciesSet)
            for fict_name, base_list in fictive_species.items():
                thermo_full[fict_name] = self._make_fictive_species(fict_name, base_list)

            species_order = reducedSpeciesSet


        Tmin = [thermo_full[n]['trange'][0] for n in species_order]
        Tmid = [thermo_full[n]['trange'][1] for n in species_order]
        Tmax = [thermo_full[n]['trange'][2] for n in species_order]
        coefs_low = [thermo_full[n]['data'][0] for n in species_order]
        coefs_high = [thermo_full[n]['data'][1] for n in species_order]
        molar_masses = []
        for n in species_order:
            if 'molar_mass' in thermo_full[n].keys():
                molar_masses.append(thermo_full[n]['molar_mass'])
            else:
                molar_masses.append(sum([num * getattr(MM, atom) for num, atom in thermo_full[n]['composition']]))

        return ThermoData_vectorized(
            species_order, Tmin, Tmid, Tmax, coefs_low, coefs_high, molar_masses, dtype, device
        )


class ThermoData_vectorized:
    ##############
    ##### Retourne tout en massique
    #############
    def __init__(self, species_names, Tmin, Tmid, Tmax, coef_low, coef_high, molar_masses, dtype, device):
        self.species = species_names  # (nspecies,)
        self.Tmin = torch.tensor(Tmin, dtype=dtype, device=device)  # (nspecies,)
        self.Tmid = torch.tensor(Tmid, dtype=dtype, device=device)  # (nspecies,)
        self.Tmax = torch.tensor(Tmax, dtype=dtype, device=device)  # (nspecies,)
        self.coef_low = torch.tensor(coef_low, dtype=dtype, device=device)  # (nspecies, 6)
        self.coef_high = torch.tensor(coef_high, dtype=dtype, device=device)  # (nspecies, 6)
        self.molar_masses = torch.tensor(molar_masses, dtype=dtype, device=device)  # (nspecies,)
        self.inv_molar_masses = 1.0 / self.molar_masses
        self.nspecies = len(species_names)

    def _select_coef(self, T):
        """
        Sélectionne les bons coefficients (low ou high) en fonction de T.
        T: (ncells, 1)
        retourne coef: (ncells, nspecies, 7)
        """
        mask_low = T <= self.Tmid
        coef_low = self.coef_low[None, :, :]
        coef_high = self.coef_high[None, :, :]
        return torch.where(mask_low[:, :, None], coef_low, coef_high)

    def cp_mix(self, T, Y):
        #Y: (ncells, nspecies)
        T = T[:, None]  # (ncells, 1)
        coef = self._select_coef(T)[:, :, :5]  # (ncells, nspecies, 5)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T, T_2, T_3, T_4], dim=-1)  # (ncells, 1, 5)
        cp_species_molar = R * torch.sum(coef * Temp, dim=-1)  # (ncells, nspecies)
        cp_species_mass = cp_species_molar * self.inv_molar_masses
        return torch.sum(Y * cp_species_mass, dim=1)  # (ncells,)

    def cv_mix(self, T, Y):
        #Y: (ncells, nspecies)
        return self.cp_mix(T, Y) - self.R_spec_from_Y(Y)  # relation Cp-Cv = Rspec


    def cp_mix_mol(self, T, X): #TODO: factorize code for having molar and mass quantities with everyone based on molar
        """
        Capacité calorifique molaire moyenne du mélange (J/mol/K)

        Args:
            T : (ncells,)
            X : (ncells, nspecies) fractions molaires

        Returns:
            cp_mol : (ncells,), J/mol/K
        """
        T = T[:, None]  # (ncells, 1)
        coef = self._select_coef(T)[:, :, :5]  # (ncells, nspecies, 5)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack(
            [torch.ones_like(T), T, T_2, T_3, T_4], dim=-1
        )  # (ncells, 1, 5)
        cp_species_molar = R * torch.sum(coef * Temp, dim=-1)  # (ncells, nspecies)
        return torch.sum(X * cp_species_molar, dim=1)  # (ncells,)

    def cv_mix_mol(self, T, X): #TODO: factorize code for having molar and mass quantities with everyone based on molar
        """
        Capacité calorifique molaire moyenne à volume constant (J/mol/K)

        Args:
            T : (ncells,)
            X : (ncells, nspecies) fractions molaires

        Returns:
            cv_mol : (ncells,), J/mol/K
        """
        cp_mol = self.cp_mix_mol(T, X)
        return cp_mol - R  # Cv = Cp - R (molaire)

    def h_mix(self, T, Y):
        #Y: (ncells, nspecies)
        T = T[:, None]
        coef = self._select_coef(T)[:, :, :6]  # (ncells, nspecies, 6)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T / 2, T_2 / 3, T_3 / 4, T_4 / 5, 1 / T], dim=-1)  # (ncells, 1, 6)

        h_species_molar = R * T * torch.sum(coef * Temp, dim=-1)  # (ncells, nspecies), J/mol
        h_species_mass = h_species_molar * self.inv_molar_masses  # J/kg

        return torch.sum(Y * h_species_mass, dim=1)

    def h_mix_mol(self, T, Y): #TODO: factorize code for having molar and mass quantities with everyone based on molar
        #Y: (ncells, nspecies)
        T = T[:, None]
        coef = self._select_coef(T)[:, :, :6]  # (ncells, nspecies, 6)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T / 2, T_2 / 3, T_3 / 4, T_4 / 5, 1 / T], dim=-1)  # (ncells, 1, 6)

        h_species_molar = R * T * torch.sum(coef * Temp, dim=-1)  # (ncells, nspecies), J/mol
        return torch.sum(Y * h_species_molar, dim=1)

    def u_mix_mol(self, T, Y):
        # T : (ncells,)
        # Y : (ncells, nspecies)

        T = T[:, None]
        coef = self._select_coef(T)[:, :, :6]  # (ncells, nspecies, 6)

        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T / 2, T_2 / 3, T_3 / 4, T_4 / 5, 1 / T], dim=-1)  # (ncells, 1, 6)

        # h_species_molar (J/mol)
        h_species_molar = R * T * torch.sum(coef * Temp, dim=-1)

        # u = h - R*T  (par espèce)
        u_species_molar = h_species_molar - R * T  # (ncells, nspecies)

        # mélange
        return torch.sum(Y * u_species_molar, dim=1)  # (ncells,)


    def s_mix(self, T, P, Y):
        #Y: (ncells, nspecies)
        T = T[:, None]
        P = P[:, None]
        coef = self._select_coef(T)  # (ncells, nspecies, 7)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.log(T), T, T_2 / 2, T_3 / 3, T_4 / 4, torch.zeros_like(T), torch.ones_like(T)], dim=-1)  # (ncells, 1, 7)

        s_species_molar = torch.sum(R * coef * Temp, dim=-1) - R * torch.log(P / Pref)  # J/mol/K
        s_species_mass = s_species_molar * self.inv_molar_masses  # J/kg/K

        return torch.sum(Y * s_species_mass, dim=1)

    def g_mix(self, T, P, Y):
        #Y: (ncells, nspecies)
        return self.h_mix(T, Y) - T * self.s_mix(T, P, Y)

    def g0_mix(self, T, Y):
        #Y: (ncells, nspecies)
        return self.g_mix(T, Pref, Y)

    def R_spec_from_Y(self, Y):
        # returns R_specific for each cell (J/kg/K)
        Mmix = self.mean_molar_mass(Y)  # kg/mol
        return R / Mmix  # (N,)

    def mean_molar_mass(self, Y):
        # Y : (N, nspecies) mass fractions
        # Mmix = 1 / sum(Y_i / W_i)
        return 1.0 / torch.sum(Y * self.inv_molar_masses[None, :], dim=1)  # (N,)

    ############################################################
    # --- Propriétés espèce par espèce (pour sous-ensembles) ---
    ############################################################
    def cp_species(self, T, idx):
        """
        Chaleur spécifique à pression constante (molaires) pour certaines espèces.
        T : (ncells,)
        idx : indices Tensor (n_idx,) ou list
        retourne : cp_molar (ncells, n_idx), J/mol/K
        """
        T = T[:, None]  # (ncells, 1)
        coef = self._select_coef(T)[:, idx, :5]  # (ncells, n_idx, 5)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T, T_2, T_3, T_4], dim=-1)
        cp_species_molar = R * torch.sum(coef * Temp, dim=-1)  # (ncells, n_idx)
        return cp_species_molar

    def cp_species_mol(self, T):
        """
        Chaleur spécifique à pression constante (molaires) pour certaines espèces.
        T : (ncells,)
        idx : indices Tensor (n_idx,) ou list
        retourne : cp_molar (ncells, n_idx), J/mol/K
        """
        T = T[:, None]  # (ncells, 1)
        coef = self._select_coef(T)[:, :, :5]  # (ncells, n_idx, 5)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T, T_2, T_3, T_4], dim=-1)
        cp_species_molar = R * torch.sum(coef * Temp, dim=-1)  # (ncells, n_idx)
        return cp_species_molar

    def dcp_species_dT(self, T, idx):
        """
        Dérivée par rapport à T des chaleurs spécifiques molaires cp (J/mol/K^2)
        pour les espèces listées dans idx.

        T : (ncells,)
        idx : indices des espèces (Tensor or list)

        Retour :
            dcp_dT : (ncells, n_idx), dérivée dc_p/dT analytique
        """

        T = T[:, None]  # (ncells, 1)
        coef = self._select_coef(T)[:, idx, 1:5]  # (ncells, n_idx, 5)

        # Puissances de T
        T2 = T * T
        T3 = T2 * T

        # dcp/dT analytique
        Temp = torch.stack([torch.ones_like(T), 2.0 * T, 3.0 * T2, 4.0 * T3], dim=-1)
        dcp_dT = R * torch.sum(coef * Temp, dim=-1)

        return dcp_dT

    def cv_species_mol(self, T, idx):
        """
        Capacité calorifique molaire à volume constant pour certaines espèces.
        T : (ncells,)
        idx : indices Tensor (n_idx,) ou list
        Retourne :
            cv_molar : (ncells, n_idx), J/mol/K
        """
        # cp_molar
        cp_species_molar = self.cp_species(T, idx)  # (ncells, n_idx)

        # Cv = Cp - R  (molaire)
        cv_species_molar = cp_species_molar - R

        return cv_species_molar

    def dcv_species_dT(self, T, idx):
        """
        Dérivée par rapport à T de cv des espèces (J/mol/K^2).
        Puisque cv = cp - R, on a dcv/dT = dcp/dT.
        T : (ncells,)
        idx : indices des espèces
        Retourne :
            dcv_dT : (ncells, n_idx)
        """
        return self.dcp_species_dT(T, idx)


    def h_species(self, T, idx, P=Pref):
        """
        Enthalpie molaire (J/mol) pour les espèces d'indice idx.
        T : (ncells,)
        idx : indices Tensor (n_idx,) ou list
        """
        T = T[:, None]
        coef = self._select_coef(T)[:, idx, :6]  # (ncells, n_idx, 6)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([
            torch.ones_like(T),
            T / 2,
            T_2 / 3,
            T_3 / 4,
            T_4 / 5,
            1 / T
        ], dim=-1)  # (ncells, 1, 6)

        h_species_molar = R * T * torch.sum(coef * Temp, dim=-1)  # (ncells, n_idx)
        return h_species_molar

    def h_species_mol(self, T):
        # T : (ncells,)

        T = T[:, None]
        coef = self._select_coef(T)[:, :, :6]  # (ncells, nspecies, 6)

        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T / 2, T_2 / 3, T_3 / 4, T_4 / 5, 1 / T], dim=-1)  # (ncells, 1, 6)

        # h_species_molar (J/mol)
        h_species_molar = R * T * torch.sum(coef * Temp, dim=-1)

        return h_species_molar  # (ncells,)

    def u_species_mol(self, T):
        # T : (ncells,)

        T = T[:, None]
        coef = self._select_coef(T)[:, :, :6]  # (ncells, nspecies, 6)

        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T / 2, T_2 / 3, T_3 / 4, T_4 / 5, 1 / T], dim=-1)  # (ncells, 1, 6)

        # h_species_molar (J/mol)
        h_species_molar = R * T * torch.sum(coef * Temp, dim=-1)

        # u = h - R*T  (par espèce)
        u_species_molar = h_species_molar - R * T  # (ncells, nspecies)

        # mélange
        return u_species_molar  # (ncells,)

    def s_species_Pref(self, T, idx):
        """
        Entropie molaire (J/mol/K) pour les espèces d'indice idx.
        T : (ncells,)
        P : (ncells,) ou scalaire
        idx : indices Tensor (n_idx,) ou list
        """
        T = T[:, None]
        coef = self._select_coef(T)[:, idx, :]  # (ncells, n_idx, 7)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([
            torch.log(T),
            T,
            T_2 / 2,
            T_3 / 3,
            T_4 / 4,
            torch.zeros_like(T),
            torch.ones_like(T)
        ], dim=-1)

        s_species_molar = torch.sum(R * coef * Temp, dim=-1)
        return s_species_molar

    def g_species_Pref(self, T, P, idx):
        """
        Énergie libre molaire de Gibbs (J/mol).
        """
        h = self.h_species(T, idx, P)
        s = self.s_species_Pref(T, idx)
        return h - T[:, None] * s


if __name__ == '__main__':
    import time
    dtype = torch.float64
    device = 'cpu'
    y = yamlParser(r'../../vectorized_thermo_network_chemestry_gpu/gri30.yaml')
    vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device, reducedSpeciesSet=["H2", "H", "O", "O2", "OH", "H2O", "HO2", "CH2", "CH3", "CH4", "CO", "CO2", "C2H2", "N2", "bin"])
    #print(th.Species)
    print(vec_th.species)
    cells = 100000
    x = torch.arange(cells, dtype=dtype, device=device)
    T = 300.0 + 0.0 * x
    P = 101325.0 + 0.0 * x
    Y = torch.zeros((cells, vec_th.nspecies), dtype=dtype, device=device)
    Y[:,0] = 1.0
    t0 = time.time()
    print(vec_th.cp_mix(T, Y))
    print(time.time() - t0)

    print(vec_th.molar_masses)