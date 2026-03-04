from numba import cuda
import yaml
import math
import numpy as np
import kineticsGPU.numba_implementation.constants as constants

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
            sum([n * getattr(constants.MM, atom) for n, atom in s['composition']])
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

    def populateVectorizedThermoObject(self, reducedSpeciesSet=None):
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
                molar_masses.append(sum([num * getattr(constants.MM, atom) for num, atom in thermo_full[n]['composition']]))

        return ThermoData_vectorized(
            species_order, Tmin, Tmid, Tmax, coefs_low, coefs_high, molar_masses
        )


class ThermoData_vectorized:
    ##############
    ##### Retourne tout en massique
    #############
    def __init__(self, species_names, Tmin, Tmid, Tmax, coef_low, coef_high, molar_masses):
        self.species = species_names  # (nspecies,)
        self.Tmin = cuda.to_device(np.ascontiguousarray(Tmin))  # (nspecies,)
        self.Tmid = cuda.to_device(np.ascontiguousarray(Tmid))  # (nspecies,)
        self.Tmax = cuda.to_device(np.ascontiguousarray(Tmax))  # (nspecies,)
        self.coef_low = cuda.to_device(np.ascontiguousarray(coef_low))  # (nspecies, 6)
        self.coef_high = cuda.to_device(np.ascontiguousarray(coef_high))  # (nspecies, 6)
        self.molar_masses = cuda.to_device(np.ascontiguousarray(molar_masses))  # (nspecies,)
        self.nspecies = len(species_names)

    def package_values(self):
        return(self.coef_high, self.coef_low, self.Tmid)

###### kernels
@cuda.jit(device=True, inline=False)
def _coefs(coefs_high, coefs_low, Tmid, T):
    if T < Tmid:
        return coefs_low
    else:
        return coefs_high

@cuda.jit(device=True, inline=False)
def h_nasa7(coef_high, coef_low, T_mid, T):
    """Molar enthalpy J/mol from 6-coefficient NASA poly (no integration constant)."""
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    coef = _coefs(coef_high, coef_low, T_mid, T)
    return constants.R * T * (coef[0] + coef[1] * T / 2 + coef[2] * T2 / 3 + coef[3] * T3 / 4 + coef[4] * T4 / 5 + coef[5] / T)

@cuda.jit(device=True, inline=False)
def s_nasa7(coef_high, coef_low, T_mid, T):
    """Molar entropy J/(mol·K) at Pref_s from 7-coefficient NASA poly."""
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    coef = _coefs(coef_high, coef_low, T_mid, T)
    return constants.R * (coef[0]*math.log(T) + coef[1]*T + coef[2]*T2/2 + coef[3]*T3/3 + coef[4]*T4/4 + coef[6])

@cuda.jit(device=True, inline=False)
def g0_nasa7(coef_high, coef_low, T_mid, T):
    return(h_nasa7(coef_high, coef_low, T_mid, T) - T * s_nasa7(coef_high, coef_low, T_mid, T))

@cuda.jit(device=True, inline=False)
def cv_nasa7(coef_high, coef_low, T_mid, T):
    """Molar cv = cp - R  (J/(mol·K))."""
    T2 = T * T; T3 = T2 * T; T4 = T3 * T
    coef = _coefs(coef_high, coef_low, T_mid, T)
    cp = constants.R * (coef[0] + coef[1]*T + coef[2]*T2 + coef[3]*T3 + coef[4]*T4)
    return cp - constants.R

@cuda.jit(device=True, inline=False)
def u_nasa7(coef_high, coef_low, T_mid, T):
    """Molar internal energy u = h - RT  (J/mol)."""
    return h_nasa7(coef_high, coef_low, T_mid, T) - constants.R * T

@cuda.jit(device=True, inline=False)
def h0_nasa7(coefs_high, coefs_low, Tmid, T):
    a = _coefs(coefs_high, coefs_low, Tmid, T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    return constants.R * (
        a[0] * T
        + a[1] * 0.5   * T2
        + a[2] / 3.0   * T2
        + a[3] * 0.25  * T4
        + a[4] * 0.2   * T5
        + a[5]
    )

@cuda.jit(device=True, inline=False)
def s0_nasa7(coefs_high, coefs_low, Tmid, T):
    a = _coefs(coefs_high, coefs_low, Tmid, T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    return constants.R * (
        a[0] * math.log(T)
        + a[1] * T
        + a[2] * 0.5  * T2
        + a[3] / 3.0  * T3
        + a[4] * 0.25 * T4
        + a[6]
    )