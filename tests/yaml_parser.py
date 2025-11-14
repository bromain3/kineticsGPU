import cantera as ct
import yaml





#####################This code is not validated













def OneWayOnly(eq):
    if '<=>' in eq:
        return(False)
    else:
        return(True)

class yamlParser:
    def __init__(self, YAML_file):
        self.Yaml_file = YAML_file
        with open(self.Yaml_file, 'r') as f:
            self.yaml_data = yaml.load(f, Loader=yaml.BaseLoader)
        self.gas = ct.Solution(self.Yaml_file)
        self.thermo = self.getThermoData()
        self.reactions = self.getReactionData()
        self.Species = self.getSpecies()

    def getThermoData(self):
        res = []
        for spe in self.yaml_data['species']:
            res.append({'name': spe['name'], 'composition': [(int(v), k) for k,v in spe['composition'].items()],
                        'modelType': spe['thermo']['model'], 'trange': [float(k) for k in spe['thermo']['temperature-ranges']],
                        'data': [[float(k) for k in spe['thermo']['data'][i]] for i in range(len(spe['thermo']['data']))]})
        return(res)

    def populateThermoObject(self, th):
        for i in self.thermo:
            th.AddSpeciePorperty(i['name'], i['composition'], i['modelType'], TRange=i['trange'], data=i['data'])

    def populateReactor(self, r, th, T):
        reactionsSet = []
        for reac in self.reactions:
            if reac['Type'] == 'ER':
                reactionsSet.append(reactions.ElementaryReaction(reac['react'], reac['prod'], reac['A'], reac['b'], reac['Ea'], th, self.Species, OneWayOnly=reac['OWO']))
            elif reac['Type'] == 'TB':
                reactionsSet.append(reactions.ThreeBody(reac['react'], reac['prod'], reac['A'], reac['b'], reac['Ea'], reac['efficiencies'], th, self.Species, OneWayOnly=reac['OWO']))
            elif reac['Type'] == 'FO':
                reactionsSet.append(reactions.Falloff(reac['react'], reac['prod'], reac['low-P-rate-constant'], reac['high-P-rate-constant'], reac['efficiencies'], th, self.Species, Troe=reac['Troe']))
        return(r(self.Species, reactionsSet, th, T))


    def getReactionData(self):
        reactions = []
        for n, reac in enumerate(self.yaml_data['reactions']):
            if 'type' not in reac:
                reactions.append({'Type': 'ER','A':float(reac['rate-constant']['A']), 'b': float(reac['rate-constant']['b']), 'Ea':float(reac['rate-constant']['Ea']) , 'OWO': OneWayOnly(['equation']), 'react': [(float(v), k) for k,v in self.gas.reaction(n).reactants.items()],'prod':[(float(v), k) for k,v in self.gas.reaction(n).products.items()]})
            elif reac['type'] == 'three-body':
                if 'efficiencies' not in reac:
                    eff = {}
                else:
                    eff = {k:float(v) for k,v in reac['efficiencies'].items()}
                reactions.append({'Type': 'TB', 'OWO': OneWayOnly(['equation']), 'efficiencies': eff,
                                  'A': float(reac['rate-constant']['A']), 'b': float(reac['rate-constant']['b']),
                                  'Ea': float(reac['rate-constant']['Ea']),
                                  'react': [(float(v), k) for k, v in self.gas.reaction(n).reactants.items()],
                                  'prod': [(float(v), k) for k, v in self.gas.reaction(n).products.items()]})
            elif reac['type'] == 'falloff':
                if 'efficiencies' not in reac:
                    eff = {}
                else:
                    eff = {k:float(v) for k,v in reac['efficiencies'].items()}
                if 'Troe' not in reac:
                    tr = None
                else:
                    tr = {k:float(v) for k,v in reac['Troe'].items()}
                reactions.append({'Type': 'FO', 'OWO': OneWayOnly(['equation']), 'efficiencies': eff, 'Troe': tr,
                                  'low-P-rate-constant' : {k:float(v) for k,v in reac['low-P-rate-constant'].items()},
                                  'high-P-rate-constant': {k: float(v) for k, v in reac['high-P-rate-constant'].items()},
                                  'react': [(float(v), k) for k, v in self.gas.reaction(n).reactants.items()],
                                  'prod': [(float(v), k) for k, v in self.gas.reaction(n).products.items()]})
        return(reactions)

    def getSpecies(self):
        return(self.yaml_data['phases'][0]['species'])

if __name__ == '__main__':
    import thermo_properties
    y = yamlParser('air.yaml')
    th = thermo_properties.ThermoData()
    y.populateThermoObject(th)
    print(th.Species)