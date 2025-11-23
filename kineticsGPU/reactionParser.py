import cantera as ct
from kineticsGPU.reactionTypes import ElementaryReaction, ThreeBodyReaction, FalloffReaction

def parseReactions(yaml_file):
    gas = ct.Solution(yaml_file)
    reactions_info = []

    for i, rxn in enumerate(gas.reactions()):
        info = {}
        info["index"] = i
        info["equation"] = rxn.equation
        info["reactants"] = rxn.reactants
        info["products"] = rxn.products
        info["reversible"] = rxn.reversible

        # === Détermination des unités globales ===
        units = rxn.input_data.units.units
        L_unit = units["length"]
        Q_unit = units["quantity"]
        E_unit = units["activation-energy"]

        # === Facteurs de conversion vers SI ===
        L_factor = 1.0 if L_unit == "m" else 1e-2 if L_unit == "cm" else None
        if L_factor is None:
            raise ValueError(f"Unité de longueur non supportée: {L_unit}")

        Q_factor = 1.0 if Q_unit == "mol" else 1e3 if Q_unit == "kmol" else None
        if Q_factor is None:
            raise ValueError(f"Unité de quantité non supportée: {Q_unit}")

        E_factor = {
            "J / mol": 1.0,
            "cal / mol": 4.184,
            "kcal / mol": 4184,
            "J / kmol": 1e-3,
        }.get(E_unit, None)
        if E_factor is None:
            raise ValueError(f"Unité d’énergie non supportée: {E_unit}")

        # === Ordre de réaction ===
        n_react = sum(rxn.reactants.values())

        # === Fonction de conversion d’A vers SI ===
        def convert_A_to_SI(A, n_react):
            # A * (L_factor^3 / Q_factor)^(n_react - 1)
            # car concentration = Q/L^3
            return A * (L_factor ** (3 * (n_react - 1))) / (Q_factor ** (n_react - 1))

        # === Cas de réaction ===
        if rxn.reaction_type == 'Arrhenius':
            info["type"] = "elementary"
            A = rxn.rate.pre_exponential_factor
            b = rxn.rate.temperature_exponent
            Ea = rxn.rate.activation_energy

            info["A_SI"] = convert_A_to_SI(A, n_react)
            info["b"] = b
            info["Ea_SI"] = Ea * E_factor

        elif rxn.reaction_type == 'three-body-Arrhenius':
            info["type"] = "threebody"
            A = rxn.rate.pre_exponential_factor
            b = rxn.rate.temperature_exponent
            Ea = rxn.rate.activation_energy

            defaultEff = rxn.third_body.default_efficiency

            info["A_SI"] = convert_A_to_SI(A, n_react + 1)  # +1 car molécule tierce
            info["b"] = b
            info["Ea_SI"] = Ea * E_factor
            info["efficiencies"] = rxn.third_body.efficiencies
            info['defaultEff'] = defaultEff

        elif 'falloff' in rxn.reaction_type:
            if rxn.reaction_type == 'falloff-Troe':
                info["type"] = "falloff-troe"
                troe = rxn.input_data['Troe']
                info["Troe_params"] = troe
            elif rxn.reaction_type == 'falloff-Lindemann':
                info["type"] = "falloff-lindemann"
            else:
                raise NotImplementedError(rxn.reaction_type)

            k0 = rxn.input_data['low-P-rate-constant']
            kinf = rxn.input_data['high-P-rate-constant']

            info["k0"] = {
                "A_SI": convert_A_to_SI(k0['A'], n_react + 1),
                "b": k0['b'],
                "Ea_SI": k0['Ea'] * E_factor,
            }
            info["kinf"] = {
                "A_SI": convert_A_to_SI(kinf['A'], n_react),
                "b": kinf['b'],
                "Ea_SI": kinf['Ea'] * E_factor,
            }
            info["efficiencies"] = rxn.third_body.efficiencies
            defaultEff = rxn.third_body.default_efficiency
            info['defaultEff'] = defaultEff


        else:
            raise TypeError(f"Type de réaction non supporté : {rxn.reaction_type}")

        reactions_info.append(info)

    return reactions_info


def build_reactions(species_names, reactions_info, thermo, device, dtype):
    """
    Construit la liste des réactions (Elementary, ThreeBody, Falloff)
    à partir des données Cantera parsées.
    """
    elem_reactions = []
    tb_reac = []
    falloff_reac = []

    system = []

    for r in reactions_info:
        rtype = r.get("type", "elementary").lower()

        # indices + coefficients stœchiométriques
        reactants_idx = [species_names.index(s) for s in r["reactants"].keys()]
        reactants_sto = list(r["reactants"].values())
        products_idx = [species_names.index(s) for s in r["products"].keys()]
        prod_sto = list(r["products"].values())

        oneWay = not r.get("reversible", True)

        # === 1. Elementary ===
        if rtype == "elementary":
            rxn = ElementaryReaction(
                reactants_idx=reactants_idx,
                products_idx=products_idx,
                reactants_sto=reactants_sto,
                products_sto=prod_sto,
                thermo=thermo,
                species=species_names,
                oneWay=oneWay,
                A=r["A_SI"],
                n=r["b"],
                Ea=r["Ea_SI"],
                device=device,
                dtype=dtype,
            )
            elem_reactions.append(rxn)

        # === 2. Three-body ===
        elif rtype in ("three-body", "threebody"):
            eff = r.get("efficiencies", None)
            default_eff = r.get("defaultEff", 1.0)

            rxn = ThreeBodyReaction(
                reactants_idx=reactants_idx,
                products_idx=products_idx,
                reactants_sto=reactants_sto,
                products_sto=prod_sto,
                thermo=thermo,
                species=species_names,
                oneWay=oneWay,
                A=r["A_SI"],
                n=r["b"],
                Ea=r["Ea_SI"],
                efficiencies=eff,
                defaultEff=default_eff,
                device=device,
                dtype=dtype,
            )
            tb_reac.append(rxn)
            eff = {thermo.species.index(spe): v for spe, v in eff.items()}


        # === 3. Falloff (Troe ou Lindemann) ===
        elif "falloff" in rtype:
            # Paramètres basse et haute pression
            k0_params = (r["k0"]["A_SI"], r["k0"]["b"], r["k0"]["Ea_SI"])
            kinf_params = (r["kinf"]["A_SI"], r["kinf"]["b"], r["kinf"]["Ea_SI"])

            eff = r.get("efficiencies", None)
            default_eff = r.get("defaultEff", 1.0)

            # Type spécifique (troe ou lindemann)
            if "troe" in rtype:
                falloff_type = "Troe"
                falloff_params = r.get("Troe_params", {})
            else:
                falloff_type = "Lindemann"
                falloff_params = {}

            rxn = FalloffReaction(
                reactants_idx=reactants_idx,
                products_idx=products_idx,
                reactants_sto=reactants_sto,
                products_sto=prod_sto,
                thermo=thermo,
                species=species_names,
                oneWay=oneWay,
                k0_params=k0_params,
                kinf_params=kinf_params,
                efficiencies=eff,
                defaultEff=default_eff,
                falloff_type=falloff_type,
                falloff_params=falloff_params,
                device=device,
                dtype=dtype,
            )
            falloff_reac.append(rxn)


        else:
            print(f"[WARN] Type de réaction non reconnu : {rtype}, ignorée.")
            continue

    return elem_reactions, tb_reac, falloff_reac, system