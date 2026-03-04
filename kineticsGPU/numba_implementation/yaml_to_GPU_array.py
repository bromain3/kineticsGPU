import numpy as np
from numba import cuda


# ─────────────────────────────────────────────────────────────────────────────
#  Sentinelles falloff
# ─────────────────────────────────────────────────────────────────────────────
FALLOFF_LINDEMANN = 0
FALLOFF_TROE      = 1


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _build_connectivity(reactions, th_object):
    """
    Construit les arrays de connectivité (indices + stoéchiométrie + delta_n)
    communs à tous les types de réactions.

    Retourne des listes Python (pas encore sur GPU).
    Ne touche pas aux paramètres Arrhenius.
    """
    react_idx, react_sto, reactant_number, reactant_offset = [], [], [], []
    prod_idx,  prod_sto,  prod_number,   prod_offset       = [], [], [], []
    delta_n = []

    r_off = p_off = 0

    for r in reactions:
        # --- réactants ---
        reactants = r["reactants"]
        reactant_number.append(len(reactants))
        reactant_offset.append(r_off)
        nu_r = 0.0
        for spe, sto in reactants.items():
            react_idx.append(th_object.species.index(spe))
            react_sto.append(sto)
            nu_r += sto
        r_off += len(reactants)

        # --- produits ---
        products = r["products"]
        prod_number.append(len(products))
        prod_offset.append(p_off)
        nu_p = 0.0
        for spe, sto in products.items():
            prod_idx.append(th_object.species.index(spe))
            prod_sto.append(sto)
            nu_p += sto
        p_off += len(products)

        delta_n.append(nu_p - nu_r)

    return (
        react_idx, react_sto, reactant_number, reactant_offset,
        prod_idx,  prod_sto,  prod_number,   prod_offset,
        delta_n,
    )


def _build_arrhenius(reactions):
    """
    Extrait les paramètres Arrhenius et le flag oneWay.
    Applicable uniquement aux réactions elementary et three-body
    (qui ont A_SI, b, Ea_SI au niveau racine).
    """
    A_arr  = [r["A_SI"]  for r in reactions]
    n_arr  = [r["b"]     for r in reactions]
    Ea_arr = [r["Ea_SI"] for r in reactions]
    oneWay = [0.0 if r["reversible"] else 1.0 for r in reactions]
    return A_arr, n_arr, Ea_arr, oneWay


def _build_efficiency_matrix(reactions, th_object):
    """
    Construit la matrice d'efficiences de taille (n_reactions, n_species).

    Pour chaque réaction r et chaque espèce s :
      - effmatrix[r, s] = valeur spécifique si précisée dans r["efficiencies"]
      - effmatrix[r, s] = r["defaultEff"]  sinon
    """
    nsp   = len(th_object.species)
    nreac = len(reactions)
    effmatrix = np.empty((nreac, nsp), dtype=np.float64)

    for i, r in enumerate(reactions):
        default = r.get("defaultEff", 1.0)
        effmatrix[i, :] = default

        for spe, val in r.get("efficiencies", {}).items():
            idx = th_object.species.index(spe)
            effmatrix[i, idx] = val

    return effmatrix


def _connectivity_to_gpu(conn):
    """
    Transfère sur GPU les arrays de connectivité (sans Arrhenius).
    conn : tuple retourné par _build_connectivity.
    """
    (react_idx, react_sto, reactant_number, reactant_offset,
     prod_idx,  prod_sto,  prod_number,   prod_offset, delta_n) = conn

    def d(arr, dt): return cuda.to_device(np.ascontiguousarray(arr, dtype=dt))

    return (
        d(react_idx,       np.int32),
        d(react_sto,       np.float64),
        d(reactant_number, np.int32),
        d(reactant_offset, np.int32),
        d(prod_idx,        np.int32),
        d(prod_sto,        np.float64),
        d(prod_number,     np.int32),
        d(prod_offset,     np.int32),
        d(delta_n,         np.float64),
    )


def _arrhenius_to_gpu(arrhenius):
    """
    Transfère sur GPU les paramètres Arrhenius + oneWay.
    arrhenius : tuple retourné par _build_arrhenius.
    """
    A_arr, n_arr, Ea_arr, oneWay = arrhenius

    def d(arr, dt): return cuda.to_device(np.ascontiguousarray(arr, dtype=dt))

    return (
        d(A_arr,  np.float64),
        d(n_arr,  np.float64),
        d(Ea_arr, np.float64),
        d(oneWay, np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  API publique
# ─────────────────────────────────────────────────────────────────────────────

def build_elementary_gpu_arrays(reactions, th_object):
    """
    Retourne les arrays GPU pour les réactions élémentaires.

    Ordre :
      react_idx, react_sto, reactant_number, reactant_offset,
      prod_idx,  prod_sto,  prod_number,  prod_offset,
      A, n, Ea, oneWay, delta_n
    """
    elem = [r for r in reactions if r["type"] == "elementary"]

    conn      = _build_connectivity(elem, th_object)
    arrhenius = _build_arrhenius(elem)

    (ri, rs, rn, ro,
     pi, ps, pn, po, dn) = _connectivity_to_gpu(conn)

    (A_d, n_d, Ea_d, ow_d) = _arrhenius_to_gpu(arrhenius)

    return (ri, rs, rn, ro,
            pi, ps, pn, po,
            A_d, n_d, Ea_d, ow_d, dn)


def build_threebody_gpu_arrays(reactions, th_object):
    """
    Retourne les arrays GPU pour les réactions three-body.

    Ordre : même que elementary + effmatrix (nreac, nsp) en dernier.
    """
    tb = [r for r in reactions if r["type"] in ("three-body", "threebody")]

    conn      = _build_connectivity(tb, th_object)
    arrhenius = _build_arrhenius(tb)

    (ri, rs, rn, ro,
     pi, ps, pn, po, dn) = _connectivity_to_gpu(conn)

    (A_d, n_d, Ea_d, ow_d) = _arrhenius_to_gpu(arrhenius)

    effmatrix_d = cuda.to_device(_build_efficiency_matrix(tb, th_object))

    return (ri, rs, rn, ro,
            pi, ps, pn, po,
            A_d, n_d, Ea_d, ow_d, dn,
            effmatrix_d)


def build_falloff_gpu_arrays(reactions, th_object):
    """
    Retourne les arrays GPU pour les réactions falloff (Lindemann + Troe).

    Ordre :
      react_idx, react_sto, reactant_number, reactant_offset,
      prod_idx,  prod_sto,  prod_number,  prod_offset,
      A0, n0, Ea0,          (k0   — basse pression)
      Ainf, ninf, Eainf,    (kinf — haute pression)
      oneWay, delta_n,
      effmatrix,            (nreac, nsp)
      falloff_type,         (int32 : 0=Lindemann, 1=Troe)
      troe_A, troe_T1, troe_T2, troe_T3
    """
    falloff = [r for r in reactions if "falloff" in r["type"]]

    def d(arr, dt): return cuda.to_device(np.ascontiguousarray(arr, dtype=dt))

    # ── connectivité ─────────────────────────────────────────────────────────
    conn = _build_connectivity(falloff, th_object)
    (ri, rs, rn, ro,
     pi, ps, pn, po, dn) = _connectivity_to_gpu(conn)

    # ── Arrhenius k0 / kinf ──────────────────────────────────────────────────
    A0_arr    = [r["k0"]["A_SI"]    for r in falloff]
    n0_arr    = [r["k0"]["b"]       for r in falloff]
    Ea0_arr   = [r["k0"]["Ea_SI"]   for r in falloff]
    Ainf_arr  = [r["kinf"]["A_SI"]  for r in falloff]
    ninf_arr  = [r["kinf"]["b"]     for r in falloff]
    Eainf_arr = [r["kinf"]["Ea_SI"] for r in falloff]

    oneWay_list = [0.0 if r["reversible"] else 1.0 for r in falloff]

    # ── efficiencies ─────────────────────────────────────────────────────────
    effmatrix_d = cuda.to_device(_build_efficiency_matrix(falloff, th_object))

    # ── type falloff + params Troe ────────────────────────────────────────────
    falloff_type_list = []
    troe_A_list, troe_T1_list, troe_T2_list, troe_T3_list = [], [], [], []

    for r in falloff:
        if "troe" in r["type"]:
            falloff_type_list.append(FALLOFF_TROE)
            p = r.get("Troe_params", {})
            troe_A_list .append(float(p.get("A",  0.0)))
            troe_T1_list.append(float(p.get("T1", 0.0)))
            troe_T2_list.append(float(p.get("T2", 0.0)))
            troe_T3_list.append(float(p.get("T3", 0.0)))
        else:  # Lindemann
            falloff_type_list.append(FALLOFF_LINDEMANN)
            troe_A_list .append(0.0)
            troe_T1_list.append(0.0)
            troe_T2_list.append(0.0)
            troe_T3_list.append(0.0)

    return (
        # connectivité
        ri, rs, rn, ro,
        pi, ps, pn, po,
        # k0
        d(A0_arr,    np.float64), d(n0_arr,    np.float64), d(Ea0_arr,   np.float64),
        # kinf
        d(Ainf_arr,  np.float64), d(ninf_arr,  np.float64), d(Eainf_arr, np.float64),
        # flags
        d(oneWay_list,       np.float64),
        dn,                              # delta_n — déjà sur GPU
        # efficiencies
        effmatrix_d,
        # falloff type + Troe params
        d(falloff_type_list, np.int32),
        d(troe_A_list,  np.float64), d(troe_T1_list, np.float64),
        d(troe_T2_list, np.float64), d(troe_T3_list, np.float64),
    )