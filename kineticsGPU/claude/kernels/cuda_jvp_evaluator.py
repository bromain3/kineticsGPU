"""
cuda_jvp_evaluator.py — Wrapper standalone pour tester _eval_jvp_pb + _adiabatic_jvp_pb.

Architecture : 1 bloc = 1 réacteur (même que l'intégrateur per-block).
Lance le kernel jvp_eval_kernel qui appelle :
  1. _eval_rhs_and_context_pb  — calcule et met en cache les quantités cinétiques
  2. _eval_jvp_pb              — JVP de omega (mol/m³/s)
  3. _adiabatic_jvp_pb         — JVP du RHS thermodynamique complet

Outputs par réacteur :
  Jv_om  (n_sp,)    — JVP du vecteur source chimique
  Jv_TC  (n_sp+1,)  — JVP du RHS adiabatique complet (dT/dt, dC/dt)

Usage
-----
ev = CUDAJVPEvaluator(mech, dtype=np.float64)
Jv_om, Jv_TC = ev.evaluate(T_np, C_np, vT_np, vC_np)
"""

import math
import numpy as np
from numba import cuda, float64, int32

# Import des device-functions depuis le module per-block existant.
# Elles sont déclarées @cuda.jit(device=True) donc réutilisables directement.
from kineticsGPU.claude.kernels.chemistry_kernels_pb import (
    _eval_rhs_and_context_pb,
    _eval_jvp_pb,
    _adiabatic_jvp_pb,
    MAX_N_STATE,
    MAX_R,
    THREADS_PER_BLOCK,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Kernel principal — 1 bloc = 1 réacteur
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(cache=True)
def jvp_eval_kernel(
    T_arr, C_arr,          # (B,) et (B, n_sp)  — état de base
    vT_arr, vC_arr,        # (B,) et (B, n_sp)  — vecteur de perturbation
    out_Jv_om,             # (B, n_sp)           — sortie JVP omega
    out_Jv_TC,             # (B, n_sp+1)         — sortie JVP RHS complet
    # ── Buffers contexte cinétique (1 ligne par réacteur) ─────────────────
    ctx_kf_e,   ctx_Kc_e,  ctx_cr_e,  ctx_cp_e,  ctx_dG0_e,
    ctx_sRre,   ctx_sRpe,
    ctx_kf_tb,  ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb,  ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f,   ctx_Kc_f,  ctx_cr_f,  ctx_cp_f,  ctx_dG0_f,
    ctx_sRrf,   ctx_sRpf,
    ctx_k0_f,   ctx_kinf_f, ctx_M_f,  ctx_Pr_f,  ctx_F_f,   ctx_kb_f,
    ctx_Fc_f,   ctx_lFc_f, ctx_Cp_f,  ctx_Np_f,  ctx_f1_f,  ctx_exp_f,
    # ── Données mécanisme ─────────────────────────────────────────────────
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b         = cuda.blockIdx.x
    tid       = cuda.threadIdx.x
    n_threads = cuda.blockDim.x

    if b >= T_arr.shape[0]:
        return

    n_state = n_sp + 1

    # ── Shared memory ────────────────────────────────────────────────────────
    sh_C       = cuda.shared.array(MAX_N_STATE, float64)
    sh_vC      = cuda.shared.array(MAX_N_STATE, float64)
    sh_om      = cuda.shared.array(MAX_N_STATE, float64)  # omega (RHS)
    sh_dTC     = cuda.shared.array(MAX_N_STATE, float64)  # RHS adiabatique (non utilisé ici mais requis)
    sh_Jv_om   = cuda.shared.array(MAX_N_STATE, float64)  # sortie JVP omega
    sh_Jv_TC   = cuda.shared.array(MAX_N_STATE, float64)  # sortie JVP RHS complet
    sh_scratch = cuda.shared.array(THREADS_PER_BLOCK, float64)

    # ── Charger l'état de base et le vecteur v ───────────────────────────────
    T_loc  = T_arr[b]
    vT_loc = vT_arr[b]

    i = tid
    while i < n_sp:
        sh_C[i]  = C_arr[b, i]
        sh_vC[i] = vC_arr[b, i]
        i += n_threads
    cuda.syncthreads()

    # ── Slices contexte pour ce réacteur ────────────────────────────────────
    e_kf   = ctx_kf_e[b];   e_Kc  = ctx_Kc_e[b]
    e_cr   = ctx_cr_e[b];   e_cp  = ctx_cp_e[b];  e_dG  = ctx_dG0_e[b]
    e_sRr  = ctx_sRre[b];   e_sRp = ctx_sRpe[b]

    tb_kf  = ctx_kf_tb[b];  tb_Kc  = ctx_Kc_tb[b]
    tb_cr  = ctx_cr_tb[b];  tb_cp  = ctx_cp_tb[b]; tb_dG = ctx_dG0_tb[b]
    tb_sRr = ctx_sRrtb[b];  tb_sRp = ctx_sRptb[b]
    tb_bRR = ctx_bRR_tb[b]; tb_M   = ctx_M_tb[b]

    f_kf   = ctx_kf_f[b];   f_Kc   = ctx_Kc_f[b]
    f_cr   = ctx_cr_f[b];   f_cp   = ctx_cp_f[b];  f_dG  = ctx_dG0_f[b]
    f_sRr  = ctx_sRrf[b];   f_sRp  = ctx_sRpf[b]
    f_k0   = ctx_k0_f[b];   f_ki   = ctx_kinf_f[b]
    f_M    = ctx_M_f[b];    f_Pr   = ctx_Pr_f[b]
    f_F    = ctx_F_f[b];    f_kb   = ctx_kb_f[b]
    f_Fc   = ctx_Fc_f[b];   f_lFc  = ctx_lFc_f[b]
    f_Cp   = ctx_Cp_f[b];   f_Np   = ctx_Np_f[b]
    f_f1   = ctx_f1_f[b];   f_exp  = ctx_exp_f[b]

    # ── Étape 1 : évaluer le RHS + remplir les caches cinétiques ────────────
    _eval_rhs_and_context_pb(
        tid, n_threads,
        T_loc, sh_C,
        e_kf, e_Kc, e_cr, e_cp, e_dG, e_sRr, e_sRp,
        tb_kf, tb_Kc, tb_cr, tb_cp, tb_dG, tb_sRr, tb_sRp, tb_bRR, tb_M,
        f_kf, f_Kc, f_cr, f_cp, f_dG, f_sRr, f_sRp,
        f_k0, f_ki, f_M, f_Pr, f_F, f_kb,
        f_Fc, f_lFc, f_Cp, f_Np, f_f1, f_exp,
        sh_om, n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # ── Étape 2 : JVP de omega (sh_Jv_om) ───────────────────────────────────
    _eval_jvp_pb(
        tid, n_threads,
        T_loc, sh_C, vT_loc, sh_vC, sh_Jv_om,
        e_kf, e_Kc, e_cr, e_cp, e_dG, e_sRr, e_sRp,
        tb_kf, tb_Kc, tb_cr, tb_cp, tb_dG, tb_sRr, tb_sRp, tb_bRR, tb_M,
        f_kf, f_Kc, f_cr, f_cp, f_dG, f_sRr, f_sRp,
        f_k0, f_ki, f_M, f_Pr, f_F, f_kb,
        f_Fc, f_lFc, f_Cp, f_Np, f_f1, f_exp,
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )

    # ── Étape 3 : JVP adiabatique complète (sh_Jv_TC) ───────────────────────
    _adiabatic_jvp_pb(
        tid, n_threads,
        T_loc, sh_C, vT_loc, sh_vC,
        sh_om, sh_Jv_om,
        n_sp, thL, thH, thM,
        sh_Jv_TC, sh_scratch,
    )

    # ── Écriture en mémoire globale ──────────────────────────────────────────
    i = tid
    while i < n_sp:
        out_Jv_om[b, i] = sh_Jv_om[i]
        i += n_threads
    i = tid
    while i < n_state:
        out_Jv_TC[b, i] = sh_Jv_TC[i]
        i += n_threads
    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper Python
# ─────────────────────────────────────────────────────────────────────────────

class CUDAJVPEvaluator:
    """
    Évalue la JVP du RHS adiabatique via les kernels per-block.

    La JVP est calculée analytiquement (pas de différences finies) en réutilisant
    exactement les mêmes device-functions que l'intégrateur Newton-GMRES.

    Parameters
    ----------
    mech : MechanismData
        Données mécanisme compilées (build_mechanism_data_from_torch).
    dtype : np.dtype
        np.float64 (default).

    Usage
    -----
    ev = CUDAJVPEvaluator(mech)
    Jv_om, Jv_TC = ev.evaluate(T_np, C_np, vT_np, vC_np)
    """

    def __init__(self, mech, dtype=np.float64):
        self.mech  = mech
        self.dtype = dtype
        self._d    = mech.to_cuda_arrays()
        self._alloc_ctx_buffers()

    # ── Allocation des buffers contexte (mêmes dimensions que l'intégrateur) ─

    def _alloc_ctx_buffers(self):
        m  = self.mech
        dt = self.dtype
        e  = m.elementary
        tb = m.threebody
        fa = m.falloff
        # Taille maximale : on sur-alloue à 1 réacteur, redimensionné dans evaluate()
        # L'allocation réelle est faite à chaque appel pour être correct sur N variable.
        pass  # lazy allocation dans evaluate()

    def _make_ctx(self, B):
        """Alloue les buffers contexte cinétique pour B réacteurs."""
        m  = self.mech
        dt = self.dtype
        e  = m.elementary  if m.elementary  else None
        tb = m.threebody   if m.threebody   else None
        fa = m.falloff     if m.falloff     else None

        n_e  = e.n_reactions  if e  else 1
        n_tb = tb.n_reactions if tb else 1
        n_f  = fa.n_reactions if fa else 1

        def _da(shape): return cuda.device_array(shape, dtype=dt)

        return {
            'kf_e':   _da((B, n_e)),
            'Kc_e':   _da((B, n_e)),
            'cr_e':   _da((B, n_e)),
            'cp_e':   _da((B, n_e)),
            'dG0_e':  _da((B, n_e)),
            'sRre':   _da((B, e.react_offset[-1] if e else 1)),
            'sRpe':   _da((B, e.prod_offset[-1]  if e else 1)),

            'kf_tb':  _da((B, n_tb)),
            'Kc_tb':  _da((B, n_tb)),
            'cr_tb':  _da((B, n_tb)),
            'cp_tb':  _da((B, n_tb)),
            'dG0_tb': _da((B, n_tb)),
            'sRrtb':  _da((B, tb.react_offset[-1] if tb else 1)),
            'sRptb':  _da((B, tb.prod_offset[-1]  if tb else 1)),
            'bRR_tb': _da((B, n_tb)),
            'M_tb':   _da((B, n_tb)),

            'kf_f':   _da((B, n_f)),
            'Kc_f':   _da((B, n_f)),
            'cr_f':   _da((B, n_f)),
            'cp_f':   _da((B, n_f)),
            'dG0_f':  _da((B, n_f)),
            'sRrf':   _da((B, fa.react_offset[-1] if fa else 1)),
            'sRpf':   _da((B, fa.prod_offset[-1]  if fa else 1)),
            'k0_f':   _da((B, n_f)),
            'kinf_f': _da((B, n_f)),
            'M_f':    _da((B, n_f)),
            'Pr_f':   _da((B, n_f)),
            'F_f':    _da((B, n_f)),
            'kb_f':   _da((B, n_f)),
            'Fc_f':   _da((B, n_f)),
            'lFc_f':  _da((B, n_f)),
            'Cp_f':   _da((B, n_f)),
            'Np_f':   _da((B, n_f)),
            'f1_f':   _da((B, n_f)),
            'exp_f':  _da((B, n_f)),
        }

    def _pack_mech_args(self):
        """Retourne la liste ordonnée des arrays mécanisme."""
        d  = self._d
        m  = self.mech
        dt = self.dtype

        def _g(k):  return d.get(k, cuda.device_array((1,), dt))
        def _gi(k): return d.get(k, cuda.device_array((1,), np.int32))

        return [
            np.int32(m.n_species),
            d['thermo_coef_low'], d['thermo_coef_high'], d['thermo_Tmid'],
            # Elementary
            np.int32(m.elementary.n_reactions if m.elementary else 0),
            _g('e_A'), _g('e_n'), _g('e_Ea'), _g('e_oneWay'), _g('e_delta_n'),
            _gi('e_react_idx'), _g('e_react_sto'), _gi('e_react_offset'),
            _gi('e_prod_idx'),  _g('e_prod_sto'),  _gi('e_prod_offset'),
            # Three-body
            np.int32(m.threebody.n_reactions if m.threebody else 0),
            _g('tb_A'), _g('tb_n'), _g('tb_Ea'), _g('tb_oneWay'), _g('tb_delta_n'),
            _gi('tb_react_idx'), _g('tb_react_sto'), _gi('tb_react_offset'),
            _gi('tb_prod_idx'),  _g('tb_prod_sto'),  _gi('tb_prod_offset'),
            _g('tb_eff_matrix'),
            # Falloff
            np.int32(m.falloff.n_reactions if m.falloff else 0),
            _g('f_A_k0'),   _g('f_b_k0'),   _g('f_Ea_k0'),
            _g('f_A_kinf'), _g('f_b_kinf'), _g('f_Ea_kinf'),
            _g('f_oneWay'), _g('f_delta_n'),
            _g('f_troe_A'), _g('f_troe_T1'), _g('f_troe_T2'), _g('f_troe_T3'),
            _gi('f_is_troe'),
            _gi('f_react_idx'), _g('f_react_sto'), _gi('f_react_offset'),
            _gi('f_prod_idx'),  _g('f_prod_sto'),  _gi('f_prod_offset'),
            _g('f_eff_matrix'),
        ]

    # ── Interface publique ────────────────────────────────────────────────────

    def evaluate(self, T_np, C_np, vT_np, vC_np):
        """
        Évalue la JVP analytique du RHS adiabatique.

        Parameters
        ----------
        T_np  : (B,)       float64 — températures (K)
        C_np  : (B, n_sp)  float64 — concentrations (mol/m³)
        vT_np : (B,)       float64 — composante T du vecteur de perturbation
        vC_np : (B, n_sp)  float64 — composante C du vecteur de perturbation

        Returns
        -------
        Jv_om : (B, n_sp)   float64 — JVP du vecteur source chimique omega
        Jv_TC : (B, n_sp+1) float64 — JVP du RHS adiabatique complet
                                       [:, 0]  = JVP de dT/dt
                                       [:, 1:] = JVP de dC/dt (= Jv_om)
        """
        dt = self.dtype
        B  = int(T_np.shape[0])
        m  = self.mech

        d_T  = cuda.to_device(np.ascontiguousarray(T_np,  dtype=dt))
        d_C  = cuda.to_device(np.ascontiguousarray(C_np,  dtype=dt))
        d_vT = cuda.to_device(np.ascontiguousarray(vT_np, dtype=dt))
        d_vC = cuda.to_device(np.ascontiguousarray(vC_np, dtype=dt))

        d_Jv_om = cuda.device_array((B, m.n_species),     dtype=dt)
        d_Jv_TC = cuda.device_array((B, m.n_species + 1), dtype=dt)

        ctx = self._make_ctx(B)

        jvp_eval_kernel[B, THREADS_PER_BLOCK](
            d_T, d_C, d_vT, d_vC,
            d_Jv_om, d_Jv_TC,
            # Contexte cinétique
            ctx['kf_e'],  ctx['Kc_e'],  ctx['cr_e'],  ctx['cp_e'],  ctx['dG0_e'],
            ctx['sRre'],  ctx['sRpe'],
            ctx['kf_tb'], ctx['Kc_tb'], ctx['cr_tb'], ctx['cp_tb'], ctx['dG0_tb'],
            ctx['sRrtb'], ctx['sRptb'], ctx['bRR_tb'], ctx['M_tb'],
            ctx['kf_f'],  ctx['Kc_f'],  ctx['cr_f'],  ctx['cp_f'],  ctx['dG0_f'],
            ctx['sRrf'],  ctx['sRpf'],
            ctx['k0_f'],  ctx['kinf_f'], ctx['M_f'],  ctx['Pr_f'],  ctx['F_f'],
            ctx['kb_f'],  ctx['Fc_f'],  ctx['lFc_f'], ctx['Cp_f'],  ctx['Np_f'],
            ctx['f1_f'],  ctx['exp_f'],
            # Mécanisme
            *self._pack_mech_args(),
        )
        cuda.synchronize()

        return d_Jv_om.copy_to_host(), d_Jv_TC.copy_to_host()