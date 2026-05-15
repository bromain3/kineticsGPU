"""
chemistry_kernels_article.py — Wrapper standalone pour validation (section RR_and_source_terms).

Utilise les device-functions per-block refactorisées au lieu de dupliquer la logique.
Architecture : 1 bloc = 1 réacteur (même que l'intégrateur per-block).

API publique inchangée :
    evaluator = CUDAReactionEvaluator(mech)
    RR_e, RR_tb, RR_f, omega = evaluator.evaluate(T_np, C_np)
"""

import math
import numpy as np
from numba import cuda, float64, int32

from kineticsGPU.cuda.kernels.per_block.chemistry_kernels_pb import (
    MAX_N_STATE,
    THREADS_PER_BLOCK,
)
from kineticsGPU.cuda.kernels.per_block.context_pb import _eval_rhs_and_context_pb


# ─────────────────────────────────────────────────────────────────────────────
#  Kernel principal — 1 bloc = 1 réacteur
#  Appelle _eval_rhs_and_context_pb pour remplir sh_om (= omega),
#  puis écrit les RR depuis les buffers contexte vers la mémoire globale.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(cache=True)
def eval_rr_and_omega_kernel(
    T_arr, C_arr,          # (B,) et (B, n_sp)
    out_RR_e,              # (B, n_e)
    out_RR_tb,             # (B, n_tb)
    out_RR_f,              # (B, n_f)
    out_omega,             # (B, n_sp)
    # ── Buffers contexte cinétique (une ligne par réacteur) ───────────────
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

    # ── Shared memory ────────────────────────────────────────────────────────
    sh_C  = cuda.shared.array(MAX_N_STATE, float64)
    sh_om = cuda.shared.array(MAX_N_STATE, float64)

    T_loc = T_arr[b]

    i = tid
    while i < n_sp:
        sh_C[i] = C_arr[b, i]
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

    # ── Évaluation RHS + remplissage des caches cinétiques ──────────────────
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

    # ── Écriture omega → mémoire globale ────────────────────────────────────
    i = tid
    while i < n_sp:
        out_omega[b, i] = sh_om[i]
        i += n_threads

    # ── Écriture RR depuis les buffers contexte → mémoire globale ───────────
    # Les ctx_*[b] contiennent déjà les taux nets calculés par _eval_rhs_and_context_pb.
    i = tid
    while i < n_e:
        out_RR_e[b, i] = e_kf[i] * (e_cr[i] - (1.0 - eOW[i]) * e_cp[i] / e_Kc[i])
        i += n_threads

    i = tid
    while i < n_tb:
        out_RR_tb[b, i] = tb_bRR[i] * tb_M[i]
        i += n_threads

    i = tid
    while i < n_f:
        out_RR_f[b, i] = f_kf[i] * (f_cr[i] - (1.0 - fOW[i]) * f_cp[i] / f_Kc[i])
        i += n_threads

    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper Python — API identique à l'ancienne version
# ─────────────────────────────────────────────────────────────────────────────

class CUDAReactionEvaluator:
    """
    Évalue les taux de réaction et omega sur un batch de (T, C).

    Usage
    -----
    evaluator = CUDAReactionEvaluator(mech)
    RR_e, RR_tb, RR_f, omega = evaluator.evaluate(T_np, C_np)
    # T_np  : (B,)       float64
    # C_np  : (B, n_sp)  float64  en mol/m³
    # RR_*  : (B, n_rxn) float64  en mol/m³/s
    # omega : (B, n_sp)  float64  en mol/m³/s
    """

    def __init__(self, mech, dtype=np.float64):
        self.mech  = mech
        self.dtype = dtype
        self._d    = mech.to_cuda_arrays()
        self._mech_args = self._pack_mech_args()

    def _make_ctx(self, B):
        """Alloue les buffers contexte cinétique pour B réacteurs."""
        m  = self.mech
        dt = self.dtype
        e  = m.elementary
        tb = m.threebody
        fa = m.falloff

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

    def evaluate(self, T_np: np.ndarray, C_np: np.ndarray):
        """
        Parameters
        ----------
        T_np : (B,)       float64
        C_np : (B, n_sp)  float64 — concentrations en mol/m³

        Returns
        -------
        RR_e   : (B, n_e)   float64
        RR_tb  : (B, n_tb)  float64
        RR_f   : (B, n_f)   float64
        omega  : (B, n_sp)  float64
        """
        m  = self.mech
        B  = int(T_np.shape[0])
        dt = self.dtype
        e  = m.elementary
        tb = m.threebody
        fa = m.falloff

        n_e  = e.n_reactions  if e  else 0
        n_tb = tb.n_reactions if tb else 0
        n_f  = fa.n_reactions if fa else 0

        d_T     = cuda.to_device(np.ascontiguousarray(T_np, dtype=dt))
        d_C     = cuda.to_device(np.ascontiguousarray(C_np, dtype=dt))
        d_RR_e  = cuda.device_array((B, max(n_e,  1)), dtype=dt)
        d_RR_tb = cuda.device_array((B, max(n_tb, 1)), dtype=dt)
        d_RR_f  = cuda.device_array((B, max(n_f,  1)), dtype=dt)
        d_omega = cuda.device_array((B, m.n_species),  dtype=dt)

        ctx = self._make_ctx(B)

        eval_rr_and_omega_kernel[B, THREADS_PER_BLOCK](
            d_T, d_C,
            d_RR_e, d_RR_tb, d_RR_f, d_omega,
            # Contexte
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
            *self._mech_args,
        )
        cuda.synchronize()

        RR_e  = d_RR_e.copy_to_host()[:, :n_e]
        RR_tb = d_RR_tb.copy_to_host()[:, :n_tb]
        RR_f  = d_RR_f.copy_to_host()[:, :n_f]
        omega = d_omega.copy_to_host()

        return RR_e, RR_tb, RR_f, omega