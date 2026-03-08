"""
cuda_integrator_pb.py — Per-block backward-Euler integrator (GMRES only).

API calquée sur le style PyTorch :
    integrator = CUDABackwardEulerIntegratorPB(mechanism, n_reactors)
    TC_next = integrator.step(TC_device, h)   # entrée/sortie sur device

Hypothèses :
  - Les entrées utilisateur (TC) sont déjà sur device (numba device array).
  - Pas de sauvegarde d'historique — utilisez step() en boucle si nécessaire.
  - 1 block CUDA = 1 réacteur (THREADS_PER_BLOCK threads coopèrent).
"""

import math
import numpy as np
import numba.cuda as cuda
from numba import int32
import warnings
from numba.core.errors import NumbaPerformanceWarning

from ..mechanism.mechanism_data import MechanismData
from ..kernels.chemistry_kernels_pb import (
    backward_euler_kernel_pb,
    MAX_R,
    THREADS_PER_BLOCK,
)


class CUDABackwardEulerIntegratorPB:
    """
    Per-BLOCK backward-Euler integrator pour des réacteurs adiabatiques en batch.
    Un block CUDA (64 threads) intègre un réacteur.

    Parameters
    ----------
    mechanism      : MechanismData
    n_reactors     : int
    newton_tol     : float   (défaut 1e-6)
    newton_maxiter : int     (défaut 20)
    gmres_restart  : int     (défaut 20, doit être <= MAX_R=20)
    gmres_tol      : float   (défaut 1e-6)
    gmres_maxiter  : int     (défaut 50)
    dtype          : numpy dtype (défaut float64)
    """

    def __init__(
        self,
        mechanism: MechanismData,
        n_reactors: int,
        newton_tol: float   = 1e-6,
        newton_maxiter: int = 20,
        gmres_restart: int  = 20,
        gmres_tol: float    = 1e-6,
        gmres_maxiter: int  = 50,
        dtype               = np.float64,
    ):
        assert gmres_restart <= 20, (
            f"gmres_restart doit être <= MAX_R={MAX_R} "
            f"(la mémoire partagée est statiquement dimensionnée à {MAX_R})"
        )

        self.mech           = mechanism
        self.B              = n_reactors
        self.N_s            = mechanism.n_species
        self.N_state        = mechanism.n_species + 1
        self.newton_tol     = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)
        self.gmres_restart  = int(gmres_restart)
        self.gmres_tol      = float(gmres_tol)
        self.gmres_maxiter  = int(gmres_maxiter)
        self.dtype          = dtype

        self._d_mech  = mechanism.to_cuda_arrays()
        self._alloc_buffers()
        self._mech_args = self._pack_mech_args()

    # ─────────────────────────────────────────────────────────────────────

    def _alloc_buffers(self):
        B    = self.B
        Nst  = self.N_state
        R    = self.gmres_restart
        dt   = self.dtype
        mech = self.mech

        e    = mech.elementary
        tb   = mech.threebody
        fall = mech.falloff

        n_e     = e.n_reactions    if e    else 0
        n_tb    = tb.n_reactions   if tb   else 0
        n_f     = fall.n_reactions if fall else 0
        n_tr_e  = len(e.react_idx)    if e    else 0
        n_tp_e  = len(e.prod_idx)     if e    else 0
        n_tr_tb = len(tb.react_idx)   if tb   else 0
        n_tp_tb = len(tb.prod_idx)    if tb   else 0
        n_tr_f  = len(fall.react_idx) if fall else 0
        n_tp_f  = len(fall.prod_idx)  if fall else 0

        def _buf(shape):  return cuda.device_array(shape, dtype=dt)
        def _d1():        return _buf((B, 1))

        # Buffer de sortie (ping-pong géré dans step())
        self._d_TC_out = _buf((B, Nst))

        # Drapeau de convergence
        self._d_conv = cuda.device_array((B,), dtype=np.int32)

        # Base de Krylov GMRES (global memory)
        self._d_gm_V = _buf((B, (R + 1) * Nst))

        # Buffers de contexte chimique (global memory, une ligne par réacteur)
        def _ce(n):  return _buf((B, n)) if n > 0 else _d1()

        self._ctx_kf_e   = _ce(n_e);    self._ctx_Kc_e   = _ce(n_e)
        self._ctx_cr_e   = _ce(n_e);    self._ctx_cp_e   = _ce(n_e)
        self._ctx_dG0_e  = _ce(n_e)
        self._ctx_sRre   = _ce(n_tr_e); self._ctx_sRpe   = _ce(n_tp_e)

        self._ctx_kf_tb  = _ce(n_tb);   self._ctx_Kc_tb  = _ce(n_tb)
        self._ctx_cr_tb  = _ce(n_tb);   self._ctx_cp_tb  = _ce(n_tb)
        self._ctx_dG0_tb = _ce(n_tb)
        self._ctx_sRrtb  = _ce(n_tr_tb);self._ctx_sRptb  = _ce(n_tp_tb)
        self._ctx_bRR_tb = _ce(n_tb);   self._ctx_M_tb   = _ce(n_tb)

        self._ctx_kf_f   = _ce(n_f);    self._ctx_Kc_f   = _ce(n_f)
        self._ctx_cr_f   = _ce(n_f);    self._ctx_cp_f   = _ce(n_f)
        self._ctx_dG0_f  = _ce(n_f)
        self._ctx_sRrf   = _ce(n_tr_f); self._ctx_sRpf   = _ce(n_tp_f)
        self._ctx_k0_f   = _ce(n_f);    self._ctx_kinf_f = _ce(n_f)
        self._ctx_M_f    = _ce(n_f);    self._ctx_Pr_f   = _ce(n_f)
        self._ctx_F_f    = _ce(n_f);    self._ctx_kb_f   = _ce(n_f)
        self._ctx_Fc_f   = _ce(n_f);    self._ctx_lFc_f  = _ce(n_f)
        self._ctx_Cp_f   = _ce(n_f);    self._ctx_Np_f   = _ce(n_f)
        self._ctx_f1_f   = _ce(n_f);    self._ctx_exp_f  = _ce(n_f)

    # ─────────────────────────────────────────────────────────────────────

    def step(self, d_TC: "cuda.devicearray.DeviceNDArray", h: float) -> "cuda.devicearray.DeviceNDArray":
        """
        Effectue un pas backward-Euler de taille h.

        Parameters
        ----------
        d_TC : numba device array, shape (B, N_state), déjà sur device
        h    : float, taille du pas

        Returns
        -------
        numba device array, shape (B, N_state), sur device
        """
        # Réinitialiser le drapeau de convergence directement via CUDA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            self._d_conv[:] = 0

        backward_euler_kernel_pb[self.B, THREADS_PER_BLOCK](
            d_TC, self._d_TC_out,
            self.dtype(h),
            self.dtype(self.newton_tol),
            np.int32(self.newton_maxiter),
            np.int32(self.gmres_restart),
            self.dtype(self.gmres_tol),
            np.int32(self.gmres_maxiter),
            self._d_conv,
            # Base de Krylov GMRES (global)
            self._d_gm_V,
            # Buffers de contexte
            self._ctx_kf_e,   self._ctx_Kc_e,  self._ctx_cr_e,
            self._ctx_cp_e,   self._ctx_dG0_e,
            self._ctx_sRre,   self._ctx_sRpe,
            self._ctx_kf_tb,  self._ctx_Kc_tb, self._ctx_cr_tb,
            self._ctx_cp_tb,  self._ctx_dG0_tb,
            self._ctx_sRrtb,  self._ctx_sRptb, self._ctx_bRR_tb, self._ctx_M_tb,
            self._ctx_kf_f,   self._ctx_Kc_f,  self._ctx_cr_f,
            self._ctx_cp_f,   self._ctx_dG0_f,
            self._ctx_sRrf,   self._ctx_sRpf,
            self._ctx_k0_f,   self._ctx_kinf_f, self._ctx_M_f,
            self._ctx_Pr_f,   self._ctx_F_f,   self._ctx_kb_f,
            self._ctx_Fc_f,   self._ctx_lFc_f, self._ctx_Cp_f,
            self._ctx_Np_f,   self._ctx_f1_f,  self._ctx_exp_f,
            # Mécanisme
            *self._mech_args,
        )
        cuda.synchronize()
        #TODO: return the convergence flag
        return self._d_TC_out

    # ─────────────────────────────────────────────────────────────────────

    def _pack_mech_args(self):
        d  = self._d_mech
        m  = self.mech
        e  = m.elementary
        tb = m.threebody
        fa = m.falloff
        Ns = self.N_s

        def _g(k):  return d.get(k, cuda.device_array((1,), self.dtype))
        def _gi(k): return d.get(k, cuda.device_array((1,), np.int32))

        return [
            np.int32(Ns),
            d['thermo_coef_low'], d['thermo_coef_high'], d['thermo_Tmid'],
            # Élémentaires
            np.int32(e.n_reactions if e else 0),
            _g('e_A'), _g('e_n'), _g('e_Ea'), _g('e_oneWay'), _g('e_delta_n'),
            _gi('e_react_idx'), _g('e_react_sto'), _gi('e_react_offset'),
            _gi('e_prod_idx'),  _g('e_prod_sto'),  _gi('e_prod_offset'),
            # Trois corps
            np.int32(tb.n_reactions if tb else 0),
            _g('tb_A'), _g('tb_n'), _g('tb_Ea'), _g('tb_oneWay'), _g('tb_delta_n'),
            _gi('tb_react_idx'), _g('tb_react_sto'), _gi('tb_react_offset'),
            _gi('tb_prod_idx'),  _g('tb_prod_sto'),  _gi('tb_prod_offset'),
            _g('tb_eff_matrix'),
            # Falloff
            np.int32(fa.n_reactions if fa else 0),
            _g('f_A_k0'),   _g('f_b_k0'),   _g('f_Ea_k0'),
            _g('f_A_kinf'), _g('f_b_kinf'), _g('f_Ea_kinf'),
            _g('f_oneWay'), _g('f_delta_n'),
            _g('f_troe_A'), _g('f_troe_T1'), _g('f_troe_T2'), _g('f_troe_T3'),
            _gi('f_is_troe'),
            _gi('f_react_idx'), _g('f_react_sto'), _gi('f_react_offset'),
            _gi('f_prod_idx'),  _g('f_prod_sto'),  _gi('f_prod_offset'),
            _g('f_eff_matrix'),
        ]