"""
cuda_integrator.py — Production backward-Euler integrator (GMRES only).

vs original v5:
  - No LU path, no timing, no reuse_jacobian, no dead branches.
  - h passed as float64 scalar — no per-step _d_h_arr fill kernel.
  - conv zeroed via lightweight device kernel (same as v5).
  - Context buffers (buf_ctx_*) kept in global memory — same as v5,
    avoids register spilling that would happen with local arrays.
  - Snapshots accumulated in pre-allocated device buffer, single
    copy_to_host at end of integrate() — same as v5.
  - _d_J, _d_LU, _d_piv, _d_F, _d_dx, _d_xk, _d_omega removed
    (those were only needed for the LU path).
"""

import math
import numpy as np
import numba.cuda as cuda
from numba import int32
import time

from ..mechanism.mechanism_data import MechanismData
from ..kernels.per_thread.chemistry_kernels import (
    backward_euler_kernel,
    _fill_int_zero,
    _store_snapshot,
    MAX_N_STATE,
    THREADS_PER_BLOCK,
)


class CUDABackwardEulerIntegrator:
    """
    Per-thread backward-Euler integrator for batched adiabatic reactors.

    Parameters
    ----------
    mechanism      : MechanismData
    n_reactors     : int
    newton_tol     : float   (default 1e-6)
    newton_maxiter : int     (default 20)
    gmres_restart  : int     (default 20)
    gmres_tol      : float   (default 1e-6)
    gmres_maxiter  : int     (default 50)
    dtype          : numpy dtype (default float64)
    """

    def __init__(
        self,
        mechanism: MechanismData,
        n_reactors: int,
        newton_tol: float   = 1e-12,
        newton_maxiter: int = 20,
        gmres_restart: int  = 20,
        gmres_tol: float    = 1e-12,
        gmres_maxiter: int  = 50,
        dtype               = np.float64,
    ):
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

        self._d_mech = mechanism.to_cuda_arrays()
        self._alloc_buffers()
        self._mech_args = self._pack_mech_args()

    # ─────────────────────────────────────────────────────────────────────

    def _alloc_buffers(self):
        B    = self.B
        Nst  = self.N_state
        Ns   = self.N_s
        R    = self.gmres_restart
        dt   = self.dtype
        mech = self.mech

        e    = mech.elementary
        tb   = mech.threebody
        fall = mech.falloff

        n_e       = e.n_reactions    if e    else 0
        n_tb      = tb.n_reactions   if tb   else 0
        n_f       = fall.n_reactions if fall else 0
        n_tr_e    = len(e.react_idx)    if e    else 0
        n_tp_e    = len(e.prod_idx)     if e    else 0
        n_tr_tb   = len(tb.react_idx)   if tb   else 0
        n_tp_tb   = len(tb.prod_idx)    if tb   else 0
        n_tr_f    = len(fall.react_idx) if fall else 0
        n_tp_f    = len(fall.prod_idx)  if fall else 0

        def _buf(shape):  return cuda.device_array(shape, dtype=dt)
        def _ibuf(shape): return cuda.device_array(shape, dtype=np.int32)
        def _d1():        return _buf((B, 1))

        # Ping-pong state
        self._d_TC_a = _buf((B, Nst))
        self._d_TC_b = _buf((B, Nst))

        # Convergence flag
        self._d_conv = _ibuf((B,))

        # GMRES scratch
        self._d_gm_V  = _buf((B, (R+1)*Nst))
        self._d_gm_H  = _buf((B, (R+1)*R))
        self._d_gm_cs = _buf((B, R))
        self._d_gm_sn = _buf((B, R))
        self._d_gm_g  = _buf((B, R+1))
        self._d_gm_w  = _buf((B, Nst))
        self._d_gm_y  = _buf((B, R))

        # Chemistry context buffers (global memory, per-reactor rows)
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

    def _blocks(self):
        return math.ceil(self.B / THREADS_PER_BLOCK)

    def _zero_conv(self):
        _fill_int_zero[self._blocks(), THREADS_PER_BLOCK](self._d_conv)

    # ─────────────────────────────────────────────────────────────────────

    def step(self, h: float):
        """Single backward-Euler step with step size h."""
        self._zero_conv()
        backward_euler_kernel[self._blocks(), THREADS_PER_BLOCK](
            self._d_TC_a, self._d_TC_b,
            self.dtype(h),
            self.dtype(self.newton_tol),
            np.int32(self.newton_maxiter),
            np.int32(self.gmres_restart),
            self.dtype(self.gmres_tol),
            np.int32(self.gmres_maxiter),
            self._d_conv,
            # GMRES scratch
            self._d_gm_V, self._d_gm_H,
            self._d_gm_cs, self._d_gm_sn, self._d_gm_g,
            self._d_gm_w, self._d_gm_y,
            # context buffers
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
            # mechanism
            *self._mech_args,
        )
        self._d_TC_a, self._d_TC_b = self._d_TC_b, self._d_TC_a

    # ─────────────────────────────────────────────────────────────────────

    def integrate(
        self,
        TC0: np.ndarray,
        t0: float,
        t_end: float,
        h: float,
        save_every: int = 1,
    ):
        """
        Integrate from t0 to t_end with fixed step h.

        Returns
        -------
        TC_hist   : np.ndarray (n_saved, B, n_state)
        t_hist    : np.ndarray (n_saved,)
        conv_hist : np.ndarray (n_saved, B)  int32
        """
        self._d_TC_a.copy_to_device(
            np.ascontiguousarray(TC0, dtype=self.dtype)
        )

        n_steps  = int(math.ceil((t_end - t0) / h))
        n_saves  = math.ceil(n_steps / save_every)
        t        = t0
        snap_idx = 0
        blk      = self._blocks()

        d_TC_hist   = cuda.device_array((n_saves, self.B, self.N_state), self.dtype)
        d_conv_hist = cuda.device_array((n_saves, self.B), np.int32)
        t_list      = []

        t_wall = 0.0
        for step_i in range(n_steps):
            h_eff = min(h, t_end - t)
            if h_eff <= 0.0:
                break

            t_wall -= time.perf_counter()
            self.step(h_eff)
            print(t)
            t_wall += time.perf_counter()

            t += h_eff

            if step_i % save_every == 0 and snap_idx < n_saves:
                _store_snapshot[blk, THREADS_PER_BLOCK](
                    self._d_TC_a, self._d_conv,
                    d_TC_hist, d_conv_hist,
                    np.int32(snap_idx),
                )
                t_list.append(t)
                snap_idx += 1

        cuda.synchronize()
        TC_hist   = d_TC_hist.copy_to_host()[:snap_idx]
        conv_hist = d_conv_hist.copy_to_host()[:snap_idx]
        t_hist    = np.array(t_list)

        print(f"integrate: {n_steps} steps in {t_wall:.4f}s wall "
              f"({1e6*t_wall/n_steps:.2f} µs/step, {self.B} reactors)")

        return TC_hist, t_hist, conv_hist

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
            # Elementary
            np.int32(e.n_reactions if e else 0),
            _g('e_A'), _g('e_n'), _g('e_Ea'), _g('e_oneWay'), _g('e_delta_n'),
            _gi('e_react_idx'), _g('e_react_sto'), _gi('e_react_offset'),
            _gi('e_prod_idx'),  _g('e_prod_sto'),  _gi('e_prod_offset'),
            # Three-body
            np.int32(tb.n_reactions if tb else 0),
            _g('tb_A'), _g('tb_n'), _g('tb_Ea'), _g('tb_oneWay'), _g('tb_delta_n'),
            _gi('tb_react_idx'), _g('tb_react_sto'), _gi('tb_react_offset'),
            _gi('tb_prod_idx'),  _g('tb_prod_sto'),  _gi('tb_prod_offset'),
            _g('tb_eff_matrix'),
            # Falloff
            np.int32(fa.n_reactions if fa else 0),
            _g('f_A_k0'),  _g('f_b_k0'),  _g('f_Ea_k0'),
            _g('f_A_kinf'),_g('f_b_kinf'),_g('f_Ea_kinf'),
            _g('f_oneWay'),_g('f_delta_n'),
            _g('f_troe_A'),_g('f_troe_T1'),_g('f_troe_T2'),_g('f_troe_T3'),
            _gi('f_is_troe'),
            _gi('f_react_idx'), _g('f_react_sto'), _gi('f_react_offset'),
            _gi('f_prod_idx'),  _g('f_prod_sto'),  _gi('f_prod_offset'),
            _g('f_eff_matrix'),
        ]