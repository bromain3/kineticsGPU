"""
CUDABackwardEulerIntegrator: high-level Python interface for the per-thread
CUDA backward Euler integrator.

Analogous to the existing BackwardEulerIntegrator (torch version) but:
  - Uses Numba CUDA kernels directly.
  - Pre-allocates all global memory buffers once at construction time.
  - Provides step() and integrate() methods.
  - Mechanism data is generic: any MechanismData instance can be used.

Usage:
    mech  = build_mechanism_data_from_torch(r_e, r_tb, r_f, vec_th)
    integ = CUDABackwardEulerIntegrator(mech, n_reactors=1000)
    TC_next = integ.step(TC_current, t, h)
    TC_final, t_hist = integ.integrate(TC0, t0, t_end, h)
"""

import math
import numpy as np
import numba.cuda as cuda

from ..mechanism.mechanism_data import MechanismData
from ..kernels.integrator_kernel import backward_euler_per_thread_kernel

THREADS_PER_BLOCK = 64   # optimal per literature (8 blocks/SM)


class CUDABackwardEulerIntegrator:
    """
    Per-thread backward Euler integrator for batched chemical reactors.

    Parameters
    ----------
    mechanism : MechanismData
        Pre-built mechanism data (from build_mechanism_data_from_torch).
    n_reactors : int
        Number of reactors (batch size B).
    newton_tol : float
        Newton convergence tolerance (default 1e-8, per literature atol=1e-10).
    newton_maxiter : int
        Max Newton iterations per step (default 20).
    reuse_jacobian : bool
        If True, reuse Jacobian across Newton iterations (reassemble only
        on step failure or first step). Recommended: True.
    dtype : np.dtype
        Floating point dtype (default float64).
    """

    def __init__(
        self,
        mechanism: MechanismData,
        n_reactors: int,
        newton_tol: float = 1e-6,
        newton_maxiter: int = 20,
        reuse_jacobian: bool = True,
        dtype=np.float64,
    ):
        self.mech        = mechanism
        self.B           = n_reactors
        self.N_s         = mechanism.n_species
        self.N_state     = mechanism.n_species + 1    # T + species
        self.newton_tol  = newton_tol
        self.newton_maxiter = newton_maxiter
        self.reuse_jac   = int(reuse_jacobian)
        self.dtype       = dtype

        # Transfer mechanism arrays to device once
        self._d_mech = mechanism.to_cuda_arrays()

        # Pre-allocate working buffers (global memory, never reallocated)
        self._alloc_buffers()

    # ─────────────────────────────────────────────────────────────────────
    #  Buffer allocation
    # ─────────────────────────────────────────────────────────────────────

    def _alloc_buffers(self):
        B  = self.B
        Ns = self.N_s
        Nst = self.N_state
        mech = self.mech
        dt = self.dtype

        e   = mech.elementary
        tb  = mech.threebody
        fall= mech.falloff

        n_e    = e.n_reactions    if e    else 0
        n_tb   = tb.n_reactions   if tb   else 0
        n_fall = fall.n_reactions if fall else 0

        n_tr_e    = len(e.react_idx)    if e    else 0
        n_tp_e    = len(e.prod_idx)     if e    else 0
        n_tr_tb   = len(tb.react_idx)   if tb   else 0
        n_tp_tb   = len(tb.prod_idx)    if tb   else 0
        n_tr_fall = len(fall.react_idx) if fall else 0
        n_tp_fall = len(fall.prod_idx)  if fall else 0

        def buf(shape):
            return cuda.device_array(shape, dtype=dt)

        def ibuf(shape):
            return cuda.device_array(shape, dtype=np.int32)

        # Newton / LU buffers
        self._d_J        = buf((B, Nst * Nst))
        self._d_LU       = buf((B, Nst * Nst))
        self._d_piv      = ibuf((B, Nst))
        self._d_F        = buf((B, Nst))
        self._d_dx       = buf((B, Nst))
        self._d_xk       = buf((B, Nst))
        self._d_omega    = buf((B, Ns))
        self._d_conv     = ibuf((B,))

        # Context – elementary
        self._d_ctx_kf_e    = buf((B, n_e))   if n_e else _dummy(dt)
        self._d_ctx_Kc_e    = buf((B, n_e))   if n_e else _dummy(dt)
        self._d_ctx_cr_e    = buf((B, n_e))   if n_e else _dummy(dt)
        self._d_ctx_cp_e    = buf((B, n_e))   if n_e else _dummy(dt)
        self._d_ctx_RR_e    = buf((B, n_e))   if n_e else _dummy(dt)
        self._d_ctx_dG0_e   = buf((B, n_e))   if n_e else _dummy(dt)
        self._d_ctx_sr_e    = buf((B, n_tr_e)) if n_tr_e else _dummy(dt)
        self._d_ctx_sp_e    = buf((B, n_tp_e)) if n_tp_e else _dummy(dt)

        # Context – three-body
        self._d_ctx_kf_tb   = buf((B, n_tb))  if n_tb else _dummy(dt)
        self._d_ctx_Kc_tb   = buf((B, n_tb))  if n_tb else _dummy(dt)
        self._d_ctx_cr_tb   = buf((B, n_tb))  if n_tb else _dummy(dt)
        self._d_ctx_cp_tb   = buf((B, n_tb))  if n_tb else _dummy(dt)
        self._d_ctx_RR_tb   = buf((B, n_tb))  if n_tb else _dummy(dt)
        self._d_ctx_bRR_tb  = buf((B, n_tb))  if n_tb else _dummy(dt)
        self._d_ctx_M_tb    = buf((B, n_tb))  if n_tb else _dummy(dt)
        self._d_ctx_dG0_tb  = buf((B, n_tb))  if n_tb else _dummy(dt)
        self._d_ctx_sr_tb   = buf((B, n_tr_tb)) if n_tr_tb else _dummy(dt)
        self._d_ctx_sp_tb   = buf((B, n_tp_tb)) if n_tp_tb else _dummy(dt)

        # Context – falloff
        self._d_ctx_kf_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_Kc_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_cr_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_cp_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_RR_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_k0_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_kinf_fall= buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_M_fall   = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_Pr_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_F_fall   = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_kb_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_Fc_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_lFc_fall = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_Cp_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_Np_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_f1_fall  = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_exp_fall = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_dG0_fall = buf((B, n_fall)) if n_fall else _dummy(dt)
        self._d_ctx_sr_fall  = buf((B, n_tr_fall)) if n_tr_fall else _dummy(dt)
        self._d_ctx_sp_fall  = buf((B, n_tp_fall)) if n_tp_fall else _dummy(dt)

    # ─────────────────────────────────────────────────────────────────────
    #  step(): one backward Euler step for all reactors
    # ─────────────────────────────────────────────────────────────────────

    def step(self, TC: np.ndarray, t: float, h: float) -> np.ndarray:
        """
        Advance all reactors by one time step h.

        Parameters
        ----------
        TC : np.ndarray, shape (B, n_species+1)
            Current state [T, C1, ..., CN] for each reactor.
        t  : float  current time
        h  : float  time step

        Returns
        -------
        TC_next : np.ndarray, shape (B, n_species+1)
        """

        B    = self.B
        Nst  = self.N_state

        # Transfer input to device
        d_TC_in  = cuda.to_device(np.ascontiguousarray(TC, dtype=self.dtype))
        d_TC_out = cuda.device_array((B, Nst), dtype=self.dtype)

        h_arr = np.full(B, h, dtype=self.dtype)
        t_arr = np.full(B, t, dtype=self.dtype)
        d_h   = cuda.to_device(h_arr)
        d_t   = cuda.to_device(t_arr)

        # Reset convergence flags
        cuda.to_device(np.zeros(B, dtype=np.int32), to=self._d_conv)

        # Launch: 64 threads/block
        blocks = math.ceil(B / THREADS_PER_BLOCK)
        mech_args = self._pack_mech_args()

        backward_euler_per_thread_kernel[blocks, THREADS_PER_BLOCK](
            d_TC_in, d_TC_out, d_h, d_t,
            self._d_J, self._d_LU, self._d_piv,
            self._d_F, self._d_dx, self._d_xk, self._d_omega,
            self._d_ctx_kf_e,   self._d_ctx_Kc_e,
            self._d_ctx_cr_e,   self._d_ctx_cp_e,  self._d_ctx_RR_e,
            self._d_ctx_dG0_e,  self._d_ctx_sr_e,  self._d_ctx_sp_e,
            self._d_ctx_kf_tb,  self._d_ctx_Kc_tb,
            self._d_ctx_cr_tb,  self._d_ctx_cp_tb,
            self._d_ctx_RR_tb,  self._d_ctx_bRR_tb, self._d_ctx_M_tb,
            self._d_ctx_dG0_tb, self._d_ctx_sr_tb,  self._d_ctx_sp_tb,
            self._d_ctx_kf_fall,  self._d_ctx_Kc_fall,
            self._d_ctx_cr_fall,  self._d_ctx_cp_fall,  self._d_ctx_RR_fall,
            self._d_ctx_k0_fall,  self._d_ctx_kinf_fall, self._d_ctx_M_fall,
            self._d_ctx_Pr_fall,  self._d_ctx_F_fall,   self._d_ctx_kb_fall,
            self._d_ctx_Fc_fall,  self._d_ctx_lFc_fall,
            self._d_ctx_Cp_fall,  self._d_ctx_Np_fall,
            self._d_ctx_f1_fall,  self._d_ctx_exp_fall,
            self._d_ctx_dG0_fall, self._d_ctx_sr_fall,  self._d_ctx_sp_fall,
            self._d_conv,
            self.newton_tol,
            self.newton_maxiter,
            self.reuse_jac,
            *mech_args,
        )
        print(t)
        cuda.synchronize()

        return d_TC_out.copy_to_host()

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
        TC_hist : np.ndarray (n_saved, B, n_state)
        t_hist  : np.ndarray (n_saved,)
        conv_hist : np.ndarray (n_saved, B)  convergence flag per step
        """
        TC = TC0.copy()
        t  = t0
        n_steps = int(math.ceil((t_end - t0) / h))

        TC_hist   = []
        t_hist    = []
        conv_hist = []

        for step_i in range(n_steps):
            h_eff = min(h, t_end - t)
            if h_eff <= 0.0:
                break
            TC = self.step(TC, t, h_eff)
            t += h_eff
            if step_i % save_every == 0:
                TC_hist.append(TC.copy())
                t_hist.append(t)
                conv_hist.append(self._d_conv.copy_to_host())

        return (
            np.stack(TC_hist,   axis=0),
            np.array(t_hist),
            np.stack(conv_hist, axis=0),
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Helper: pack mechanism device arrays in the order expected by kernel
    # ─────────────────────────────────────────────────────────────────────

    def _pack_mech_args(self):
        d  = self._d_mech
        m  = self.mech
        e  = m.elementary
        tb = m.threebody
        fa = m.falloff

        def _get(key, fallback_shape):
            if key in d:
                return d[key]
            return cuda.device_array(fallback_shape, dtype=self.dtype)

        def _iget(key, fallback_shape):
            if key in d:
                return d[key]
            return cuda.device_array(fallback_shape, dtype=np.int32)

        Ns = self.N_s
        args = [
            np.int32(Ns),
            d['thermo_coef_low'],
            d['thermo_coef_high'],
            d['thermo_Tmid'],
            # elementary
            np.int32(e.n_reactions if e else 0),
            _get('elem_A',    (1,)), _get('elem_n',    (1,)), _get('elem_Ea',   (1,)),
            _get('elem_oneWay',(1,)), _get('elem_delta_n',(1,)),
            _iget('elem_react_idx',(1,)), _get('elem_react_sto',(1,)), _iget('elem_react_offset',(1,)),
            _iget('elem_prod_idx', (1,)), _get('elem_prod_sto', (1,)), _iget('elem_prod_offset', (1,)),
            _iget('elem_idx_r_scatter',(1,)), _iget('elem_idx_p_scatter',(1,)),
            # three-body
            np.int32(tb.n_reactions if tb else 0),
            _get('tb_A',(1,)),    _get('tb_n',(1,)),    _get('tb_Ea',(1,)),
            _get('tb_oneWay',(1,)), _get('tb_delta_n',(1,)),
            _iget('tb_react_idx',(1,)), _get('tb_react_sto',(1,)), _iget('tb_react_offset',(1,)),
            _iget('tb_prod_idx', (1,)), _get('tb_prod_sto', (1,)), _iget('tb_prod_offset', (1,)),
            _iget('tb_idx_r_scatter',(1,)), _iget('tb_idx_p_scatter',(1,)),
            _get('tb_eff_matrix',(1, Ns)),
            # falloff
            np.int32(fa.n_reactions if fa else 0),
            _get('fall_A_k0',(1,)),   _get('fall_b_k0',(1,)),   _get('fall_Ea_k0',(1,)),
            _get('fall_A_kinf',(1,)),  _get('fall_b_kinf',(1,)),  _get('fall_Ea_kinf',(1,)),
            _get('fall_oneWay',(1,)),  _get('fall_delta_n',(1,)),
            _get('fall_troe_A',(1,)),  _get('fall_troe_T1',(1,)),
            _get('fall_troe_T2',(1,)), _get('fall_troe_T3',(1,)),
            _iget('fall_is_troe',(1,)),
            _iget('fall_react_idx',(1,)), _get('fall_react_sto',(1,)), _iget('fall_react_offset',(1,)),
            _iget('fall_prod_idx', (1,)), _get('fall_prod_sto', (1,)), _iget('fall_prod_offset', (1,)),
            _iget('fall_idx_r_scatter',(1,)), _iget('fall_idx_p_scatter',(1,)),
            _get('fall_eff_matrix',(1, Ns)),
        ]
        return args


# ─── Utilities ────────────────────────────────────────────────────────────────

def _dummy(dtype):
    """Returns a 1-element device array used as placeholder for unused buffers."""
    return cuda.device_array((1,), dtype=dtype)