"""
CUDABackwardEulerIntegrator — v4 (LU + GMRES + timing)

New vs v3
─────────
• BUG FIX: GMRES dummy buffers are now shape (B, 1) — Numba requires every
  2-D slice buf[b] to return a 1-D array regardless of whether GMRES is used.
  The old (1,) dummies caused a TypingError when the LU path was compiled
  because Numba inferred a scalar type.

• Timing system:
    - Pass  do_timing=True  to step() or integrate() to enable.
    - Cycle counts are accumulated in buf_timers (B, 7) int64 on device.
    - print_timers() reports mean/max/std over reactors, converted to µs
      using the SM clock frequency queried at construction time.
    - reset_timers() zeroes the buffer.

Timer phases
────────────
  0  PREDICTOR    initial RHS evaluation + explicit Euler step
  1  RHS          _eval_rhs_and_context  inside Newton loop  (per Newton iter)
  2  RESIDUAL     residual norm          inside Newton loop
  3  JAC_ASSEM    Jacobian assembly      (once per solve when reuse_jacobian)
  4  LU_FACTOR    LU factorisation       (LU path only, once per Jacobian)
  5  LINEAR_SOLVE back-substitution (LU) or full GMRES call (per Newton iter)
  6  UPDATE       xk += dx + physical constraints

Usage
─────
    integ = CUDABackwardEulerIntegrator(mech, 1000)
    TC_hist, t_hist, conv = integ.integrate(TC0, 0, 3e-4, 1e-7, do_timing=True)
    integ.print_timers()          # pretty summary
    raw = integ.get_timers()      # numpy array (B, 7)
"""

import math
import numpy as np
import numba.cuda as cuda
from numba.cuda import get_current_device

from ..mechanism.mechanism_data import MechanismData
from ..kernels.integrator_kernel import (
    backward_euler_per_thread_kernel,
    N_TIMER_PHASES,
    _T_PREDICTOR, _T_RHS, _T_RESIDUAL, _T_JAC_ASSEM,
    _T_LU_FACTOR, _T_LINEAR_SOLVE, _T_UPDATE,
)

THREADS_PER_BLOCK = 64

_PHASE_NAMES = [
    "PREDICTOR ",
    "RHS       ",
    "RESIDUAL  ",
    "JAC_ASSEM ",
    "LU_FACTOR ",
    "LIN_SOLVE ",
    "UPDATE    ",
]


class CUDABackwardEulerIntegrator:
    """
    Per-thread backward Euler integrator for batched chemical reactors.

    Parameters
    ----------
    mechanism : MechanismData
    n_reactors : int
    newton_tol : float
    newton_maxiter : int
    reuse_jacobian : bool
    linear_solver : str
        'lu'    — dense LU (exact, O(N³) factor cost)
        'gmres' — restarted GMRES with Givens rotations
    gmres_restart : int
        Krylov subspace size per restart cycle.
    gmres_tol : float
        GMRES inner tolerance (can be set looser than newton_tol).
    gmres_maxiter : int
        Maximum GMRES restart cycles per Newton step.
    dtype : np.dtype
    """

    _SOLVER_MAP = {'lu': 0, 'gmres': 1}

    def __init__(
        self,
        mechanism: MechanismData,
        n_reactors: int,
        newton_tol: float = 1e-6,
        newton_maxiter: int = 20,
        reuse_jacobian: bool = True,
        linear_solver: str = 'gmres',
        gmres_restart: int = 20,
        gmres_tol: float = 1e-6,
        gmres_maxiter: int = 50,
        dtype=np.float64,
    ):
        if linear_solver not in self._SOLVER_MAP:
            raise ValueError(
                f"linear_solver must be 'lu' or 'gmres', got '{linear_solver}'"
            )

        self.mech            = mechanism
        self.B               = n_reactors
        self.N_s             = mechanism.n_species
        self.N_state         = mechanism.n_species + 1
        self.newton_tol      = newton_tol
        self.newton_maxiter  = newton_maxiter
        self.reuse_jac       = int(reuse_jacobian)
        self.linear_solver   = self._SOLVER_MAP[linear_solver]
        self.gmres_restart   = int(gmres_restart)
        self.gmres_tol       = float(gmres_tol)
        self.gmres_maxiter   = int(gmres_maxiter)
        self.dtype           = dtype

        # SM clock frequency in kHz → convert cycles to µs: cycles / (kHz * 1e3)
        self._clock_khz = get_current_device().CLOCK_RATE   # int, kHz
        self._d_mech = mechanism.to_cuda_arrays()
        self._alloc_buffers()
        self._mech_args = self._pack_mech_args()

    # ─────────────────────────────────────────────────────────────────────
    #  Buffer allocation
    # ─────────────────────────────────────────────────────────────────────

    def _alloc_buffers(self):
        B   = self.B
        Ns  = self.N_s
        Nst = self.N_state
        mech = self.mech
        dt   = self.dtype
        R    = self.gmres_restart

        e    = mech.elementary
        tb   = mech.threebody
        fall = mech.falloff

        n_e       = e.n_reactions    if e    else 0
        n_tb      = tb.n_reactions   if tb   else 0
        n_fall    = fall.n_reactions if fall else 0
        n_tr_e    = len(e.react_idx)    if e    else 0
        n_tp_e    = len(e.prod_idx)     if e    else 0
        n_tr_tb   = len(tb.react_idx)   if tb   else 0
        n_tp_tb   = len(tb.prod_idx)    if tb   else 0
        n_tr_fall = len(fall.react_idx) if fall else 0
        n_tp_fall = len(fall.prod_idx)  if fall else 0

        def buf(shape):  return cuda.device_array(shape, dtype=dt)
        def ibuf(shape): return cuda.device_array(shape, dtype=np.int32)
        # Dummy 2-D buffer: shape (B, 1) so that buf[b] is always a typed
        # 1-D array of length 1 — Numba infers the correct array type.
        def dummy2d():   return cuda.device_array((B, 1), dtype=dt)

        # ── Ping-pong state + scalars ─────────────────────────────────────
        self._d_TC_a  = buf((B, Nst))
        self._d_TC_b  = buf((B, Nst))
        self._d_h_arr = buf((B,))
        self._d_t_arr = buf((B,))

        # ── Newton / LU ───────────────────────────────────────────────────
        self._d_J     = buf((B, Nst * Nst))
        self._d_LU    = buf((B, Nst * Nst))
        self._d_piv   = ibuf((B, Nst))
        self._d_F     = buf((B, Nst))
        self._d_dx    = buf((B, Nst))
        self._d_xk    = buf((B, Nst))
        self._d_omega = buf((B, Ns))
        self._d_conv  = ibuf((B,))

        # ── GMRES ─────────────────────────────────────────────────────────
        # Full allocation when linear_solver='gmres'; 2-D dummies (B,1)
        # otherwise so that buf_gmres_X[b] is always a typed 1-D array.
        if self.linear_solver == 1:
            self._d_gmres_V  = buf((B, (R + 1) * Nst))
            self._d_gmres_H  = buf((B, (R + 1) * R))
            self._d_gmres_cs = buf((B, R))
            self._d_gmres_sn = buf((B, R))
            self._d_gmres_g  = buf((B, R + 1))
            self._d_gmres_w  = buf((B, Nst))
            self._d_gmres_y  = buf((B, R))
            self._d_gmres_vC  = buf((B, Nst))   # ← NEW: v_C scratch
            self._d_gmres_Jvo = buf((B, Nst))   # ← NEW: Jv_omega scratch
        else:
            self._d_gmres_V  = dummy2d()
            self._d_gmres_H  = dummy2d()
            self._d_gmres_cs = dummy2d()
            self._d_gmres_sn = dummy2d()
            self._d_gmres_g  = dummy2d()
            self._d_gmres_w  = dummy2d()
            self._d_gmres_y  = dummy2d()
            self._d_gmres_vC  = dummy2d()        # ← NEW dummy
            self._d_gmres_Jvo = dummy2d()        # ← NEW dummy

        # ── Timing buffer — always allocated, zeros until do_timing=True ──
        self._d_timers = cuda.device_array((B, N_TIMER_PHASES), dtype=np.int64)
        self._d_timers.copy_to_device(
            np.zeros((B, N_TIMER_PHASES), dtype=np.int64)
        )

        self._d_iters = cuda.device_array((B, 2), dtype=np.int64)
        self._d_iters.copy_to_device(
            np.zeros((B, 2), dtype=np.int64)
        )

        # ── Chemistry context ─────────────────────────────────────────────
        self._d_ctx_kf_e    = buf((B, n_e))     if n_e     else dummy2d()
        self._d_ctx_Kc_e    = buf((B, n_e))     if n_e     else dummy2d()
        self._d_ctx_cr_e    = buf((B, n_e))     if n_e     else dummy2d()
        self._d_ctx_cp_e    = buf((B, n_e))     if n_e     else dummy2d()
        self._d_ctx_RR_e    = buf((B, n_e))     if n_e     else dummy2d()
        self._d_ctx_dG0_e   = buf((B, n_e))     if n_e     else dummy2d()
        self._d_ctx_sr_e    = buf((B, n_tr_e))  if n_tr_e  else dummy2d()
        self._d_ctx_sp_e    = buf((B, n_tp_e))  if n_tp_e  else dummy2d()

        self._d_ctx_kf_tb   = buf((B, n_tb))    if n_tb    else dummy2d()
        self._d_ctx_Kc_tb   = buf((B, n_tb))    if n_tb    else dummy2d()
        self._d_ctx_cr_tb   = buf((B, n_tb))    if n_tb    else dummy2d()
        self._d_ctx_cp_tb   = buf((B, n_tb))    if n_tb    else dummy2d()
        self._d_ctx_RR_tb   = buf((B, n_tb))    if n_tb    else dummy2d()
        self._d_ctx_bRR_tb  = buf((B, n_tb))    if n_tb    else dummy2d()
        self._d_ctx_M_tb    = buf((B, n_tb))    if n_tb    else dummy2d()
        self._d_ctx_dG0_tb  = buf((B, n_tb))    if n_tb    else dummy2d()
        self._d_ctx_sr_tb   = buf((B, n_tr_tb)) if n_tr_tb else dummy2d()
        self._d_ctx_sp_tb   = buf((B, n_tp_tb)) if n_tp_tb else dummy2d()

        self._d_ctx_kf_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_Kc_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_cr_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_cp_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_RR_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_k0_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_kinf_fall = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_M_fall    = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_Pr_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_F_fall    = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_kb_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_Fc_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_lFc_fall  = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_Cp_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_Np_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_f1_fall   = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_exp_fall  = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_dG0_fall  = buf((B, n_fall))    if n_fall    else dummy2d()
        self._d_ctx_sr_fall   = buf((B, n_tr_fall)) if n_tr_fall else dummy2d()
        self._d_ctx_sp_fall   = buf((B, n_tp_fall)) if n_tp_fall else dummy2d()

    # ─────────────────────────────────────────────────────────────────────
    #  step()
    # ─────────────────────────────────────────────────────────────────────

    def step(self, t: float, h: float, do_timing: bool = False, do_count: bool = False):
        B = self.B
        self._d_h_arr.copy_to_device(np.full(B, h, dtype=self.dtype))
        self._d_t_arr.copy_to_device(np.full(B, t, dtype=self.dtype))
        self._d_conv.copy_to_device(np.zeros(B, dtype=np.int32))

        blocks = math.ceil(B / THREADS_PER_BLOCK)

        backward_euler_per_thread_kernel[blocks, THREADS_PER_BLOCK](
            self._d_TC_a, self._d_TC_b,
            self._d_h_arr, self._d_t_arr,
            # Newton / LU
            self._d_J, self._d_LU, self._d_piv,
            self._d_F, self._d_dx, self._d_xk, self._d_omega,
            # GMRES (dummies when linear_solver='lu', all shape (B, stride))
            self._d_gmres_V,
            self._d_gmres_H,
            self._d_gmres_cs,
            self._d_gmres_sn,
            self._d_gmres_g,
            self._d_gmres_w,
            self._d_gmres_y,
            self._d_gmres_vC,  # ← NEW
            self._d_gmres_Jvo,  # ← NEW
            # Context – elementary
            self._d_ctx_kf_e,   self._d_ctx_Kc_e,
            self._d_ctx_cr_e,   self._d_ctx_cp_e,  self._d_ctx_RR_e,
            self._d_ctx_dG0_e,  self._d_ctx_sr_e,  self._d_ctx_sp_e,
            # Context – three-body
            self._d_ctx_kf_tb,  self._d_ctx_Kc_tb,
            self._d_ctx_cr_tb,  self._d_ctx_cp_tb,
            self._d_ctx_RR_tb,  self._d_ctx_bRR_tb, self._d_ctx_M_tb,
            self._d_ctx_dG0_tb, self._d_ctx_sr_tb,  self._d_ctx_sp_tb,
            # Context – falloff
            self._d_ctx_kf_fall,  self._d_ctx_Kc_fall,
            self._d_ctx_cr_fall,  self._d_ctx_cp_fall,  self._d_ctx_RR_fall,
            self._d_ctx_k0_fall,  self._d_ctx_kinf_fall, self._d_ctx_M_fall,
            self._d_ctx_Pr_fall,  self._d_ctx_F_fall,   self._d_ctx_kb_fall,
            self._d_ctx_Fc_fall,  self._d_ctx_lFc_fall,
            self._d_ctx_Cp_fall,  self._d_ctx_Np_fall,
            self._d_ctx_f1_fall,  self._d_ctx_exp_fall,
            self._d_ctx_dG0_fall, self._d_ctx_sr_fall,  self._d_ctx_sp_fall,
            self._d_conv,
            # Solver parameters
            self.newton_tol,
            self.newton_maxiter,
            self.reuse_jac,
            np.int32(self.linear_solver),
            np.int32(self.gmres_restart),
            self.gmres_tol,
            np.int32(self.gmres_maxiter),
            # Timing
            self._d_timers,
            np.int32(1 if do_timing else 0),
            # Counting
            self._d_iters,
            np.int32(1 if do_count else 0),
            # Mechanism
            *self._mech_args,
        )
        print(t)
        self._d_TC_a, self._d_TC_b = self._d_TC_b, self._d_TC_a

    # ─────────────────────────────────────────────────────────────────────
    #  integrate()
    # ─────────────────────────────────────────────────────────────────────

    def integrate(
        self,
        TC0: np.ndarray,
        t0: float,
        t_end: float,
        h: float,
        save_every: int = 1,
        do_timing: bool = False,
        do_count: bool = False,
    ):
        """
        Integrate from t0 to t_end with fixed step h.

        Returns
        -------
        TC_hist   : np.ndarray (n_saved, B, n_state)
        t_hist    : np.ndarray (n_saved,)
        conv_hist : np.ndarray (n_saved, B)
        """
        self._d_TC_a.copy_to_device(
            np.ascontiguousarray(TC0, dtype=self.dtype)
        )

        t       = t0
        n_steps = int(math.ceil((t_end - t0) / h))

        TC_hist   = []
        t_hist    = []
        conv_hist = []

        for step_i in range(n_steps):
            h_eff = min(h, t_end - t)
            if h_eff <= 0.0:
                break

            self.step(t, h_eff, do_timing=do_timing, do_count=do_count)
            t += h_eff

            if step_i % save_every == 0:
                TC_hist.append(self._d_TC_a.copy_to_host())
                t_hist.append(t)
                conv_hist.append(self._d_conv.copy_to_host())

        cuda.synchronize()

        return (
            np.stack(TC_hist,   axis=0),
            np.array(t_hist),
            np.stack(conv_hist, axis=0),
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Timing helpers
    # ─────────────────────────────────────────────────────────────────────

    def reset_timers(self):
        """Zero all accumulated cycle counts on device."""
        self._d_timers.copy_to_device(
            np.zeros((self.B, N_TIMER_PHASES), dtype=np.int64)
        )

    def get_timers(self) -> np.ndarray:
        """Return raw timer array (B, 7) of accumulated SM cycle counts."""
        return self._d_timers.copy_to_host()

    def get_timers_us(self) -> np.ndarray:
        """Return timer array (B, 7) converted to microseconds."""
        raw = self.get_timers().astype(np.float64)
        return raw / (self._clock_khz * 1e3)

    def print_timers(self, n_steps: int = 1):
        """
        Print a summary table of per-phase timings.

        Parameters
        ----------
        n_steps : int
            Number of integration steps over which timers were accumulated.
            Used to compute per-step averages.
        """
        us = self.get_timers_us()  # (B, 7)

        # Mean across reactors, then divide by number of steps
        mean_us  = us.mean(axis=0)
        max_us   = us.max(axis=0)
        std_us   = us.std(axis=0)
        total_us = mean_us.sum()

        solver_name = 'LU' if self.linear_solver == 0 else 'GMRES'
        print(f"\n{'─'*62}")
        print(f"  Solver timing summary  [{solver_name}, B={self.B}, steps={n_steps}]")
        print(f"{'─'*62}")
        print(f"  {'Phase':<14}  {'Mean(µs)':>10}  {'Max(µs)':>10}  "
              f"{'Std(µs)':>10}  {'%total':>7}")
        print(f"{'─'*62}")
        for ph in range(N_TIMER_PHASES):
            pct = 100.0 * mean_us[ph] / total_us if total_us > 0 else 0.0
            print(f"  {_PHASE_NAMES[ph]:<14}  {mean_us[ph]:>10.2f}  "
                  f"{max_us[ph]:>10.2f}  {std_us[ph]:>10.2f}  {pct:>6.1f}%")
        print(f"{'─'*62}")
        print(f"  {'TOTAL':<14}  {total_us:>10.2f}")
        print(f"{'─'*62}\n")

        # Per-step breakdown
        if n_steps > 1:
            print(f"  Per-step averages (÷{n_steps}):")
            for ph in range(N_TIMER_PHASES):
                print(f"    {_PHASE_NAMES[ph]}: {mean_us[ph]/n_steps:.3f} µs")
            print()

    # ─────────────────────────────────────────────────────────────────────
    #  _pack_mech_args  (unchanged)
    # ─────────────────────────────────────────────────────────────────────

    def _pack_mech_args(self):
        d  = self._d_mech
        m  = self.mech
        e  = m.elementary
        tb = m.threebody
        fa = m.falloff

        def _get(key, fs):
            return d[key] if key in d else cuda.device_array(fs, dtype=self.dtype)
        def _iget(key, fs):
            return d[key] if key in d else cuda.device_array(fs, dtype=np.int32)

        Ns = self.N_s
        return [
            np.int32(Ns),
            d['thermo_coef_low'], d['thermo_coef_high'], d['thermo_Tmid'],
            np.int32(e.n_reactions if e else 0),
            _get('elem_A',(1,)),   _get('elem_n',(1,)),   _get('elem_Ea',(1,)),
            _get('elem_oneWay',(1,)), _get('elem_delta_n',(1,)),
            _iget('elem_react_idx',(1,)), _get('elem_react_sto',(1,)),
            _iget('elem_react_offset',(1,)),
            _iget('elem_prod_idx',(1,)),  _get('elem_prod_sto',(1,)),
            _iget('elem_prod_offset',(1,)),
            _iget('elem_idx_r_scatter',(1,)), _iget('elem_idx_p_scatter',(1,)),
            np.int32(tb.n_reactions if tb else 0),
            _get('tb_A',(1,)),   _get('tb_n',(1,)),   _get('tb_Ea',(1,)),
            _get('tb_oneWay',(1,)), _get('tb_delta_n',(1,)),
            _iget('tb_react_idx',(1,)), _get('tb_react_sto',(1,)),
            _iget('tb_react_offset',(1,)),
            _iget('tb_prod_idx',(1,)),  _get('tb_prod_sto',(1,)),
            _iget('tb_prod_offset',(1,)),
            _iget('tb_idx_r_scatter',(1,)), _iget('tb_idx_p_scatter',(1,)),
            _get('tb_eff_matrix',(1, Ns)),
            np.int32(fa.n_reactions if fa else 0),
            _get('fall_A_k0',(1,)),    _get('fall_b_k0',(1,)),
            _get('fall_Ea_k0',(1,)),
            _get('fall_A_kinf',(1,)),   _get('fall_b_kinf',(1,)),
            _get('fall_Ea_kinf',(1,)),
            _get('fall_oneWay',(1,)),   _get('fall_delta_n',(1,)),
            _get('fall_troe_A',(1,)),   _get('fall_troe_T1',(1,)),
            _get('fall_troe_T2',(1,)),  _get('fall_troe_T3',(1,)),
            _iget('fall_is_troe',(1,)),
            _iget('fall_react_idx',(1,)), _get('fall_react_sto',(1,)),
            _iget('fall_react_offset',(1,)),
            _iget('fall_prod_idx',(1,)),  _get('fall_prod_sto',(1,)),
            _iget('fall_prod_offset',(1,)),
            _iget('fall_idx_r_scatter',(1,)), _iget('fall_idx_p_scatter',(1,)),
            _get('fall_eff_matrix',(1, Ns)),
        ]