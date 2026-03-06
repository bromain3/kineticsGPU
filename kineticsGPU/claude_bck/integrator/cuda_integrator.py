"""
CUDABackwardEulerIntegrator — v5 (LU + GMRES + timing)

Perf fixes vs v4
────────────────
• Point 2 — élimination des copies CPU→GPU dans la boucle chaude :
    - h_arr  : copié une seule fois dans integrate() si h est constant,
               et uniquement si h_eff change (dernier pas tronqué).
    - t_arr  : plus jamais copié — le kernel n'en a pas besoin pour le calcul
               (t ne sert pas dans la physique, seulement pour du logging).
               Le buffer _d_t_arr est supprimé.
    - conv   : zeroed sur device via un petit kernel dédié (_zero_int_kernel)
               au lieu d'allouer np.zeros + copy_to_device à chaque pas.
    - Les buffers h_host et conv_zeros sont pré-alloués une fois dans __init__.

• Point 3 — copies device→host limitées :
    - Les snapshots demandés par save_every sont accumulés dans un buffer
      device pré-alloué  _d_TC_hist  (n_saves, B, N_state)  et
      _d_conv_hist  (n_saves, B).
    - Un kernel _store_snapshot_kernel copie TC_a et conv dans ce buffer.
    - Une seule paire copy_to_host() à la fin de integrate() transfère
      tout l'historique en un seul appel.
    - t_hist reste une simple liste Python (pas de transfer GPU).
"""

import math
import numpy as np
import numba.cuda as cuda
from numba.cuda import get_current_device
from numba import cuda as nb_cuda, int32, float64
import time

from ..mechanism.mechanism_data import MechanismData
from ..kernels.integrator_kernel import (
    backward_euler_per_thread_kernel,
    N_TIMER_PHASES, PHASE_NAMES
)
_GROUPS = [
    ("PREDICTOR",       [0, 1, 2],               "─"),
    ("NEWTON / RHS",    [3, 4, 5],               "─"),
    ("LU — Jacobian",   [7, 8, 9, 10, 11, 12],   "─"),
    ("GMRES — Krylov",  [13, 14, 15, 16, 17, 18, 19, 20, 21],  "─"),
    ("STEP UPDATE",     [22],                    "─"),
    ("AGGREGATES",      [23, 24, 25, 26],        "═"),
]
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


# ─────────────────────────────────────────────────────────────────────────────
#  Helper kernels  (device-side utilities, launched once per step or snapshot)
# ─────────────────────────────────────────────────────────────────────────────

@nb_cuda.jit
def _zero_int_kernel(arr):
    """Zero a 1-D int32 array in-place on device — avoids CPU allocation."""
    i = nb_cuda.blockIdx.x * nb_cuda.blockDim.x + nb_cuda.threadIdx.x
    if i < arr.shape[0]:
        arr[i] = int32(0)


@nb_cuda.jit
def _fill_float_kernel(arr, val):
    """Fill a 1-D float array with val on device — used to broadcast h_eff."""
    i = nb_cuda.blockIdx.x * nb_cuda.blockDim.x + nb_cuda.threadIdx.x
    if i < arr.shape[0]:
        arr[i] = val


@nb_cuda.jit
def _store_snapshot_kernel(TC_src, conv_src, TC_hist, conv_hist, snap_idx):
    """
    Copy one row of TC_src (B, N_state) and conv_src (B,) into the
    pre-allocated history buffers at position snap_idx.
    One thread per reactor.
    """
    b = nb_cuda.blockIdx.x * nb_cuda.blockDim.x + nb_cuda.threadIdx.x
    if b >= TC_src.shape[0]:
        return
    n_state = TC_src.shape[1]
    for i in range(n_state):
        TC_hist[snap_idx, b, i] = TC_src[b, i]
    conv_hist[snap_idx, b] = conv_src[b]


# ─────────────────────────────────────────────────────────────────────────────
#  Main integrator class
# ─────────────────────────────────────────────────────────────────────────────

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
    gmres_tol : float
    gmres_maxiter : int
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

        self._clock_khz = get_current_device().CLOCK_RATE
        self._d_mech = mechanism.to_cuda_arrays()
        self._alloc_buffers()
        self._mech_args = self._pack_mech_args()

        # ── Pré-allocation des buffers hôte réutilisés dans la boucle ────
        # Évite numpy allocations répétées dans step()
        self._h_host    = np.empty(n_reactors, dtype=dtype)   # pour h_arr
        self._h_current = None   # dernière valeur broadcastée dans _d_h_arr

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
        def dummy2d():   return cuda.device_array((B, 1), dtype=dt)

        # ── Ping-pong state ───────────────────────────────────────────────
        self._d_TC_a  = buf((B, Nst))
        self._d_TC_b  = buf((B, Nst))

        # ── h_arr : alloué une fois, mis à jour seulement si h change ────
        # t_arr : SUPPRIMÉ — le kernel n'utilise pas t pour la physique.
        self._d_h_arr = buf((B,))

        # ── conv : buffer device permanent, zeroé via kernel ─────────────
        self._d_conv  = ibuf((B,))

        # ── Newton / LU ───────────────────────────────────────────────────
        self._d_J     = buf((B, Nst * Nst))
        self._d_LU    = buf((B, Nst * Nst))
        self._d_piv   = ibuf((B, Nst))
        self._d_F     = buf((B, Nst))
        self._d_dx    = buf((B, Nst))
        self._d_xk    = buf((B, Nst))
        self._d_omega = buf((B, Ns))

        # ── GMRES ─────────────────────────────────────────────────────────
        if self.linear_solver == 1:
            self._d_gmres_V   = buf((B, (R + 1) * Nst))
            self._d_gmres_H   = buf((B, (R + 1) * R))
            self._d_gmres_cs  = buf((B, R))
            self._d_gmres_sn  = buf((B, R))
            self._d_gmres_g   = buf((B, R + 1))
            self._d_gmres_w   = buf((B, Nst))
            self._d_gmres_y   = buf((B, R))
            self._d_gmres_vC  = buf((B, Nst))
            self._d_gmres_Jvo = buf((B, Nst))
        else:
            self._d_gmres_V   = dummy2d()
            self._d_gmres_H   = dummy2d()
            self._d_gmres_cs  = dummy2d()
            self._d_gmres_sn  = dummy2d()
            self._d_gmres_g   = dummy2d()
            self._d_gmres_w   = dummy2d()
            self._d_gmres_y   = dummy2d()
            self._d_gmres_vC  = dummy2d()
            self._d_gmres_Jvo = dummy2d()

        # ── Timing / counting ─────────────────────────────────────────────
        self._d_timers = cuda.device_array((B, N_TIMER_PHASES), dtype=np.int64)
        self._d_timers.copy_to_device(
            np.zeros((B, N_TIMER_PHASES), dtype=np.int64)
        )
        self._d_iters = cuda.device_array((B, 2), dtype=np.int64)
        self._d_iters.copy_to_device(np.zeros((B, 2), dtype=np.int64))

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
    #  Helpers pour lancer les kernels utilitaires
    # ─────────────────────────────────────────────────────────────────────

    def _blocks_for(self, n):
        return math.ceil(n / THREADS_PER_BLOCK)

    def _zero_conv(self):
        """Zéroer _d_conv entièrement sur device, sans aucune copie CPU."""
        _zero_int_kernel[self._blocks_for(self.B), THREADS_PER_BLOCK](
            self._d_conv
        )

    def _set_h(self, h_eff: float):
        """
        Broadcaster h_eff dans _d_h_arr uniquement si la valeur a changé.
        Évite un copy_to_device + np.full à chaque pas quand h est constant.
        """
        if h_eff != self._h_current:
            _fill_float_kernel[self._blocks_for(self.B), THREADS_PER_BLOCK](
                self._d_h_arr, self.dtype(h_eff)
            )
            self._h_current = h_eff

    # ─────────────────────────────────────────────────────────────────────
    #  step()
    # ─────────────────────────────────────────────────────────────────────

    def step(self, h_eff: float, do_timing: bool = False, do_count: bool = False):
        """
        Effectue un pas de backward Euler.

        Paramètre t supprimé : le kernel n'utilise pas t pour la physique.
        h_eff est broadcasté sur device uniquement s'il a changé depuis le
        dernier appel (cas du dernier pas tronqué).
        conv est zéroé via un kernel device (pas de copie CPU).
        """
        self._set_h(h_eff)
        self._zero_conv()

        blocks = self._blocks_for(self.B)
        backward_euler_per_thread_kernel[blocks, THREADS_PER_BLOCK](
            self._d_TC_a, self._d_TC_b,
            self._d_h_arr,
            # Newton / LU
            self._d_J, self._d_LU, self._d_piv,
            self._d_F, self._d_dx, self._d_xk, self._d_omega,
            # GMRES
            self._d_gmres_V,
            self._d_gmres_H,
            self._d_gmres_cs,
            self._d_gmres_sn,
            self._d_gmres_g,
            self._d_gmres_w,
            self._d_gmres_y,
            self._d_gmres_vC,
            self._d_gmres_Jvo,
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
            self.newton_tol,
            self.newton_maxiter,
            self.reuse_jac,
            np.int32(self.linear_solver),
            np.int32(self.gmres_restart),
            self.gmres_tol,
            np.int32(self.gmres_maxiter),
            self._d_timers,
            np.int32(1 if do_timing else 0),
            self._d_iters,
            np.int32(1 if do_count else 0),
            *self._mech_args,
        )
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
        Intègre de t0 à t_end avec pas fixe h.

        Optimisations vs v4 :
        - h_arr et conv ne sont plus copiés via CPU dans la boucle.
        - Les snapshots sont écrits dans un buffer device pré-alloué
          (_d_TC_hist, _d_conv_hist) via un kernel léger.
        - Une seule paire copy_to_host() est effectuée à la fin.

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
        n_saves = math.ceil(n_steps / save_every)

        # ── Allouer les buffers d'historique sur device ───────────────────
        d_TC_hist   = cuda.device_array(
            (n_saves, self.B, self.N_state), dtype=self.dtype
        )
        d_conv_hist = cuda.device_array(
            (n_saves, self.B), dtype=np.int32
        )
        t_hist_list = []   # liste Python légère, pas de GPU pour les scalaires

        snap_idx   = 0
        blocks_b   = self._blocks_for(self.B)

        wall_timer = 0.0

        for step_i in range(n_steps):
            h_eff = min(h, t_end - t)
            if h_eff <= 0.0:
                break
            print(t)
            wall_timer -= time.perf_counter()
            self.step(h_eff, do_timing=do_timing, do_count=do_count)
            wall_timer += time.perf_counter()

            t += h_eff

            if step_i % save_every == 0 and snap_idx < n_saves:
                # Copie device→device via kernel : pas de synchronisation forcée
                _store_snapshot_kernel[blocks_b, THREADS_PER_BLOCK](
                    self._d_TC_a,
                    self._d_conv,
                    d_TC_hist,
                    d_conv_hist,
                    np.int32(snap_idx),
                )
                t_hist_list.append(t)
                snap_idx += 1

        # ── Synchronisation unique + copie hôte en bloc ───────────────────
        cuda.synchronize()

        TC_hist   = d_TC_hist.copy_to_host()[:snap_idx]     # (n_saved, B, N_state)
        conv_hist = d_conv_hist.copy_to_host()[:snap_idx]   # (n_saved, B)
        t_hist    = np.array(t_hist_list)

        print(f'simulation ended in {wall_timer:.4f}s (wall, kernel launches only)')

        return TC_hist, t_hist, conv_hist

    # ─────────────────────────────────────────────────────────────────────
    #  Timing helpers
    # ─────────────────────────────────────────────────────────────────────

    def reset_timers(self):
        self._d_timers.copy_to_device(
            np.zeros((self.B, N_TIMER_PHASES), dtype=np.int64)
        )

    def get_timers(self) -> np.ndarray:
        return self._d_timers.copy_to_host()

    def get_timers_us(self) -> np.ndarray:
        raw = self.get_timers().astype(np.float64)
        return raw / (self._clock_khz * 1e3)

    def print_timers(self, n_steps: int = 1):
        """
        Remplace print_timers() dans CUDABackwardEulerIntegrator.
        Affiche un tableau groupé avec mean / max / std / % pour chaque
        des 27 phases, plus les totaux agrégés.

        Usage:
            integrator.print_timers_detailed(n_steps=n_steps)
        """

        raw_clk = self.get_timers()  # (B, 27) int64
        us_all = raw_clk.astype(np.float64) / (self._clock_khz * 1e3)

        mean_us = us_all.mean(axis=0)
        max_us = us_all.max(axis=0)
        std_us = us_all.std(axis=0)

        # Total = somme des phases de base (pas les agrégats 23..26)
        total_us = mean_us[:23].sum()

        solver_name = 'LU' if self.linear_solver == 0 else 'GMRES'
        W = 76
        print(f"\n{'═' * W}")
        print(f"  Ultra-detailed timing  [{solver_name}, B={self.B}, steps={n_steps}]")
        print(f"{'═' * W}")
        print(f"  {'Phase':<22}  {'Mean µs':>10}  {'Max µs':>10}  "
              f"{'Std µs':>9}  {'% total':>8}  {'µs/step':>9}")
        print(f"{'─' * W}")

        for group_name, indices, sep in _GROUPS:
            print(f"  ── {group_name}")
            for idx in indices:
                if idx >= len(PHASE_NAMES):
                    continue
                name = PHASE_NAMES[idx]
                m = mean_us[idx]
                mx = max_us[idx]
                sd = std_us[idx]
                pct = 100.0 * m / total_us if total_us > 0 else 0.0
                mps = m / n_steps if n_steps > 0 else m
                marker = "  " if not name.startswith("=") else "→ "
                print(f"  {marker}{name:<22}  {m:>10.3f}  {mx:>10.3f}  "
                      f"{sd:>9.3f}  {pct:>7.2f}%  {mps:>9.4f}")
            print(f"  {sep * W}")

        print(f"  {'WALL TOTAL (base phases)':<22}  {total_us:>10.3f}")
        print(f"{'═' * W}\n")

        # ── Per-step breakdown (si multi-pas) ────────────────────────────────
        if n_steps > 1:
            print(f"  Per-step mean (÷{n_steps}) for phases > 0.01 µs/step:")
            for idx in range(23):  # base phases only
                mps = mean_us[idx] / n_steps
                if mps > 0.01:
                    print(f"    [{idx:2d}] {PHASE_NAMES[idx]}  {mps:.4f} µs/step")
            print()

        # ── JVP call count estimate ───────────────────────────────────────────
        # GMRES : chaque Krylov vector = 1 JVP (phases 14+15)
        # LU    : chaque Jacobian assem = n_state JVP calls (phases 7+8)
        total_jvp_time = mean_us[7] + mean_us[8] + mean_us[14] + mean_us[15] \
                         + mean_us[20]
        print(f"  ∑ JVP time (all sites) : {total_jvp_time:.3f} µs  "
              f"({100 * total_jvp_time / total_us:.1f}% of total)\n")

    # ─────────────────────────────────────────────────────────────────────
    #  _pack_mech_args  (inchangé)
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