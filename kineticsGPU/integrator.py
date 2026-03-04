"""
integrator.py
=============
Backward-Euler integrator for chemical kinetics — Numba CUDA back-end.

API is intentionally close to the existing Torch version
(BDFTorchContext.BackwardEulerIntegrator) so the two can be swapped for
benchmarking with minimal code changes.

Key design choices
------------------
* NO CPU↔GPU transfer inside the integration loop.
  Torch tensors are accessed via __cuda_array_interface__ so Numba device
  arrays point to the same CUDA memory — zero-copy bridge.

* Strategy flag  ('per_thread'  or  'per_block')
  Controls the launch configuration for every kernel call.
  Both strategies use:
    - Analytic JVP   (pt_jvp_* / pb_jvp_* kernels)
    - Restarted GMRES with Givens rotations   (pt/pb_gmres_step)

* Three reaction-type batches are dispatched sequentially per step:
  elementary → three-body → falloff.
  All writes to omega / Jv use CUDA atomicAdd (safe, no temporary
  per-type arrays needed).

* Workspace arrays (V, H, cs, sn, g_vec, beta, F, Av) are preallocated
  once in __init__ and reused across steps to avoid repeated allocation.

* The Newton loop and GMRES restart loop run on the HOST (Python),
  but every operation inside is a CUDA kernel launch — no data copies.

Usage
-----
    solver = NumbaSolver(
        packed,                  # NumbaPacked object
        strategy='per_thread',   # or 'per_block'
        tpb=64,                  # threads per block
        newton_tol=1e-6,
        newton_maxiter=20,
        gmres_restart=20,
        gmres_tol=1e-6,
        gmres_maxiter=50,
        verbose=False,
    )
    # TC_torch : (ncells, 1+nsp) torch Tensor on CUDA
    TC_new = solver.step(TC_torch, t, h)

Adiabatic constant-volume RHS
------------------------------
The RHS and JVP for the adiabatic CV reactor are provided as
  rhs_adiabatic_cv(solver, TC_dev)  → dTC_dev
  jvp_adiabatic_cv(solver, TC_dev, dTC_dev)  → J·dTC_dev
These are stand-alone functions that call the Numba kernels and can be
replaced by the user for other reactor types.
"""

import math
import numpy as np
import torch
from numba import cuda
import numba as nb

from .kernels.rhs_jvp import (
    pt_rhs_elementary, pt_rhs_threebody, pt_rhs_falloff,
    pt_jvp_elementary, pt_jvp_threebody, pt_jvp_falloff,
    pb_rhs_elementary,                   pb_jvp_elementary,
    zero_2d,
)
from .kernels.gmres import (
    pt_gmres_step, pb_gmres_step,
    compute_residual_kernel,
    compute_matvec_preprocess,
    initialise_gmres,
)

# NASA thermo helpers (device kernels for cv, u, needed in RHS dT/dt)
from .kernels.thermo_cv import (
    pt_dT_from_omega,   # fills dT/dt from omega and current state
    pt_jvp_dT,         # JVP of dT/dt
)

R_GAS = 8.31446261815324

# ─────────────────────────────────────────────────────────────────────────────
# Torch → Numba zero-copy bridge
# ─────────────────────────────────────────────────────────────────────────────

def torch_to_numba(tensor):
    """
    Return a Numba DeviceNDArray that shares memory with a CUDA torch Tensor.
    No data is copied.  The tensor must be contiguous and on CUDA.
    """
    assert tensor.is_cuda and tensor.is_contiguous(), \
        "Tensor must be contiguous and on CUDA"
    return cuda.as_cuda_array(tensor)


# ─────────────────────────────────────────────────────────────────────────────
# Launch configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pt_config(ncells, tpb):
    """Per-thread: 1-D grid, tpb threads/block."""
    return (math.ceil(ncells / tpb),), (tpb,)

def _pb_config(ncells, tpb):
    """Per-block: ncells blocks, tpb threads/block (power of 2)."""
    return (ncells,), (tpb,)


# ─────────────────────────────────────────────────────────────────────────────
# Main solver class
# ─────────────────────────────────────────────────────────────────────────────

class NumbaSolver:
    """
    GPU backward-Euler integrator with analytic JVP + GMRES.

    Parameters
    ----------
    packed        : NumbaPacked   — preallocated device data (reactions + thermo)
    strategy      : str           — 'per_thread' or 'per_block'
    tpb           : int           — threads per block (must be power of 2 for pb)
    newton_tol    : float
    newton_maxiter: int
    gmres_restart : int           — Krylov restart dimension m
    gmres_tol     : float
    gmres_maxiter : int           — max GMRES iterations (multiple restarts)
    verbose       : bool
    """

    def __init__(self,
                 packed,
                 strategy: str = 'per_thread',
                 tpb: int = 64,
                 newton_tol: float = 1e-6,
                 newton_maxiter: int = 20,
                 gmres_restart: int = 20,
                 gmres_tol: float = 1e-6,
                 gmres_maxiter: int = 50,
                 verbose: bool = False):

        self.packed   = packed
        self.strategy = strategy
        self.tpb      = tpb
        self.N        = 1 + packed.n_species   # state size = T + C
        self.verbose  = verbose

        self.newton_tol     = newton_tol
        self.newton_maxiter = newton_maxiter
        self.gmres_restart  = gmres_restart
        self.gmres_tol      = gmres_tol
        self.gmres_maxiter  = gmres_maxiter

        # Workspace arrays — allocated lazily on first step call
        self._ncells_alloc = None
        self._workspace    = {}

    # ── Lazy workspace allocation ─────────────────────────────────────────────
    def _ensure_workspace(self, ncells):
        if self._ncells_alloc == ncells:
            return
        N = self.N; m = self.gmres_restart
        alloc = lambda shape: cuda.device_array(shape, dtype=np.float64)

        self._workspace = {
            'omega':     alloc((ncells, N - 1)),   # species production rates
            'Jv_omega':  alloc((ncells, N - 1)),   # JVP of omega
            'dTC':       alloc((ncells, N)),        # full state RHS
            'Jv_dTC':    alloc((ncells, N)),        # JVP of full RHS
            'F':         alloc((ncells, N)),        # Newton residual
            'Av':        alloc((ncells, N)),        # matvec output  A·v
            'V':         alloc((ncells, m + 1, N)),
            'H':         alloc((ncells, m + 1, m)),
            'cs':        alloc((ncells, m)),
            'sn':        alloc((ncells, m)),
            'g_vec':     alloc((ncells, m + 1)),
            'beta':      alloc((ncells,)),
            'res_norms': alloc((ncells,)),
            'xk':        alloc((ncells, N)),        # Newton iterate
        }
        self._ncells_alloc = ncells

    # ── Zero helper (avoids cudaMemset Python overhead) ───────────────────────
    def _zero(self, arr):
        if arr.ndim == 2:
            grid = (math.ceil(arr.shape[0] / 16),
                    math.ceil(arr.shape[1] / 16))
            block = (16, 16)

            zero_2d[grid, block](arr)
        else:
            raise NotImplementedError("Only 2-D arrays handled by _zero")

    # ── Select launch config based on strategy ────────────────────────────────
    def _cfg(self, ncells):
        if self.strategy == 'per_thread':
            return _pt_config(ncells, self.tpb)
        else:
            return _pb_config(ncells, self.tpb)

    # ── Dispatch RHS kernels (omega only — no dT yet) ─────────────────────────
    def _compute_omega(self, T_dev, C_dev, omega_dev, ncells):
        """Fill omega_dev = sum of all reaction contributions."""
        self._zero(omega_dev)
        p  = self.packed
        cl = self._cfg(ncells)

        if p.has_elem:
            e = p.elem
            if self.strategy == 'per_thread':
                pt_rhs_elementary[cl[0], cl[1]](
                    T_dev, C_dev,
                    p.d_coef_low, p.d_coef_high, p.d_Tmid,
                    e['react_idx'], e['react_sto'], e['r_start'], e['r_end'],
                    e['prod_idx'],  e['prod_sto'],  e['p_start'], e['p_end'],
                    e['A_arr'], e['n_arr'], e['Ea_arr'], e['oneWay'], e['delta_n'],
                    omega_dev,
                )
            else:
                pb_rhs_elementary[cl[0], cl[1]](
                    T_dev, C_dev,
                    p.d_coef_low, p.d_coef_high, p.d_Tmid,
                    e['react_idx'], e['react_sto'], e['r_start'], e['r_end'],
                    e['prod_idx'],  e['prod_sto'],  e['p_start'], e['p_end'],
                    e['A_arr'], e['n_arr'], e['Ea_arr'], e['oneWay'], e['delta_n'],
                    omega_dev,
                )

        if p.has_tb:
            t = p.tb
            if self.strategy == 'per_thread':
                pt_rhs_threebody[cl[0], cl[1]](
                    T_dev, C_dev,
                    p.d_coef_low, p.d_coef_high, p.d_Tmid,
                    t['react_idx'], t['react_sto'], t['r_start'], t['r_end'],
                    t['prod_idx'],  t['prod_sto'],  t['p_start'], t['p_end'],
                    t['A_arr'], t['n_arr'], t['Ea_arr'], t['oneWay'], t['delta_n'],
                    t['eff_matrix'],
                    omega_dev,
                )

        if p.has_falloff:
            f = p.fo
            if self.strategy == 'per_thread':
                pt_rhs_falloff[cl[0], cl[1]](
                    T_dev, C_dev,
                    p.d_coef_low, p.d_coef_high, p.d_Tmid,
                    f['react_idx'], f['react_sto'], f['r_start'], f['r_end'],
                    f['prod_idx'],  f['prod_sto'],  f['p_start'], f['p_end'],
                    f['A_k0'], f['b_k0'], f['Ea_k0'],
                    f['A_kinf'], f['b_kinf'], f['Ea_kinf'],
                    f['oneWay'], f['delta_n'],
                    f['eff_matrix'],
                    f['troe_A'], f['troe_T1'], f['troe_T2'], f['troe_T3'],
                    f['is_troe'],
                    omega_dev,
                )

    # ── Dispatch JVP kernels ──────────────────────────────────────────────────
    def _compute_jvp(self, T_dev, C_dev, vT_dev, vC_dev, Jv_omega_dev, ncells):
        """Fill Jv_omega_dev = (d omega / d[T,C]) · [vT, vC]."""
        self._zero(Jv_omega_dev)
        p  = self.packed
        cl = self._cfg(ncells)

        if p.has_elem:
            e = p.elem
            if self.strategy == 'per_thread':
                pt_jvp_elementary[cl[0], cl[1]](
                    T_dev, C_dev,
                    p.d_coef_low, p.d_coef_high, p.d_Tmid,
                    e['react_idx'], e['react_sto'], e['r_start'], e['r_end'],
                    e['prod_idx'],  e['prod_sto'],  e['p_start'], e['p_end'],
                    e['A_arr'], e['n_arr'], e['Ea_arr'], e['oneWay'], e['delta_n'],
                    vT_dev, vC_dev,
                    Jv_omega_dev,
                )
            else:
                pb_jvp_elementary[cl[0], cl[1]](
                    T_dev, C_dev,
                    p.d_coef_low, p.d_coef_high, p.d_Tmid,
                    e['react_idx'], e['react_sto'], e['r_start'], e['r_end'],
                    e['prod_idx'],  e['prod_sto'],  e['p_start'], e['p_end'],
                    e['A_arr'], e['n_arr'], e['Ea_arr'], e['oneWay'], e['delta_n'],
                    vT_dev, vC_dev,
                    Jv_omega_dev,
                )

        if p.has_tb:
            t = p.tb
            if self.strategy == 'per_thread':
                pt_jvp_threebody[cl[0], cl[1]](
                    T_dev, C_dev,
                    p.d_coef_low, p.d_coef_high, p.d_Tmid,
                    t['react_idx'], t['react_sto'], t['r_start'], t['r_end'],
                    t['prod_idx'],  t['prod_sto'],  t['p_start'], t['p_end'],
                    t['A_arr'], t['n_arr'], t['Ea_arr'], t['oneWay'], t['delta_n'],
                    t['eff_matrix'],
                    vT_dev, vC_dev,
                    Jv_omega_dev,
                )

        if p.has_falloff:
            f = p.fo
            if self.strategy == 'per_thread':
                pt_jvp_falloff[cl[0], cl[1]](
                    T_dev, C_dev,
                    p.d_coef_low, p.d_coef_high, p.d_Tmid,
                    f['react_idx'], f['react_sto'], f['r_start'], f['r_end'],
                    f['prod_idx'],  f['prod_sto'],  f['p_start'], f['p_end'],
                    f['A_k0'], f['b_k0'], f['Ea_k0'],
                    f['A_kinf'], f['b_kinf'], f['Ea_kinf'],
                    f['oneWay'], f['delta_n'],
                    f['eff_matrix'],
                    f['troe_A'], f['troe_T1'], f['troe_T2'], f['troe_T3'],
                    f['is_troe'],
                    vT_dev, vC_dev,
                    Jv_omega_dev,
                )

    # ── Public step API ───────────────────────────────────────────────────────
    def step(self, TC_torch: torch.Tensor, t: float, h: float) -> torch.Tensor:
        """
        Advance the state by one time step using implicit Euler + GMRES.

        Parameters
        ----------
        TC_torch : torch.Tensor  (ncells, 1+nsp)  on CUDA, float64
                   Columns: [T, C_0, C_1, ..., C_{nsp-1}]
        t        : float   current time (used for RHS if time-dependent)
        h        : float   time step size

        Returns
        -------
        torch.Tensor  (ncells, 1+nsp)  on same CUDA device, same dtype.
                      The returned tensor is a NEW allocation (not in-place).
        """
        assert TC_torch.is_cuda and TC_torch.dtype == torch.float64
        ncells = TC_torch.shape[0]
        self._ensure_workspace(ncells)
        ws = self._workspace

        # Zero-copy bridge: Torch → Numba device arrays
        TC_nb = torch_to_numba(TC_torch.contiguous())

        # Allocate output (will hold updated state — starts as copy of TC_torch)
        TC_new_torch = TC_torch.clone()
        xk = torch_to_numba(TC_new_torch)

        # x0 is the original state (read-only)
        x0 = TC_nb

        # Split views:  T = TC[:,0],  C = TC[:,1:]
        # We pass the full TC array and use offset=0 for T, offset=1 for C
        # inside the kernels via a wrapper kernel.

        T_nb = torch_to_numba(TC_torch[:, 0].contiguous())
        C_nb = torch_to_numba(TC_torch[:, 1:].contiguous())

        # ── Initial guess: explicit Euler ─────────────────────────────────────
        # Compute omega(T0, C0) → dTC → xk = x0 + h * f(x0)
        self._compute_omega(T_nb, C_nb, ws['omega'], ncells)

        pt_dT_from_omega[self._cfg(ncells)[0], self._cfg(ncells)[1]](
            T_nb, C_nb, ws['omega'], ws['dTC'],
            self.packed.d_coef_low, self.packed.d_coef_high, self.packed.d_Tmid,
        )
        # xk = x0 + h * dTC
        _axpby_kernel[self._cfg(ncells)[0], self._cfg(ncells)[1]](
            xk, TC_nb, ws['dTC'], h, self.N)

        # ── Newton iterations ─────────────────────────────────────────────────
        for nit in range(self.newton_maxiter):

            # Split xk into T and C views
            xk_torch = TC_new_torch   # kept in sync
            T_xk = torch_to_numba(xk_torch[:, 0].contiguous())
            C_xk = torch_to_numba(xk_torch[:, 1:].contiguous())

            # Residual  F = xk - x0 - h * f(xk)
            self._compute_omega(T_xk, C_xk, ws['omega'], ncells)
            pt_dT_from_omega[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                T_xk, C_xk, ws['omega'], ws['dTC'],
                self.packed.d_coef_low, self.packed.d_coef_high, self.packed.d_Tmid,
            )
            compute_residual_kernel[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                xk, x0, ws['dTC'], h, ws['F'], ws['res_norms'])

            # Check convergence (single max reduction — done on GPU via cupy)
            import cupy as cp
            res_max = float(cp.max(cp.asarray(ws['res_norms'])))
            if self.verbose:
                print(f"  [Newton {nit}] max residual = {res_max:.3e}")
            if res_max < self.newton_tol:
                break

            # ── GMRES to solve  (I + h*J) · dx = -F ──────────────────────────
            # Initialise: r = -F,  V[:,0] = r/‖r‖
            _negate_inplace[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                ws['F'], ws['Av'])   # Av temporarily holds -F = rhs_gmres
            initialise_gmres[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                ws['Av'], ws['V'], ws['g_vec'], ws['beta'], self.N)

            m = self.gmres_restart
            max_restarts = math.ceil(self.gmres_maxiter / m)

            for restart_idx in range(max_restarts):
                # Reset H, cs, sn for fresh restart
                # (g_vec[:,0] and V[:,0,:] are set at end of previous restart)

                for j in range(m):
                    # ── matvec: Av = (I + h*J) · V[:,j,:] ────────────────────
                    Vj_torch = torch.as_tensor(ws['V'][:, j, :],
                                               device=TC_torch.device)
                    vT = Vj_torch[:, 0].contiguous()
                    vC = Vj_torch[:, 1:].contiguous()
                    vT_nb = torch_to_numba(vT)
                    vC_nb = torch_to_numba(vC)

                    # J · vj via analytic JVP
                    self._compute_jvp(T_xk, C_xk, vT_nb, vC_nb,
                                      ws['Jv_omega'], ncells)
                    # also need JVP of dT/dt
                    pt_jvp_dT[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                        T_xk, C_xk, ws['omega'], ws['dTC'],
                        vT_nb, vC_nb, ws['Jv_omega'], ws['Jv_dTC'],
                        self.packed.d_coef_low, self.packed.d_coef_high,
                        self.packed.d_Tmid,
                    )
                    # Av = Vj + h * J·Vj
                    compute_matvec_preprocess[
                        (math.ceil(ncells * self.N / (self.tpb * self.tpb)),),
                        (self.tpb, self.tpb)](
                        torch_to_numba(Vj_torch.contiguous()),
                        ws['Jv_dTC'], h, ws['Av'])

                    # ── GMRES Arnoldi + Givens step ───────────────────────────
                    if self.strategy == 'per_thread':
                        pt_gmres_step[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                            xk, ws['F'], ws['V'], ws['H'],
                            ws['cs'], ws['sn'], ws['g_vec'],
                            ws['Av'], j, ws['beta'], self.gmres_tol,
                            self.N, 0, m,
                        )
                    else:
                        shm_bytes = self.tpb * 8   # sizeof(float64)
                        pb_gmres_step[self._cfg(ncells)[0],
                                      self._cfg(ncells)[1],
                                      shm_bytes](
                            xk, ws['F'], ws['V'], ws['H'],
                            ws['cs'], ws['sn'], ws['g_vec'],
                            ws['Av'], j, ws['beta'], self.gmres_tol,
                            self.N, 0, m,
                        )

                    # Check GMRES convergence via |g_vec[:,j+1]|
                    # (this is a host-side check on a single scalar copied from
                    # GPU — the only CPU↔GPU transfer allowed per iteration)
                    g_residual = float(
                        cp.max(cp.abs(cp.asarray(ws['g_vec'])[:, j+1])))
                    if self.verbose:
                        print(f"    [GMRES j={j}] g_res = {g_residual:.3e}")
                    if g_residual < self.gmres_tol:
                        # back-solve
                        if self.strategy == 'per_thread':
                            pt_gmres_step[self._cfg(ncells)[0],
                                          self._cfg(ncells)[1]](
                                xk, ws['F'], ws['V'], ws['H'],
                                ws['cs'], ws['sn'], ws['g_vec'],
                                ws['Av'], j+1, ws['beta'], self.gmres_tol,
                                self.N, 1, j+1,
                            )
                        else:
                            pb_gmres_step[self._cfg(ncells)[0],
                                          self._cfg(ncells)[1],
                                          shm_bytes](
                                xk, ws['F'], ws['V'], ws['H'],
                                ws['cs'], ws['sn'], ws['g_vec'],
                                ws['Av'], j+1, ws['beta'], self.gmres_tol,
                                self.N, 1, j+1,
                            )
                        break
                else:
                    # Full restart: back-solve with full m
                    if self.strategy == 'per_thread':
                        pt_gmres_step[self._cfg(ncells)[0],
                                      self._cfg(ncells)[1]](
                            xk, ws['F'], ws['V'], ws['H'],
                            ws['cs'], ws['sn'], ws['g_vec'],
                            ws['Av'], m, ws['beta'], self.gmres_tol,
                            self.N, 1, m,
                        )
                    else:
                        pb_gmres_step[self._cfg(ncells)[0],
                                      self._cfg(ncells)[1],
                                      shm_bytes](
                            xk, ws['F'], ws['V'], ws['H'],
                            ws['cs'], ws['sn'], ws['g_vec'],
                            ws['Av'], m, ws['beta'], self.gmres_tol,
                            self.N, 1, m,
                        )
                    # Re-initialise for next restart
                    # recompute residual
                    T_xk2 = torch_to_numba(TC_new_torch[:, 0].contiguous())
                    C_xk2 = torch_to_numba(TC_new_torch[:, 1:].contiguous())
                    self._compute_omega(T_xk2, C_xk2, ws['omega'], ncells)
                    pt_dT_from_omega[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                        T_xk2, C_xk2, ws['omega'], ws['dTC'],
                        self.packed.d_coef_low, self.packed.d_coef_high,
                        self.packed.d_Tmid,
                    )
                    compute_residual_kernel[
                        self._cfg(ncells)[0], self._cfg(ncells)[1]](
                        xk, x0, ws['dTC'], h, ws['F'], ws['res_norms'])
                    _negate_inplace[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                        ws['F'], ws['Av'])
                    initialise_gmres[self._cfg(ncells)[0], self._cfg(ncells)[1]](
                        ws['Av'], ws['V'], ws['g_vec'], ws['beta'], self.N)

                    g_res_restart = float(
                        cp.max(cp.abs(cp.asarray(ws['beta']))))
                    if g_res_restart < self.gmres_tol:
                        break

            # Sync TC_new_torch ← xk (already pointing to same CUDA memory)
            # No copy needed; both are views of TC_new_torch.

        return TC_new_torch


# ─────────────────────────────────────────────────────────────────────────────
# Small utility kernels
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit
def _axpby_kernel(out, x0, f, h, N):
    """out[cell, k] = x0[cell, k] + h * f[cell, k]"""
    cell, k = cuda.grid(2)
    if cell < out.shape[0] and k < N:
        out[cell, k] = x0[cell, k] + h * f[cell, k]


@cuda.jit
def _negate_inplace(src, dst):
    """dst = -src  (per element, 2-D)"""
    cell, k = cuda.grid(2)
    if cell < src.shape[0] and k < src.shape[1]:
        dst[cell, k] = -src[cell, k]