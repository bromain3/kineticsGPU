"""
integrator_host.py
==================
Couche host Python pour l'intégrateur Backward Euler Numba.

Solution au dépassement de 4096 bytes de paramètres kernel :
  Tous les arrays chimiques sont sérialisés dans UN SEUL vecteur 1D float64
  (data_flat), accompagné de deux petits arrays int32 (offsets, ncols) qui
  permettent de reconstruire l'accès [i] ou [row, col] dans le kernel.

  data_flat[offsets[k] + j]                → array 1D k, élément j
  data_flat[offsets[k] + row*ncols[k]+col] → array 2D k, élément [row,col]

  Le kernel ne reçoit donc que 3 DeviceNDArrays au lieu de 52.

Layout des 52 arrays (identique à l'ancienne convention _E/_TB/_FO/_TH) :
  [0..12]  : elem_arrays[0..12]
  [13..26] : tb_arrays[0..13]
  [27..48] : fo_arrays[0..21]
  [49..51] : thermo_arrays[0..2]
"""

import math
import numpy as np
from numba import cuda

from kineticsGPU.numba_implementation.elementary_reaction import N_CTX_FIELDS   as N_CTX_ELEM
from kineticsGPU.numba_implementation.three_body_reaction import N_CTX_TB_FIELDS as N_CTX_TB
from kineticsGPU.numba_implementation.falloff_reaction    import N_CTX_FO_FIELDS as N_CTX_FO
from kineticsGPU.numba_implementation.backward_euler import backward_euler_step_kernel


# =============================================================================
#  Sérialisation des arrays chimiques
# =============================================================================

def pack_all_arrays(elem_arrays, tb_arrays, fo_arrays, thermo_arrays):
    """
    Sérialise les 52 arrays chimiques en trois DeviceNDArrays :
      data_flat (float64, 1D) : données concaténées (valeurs aplaties row-major)
      offsets   (int32,   1D) : offsets[k] = indice dans data_flat du 1er élément du k-ème array
      ncols     (int32,   1D) : ncols[k] = nb de colonnes si 2D, 0 si 1D

    Retourne (data_flat_dev, offsets_dev, ncols_dev).
    """
    arrays_list = (
        list(elem_arrays[:13])
        + list(tb_arrays[:14])
        + list(fo_arrays[:22])
        + list(thermo_arrays[:3])
    )
    n_arrays = len(arrays_list)  # 52

    offsets_np = np.zeros(n_arrays, dtype=np.int32)
    ncols_np   = np.zeros(n_arrays, dtype=np.int32)

    # ── Première passe : calcul des offsets ──────────────────────────────
    cursor = 0
    host_arrays = []
    for i, arr in enumerate(arrays_list):
        if hasattr(arr, 'copy_to_host'):
            a = arr.copy_to_host()
        else:
            a = np.asarray(arr, dtype=np.float64)
        if a.dtype != np.float64:
            a = a.astype(np.float64)
        host_arrays.append(a)

        offsets_np[i] = cursor
        ncols_np[i]   = a.shape[1] if a.ndim == 2 else 0
        cursor += a.size

    # ── Deuxième passe : remplissage du vecteur plat ─────────────────────
    flat_np = np.zeros(cursor, dtype=np.float64)
    for i, a in enumerate(host_arrays):
        start = offsets_np[i]
        flat_np[start : start + a.size] = a.ravel()

    return (
        cuda.to_device(flat_np),
        cuda.to_device(offsets_np),
        cuda.to_device(ncols_np),
    )


# =============================================================================
#  Grille CUDA
# =============================================================================

def _grid(ncells, tpb=128):
    """Retourne (tpb,), (blocks,) — blocks couvre ncells threads."""
    return (tpb,), ((ncells + tpb - 1) // tpb,)


# =============================================================================
#  Intégrateur
# =============================================================================

class BackwardEulerNumba:
    """
    Intégrateur Backward Euler implicite GPU — 1 thread par réacteur.

    Tous les buffers GPU sont alloués une seule fois dans __init__.

    Paramètres
    ----------
    elem_arrays, tb_arrays, fo_arrays, thermo_arrays :
        Tuples d'arrays Numba CUDA issus de build_*_gpu_arrays().
    nsp         : nombre d'espèces
    ncells      : nombre de réacteurs
    nreac_elem  : nombre de réactions élémentaires
    nreac_tb    : nombre de réactions three-body
    nreac_fo    : nombre de réactions falloff
    restart     : dimension sous-espace Krylov (défaut 20)
    gmres_tol   : tolérance relative GMRES (défaut 1e-8)
    gmres_maxiter : nb max de cycles restart (défaut 5)
    newton_tol  : tolérance Newton (défaut 1e-8)
    newton_maxiter : nb max itérations Newton (défaut 15)
    """

    def __init__(self,
                 elem_arrays, tb_arrays, fo_arrays, thermo_arrays,
                 nsp, ncells,
                 nreac_elem, nreac_tb, nreac_fo,
                 restart=20,
                 gmres_tol=1e-8,
                 gmres_maxiter=5,
                 newton_tol=1e-8,
                 newton_maxiter=15):

        # ── Sérialisation en un seul vecteur plat ─────────────────────────
        self.data_flat, self.arr_offsets, self.arr_ncols = pack_all_arrays(
            elem_arrays, tb_arrays, fo_arrays, thermo_arrays
        )

        self.nsp      = nsp
        self.ncells   = ncells
        self.N        = nsp + 1
        self.restart  = restart

        self.gmres_tol      = float(gmres_tol)
        self.gmres_maxiter  = int(gmres_maxiter)
        self.newton_tol     = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)

        N   = self.N
        rst = restart
        nc  = ncells

        # ── nreacs : [nreac_elem, nreac_tb, nreac_fo] passé au kernel ────────
        self.nreacs = cuda.to_device(
            np.array([nreac_elem, nreac_tb, nreac_fo], dtype=np.int32))

        # ── Contextes chimiques (gardés pour compatibilité future) ─────────
        self.elem_ctx_buf = cuda.device_array(
            (nc, N_CTX_ELEM, nreac_elem), dtype=np.float64)
        self.tb_ctx_buf   = cuda.device_array(
            (nc, N_CTX_TB,   nreac_tb),   dtype=np.float64)
        self.fo_ctx_buf   = cuda.device_array(
            (nc, N_CTX_FO,   nreac_fo),   dtype=np.float64)

        # ── Buffers réacteur ───────────────────────────────────────────────
        self.rhs_buf     = cuda.device_array((nc, N),   dtype=np.float64)
        self.jvp_tmp_buf = cuda.device_array((nc, N),   dtype=np.float64)
        self.tmp_fwd_buf = cuda.device_array((nc, N),   dtype=np.float64)
        self.vC_buf      = cuda.device_array((nc, nsp), dtype=np.float64)
        self.F_buf       = cuda.device_array((nc, N),   dtype=np.float64)

        # ── Buffers GMRES ──────────────────────────────────────────────────
        self.gmres_V   = cuda.device_array((nc, (rst + 1) * N),  dtype=np.float64)
        self.gmres_H   = cuda.device_array((nc, rst * (rst + 1)), dtype=np.float64)
        self.gmres_cs  = cuda.device_array((nc, rst),             dtype=np.float64)
        self.gmres_sn  = cuda.device_array((nc, rst),             dtype=np.float64)
        self.gmres_e1  = cuda.device_array((nc, rst + 1),         dtype=np.float64)
        self.gmres_y   = cuda.device_array((nc, rst),             dtype=np.float64)
        self.gmres_tmp = cuda.device_array((nc, N),               dtype=np.float64)
        self.gmres_sol = cuda.device_array((nc, N),               dtype=np.float64)

        # Info mémoire
        mb = (nc * (N_CTX_ELEM * nreac_elem +
                    N_CTX_TB   * nreac_tb   +
                    N_CTX_FO   * nreac_fo   +
                    N * 6 + nsp +
                    (rst + 1) * N + rst * (rst + 1) +
                    rst * 3 + (rst + 1))) * 8 / 1e6
        flat_mb = self.data_flat.size * 8 / 1e6
        print(f"[BackwardEulerNumba] Mémoire GPU : {mb:.1f} Mo (buffers) "
              f"+ {flat_mb:.1f} Mo (arrays chimiques plats) "
              f"| ncells={nc}, nsp={nsp}, restart={rst}")

    # -------------------------------------------------------------------------
    def step(self, TC_in_dev, TC_out_dev, h):
        """
        Un pas de temps Backward Euler.
        TC_in_dev  : (ncells, N) DeviceNDArray — état au temps t_n
        TC_out_dev : (ncells, N) DeviceNDArray — état au temps t_{n+1}
        h          : float — pas de temps (s)
        """
        tpb, blocks = _grid(self.ncells)
        backward_euler_step_kernel[blocks, tpb](
            TC_in_dev, TC_out_dev,
            np.float64(h),
            self.data_flat,      # float64[:]  — toutes les données chimiques
            self.arr_offsets,    # int32[:]    — offsets des 52 arrays
            self.arr_ncols,      # int32[:]    — ncols (0=1D, >0=2D)
            self.nreacs,         # int32[3]    — [nreac_elem, nreac_tb, nreac_fo]
            self.rhs_buf,
            self.jvp_tmp_buf,
            self.tmp_fwd_buf,
            self.vC_buf,
            self.F_buf,
            self.gmres_V,
            self.gmres_H,
            self.gmres_cs,
            self.gmres_sn,
            self.gmres_e1,
            self.gmres_y,
            self.gmres_tmp,
            self.gmres_sol,
            np.int32(self.restart),
            np.float64(self.gmres_tol),
            np.int32(self.gmres_maxiter),
            np.float64(self.newton_tol),
            np.int32(self.newton_maxiter),
        )
        cuda.synchronize()

    # -------------------------------------------------------------------------
    def integrate(self, TC_dev, t_end, h,
                  t_start=0.0, callback=None, callback_every=1, verbose=False):
        """
        Intègre de t_start à t_end par pas constants h.
        TC_dev est modifié en place au retour.
        Retourne t_final.
        """
        TC_a = TC_dev
        TC_b = cuda.device_array_like(TC_dev)

        t       = t_start
        n_steps = int(math.ceil((t_end - t_start) / h))
        step_i  = 0

        for _ in range(n_steps):
            h_eff = min(h, t_end - t)
            if h_eff < 1e-15 * abs(t_end):
                break
            self.step(TC_a, TC_b, h_eff)
            TC_a, TC_b = TC_b, TC_a
            t      += h_eff
            step_i += 1
            if verbose and step_i % max(1, n_steps // 10) == 0:
                print(f"  t = {t:.3e} / {t_end:.3e}  (step {step_i}/{n_steps})")
            if callback is not None and step_i % callback_every == 0:
                callback(t, TC_a)

        # Copie résultat dans TC_dev si nécessaire
        if TC_a is not TC_dev:
            cuda.synchronize()
            TC_dev[:] = TC_a
            cuda.synchronize()

        return t