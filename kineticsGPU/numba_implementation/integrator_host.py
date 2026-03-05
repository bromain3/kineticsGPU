"""
integrator_host.py
==================
Couche host Python pour l'intégrateur Backward Euler Numba.

Point clé : pack_all_arrays() fusionne les 4 tuples d'arrays chimiques
en un seul tuple de 52 éléments pour contourner la limite de 4096 bytes
d'espace de paramètres des kernels CUDA.

Usage :
    integrator = BackwardEulerNumba(
        elem_arrays, tb_arrays, fo_arrays, thermo_arrays,
        nsp=53, ncells=512,
        nreac_elem=206, nreac_tb=15, nreac_fo=21,
        restart=20, gmres_tol=1e-8, gmres_maxiter=5,
        newton_tol=1e-8, newton_maxiter=15,
    )
    # Pas de temps unique
    integrator.step(TC_in_dev, TC_out_dev, h)

    # Intégration complète sur [0, t_end]
    integrator.integrate(TC_dev, t_end=1e-3, h=1e-6)
"""

import math
import numpy as np
from numba import cuda

from kineticsGPU.numba_implementation.elementary_reaction import N_CTX_FIELDS   as N_CTX_ELEM
from kineticsGPU.numba_implementation.three_body_reaction import N_CTX_TB_FIELDS as N_CTX_TB
from kineticsGPU.numba_implementation.falloff_reaction    import N_CTX_FO_FIELDS as N_CTX_FO
from kineticsGPU.numba_implementation.backward_euler_numba import backward_euler_step_kernel


def pack_all_arrays(elem_arrays, tb_arrays, fo_arrays, thermo_arrays):
    """
    Fusionne les 4 tuples d'arrays chimiques en un seul tuple de 52 éléments.

    Layout résultant (doit correspondre aux constantes _E, _TB, _FO, _TH
    dans backward_euler_numba.py) :
        [0..12]  : elem_arrays[0..12]   (13 arrays élémentaires)
        [13..26] : tb_arrays[0..13]     (14 arrays three-body)
        [27..48] : fo_arrays[0..21]     (22 arrays falloff)
        [49..51] : thermo_arrays[0..2]  (3 arrays thermo)

    Note : les fonctions pack_*_context_packaged utilisent 15 arrays pour
    élémentaires (indices 0..14 dans leur tuple d'origine), mais les arrays
    thermo sont extraits séparément. On ne garde que les 13 premiers (non-thermo).
    Vérifiez les indices avec les fichiers yaml_to_GPU_array.py de votre projet.
    """
    return (
        # Élémentaires : react_idx, react_sto, reactant_number, reactant_offset,
        #                prod_idx, prod_sto, prod_number, prod_offset,
        #                A, n, Ea, oneWay, delta_n
        *elem_arrays[:13],
        # Three-body : même layout + effmatrix en [13]
        *tb_arrays[:14],
        # Falloff : 22 arrays (indices 0..21)
        *fo_arrays[:22],
        # Thermo : coef_high, coef_low, Tmid
        *thermo_arrays[:3],
    )


def _grid(ncells, tpb=128):
    return (tpb,), ((ncells + tpb - 1) // tpb,)


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

        # Fusion en un seul tuple pour le kernel (évite le dépassement de 4096 bytes)
        self.all_arrays = pack_all_arrays(elem_arrays, tb_arrays,
                                          fo_arrays, thermo_arrays)

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

        # ── Contextes chimiques ────────────────────────────────────────────
        self.elem_ctx_buf = cuda.device_array(
            (nc, N_CTX_ELEM, nreac_elem), dtype=np.float64)
        self.tb_ctx_buf   = cuda.device_array(
            (nc, N_CTX_TB,   nreac_tb),   dtype=np.float64)
        self.fo_ctx_buf   = cuda.device_array(
            (nc, N_CTX_FO,   nreac_fo),   dtype=np.float64)

        # ── Buffers réacteur ───────────────────────────────────────────────
        self.rhs_buf     = cuda.device_array((nc, N),   dtype=np.float64)
        self.jvp_tmp_buf = cuda.device_array((nc, N),   dtype=np.float64)
        self.vC_buf      = cuda.device_array((nc, nsp), dtype=np.float64)
        self.F_buf       = cuda.device_array((nc, N),   dtype=np.float64)

        # ── Buffers GMRES ──────────────────────────────────────────────────
        self.gmres_V   = cuda.device_array((nc, (rst + 1) * N),      dtype=np.float64)
        self.gmres_H   = cuda.device_array((nc, rst * (rst + 1)),     dtype=np.float64)
        self.gmres_cs  = cuda.device_array((nc, rst),                 dtype=np.float64)
        self.gmres_sn  = cuda.device_array((nc, rst),                 dtype=np.float64)
        self.gmres_e1  = cuda.device_array((nc, rst + 1),             dtype=np.float64)
        self.gmres_y   = cuda.device_array((nc, rst),                 dtype=np.float64)
        self.gmres_tmp = cuda.device_array((nc, N),                   dtype=np.float64)
        self.gmres_sol = cuda.device_array((nc, N),                   dtype=np.float64)

        # Info mémoire
        mb = (nc * (N_CTX_ELEM * nreac_elem +
                    N_CTX_TB   * nreac_tb   +
                    N_CTX_FO   * nreac_fo   +
                    N * 6 + nsp +
                    (rst + 1) * N + rst * (rst + 1) +
                    rst * 3 + (rst + 1))) * 8 / 1e6
        print(f"[BackwardEulerNumba] Mémoire GPU : {mb:.1f} Mo "
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
            self.all_arrays,
            self.elem_ctx_buf,
            self.tb_ctx_buf,
            self.fo_ctx_buf,
            self.rhs_buf,
            self.jvp_tmp_buf,
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