"""
kineticsGPU/numba_implementation/integrator.py

Backward Euler + GMRES (rotations de Givens), per-thread GPU.

Architecture mémoire : globale pour les gros buffers (Krylov, Hessenberg),
registres (local.array) pour les vecteurs de travail courants.

Convention d'indexation stricte :
  - V_buf[cell, j*n + i]   : accès scalaire uniquement, jamais de slice
  - Tous les vecteurs courants (v_in, w_out) sont des local.array (registres)
    copiés depuis/vers V_buf explicitement
  - tmp_vT[cell] = scalaire  => toujours un float64 scalaire
"""

import math
import numpy as np
from numba import cuda, float64 as numba_float64
import kineticsGPU.numba_implementation.reactor as reactor_mod

# ─────────────────────────────────────────────────────────────────────────────
#  Paramètres
# ─────────────────────────────────────────────────────────────────────────────
MAX_RESTART       = 20
MAX_NSP           = 54        # nsp + 1  (T + espèces)
NEWTON_MAXITER    = 10
NEWTON_TOL        = 1e-6
GMRES_TOL         = 1e-6
THREADS_PER_BLOCK = 32

_N_STATE   = MAX_NSP
_V_SIZE    = (MAX_RESTART + 1) * _N_STATE   # base de Krylov aplatie
_H_SIZE    = (MAX_RESTART + 1) * MAX_RESTART
_CS_SIZE   = MAX_RESTART
_SN_SIZE   = MAX_RESTART
_BETA_SIZE = MAX_RESTART + 1


# ─────────────────────────────────────────────────────────────────────────────
#  Matvec : jvp_out[:n] = (I - h*J) * v_in[:n]
#
#  v_in  : local.array(MAX_NSP) — vecteur d'entrée (registres)
#  w_out : local.array(MAX_NSP) — vecteur de sortie (registres)
#
#  tmp_T/tmp_C contiennent déjà xk courant (écrits avant l'appel GMRES).
#  On écrit seulement vT/vC à partir de v_in.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _matvec(
        v_in,       # local.array(MAX_NSP) — vecteur direction
        w_out,      # local.array(MAX_NSP) — résultat w = (I-h*J)*v
        h, n, cell_idx,
        tmp_T, tmp_C, tmp_vT, tmp_vC,
        tmp_omega, tmp_jvp_omega,
        coef_low, coef_high, Tmid, nsp,
        e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
        fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff, fo_ftype,
        fo_tA, fo_tT1, fo_tT2, fo_tT3,
        rhs_fwd,    # local.array(MAX_NSP) — réécrit par per_thread_jvp_rhs
        jvp_loc):   # local.array(MAX_NSP) — sortie : J*v

    # v_in[0] est un scalaire float64 ✓
    tmp_vT[cell_idx] = v_in[0]
    for k in range(nsp):
        tmp_vC[cell_idx, k] = v_in[k + 1]

    reactor_mod.per_thread_jvp_rhs(
        tmp_T, tmp_C, cell_idx,
        tmp_vT, tmp_vC,
        coef_low, coef_high, Tmid, nsp,
        e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
        fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff, fo_ftype,
        fo_tA, fo_tT1, fo_tT2, fo_tT3,
        tmp_omega, tmp_jvp_omega,
        rhs_fwd,
        jvp_loc)

    # w = v - h * J*v
    for i in range(n):
        w_out[i] = v_in[i] - h * jvp_loc[i]


# ─────────────────────────────────────────────────────────────────────────────
#  GMRES Givens
#  Résout (I - h*J) * dx = b_loc
#
#  V_buf[cell, j*n + i] : stockage scalaire de la base de Krylov
#  Toutes les opérations vecteur utilisent des local.array intermédiaires
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=False)
def _gmres(
        b_loc,      # local.array(MAX_NSP) — RHS = -F
        dx_loc,     # local.array(MAX_NSP) — sortie dx
        h, n, cell_idx,
        V_buf,      # (ncells, _V_SIZE)    — base de Krylov
        H_buf,      # (ncells, _H_SIZE)    — Hessenberg
        cs_buf,     # (ncells, _CS_SIZE)
        sn_buf,     # (ncells, _SN_SIZE)
        bt_buf,     # (ncells, _BETA_SIZE)
        tmp_T, tmp_C, tmp_vT, tmp_vC,
        tmp_omega, tmp_jvp_omega,
        coef_low, coef_high, Tmid, nsp,
        e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
        fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff, fo_ftype,
        fo_tA, fo_tT1, fo_tT2, fo_tT3,
        rhs_fwd,    # local.array(MAX_NSP) — réutilisé pour matvec
        jvp_loc,    # local.array(MAX_NSP) — buffer jvp
        v_cur,      # local.array(MAX_NSP) — vecteur courant V[j]
        w_cur):     # local.array(MAX_NSP) — résultat matvec w = A*V[j]

    for i in range(n):
        dx_loc[i] = 0.0

    # Norme de b
    r_norm = 0.0
    for i in range(n):
        r_norm += b_loc[i] * b_loc[i]
    r_norm = math.sqrt(r_norm)
    if r_norm < GMRES_TOL:
        return

    bt_buf[cell_idx, 0] = r_norm

    # V[0] = b / ||b||  — écriture scalaire dans V_buf
    inv_r = 1.0 / r_norm
    for i in range(n):
        V_buf[cell_idx, i] = b_loc[i] * inv_r

    j = 0
    while j < MAX_RESTART:

        # Charger V[j] dans v_cur (local.array)
        j_off = j * n
        for i in range(n):
            v_cur[i] = V_buf[cell_idx, j_off + i]

        # w = A * v_cur
        _matvec(
            v_cur, w_cur,
            h, n, cell_idx,
            tmp_T, tmp_C, tmp_vT, tmp_vC,
            tmp_omega, tmp_jvp_omega,
            coef_low, coef_high, Tmid, nsp,
            e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
            e_A, e_n, e_Ea, e_ow, e_dn,
            tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
            tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
            fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
            fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
            fo_ow, fo_dn, fo_eff, fo_ftype,
            fo_tA, fo_tT1, fo_tT2, fo_tT3,
            rhs_fwd, jvp_loc)

        # Gram-Schmidt modifié sur w_cur
        for i in range(j + 1):
            i_off = i * n
            h_ij = 0.0
            for k in range(n):
                h_ij += w_cur[k] * V_buf[cell_idx, i_off + k]
            H_buf[cell_idx, i * MAX_RESTART + j] = h_ij
            for k in range(n):
                w_cur[k] -= h_ij * V_buf[cell_idx, i_off + k]

        # ||w||
        w_norm = 0.0
        for k in range(n):
            w_norm += w_cur[k] * w_cur[k]
        w_norm = math.sqrt(w_norm)
        H_buf[cell_idx, (j + 1) * MAX_RESTART + j] = w_norm

        # V[j+1] = w / ||w||
        if w_norm > 1e-300:
            inv_w   = 1.0 / w_norm
            jp1_off = (j + 1) * n
            for k in range(n):
                V_buf[cell_idx, jp1_off + k] = w_cur[k] * inv_w

        # Rotations de Givens existantes
        for i in range(j):
            h_ij   = H_buf[cell_idx, i       * MAX_RESTART + j]
            h_ip1j = H_buf[cell_idx, (i + 1) * MAX_RESTART + j]
            t0 =  cs_buf[cell_idx, i] * h_ij + sn_buf[cell_idx, i] * h_ip1j
            t1 = -sn_buf[cell_idx, i] * h_ij + cs_buf[cell_idx, i] * h_ip1j
            H_buf[cell_idx, i       * MAX_RESTART + j] = t0
            H_buf[cell_idx, (i + 1) * MAX_RESTART + j] = t1

        # Nouvelle rotation
        h_jj   = H_buf[cell_idx, j       * MAX_RESTART + j]
        h_jp1j = H_buf[cell_idx, (j + 1) * MAX_RESTART + j]
        denom  = math.sqrt(h_jj * h_jj + h_jp1j * h_jp1j)
        if denom < 1e-300:
            denom = 1e-300
        cj =  h_jj   / denom
        sj =  h_jp1j / denom
        cs_buf[cell_idx, j] = cj
        sn_buf[cell_idx, j] = sj

        H_buf[cell_idx, j       * MAX_RESTART + j] = denom
        H_buf[cell_idx, (j + 1) * MAX_RESTART + j] = 0.0

        bt_buf[cell_idx, j + 1] = -sj * bt_buf[cell_idx, j]
        bt_buf[cell_idx, j]     =  cj * bt_buf[cell_idx, j]

        j += 1
        if math.fabs(bt_buf[cell_idx, j]) < GMRES_TOL:
            break

    # Back-substitution
    m = j
    for i in range(m - 1, -1, -1):
        s = bt_buf[cell_idx, i]
        for k in range(i + 1, m):
            s -= H_buf[cell_idx, i * MAX_RESTART + k] * bt_buf[cell_idx, k]
        h_ii = H_buf[cell_idx, i * MAX_RESTART + i]
        if math.fabs(h_ii) < 1e-300:
            h_ii = 1e-300
        bt_buf[cell_idx, i] = s / h_ii

    # dx = Σ beta[i] * V[i]
    for i in range(m):
        i_off = i * n
        bi    = bt_buf[cell_idx, i]
        for k in range(n):
            dx_loc[k] += bi * V_buf[cell_idx, i_off + k]


# ─────────────────────────────────────────────────────────────────────────────
#  Kernel principal
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit
def kernel_integrate(
        TC_in, t_start, dt, n_steps,
        coef_low, coef_high, Tmid, nsp,
        e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
        fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff, fo_ftype,
        fo_tA, fo_tT1, fo_tT2, fo_tT3,
        tmp_omega, tmp_jvp_omega,
        tmp_T, tmp_C, tmp_vT, tmp_vC,
        V_buf, H_buf, cs_buf, sn_buf, bt_buf,
        TC_out):

    cell_idx = cuda.grid(1)
    if cell_idx >= TC_in.shape[0]:
        return

    n = nsp + 1

    # ── Buffers registres ─────────────────────────────────────────────────
    xk      = cuda.local.array(MAX_NSP, dtype=numba_float64)
    x0      = cuda.local.array(MAX_NSP, dtype=numba_float64)
    F_loc   = cuda.local.array(MAX_NSP, dtype=numba_float64)
    dx_loc  = cuda.local.array(MAX_NSP, dtype=numba_float64)
    rhs_loc = cuda.local.array(MAX_NSP, dtype=numba_float64)
    jvp_loc = cuda.local.array(MAX_NSP, dtype=numba_float64)
    v_cur   = cuda.local.array(MAX_NSP, dtype=numba_float64)  # V[j] courant
    w_cur   = cuda.local.array(MAX_NSP, dtype=numba_float64)  # A*V[j]

    # Lecture état initial
    for i in range(n):
        xk[i] = TC_in[cell_idx, i]

    # ── Boucle temporelle ─────────────────────────────────────────────────
    for _step in range(n_steps):

        for i in range(n):
            x0[i] = xk[i]

        # Écrire xk dans buffers cinétiques globaux
        tmp_T[cell_idx] = xk[0]
        for k in range(nsp):
            tmp_C[cell_idx, k] = xk[k + 1]

        # Prédicteur explicite : xk = x0 + dt * f(x0)
        reactor_mod.per_thread_rhs(
            tmp_T, tmp_C, cell_idx,
            coef_low, coef_high, Tmid, nsp,
            e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
            e_A, e_n, e_Ea, e_ow, e_dn,
            tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
            tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
            fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
            fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
            fo_ow, fo_dn, fo_eff, fo_ftype,
            fo_tA, fo_tT1, fo_tT2, fo_tT3,
            tmp_omega, rhs_loc)

        for i in range(n):
            xk[i] = x0[i] + dt * rhs_loc[i]

        # ── Newton ────────────────────────────────────────────────────────
        for _nit in range(NEWTON_MAXITER):

            tmp_T[cell_idx] = xk[0]
            for k in range(nsp):
                tmp_C[cell_idx, k] = xk[k + 1]

            reactor_mod.per_thread_rhs(
                tmp_T, tmp_C, cell_idx,
                coef_low, coef_high, Tmid, nsp,
                e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
                e_A, e_n, e_Ea, e_ow, e_dn,
                tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
                tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
                fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
                fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
                fo_ow, fo_dn, fo_eff, fo_ftype,
                fo_tA, fo_tT1, fo_tT2, fo_tT3,
                tmp_omega, rhs_loc)

            # Résidu F = xk - x0 - dt*f(xk)
            res_sq = 0.0
            for i in range(n):
                F_loc[i] = xk[i] - x0[i] - dt * rhs_loc[i]
                res_sq  += F_loc[i] * F_loc[i]

            if math.sqrt(res_sq) < NEWTON_TOL:
                break

            # Init H et beta à zéro
            for i in range(_H_SIZE):
                H_buf[cell_idx, i] = 0.0
            for i in range(_BETA_SIZE):
                bt_buf[cell_idx, i] = 0.0

            # b = -F
            for i in range(n):
                rhs_loc[i] = -F_loc[i]

            # Résoudre (I - dt*J)*dx = -F
            _gmres(
                rhs_loc, dx_loc,
                dt, n, cell_idx,
                V_buf, H_buf, cs_buf, sn_buf, bt_buf,
                tmp_T, tmp_C, tmp_vT, tmp_vC,
                tmp_omega, tmp_jvp_omega,
                coef_low, coef_high, Tmid, nsp,
                e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
                e_A, e_n, e_Ea, e_ow, e_dn,
                tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
                tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
                fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
                fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
                fo_ow, fo_dn, fo_eff, fo_ftype,
                fo_tA, fo_tT1, fo_tT2, fo_tT3,
                rhs_loc, jvp_loc,
                v_cur, w_cur)

            for i in range(n):
                xk[i] += dx_loc[i]

    # Écriture résultat
    for i in range(n):
        TC_out[cell_idx, i] = xk[i]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper de lancement
# ─────────────────────────────────────────────────────────────────────────────

def launch_integrate(
        TC_in_d, t_start, dt, n_steps,
        coef_low, coef_high, Tmid, nsp,
        elem_arrays, tb_arrays, fo_arrays,
        tmp_omega, tmp_jvp_omega,
        tmp_T, tmp_C, tmp_vT, tmp_vC):

    ncells   = TC_in_d.shape[0]
    TC_out_d = cuda.device_array_like(TC_in_d)

    # Buffers GMRES 2D — accès scalaire [cell, i] uniquement
    V_buf  = cuda.device_array((ncells, _V_SIZE),    dtype=np.float64)
    H_buf  = cuda.device_array((ncells, _H_SIZE),    dtype=np.float64)
    cs_buf = cuda.device_array((ncells, _CS_SIZE),   dtype=np.float64)
    sn_buf = cuda.device_array((ncells, _SN_SIZE),   dtype=np.float64)
    bt_buf = cuda.device_array((ncells, _BETA_SIZE), dtype=np.float64)

    blocks = (ncells + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    (e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
     e_A, e_n, e_Ea, e_ow, e_dn) = elem_arrays

    (tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
     tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff) = tb_arrays

    (fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
     fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
     fo_ow, fo_dn, fo_eff, fo_ftype,
     fo_tA, fo_tT1, fo_tT2, fo_tT3) = fo_arrays

    kernel_integrate[blocks, THREADS_PER_BLOCK](
        TC_in_d, t_start, dt, n_steps,
        coef_low, coef_high, Tmid, nsp,
        e_ri, e_rs, e_rn, e_ro, e_pi, e_ps, e_pn, e_po,
        e_A, e_n, e_Ea, e_ow, e_dn,
        tb_ri, tb_rs, tb_rn, tb_ro, tb_pi, tb_ps, tb_pn, tb_po,
        tb_A, tb_n, tb_Ea, tb_ow, tb_dn, tb_eff,
        fo_ri, fo_rs, fo_rn, fo_ro, fo_pi, fo_ps, fo_pn, fo_po,
        fo_A0, fo_n0, fo_Ea0, fo_Ainf, fo_ninf, fo_Eainf,
        fo_ow, fo_dn, fo_eff, fo_ftype,
        fo_tA, fo_tT1, fo_tT2, fo_tT3,
        tmp_omega, tmp_jvp_omega,
        tmp_T, tmp_C, tmp_vT, tmp_vC,
        V_buf, H_buf, cs_buf, sn_buf, bt_buf,
        TC_out_d)

    return TC_out_d


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic mémoire (utilisé par comparison_reactor.py)
# ─────────────────────────────────────────────────────────────────────────────

def gmres_buffer_memory_mb(ncells):
    total = (ncells * _V_SIZE
             + ncells * _H_SIZE
             + ncells * _CS_SIZE
             + ncells * _SN_SIZE
             + ncells * _BETA_SIZE)
    return total * 8 / 1e6


def print_memory_layout(ncells, nsp):
    print(f"\n  ── Buffers GMRES globaux ({ncells} réacteur(s)) ──")
    print(f"  V  (Krylov)    : {ncells} × {_V_SIZE:5d} f64 = {ncells*_V_SIZE*8/1024:.1f} KB")
    print(f"  H  (Hessenberg): {ncells} × {_H_SIZE:5d} f64 = {ncells*_H_SIZE*8/1024:.1f} KB")
    print(f"  cs, sn, beta   : {ncells} × {_CS_SIZE+_SN_SIZE+_BETA_SIZE:5d} f64")
    print(f"  TOTAL GMRES    : {gmres_buffer_memory_mb(ncells):.3f} MB")
    print(f"  Kin. buffers   : {ncells*(4*nsp+2)*8/1024:.1f} KB")
    print(f"  THREADS/BLOCK  : {THREADS_PER_BLOCK}  (sans contrainte shared ✓)")
    print(f"  v_cur, w_cur   : local.array (registres, pas de buffer global)")