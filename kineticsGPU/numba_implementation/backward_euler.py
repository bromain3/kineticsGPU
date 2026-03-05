"""
backward_euler_numba.py
=======================
Intégrateur Backward Euler implicite — GMRES + JVP analytique.

Philosophie :
  - 1 thread CUDA = 1 réacteur
  - Zéro mémoire locale ou partagée (tout en mémoire globale)
  - GMRES depuis gmres_numba.py (pas de duplication)

Correction "Formal parameter space overflowed" :
  Tous les arrays chimiques sont regroupés dans un SEUL tuple `all_arrays`
  passé au kernel. Des device functions wrapper extraient les bons indices
  pour appeler pack_*, rhs, jvp avec leurs signatures d'origine.

Layout de all_arrays (construit par pack_all_arrays() dans integrator_host.py) :
  Index   Contenu
  ──────────────────────────────────────────────────────────────
  ELEM (15 arrays, indices identiques à build_elementary_gpu_arrays) :
   0  react_idx      1  react_sto      2  reactant_number  3  reactant_offset
   4  prod_idx       5  prod_sto       6  prod_number       7  prod_offset
   8  A_arr          9  n_arr          10 Ea_arr
   11 oneWay         12 delta_n
   (pas d'effmatrix pour élémentaires)
  TB (14 arrays, décalage +13 = index 13..26) :
   13 react_idx  14 react_sto  15 reactant_number  16 reactant_offset
   17 prod_idx   18 prod_sto   19 prod_number       20 prod_offset
   21 A_arr      22 n_arr      23 Ea_arr
   24 oneWay     25 delta_n    26 effmatrix
  FO (22 arrays, décalage +27 = index 27..48) :
   27 react_idx  28 react_sto  29 reactant_number  30 reactant_offset
   31 prod_idx   32 prod_sto   33 prod_number       34 prod_offset
   35 A0         36 n0         37 Ea0
   38 Ainf       39 ninf       40 Eainf
   41 oneWay     42 delta_n    43 effmatrix
   44 falloff_type_arr
   45 troe_A     46 troe_T1    47 troe_T2    48 troe_T3
  THERMO (3 arrays, index 49..51) :
   49 coef_high  50 coef_low   51 Tmid

Total : 52 arrays dans all_arrays.
"""

from numba import cuda
import math

from kineticsGPU.numba_implementation.GMRES import (
    norm1d, norm_off, dot_off, axpy_off, scale_off, zero_off,
    givens_rotation, back_substitution,
)
from kineticsGPU.numba_implementation.elementary_reaction import (
    pack_elementary_context, elem_omega, elem_jvp,
    N_CTX_FIELDS as N_CTX_ELEM,
)
from kineticsGPU.numba_implementation.three_body_reaction import (
    pack_threebody_context, tb_omega, tb_jvp,
    N_CTX_TB_FIELDS as N_CTX_TB,
)
from kineticsGPU.numba_implementation.falloff_reaction import (
    pack_falloff_context, fo_omega, fo_jvp,
    N_CTX_FO_FIELDS as N_CTX_FO,
)
from kineticsGPU.numba_implementation.constant_volume_reactor import (
    constant_volume_reactor_rhs,
    constant_volume_reactor_jvp,
)

# Offsets dans all_arrays
_E  =  0   # elem  : 0..12
_TB = 13   # tb    : 13..26
_FO = 27   # fo    : 27..48
_TH = 49   # thermo: 49..51


# =============================================================================
#  Wrappers pack/omega/jvp qui extraient les sous-arrays depuis all_arrays
# =============================================================================

@cuda.jit(device=True, inline=False)
def _pack_elem(T, C, all_arrays, ctx):
    pack_elementary_context(
        T, C,
        all_arrays[_E+0],  all_arrays[_E+1],  all_arrays[_E+2],  all_arrays[_E+3],
        all_arrays[_E+4],  all_arrays[_E+5],  all_arrays[_E+6],  all_arrays[_E+7],
        all_arrays[_E+8],  all_arrays[_E+9],  all_arrays[_E+10],
        all_arrays[_E+11], all_arrays[_E+12],
        all_arrays[_TH+1], all_arrays[_TH+0], all_arrays[_TH+2],
        ctx)

@cuda.jit(device=True, inline=False)
def _pack_tb(T, C, all_arrays, ctx):
    pack_threebody_context(
        T, C,
        all_arrays[_TB+0],  all_arrays[_TB+1],  all_arrays[_TB+2],  all_arrays[_TB+3],
        all_arrays[_TB+4],  all_arrays[_TB+5],  all_arrays[_TB+6],  all_arrays[_TB+7],
        all_arrays[_TB+8],  all_arrays[_TB+9],  all_arrays[_TB+10],
        all_arrays[_TB+11], all_arrays[_TB+12],
        all_arrays[_TB+13],
        all_arrays[_TH+1], all_arrays[_TH+0], all_arrays[_TH+2],
        ctx)

@cuda.jit(device=True, inline=False)
def _pack_fo(T, C, all_arrays, ctx):
    pack_falloff_context(
        T, C,
        all_arrays[_FO+0],  all_arrays[_FO+1],  all_arrays[_FO+2],  all_arrays[_FO+3],
        all_arrays[_FO+4],  all_arrays[_FO+5],  all_arrays[_FO+6],  all_arrays[_FO+7],
        all_arrays[_FO+8],  all_arrays[_FO+9],  all_arrays[_FO+10],
        all_arrays[_FO+11], all_arrays[_FO+12], all_arrays[_FO+13],
        all_arrays[_FO+14], all_arrays[_FO+15],
        all_arrays[_FO+16],
        all_arrays[_FO+17],
        all_arrays[_FO+18], all_arrays[_FO+19], all_arrays[_FO+20], all_arrays[_FO+21],
        all_arrays[_TH+1], all_arrays[_TH+0], all_arrays[_TH+2],
        ctx)

@cuda.jit(device=True, inline=False)
def _omega_elem(all_arrays, ctx, omega):
    elem_omega(
        all_arrays[_E+0], all_arrays[_E+1], all_arrays[_E+2], all_arrays[_E+3],
        all_arrays[_E+4], all_arrays[_E+5], all_arrays[_E+6], all_arrays[_E+7],
        ctx, omega)

@cuda.jit(device=True, inline=False)
def _omega_tb(all_arrays, ctx, omega):
    tb_omega(
        all_arrays[_TB+0], all_arrays[_TB+1], all_arrays[_TB+2], all_arrays[_TB+3],
        all_arrays[_TB+4], all_arrays[_TB+5], all_arrays[_TB+6], all_arrays[_TB+7],
        ctx, omega)

@cuda.jit(device=True, inline=False)
def _omega_fo(all_arrays, ctx, omega):
    fo_omega(
        all_arrays[_FO+0], all_arrays[_FO+1], all_arrays[_FO+2], all_arrays[_FO+3],
        all_arrays[_FO+4], all_arrays[_FO+5], all_arrays[_FO+6], all_arrays[_FO+7],
        ctx, omega)

@cuda.jit(device=True, inline=False)
def _jvp_elem(C, all_arrays, ctx, vT, vC, jvp):
    elem_jvp(
        C,
        all_arrays[_E+0], all_arrays[_E+1], all_arrays[_E+2], all_arrays[_E+3],
        all_arrays[_E+4], all_arrays[_E+5], all_arrays[_E+6], all_arrays[_E+7],
        all_arrays[_E+11],
        ctx, vT, vC, jvp)

@cuda.jit(device=True, inline=False)
def _jvp_tb(C, vT, vC, all_arrays, ctx, jvp):
    tb_jvp(
        C,
        all_arrays[_TB+0], all_arrays[_TB+1], all_arrays[_TB+2], all_arrays[_TB+3],
        all_arrays[_TB+4], all_arrays[_TB+5], all_arrays[_TB+6], all_arrays[_TB+7],
        all_arrays[_TB+11], all_arrays[_TB+13],
        ctx, vT, vC, jvp)

@cuda.jit(device=True, inline=False)
def _jvp_fo(C, vT, vC, all_arrays, ctx, jvp):
    fo_jvp(
        C,
        all_arrays[_FO+0], all_arrays[_FO+1], all_arrays[_FO+2], all_arrays[_FO+3],
        all_arrays[_FO+4], all_arrays[_FO+5], all_arrays[_FO+6], all_arrays[_FO+7],
        all_arrays[_FO+14], all_arrays[_FO+16],
        ctx, vT, vC, jvp)


# =============================================================================
#  RHS réacteur adiabatique volume constant
#  (inlining de reactor.py pour éviter de repasser les tuples séparés)
# =============================================================================

@cuda.jit(device=True, inline=False)
def _reactor_rhs(Tcell, Ccell, elem_c, tb_c, fo_c, all_arrays, dXdt):
    """
    Calcule [dCdt_0..dCdt_{nsp-1}, dTdt] dans dXdt.
    Utilise les wrappers _omega_* et les arrays thermo depuis all_arrays.
    """
    import kineticsGPU.numba_implementation.thermo as _thermo
    import kineticsGPU.numba_implementation.constants as _constants

    nsp = Ccell.shape[0]

    for i in range(nsp): dXdt[i] = 0.0

    _omega_elem(all_arrays, elem_c, dXdt)
    _omega_tb  (all_arrays, tb_c,   dXdt)
    _omega_fo  (all_arrays, fo_c,   dXdt)

    coef_high = all_arrays[_TH + 0]
    coef_low  = all_arrays[_TH + 1]
    Tmid      = all_arrays[_TH + 2]

    n_tot = 0.0
    for i in range(nsp): n_tot += Ccell[i]

    cp_mol_mix = 0.0
    for i in range(nsp):
        Xi = Ccell[i] / n_tot
        cp_mol_mix += Xi * _thermo.cp_mol_sp(coef_high, coef_low, Tmid, i, Tcell)

    cv_mol_mix = cp_mol_mix - _constants.R

    dUdt = 0.0
    for i in range(nsp):
        u_i   = _thermo.u_mol_sp(coef_high, coef_low, Tmid, i, Tcell)
        dUdt += u_i * dXdt[i]

    dXdt[nsp] = -dUdt / (n_tot * cv_mol_mix)


@cuda.jit(device=True, inline=False)
def _reactor_jvp(Tcell, Ccell, vT, vC,
                 elem_c, tb_c, fo_c, all_arrays,
                 dXdt, jvp_out):
    """
    JVP du RHS réacteur. Layout jvp_out : [dCdt', dTdt'].
    """
    import kineticsGPU.numba_implementation.thermo as _thermo
    import kineticsGPU.numba_implementation.constants as _constants

    nsp = Ccell.shape[0]
    coef_high = all_arrays[_TH + 0]
    coef_low  = all_arrays[_TH + 1]
    Tmid      = all_arrays[_TH + 2]

    for i in range(nsp): jvp_out[i] = 0.0

    _jvp_elem(Ccell, all_arrays, elem_c, vT, vC, jvp_out)
    _jvp_tb  (Ccell, vT, vC, all_arrays, tb_c, jvp_out)
    _jvp_fo  (Ccell, vT, vC, all_arrays, fo_c, jvp_out)

    n_tot = 0.0
    for i in range(nsp): n_tot += Ccell[i]

    dn_tot = 0.0
    for i in range(nsp): dn_tot += vC[i]

    cv_mix = 0.0; dcv_mix_X = 0.0; dcv_mix_T = 0.0
    U_prime = 0.0; dU_prime_C = 0.0; dU_prime_T = 0.0

    for i in range(nsp):
        Xi   = Ccell[i] / n_tot
        dXi  = vC[i] / n_tot - Ccell[i] * dn_tot / (n_tot * n_tot)
        cv_i  = _thermo.cv_mol_sp   (coef_high, coef_low, Tmid, i, Tcell)
        dcv_i = _thermo.dcv_mol_sp_dT(coef_high, coef_low, Tmid, i, Tcell)
        u_i   = _thermo.u_mol_sp    (coef_high, coef_low, Tmid, i, Tcell)

        cv_mix     += Xi  * cv_i
        dcv_mix_X  += dXi * cv_i
        dcv_mix_T  += Xi  * dcv_i

        U_prime    += u_i * dXdt[i]
        dU_prime_C += u_i * jvp_out[i]
        dU_prime_T += cv_i * dXdt[i]

    dcv_mix  = dcv_mix_X + dcv_mix_T * vT
    dU_prime = dU_prime_C + dU_prime_T * vT

    denom1 = n_tot * cv_mix
    denom2 = n_tot * n_tot * cv_mix
    denom3 = n_tot * cv_mix * cv_mix

    jvp_out[nsp] = (- dU_prime / denom1
                    + U_prime * dn_tot  / denom2
                    + U_prime * dcv_mix / denom3)


# =============================================================================
#  Matvec : w = v + h * J @ v
# =============================================================================

@cuda.jit(device=True, inline=False)
def _matvec(V_flat, v_off, h,
            Tcell, Ccell, N,
            elem_c, tb_c, fo_c,
            all_arrays,
            rhs_c, jvp_tmp, vC_buf,
            out):
    nsp = N - 1
    vT  = V_flat[v_off]

    for i in range(nsp):
        vC_buf[i] = V_flat[v_off + 1 + i]

    for i in range(N):
        jvp_tmp[i] = 0.0

    _reactor_jvp(Tcell, Ccell, vT, vC_buf,
                 elem_c, tb_c, fo_c, all_arrays,
                 rhs_c, jvp_tmp)

    # Réordonne jvp [dC.., dT] → out layout TC [T, C..]
    out[0] = vT + h * jvp_tmp[nsp]
    for i in range(nsp):
        out[i + 1] = vC_buf[i] + h * jvp_tmp[i]


# =============================================================================
#  GMRES redémarré — utilise gmres_numba.py pour les utilitaires
# =============================================================================

@cuda.jit(device=True, inline=False)
def _gmres(b, x, h,
           Tcell, Ccell, N,
           elem_c, tb_c, fo_c,
           all_arrays,
           rhs_c, jvp_tmp, vC_buf,
           V, H, cs, sn, e1, yv, tmp,
           restart, tol, maxiter):

    b_norm = norm1d(b, N)
    if b_norm < 1e-300:
        for i in range(N): x[i] = 0.0
        return

    abs_tol = tol * b_norm

    for _outer in range(maxiter):

        for i in range(N): V[i] = x[i]
        zero_off(tmp, 0, N)
        _matvec(V, 0, h, Tcell, Ccell, N,
                elem_c, tb_c, fo_c, all_arrays,
                rhs_c, jvp_tmp, vC_buf, tmp)
        for i in range(N): V[i] = b[i] - tmp[i]

        beta = norm_off(V, 0, N)
        if beta < abs_tol: return
        scale_off(1.0 / beta, V, 0, N)

        for i in range(restart + 1): e1[i] = 0.0
        e1[0] = beta

        j_conv = restart
        for j in range(restart):
            vj = j * N
            zero_off(tmp, 0, N)
            _matvec(V, vj, h, Tcell, Ccell, N,
                    elem_c, tb_c, fo_c, all_arrays,
                    rhs_c, jvp_tmp, vC_buf, tmp)

            for i in range(j + 1):
                vi   = i * N
                h_ij = dot_off(tmp, 0, V, vi, N)
                H[j * (restart + 1) + i] = h_ij
                axpy_off(-h_ij, V, vi, tmp, 0, N)

            h_jp1j = norm_off(tmp, 0, N)
            H[j * (restart + 1) + (j + 1)] = h_jp1j

            vj1 = (j + 1) * N
            if h_jp1j > 1e-300:
                inv_h = 1.0 / h_jp1j
                for k in range(N): V[vj1 + k] = tmp[k] * inv_h
            else:
                zero_off(V, vj1, N)

            for i in range(j):
                ci = cs[i]; si = sn[i]
                h0 = H[j * (restart + 1) + i]
                h1 = H[j * (restart + 1) + (i + 1)]
                H[j * (restart + 1) + i]       =  ci * h0 + si * h1
                H[j * (restart + 1) + (i + 1)] = -si * h0 + ci * h1

            h0 = H[j * (restart + 1) + j]
            h1 = H[j * (restart + 1) + (j + 1)]
            cs_j, sn_j = givens_rotation(h0, h1)
            cs[j] = cs_j; sn[j] = sn_j
            H[j * (restart + 1) + j]       =  cs_j * h0 + sn_j * h1
            H[j * (restart + 1) + (j + 1)] = -sn_j * h0 + cs_j * h1

            e1[j + 1] = -sn_j * e1[j]
            e1[j]     =  cs_j * e1[j]

            if math.fabs(e1[j + 1]) < abs_tol:
                j_conv = j + 1
                break
        else:
            j_conv = restart

        for i in range(j_conv): yv[i] = 0.0
        back_substitution(H, e1, yv, restart, j_conv)

        for i in range(j_conv):
            vi = i * N; yi = yv[i]
            for k in range(N): x[k] += yi * V[vi + k]

        for i in range(N): V[i] = x[i]
        zero_off(tmp, 0, N)
        _matvec(V, 0, h, Tcell, Ccell, N,
                elem_c, tb_c, fo_c, all_arrays,
                rhs_c, jvp_tmp, vC_buf, tmp)
        res2 = 0.0
        for k in range(N):
            d = b[k] - tmp[k]; res2 += d * d
        if math.sqrt(res2) < abs_tol: return


# =============================================================================
#  Kernel principal — UN SEUL tuple all_arrays
# =============================================================================

@cuda.jit
def backward_euler_step_kernel(
        TC_in,          # (ncells, N)
        TC_out,         # (ncells, N)
        h_step,
        all_arrays,     # tuple unique de tous les arrays chimiques (52 éléments)
        elem_ctx_buf,   # (ncells, N_CTX_ELEM, nreac_elem)
        tb_ctx_buf,     # (ncells, N_CTX_TB,   nreac_tb)
        fo_ctx_buf,     # (ncells, N_CTX_FO,   nreac_fo)
        rhs_buf,        # (ncells, N)
        jvp_tmp_buf,    # (ncells, N)
        vC_buf_gl,      # (ncells, nsp)
        F_buf,          # (ncells, N)
        gmres_V,        # (ncells, (restart+1)*N)
        gmres_H,        # (ncells, restart*(restart+1))
        gmres_cs,       # (ncells, restart)
        gmres_sn,       # (ncells, restart)
        gmres_e1,       # (ncells, restart+1)
        gmres_y,        # (ncells, restart)
        gmres_tmp,      # (ncells, N)
        gmres_sol,      # (ncells, N)
        restart,
        gmres_tol,
        gmres_maxiter,
        newton_tol,
        newton_maxiter,
):
    cell = cuda.grid(1)
    if cell >= TC_in.shape[0]:
        return

    N   = TC_in.shape[1]
    nsp = N - 1

    TC_n    = TC_in[cell]
    TC_k    = TC_out[cell]
    rhs_c   = rhs_buf[cell]
    jvp_tmp = jvp_tmp_buf[cell]
    vC_buf  = vC_buf_gl[cell]
    F_c     = F_buf[cell]
    elem_c  = elem_ctx_buf[cell]
    tb_c    = tb_ctx_buf[cell]
    fo_c    = fo_ctx_buf[cell]
    V_c     = gmres_V[cell]
    H_c     = gmres_H[cell]
    cs_c    = gmres_cs[cell]
    sn_c    = gmres_sn[cell]
    e1_c    = gmres_e1[cell]
    y_c     = gmres_y[cell]
    tmp_c   = gmres_tmp[cell]
    sol_c   = gmres_sol[cell]

    Tn = TC_n[0]
    Cn = TC_n[1:]

    # ── Prédicteur explicite ───────────────────────────────────────────────
    _pack_elem(Tn, Cn, all_arrays, elem_c)
    _pack_tb  (Tn, Cn, all_arrays, tb_c)
    _pack_fo  (Tn, Cn, all_arrays, fo_c)

    for i in range(N): rhs_c[i] = 0.0
    _reactor_rhs(Tn, Cn, elem_c, tb_c, fo_c, all_arrays, rhs_c)

    TC_k[0] = TC_n[0] + h_step * rhs_c[nsp]
    for i in range(nsp):
        TC_k[i + 1] = TC_n[i + 1] + h_step * rhs_c[i]

    # ── Boucle Newton ─────────────────────────────────────────────────────
    for _nit in range(newton_maxiter):
        Tk = TC_k[0]
        Ck = TC_k[1:]

        _pack_elem(Tk, Ck, all_arrays, elem_c)
        _pack_tb  (Tk, Ck, all_arrays, tb_c)
        _pack_fo  (Tk, Ck, all_arrays, fo_c)

        for i in range(N): rhs_c[i] = 0.0
        _reactor_rhs(Tk, Ck, elem_c, tb_c, fo_c, all_arrays, rhs_c)

        F_norm2 = 0.0
        F_c[0] = TC_k[0] - TC_n[0] - h_step * rhs_c[nsp]
        F_norm2 += F_c[0] * F_c[0]
        for i in range(nsp):
            F_c[i + 1] = TC_k[i + 1] - TC_n[i + 1] - h_step * rhs_c[i]
            F_norm2 += F_c[i + 1] * F_c[i + 1]

        if math.sqrt(F_norm2) < newton_tol:
            break

        for i in range(N):
            F_c[i]   = -F_c[i]
            sol_c[i] = 0.0

        _gmres(F_c, sol_c, h_step,
               Tk, Ck, N,
               elem_c, tb_c, fo_c,
               all_arrays,
               rhs_c, jvp_tmp, vC_buf,
               V_c, H_c, cs_c, sn_c, e1_c, y_c, tmp_c,
               restart, gmres_tol, gmres_maxiter)

        for i in range(N):
            TC_k[i] += sol_c[i]