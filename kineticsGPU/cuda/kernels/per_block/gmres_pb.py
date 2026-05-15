"""
kernels/per_block/gmres_pb.py — GMRES redémarré matriciel-libre (per-block).

Architecture : 1 bloc = 1 réacteur. Les threads coopèrent.

Fonction exportée :
    _gmres_pb(tid, n_threads, T, C, h, omega,
              sh_rhs, sh_dx, sh_w, sh_g, sh_y, sh_cs, sh_sn, sh_H,
              sh_scratch, sh_Jvo, sh_vC, gm_V, n_state, R, tol, maxiter,
              <ctx_slices>, <mech_args>)

Budget mémoire partagée (R=20, n_state=54) :
    Vecteurs partagés : ~8 × 54 × 8B ≈ 3.5 KB
    H : 21×20×8B = 3.36 KB
    g,y,cs,sn : ~4×21×8B = 0.67 KB
    Total ≈ 7.5 KB (tient dans 12 KB/bloc @ 8 blocs/SM)

V (base de Krylov) reste en mémoire globale (21×54×8B ≈ 9 KB / réacteur).
"""

import math
from numba import cuda, float64

from .reactor_odes_pb import _adiabatic_jvp_pb
from .jvp_pb import _eval_jvp_pb
from .reductions import _block_norm2, _block_dot, _block_norm2_inplace_sub


@cuda.jit(device=True)
def _gmres_pb(
    tid, n_threads,
    T, C, h, omega,
    sh_rhs, sh_dx,
    sh_w,
    sh_g, sh_y, sh_cs, sh_sn,
    sh_H,
    sh_scratch,
    sh_Jvo,
    sh_vC,
    gm_V,       # mémoire globale (R+1)*n_state pour ce réacteur
    n_state, R, tol, maxiter,
    # ── Contexte cinétique ────────────────────────────────────────────
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    # ── Données mécanisme ─────────────────────────────────────────────
    n_sp, thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    """
    Résout (I - h*J) dx = sh_rhs par GMRES(R) parallèle.
    Les tableaux sh_* sont en mémoire partagée.
    gm_V est en mémoire globale (une ligne par réacteur).
    """
    # Initialisation de sh_dx = 0
    i = tid
    while i < n_state:
        sh_dx[i] = float64(0.0)
        i += n_threads
    cuda.syncthreads()

    # beta = ||rhs||
    _block_norm2(sh_rhs, sh_scratch, tid, n_state, n_threads)
    cuda.syncthreads()
    if tid == 0:
        sh_g[0] = math.sqrt(sh_scratch[0])
    cuda.syncthreads()
    beta = sh_g[0]
    if beta < tol:
        return

    for outer in range(maxiter):
        # Initialisation H, cs, sn, g
        i = tid
        while i < (R + 1) * R:
            sh_H[i] = float64(0.0)
            i += n_threads
        i = tid
        while i < R:
            sh_cs[i] = float64(0.0)
            sh_sn[i] = float64(0.0)
            i += n_threads
        i = tid
        while i < R + 1:
            sh_g[i] = float64(0.0)
            i += n_threads
        cuda.syncthreads()

        if outer == 0:
            inv = float64(1.0) / beta
            i   = tid
            while i < n_state:
                gm_V[i] = sh_rhs[i] * inv
                i += n_threads
            cuda.syncthreads()
        else:
            # Recalcul du résidu vrai pour le restart
            vT_ = sh_dx[0]
            i   = tid
            while i < n_sp:
                sh_vC[i] = sh_dx[i + 1]
                i += n_threads
            cuda.syncthreads()
            _eval_jvp_pb(
                tid, n_threads,
                T, C, vT_, sh_vC, sh_Jvo,
                kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
                kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
                kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
                k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
                Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
                n_sp, thL, thH, thM,
                n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
                n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
                n_f, fA0, fb0, fE0, fAi, fbi, fEi,
                fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
            )
            _adiabatic_jvp_pb(
                tid, n_threads,
                T, C, vT_, sh_vC, omega, sh_Jvo, n_sp, thL, thH, thM,
                sh_w, sh_scratch,
            )
            _block_norm2_inplace_sub(
                sh_rhs, sh_dx, sh_w, h, sh_scratch, sh_w,
                tid, n_state, n_threads,
            )
            cuda.syncthreads()
            beta = math.sqrt(sh_scratch[0])
            if beta < tol:
                return
            inv = float64(1.0) / beta
            i   = tid
            while i < n_state:
                gm_V[i] = sh_w[i] * inv
                i += n_threads
            cuda.syncthreads()

        if tid == 0:
            sh_g[0] = beta
        cuda.syncthreads()

        j_conv = R

        # ── Boucle d'Arnoldi ──────────────────────────────────────────
        for j in range(R):
            vj_off = j * n_state

            vT_ = gm_V[vj_off]
            i   = tid
            while i < n_sp:
                sh_vC[i] = gm_V[vj_off + i + 1]
                i += n_threads
            cuda.syncthreads()

            # Produit matvec : sh_w = (I - h*J) * V[j]
            _eval_jvp_pb(
                tid, n_threads,
                T, C, vT_, sh_vC, sh_Jvo,
                kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
                kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
                kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
                k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
                Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
                n_sp, thL, thH, thM,
                n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
                n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
                n_f, fA0, fb0, fE0, fAi, fbi, fEi,
                fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
            )
            _adiabatic_jvp_pb(
                tid, n_threads,
                T, C, vT_, sh_vC, omega, sh_Jvo, n_sp, thL, thH, thM,
                sh_w, sh_scratch,
            )
            i = tid
            while i < n_state:
                sh_w[i] = gm_V[vj_off + i] - h * sh_w[i]
                i += n_threads
            cuda.syncthreads()

            # Gram–Schmidt modifié
            for ii in range(j + 1):
                vi_off = ii * n_state
                _block_dot(sh_w, gm_V[vi_off:vi_off + n_state],
                           sh_scratch, tid, n_state, n_threads)
                cuda.syncthreads()
                hij = sh_scratch[0]
                if tid == 0:
                    sh_H[ii * R + j] = hij
                cuda.syncthreads()
                i = tid
                while i < n_state:
                    sh_w[i] -= hij * gm_V[vi_off + i]
                    i += n_threads
                cuda.syncthreads()

            _block_norm2(sh_w, sh_scratch, tid, n_state, n_threads)
            cuda.syncthreads()
            h_j1j = math.sqrt(sh_scratch[0])
            if tid == 0:
                sh_H[(j + 1) * R + j] = h_j1j
            cuda.syncthreads()

            vj1_off = (j + 1) * n_state
            if h_j1j > 1e-300:
                inv2 = float64(1.0) / h_j1j
                i = tid
                while i < n_state:
                    gm_V[vj1_off + i] = sh_w[i] * inv2
                    i += n_threads
            else:
                i = tid
                while i < n_state:
                    gm_V[vj1_off + i] = float64(0.0)
                    i += n_threads
            cuda.syncthreads()

            # Rotations de Givens (thread 0 seulement — scalaire)
            if tid == 0:
                for ii in range(j):
                    hi  = sh_H[ii * R + j]; hi1 = sh_H[(ii + 1) * R + j]
                    c_  = sh_cs[ii];        s_  = sh_sn[ii]
                    sh_H[ii * R + j]       =  c_ * hi + s_ * hi1
                    sh_H[(ii + 1) * R + j] = -s_ * hi + c_ * hi1
                hjj   = sh_H[j * R + j]; hj1j = sh_H[(j + 1) * R + j]
                rdiag = math.sqrt(hjj * hjj + hj1j * hj1j)
                if rdiag > 1e-300:
                    cn = hjj / rdiag; sn_ = hj1j / rdiag
                else:
                    cn = float64(1.0); sn_ = float64(0.0)
                sh_cs[j] = cn; sh_sn[j] = sn_
                sh_H[j * R + j]       = rdiag
                sh_H[(j + 1) * R + j] = float64(0.0)
                gj = sh_g[j]
                sh_g[j]     =  cn  * gj
                sh_g[j + 1] = -sn_ * gj
            cuda.syncthreads()

            if abs(sh_g[j + 1]) < tol:
                j_conv = j + 1
                break

        # ── Substitution arrière (thread 0) ──────────────────────────
        m = j_conv
        if tid == 0:
            for k in range(m - 1, -1, -1):
                s_ = sh_g[k]
                for l in range(k + 1, m):
                    s_ -= sh_H[k * R + l] * sh_y[l]
                d     = sh_H[k * R + k]
                sh_y[k] = s_ / d if abs(d) > 1e-300 else float64(0.0)
        cuda.syncthreads()

        i = tid
        while i < n_state:
            acc = float64(0.0)
            for k in range(m):
                acc += gm_V[k * n_state + i] * sh_y[k]
            sh_dx[i] += acc
            i += n_threads
        cuda.syncthreads()

        # ── Vérification du résidu vrai ───────────────────────────────
        vT_ = sh_dx[0]
        i   = tid
        while i < n_sp:
            sh_vC[i] = sh_dx[i + 1]
            i += n_threads
        cuda.syncthreads()
        _eval_jvp_pb(
            tid, n_threads,
            T, C, vT_, sh_vC, sh_Jvo,
            kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
            kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
            kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
            k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
            Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
            n_sp, thL, thH, thM,
            n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
            n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
            n_f, fA0, fb0, fE0, fAi, fbi, fEi,
            fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
        )
        _adiabatic_jvp_pb(
            tid, n_threads,
            T, C, vT_, sh_vC, omega, sh_Jvo, n_sp, thL, thH, thM,
            sh_w, sh_scratch,
        )
        _block_norm2_inplace_sub(
            sh_rhs, sh_dx, sh_w, h, sh_scratch, sh_w,
            tid, n_state, n_threads,
        )
        cuda.syncthreads()
        res_true = math.sqrt(sh_scratch[0])

        if res_true < tol or j_conv < R:
            return
        beta = res_true

    return