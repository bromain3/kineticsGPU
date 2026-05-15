"""
kernels/per_thread/gmres.py — GMRES redémarré matriciel-libre (per-thread).

Architecture : 1 thread = 1 réacteur (scalaire).
Tout le scratch GMRES (V, H, cs, sn, g, w, y) est en mémoire globale
(row-slices pré-allouées) pour éviter le register spilling.

Fonction exportée :
    _gmres(T, C, h, omega, rhs, dx, V, H, cs, sn, g, w, y,
           vC_tmp, Jvo_tmp, n_state, R, tol, maxiter,
           <ctx_slices>, <mech_args>)

Algorithme : GMRES(R) avec vérification du résidu vrai après chaque cycle.
Le produit matrice-vecteur est remplacé par la JVP analytique :
    (I - h*J) · v  ≈  v - h * _adiabatic_jvp(v)
"""

import math
from numba import cuda, float64

from kineticsGPU.cuda.kernels.per_thread.reactor_odes import _adiabatic_jvp
from .jvp import _eval_jvp


@cuda.jit(device=True)
def _gmres(
    T, C, h, omega,
    rhs, dx,
    V, H, cs, sn, g, w, y,
    vC_tmp, Jvo_tmp,
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
    Résout (I - h*J) dx = rhs par GMRES(R) matriciel-libre.

    Le produit Jacobien-vecteur (I - h*J)·v est évalué comme :
        v - h * adiabatic_jvp(v)
    en réutilisant le contexte cinétique déjà calculé.
    """
    # Initialisation de dx = 0, calcul de beta = ||rhs||
    beta = float64(0.0)
    for i in range(n_state):
        dx[i] = float64(0.0)
        beta  += rhs[i] * rhs[i]
    beta = math.sqrt(beta)
    if beta < tol:
        return

    for outer in range(maxiter):
        # Initialisation des buffers GMRES
        for i in range((R + 1) * R): H[i]  = float64(0.0)
        for i in range(R):           cs[i]  = float64(0.0); sn[i] = float64(0.0)
        for i in range(R + 1):       g[i]   = float64(0.0)

        if outer == 0:
            # Premier vecteur de Krylov : v0 = rhs / beta
            inv = float64(1.0) / beta
            for i in range(n_state):
                V[i] = rhs[i] * inv
        else:
            # Recalcul du résidu vrai pour le vecteur de restart
            vT_ = dx[0]
            for s in range(n_sp): vC_tmp[s] = dx[s + 1]
            _eval_jvp(
                T, C, vT_, vC_tmp, Jvo_tmp,
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
            _adiabatic_jvp(T, C, vT_, vC_tmp, omega, Jvo_tmp, n_sp, thL, thH, thM, w)
            beta = float64(0.0)
            for i in range(n_state):
                ri    = rhs[i] - (dx[i] - h * w[i])
                w[i]  = ri; beta += ri * ri
            beta = math.sqrt(beta)
            if beta < tol: return
            inv = float64(1.0) / beta
            for i in range(n_state):
                V[i] = w[i] * inv

        g[0]   = beta
        j_conv = R

        # ── Boucle d'Arnoldi ──────────────────────────────────────────
        for j in range(R):
            vj   = j * n_state
            vT_  = V[vj]
            for s in range(n_sp): vC_tmp[s] = V[vj + s + 1]

            # Produit matvec : w = (I - h*J) * V[j]
            _eval_jvp(
                T, C, vT_, vC_tmp, Jvo_tmp,
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
            _adiabatic_jvp(T, C, vT_, vC_tmp, omega, Jvo_tmp, n_sp, thL, thH, thM, w)
            for i in range(n_state):
                w[i] = V[vj + i] - h * w[i]

            # Gram–Schmidt modifié
            for i in range(j + 1):
                vi  = i * n_state
                hij = float64(0.0)
                for k in range(n_state): hij += w[k] * V[vi + k]
                H[i * R + j] = hij
                for k in range(n_state): w[k] -= hij * V[vi + k]

            h_j1j = float64(0.0)
            for k in range(n_state): h_j1j += w[k] * w[k]
            h_j1j = math.sqrt(h_j1j)
            H[(j + 1) * R + j] = h_j1j

            vj1 = (j + 1) * n_state
            if h_j1j > 1e-300:
                inv2 = float64(1.0) / h_j1j
                for k in range(n_state): V[vj1 + k] = w[k] * inv2
            else:
                for k in range(n_state): V[vj1 + k] = float64(0.0)

            # Rotations de Givens
            for i in range(j):
                hi  = H[i * R + j]; hi1 = H[(i + 1) * R + j]
                c_  = cs[i];        s_  = sn[i]
                H[i * R + j]       =  c_ * hi + s_ * hi1
                H[(i + 1) * R + j] = -s_ * hi + c_ * hi1

            hjj   = H[j * R + j]; hj1j = H[(j + 1) * R + j]
            rdiag = math.sqrt(hjj * hjj + hj1j * hj1j)
            if rdiag > 1e-300:
                cn = hjj / rdiag; sn_ = hj1j / rdiag
            else:
                cn = float64(1.0); sn_ = float64(0.0)
            cs[j] = cn; sn[j] = sn_
            H[j * R + j]       = rdiag
            H[(j + 1) * R + j] = float64(0.0)
            gj = g[j]
            g[j]     =  cn  * gj
            g[j + 1] = -sn_ * gj

            if abs(g[j + 1]) < tol:
                j_conv = j + 1
                break

        # ── Substitution arrière ──────────────────────────────────────
        m = j_conv
        for k in range(m - 1, -1, -1):
            s_ = g[k]
            for l in range(k + 1, m): s_ -= H[k * R + l] * y[l]
            d  = H[k * R + k]
            y[k] = s_ / d if abs(d) > 1e-300 else float64(0.0)

        for i in range(n_state):
            acc = float64(0.0)
            for k in range(m): acc += V[k * n_state + i] * y[k]
            dx[i] += acc

        # ── Vérification du résidu vrai ───────────────────────────────
        vT_ = dx[0]
        for s in range(n_sp): vC_tmp[s] = dx[s + 1]
        _eval_jvp(
            T, C, vT_, vC_tmp, Jvo_tmp,
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
        _adiabatic_jvp(T, C, vT_, vC_tmp, omega, Jvo_tmp, n_sp, thL, thH, thM, w)
        res_true = float64(0.0)
        for i in range(n_state):
            ri        = rhs[i] - (dx[i] - h * w[i])
            res_true += ri * ri
        res_true = math.sqrt(res_true)

        if res_true < tol or j_conv < R:
            return
        beta = res_true

    return