"""
kernels/per_thread/jvp.py — JVP du vecteur source chimique (per-thread).

Architecture : 1 thread = 1 réacteur (scalaire).

Fonction exportée :
    _eval_jvp(T, C, vT, vC, Jv_om, <ctx_slices>, <mech_args>)

Lit le contexte cinétique mis en cache par _eval_rhs_and_context (context.py)
pour calculer la dérivée directionnelle de omega dans la direction (vT, vC).
La JVP du RHS adiabatique complet est dans shared/reactor_odes.py (_adiabatic_jvp).

CORRECTION vs version précédente :
    dlnKc/dT calculé via Σ νk × dBk/dT (PyJac eq. 73-75, paper §3.1)
    au lieu de (dSp - dSr)/(R*T) qui utilisait sR_re/sR_pe à tort.

    dBk/dT = (a0-1)/T + a5/T² + a1/2 + T*(a2/3 + T*(a3/4 + a4/5*T))
    dlnKc  = Σ_{k∈reac} (-ν_k) * dBk/dT + Σ_{k∈prod} ν_k * dBk/dT
           = Σ_k  ν_k_net * dBk/dT      (PyJac eq. 74)

    Les tableaux sR_re / sR_pe ne sont PAS utilisés pour la direction vT.
    Ils servent uniquement en interne à _eval_rhs_and_context pour
    calculer les taux de réaction (produit de concentrations en log),
    et n'ont aucun sens thermodynamique utilisable ici.
"""

import math
from numba import cuda, float64

from ..shared.thermo import R_CONST, EPS, LN10, INV_LN10


# ---------------------------------------------------------------------------
#  Helper interne : dBk/dT pour l'espèce sp à la température T
#  PyJac eq. (74) :
#    Bk = a6 - a0 + (a0-1)*lnT + T*(a1/2 + T*(a2/6 + T*(a3/12 + a4/20*T))) - a5/T
#    dBk/dT = (a0-1)/T + a5/T² + a1/2 + T*(a2/3 + T*(a3/4 + a4/5*T))
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _dB_dT(T, sp, thL, thH, thM):
    """
    Dérivée de Bk par rapport à T pour l'espèce sp.
    Sélectionne automatiquement les coefficients low/high selon T vs Tmid.
    """
    base = sp * 7
    if T <= thM[sp]:
        a = thL
    else:
        a = thH
    a0 = a[base + 0]
    a1 = a[base + 1]
    a2 = a[base + 2]
    a3 = a[base + 3]
    a4 = a[base + 4]
    a5 = a[base + 5]
    return ((a0 - float64(1.0)) / T
            + a5 / (T * T)
            + a1 / float64(2.0)
            + T * (a2 / float64(3.0)
                   + T * (a3 / float64(4.0)
                          + a4 / float64(5.0) * T)))


# ---------------------------------------------------------------------------
#  Helper interne : dlnKc/dT pour une réaction
#  PyJac eq. (73) : d(ln Kc)/dT = Σ_k ν_k * dBk/dT
#  avec ν_k = stoichio produits - stoichio réactants
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _dlnKc_dT_elem(T, thL, thH, thM,
                   RI, RS, RO_j, RO_j1,
                   PI, PS, PO_j, PO_j1):
    """
    Calcule d(ln Kc)/dT pour une réaction élémentaire ou trois-corps.
    Somme sur réactants (signe -) et produits (signe +).
    """
    val = float64(0.0)
    for k in range(RO_j, RO_j1):
        val -= RS[k] * _dB_dT(T, RI[k], thL, thH, thM)
    for k in range(PO_j, PO_j1):
        val += PS[k] * _dB_dT(T, PI[k], thL, thH, thM)
    return val


@cuda.jit(device=True)
def _eval_jvp(
    T, C, vT, vC, Jv_om,
    # ── Contexte cinétique (row-slices 1D, déjà calculé) ─────────────
    kf_e, Kc_e, cr_e, cp_e, dG0_e, sR_re, sR_pe,
    kf_tb, Kc_tb, cr_tb, cp_tb, dG0_tb, sR_rtb, sR_ptb, bRR_tb, M_tb,
    kf_f, Kc_f, cr_f, cp_f, dG0_f, sR_rf, sR_pf,
    k0_f, kinf_f, M_f, Pr_f, F_f, kb_f,
    Fcent_f, lFcent_f, Cpar_f, Npar_f, f1_f, exp_f,
    # ── Données mécanisme ─────────────────────────────────────────────
    n_sp,
    thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT,
    fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    """
    Calcule la JVP du vecteur source chimique omega.

    J·v où J = d(omega)/d(T,C) et v = (vT, vC).

    Résultat écrit dans Jv_om (taille n_sp).
    """
    for s in range(n_sp):
        Jv_om[s] = float64(0.0)

    # ── Élémentaires ──────────────────────────────────────────────────
    for j in range(n_e):
        kf  = kf_e[j]; Kc = Kc_e[j]
        cr  = cr_e[j]; cp = cp_e[j]
        ow  = eOW[j];  fac = float64(1.0) - ow

        dkf_dT = kf * (en[j] / T + eEa[j] / (R_CONST * T * T))

        # CORRECTION : d(ln Kc)/dT via dBk/dT  [PyJac eq.(73-74)]
        dlnKc = _dlnKc_dT_elem(
            T, thL, thH, thM,
            eRI, eRS, eRO[j], eRO[j + 1],
            ePI, ePS, ePO[j], ePO[j + 1],
        )
        dKc = Kc * dlnKc

        dRR_v = (dkf_dT * (cr - fac * cp / Kc)
                 + kf * fac * cp / (Kc * Kc) * dKc) * vT

        for k in range(eRO[j], eRO[j + 1]):
            sp = eRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf * cr * (eRS[k] / c) * vC[sp]
        for k in range(ePO[j], ePO[j + 1]):
            sp = ePI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf * fac / Kc * cp * (ePS[k] / c) * vC[sp]

        for k in range(eRO[j], eRO[j + 1]):
            Jv_om[eRI[k]] -= eRS[k] * dRR_v
        for k in range(ePO[j], ePO[j + 1]):
            Jv_om[ePI[k]] += ePS[k] * dRR_v

    # ── Trois-corps ───────────────────────────────────────────────────
    for j in range(n_tb):
        kf  = kf_tb[j]; Kc = Kc_tb[j]
        cr  = cr_tb[j]; cp = cp_tb[j]
        M   = M_tb[j];  br = bRR_tb[j]; ow = tbOW[j]; fac = float64(1.0) - ow

        dkf_dT = kf * (tbn[j] / T + tbEa[j] / (R_CONST * T * T))

        # CORRECTION : d(ln Kc)/dT via dBk/dT  [PyJac eq.(73-74)]
        dlnKc = _dlnKc_dT_elem(
            T, thL, thH, thM,
            tbRI, tbRS, tbRO[j], tbRO[j + 1],
            tbPI, tbPS, tbPO[j], tbPO[j + 1],
        )
        dKc = Kc * dlnKc

        dbR_dT = (dkf_dT * (cr - fac * cp / Kc)
                  + kf * fac * cp / (Kc * Kc) * dKc)

        dM_v = float64(0.0)
        for s in range(n_sp):
            dM_v += vC[s] * tbEff[j * n_sp + s]

        dRR_v = dbR_dT * vT * M

        for k in range(tbRO[j], tbRO[j + 1]):
            sp = tbRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf * cr * (tbRS[k] / c) * vC[sp] * M
        for k in range(tbPO[j], tbPO[j + 1]):
            sp = tbPI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf * fac / Kc * cp * (tbPS[k] / c) * vC[sp] * M
        dRR_v += br * dM_v

        for k in range(tbRO[j], tbRO[j + 1]):
            Jv_om[tbRI[k]] -= tbRS[k] * dRR_v
        for k in range(tbPO[j], tbPO[j + 1]):
            Jv_om[tbPI[k]] += tbPS[k] * dRR_v

    # ── Falloff ───────────────────────────────────────────────────────
    for j in range(n_f):
        kf_v = kf_f[j];  Kc  = Kc_f[j]
        cr   = cr_f[j];  cp  = cp_f[j]
        k0_v = k0_f[j];  ki  = kinf_f[j]; M = M_f[j]
        Pr   = Pr_f[j];  F   = F_f[j];    kb = kb_f[j]
        Fc   = Fcent_f[j]; lFc = lFcent_f[j]
        Cp   = Cpar_f[j];  Np  = Npar_f[j]
        f1_v = f1_f[j];   ex  = exp_f[j]
        ow   = fOW[j]; fac = float64(1.0) - ow
        ki_s = ki if ki > EPS else EPS
        pr_ok = (Pr > EPS) and (M > EPS)

        dk0_dT = k0_v * (fb0[j] / T + fE0[j] / (R_CONST * T * T))
        dki_dT = ki   * (fbi[j] / T + fEi[j] / (R_CONST * T * T))

        dM_v = float64(0.0)
        for s in range(n_sp):
            dM_v += vC[s] * fEff[j * n_sp + s]

        if pr_ok:
            dPr_dT = (dk0_dT * M - k0_v * M * dki_dT / ki_s) / ki_s
            dPr_vC = (k0_v / ki_s) * dM_v
        else:
            dPr_dT = float64(0.0); dPr_vC = float64(0.0)

        # CORRECTION : d(ln Kc)/dT via dBk/dT  [PyJac eq.(73-74)]
        dlnKc = _dlnKc_dT_elem(
            T, thL, thH, thM,
            fRI, fRS, fRO[j], fRO[j + 1],
            fPI, fPS, fPO[j], fPO[j + 1],
        )
        dKc = Kc * dlnKc
        ce = cr - fac * cp / Kc

        if fIT[j]:
            t1 = fT1[j]; t2 = fT2[j]; t3 = fT3[j]; tA = fTA[j]
            dFc_dT = float64(0.0)
            if t1 > EPS: dFc_dT += tA * (-float64(1.0) / t1) * math.exp(-T / t1)
            if t3 > EPS: dFc_dT += (float64(1.0) - tA) * (-float64(1.0) / t3) * math.exp(-T / t3)
            if t2 > EPS: dFc_dT += math.exp(-t2 / T) * (t2 / (T * T))

            fc_ok = Fc > EPS
            dlFc_dT = (dFc_dT / Fc * INV_LN10) if fc_ok else float64(0.0)
            dCp_dT = -float64(0.67) * dlFc_dT
            dNp_dT = -float64(1.27) * dlFc_dT

            Prr      = Pr if Pr > EPS else EPS
            dlPr_dPr = (float64(1.0) / (Prr * LN10)) if pr_ok else float64(0.0)
            lPr      = math.log10(Prr) if pr_ok else float64(0.0)
            nf1      = lPr + Cp
            df       = Np - float64(0.14) * nf1
            abs_df   = abs(df)
            df_s     = df if abs_df > EPS else (EPS if df >= float64(0.0) else -EPS)
            df_sq    = df_s * df_s; df_ok = abs_df > EPS

            df1_dPr = ((dlPr_dPr * df_s + float64(0.14) * nf1 * dlPr_dPr) / df_sq
                       if (df_ok and pr_ok) else float64(0.0))
            nT_     = dCp_dT
            dT_     = dNp_dT - float64(0.14) * dCp_dT
            df1_dT  = (nT_ * df_s - nf1 * dT_) / df_sq if df_ok else float64(0.0)

            f1sq   = f1_v * f1_v
            de_dlF = float64(1.0) / (float64(1.0) + f1sq)
            de_df1 = -float64(2.0) * f1_v * lFc / ((float64(1.0) + f1sq) ** 2)
            de_dT  = de_dlF * dlFc_dT + de_df1 * df1_dT
            de_dPr = de_df1 * df1_dPr
            dF_dT  = F * LN10 * de_dT
            dF_dPr = F * LN10 * de_dPr
        else:
            dF_dT = float64(0.0); dF_dPr = float64(0.0)

        Pd       = (float64(1.0) + Pr) * (float64(1.0) + Pr)
        dkb_dT   = dki_dT * (Pr / (float64(1.0) + Pr)) + ki_s * (float64(1.0) / Pd) * dPr_dT
        dkb_vC   = ki_s * (float64(1.0) / Pd) * dPr_vC
        dkf_dT_v = dkb_dT * F + kb * (dF_dT + dF_dPr * dPr_dT)
        dkf_vC   = dkb_vC * F + kb * dF_dPr * dPr_vC

        dRR_v = dkf_dT_v * vT * ce + kf_v * fac * cp / (Kc * Kc) * dKc * vT

        for k in range(fRO[j], fRO[j + 1]):
            sp = fRI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v += kf_v * cr * (fRS[k] / c) * vC[sp]
        for k in range(fPO[j], fPO[j + 1]):
            sp = fPI[k]; c = C[sp] if C[sp] > EPS else EPS
            dRR_v -= kf_v * fac / Kc * cp * (fPS[k] / c) * vC[sp]
        dRR_v += dkf_vC * ce

        for k in range(fRO[j], fRO[j + 1]):
            Jv_om[fRI[k]] -= fRS[k] * dRR_v
        for k in range(fPO[j], fPO[j + 1]):
            Jv_om[fPI[k]] += fPS[k] * dRR_v


"""
kernels/per_thread/jvp_wrapper.py — Wrappers Python pour JVP et RHS (per-thread).

Classes exportées
-----------------
CUDARHSEvaluatorPT
    Évalue omega = d(C)/dt au point (T, C).
    → omega : (B, n_sp)

CUDAJVPEvaluatorPT
    Évalue la dérivée directionnelle J·v de omega dans la direction (vT, vC).
    Précalcule automatiquement le contexte cinétique avant la JVP.
    → Jv_om : (B, n_sp)

CUDAAdiabaticJVPEvaluatorPT
    Évalue la JVP du RHS adiabatique complet (dT/dt, dC/dt) dans la direction
    (vT, vC). Précalcule omega et le contexte cinétique.
    → Jv_TC : (B, n_state)
"""

import numpy as np
from .context import _eval_context

MAX_N_STATE = 54


@cuda.jit(cache=True)
def _jvp_kernel_pt(
    TC_in, VTC_in, Jv_out,
    ctx_kf_e, ctx_Kc_e, ctx_cr_e, ctx_cp_e, ctx_dG0_e, ctx_sRre, ctx_sRpe,
    ctx_kf_tb, ctx_Kc_tb, ctx_cr_tb, ctx_cp_tb, ctx_dG0_tb,
    ctx_sRrtb, ctx_sRptb, ctx_bRR_tb, ctx_M_tb,
    ctx_kf_f, ctx_Kc_f, ctx_cr_f, ctx_cp_f, ctx_dG0_f, ctx_sRrf, ctx_sRpf,
    ctx_k0_f, ctx_kinf_f, ctx_M_f, ctx_Pr_f, ctx_F_f, ctx_kb_f,
    ctx_Fc_f, ctx_lFc_f, ctx_Cp_f, ctx_Np_f, ctx_f1_f, ctx_exp_f,
    n_sp, thL, thH, thM,
    n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
    n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
    n_f, fA0, fb0, fE0, fAi, fbi, fEi,
    fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
):
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b >= TC_in.shape[0]:
        return
    T_loc = TC_in[b, 0]
    vT    = VTC_in[b, 0]
    C     = cuda.local.array(MAX_N_STATE, float64)
    vC    = cuda.local.array(MAX_N_STATE, float64)
    Jv_om = cuda.local.array(MAX_N_STATE, float64)
    for s in range(n_sp):
        C[s]  = TC_in[b,  s + 1]
        vC[s] = VTC_in[b, s + 1]

    _eval_context(
        T_loc, C,
        ctx_kf_e[b],  ctx_Kc_e[b],  ctx_cr_e[b],  ctx_cp_e[b],  ctx_dG0_e[b],
        ctx_sRre[b],  ctx_sRpe[b],
        ctx_kf_tb[b], ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
        ctx_sRrtb[b], ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
        ctx_kf_f[b],  ctx_Kc_f[b],  ctx_cr_f[b],  ctx_cp_f[b],  ctx_dG0_f[b],
        ctx_sRrf[b],  ctx_sRpf[b],
        ctx_k0_f[b],  ctx_kinf_f[b], ctx_M_f[b],  ctx_Pr_f[b],  ctx_F_f[b],
        ctx_kb_f[b],  ctx_Fc_f[b],  ctx_lFc_f[b], ctx_Cp_f[b],  ctx_Np_f[b],
        ctx_f1_f[b],  ctx_exp_f[b],
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )
    _eval_jvp(
        T_loc, C, vT, vC, Jv_om,
        ctx_kf_e[b],  ctx_Kc_e[b],  ctx_cr_e[b],  ctx_cp_e[b],  ctx_dG0_e[b],
        ctx_sRre[b],  ctx_sRpe[b],
        ctx_kf_tb[b], ctx_Kc_tb[b], ctx_cr_tb[b], ctx_cp_tb[b], ctx_dG0_tb[b],
        ctx_sRrtb[b], ctx_sRptb[b], ctx_bRR_tb[b], ctx_M_tb[b],
        ctx_kf_f[b],  ctx_Kc_f[b],  ctx_cr_f[b],  ctx_cp_f[b],  ctx_dG0_f[b],
        ctx_sRrf[b],  ctx_sRpf[b],
        ctx_k0_f[b],  ctx_kinf_f[b], ctx_M_f[b],  ctx_Pr_f[b],  ctx_F_f[b],
        ctx_kb_f[b],  ctx_Fc_f[b],  ctx_lFc_f[b], ctx_Cp_f[b],  ctx_Np_f[b],
        ctx_f1_f[b],  ctx_exp_f[b],
        n_sp, thL, thH, thM,
        n_e, eA, en, eEa, eOW, eDn, eRI, eRS, eRO, ePI, ePS, ePO,
        n_tb, tbA, tbn, tbEa, tbOW, tbDn, tbRI, tbRS, tbRO, tbPI, tbPS, tbPO, tbEff,
        n_f, fA0, fb0, fE0, fAi, fbi, fEi,
        fOW, fDn, fTA, fT1, fT2, fT3, fIT, fRI, fRS, fRO, fPI, fPS, fPO, fEff,
    )
    for s in range(n_sp):
        Jv_out[b, s] = Jv_om[s]


# =============================================================================
#  Classe de base — allocation du contexte (partagée entre les trois wrappers)
# =============================================================================

class _BaseEvaluatorPT:
    THREADS = 64

    def __init__(self, mech, dtype=np.float64):
        self._mech    = mech
        self._dtype   = dtype
        self._n_sp    = mech.n_species
        self._n_state = self._n_sp + 1
        self._ctx     = None
        self._B_alloc = -1

    def _alloc_context(self, B):
        dt = self._dtype
        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff

        def z(shape):
            return cuda.device_array(shape, dtype=dt)

        n_e_flat  = e.react_offset[-1]  if e.n_reactions  else 1
        n_tb_flat = tb.react_offset[-1] if tb.n_reactions else 1
        n_f_flat  = f.react_offset[-1]  if f.n_reactions  else 1

        self._ctx = dict(
            kf_e   = z((B, e.n_reactions)),    Kc_e   = z((B, e.n_reactions)),
            cr_e   = z((B, e.n_reactions)),    cp_e   = z((B, e.n_reactions)),
            dG0_e  = z((B, e.n_reactions)),
            sRre   = z((B, n_e_flat)),         sRpe   = z((B, n_e_flat)),
            kf_tb  = z((B, tb.n_reactions)),   Kc_tb  = z((B, tb.n_reactions)),
            cr_tb  = z((B, tb.n_reactions)),   cp_tb  = z((B, tb.n_reactions)),
            dG0_tb = z((B, tb.n_reactions)),
            sRrtb  = z((B, n_tb_flat)),        sRptb  = z((B, n_tb_flat)),
            bRR_tb = z((B, tb.n_reactions)),   M_tb   = z((B, tb.n_reactions)),
            kf_f   = z((B, f.n_reactions)),    Kc_f   = z((B, f.n_reactions)),
            cr_f   = z((B, f.n_reactions)),    cp_f   = z((B, f.n_reactions)),
            dG0_f  = z((B, f.n_reactions)),
            sRrf   = z((B, n_f_flat)),         sRpf   = z((B, n_f_flat)),
            k0_f   = z((B, f.n_reactions)),    kinf_f = z((B, f.n_reactions)),
            M_f    = z((B, f.n_reactions)),    Pr_f   = z((B, f.n_reactions)),
            F_f    = z((B, f.n_reactions)),    kb_f   = z((B, f.n_reactions)),
            Fc_f   = z((B, f.n_reactions)),    lFc_f  = z((B, f.n_reactions)),
            Cp_f   = z((B, f.n_reactions)),    Np_f   = z((B, f.n_reactions)),
            f1_f   = z((B, f.n_reactions)),    exp_f  = z((B, f.n_reactions)),
        )
        self._B_alloc = B

    def _prepare(self, T_arr, C_arr):
        T_arr = np.asarray(T_arr, dtype=self._dtype)
        C_arr = np.asarray(C_arr, dtype=self._dtype)
        B     = T_arr.shape[0]
        nst   = self._n_state
        if self._B_alloc != B:
            self._alloc_context(B)
        TC = np.empty((B, nst), dtype=self._dtype)
        TC[:, 0]  = T_arr
        TC[:, 1:] = C_arr
        return B, cuda.to_device(TC)

    def _mech_args(self):
        m  = self._mech
        e  = m.elementary
        tb = m.threebody
        f  = m.falloff
        cx = self._ctx
        return (
            cx['kf_e'],  cx['Kc_e'],  cx['cr_e'],  cx['cp_e'],  cx['dG0_e'],
            cx['sRre'],  cx['sRpe'],
            cx['kf_tb'], cx['Kc_tb'], cx['cr_tb'], cx['cp_tb'], cx['dG0_tb'],
            cx['sRrtb'], cx['sRptb'], cx['bRR_tb'], cx['M_tb'],
            cx['kf_f'],  cx['Kc_f'],  cx['cr_f'],  cx['cp_f'],  cx['dG0_f'],
            cx['sRrf'],  cx['sRpf'],
            cx['k0_f'],  cx['kinf_f'], cx['M_f'],  cx['Pr_f'],  cx['F_f'],
            cx['kb_f'],  cx['Fc_f'],  cx['lFc_f'], cx['Cp_f'],  cx['Np_f'],
            cx['f1_f'],  cx['exp_f'],
            m.n_species,
            m.thermo.coef_low, m.thermo.coef_high, m.thermo.Tmid,
            e.n_reactions, e.A, e.n, e.Ea, e.oneWay,
            e.delta_n, e.react_idx, e.react_sto,
            e.react_offset, e.prod_idx, e.prod_sto, e.prod_offset,
            tb.n_reactions, tb.A, tb.n, tb.Ea, tb.oneWay,
            tb.delta_n, tb.react_idx, tb.react_sto,
            tb.react_offset, tb.prod_idx, tb.prod_sto, tb.prod_offset, tb.eff_matrix,
            f.n_reactions, f.A_k0, f.b_k0, f.Ea_k0, f.A_kinf, f.b_kinf, f.Ea_kinf,
            f.oneWay, f.delta_n,
            f.troe_A, f.troe_T1, f.troe_T2, f.troe_T3, f.is_troe,
            f.react_idx, f.react_sto, f.react_offset,
            f.prod_idx, f.prod_sto, f.prod_offset, f.eff_matrix,
        )

    def _launch(self, kernel, B, *args):
        threads = self.THREADS
        blocks  = (B + threads - 1) // threads
        kernel[blocks, threads](*args)
        cuda.synchronize()


# =============================================================================
#  CUDAJVPEvaluatorPT
# =============================================================================

class CUDAJVPEvaluatorPT(_BaseEvaluatorPT):
    """
    Évalue la JVP chimique J·v  où  J = d(omega)/d(T,C),  v = (vT, vC).

    Usage
    -----
    ev    = CUDAJVPEvaluatorPT(mech)
    Jv_om = ev.evaluate(T_arr, C_arr, vT_arr, vC_arr)   # (B, n_sp)
    """

    def evaluate(self, T_arr, C_arr, vT_arr, vC_arr):
        B, d_TC = self._prepare(T_arr, C_arr)
        nst     = self._n_state
        #vT_arr  = np.asarray(vT_arr, dtype=self._dtype)
        #vC_arr  = np.asarray(vC_arr, dtype=self._dtype)
        VTC     = np.empty((B, nst), dtype=self._dtype)
        VTC[:, 0]  = vT_arr
        VTC[:, 1:] = vC_arr
        d_VTC   = cuda.to_device(VTC)
        d_Jv    = cuda.device_array((B, self._n_sp), dtype=self._dtype)
        self._launch(_jvp_kernel_pt, B, d_TC, d_VTC, d_Jv, *self._mech_args())
        return d_Jv.copy_to_host()

