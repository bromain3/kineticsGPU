"""
thermo_cv.py
============
CUDA kernels for the thermodynamic part of the adiabatic constant-volume RHS:

  dT/dt = - (Σ_k  u_k(T) · omega_k)  /  (n_tot · cv_mix)

and its analytic JVP:

  d(dT/dt)/d[T,C] · [vT, vC]

These kernels complement the reaction-rate kernels in rhs_jvp.py.
They are per-thread only (per-block variant can be added analogously).

Layout of the full state vector:
  TC[cell, 0]        = T       [K]
  TC[cell, 1:nsp+1]  = C_k    [mol/m³]
"""

import math
from numba import cuda

R_GAS  = 8.31446261815324

# ── NASA polynomial helpers (must match rhs_jvp.py) ───────────────────────────
@cuda.jit(device=True, inline=True)
def _h_nasa(coef, T):
    T2 = T*T; T3 = T2*T; T4 = T3*T
    return R_GAS*T*(coef[0] + coef[1]*T/2 + coef[2]*T2/3
                    + coef[3]*T3/4 + coef[4]*T4/5 + coef[5]/T)

@cuda.jit(device=True, inline=True)
def _u_nasa(coef, T):
    return _h_nasa(coef, T) - R_GAS * T

@cuda.jit(device=True, inline=True)
def _cv_nasa(coef, T):
    T2 = T*T; T3 = T2*T; T4 = T3*T
    cp = R_GAS*(coef[0] + coef[1]*T + coef[2]*T2 + coef[3]*T3 + coef[4]*T4)
    return cp - R_GAS

@cuda.jit(device=True, inline=True)
def _dcp_dT_nasa(coef, T):
    """d cp / dT = R * (b2 + 2*b3*T + 3*b4*T² + 4*b5*T³)"""
    T2 = T*T; T3 = T2*T
    return R_GAS*(coef[1] + 2.0*coef[2]*T + 3.0*coef[3]*T2 + 4.0*coef[4]*T3)

@cuda.jit(device=True, inline=True)
def _dcv_dT_nasa(coef, T):
    return _dcp_dT_nasa(coef, T)   # dcv/dT = dcp/dT


# ══════════════════════════════════════════════════════════════════════════════
# §1  RHS  dT/dt  (adiabatic CV)
# ══════════════════════════════════════════════════════════════════════════════

@cuda.jit
def pt_dT_from_omega(
    T,          # (ncells,)
    C,          # (ncells, nsp)
    omega,      # (ncells, nsp)   precomputed species production rates
    dTC,        # (ncells, nsp+1) output full state derivative
                #   dTC[:,0]  = dT/dt
                #   dTC[:,1:] = omega  (simply copied)
    coef_low,   # (nsp, 7)
    coef_high,  # (nsp, 7)
    Tmid,       # (nsp,)
):
    """
    Per-thread kernel.
    Fills dTC[:, 0]   = dT/dt
          dTC[:, 1:]  = omega   (copy)

    dT/dt = - (Σ_k  u_k(T) · omega_k) / (n_tot · cv_mix_mol)

    where:
      n_tot        = Σ_k C_k                    [mol/m³]
      cv_mix_mol   = Σ_k X_k · cv_k(T)          [J/(mol·K)]
      X_k          = C_k / n_tot
    """
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc   = T[cell]
    nsp  = C.shape[1]

    n_tot  = 0.0
    U_dot  = 0.0   # Σ u_k · omega_k
    cv_sum = 0.0   # Σ C_k · cv_k  (used for molar cv_mix = cv_sum / n_tot)

    for k in range(nsp):
        Ck = C[cell, k]
        if Tc > Tmid[k]:
            u_k  = _u_nasa(coef_high[k], Tc)
            cv_k = _cv_nasa(coef_high[k], Tc)
        else:
            u_k  = _u_nasa(coef_low[k],  Tc)
            cv_k = _cv_nasa(coef_low[k],  Tc)
        n_tot  += Ck
        U_dot  += u_k  * omega[cell, k]
        cv_sum += Ck   * cv_k

    # cv_mix_mol = cv_sum / n_tot  (avoid div by zero)
    n_safe = n_tot if n_tot > 1e-300 else 1e-300
    cv_mix = cv_sum / n_safe

    dT_dt = -U_dot / (n_safe * cv_mix)

    dTC[cell, 0] = dT_dt
    for k in range(nsp):
        dTC[cell, k + 1] = omega[cell, k]


# ══════════════════════════════════════════════════════════════════════════════
# §2  JVP of dT/dt
# ══════════════════════════════════════════════════════════════════════════════

@cuda.jit
def pt_jvp_dT(
    T,           # (ncells,)
    C,           # (ncells, nsp)
    omega,       # (ncells, nsp)      f(T,C)
    dTC,         # (ncells, nsp+1)    RHS (dT/dt, omega)
    v_T,         # (ncells,)          perturbation in T
    v_C,         # (ncells, nsp)      perturbation in C
    Jv_omega,    # (ncells, nsp)      (d omega / d[T,C]) · [vT,vC]  (precomputed)
    Jv_dTC,      # (ncells, nsp+1)    output JVP of full state derivative
    coef_low,    # (nsp, 7)
    coef_high,   # (nsp, 7)
    Tmid,        # (nsp,)
):
    """
    Per-thread JVP of the full RHS [dT/dt, omega]:

      Jv_dTC[:,0]  = d(dT/dt)/d[T,C] · [vT, vC]
      Jv_dTC[:,1:] = Jv_omega   (already computed externally)

    Derivation (notation: U' = Σ u_k · omega_k, n = Σ C_k,
                                cv_m = Σ C_k cv_k / n):
      d(dT/dt) = -[dU'·n·cv_m - U'·(dn·cv_m + n·d_cv_m)] / (n·cv_m)²
    """
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc  = T[cell]
    nsp = C.shape[1]

    # Compute base quantities
    n_tot  = 0.0; U_dot  = 0.0; cv_sum = 0.0
    for k in range(nsp):
        Ck = C[cell, k]
        if Tc > Tmid[k]:
            u_k  = _u_nasa(coef_high[k], Tc)
            cv_k = _cv_nasa(coef_high[k], Tc)
        else:
            u_k  = _u_nasa(coef_low[k],  Tc)
            cv_k = _cv_nasa(coef_low[k],  Tc)
        n_tot  += Ck
        U_dot  += u_k  * omega[cell, k]
        cv_sum += Ck   * cv_k

    n_safe = n_tot if n_tot > 1e-300 else 1e-300
    cv_mix = cv_sum / n_safe          # molar cv_mix
    denom  = n_safe * cv_mix          # = cv_sum

    # Perturbation terms:
    # dU' = Σ_k ( u_k · Jv_omega_k  +  cv_k · omega_k · vT )
    # dn  = Σ_k vC_k
    # d(cv_sum) = Σ_k ( vC_k·cv_k + C_k·dcv_k/dT·vT )
    dU_prime  = 0.0
    dn        = 0.0
    d_cv_sum  = 0.0

    for k in range(nsp):
        Ck = C[cell, k]
        if Tc > Tmid[k]:
            u_k     = _u_nasa(coef_high[k], Tc)
            cv_k    = _cv_nasa(coef_high[k], Tc)
            dcv_dT  = _dcv_dT_nasa(coef_high[k], Tc)
        else:
            u_k     = _u_nasa(coef_low[k],  Tc)
            cv_k    = _cv_nasa(coef_low[k],  Tc)
            dcv_dT  = _dcv_dT_nasa(coef_low[k],  Tc)

        dU_prime += u_k  * Jv_omega[cell, k] + cv_k * omega[cell, k] * v_T[cell]
        dn       += v_C[cell, k]
        d_cv_sum += v_C[cell, k] * cv_k + Ck * dcv_dT * v_T[cell]

    # d(dT/dt) via quotient rule:
    #   dT/dt = -U' / denom    where denom = cv_sum = n * cv_mix
    #   d(dT/dt) = -(dU' * denom - U_dot * d_cv_sum) / denom²
    dT_dt_jvp = -(dU_prime * denom - U_dot * d_cv_sum) / (denom * denom)

    Jv_dTC[cell, 0] = dT_dt_jvp
    for k in range(nsp):
        Jv_dTC[cell, k + 1] = Jv_omega[cell, k]