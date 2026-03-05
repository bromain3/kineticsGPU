"""
test_reactor.py
===============
Script de validation du réacteur à volume constant (RHS + JVP).

Vérifie :
  1. Compilation des device functions (constant_volume_reactor_rhs / _jvp)
  2. Cohérence numérique du RHS vs une référence torch (rhs_adiabatic)
  3. Cohérence numérique du JVP vs différences finies sur le RHS

Usage :
    python test_reactor.py
"""

import time
import math
import numpy as np
import torch
from numba import cuda

# ── imports projet ────────────────────────────────────────────────────────────
import kineticsGPU.numba_implementation.thermo as thermo
import thermo_classique as th_clas
import constants

from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.numba_implementation.yaml_to_GPU_array import (
    build_elementary_gpu_arrays,
    build_threebody_gpu_arrays,
    build_falloff_gpu_arrays,
)
from kineticsGPU.numba_implementation.elementary_reaction import (
    pack_elementary_context_packaged,
    N_CTX_FIELDS as N_CTX_ELEM,
)
from kineticsGPU.numba_implementation.three_body_reaction import (
    pack_threebody_context_packaged,
    N_CTX_TB_FIELDS as N_CTX_TB,
)
from kineticsGPU.numba_implementation.falloff_reaction import (
    pack_falloff_context_packaged,
    N_CTX_FO_FIELDS as N_CTX_FO,
)

# La device function à tester
from constant_volume_reactor import (        # ← adapter si le module s'appelle autrement
    constant_volume_reactor_rhs,
    constant_volume_reactor_jvp,
)

from reactionTypesContext import (
    ElementaryReactionBatchContext,
    ThreeBodyReactionBatchContext,
    FalloffReactionBatchContext,
)

# =============================================================================
#  PARAMÈTRES
# =============================================================================

YAML_FILE  = '../gri30.yaml'
NCELLS     = 256          # petit pour que le test soit rapide
EPS_FD     = 1e-5         # pas différences finies
TOL_RHS    = 1e-8         # tolérance erreur relative RHS vs torch
TOL_JVP    = 1e-5         # tolérance erreur relative JVP vs FD
THREADS    = 64
GRID       = (NCELLS + THREADS - 1) // THREADS

# =============================================================================
#  SETUP : thermo + réactions
# =============================================================================

print("── Setup ────────────────────────────────────────────────────────────────")
lecteur      = thermo.yamlParser(YAML_FILE)
lecteur_clas = th_clas.yamlParser(YAML_FILE)
th_object      = lecteur.populateVectorizedThermoObject()
clas_th_object = lecteur_clas.populateVectorizedThermoObject(device='cuda:0', dtype=torch.float64)

reactions = parseReactions(YAML_FILE)
r_e, r_tb, r_f, system = build_reactions(
    clas_th_object.species, reactions, clas_th_object, 'cuda:0', torch.float64)

batched_elem    = ElementaryReactionBatchContext(r_e,  device='cuda:0', dtype=torch.float64)
batched_tb      = ThreeBodyReactionBatchContext (r_tb, device='cuda:0', dtype=torch.float64)
batched_falloff = FalloffReactionBatchContext   (r_f,  device='cuda:0', dtype=torch.float64)

elem_arrays   = build_elementary_gpu_arrays(reactions, th_object)
tb_arrays     = build_threebody_gpu_arrays (reactions, th_object)
fo_arrays     = build_falloff_gpu_arrays   (reactions, th_object)
thermo_arrays = th_object.package_values()

nspecies   = thermo_arrays[2].shape[0]
nreac_elem = elem_arrays[8].shape[0]
nreac_tb   = tb_arrays  [8].shape[0]
nreac_fo   = fo_arrays  [8].shape[0]

print(f"  nspecies = {nspecies}  |  elem={nreac_elem}  tb={nreac_tb}  fo={nreac_fo}")

# ── données d'entrée ─────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
TC  = rng.random((NCELLS, nspecies + 1))
TC[:, 0] = 2000.0                                    # T fixée
TC[:, 1:] = np.abs(TC[:, 1:]) * 1e-3 + 1e-6         # concentrations > 0

vTC = rng.random((NCELLS, nspecies + 1))

# Numpy / CUDA
T_d  = cuda.to_device(np.ascontiguousarray(TC[:,  0],  dtype=np.float64))
C_d  = cuda.to_device(np.ascontiguousarray(TC[:, 1:],  dtype=np.float64))
vT_d = cuda.to_device(np.ascontiguousarray(vTC[:,  0], dtype=np.float64))
vC_d = cuda.to_device(np.ascontiguousarray(vTC[:, 1:], dtype=np.float64))

# Torch
T_t  = torch.tensor(TC[:,  0],  device='cuda:0', dtype=torch.float64)
C_t  = torch.tensor(TC[:, 1:],  device='cuda:0', dtype=torch.float64)
vT_t = torch.tensor(vTC[:,  0], device='cuda:0', dtype=torch.float64)
vC_t = torch.tensor(vTC[:, 1:], device='cuda:0', dtype=torch.float64)

print("  Setup OK")

# =============================================================================
#  KERNELS WRAPPER
# =============================================================================

@cuda.jit
def kernel_rhs(T, C,
               elem_react_arrays, tb_react_arrays, fo_react_arrays,
               thermo_arrays,
               elem_ctx, tb_ctx, fo_ctx,
               dXdt_out):
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc      = T[cell]
    Cc      = C[cell]
    e_ctx   = elem_ctx[cell]
    t_ctx   = tb_ctx  [cell]
    f_ctx   = fo_ctx  [cell]
    dXdt    = dXdt_out[cell]

    # remplissage des contextes
    pack_elementary_context_packaged(Tc, Cc, elem_react_arrays, thermo_arrays, e_ctx)
    pack_threebody_context_packaged (Tc, Cc, tb_react_arrays,   thermo_arrays, t_ctx)
    pack_falloff_context_packaged   (Tc, Cc, fo_react_arrays,   thermo_arrays, f_ctx)

    constant_volume_reactor_rhs(
        Tc, Cc,
        e_ctx, elem_react_arrays,
        t_ctx, tb_react_arrays,
        f_ctx, fo_react_arrays,
        thermo_arrays,
        dXdt,
    )


@cuda.jit
def kernel_jvp(T, C, vT, vC,
               elem_react_arrays, tb_react_arrays, fo_react_arrays,
               thermo_arrays,
               elem_ctx, tb_ctx, fo_ctx,
               dXdt_buf, jvp_out):
    cell = cuda.grid(1)
    if cell >= T.shape[0]:
        return

    Tc     = T [cell];  Cc     = C [cell]
    vTc    = vT[cell];  vCc    = vC[cell]
    e_ctx  = elem_ctx[cell]
    t_ctx  = tb_ctx  [cell]
    f_ctx  = fo_ctx  [cell]
    dXdt   = dXdt_buf[cell]
    jvp    = jvp_out [cell]

    pack_elementary_context_packaged(Tc, Cc, elem_react_arrays, thermo_arrays, e_ctx)
    pack_threebody_context_packaged (Tc, Cc, tb_react_arrays,   thermo_arrays, t_ctx)
    pack_falloff_context_packaged   (Tc, Cc, fo_react_arrays,   thermo_arrays, f_ctx)

    constant_volume_reactor_rhs(
        Tc, Cc,
        e_ctx, elem_react_arrays,
        t_ctx, tb_react_arrays,
        f_ctx, fo_react_arrays,
        thermo_arrays, dXdt,
    )
    constant_volume_reactor_jvp(
        Tc, Cc, vTc, vCc,
        e_ctx, elem_react_arrays,
        t_ctx, tb_react_arrays,
        f_ctx, fo_react_arrays,
        thermo_arrays,
        dXdt, jvp,
    )


# =============================================================================
#  UTILITAIRES
# =============================================================================

def rel_err(a: np.ndarray, b: np.ndarray) -> dict:
    """Erreur relative |a-b| / (|b| + 1e-300)."""
    e = np.abs(a - b) / (np.abs(b) + 1e-300)
    return dict(min=e.min(), max=e.max(), mean=e.mean(), p99=np.percentile(e, 99))


def fmt(d: dict) -> str:
    return (f"min={d['min']:.2e}  max={d['max']:.2e}  "
            f"mean={d['mean']:.2e}  p99={d['p99']:.2e}")


def alloc_ctx():
    return (
        cuda.to_device(np.zeros((NCELLS, N_CTX_ELEM, nreac_elem), dtype=np.float64)),
        cuda.to_device(np.zeros((NCELLS, N_CTX_TB,   nreac_tb),   dtype=np.float64)),
        cuda.to_device(np.zeros((NCELLS, N_CTX_FO,   nreac_fo),   dtype=np.float64)),
    )


# =============================================================================
#  TEST 1 — COMPILATION + RHS
# =============================================================================

print("\n── Test 1 : compilation + RHS ───────────────────────────────────────────")

e_ctx, t_ctx, f_ctx = alloc_ctx()
dXdt_d  = cuda.to_device(np.zeros((NCELLS, nspecies + 1), dtype=np.float64))

t0 = time.perf_counter()
kernel_rhs[GRID, THREADS](
    T_d, C_d,
    elem_arrays, tb_arrays, fo_arrays,
    thermo_arrays,
    e_ctx, t_ctx, f_ctx,
    dXdt_d,
)
cuda.synchronize()
t_compile = time.perf_counter() - t0
print(f"  Compile + 1er run : {t_compile:.4f} s  → ✓ Compilation OK")

# 2e run (runtime seul)
t0 = time.perf_counter()
kernel_rhs[GRID, THREADS](
    T_d, C_d,
    elem_arrays, tb_arrays, fo_arrays,
    thermo_arrays,
    e_ctx, t_ctx, f_ctx,
    dXdt_d,
)
cuda.synchronize()
print(f"  Runtime seul      : {time.perf_counter() - t0:.6f} s")

dXdt_np = dXdt_d.copy_to_host()  # (NCELLS, nsp+1)

# ── Référence torch : rhs_adiabatic ──────────────────────────────────────────
# On reconstruit le RHS torch manuellement (même logique que le device fn).
def rhs_torch(T, C):
    """Référence : RHS réacteur adiabatique à volume constant (torch)."""
    ctx_e = batched_elem.evaluate_context(T, C)
    ctx_t = batched_tb  .evaluate_context(T, C)
    ctx_f = batched_falloff.evaluate_context(T, C)

    omega = (batched_elem   .omega_from_RR(ctx_e.RR)
           + batched_tb     .omega_from_RR(ctx_t.RR)
           + batched_falloff.omega_from_RR(ctx_f.RR))   # (ncells, nsp)

    n_tot = C.sum(dim=1, keepdim=True)                   # (ncells, 1)
    X     = C / n_tot                                    # fractions molaires

    # cp_mol_i via torch (si exposé), sinon on accepte petite approximation
    # Ici on utilise la même logique que le code Numba.
    # ──  cp_mol_mix = Σ X[i] * cp_mol_i  ──
    # On suppose que clas_th_object fournit cp_mol(T) → (ncells, nsp)
    cp_mol = clas_th_object.cp_mol(T)                    # (ncells, nsp)
    cv_mol = cp_mol - constants.R
    cv_mix = (X * cv_mol).sum(dim=1)                     # (ncells,)

    # u_mol_i
    u_mol  = clas_th_object.u_mol(T)                     # (ncells, nsp)
    dUdt   = (u_mol * omega).sum(dim=1)                  # (ncells,)

    dTdt   = -dUdt / (n_tot.squeeze(1) * cv_mix)

    return torch.cat([omega, dTdt.unsqueeze(1)], dim=1)  # (ncells, nsp+1)

t0 = time.perf_counter()
rhs_ref = rhs_torch(T_t, C_t)
print(f"  Torch runtime     : {time.perf_counter() - t0:.4f} s")

ref_np = rhs_ref.cpu().numpy()

# ── Erreurs ───────────────────────────────────────────────────────────────────
err_dCdt = rel_err(dXdt_np[:, :nspecies], ref_np[:, :nspecies])
err_dTdt = rel_err(dXdt_np[:, nspecies:], ref_np[:, nspecies:])

print(f"\n  dCdt  erreur relative :  {fmt(err_dCdt)}")
print(f"  dTdt  erreur relative :  {fmt(err_dTdt)}")

ok_rhs = err_dCdt['max'] < TOL_RHS and err_dTdt['max'] < TOL_RHS
print(f"\n  → RHS {'✓ OK' if ok_rhs else '✗ ÉCHEC'}"
      f"  (tolérance {TOL_RHS:.0e})")

# =============================================================================
#  TEST 2 — JVP vs différences finies
# =============================================================================

print("\n── Test 2 : JVP vs différences finies ───────────────────────────────────")

e_ctx, t_ctx, f_ctx = alloc_ctx()
dXdt_buf = cuda.to_device(np.zeros((NCELLS, nspecies + 1), dtype=np.float64))
jvp_d    = cuda.to_device(np.zeros((NCELLS, nspecies + 1), dtype=np.float64))

t0 = time.perf_counter()
kernel_jvp[GRID, THREADS](
    T_d, C_d, vT_d, vC_d,
    elem_arrays, tb_arrays, fo_arrays,
    thermo_arrays,
    e_ctx, t_ctx, f_ctx,
    dXdt_buf, jvp_d,
)
cuda.synchronize()
print(f"  Compile + 1er run : {time.perf_counter() - t0:.4f} s  → ✓ Compilation OK")

t0 = time.perf_counter()
kernel_jvp[GRID, THREADS](
    T_d, C_d, vT_d, vC_d,
    elem_arrays, tb_arrays, fo_arrays,
    thermo_arrays,
    e_ctx, t_ctx, f_ctx,
    dXdt_buf, jvp_d,
)
cuda.synchronize()
print(f"  Runtime seul      : {time.perf_counter() - t0:.6f} s")

jvp_np = jvp_d.copy_to_host()

# ── Référence : différences finies (torch) ────────────────────────────────────
vTC_t = torch.cat([vT_t.unsqueeze(1), vC_t], dim=1)  # (ncells, nsp+1)
TC_t  = torch.cat([T_t .unsqueeze(1), C_t ], dim=1)

TC_plus  = TC_t + EPS_FD * vTC_t
TC_minus = TC_t - EPS_FD * vTC_t

T_plus  = TC_plus [:, 0];  C_plus  = TC_plus [:, 1:]
T_minus = TC_minus[:, 0];  C_minus = TC_minus[:, 1:]

rhs_plus  = rhs_torch(T_plus,  C_plus ).cpu().numpy()
rhs_minus = rhs_torch(T_minus, C_minus).cpu().numpy()

jvp_fd = (rhs_plus - rhs_minus) / (2.0 * EPS_FD)   # (ncells, nsp+1)

# ── Erreurs ───────────────────────────────────────────────────────────────────
err_jvp_C = rel_err(jvp_np[:, :nspecies], jvp_fd[:, :nspecies])
err_jvp_T = rel_err(jvp_np[:, nspecies:], jvp_fd[:, nspecies:])

print(f"\n  jvp(dCdt)  erreur relative : {fmt(err_jvp_C)}")
print(f"  jvp(dTdt)  erreur relative : {fmt(err_jvp_T)}")

ok_jvp = err_jvp_C['max'] < TOL_JVP and err_jvp_T['max'] < TOL_JVP
print(f"\n  → JVP {'✓ OK' if ok_jvp else '✗ ÉCHEC'}"
      f"  (tolérance {TOL_JVP:.0e})")

# =============================================================================
#  TEST 3 — SANITY CHECKS physiques
# =============================================================================

print("\n── Test 3 : sanity checks physiques ────────────────────────────────────")

# dTdt doit être un scalaire fini par cellule
dTdt_np = dXdt_np[:, nspecies]
assert np.all(np.isfinite(dTdt_np)), "dTdt contient des NaN/Inf !"
print(f"  dTdt  ∈ [{dTdt_np.min():.3e}, {dTdt_np.max():.3e}]  K/s  → ✓ fini")

# Conservation de la masse : Σ dCdt * M[i] ≈ 0  (pas accessible ici, on vérifie Σ omega ≠ 0 juste)
sum_omega = dXdt_np[:, :nspecies].sum(axis=1)
print(f"  Σ_i dCdt[i]  ∈ [{sum_omega.min():.3e}, {sum_omega.max():.3e}]  mol/m³/s")

# JVP linéarité : JVP(alpha*v) ≈ alpha * JVP(v) — testé sur alpha = 2
print("\n  Test linéarité JVP (alpha=2) …")
alpha = 2.0
vTC2    = alpha * vTC
vT2_d   = cuda.to_device(np.ascontiguousarray(vTC2[:,  0], dtype=np.float64))
vC2_d   = cuda.to_device(np.ascontiguousarray(vTC2[:, 1:], dtype=np.float64))
jvp2_d  = cuda.to_device(np.zeros((NCELLS, nspecies + 1), dtype=np.float64))
dXdt2_d = cuda.to_device(np.zeros((NCELLS, nspecies + 1), dtype=np.float64))

kernel_jvp[GRID, THREADS](
    T_d, C_d, vT2_d, vC2_d,
    elem_arrays, tb_arrays, fo_arrays,
    thermo_arrays,
    e_ctx, t_ctx, f_ctx,
    dXdt2_d, jvp2_d,
)
cuda.synchronize()

jvp2_np = jvp2_d.copy_to_host()
lin_err  = rel_err(jvp2_np, alpha * jvp_np)
print(f"  Erreur relative JVP(2v) vs 2·JVP(v) : {fmt(lin_err)}")
ok_lin = lin_err['max'] < 1e-10
print(f"  → Linéarité {'✓ OK' if ok_lin else '✗ ÉCHEC'}")

# =============================================================================
#  RÉSUMÉ
# =============================================================================

print("\n" + "=" * 65)
print("  RÉSUMÉ")
print("=" * 65)
all_ok = ok_rhs and ok_jvp and ok_lin
results = [
    ("RHS vs torch",               ok_rhs),
    ("JVP vs différences finies",  ok_jvp),
    ("Linéarité JVP",              ok_lin),
]
for name, ok in results:
    print(f"  {'✓' if ok else '✗'}  {name}")

print("=" * 65)
print(f"  {'✓ TOUS LES TESTS PASSENT' if all_ok else '✗ DES TESTS ONT ÉCHOUÉ'}")
print("=" * 65)