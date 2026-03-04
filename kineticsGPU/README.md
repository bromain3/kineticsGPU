# numba_kinetics — GPU Chemical Kinetics Integrator (Numba CUDA)

A research-grade, GPU-accelerated backward-Euler integrator for stiff
chemical kinetics ODEs, built on **Numba CUDA** kernels.

## Architecture

```
numba_kinetics/
├── __init__.py
├── packing.py              # Torch → Numba device-array bridge (zero-copy)
├── integrator.py           # NumbaSolver  (backward Euler + GMRES)
├── benchmark.py            # Strategy comparison + divergence analysis
└── kernels/
    ├── __init__.py
    ├── rhs_jvp.py           # RHS and analytic JVP for all reaction types
    ├── gmres.py             # Batched GMRES with Givens rotations
    └── thermo_cv.py         # dT/dt and its JVP (adiabatic CV reactor)
```

## Two parallelization strategies

| Feature | `per_thread` | `per_block` |
|---------|-------------|-------------|
| Launch config | `<<<ceil(N/TPB), TPB>>>` | `<<<N, TPB>>>` |
| 1 GPU thread handles | 1 cell | reactions in 1 cell |
| Concurrency | ncells × nreac threads | ncells blocks |
| Shared memory | None | Yes (dot products / norm) |
| Thread divergence | Higher (Newton iters differ) | Lower (warp executes same path) |
| Register pressure | Lower | Higher |
| Best for | `ncells ≫ nspecies` | Large mechanisms |

Recommendation from the cited paper (Niemeyer & Sung 2014):
> per-thread outperforms per-block even in best case (identical ICs) when
> parallelism over cells is sufficient.  Use per-block for very large mechanisms
> (> 500 species) where per-thread register spill becomes a bottleneck.

## Key design choices

### No CPU↔GPU transfers in the integration loop
Torch tensors and Numba device arrays share the same CUDA memory via
`__cuda_array_interface__`.  The only host-side operation inside Newton
iterations is reading a **single scalar** (max residual norm) via CuPy
to check convergence.

### Analytic JVP (not autograd)
All three reaction types implement hand-derived Jacobian-vector products
(see equations in `kernels/rhs_jvp.py` docstrings).  This avoids the
overhead of autograd tape construction and is directly compatible with
`torch.compile` / `torch.vmap`.

### GMRES with Givens rotations
Implements restarted GMRES(m) without `lstsq` (back-substitution only).
The Krylov basis `V` is stored in global memory `(ncells, m+1, N)`.
For per-block, dot products in the Arnoldi step use shared-memory
parallel reduction (O(log TPB) steps).

### Workspace pre-allocation
All temporary arrays (`V`, `H`, `cs`, `sn`, `g_vec`, `F`, `Av`) are
allocated once in `NumbaSolver.__init__` and reused across steps.

## Usage

```python
from numba_kinetics import NumbaPacked, NumbaSolver
import kineticsGPU.thermo as yaml_thermo
from kineticsGPU.reactionParser import parseReactions, build_reactions
from kineticsGPU.reactionTypesContext import (
    ElementaryReactionBatchContext,
    ThreeBodyReactionBatchContext,
    FalloffReactionBatchContext,
)

# 1. Build reaction system (same as Torch version)
y        = yaml_thermo.yamlParser('gri30.yaml')
vec_th   = y.populateVectorizedThermoObject(dtype=torch.float64, device='cuda')
reactions = parseReactions('gri30.yaml')
r_e, r_tb, r_f, _ = build_reactions(vec_th.species, reactions, vec_th, 'cuda', torch.float64)

# 2. Pack to device (ONE TIME — no copy after this)
packed = NumbaPacked(
    vec_th,
    ElementaryReactionBatchContext(r_e,  device='cuda', dtype=torch.float64),
    ThreeBodyReactionBatchContext(r_tb, device='cuda', dtype=torch.float64),
    FalloffReactionBatchContext(r_f,   device='cuda', dtype=torch.float64),
)

# 3. Create solver
solver = NumbaSolver(packed, strategy='per_thread', tpb=64)
# or:   NumbaSolver(packed, strategy='per_block',  tpb=64)

# 4. Integrate
# TC_torch shape: (ncells, 1 + n_species),  columns = [T, C_0, ..., C_{ns-1}]
TC_new = solver.step(TC_torch, t=0.0, h=1e-7)
```

## Benchmark

```bash
python benchmark.py --yaml gri30.yaml --ncells 1000 --nsteps 100 --h 1e-7

# Thread-divergence sweep
python benchmark.py --yaml gri30.yaml --ncells 1000 --nsteps 10 \
                    --divergence-analysis
```

## Dependencies

```
numba >= 0.59
cupy  >= 12.0      (for single-scalar max norm in Newton loop)
numpy
torch (CUDA build)
```

## Known limitations & TODOs

- `per_block` RHS for three-body and falloff reactions: the stubs in
  `rhs_jvp.py` dispatch to per-thread kernels.  Full per-block versions
  should split reactions across threads with `for i in range(tid, nreac, TPB)`.
  The elementary per-block kernel (`pb_rhs_elementary`) is complete and
  serves as the template.

- The single `cp.max(cp.asarray(...))` call inside the Newton convergence
  check is the only CPU↔GPU synchronisation in the hot loop.  It can be
  eliminated by keeping a convergence flag on device and checking it via a
  tiny kernel; this is left as a future optimisation.

- `torch.compile` compatibility: the Numba kernels are called from Python
  and are not yet wrapped in a `torch.autograd.Function`, so they are
  opaque to `torch.compile`.  For full graph capture, wrap each kernel
  dispatch in a `triton` kernel or use `cupy.RawKernel`.

- Adaptive time-stepping (step-size control based on local error estimate)
  is not implemented.  The backward-Euler method is first-order; for
  production use consider BDF2 or SDIRK2.