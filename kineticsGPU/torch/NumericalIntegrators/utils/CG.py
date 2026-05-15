import time
import torch
from typing import Callable, Optional, Dict, Tuple

#TODO: validate the methode
@torch.no_grad()
def cg_batched(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    m_inv: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,  # preconditioner (B,N)->(B,N)
    tol: float = 1e-6,
    maxiter: int = 100,
    verbose: bool = False,
    profile: bool = False,
    device: Optional[torch.device] = None,
    atol: float = 0.0,   # absolute tolerance optional
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    """
    Batched Conjugate Gradient (CG) solver.
    - matvec: (B,N)->(B,N) vectorized on batch dimension.
    - rhs: (B,N)
    - m_inv: preconditioner function (B,N)->(B,N). Default identity.
    Returns (x, timings) if profile=True else (x, None).
    All heavy matvec/m_inv outputs are detached to avoid autograd memory blowup.
    """
    if device is None:
        device = rhs.device

    B, N = rhs.shape
    dtype = rhs.dtype

    # timers
    timers = {"matvec": 0.0, "m_inv": 0.0, "vector_ops": 0.0, "total": 0.0}
    t_start_total = time.time()

    # initial guess x = 0
    x = torch.zeros_like(rhs, device=device, dtype=dtype)

    # r0 = b - A x0
    t0 = time.time()
    Ax0 = matvec(x).detach()
    timers["matvec"] += time.time() - t0

    r = (rhs - Ax0).detach()

    # apply preconditioner z = M^{-1} r
    t0 = time.time()
    z = m_inv(r).detach()
    timers["m_inv"] += time.time() - t0

    p = z.clone()       # search direction
    # scalar products per-batch
    rz_old = torch.einsum("bn,bn->b", r, z)            # (B,)

    # norms and convergence
    r_norm = torch.norm(r, dim=1)                     # (B,)
    rhs_norm = torch.norm(rhs, dim=1)
    # avoid zero rhs_norm for relative tolerance
    rhs_norm_safe = torch.where(rhs_norm == 0, torch.ones_like(rhs_norm), rhs_norm)
    converged = (r_norm <= torch.maximum(atol * torch.ones_like(r_norm), tol * rhs_norm_safe))
    if verbose:
        print(f"[CG] initial max residual = {r_norm.max().item():.3e}")

    for it in range(maxiter):
        if converged.all():
            if verbose:
                print(f"[CG] all converged at iter {it}")
            break

        # Ap = A p
        t0 = time.time()
        Ap = matvec(p).detach()         # (B,N)
        timers["matvec"] += time.time() - t0

        # alpha = (r,z) / (p,Ap)
        t0 = time.time()
        pAp = torch.einsum("bn,bn->b", p, Ap)           # (B,)
        # safe divide: if denominator very small, set to tiny to avoid nan
        eps = torch.finfo(dtype).tiny
        denom = torch.where(pAp.abs() < eps, torch.full_like(pAp, eps), pAp)
        alpha = (rz_old / denom).detach()              # (B,)
        timers["vector_ops"] += time.time() - t0

        # mask updates for converged batches to avoid numerical noise
        mask = ~converged     # True for active batches

        # x = x + alpha * p
        t0 = time.time()
        x = x + (alpha.unsqueeze(1) * p)
        x = x.detach()
        timers["vector_ops"] += time.time() - t0

        # r = r - alpha * Ap
        t0 = time.time()
        r = r - (alpha.unsqueeze(1) * Ap)
        r = r.detach()
        timers["vector_ops"] += time.time() - t0

        # apply preconditioner z = M^{-1} r
        t0 = time.time()
        z = m_inv(r).detach()
        timers["m_inv"] += time.time() - t0

        # rz_new = (r, z)
        t0 = time.time()
        rz_new = torch.einsum("bn,bn->b", r, z).detach()
        timers["vector_ops"] += time.time() - t0

        # beta = rz_new / rz_old
        t0 = time.time()
        # avoid division by zero
        rz_old_safe = torch.where(rz_old.abs() < eps, torch.full_like(rz_old, eps), rz_old)
        beta = (rz_new / rz_old_safe).detach()
        timers["vector_ops"] += time.time() - t0

        # p = z + beta * p
        t0 = time.time()
        p = (z + beta.unsqueeze(1) * p).detach()
        timers["vector_ops"] += time.time() - t0

        rz_old = rz_new

        # update residual norm and convergence mask
        t0 = time.time()
        r_norm = torch.norm(r, dim=1)
        converged = (r_norm <= torch.maximum(atol * torch.ones_like(r_norm), tol * rhs_norm_safe))
        timers["vector_ops"] += time.time() - t0

        if verbose and (it % 10 == 0 or converged.all()):
            print(f"[CG] iter {it+1}, max residual = {r_norm.max().item():.3e}, #active = {mask.sum().item()}")

    timers["total"] = time.time() - t_start_total

    if profile:
        return x, timers
    else:
        return x, None
