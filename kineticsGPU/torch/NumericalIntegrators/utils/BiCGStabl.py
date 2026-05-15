import torch
import time
import math
from typing import Callable, Optional, Dict, Tuple


def bicgstab_l_batched(
        matvec: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        m_inv: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        l: int = 2,
        tol: float = 1e-6,
        maxiter: int = 50,
        verbose: bool = False,
        profile: bool = False,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Batched BiCGStab(l) solver optimized for GPU.

    BiCGStab(l) is a stabilized version of BiCG that performs l BiCG steps
    followed by a polynomial stabilization (MR-like minimization).
    Often more efficient than GMRES for non-symmetric systems.

    Args:
        matvec: Callable(B,N)->(B,N). Vectorized matrix-vector product.
        rhs: (B, N) right-hand side vectors
        m_inv: Preconditioner, defaults to identity
        l: Degree of polynomial (l=1 is BiCGStab, l=2 is BiCGStab(2), etc.)
        tol: Convergence tolerance
        maxiter: Maximum number of iterations
        verbose: Print convergence info
        profile: Return timing statistics
        device: Device to use

    Returns:
        (x, timings_dict) if profile=True, else (x, None)

    Memory: Detaches intermediate results to avoid autograd overhead.
    """

    if device is None:
        device = rhs.device

    B, N = rhs.shape
    dtype = rhs.dtype

    # Initialize solution
    x = torch.zeros_like(rhs, device=device, dtype=dtype)

    # Initial residual
    r = (rhs - matvec(x).detach()).detach()
    r0_hat = r.clone()  # Shadow residual (arbitrary choice)

    # Check initial convergence
    res_norms = torch.norm(r, dim=1)
    rhs_norms = torch.norm(rhs, dim=1)
    rhs_norms = torch.where(rhs_norms == 0, torch.ones_like(rhs_norms), rhs_norms)

    if res_norms.max().item() / rhs_norms.max().item() < tol:
        if verbose:
            print(f"[BiCGStab({l})] Converged at initialization")
        return (x, None) if not profile else (x, {})

    # Storage for r_i vectors (l+1 residuals)
    r_vecs = torch.zeros((B, l + 1, N), device=device, dtype=dtype)
    # Storage for u_i vectors (search directions)
    u_vecs = torch.zeros((B, l + 1, N), device=device, dtype=dtype)

    # Timing accumulators
    timers = {
        "matvec_call": 0,
        "matvec": 0.0,
        "m_inv": 0.0,
        "bicg_step": 0.0,
        "gmres_step": 0.0,
        "update": 0.0,
        "total": 0.0,
    }
    start_total = time.time()

    rho_prev = torch.ones(B, device=device, dtype=dtype)
    alpha = torch.zeros(B, device=device, dtype=dtype)
    omega = torch.ones(B, device=device, dtype=dtype)

    for iter_count in range(maxiter):
        # Store r_0 = r
        r_vecs[:, 0, :] = r

        # --- BiCG part: build Krylov subspace ---
        t0 = time.time()

        # rho = <r0_hat, r>
        rho = torch.einsum("bn,bn->b", r0_hat, r)

        # Check for breakdown
        if (torch.abs(rho) < 1e-30).any():
            if verbose:
                print(f"[BiCGStab({l})] BiCG breakdown at iteration {iter_count}")
            break

        # beta = (rho / rho_prev) * (alpha / omega)
        if iter_count > 0:
            beta = (rho / rho_prev) * (alpha / omega)
            # u_0 = r - beta * omega * u_0
            u_vecs[:, 0, :] = r - beta.unsqueeze(1) * omega.unsqueeze(1) * u_vecs[:, 0, :]
        else:
            u_vecs[:, 0, :] = r.clone()

        # Build l BiCG steps
        for j in range(l):
            # u_{j+1} = A * M^{-1} * u_j
            t_mv = time.time()
            u_prec = m_inv(u_vecs[:, j, :]).detach()
            timers["m_inv"] += (time.time() - t_mv)

            t_mv = time.time()
            u_vecs[:, j + 1, :] = matvec(u_prec).detach()
            timers["matvec_call"] += 1
            timers["matvec"] += (time.time() - t_mv)

            # alpha = rho / <r0_hat, u_{j+1}>
            sigma = torch.einsum("bn,bn->b", r0_hat, u_vecs[:, j + 1, :])

            # Check for breakdown
            if (torch.abs(sigma) < 1e-30).any():
                if verbose:
                    print(f"[BiCGStab({l})] Breakdown in BiCG at j={j}, iter={iter_count}")
                # Use previous alpha or set to small value
                if j == 0 and iter_count == 0:
                    alpha = torch.ones_like(sigma) * 1e-10
                # else keep previous alpha
            else:
                alpha = rho / sigma

            # r_j = r_{j-1} - alpha * u_{j+1}
            # Note: r_vecs[:, 0, :] already set to r
            if j == 0:
                r_vecs[:, 1, :] = r_vecs[:, 0, :] - alpha.unsqueeze(1) * u_vecs[:, 1, :]
            else:
                r_vecs[:, j + 1, :] = r_vecs[:, j, :] - alpha.unsqueeze(1) * u_vecs[:, j + 1, :]

        timers["bicg_step"] += (time.time() - t0)

        # --- MR part: minimize residual over generated subspace ---
        t0 = time.time()

        # Compute tau matrix: tau[i,j] = <r_i, r_j> for i,j = 1..l
        # This is a small (l x l) system per batch element
        tau = torch.zeros((B, l, l), device=device, dtype=dtype)
        for i in range(1, l + 1):
            for j in range(i, l + 1):  # Symmetric
                tau[:, i - 1, j - 1] = torch.einsum("bn,bn->b", r_vecs[:, i, :], r_vecs[:, j, :])
                if i != j:
                    tau[:, j - 1, i - 1] = tau[:, i - 1, j - 1]

        # sigma vector: sigma[i] = <r_i, r_0> for i = 1..l
        sigma_vec = torch.zeros((B, l), device=device, dtype=dtype)
        for i in range(1, l + 1):
            sigma_vec[:, i - 1] = torch.einsum("bn,bn->b", r_vecs[:, i, :], r_vecs[:, 0, :])

        # Solve tau @ gamma = sigma for gamma
        # Use torch.linalg.solve which is batched
        try:
            # Add small regularization for stability
            tau_reg = tau + 1e-12 * torch.eye(l, device=device, dtype=dtype).unsqueeze(0)
            gamma = torch.linalg.solve(tau_reg, sigma_vec.unsqueeze(-1)).squeeze(-1)  # (B, l)
        except:
            # Fallback: use pseudo-inverse if singular
            gamma = torch.linalg.lstsq(tau, sigma_vec.unsqueeze(-1)).solution.squeeze(-1)

        # gamma'[0] = gamma[0], gamma'[i] = gamma[i] - gamma[i-1] for i > 0
        gamma_prime = gamma.clone()
        for i in range(1, l):
            gamma_prime[:, i] = gamma[:, i] - gamma[:, i - 1]

        timers["gmres_step"] += (time.time() - t0)

        # --- Update solution and residual ---
        t0 = time.time()

        # x = x + gamma'[0] * M^{-1} * u_0 + sum_{i=1}^{l-1} gamma'[i] * M^{-1} * u_i + gamma[l-1] * M^{-1} * r_l
        delta = torch.zeros_like(x)

        # First term
        u0_prec = m_inv(u_vecs[:, 0, :]).detach()
        delta += gamma_prime[:, 0].unsqueeze(1) * u0_prec

        # Middle terms
        for i in range(1, l):
            ui_prec = m_inv(u_vecs[:, i, :]).detach()
            delta += gamma_prime[:, i].unsqueeze(1) * ui_prec

        # Last term
        rl_prec = m_inv(r_vecs[:, l, :]).detach()
        delta += gamma[:, l - 1].unsqueeze(1) * rl_prec

        x = (x + delta).detach()

        # r = r_0 - sum_{i=1}^l gamma'[i-1] * r_i
        r = r_vecs[:, 0, :].clone()
        for i in range(1, l + 1):
            r -= gamma_prime[:, i - 1].unsqueeze(1) * r_vecs[:, i, :]
        r = r.detach()

        # u_0 = u_0 - sum_{i=1}^l gamma'[i-1] * u_i
        u_new = u_vecs[:, 0, :].clone()
        for i in range(1, l + 1):
            u_new -= gamma_prime[:, i - 1].unsqueeze(1) * u_vecs[:, i, :]
        u_vecs[:, 0, :] = u_new.detach()

        timers["update"] += (time.time() - t0)

        # Check convergence
        res_norms = torch.norm(r, dim=1)
        rel_res = res_norms / rhs_norms

        if verbose:
            print(f"[BiCGStab({l})] Iter {iter_count}: max residual = {res_norms.max().item():.3e}, "
                  f"max rel_res = {rel_res.max().item():.3e}")

        if rel_res.max().item() < tol:
            if verbose:
                print(f"[BiCGStab({l})] Converged at iteration {iter_count}")
            break

        # omega = gamma[l-1] for next iteration
        omega = gamma[:, l - 1]
        rho_prev = rho

    timers["total"] = time.time() - start_total

    print(timers)

    return x