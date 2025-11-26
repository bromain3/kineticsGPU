# Batched restarted GMRES solver.
from typing import Callable, Tuple, Optional, Dict
import math
import torch

import time


def gmres_solve(matvec: Callable[[torch.Tensor], torch.Tensor],
                rhs: torch.Tensor,
                m_inv: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                restart: int = 20,
                tol: float = 1e-6,
                maxiter: int = 50,
                verbose: bool = False) -> torch.Tensor:
    B, N = rhs.shape
    device = rhs.device
    dtype = rhs.dtype

    t_ini = time.time()

    matvec_time = []
    lst_square_time = []

    x_sol = torch.zeros_like(rhs, device=device, dtype=dtype)

    t0 = time.time()
    r = (rhs - matvec(x_sol)).detach()
    matvec_time.append(time.time() - t0)

    z = m_inv(r).detach()
    beta = torch.norm(z, dim=1)
    beta = torch.where(beta == 0, torch.tensor(1.0, device=device, dtype=dtype), beta)


    max_restarts = math.ceil(maxiter / restart)
    remaining_iters = maxiter


    m_max = min(restart, remaining_iters)
    V = torch.zeros((B, m_max + 1, N), device=device, dtype=dtype)
    H = torch.zeros((B, m_max + 1, m_max), device=device, dtype=dtype)
    g = torch.zeros((B, m_max + 1), device=device, dtype=dtype)


    for restart_idx in range(max_restarts):
        m = min(restart, remaining_iters)
        V[:, :m+1, :].zero_()
        H[:, :m+1, :m].zero_()
        g[:, :m+1].zero_()


        v0 = (z / beta.unsqueeze(1)).detach()
        V[:, 0, :] = v0
        g[:, 0] = beta


        for j in range(m):
            vj = V[:, j, :].detach()

            t0 = time.time()
            w = matvec(vj).detach()
            matvec_time.append(time.time() - t0)

            w = m_inv(w).detach()


            for i in range(j + 1):
                vi = V[:, i, :].detach()
                hij = torch.sum(w * vi, dim=1)
                H[:, i, j] = hij
                w = w - hij.unsqueeze(1) * vi


            hj1 = torch.norm(w, dim=1)
            H[:, j + 1, j] = hj1
            nonzero = hj1 != 0
            if nonzero.any():
                V[nonzero, j + 1, :] = (w[nonzero] / hj1[nonzero].unsqueeze(1))
            else:
                V[:, j + 1, :].zero_()


        y = torch.zeros((B, m), device=device, dtype=dtype)
        t0 = time.time()
        for b in range(B):
            Hb = H[b, :m+1, :m].detach()
            gb = g[b, :m+1].detach()
            sol = torch.linalg.lstsq(Hb, gb.unsqueeze(1)).solution.squeeze(1)
            y[b, :] = sol
        lst_square_time.append(time.time() - t0)

        V_nm = V[:, :m, :].permute(0, 2, 1).detach()
        delta = torch.matmul(V_nm, y.unsqueeze(-1)).squeeze(-1).detach()
        x_sol = (x_sol + delta).detach()

        t0 = time.time()
        r = (rhs - matvec(x_sol)).detach()
        matvec_time.append(time.time() - t0)

        res_norms = torch.norm(r, dim=1)
        if verbose:
            print(f"[GMRES restart {restart_idx}] max residual = {res_norms.max().item():.3e}")
        if res_norms.max().item() < tol:
            break


        z = m_inv(r).detach()
        beta = torch.norm(z, dim=1)
        beta = torch.where(beta == 0, torch.tensor(1.0, device=device, dtype=dtype), beta)
        remaining_iters -= m
        if remaining_iters <= 0:
            break

    print('total time:', time.time() - t_ini)
    print('matvec time:', sum(matvec_time))
    print('lstsquare time:', sum(lst_square_time))
    return x_sol


def gmres_solve_batched(matvec: Callable[[torch.Tensor], torch.Tensor],
                        rhs: torch.Tensor,
                        m_inv: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                        restart: int = 20,
                        tol: float = 1e-6,
                        maxiter: int = 50,
                        verbose: bool = False,
                        ):
    B, N = rhs.shape
    device = rhs.device
    dtype = rhs.dtype

    t_ini = time.time()

    matvec_time = []
    lst_square_time = []

    # Solution vector
    x = torch.zeros_like(rhs, device=device, dtype=dtype)

    # Initial residual
    t0 = time.time()
    r = (rhs - matvec(x)).detach()
    matvec_time.append(time.time() - t0)

    z = m_inv(r).detach()
    beta = torch.norm(z, dim=1)
    beta = torch.where(beta == 0, torch.ones_like(beta), beta)

    # Preallocation
    m_max = min(restart, maxiter)
    V = torch.zeros((B, m_max + 1, N), device=device, dtype=dtype)
    H = torch.zeros((B, m_max + 1, m_max), device=device, dtype=dtype)
    g = torch.zeros((B, m_max + 1), device=device, dtype=dtype)

    remaining_iters = maxiter
    max_restarts = math.ceil(maxiter / restart)

    for restart_idx in range(max_restarts):

        m = min(restart, remaining_iters)
        V[:, :m+1].zero_()
        H[:, :m+1, :m].zero_()
        g[:, :m+1].zero_()

        # Normalize v0
        V[:, 0, :] = (z / beta.unsqueeze(1)).detach()
        g[:, 0] = beta

        # Arnoldi
        for j in range(m):
            vj = V[:, j, :].detach()

            # Important: detach matvec output immediately
            t0 = time.time()
            w = matvec(vj).detach()
            matvec_time.append(time.time() - t0)

            w = m_inv(w).detach()

            # Compute projections via einsum
            hij = torch.einsum("bn,bkn->bk", w, V[:, :j+1, :])
            H[:, :j+1, j] = hij.detach()

            # Remove projections
            w = (w - torch.einsum("bk,bkn->bn", hij, V[:, :j+1, :])).detach()

            hj1 = torch.norm(w, dim=1)
            H[:, j+1, j] = hj1.detach()

            # Normalize new basis vector
            mask = hj1 > 0
            if mask.any():
                V[mask, j+1, :] = (w[mask] / hj1[mask].unsqueeze(1))

        # Solve batched least squares
        Hb = H[:, :m+1, :m].detach()
        gb = g[:, :m+1].unsqueeze(-1).detach()

        t0 = time.time()
        y = torch.linalg.lstsq(Hb, gb).solution.squeeze(-1).detach()
        lst_square_time.append(time.time() - t0)

        # Update x
        delta = torch.einsum("bkn,bk->bn", V[:, :m, :], y).detach()
        x = (x + delta).detach()

        # Residual
        t0 = time.time()
        r = (rhs - matvec(x)).detach()
        matvec_time.append(time.time() - t0)

        res_norm = torch.norm(r, dim=1)

        if verbose:
            print(f"[GMRES restart {restart_idx}] max residual = {res_norm.max().item():.3e}")

        if res_norm.max() < tol:
            break

        # Prepare next restart
        z = m_inv(r).detach()
        beta = torch.norm(z, dim=1)
        beta = torch.where(beta == 0, torch.ones_like(beta), beta).detach()

        remaining_iters -= m
        if remaining_iters <= 0:
            break
    print('total time:', time.time() - t_ini)
    print('matvec time:', sum(matvec_time))
    print('lstsquare time:', sum(lst_square_time))

    return x


def gmres_givens_batched( #TODO: add a functionality to give specific indexes to jvp and rhs so that the solving only occurs for unconverged reactors
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    m_inv: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    restart: int = 20,
    tol: float = 1e-6,
    maxiter: int = 50,
    verbose: bool = False,
    profile: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Batched restarted GMRES using incremental Givens rotations (no lstsq).
    - matvec: Callable(B,N)->(B,N). IMPORTANT: matvec should be vectorized on batch dim.
    - rhs: (B, N)
    Returns: (x, timings_dict) if profile=True, else (x, None)
    Memory: detaches matvec/m_inv outputs to avoid building huge autograd graphs.
    """

    if device is None:
        device = rhs.device

    B, N = rhs.shape
    dtype = rhs.dtype

    x = torch.zeros_like(rhs, device=device, dtype=dtype)  # solution
    r = (rhs - matvec(x).detach()).detach()
    z = m_inv(r).detach()
    beta = torch.norm(z, dim=1)
    beta = torch.where(beta == 0, torch.ones_like(beta), beta)

    m_max = min(restart, maxiter)
    V = torch.zeros((B, m_max + 1, N), device=device, dtype=dtype)     # basis
    H = torch.zeros((B, m_max + 1, m_max), device=device, dtype=dtype) # upper-Hessenberg
    cs = torch.zeros((B, m_max), device=device, dtype=dtype)           # cosines of Givens
    sn = torch.zeros((B, m_max), device=device, dtype=dtype)           # sines of Givens
    g = torch.zeros((B, m_max + 1), device=device, dtype=dtype)       # right-hand side (rotated)

    remaining_iters = maxiter
    max_restarts = math.ceil(maxiter / restart)

    # timing accumulators
    timers = {
        "matvec_call": 0,
        "matvec": 0.0,
        "m_inv": 0.0,
        "proj": 0.0,
        "givens": 0.0,
        "backsolve": 0.0,
        "update": 0.0,
        "residual_matvec": 0.0,
        "total": 0.0,
    }
    start_total = time.time()

    for restart_idx in range(max_restarts):
        m = min(restart, remaining_iters)
        V[:, :m+1, :].zero_()
        H[:, :m+1, :m].zero_()
        cs[:, :m].zero_()
        sn[:, :m].zero_()
        g[:, :m+1].zero_()

        V[:, 0, :] = (z / beta.unsqueeze(1))
        g[:, 0] = beta

        # Arnoldi + Givens applied incrementally
        for j in range(m):
            # matvec on current basis vector
            t0 = time.time()
            vj = V[:, j, :]

            w = matvec(vj).detach()
            timers["matvec_call"] += 1
            timers["matvec"] += (time.time() - t0)

            t0 = time.time()
            w = m_inv(w).detach()
            timers["m_inv"] += (time.time() - t0)

            # projection onto V[:, :j+1, :]
            t0 = time.time()
            if j + 1 > 0:
                hij = torch.einsum("bn,bkn->bk", w, V[:, :j+1, :])   # (B, j+1)
                H[:, :j+1, j] = hij
                w = (w - torch.einsum("bk,bkn->bn", hij, V[:, :j+1, :])).detach()
            timers["proj"] += (time.time() - t0)

            # norm and new basis vector
            hj1 = torch.norm(w, dim=1)
            H[:, j+1, j] = hj1
            mask = hj1 > 0
            if mask.any():
                V[mask, j+1, :] = (w[mask] / hj1[mask].unsqueeze(1))

            # Apply previous Givens rotations to the new column H[:, :j+2, j]
            t0 = time.time()
            # For i in 0..j-1 apply (cs_i, sn_i) to H[i:i+2, j]
            # vectorized loop over i
            for i in range(j):
                # temp = cs[:,i]*H[:,i,j] + sn[:,i]*H[:,i+1,j]
                hi = H[:, i, j]
                hi1 = H[:, i+1, j]
                c = cs[:, i]
                s = sn[:, i]
                temp = c * hi + s * hi1
                H[:, i+1, j] = -s * hi + c * hi1
                H[:, i, j] = temp

            # Now compute new Givens to eliminate H[j+1,j]
            h_jj = H[:, j, j]
            h_j1j = H[:, j+1, j]

            # r = sqrt(h_jj^2 + h_j1j^2), but avoid zero divisions
            r_diag = torch.sqrt(h_jj * h_jj + h_j1j * h_j1j)
            # handle r==0
            nonzero = r_diag != 0
            c_new = torch.ones_like(r_diag)
            s_new = torch.zeros_like(r_diag)
            c_new[nonzero] = h_jj[nonzero] / r_diag[nonzero]
            s_new[nonzero] = h_j1j[nonzero] / r_diag[nonzero]

            cs[:, j] = c_new
            sn[:, j] = s_new

            # apply rotation: [h_jj; h_j1j] := [r; 0]
            H[:, j, j] = r_diag
            H[:, j+1, j] = 0.0

            # rotate g similarly
            g_j = g[:, j].clone()
            g[:, j] = c_new * g_j
            g[:, j+1] = -s_new * g_j
            timers["givens"] += (time.time() - t0)

        # Solve R y = g_head where R is upper triangular of shape (m,m) stored in H[:, :m, :m]
        t0 = time.time()
        R = H[:, :m, :m]           # (B, m, m) upper triangular
        g_head = g[:, :m]         # (B, m)
        y = torch.zeros((B, m), device=device, dtype=dtype)

        # Back substitution batched
        # start from last row
        # ensure diagonal elements not zero (if zero, set to tiny to avoid NaN)
        diag = R[:, torch.arange(m), torch.arange(m)]
        diag_safe = torch.where(diag == 0, torch.full_like(diag, 1e-30), diag)

        # compute y[:, m-1]
        y[:, m - 1] = g_head[:, m - 1] / diag_safe[:, m - 1]
        if m - 2 >= 0:
            for k in range(m - 2, -1, -1):
                # y[:,k] = (g_head[:,k] - sum_{l>k} R[:,k,l]*y[:,l]) / R[:,k,k]
                # compute sum via einsum over l=k+1..m-1
                if k + 1 <= m - 1:
                    R_sub = R[:, k, k+1:m]            # (B, m-k-1)
                    y_sub = y[:, k+1:m]               # (B, m-k-1)
                    dot = torch.einsum("bk,bk->b", R_sub, y_sub)
                else:
                    dot = torch.zeros(B, device=device, dtype=dtype)
                y[:, k] = (g_head[:, k] - dot) / diag_safe[:, k]
        timers["backsolve"] += (time.time() - t0)

        # Update solution x += V[:, :m, :] @ y
        t0 = time.time()
        delta = torch.einsum("bkn,bk->bn", V[:, :m, :], y).detach()
        x = (x + delta).detach()
        timers["update"] += (time.time() - t0)

        # compute residual and test convergence
        t0 = time.time()
        r = (rhs - matvec(x).detach()).detach()
        timers["matvec_call"] += 1
        timers["residual_matvec"] += (time.time() - t0)

        res_norms = torch.norm(r, dim=1)
        if verbose:
            print(f"[GMRES-Givens restart {restart_idx}] max residual = {res_norms.max().item():.3e}")

        if res_norms.max().item() < tol:
            break

        # prepare next restart
        t0 = time.time()
        z = m_inv(r).detach()
        timers["m_inv"] += (time.time() - t0)
        beta = torch.norm(z, dim=1)
        beta = torch.where(beta == 0, torch.ones_like(beta), beta)
        remaining_iters -= m
        if remaining_iters <= 0:
            break

    timers["total"] = time.time() - start_total
    print(timers)
    return x
"""    if profile:
        # return timings summary
        return x, timers
    else:
        return x, None"""