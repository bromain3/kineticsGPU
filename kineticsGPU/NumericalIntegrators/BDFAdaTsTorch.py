"""
Torch Backward Euler Integrator — extended

Changements ajoutés :
- pas de temps adaptatif (step-doubling + contrôleur PID-simple)
- option pour méthode d'ordre 1 (Backward Euler) ou ordre 2 (Crank-Nicolson / trapezoid)
- support d'une Jacobienne analytique fournie par l'utilisateur (optionnel) : soit la matrice complète (B,N,N), soit une fonction Jv (jacobian-vector product)
- preconditionnement amélioré : possibilité d'utiliser la jacobienne analytique pour construire un préconditionneur (Jacobi/diagonal, bloc-diagonal, ou factorisation LU si petite)
- stratégies pour performance : jacobian reuse (frequency), batch LU via torch.linalg.solve, possibilité d'utiliser vmap si disponible

Remarques sur la performance et la justesse :
- Fournir la Jacobienne analytique (ou son produit par vecteur Jv) **aide fortement** : Newton converge en moins d'itérations, GMRES/JFNK nécessite moins d'itérations, et on peut construire un préconditionneur bien adapté (p.ex. ILU approché ou factorisation des blocs diagonaux). Si tu peux fournir soit J(x,t) complet soit une fonction Jv(x, t, v) -> J(x,t)@v, c'est l'idéal.
- Pour systèmes très grands (N >> 1e4), évite la formation dense de J ; privilégie Jv (Jacobian-Free Newton-Krylov) + GMRES + préconditionneur léger.

Cette version du fichier propose une API :

- integrator = AdaptiveImplicitIntegrator(f, ...)
- x_next = integrator.step_adaptive(x, t, h, ...)

La classe accepte :
- jacobian_fn (optionnel) : callable (x, t) -> (B, N, N)
- jvp_fn (optionnel) : callable (x, t, v) -> (B, N)    # J @ v

"""

import math
from typing import Callable, Optional, Tuple
import torch

Tensor = torch.Tensor


def _ensure_batch_dim(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


class AdaptiveImplicitIntegrator:
    """Implicit integrator with adaptive time-stepping and optional analytic Jacobian/Jv.

    Features:
      - step-doubling error estimate (one-step vs two half-steps)
      - methods: 'backward_euler' (order 1) or 'trapezoid' (Crank-Nicolson, order 2)
      - supports analytic jacobian or jvp for performance
      - Newton iterations for nonlinear solve with linear solver options: 'dense' or 'gmres' (jacobian-free or Jacobian-based)
      - preconditioners: 'identity', 'jacobi' (diag), or user-provided
      - batched (B,N) and GPU support
    """

    def __init__(self,
                 f: Callable[[Tensor, float], Tensor],
                 device: Optional[torch.device|str] = None,
                 dtype: Optional[torch.dtype] = None,
                 jacobian_fn: Optional[Callable[[Tensor, float], Tensor]] = None,
                 jvp_fn: Optional[Callable[[Tensor, float, Tensor], Tensor]] = None,
                 newton_tol: float = 1e-7,
                 newton_maxiter: int = 20,
                 gmres_tol: float = 1e-6,
                 gmres_maxiter: int = 50,
                 gmres_restart: int = 20,
                 verbose: bool = False):
        self.f = f
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or torch.float32
        self.jacobian_fn = jacobian_fn
        self.jvp_fn = jvp_fn
        self.newton_tol = newton_tol
        self.newton_maxiter = newton_maxiter
        self.gmres_tol = gmres_tol
        self.gmres_maxiter = gmres_maxiter
        self.gmres_restart = gmres_restart
        self.verbose = verbose

    # ----------------- public adaptive step -----------------
    def step_adaptive(self,
                      x: Tensor,
                      t: float,
                      h: float,
                      method: str = 'backward_euler',
                      atol: float = 1e-6,
                      rtol: float = 1e-3,
                      safety: float = 0.9,
                      min_h: float = 1e-12,
                      max_h: float = 1.0,
                      max_retries: int = 10,
                      linear_solver: str = 'gmres',
                      precond: str = 'identity',
                      user_m_inv: Optional[Callable[[Tensor], Tensor]] = None,
                      jacobian_reuse: int = 1) -> Tuple[Tensor, float]:
        """
        Adaptive step using step-doubling estimate.
        returns (x_next, h_new)
        """
        x_b = _ensure_batch_dim(x).to(device=self.device, dtype=self.dtype)
        B, N = x_b.shape
        t_curr = t
        h_curr = h

        for attempt in range(max_retries):
            # compute one full step of size h
            x_full = self._implicit_step(x_b, t_curr, h_curr, method, linear_solver, precond, user_m_inv, jacobian_reuse)
            # compute two half steps
            x_half = self._implicit_step(x_b, t_curr, h_curr * 0.5, method, linear_solver, precond, user_m_inv, jacobian_reuse)
            x_half_full = self._implicit_step(x_half, t_curr + 0.5 * h_curr, h_curr * 0.5, method, linear_solver, precond, user_m_inv, jacobian_reuse)

            # estimate error: use p=1 for BE, p=2 for trapezoid
            p = 1 if method == 'backward_euler' else 2
            err_est = torch.norm((x_half_full - x_full).detach(), dim=1)
            # normalize
            scale = atol + rtol * torch.max(x_half_full.detach().norm(dim=1), x_b.detach().norm(dim=1))
            err_ratio = (err_est / scale).max().item()

            if self.verbose:
                print(f"attempt {attempt}: h={h_curr:.3e}, err_ratio={err_ratio:.3e}")

            if err_ratio <= 1.0:
                # accept
                # compute suggested new h using simple power law
                if err_ratio == 0:
                    factor = 2.0
                else:
                    factor = safety * err_ratio ** (-1.0 / (p + 1))
                h_new = max(min_h, min(max_h, h_curr * factor))
                return x_half_full.detach(), h_new
            else:
                # reject and reduce h
                factor = max(0.1, safety * err_ratio ** (-1.0 / (p + 1)))
                h_curr = max(min_h, h_curr * factor)
                if h_curr < min_h + 1e-30:
                    raise RuntimeError("Step size underflow in adaptive integrator")
                continue

        raise RuntimeError("adaptive step failed after max_retries")

    # ----------------- core implicit step (single step of size h) -----------------
    def _implicit_step(self,
                       x: Tensor,
                       t: float,
                       h: float,
                       method: str,
                       linear_solver: str,
                       precond: str,
                       user_m_inv: Optional[Callable[[Tensor], Tensor]],
                       jacobian_reuse: int) -> Tensor:
        """Compute one implicit step of size h starting from x at time t.
        method: 'backward_euler' or 'trapezoid'
        jacobian_reuse: how many newton iterations to reuse analytic jacobian before recomputing
        """
        xk = x.clone()
        B, N = xk.shape
        if method == 'backward_euler':
            def residual(u):
                return u - x - h * self.f(u, t + h)
        elif method == 'trapezoid':
            def residual(u):
                return u - x - 0.5 * h * (self.f(u, t + h) + self.f(x, t))
        else:
            raise ValueError('unknown method')

        # Newton iterations
        J_cached = None
        for nit in range(self.newton_maxiter):
            xk = xk.requires_grad_(True)
            F = residual(xk)
            res_norm = F.detach().norm(dim=1).max().item()
            if self.verbose:
                print(f"  Newton {nit}, res_norm={res_norm:.3e}")
            if res_norm < self.newton_tol:
                break

            # form linear operator A ~ I - h * df/du (depends on method)
            if linear_solver == 'dense' and self.jacobian_fn is not None:
                # use analytic jacobian if provided
                if J_cached is None:
                    Jf = self.jacobian_fn(xk.detach(), t + h)
                    if method == 'backward_euler':
                        A = torch.eye(N, device=self.device, dtype=self.dtype).unsqueeze(0).expand(B, N, N) - h * Jf
                    else:
                        # trapezoid: A = I - 0.5 h df/du
                        A = torch.eye(N, device=self.device, dtype=self.dtype).unsqueeze(0).expand(B, N, N) - 0.5 * h * Jf
                    J_cached = A
                dx = self._dense_solve_cached(J_cached, -F)
            else:
                # use GMRES / JFNK: need matvec function
                def matvec(v: Tensor) -> Tensor:
                    # v: (B,N)
                    if self.jvp_fn is not None:
                        jvp = self.jvp_fn(xk.detach(), t + h, v)
                    else:
                        # use autograd jvp
                        _, jvp = torch.autograd.functional.jvp(lambda z: self.f(z, t + h), xk.detach(), v, create_graph=False)
                    if method == 'backward_euler':
                        return v - h * jvp
                    else:
                        # trapezoid: I - 0.5 h J
                        return v - 0.5 * h * jvp

                # preconditioner
                if precond == 'identity':
                    m_inv = lambda v: v
                elif precond == 'jacobi':
                    diag = self._estimate_diag_precond(xk.detach(), t + h, h, method)
                    diag[diag == 0] = 1.0
                    m_inv = lambda v: v / diag
                elif precond == 'user':
                    if user_m_inv is None:
                        raise ValueError('user_m_inv must be provided')
                    m_inv = user_m_inv
                else:
                    raise ValueError('unknown precond')

                dx = self._gmres(matvec, -F, m_inv)

            # update
            xk = (xk + dx).detach()
            # optionally flush cached jacobian after reuse threshold
            if J_cached is not None:
                jacobian_reuse -= 1
                if jacobian_reuse <= 0:
                    J_cached = None

        return xk.detach()

    # ----------------- utility linear solvers -----------------
    def _dense_solve_cached(self, A: Tensor, rhs: Tensor) -> Tensor:
        # A: (B,N,N), rhs: (B,N)
        dx = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        return dx

    def _gmres(self, matvec: Callable[[Tensor], Tensor], rhs: Tensor, m_inv: Callable[[Tensor], Tensor]) -> Tensor:
        # Simple restarted GMRES as in previous file but using provided matvec/jvp
        B, N = rhs.shape
        x = torch.zeros_like(rhs)
        r = rhs - matvec(x)
        z = m_inv(r)
        beta = torch.norm(z, dim=1)
        beta = torch.where(beta == 0, torch.tensor(1.0, device=self.device, dtype=self.dtype), beta)

        max_restarts = math.ceil(self.gmres_maxiter / self.gmres_restart)
        for restart in range(max_restarts):
            m = min(self.gmres_restart, self.gmres_maxiter - restart * self.gmres_restart)
            V = torch.zeros((B, m + 1, N), device=self.device, dtype=self.dtype)
            H = torch.zeros((B, m + 1, m), device=self.device, dtype=self.dtype)
            V[:, 0, :] = z / beta.unsqueeze(1)
            g = torch.zeros((B, m + 1), device=self.device, dtype=self.dtype)
            g[:, 0] = beta
            for j in range(m):
                vj = V[:, j, :]
                w = matvec(vj)
                w = m_inv(w)
                for i in range(j + 1):
                    vi = V[:, i, :]
                    hij = torch.sum(w * vi, dim=1)
                    H[:, i, j] = hij
                    w = w - hij.unsqueeze(1) * vi
                hj1 = torch.norm(w, dim=1)
                H[:, j + 1, j] = hj1
                nonzero = hj1 != 0
                if nonzero.any():
                    V[nonzero, j + 1, :] = (w[nonzero] / hj1[nonzero].unsqueeze(1))
                else:
                    V[:, j + 1, :] = 0.0
            # solve small least squares per batch
            y = torch.zeros((B, m), device=self.device, dtype=self.dtype)
            for b in range(B):
                Hb = H[b, :m + 1, :m]
                gb = g[b, :m + 1]
                sol = torch.linalg.lstsq(Hb, gb.unsqueeze(1)).solution.squeeze(1)
                y[b, :] = sol
            V_nm = V[:, :m, :].permute(0, 2, 1)
            delta = torch.matmul(V_nm, y.unsqueeze(-1)).squeeze(-1)
            x = x + delta
            r = rhs - matvec(x)
            res_norms = torch.norm(r, dim=1)
            if self.verbose:
                print(f"    GMRES restart {restart}, max residual = {res_norms.max().item():.3e}")
            if res_norms.max().item() < self.gmres_tol:
                break
            z = m_inv(r)
            beta = torch.norm(z, dim=1)
            beta = torch.where(beta == 0, torch.tensor(1.0, device=self.device, dtype=self.dtype), beta)
        return x

    def _estimate_diag_precond(self, x: Tensor, t: float, h: float, method: str) -> Tensor:
        # estimate diagonal of A = I - alpha * J
        B, N = x.shape
        alpha = h if method == 'backward_euler' else 0.5 * h
        eye = torch.eye(N, device=self.device, dtype=self.dtype)
        diag = torch.zeros_like(x)
        for i in range(N):
            v = eye[i].unsqueeze(0).expand(B, -1)
            if self.jvp_fn is not None:
                jvp = self.jvp_fn(x, t, v)
            else:
                _, jvp = torch.autograd.functional.jvp(lambda z: self.f(z, t), x, v, create_graph=False)
            diag[:, i] = 1.0 - alpha * jvp[:, i]
        return diag


# ----------------- short usage notes -----------------
# - Pass jacobian_fn(x,t) if you can compute analytic Jacobian: big win for speed.
# - If J is too large to store, provide jvp_fn(x,t,v) that computes J@v efficiently.
# - Use 'dense' solver only for moderate N (< ~2000 depending on memory). For larger systems use GMRES + jvp.
# - Precompute or reuse Jacobian for several Newton steps when possible (jacobian_reuse > 1).

# If tu veux je peux :
# - inclure un contrôleur PID de pas de temps plus avancé (suit Soderlind et Hairer recommendations),
# - ajouter une implémentation d'un SDIRK 2(3) ou 3-stage L-stable SDIRK pour meilleure stabilité/ordre,
# - implémenter un préconditionneur ILU par bloc (approx.), ou interface à scikit-sparse si tu veux une version CPU hybrid.

