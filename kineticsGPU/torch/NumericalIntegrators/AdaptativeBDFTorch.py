"""
Backward (implicit) Euler integrator with adaptive timestep and batched GPU support.
Features:
- Adaptive timestep control based on convergence rate and error estimation
- Implicit backward Euler with Newton iterations
- Linear solver options: dense LU, GMRES, BiCGStab-l
- Preconditioners: identity, Jacobi, chemistry
- Batched systems (B, N) on GPU/CPU
- Per-batch timestep control for heterogeneous systems
"""

from typing import Callable, Optional, Tuple
import torch
from kineticsGPU.torch.NumericalIntegrators.utils.utils import ensure_batch_dim, one
from kineticsGPU.torch.NumericalIntegrators.utils.residual import compute_residual
from kineticsGPU.torch.NumericalIntegrators.utils.linear_operators import make_matvec_autograd, make_matvec_fd, \
    make_matvec_user_defined
from kineticsGPU.torch.NumericalIntegrators.utils.preconditioners import identity_precond, make_jacobi_autograd, \
    chemistry_preconditioner
from kineticsGPU.torch.NumericalIntegrators.utils.dense import dense_solve
from kineticsGPU.torch.NumericalIntegrators.utils.GMRES import gmres_givens_batched
from kineticsGPU.torch.NumericalIntegrators.utils.BiCGStabl import bicgstab_l_batched


class BackwardEulerIntegrator:
    def __init__(self,
                 f: Callable[[torch.Tensor, float], torch.Tensor],
                 jvp: Optional[Callable[[torch.Tensor, float, torch.Tensor], torch.Tensor]] = None,
                 device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None,
                 newton_tol: float = 1e-6,
                 newton_maxiter: int = 20,
                 gmres_tol: float = 1e-6,
                 gmres_maxiter: int = 50,
                 gmres_restart: int = 20,
                 verbose: bool = False,
                 use_autograd_jvp: bool = True,
                 # Adaptive timestep parameters
                 adaptive: bool = False,
                 atol: float = 1e-6,
                 rtol: float = 1e-4,
                 h_min: float = 1e-10,
                 h_max: float = 1.0,
                 safety_factor: float = 0.9,
                 max_rejections: int = 5):
        self.f = f
        self.jvp = jvp
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype or torch.float32
        self.newton_tol = newton_tol
        self.newton_maxiter = newton_maxiter
        self.gmres_tol = gmres_tol
        self.gmres_maxiter = gmres_maxiter
        self.gmres_restart = gmres_restart
        self.verbose = verbose
        self.use_autograd_jvp = use_autograd_jvp
        self._one = one(device=self.device, dtype=self.dtype)

        # Adaptive timestep
        self.adaptive = adaptive
        self.atol = atol
        self.rtol = rtol
        self.h_min = h_min
        self.h_max = h_max
        self.safety_factor = safety_factor
        self.max_rejections = max_rejections

    def _estimate_error(self, x_full: torch.Tensor, x_half1: torch.Tensor,
                        x_half2: torch.Tensor) -> torch.Tensor:
        """
        Estimate local truncation error using Richardson extrapolation.
        Compare full step vs two half steps.
        Returns per-batch error estimates.
        """
        # Error estimate: ||x_full - x_half2|| / ||x_half2||
        error = torch.abs(x_full - x_half2)
        scale = self.atol + self.rtol * torch.abs(x_half2)
        normalized_error = error / scale
        # Max error per batch
        return normalized_error.max(dim=1)[0]  # (B,)

    def _compute_new_stepsize(self, h: torch.Tensor, error: torch.Tensor,
                              order: int = 1) -> torch.Tensor:
        """
        Compute new timestep based on error estimate.
        h_new = safety * h * (tol/error)^(1/(order+1))
        """
        # Avoid division by zero
        error = torch.clamp(error, min=1e-14)
        factor = torch.pow(1.0 / error, 1.0 / (order + 1))
        factor = torch.clamp(factor, 0.2, 5.0)  # Limit growth/shrinkage
        h_new = self.safety_factor * h * factor
        h_new = torch.clamp(h_new, self.h_min, self.h_max)
        return h_new

    def step(self,
             x: torch.Tensor,
             t: float,
             h: float,
             method: str = 'gmres',
             precond: str = 'identity',
             m_inv_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        """Single timestep - delegates to adaptive or fixed step."""
        if self.adaptive:
            return self.step_adaptive(x, t, h, method, precond, m_inv_fn)[0]
        else:
            return self.step_fixed(x, t, h, method, precond, m_inv_fn)

    def step_fixed(self,
                   x: torch.Tensor,
                   t: float,
                   h: float,
                   method: str = 'gmres',
                   precond: str = 'identity',
                   m_inv_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        """Original fixed timestep implementation."""
        x_b = ensure_batch_dim(x).to(device=self.device, dtype=self.dtype)
        B, N = x_b.shape
        t_new = t + h

        with torch.no_grad():
            xk = (x_b + h * self.f(x_b, t)).to(device=self.device, dtype=self.dtype)

        for nit in range(self.newton_maxiter):
            with torch.no_grad():
                F, norms = compute_residual(xk, x_b, h, self.f, t_new)
                res_norm = norms.max().item()
            if self.verbose:
                print(f"[Newton {nit}] residual max = {res_norm:.3e}")
            if res_norm < self.newton_tol:
                break

            x_lin = xk.clone().detach().requires_grad_(self.use_autograd_jvp)

            m_inv_fn_eff = self._get_preconditioner(precond, m_inv_fn, x_lin, t_new, h)
            rhs = (-F).detach()
            matvec = self._get_matvec(x_lin, t_new, h, method)
            dx = self._solve_linear_system(method, matvec, rhs, m_inv_fn_eff, x_lin, t_new, h)

            with torch.no_grad():
                xk = xk + dx
                xk = xk.to(device=self.device, dtype=self.dtype)

        return xk.detach()

    def step_adaptive(self,
                      x: torch.Tensor,
                      t: float,
                      h: float,
                      method: str = 'gmres',
                      precond: str = 'identity',
                      m_inv_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> Tuple[
        torch.Tensor, float, bool]:
        """
        Adaptive timestep with error control.
        Returns: (x_new, h_accepted, success)
        """
        x_b = ensure_batch_dim(x).to(device=self.device, dtype=self.dtype)
        B, N = x_b.shape

        h_current = h
        rejections = 0

        while rejections < self.max_rejections:
            # Full step with h
            x_full = self._newton_solve(x_b, t, h_current, method, precond, m_inv_fn)

            # Two half steps with h/2
            x_half1 = self._newton_solve(x_b, t, h_current / 2, method, precond, m_inv_fn)
            x_half2 = self._newton_solve(x_half1, t + h_current / 2, h_current / 2, method, precond, m_inv_fn)

            # Estimate error
            error = self._estimate_error(x_full, x_half1, x_half2)
            max_error = error.max().item()

            if self.verbose:
                print(f"[Adaptive] h={h_current:.3e}, error={max_error:.3e}, rejections={rejections}")

            # Accept or reject
            if max_error <= 1.0:
                # Step accepted - use more accurate half-step result
                return x_half2.detach(), h_current, True
            else:
                # Reject and reduce stepsize
                h_current = h_current * max(0.2, self.safety_factor * (1.0 / max_error) ** 0.5)
                h_current = max(h_current, self.h_min)
                rejections += 1

        # Max rejections reached - accept with warning
        if self.verbose:
            print(f"[Adaptive] WARNING: Max rejections reached, accepting with h={h_current:.3e}")
        return x_full.detach(), h_current, False

    def _newton_solve(self,
                      x_b: torch.Tensor,
                      t: float,
                      h: float,
                      method: str,
                      precond: str,
                      m_inv_fn: Optional[Callable]) -> torch.Tensor:
        """Internal Newton solver for a single step."""
        B, N = x_b.shape
        t_new = t + h

        with torch.no_grad():
            xk = (x_b + h * self.f(x_b, t)).to(device=self.device, dtype=self.dtype)

        for nit in range(self.newton_maxiter):
            with torch.no_grad():
                F, norms = compute_residual(xk, x_b, h, self.f, t_new)
                res_norm = norms.max().item()
                if self.verbose:
                    print(f"[Newton {nit}] residual max = {res_norm:.3e}")
            if res_norm < self.newton_tol:
                break

            x_lin = xk.clone().detach().requires_grad_(self.use_autograd_jvp)
            m_inv_fn_eff = self._get_preconditioner(precond, m_inv_fn, x_lin, t_new, h)
            rhs = (-F).detach()
            matvec = self._get_matvec(x_lin, t_new, h, method)
            dx = self._solve_linear_system(method, matvec, rhs, m_inv_fn_eff, x_lin, t_new, h)

            with torch.no_grad():
                xk = xk + dx
                xk = xk.to(device=self.device, dtype=self.dtype)

        return xk

    def _get_preconditioner(self, precond: str, m_inv_fn: Optional[Callable],
                            x_lin: torch.Tensor, t_new: float, h: float) -> Callable:
        """Get preconditioner function."""
        if precond == 'identity':
            return identity_precond
        elif precond == 'jacobi':
            if self.use_autograd_jvp:
                return make_jacobi_autograd(self.f, x_lin, t_new, h)
            else:
                return chemistry_preconditioner(self.f, x_lin, t_new, h)
        elif precond == 'chemistry':
            return chemistry_preconditioner(self.f, x_lin, t_new, h)
        elif precond == 'user':
            if m_inv_fn is None:
                raise ValueError("m_inv_fn must be provided when precond='user'")
            return m_inv_fn
        else:
            raise ValueError(f"Unknown precond {precond}")

    def _get_matvec(self, x_lin: torch.Tensor, t_new: float, h: float, method: str) -> Callable:
        """Get matrix-vector product function."""
        if self.jvp is not None:
            return make_matvec_user_defined(self.jvp, x_lin, t_new, h)
        else:
            if self.use_autograd_jvp and method == 'gmres':
                return make_matvec_autograd(self.f, x_lin, t_new, h)
            else:
                return make_matvec_fd(self.f, x_lin, t_new, h)

    def _solve_linear_system(self, method: str, matvec: Callable, rhs: torch.Tensor,
                             m_inv_fn: Callable, x_lin: torch.Tensor,
                             t_new: float, h: float) -> torch.Tensor:
        """Solve linear system with specified method."""
        if method == 'dense':
            return dense_solve(self.f, x_lin, t_new, h, rhs)
        elif method == 'gmres':
            return gmres_givens_batched(matvec=matvec,
                                        rhs=rhs,
                                        m_inv=m_inv_fn,
                                        restart=self.gmres_restart,
                                        tol=self.gmres_tol,
                                        maxiter=self.gmres_maxiter,
                                        verbose=self.verbose)
        elif method == 'bicgstabl':
            return bicgstab_l_batched(matvec=matvec,
                                      rhs=rhs,
                                      m_inv=m_inv_fn,
                                      l=2,
                                      tol=self.gmres_tol,
                                      maxiter=self.gmres_maxiter,
                                      verbose=self.verbose)
        else:
            raise ValueError(f"Unknown linear solver method {method}")

    def integrate(self,
                  x0: torch.Tensor,
                  t_span: Tuple[float, float],
                  h_init: float = 0.01,
                  method: str = 'gmres',
                  precond: str = 'identity',
                  m_inv_fn: Optional[Callable] = None,
                  save_every: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate from t_span[0] to t_span[1] with adaptive timestep.
        Returns: (times, states) where states has shape (n_saved, B, N)
        """
        x = ensure_batch_dim(x0).to(device=self.device, dtype=self.dtype)
        t_start, t_end = t_span
        t = t_start
        h = h_init

        times = [t]
        states = [x.clone()]
        step_count = 0

        while t < t_end:
            h = min(h, t_end - t)

            if self.adaptive:
                x_new, h_accepted, success = self.step_adaptive(x, t, h, method, precond, m_inv_fn)
                # Update h for next step based on current success
                if success:
                    h = min(h_accepted * 1.2, self.h_max)  # Cautiously increase
                t += h_accepted
            else:
                x_new = self.step_fixed(x, t, h, method, precond, m_inv_fn)
                t += h

            x = x_new
            step_count += 1

            if step_count % save_every == 0:
                times.append(t)
                states.append(x.clone())

            if self.verbose and step_count % 10 == 0:
                print(f"[Integrate] t={t:.6f}, h={h:.3e}, steps={step_count}")

        return torch.tensor(times), torch.stack(states)


# Example usage
if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Stiff Van der Pol oscillator
    def van_der_pol(x, t, mu=1000.0):
        B, N = x.shape
        dx = torch.zeros_like(x)
        if N >= 2:
            dx[:, 0] = x[:, 1]
            dx[:, 1] = mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]
        return dx


    B = 100
    N = 2
    x0 = torch.tensor([[2.0, 0.0]] * B, device=device)

    print("Testing adaptive timestep integrator...")
    integrator_adaptive = BackwardEulerIntegrator(
        van_der_pol,
        device=device,
        verbose=True,
        adaptive=True,
        atol=1e-4,
        rtol=1e-3,
        h_min=1e-10,
        h_max=0.1
    )

    t_start = time.perf_counter()
    times, states = integrator_adaptive.integrate(x0, (0.0, 3.0), h_init=0.01, method='gmres')
    t_elapsed = time.perf_counter() - t_start

    print(f"Adaptive: {len(times)} saved steps in {t_elapsed:.3f}s")
    print(f"Final state: {states[-1][0]}")