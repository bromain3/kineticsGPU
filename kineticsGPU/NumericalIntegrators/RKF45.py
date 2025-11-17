import math
from typing import Callable, Optional, Tuple
import torch

Tensor = torch.Tensor

def _ensure_batch_dim(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x

class AdaptiveExplicitRKF45:
    """
    Adaptive explicit RKF45 (Fehlberg 4(5)) integrator with variable step size.

    Usage:
        integrator = AdaptiveExplicitRKF45(f, device=device, dtype=torch.float32)
        x_next, h_new = integrator.step_adaptive(x, t, h, ...)

    Features:
      - Fehlberg 4(5) coefficients (embedded 4th/5th order pair)
      - batched (B,N) support
      - absolute/tolerances like implicit integrator (atol, rtol)
      - safety, max/min step, growth/shrink limits
      - optional 'controller' = 'simple' (power law) or 'PI' (better step smoothness)
    """

    def __init__(self,
                 f: Callable[[Tensor, float], Tensor],
                 device: Optional[torch.device] | str = None,
                 dtype: Optional[torch.dtype] = None,
                 newton_tol: float = 1e-7,  # unused but kept for API similarity
                 verbose: bool = False):
        self.f = f
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype or torch.float32
        self.verbose = verbose

        # Fehlberg RKF45 coefficients (classic Fehlberg)
        # But we use the commonly used coefficients for the 6-stage Fehlberg method.
        self.a = [0.0,
                  1/4,
                  3/8,
                  12/13,
                  1.0,
                  1/2]

        self.b = [
            [],
            [1/4],
            [3/32, 9/32],
            [1932/2197, -7200/2197, 7296/2197],
            [439/216, -8, 3680/513, -845/4104],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40]
        ]

        # weights for 5th order solution (y5)
        self.c5 = [16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55]

        # weights for 4th order solution (y4) (embedded) -> used to estimate error
        self.c4 = [25/216, 0.0, 1408/2565, 2197/4104, -1/5, 0.0]

        # order of error (error ~ h^5) -> exponent = 1/5
        self.error_exponent = 1.0 / 5.0

    def step_adaptive(self,
                      x: Tensor,
                      t: float,
                      h: float,
                      atol: float = 1e-6,
                      rtol: float = 1e-3,
                      safety: float = 0.9,
                      min_h: float = 1e-12,
                      max_h: float = 1.0,
                      max_retries: int = 10,
                      max_growth: float = 5.0,
                      max_shrink: float = 0.1,
                      controller: str = 'simple',  # 'simple' or 'PI'
                      prev_err_ratio: Optional[float] = None,  # for PI controller memory
                      verbose: bool = False) -> Tuple[Tensor, float]:
        """
        Perform an adaptive RKF45 step from (x,t) with initial guess h.
        Returns (x_next, h_new).

        If step rejected, tries up to max_retries reducing h.
        """

        x_b = _ensure_batch_dim(x).to(device=self.device, dtype=self.dtype)
        B, N = x_b.shape
        t_curr = float(t)
        h_curr = float(h)
        prev_err_ratio_local = prev_err_ratio

        for attempt in range(max_retries):
            if h_curr < min_h:
                raise RuntimeError("Step size underflow in RKF45 adaptive integrator")

            # compute ks
            ks = [None] * 6
            # k1
            ks[0] = self.f(x_b, t_curr).to(device=self.device, dtype=self.dtype)

            # k2..k6
            for i in range(1, 6):
                # compute stage input
                xi = x_b.clone()
                for j in range(i):
                    if j < len(self.b[i]):
                        xi = xi + (h_curr * self.b[i][j]) * ks[j]
                ti = t_curr + self.a[i] * h_curr
                ks[i] = self.f(xi, ti)

            # construct y4 and y5
            y4 = x_b.clone()
            y5 = x_b.clone()
            for i in range(6):
                y4 = y4 + h_curr * self.c4[i] * ks[i]
                y5 = y5 + h_curr * self.c5[i] * ks[i]

            # error estimate per batch (vector)
            err = (y5 - y4)
            err_est = torch.norm(err, dim=1)  # L2 per batch
            # scaling
            scale = atol + rtol * torch.max(y5.detach().norm(dim=1), x_b.detach().norm(dim=1))
            # avoid division by zero
            scale = torch.where(scale == 0.0, torch.tensor(atol, device=self.device, dtype=self.dtype), scale)
            err_ratio_batch = err_est / scale
            # take worst-case across batch (like in your implicit integrator)
            err_ratio = err_ratio_batch.max().item()

            if verbose or self.verbose:
                print(f"attempt {attempt}: h={h_curr:.3e}, err_ratio={err_ratio:.3e}")

            # accept / reject
            if err_ratio <= 1.0:
                # accepted
                # compute new h using controller
                if controller == 'simple':
                    if err_ratio == 0:
                        factor = max_growth
                    else:
                        factor = safety * (err_ratio ** (-self.error_exponent))
                elif controller == 'PI':
                    # simple PI controller (H211-like); needs prev_err_ratio
                    # formula: h_new = h * safety * err_ratio^{-kI} * prev_err_ratio^{-kP}
                    # choose gains (empirical): kP = 0.7/(order+1), kI = 0.4/(order+1) (example)
                    order_plus_one = 5.0
                    kP = 0.7 / order_plus_one
                    kI = 0.4 / order_plus_one
                    if prev_err_ratio_local is None or prev_err_ratio_local <= 0:
                        prev_err_ratio_local = err_ratio
                    factor = safety * (err_ratio ** (-kP)) * (prev_err_ratio_local ** (-kI))
                    prev_err_ratio_local = err_ratio
                else:
                    raise ValueError("unknown controller")

                # clamp growth/shrink
                factor = max(max_shrink, min(max_growth, factor))
                h_new = max(min_h, min(max_h, h_curr * factor))

                return y5.detach(), h_new
            else:
                # reject and shrink step
                # shrink factor like simple controller but limited
                factor = max(max_shrink, min(1.0, safety * (err_ratio ** (-self.error_exponent))))
                h_curr = max(min_h, h_curr * factor)
                if h_curr < min_h + 1e-30:
                    raise RuntimeError("Step size underflow in RKF45 adaptive integrator")
                # continue and retry with smaller h

        raise RuntimeError("adaptive RKF45 step failed after max_retries")


# ------------------ Example usage (simple test) ------------------
if __name__ == "__main__":
    # test on a scalar ODE dy/dt = -y  => exact solution y = exp(-t)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    def f(y, t):
        # y: (B,N) or (N,) depending on batch; here we assume batch dimension present
        return -y

    integrator = AdaptiveExplicitRKF45(f, device=device, dtype=dtype, verbose=True)

    y0 = torch.tensor([1.0], device=device, dtype=dtype)  # B=1, N=1
    t = 0.0
    h = 0.1
    steps = 10
    for i in range(steps):
        y0, h = integrator.step_adaptive(y0, t, h, atol=1e-10, rtol=1e-8, safety=0.9, controller='PI', verbose=True)
        t += h
        print(f"step {i}: t={t:.6f}, y={y0.squeeze().cpu().item():.12f}, exact={math.exp(-t):.12f}, h_next={h:.3e}")
