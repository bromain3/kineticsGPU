# Residual computation for backward Euler Newton iterations.
from typing import Callable
import torch
from kineticsGPU.NumericalIntegrators.utils.utils import ensure_batch_dim




def compute_residual(xk: torch.Tensor, x_prev: torch.Tensor, h: float, f: Callable[[torch.Tensor, float], torch.Tensor], t_new: float):
    # Return residual F = xk - x_prev - h * f(xk, t_new) and its per-batch norms.
    xk = ensure_batch_dim(xk)
    x_prev = ensure_batch_dim(x_prev)
    fxk = f(xk, t_new)
    F = xk - x_prev - h * fxk
    norms = F.norm(dim=1)
    return F, norms