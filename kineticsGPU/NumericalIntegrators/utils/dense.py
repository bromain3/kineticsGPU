# Direct dense solver for batched systems J dx = rhs where J = I - h * Jf.
from typing import Callable
import torch
from kineticsGPU.NumericalIntegrators.utils.jacobian import jacobian_dense_autograd




def dense_solve(f: Callable[[torch.Tensor, float], torch.Tensor], x_lin: torch.Tensor, t: float, h: float, rhs: torch.Tensor) -> torch.Tensor:
    B, N = rhs.shape
    Js = jacobian_dense_autograd(f, x_lin, t)
    As = torch.eye(N, device=Js.device, dtype=Js.dtype).unsqueeze(0).expand(B, -1, -1) - h * Js
    dx = torch.linalg.solve(As, rhs.unsqueeze(-1)).squeeze(-1)
    return dx