# Preconditioners that operate on batched (B,N) tensors.
from typing import Callable
import torch
from kineticsGPU.torch.NumericalIntegrators.utils.jacobian import estimate_jacobian_diag_autograd
from kineticsGPU.torch.NumericalIntegrators.utils.utils import one




def identity_precond(v: torch.Tensor) -> torch.Tensor:
    return v




def jacobi_precond_from_diag(diag: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def precond(v: torch.Tensor) -> torch.Tensor:
        return v / diag
    return precond




def make_jacobi_autograd(f: Callable[[torch.Tensor, float], torch.Tensor], x_lin: torch.Tensor, t: float, h: float) -> Callable[[torch.Tensor], torch.Tensor]:
    diag = estimate_jacobian_diag_autograd(f, x_lin, t, h)
    diag = torch.where(diag == 0.0, one(device=diag.device, dtype=diag.dtype), diag)
    return jacobi_precond_from_diag(diag)




def chemistry_preconditioner(f: Callable[[torch.Tensor, float], torch.Tensor], x: torch.Tensor, t: float, h: float, eps: float = 1e-8) -> Callable[[torch.Tensor], torch.Tensor]:
    B, N = x.shape
    fx = f(x, t)
    diag_est = torch.empty_like(x)
    for i in range(N):
        ei = torch.zeros_like(x)
        scale = torch.maximum(x[:, i].abs(), torch.ones_like(x[:, i]))
        ei[:, i] = eps * scale
        f_p = f(x + ei, t)
        dfii = (f_p[:, i] - fx[:, i]) / (ei[:, i] + 1e-20)
        diag_est[:, i] = 1.0 - h * dfii
    diag_est = torch.where(diag_est == 0.0, one(device=diag_est.device, dtype=diag_est.dtype), diag_est)


    def precond(v: torch.Tensor) -> torch.Tensor:
        return v / diag_est


    return precond