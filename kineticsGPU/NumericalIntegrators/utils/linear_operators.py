# Factory for matvec A(v) = v - h * J v using either autograd Jv or finite differences.
from typing import Callable, Optional
import torch
from kineticsGPU.NumericalIntegrators.utils.jacobian import jv_fd




def make_matvec_autograd(f: Callable[[torch.Tensor, float], torch.Tensor], x_lin: torch.Tensor, t: float, h: float) -> Callable[[torch.Tensor], torch.Tensor]:
    x_lin = x_lin.clone().detach().requires_grad_(True)


    def matvec(v: torch.Tensor) -> torch.Tensor:
        v_det = v.detach()
        _, jvp = torch.autograd.functional.jvp(lambda z: f(z, t), x_lin, v_det, create_graph=False)
        return v_det - h * jvp


    return matvec




def make_matvec_fd(f: Callable[[torch.Tensor, float], torch.Tensor], x: torch.Tensor, t: float, h: float, eps: Optional[float] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    def matvec(v: torch.Tensor) -> torch.Tensor:
        jv = jv_fd(f, x, v, t, eps=eps)
        return v - h * jv


    return matvec

def make_matvec_user_defined(jvp: Callable[[torch.Tensor, float, torch.Tensor], torch.Tensor], x: torch.Tensor, t: float, h: float, eps: Optional[float] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    def matvec(v: torch.Tensor) -> torch.Tensor:
        jv = jvp(x, t, v)
        return v - h * jv
    return matvec

def make_matvec_user_defined_context(jvp: Callable[[float, torch.Tensor], torch.Tensor], t: float, h: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def matvec(v: torch.Tensor) -> torch.Tensor:
        jv = jvp(t, v)
        return v - h * jv
    return matvec