# Jacobian-related utilities: diag estimation, dense jacobian and finite-diff Jv.
from typing import Callable, Optional
import torch




def estimate_jacobian_diag_autograd(f: Callable[[torch.Tensor, float], torch.Tensor], x_lin: torch.Tensor, t: float, h: float) -> torch.Tensor:
    B, N = x_lin.shape
    x_lin = x_lin.clone().detach().requires_grad_(True)
    fx = f(x_lin, t)
    diag = torch.empty_like(x_lin)
    eye = torch.eye(N, device=x_lin.device, dtype=x_lin.dtype)
    for i in range(N):
        grad_out = eye[i].unsqueeze(0).expand(B, -1)
        vjp = torch.autograd.grad(outputs=fx, inputs=x_lin, grad_outputs=grad_out, retain_graph=True, create_graph=False)[0]
        diag[:, i] = 1.0 - h * vjp[:, i]
    return diag




def jv_fd(f: Callable[[torch.Tensor, float], torch.Tensor], x: torch.Tensor, v: torch.Tensor, t: float, eps: Optional[float] = None) -> torch.Tensor:
    dtype = x.dtype
    device = x.device
    eps = eps or (1e-6 if dtype==torch.float32 else 1e-8)
    x = x.detach()
    v = v.detach()
    fx = f(x, t)
    x_pert = x + eps * v
    fxp = f(x_pert, t)
    jv = (fxp - fx) / eps
    return jv




def jacobian_dense_autograd(f: Callable[[torch.Tensor, float], torch.Tensor], x: torch.Tensor, t: float) -> torch.Tensor:
    B, N = x.shape
    Js = torch.empty((B, N, N), device=x.device, dtype=x.dtype)
    for b in range(B):
        xb = x[b:b+1].requires_grad_(True)
        Jf = torch.autograd.functional.jacobian(lambda z: f(z, t)[0], xb, create_graph=False)
        if Jf.dim() == 3:
            Jf = Jf[0]
            Js[b] = Jf
    return Js