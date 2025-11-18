# Helpers for device/dtype and small utilities.
from typing import Optional
import torch




def ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x




def to_device_dtype(x: torch.Tensor, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> torch.Tensor:
    if device is None and dtype is None:
        return x
    return x.to(device=device, dtype=dtype)




# small constant
def one(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.tensor(1.0, device=device, dtype=dtype)