
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from typing import Optional
import contextlib

class MyScaler:
    """disable amp by passing amp_dtype=torch.float32"""

    def __init__(self, amp_dtype: torch.dtype, origin_dtype: torch.dtype):
        assert amp_dtype in [torch.float32, torch.float16, torch.bfloat16]

        use_amp = amp_dtype in [torch.float16, torch.bfloat16] and \
            origin_dtype in [torch.float32, torch.float64]

        self._scaler = GradScaler() if use_amp else None
        self.device_type = "cuda"
        self.dtype = amp_dtype

    def autocast(self):
        if self._scaler is not None:
            return autocast(device_type=self.device_type, dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        if self._scaler is not None:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        if self._scaler is not None:
            self._scaler.update()

    def state_dict(self) -> Optional[dict]:
        if self._scaler is not None:
            return self._scaler.state_dict()

    def load_state_dict(self, state: Optional[dict]):
        if self._scaler is not None:
            assert isinstance(state, dict)
            self._scaler.load_state_dict(state)
