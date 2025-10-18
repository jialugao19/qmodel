
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from typing import Optional
import contextlib

class MyScaler:
    def __init__(self, amp_dtype: torch.dtype | None):
        use_amp = amp_dtype is not None
        self._scaler = GradScaler() if use_amp else None
        self.device_type = "cuda"
        self.dtype = amp_dtype if isinstance(amp_dtype, torch.dtype) else torch.float32

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
