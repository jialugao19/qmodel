
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from typing import Optional
import contextlib

class MyScaler:
    """disable amp by passing amp_dtype=torch.float32"""

    def __init__(self, use_amp: str, amp_dtype: torch.dtype):
        assert amp_dtype in [torch.float32, torch.float16, torch.bfloat16]

        # use_amp = amp_dtype in [torch.float16, torch.bfloat16] and \
        #     origin_dtype in [torch.float32, torch.float64]
        assert use_amp in ["torch", "custom", "none"]

        self._scaler = None
        if use_amp == "torch":
            self._scaler = GradScaler()
        elif use_amp == "custom":
            self._scaler = CudaGraphAMPScaler(init_scale=2.**16)
        else:
            assert use_amp == "none"

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

    def get_scale(self) -> float:
        """Return current scaling factor for logging/monitoring."""
        if self._scaler is None:
            return 1.0
        if hasattr(self._scaler, "get_scale"):
            return float(self._scaler.get_scale())
        # torch GradScaler stores scale tensor on device; cast to float
        return float(self._scaler._scale.cpu().item())


class CudaGraphAMPScaler:
    def __init__(self, init_scale=2.**16, growth=2.0, backoff=0.5, growth_interval=2000, device="cuda"):
        self.scaler  = torch.tensor(float(init_scale),      dtype=torch.float32, device=device)
        self.growth  = torch.tensor(float(growth),          dtype=torch.float32, device=device)
        self.backoff = torch.tensor(float(backoff),         dtype=torch.float32, device=device)
        self.gi      = torch.tensor(float(growth_interval), dtype=torch.float32, device=device)
        self.step_counter = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.found_inf    = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.acc          = torch.tensor(0.0, dtype=torch.float32, device=device)

    def state_dict(self) -> dict:
        return {
            "scaler":        self.scaler.cpu(),
            "growth":        self.growth.cpu(),
            "backoff":       self.backoff.cpu(),
            "growth_interval": self.gi.cpu(),
            "step_counter":  self.step_counter.cpu(),
        }

    def load_state_dict(self, state: dict):
        self.scaler       = state["scaler"].to(self.scaler.device)
        self.growth       = state["growth"].to(self.growth.device)
        self.backoff      = state["backoff"].to(self.backoff.device)
        self.gi           = state["growth_interval"].to(self.gi.device)
        self.step_counter = state["step_counter"].to(self.step_counter.device)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scaler

    def step(self, optimizer):
        with torch.no_grad():
            inv = self.scaler.double().reciprocal().float()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        p.grad.mul_(inv)
            self.found_inf.zero_()
            self.acc.zero_()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        grad = p.grad
                        self.acc += torch.isinf(grad).any().float()
                        self.acc += torch.isnan(grad).any().float()
            self.acc.clamp_(0, 1.0)
            self.found_inf.add_(self.acc).clamp_(0, 1.0)
            # 先消毒再掩码；NaN*0 仍是 NaN，因此需要 nan_to_num_
            mask = 1.0 - self.found_inf  # 1=正常更新，0=整体屏蔽
            for g in optimizer.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        p.grad.mul_(mask)

        optimizer.step()

    def update(self):
        # 纯张量逻辑：若无溢出按区间增长，否则退避缩小；并用计数器控制增长频率
        ok  = 1.0 - self.found_inf
        self.step_counter.add_(ok)                    # 仅无溢出时累计
        grow_mask = (self.step_counter >= self.gi).float() * ok  # 达到增长间隔且无溢出
        # new_scale = scale * (backoff if found_inf else (growth if reach interval else 1))
        factor = self.found_inf * self.backoff + grow_mask * self.growth + (1.0 - self.found_inf) * (1.0 - grow_mask) * 1.0
        self.scaler.mul_(factor)
        # 达到增长后把计数器清零（张量化）
        self.step_counter.mul_(1.0 - grow_mask)

    def get_scale(self) -> float:
        return float(self.scaler.item())
