
import os
import torch
from pathlib import Path
from typing import Optional

from qmodel.components.amp_scaler import MyScaler

from qmodel.config import QConfig

class CheckpointSaver:
    def __init__(self, root_dir: str, config: QConfig, device=None):
        self.dir = Path(root_dir) / "ckpt"

        os.makedirs(self.dir, exist_ok=True)

        self.device = device
        self.config = config

        self.ckpt = {}

    def save(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        scaler: MyScaler, it: int
    ) -> str:
        self.ckpt = {
            "iteration": it,
            "model"    : model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler"   : scaler.state_dict(),
            "config"   : self.config,
        }

        path = self.dir / f"iter_{it}.pt"
        torch.save(self.ckpt, path)
        return path.absolute().as_posix()

    def load(
        self, path: Path | str,
        model: Optional[torch.nn.Module],
        optim: Optional[torch.optim.Optimizer],
        sched: Optional[torch.optim.lr_scheduler.LRScheduler],
        scaler: Optional[MyScaler]
    ) -> int:
        """load the state dict into the provided models"""

        self.ckpt = ckpt = torch.load(
            path, map_location=self.device, weights_only=False
        )

        if model:
            model.load_state_dict(ckpt["model"])
        if optim:
            optim.load_state_dict(ckpt["optimizer"])
        if sched:
            sched.load_state_dict(ckpt["scheduler"])
        if scaler:
            scaler.load_state_dict(ckpt["scaler"])

        return ckpt["iteration"]