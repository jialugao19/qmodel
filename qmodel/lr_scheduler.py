from dataclasses import dataclass
from typing import Optional
import torch.optim as optim


def custom_scheduler(optimizer: optim.Optimizer, config):
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.start_warmup_factor,
        end_factor=config.end_warmup_factor,
        total_iters=config.warmup_iters
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.finish_decay_iter - config.warmup_iters,
        eta_min=config.eta_min
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_iters]
    )
    return scheduler