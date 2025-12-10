import torch.optim as optim
import torch
import math
from typing import TypeAlias, Callable
from qmodel.config import QConfig

Scheduler: TypeAlias = "optim.lr_scheduler.LRScheduler | GraphLRScheduler"


def custom_scheduler(optimizer: optim.Optimizer, config: QConfig):
    lrconf = config.lr_scheduler
    assert config.use_lr_sched in ["torch", "custom"]

    if config.use_lr_sched == "custom":
        return GraphLRScheduler(
            optimizer=optimizer,
            warmup_iters=lrconf.warmup_iters,
            total_iters=lrconf.finish_decay_iter,
            start_factor=lrconf.start_warmup_factor,
            end_factor=lrconf.end_warmup_factor,
            eta_min=lrconf.eta_min,
            base_lr=config.learning_rate,
            device=config.device
        )

    # todo: assert not using CUDA Graph

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=lrconf.start_warmup_factor,
        end_factor=lrconf.end_warmup_factor,
        total_iters=lrconf.warmup_iters
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=lrconf.finish_decay_iter - lrconf.warmup_iters,
        eta_min=lrconf.eta_min
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[lrconf.warmup_iters]
    )

    # update lr to tensor instead of raw float, to correctly allow CUDA Graph capture
    # optimizer correctly (but not LR Schuduler!) LR Scheduler needs to be done outside CUDA Graph
    for group in optimizer.param_groups:
        group['lr'] = torch.tensor(config.learning_rate * lrconf.start_warmup_factor, device=config.device, dtype=torch.float64, requires_grad=False)
    return scheduler


class GraphLRScheduler:
    """
    An exact match to Torch's Linear + CosineAnnealing LR scheduler,
    done manually to verify correctness.

    Also it makes it possible to be put inside the CUDA Graph.
    """
    def __init__(self, optimizer,
                 warmup_iters, total_iters,
                 start_factor, end_factor,
                 eta_min, base_lr,
                 device):
        self.optimizer = optimizer

        # convert lr to tensor
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            param_group['lr'] = torch.tensor(lr, device=device, dtype=torch.float64, requires_grad=False)

        self.global_step = torch.zeros((), dtype=torch.int64, device=device)
        self.param_groups = optimizer.param_groups

        self.warmup_iters = torch.tensor(warmup_iters, device=device, dtype=torch.int64)
        self.total_iters = torch.tensor(total_iters, device=device, dtype=torch.int64)

        self.start_factor = torch.tensor(start_factor, device=device, dtype=torch.float64)
        self.end_factor   = torch.tensor(end_factor, device=device, dtype=torch.float64)
        self.eta_min      = torch.tensor(eta_min, device=device, dtype=torch.float64)
        self.base_lr      = torch.tensor(base_lr, device=device, dtype=torch.float64)

        self.cosine_base_lr = torch.tensor(
            self.eta_min + (self.base_lr - self.eta_min),
            device=device, dtype=torch.float64
        )

        self.set_lr(self.base_lr * self.start_factor)

    def get_lr(self, t: torch.Tensor, prev_lr: torch.Tensor):
        """in torch's impl, for each scheduler,
        the first call gets lr from closed form,
        then always get from chainable form.
        """
        # linear warmup lr that is exacly the same as torch's impl
        warmup_lr = prev_lr * (
            1.0 + (self.end_factor - self.start_factor).double() / (
                self.warmup_iters * self.start_factor
                + t * (self.end_factor - self.start_factor)
            ).double()
        )

        # cosine
        last_epoch = (t - self.warmup_iters + 1).clamp(min=0).double()
        T_max = (self.total_iters - self.warmup_iters).double()

        cosine_lr1 = prev_lr + (
            (self.cosine_base_lr - self.eta_min)
            * (1 - torch.cos(math.pi / T_max)) / 2
        )

        cosine_lr2 = (
            (1 + torch.cos(math.pi * last_epoch / T_max))
            / (1 + torch.cos(math.pi * (last_epoch - 1) / T_max))
            * (prev_lr - self.eta_min)
            + self.eta_min
        )

        lr = torch.where(
            t < self.warmup_iters - 1, warmup_lr, torch.where(
                t == self.warmup_iters - 1, self.cosine_base_lr, torch.where(
                    (last_epoch - 1 - T_max) % (2 * T_max) == 0,
                    cosine_lr1,
                    cosine_lr2
                )
            )
        )
        return lr

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'].copy_(lr)

    @torch.no_grad()
    def step(self):
        for param_group in self.param_groups:
            prev_lr = param_group['lr']
            lr = self.get_lr(self.global_step, prev_lr)
            param_group['lr'].copy_(lr)

        self.global_step.add_(1)  # step += 1

    def state_dict(self):
        return {
            "global_step": self.global_step.cpu(),
            "warmup_iters": self.warmup_iters.cpu(),
            "total_iters":  self.total_iters.cpu(),
            "start_factor": self.start_factor.cpu(),
            "end_factor":   self.end_factor.cpu(),
            "eta_min":      self.eta_min.cpu(),
            "base_lr":      self.base_lr.cpu(),
        }

    def load_state_dict(self, state):
        self.global_step.copy_(state["global_step"].to(self.global_step.device))
        self.warmup_iters.copy_(state["warmup_iters"].to(self.warmup_iters.device))
        self.total_iters.copy_(state["total_iters"].to(self.total_iters.device))
        self.start_factor.copy_(state["start_factor"].to(self.start_factor.device))
        self.end_factor.copy_(state["end_factor"].to(self.end_factor.device))
        self.eta_min.copy_(state["eta_min"].to(self.eta_min.device))
        self.base_lr.copy_(state["base_lr"].to(self.base_lr.device))
