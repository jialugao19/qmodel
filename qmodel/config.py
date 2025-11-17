
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.optim as optim

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from qmodel.models import QModel
    from qmodel.data.dataset import QDataset


@dataclass
class ProfilerConfig:
    profile_section: str  # "train", "eval", "none"
    profile_dir: str
    wait: int = 0
    warmup: int = 0
    active: int = 120
    repeat: int = 2

@dataclass
class LRSchedulerConfig:
    start_warmup_factor: float = 0.001
    end_warmup_factor: float = 1.0
    warmup_iters: int = 10**3
    finish_decay_iter: int = 10**6
    eta_min: float = 1e-6

@dataclass
class ModelConfig:
    dtype: torch.dtype = torch.float32

@dataclass
class EvalConfig:
    eval_checkpoint_iter: Sequence[int] = field(init=False)
    eval_all_num_iters: int = field(init=False)
    eval_batch_size: int = field(init=False)


# type hint is mandatory to allow dataclass to recognize fields correctly
@dataclass
class QConfig:
    device: torch.device = torch.device("cuda")

    # data / dataset settings
    window_size: int = field(init=False)
    ret_col_name: str = field(init=False)
    dataset_class: Callable[..., "QDataset"] = field(init=False)

    # training components
    model: ModelConfig = field(default_factory=ModelConfig)
    model_class: Type[nn.Module] = field(init=False)
    seed: int = field(init=False)

    amp_dtype: torch.dtype = field(init=False)
    eval_dtype: torch.dtype = field(init=False)
    train_dtype: torch.dtype = field(init=False)
    criterion: Callable = field(init=False)
    optimizer_class: Type[optim.Adam] = field(init=False)
    learning_rate: float = field(init=False)
    use_amp: str = field(init=False)
    use_lr_sched: str = field(init=False)

    # run metadata
    date: str = field(default_factory=lambda: time.strftime("%m%d"))
    expr_name: str = field(init=False)

    # iteration configuration
    batch_size: int = field(init=False)
    num_workers: int = field(init=False)
    num_iters: int = field(init=False)
    save_every: int = field(init=False)
    eval_every: int = field(init=False)
    eval_during: bool = field(init=False)
    eval_during_num_iters: int = field(init=False)


    load_from_iter: Optional[int] = field(init=False)
    log_every: int = field(init=False)
    mean_loss_length: int = field(init=False)

    # derived paths
    root_dir: str = field(init=False)
    tensorboard_dir: str = field(init=False)

    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    profiler: ProfilerConfig = field(init=False)
    evaluator: EvalConfig = field(default_factory=EvalConfig)

