import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from dataclasses import dataclass, asdict
import os
import shutil
import time
from functools import partial
from pathlib import Path

from typing import Any, Callable, Type, Optional, Sequence

from qmodel.models import *
from qmodel import Trainer, Evaluator

from qmodel.logger import logger
from qmodel.config import QConfig, ModelConfig, ProfilerConfig, LRSchedulerConfig, EvalConfig

from gru1 import GRU3
from dataset import AQ2Dataset

@dataclass
class ModelConfig1(ModelConfig):

    input_dim: int = 8
    output_dim = 1
    gru_hidden_dim = 192
    gru_num_layers = 4
    gru_bidirectional = False
    gru_dropout = 0.0

    dense_dropout = 0.0
    dense_hidden_dim = 512

@dataclass
class EvalConfig1(EvalConfig):
    eval_checkpoint_iter: Sequence[int]
    eval_all_num_iters: int = 0
    eval_batch_size: int = 16384

@dataclass
class Config(QConfig):


    # data / dataset settings
    window_size = 32
    ret_col_name: str = "ret_1m_5m"
    dataset_class: Any = partial(AQ2Dataset, ret_col_name=ret_col_name, window_size=window_size)


    # training components
    model: ModelConfig = ModelConfig1()
    model_class = GRU3
    seed = 42

    amp_dtype: torch.dtype = torch.float16  # set to float32 to disable amp
    eval_dtype: torch.dtype = torch.float16
    train_dtype: torch.dtype = torch.float32
    criterion: Callable = nn.HuberLoss()
    optimizer_class: Type[optim.Adam] = optim.Adam
    learning_rate: float = 5e-4

    date = f"{time.strftime('%m%d')}"
    expr_name = f"AQ2_1min_convgru1_clip_{time.strftime('%m%d_%H%M%S')}"
    # expr_name = f"AQ2_1min_convgru1_clip_0920_160131"

    # date = "best"
    # expr_name = "best2"

    load_from_iter = None

    batch_size = 1024
    num_workers = 32
    num_iters = 1 * 10**6 + 1
    save_every = 10000
    eval_every = 10000
    eval_during = True
    eval_during_num_iters = 200

    load_from_iter: Optional[int] = None
    log_every = 1000
    mean_loss_length = 2000

    root_dir = os.path.join('train_logs', date, expr_name)
    tensorboard_dir = os.path.join('runs', date, expr_name)

    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    profiler: ProfilerConfig = ProfilerConfig(
        profile_section="none",  # "train", "eval", "none"
        profile_dir=os.path.join(tensorboard_dir, "profile")
    )
    evaluator: EvalConfig = EvalConfig1(
        eval_checkpoint_iter=np.arange(0, 1_000_000, 10_000).tolist()[::-1][:1]
    )

    def __post_init__(self):
        os.makedirs(self.root_dir, exist_ok=True)

        log_file = f"{self.root_dir}/logs.log"
        logger.add_file_log(log_file)

        for k, v in asdict(self).items():
            logger.info(f"{k}: {v}")


def copy_config_file(config):
    os.makedirs(config.root_dir, exist_ok=True)
    config_file_path = __file__
    dest_path = os.path.join(config.root_dir, os.path.basename(config_file_path))
    shutil.copy2(config_file_path, dest_path)
    logger.info(f"Config file copied to: {dest_path}")


def validate_conf(conf: Config):
    assert conf.profiler.profile_section in ["train", "eval", "none"]
    assert not hasattr(conf, "do_profile")
    assert conf.eval_every >= conf.num_iters or \
        conf.eval_every % conf.save_every == 0, "must save at each eval"


def main(group: str):
    conf = Config()
    validate_conf(conf)

    copy_config_file(conf)
    # set seed
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    random.seed(conf.seed)

    if group == "train":
        assert conf.date != "best"
        trainer = Trainer(conf)
        trainer.train()
    else:
        # eval only
        evaluator = Evaluator(conf, group=group)
        evaluator.evaluate()


if __name__ == "__main__":
    import argparse
    # --group must be train/test/predict
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")

    parser.add_argument("--group", type=str, choices=["train", "test", "predict"], default="train")
    args = parser.parse_args()

    group = args.group  # type: ignore
    main(group)

