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

from qmodel.models import *
from qmodel.trainer import Trainer
from qmodel.evaluator import Evaluator
from qmodel.data.dataset import AQ2Dataset

from qmodel.logger import logger

@dataclass
class Config:

    input_dim = 8
    output_dim = 1
    gru_hidden_dim = 192
    gru_num_layers = 4
    gru_bidirectional = False
    gru_dropout = 0.0

    dense_dropout = 0.0
    dense_hidden_dim = 512

    window_size = 32

    # conv_out_channels = 64
    # conv_kernel_size = 3
    # conv_stride = 2
    # conv_dropout = 0.0

    # model = ConvGRU1
    model = GRU3
    seed = 42

    date = f"{time.strftime('%m%d')}"
    expr_name = f"AQ2_1min_convgru1_clip_{time.strftime('%m%d_%H%M%S')}"

    # expr_name = f"AQ2_1min_convgru1_clip_0920_160131"

    # date = "best"
    # expr_name = "best2"

    # load_from_iter = 5
    load_from_iter = None

    batch_size = 1024
    eval_batch_size = 16384

    num_workers = 32

    num_iters = 1 * 10**6 + 1

    device = "cuda"
    amp_dtype = torch.float16  # or None
    eval_dtype = torch.float16

    criterion = nn.HuberLoss()
    optimizer_class = optim.Adam

    learning_rate = 5e-4
    warmup_iters = 1000
    finish_decay_iter = 10**6
    start_warmup_factor = 0.001
    end_warmup_factor = 1.0
    eta_min = 1e-6

    root_dir = os.path.join('train_logs', date, expr_name)
    save_every = 10**4
    eval_every = 10000

    # num_iters during training / eval only
    eval_num_iters = 100
    eval_all_num_iters = 0   # 0 means all
    # before that, need to run sync trainer to verify result

    profile_section = "none"  # must be in "train" / "eval" / "none"
    tensorboard_dir = os.path.join('runs', date, expr_name)
    profile_dir     = os.path.join(tensorboard_dir, "profile")
    log_every = 1000
    mean_loss_length = 2000

    eval_checkpoint_iter = np.arange(0, num_iters - 1, eval_every).tolist()[::-1][:1]
    ret_col_name = "ret_1m_5m"
    dataset_class = partial(AQ2Dataset, ret_col_name=ret_col_name, window_size=window_size)

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
    assert conf.profile_section in ["train", "eval", "none"]
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

