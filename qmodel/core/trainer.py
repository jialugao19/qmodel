import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from qmodel.core.eval_trainer import AsyncTrainer, SyncTrainer
from qmodel.core.evaluator import Evaluator
from qmodel.components.lr_scheduler import custom_scheduler
from qmodel.components.checkpoint import CheckpointSaver
from qmodel.components.amp_scaler import MyScaler
from qmodel.components.profiler import MyProfiler
from qmodel.data.dataloader import setup_train_dataloader
from qmodel.logger import logger

from qmodel.config import QConfig

from line_profiler import profile
import time

class Trainer(AsyncTrainer):
    def __init__(self, config: QConfig) -> None:
        self.config = config

        device = config.device
        model  = config.model_class(config.model).to(device)

        loss_fn   = config.criterion
        lr        = config.learning_rate
        optimizer = config.optimizer_class(model.parameters(), lr=lr)
        lr_sched  = custom_scheduler(optimizer, config.lr_scheduler)
        amp_scaler  = MyScaler(config.amp_dtype, config.train_dtype)

        do_profile  = config.profiler.profile_section == "train"
        profiler    = MyProfiler(real_run=do_profile, config=config.profiler)

        dataloader, sampler = setup_train_dataloader(config, group="train", shuffle=True)

        super().__init__(
            model=model, dataloader=dataloader, loss_fn=loss_fn, lr_sched=lr_sched, amp_scaler=amp_scaler, profiler=profiler,
            optimizer=optimizer, device=device
        )

        self.evaluator = Evaluator(config, group="test")

        os.makedirs(config.root_dir, exist_ok=True)
        os.makedirs(config.tensorboard_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=config.tensorboard_dir)
        self.checkpointer = CheckpointSaver(config.root_dir, config, device)
        self.sampler = sampler
        self.start_iter = 0

        self.load_from_checkpoint()

    def load_from_checkpoint(self):
        if self.config.load_from_iter is None:
            return
        iteration: int = self.config.load_from_iter
        path = Path(self.config.root_dir) / f"ckpt/iter_{iteration}.pt"

        logger.info(f"load from iteration {iteration}")
        self.checkpointer.load(
            path, self.model, self.optimizer, self.lr_sched, self.amp_scaler
        )
        self.sampler.set_start_iter(iteration * self.config.batch_size)
        self.start_iter = iteration

    def train(self):
        assert isinstance(self.dataloader, torch.utils.data.DataLoader)
        n_iter = min(len(self.dataloader), self.config.num_iters)
        self.run(n_iter=n_iter, start=self.start_iter)
        self.writer.close()

    def _pre_step(self, curr_it):
        if curr_it % self.config.save_every == 0:
            self.checkpointer.save(
                self.model, self.optimizer, self.lr_sched,
                self.amp_scaler, curr_it
            )

        if 0 and curr_it % self.config.eval_every == 0:
            torch.cuda.empty_cache()
            metrics = self.evaluator.eval_single(curr_it)
            torch.cuda.empty_cache()

            ret_col = self.config.ret_col_name
            for k, v in metrics.items():
                self.writer.add_scalar(f'{ret_col}/{k}', v, curr_it)

    def _post_step(self, res, buffer, target, other_meta, curr_it):

        loss_length = self.config.mean_loss_length
        if len(res) < loss_length:
            res.extend([np.nan for _ in range(loss_length-len(res))])

        loss_val = buffer.item()
        res[curr_it % loss_length] = loss_val
        if curr_it % self.config.log_every == 0:
            logger.info(
                f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}, '
                f'Batch {curr_it}: Loss: {loss_val:.4f}, Mean loss: {np.nanmean(res):.4f}'
            )

            self.writer.add_scalar('Loss/Train', loss_val, curr_it)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], curr_it)
