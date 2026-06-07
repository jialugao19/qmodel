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
from qmodel.models import build_model

from qmodel.config import QConfig

from line_profiler import profile
import time

class Trainer(AsyncTrainer):
    def __init__(self, config: QConfig) -> None:
        self.config = config

        device = config.device
        dtype  = config.train_dtype
        model     = build_model(config.model_class, config.model).to(device=device, dtype=dtype)
        loss_fn   = config.criterion
        lr        = config.learning_rate
        optimizer = config.optimizer_class(model.parameters(), lr=lr)
        lr_sched  = custom_scheduler(optimizer, config)
        amp_scaler  = MyScaler(config.use_amp, config.amp_dtype)

        do_profile  = config.profiler.profile_section == "train"
        profiler    = MyProfiler(real_run=do_profile, config=config.profiler)

        dataloader, sampler = setup_train_dataloader(config, group="train", shuffle=True)

        num_normal_steps = config.num_normal_steps if config.train_use_cuda_graph else 9999999999

        super().__init__(
            model=model, dataloader=dataloader, loss_fn=loss_fn, lr_sched=lr_sched, amp_scaler=amp_scaler, profiler=profiler,
            optimizer=optimizer, device=device, num_normal_steps=num_normal_steps
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
        res = self.run(n_iter=n_iter, start=self.start_iter, use_tqdm=False)
        self.writer.close()
        return res

    def _pre_step(self, curr_it):
        if curr_it % self.config.save_every == 0:
            self.checkpointer.save(
                self.model, self.optimizer, self.lr_sched,
                self.amp_scaler, curr_it
            )

        if self.config.eval_during and \
            curr_it % self.config.eval_every == 0:

            torch.cuda.empty_cache()
            metrics = self.evaluator.eval_partial(
                curr_it, self.config.eval_during_num_iters
            )
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
            self.writer.add_scalar('Loss/TrainMean', np.nanmean(res), curr_it)
            self._log_param_stats(curr_it)

    def _log_param_stats(self, curr_it: int):
        """Write per-parameter norm and gradient statistics to TensorBoard."""
        param_norms: list[tuple[str, float]] = []
        grad_norms: list[tuple[str, float]] = []

        for name, param in self.model.named_parameters():
            if param is None:
                continue

            tensor = param.detach()
            if tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
            norm = torch.linalg.vector_norm(tensor).item()
            param_norms.append((name, norm))

            grad = param.grad
            if grad is not None:
                grad_tensor = grad.detach()
                if grad_tensor.dtype != torch.float32:
                    grad_tensor = grad_tensor.to(dtype=torch.float32)
                grad_norm = torch.linalg.vector_norm(grad_tensor).item()
                grad_norms.append((name, grad_norm))

        for name, value in param_norms:
            self.writer.add_scalar(f'ParamNorm/{name}', value, curr_it)

        for name, value in grad_norms:
            self.writer.add_scalar(f'GradNorm/{name}', value, curr_it)

        if param_norms:
            values = [v for _, v in param_norms]
            self.writer.add_scalar('ParamNorm/max', max(values), curr_it)
            self.writer.add_scalar('ParamNorm/min', min(values), curr_it)
            self.writer.add_scalar('ParamNorm/mean', float(np.mean(values)), curr_it)

        if grad_norms:
            gvalues = [v for _, v in grad_norms]
            self.writer.add_scalar('GradNorm/max', max(gvalues), curr_it)
            self.writer.add_scalar('GradNorm/min', min(gvalues), curr_it)
            self.writer.add_scalar('GradNorm/mean', float(np.mean(gvalues)), curr_it)

        amp_scale = self.amp_scaler.get_scale()
        self.writer.add_scalar('AMP/scale', amp_scale, curr_it)
