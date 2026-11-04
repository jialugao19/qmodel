import torch
import torch.nn as nn
import numpy as np
import os
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from qmodel.core.eval_trainer import AsyncTrainer, SyncTrainer
from qmodel.core.evaluator import Evaluator
from qmodel.components.distributed import DistributedContext
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
        """Build the training runtime from config."""
        # Initialize distributed placement before constructing CUDA modules.
        self.config = config
        self.distributed = DistributedContext.from_config(config)
        self.config.device = self.distributed.device
        self.ddp_graph_stream: torch.cuda.Stream | None = None

        # Build model, optimizer, scheduler, scaler, and profiler.
        device = self.distributed.device
        dtype  = config.train_dtype
        raw_model = build_model(config.model_class, config.model).to(device=device, dtype=dtype)
        loss_fn   = config.criterion
        lr        = config.learning_rate
        optimizer = config.optimizer_class(raw_model.parameters(), lr=lr)
        lr_sched  = custom_scheduler(optimizer, config)
        amp_scaler  = MyScaler(config.use_amp, config.amp_dtype)

        # Validate graph-only component choices before capture.
        if config.train_use_cuda_graph:
            assert config.use_lr_sched == "custom"
            assert config.use_amp in ["custom", "none"]

        # Wrap distributed training with the DDP reducer in every mode.
        model: nn.Module = raw_model
        if self.distributed.enabled:
            model = self._wrap_distributed_model(raw_model)

        # Build the profiler and rank-local dataloader.
        do_profile  = config.profiler.profile_section == "train"
        profiler    = MyProfiler(real_run=do_profile, config=config.profiler)

        dataloader, sampler = setup_train_dataloader(
            config,
            group="train",
            shuffle=True,
            distributed=self.distributed,
        )

        # Configure CUDA graph warmup behavior.
        num_normal_steps = config.num_normal_steps if config.train_use_cuda_graph else 9999999999

        super().__init__(
            model=model, dataloader=dataloader, loss_fn=loss_fn, lr_sched=lr_sched, amp_scaler=amp_scaler, profiler=profiler,
            optimizer=optimizer, device=device, num_normal_steps=num_normal_steps,
            distributed=self.distributed,
        )

        # Reuse the DDP construction stream for graph warmup and capture.
        if config.train_use_cuda_graph:
            assert self.ddp_graph_stream is not None
            self.model_stream = self.ddp_graph_stream

        # Store raw model and evaluator for checkpoint/eval paths.
        self.raw_model = raw_model
        self.graph_model = raw_model
        self.evaluator = Evaluator(config, group="test", distributed=self.distributed)

        # Create persistent writers on rank zero.
        if self.distributed.is_main:
            os.makedirs(config.root_dir, exist_ok=True)
            os.makedirs(config.tensorboard_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=config.tensorboard_dir) if self.distributed.is_main else None
        self.checkpointer = CheckpointSaver(config.root_dir, config, device)
        self.sampler = sampler
        self.start_iter = 0

        self.load_from_checkpoint()

    def _wrap_distributed_model(self, raw_model: nn.Module) -> DistributedDataParallel:
        """Wrap the raw model with DDP using graph-safe construction."""
        # Construct graph-captured DDP on a side stream as required by PyTorch.
        if self.config.train_use_cuda_graph:
            stream = torch.cuda.Stream(device=self.distributed.device)
            self.ddp_graph_stream = stream
            stream.wait_stream(torch.cuda.current_stream(self.distributed.device))
            with torch.cuda.stream(stream):
                model = DistributedDataParallel(
                    raw_model,
                    device_ids=[self.distributed.local_rank],
                    output_device=self.distributed.local_rank,
                    static_graph=True,
                )
            torch.cuda.current_stream(self.distributed.device).wait_stream(stream)
            return model

        # Construct normal eager DDP on the current stream.
        return DistributedDataParallel(
            raw_model,
            device_ids=[self.distributed.local_rank],
            output_device=self.distributed.local_rank,
        )

    def load_from_checkpoint(self):
        if self.config.load_from_iter is None:
            return
        iteration: int = self.config.load_from_iter
        path = Path(self.config.root_dir) / f"ckpt/iter_{iteration}.pt"

        logger.info(f"load from iteration {iteration}")
        self.checkpointer.load(
            path, self.raw_model, self.optimizer, self.lr_sched, self.amp_scaler
        )
        self.sampler.set_start_iter(iteration * self.config.batch_size)
        self.start_iter = iteration
        self.distributed.barrier()

    def train(self):
        assert isinstance(self.dataloader, torch.utils.data.DataLoader)
        n_iter = min(len(self.dataloader), self.config.num_iters)
        res = self.run(n_iter=n_iter, start=self.start_iter, use_tqdm=False)
        if self.writer is not None:
            self.writer.close()
        return res

    def _pre_step(self, curr_it):
        if curr_it % self.config.save_every == 0:
            # Finish async training work before materializing checkpoint tensors.
            torch.cuda.synchronize(self.device)
            if self.distributed.is_main:
                self.checkpointer.save(
                    self.raw_model, self.optimizer, self.lr_sched,
                    self.amp_scaler, curr_it
                )
            self.distributed.barrier()

        if self.config.eval_during and \
            curr_it % self.config.eval_every == 0:

            torch.cuda.empty_cache()
            metrics = self.evaluator.eval_partial(
                curr_it, self.config.eval_during_num_iters
            )
            torch.cuda.empty_cache()

            if self.distributed.is_main:
                assert self.writer is not None
                ret_col = self.config.ret_col_name
                for k, v in metrics.items():
                    self.writer.add_scalar(f'{ret_col}/{k}', v, curr_it)

    def _post_step(self, res, buffer, target, other_meta, curr_it):
        """Log rank-zero train metrics after one train step."""
        # Keep persistent logging on rank zero only.
        if not self.distributed.is_main:
            return

        # Update the rolling loss window.
        loss_length = self.config.mean_loss_length
        if len(res) < loss_length:
            res.extend([np.nan for _ in range(loss_length-len(res))])

        loss_val = buffer.item()
        res[curr_it % loss_length] = loss_val
        if curr_it % self.config.log_every == 0:
            assert self.writer is not None

            # Write scalar train metrics.
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
        # Collect raw model parameter and gradient norms.
        assert self.writer is not None
        param_norms: list[tuple[str, float]] = []
        grad_norms: list[tuple[str, float]] = []

        for name, param in self.raw_model.named_parameters():
            if param is None:
                continue

            # Convert parameter norm to a scalar for TensorBoard.
            tensor = param.detach()
            if tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
            norm = torch.linalg.vector_norm(tensor).item()
            param_norms.append((name, norm))

            # Convert gradient norm to a scalar for TensorBoard.
            grad = param.grad
            if grad is not None:
                grad_tensor = grad.detach()
                if grad_tensor.dtype != torch.float32:
                    grad_tensor = grad_tensor.to(dtype=torch.float32)
                grad_norm = torch.linalg.vector_norm(grad_tensor).item()
                grad_norms.append((name, grad_norm))

        # Write per-parameter metrics.
        for name, value in param_norms:
            self.writer.add_scalar(f'ParamNorm/{name}', value, curr_it)

        for name, value in grad_norms:
            self.writer.add_scalar(f'GradNorm/{name}', value, curr_it)

        # Write aggregate parameter norm metrics.
        if param_norms:
            values = [v for _, v in param_norms]
            self.writer.add_scalar('ParamNorm/max', max(values), curr_it)
            self.writer.add_scalar('ParamNorm/min', min(values), curr_it)
            self.writer.add_scalar('ParamNorm/mean', float(np.mean(values)), curr_it)

        # Write aggregate gradient norm metrics.
        if grad_norms:
            gvalues = [v for _, v in grad_norms]
            self.writer.add_scalar('GradNorm/max', max(gvalues), curr_it)
            self.writer.add_scalar('GradNorm/min', min(gvalues), curr_it)
            self.writer.add_scalar('GradNorm/mean', float(np.mean(gvalues)), curr_it)

        # Write AMP scale for the current train step.
        amp_scale = self.amp_scaler.get_scale()
        self.writer.add_scalar('AMP/scale', amp_scale, curr_it)
