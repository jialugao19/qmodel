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
from qmodel.components.checkpoint import CheckpointSaver, capture_rng, restore_rng
from qmodel.components.amp_scaler import MyScaler
from qmodel.components.profiler import MyProfiler
from qmodel.data.dataloader import setup_train_dataloader
from qmodel.logger import logger
from qmodel.models import build_model

from qmodel.config import QConfig

from line_profiler import profile
import time
import pickle

class Trainer(AsyncTrainer):
    def __init__(self, config: QConfig) -> None:
        """Build the training runtime from config."""
        # Initialize distributed placement before constructing CUDA modules.
        restored = CheckpointSaver.latest(config.root_dir)
        if restored is not None:
            config = restored[2]["config"]
        self.signal = 0
        self.global_step = 0
        self.consumed_batches = 0
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
        if config.train_use_cuda_graph and self.distributed.enabled:
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

        self.pending_restore = restored

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

    def restore_training_state(self):
        # Delay project restoration until subclass initialization is complete.
        self.load_from_checkpoint(self.pending_restore)
        self.pending_restore = None

    def load_from_checkpoint(self, restored=None):
        # Restore objects before creating the loader iterator.
        if restored is None:
            return
        _, path, state = restored
        self.checkpointer.load(
            path, self.raw_model, self.optimizer, self.lr_sched, self.amp_scaler
        )
        self.global_step = state.get("global_step", state["iteration"])
        if "progress" in state:
            progress = state["progress"]
            if len(progress["ranks"]) != self.distributed.world_size:
                raise ValueError("Checkpoint world size differs from current runtime")
            local = progress["ranks"][self.distributed.rank]
            self.consumed_batches = local["sampler"]["consumed_batches"]
            self.sampler.load_state_dict(local["sampler"])
            self.raw_model.load_state_dict(local["model"])
            if state.get("format_version") == 2:
                self.statistics = local["statistics"]
                self.load_project_state(local["project"])
            else:
                logger.warning("Legacy checkpoint: statistics and project state are missing")
            restore_rng(local["rng"])
        else:
            logger.warning("Legacy checkpoint: RNG state is missing; exact continuation is not guaranteed")
            self.consumed_batches = state["iteration"]
            self.sampler.set_start_iter(self.consumed_batches * self.config.batch_size)
        self.start_iter = self.consumed_batches
        # Warm the new DDP reducer with real updates before recapturing NCCL.
        if self.config.train_use_cuda_graph and self.distributed.enabled:
            self.num_normal_steps += self.start_iter
        self.distributed.barrier()

    def train(self):
        # Keep the configured total batch budget across restarts.
        n_iter = min(len(self.dataloader), self.config.num_iters)
        res = self.run(n_iter=n_iter, start=self.start_iter, use_tqdm=False)
        return res

    def close_training_resources(self):
        # Close persistent logging before returning a pause result.
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            if (self.pause_requested or self.run_failed) and self.distributed.owns_process_group:
                torch.distributed.destroy_process_group()

    def _pre_step(self, curr_it):
        # Saving and evaluation belong to the completed update boundary.
        pass

    def _completed_step(self, curr_it, post):
        # Count consumed batches separately from successful optimizer updates.
        self.consumed_batches = curr_it + 1
        if self.update_completed:
            self.global_step += 1
        request = self.signal if self.distributed.is_main else 0
        if self.distributed.enabled:
            control = torch.tensor(request, device=self.device)
            torch.distributed.broadcast(control, src=0)
            request = control.item()
        periodic = self.update_completed and self.global_step % self.config.save_every == 0
        evaluate = self.update_completed and self.config.eval_during and self.global_step % self.config.eval_every == 0
        if not self.update_completed or not (request or periodic or evaluate):
            return

        # Drain logging and GPU work before collecting the immutable boundary.
        export_error = None
        try:
            if post is not None:
                post.result()
                self._check_completed_posts()
            torch.cuda.synchronize(self.device)
            local = {
                "project": self.get_project_state(),
                "statistics": list(self.statistics),
                "rng": capture_rng(),
                "sampler": self.sampler.state_dict(self.consumed_batches),
                "model": {key: value.detach().cpu().clone() for key, value in self.raw_model.state_dict().items()},
            }
            if self.distributed.enabled:
                pickle.dumps(local)
        except Exception as failure:
            export_error = failure
        if self.distributed.enabled:
            errors = [None] * self.distributed.world_size
            torch.distributed.all_gather_object(errors, repr(export_error) if export_error else None)
            if any(error is not None for error in errors) and export_error is None:
                raise RuntimeError(f"Checkpoint state export failed: {errors}")
        if export_error is not None:
            raise export_error
        ranks = self.distributed.gather_object(local)
        error = None
        if self.distributed.is_main:
            try:
                self.checkpoint_path = self.checkpointer.save(
                    self.raw_model, self.optimizer, self.lr_sched,
                    self.amp_scaler, self.global_step, {"ranks": ranks}
                )
            except Exception as failure:
                error = failure
        if self.distributed.enabled:
            status = [repr(error) if error is not None else None]
            torch.distributed.broadcast_object_list(status, src=0)
            if status[0] is not None and error is None:
                raise RuntimeError(f"Rank zero checkpoint save failed: {status[0]}")
        if error is not None:
            raise error
        paths = [self.checkpoint_path]
        if self.distributed.enabled:
            torch.distributed.broadcast_object_list(paths, src=0)
        self.checkpoint_path = paths[0]
        if request:
            self.pause_requested = True
            return True

        # Evaluation must not perturb the subsequent training random sequence.
        if evaluate:
            metrics = self.evaluator.eval_partial(self.global_step, self.config.eval_during_num_iters)
            restore_rng(local["rng"])
            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"{self.config.ret_col_name}/{key}", value, self.global_step)

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
