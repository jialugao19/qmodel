import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from qmodel.config import QConfig


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @classmethod
    def from_config(cls, config: QConfig) -> "DistributedContext":
        """Build distributed runtime context from config."""
        # Use the existing single-process device when DDP is disabled.
        if not config.distributed.enabled:
            return cls.single(config.device)

        # Disable PGNCCL watchdog handling before graph-captured DDP collectives.
        if config.train_use_cuda_graph and config.distributed.backend == "nccl":
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        # Read torchrun process placement and initialize NCCL.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend=config.distributed.backend)
        return cls(True, rank, world_size, local_rank, device)

    @classmethod
    def single(cls, device: torch.device | str) -> "DistributedContext":
        """Build a single-process context."""
        # Represent single GPU execution with rank zero.
        return cls(False, 0, 1, 0, torch.device(device))

    @property
    def is_main(self) -> bool:
        """Return whether this rank owns persistent output writes."""
        # Rank zero is the only persistent writer.
        return self.rank == 0

    def barrier(self):
        """Synchronize all ranks when DDP is enabled."""
        # DDP collectives are only valid in distributed mode.
        if self.enabled:
            dist.barrier(device_ids=[self.local_rank])

    def average(self, tensor: torch.Tensor) -> torch.Tensor:
        """Average a tensor across all ranks."""
        # Reduce the tensor in place and scale to the global mean.
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(self.world_size)
        return tensor

    def average_gradients(self, parameters):
        """Average parameter gradients across all ranks."""
        # Reduce every materialized gradient in place.
        if self.enabled:
            for parameter in parameters:
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(self.world_size)

    def broadcast_module(self, module: torch.nn.Module):
        """Broadcast module parameters and buffers from rank zero."""
        # Synchronize trainable parameters.
        if self.enabled:
            for parameter in module.parameters():
                dist.broadcast(parameter.data, src=0)

            # Synchronize module buffers.
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=0)

    def gather_object(self, value):
        """Gather one Python object per rank onto rank zero."""
        # Single-process execution already has the full result.
        if not self.enabled:
            return [value]

        # Gather rank-local objects to the main process.
        output = [None for _ in range(self.world_size)] if self.is_main else None
        dist.gather_object(value, output, dst=0)
        if self.is_main:
            return output
        return []
