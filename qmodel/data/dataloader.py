
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from qmodel.components.distributed import DistributedContext
from qmodel.data.sampler import CustomSampler
from qmodel.config import QConfig

def custom_collate_fn(batch):
    """Keep dataset batch objects unchanged."""
    # The datasets already return batched tensors through __getitems__.
    return batch

def setup_train_dataloader(
    config: QConfig,
    group: str,
    shuffle: bool = True,
    distributed: DistributedContext | None = None,
) -> Tuple[DataLoader, CustomSampler]:
    """Build a train dataloader with global-batch rank sectioning."""
    # Resolve distributed geometry and local batch size.
    if distributed is None:
        distributed = DistributedContext.single(config.device)
    dtype     = config.train_dtype
    dataset   = config.dataset_class(group, dtype)
    seed      = config.seed
    bs        = config.batch_size
    assert bs % distributed.world_size == 0
    local_bs  = bs // distributed.world_size
    n_workers = config.num_workers

    # Build a sampler that yields only this rank's contiguous section.
    sampler = CustomSampler(
        dataset,
        seed=seed,
        shuffle=shuffle,
        infinite=True,
        global_batch_size=bs,
        rank=distributed.rank,
        world_size=distributed.world_size,
        drop_last=True,
    )
    return DataLoader(
        dataset,
        batch_size=local_bs,
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=n_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    ), sampler


def setup_eval_dataloader(
    config: QConfig,
    group,
    shuffle=False,
    distributed: DistributedContext | None = None,
):
    """Build an eval dataloader with global-batch rank sectioning."""
    # Resolve distributed geometry and local eval batch size.
    if distributed is None:
        distributed = DistributedContext.single(config.device)
    dtype = config.eval_dtype
    dataset = config.dataset_class(group, dtype)
    seed = config.seed
    bs = config.evaluator.eval_batch_size
    assert bs % distributed.world_size == 0
    local_bs = bs // distributed.world_size

    # Keep final partial eval batches so rank0 can reconstruct all rows.
    sampler = CustomSampler(
        dataset,
        seed=seed,
        shuffle=shuffle,
        infinite=False,
        global_batch_size=bs,
        rank=distributed.rank,
        world_size=distributed.world_size,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_size=local_bs,
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
