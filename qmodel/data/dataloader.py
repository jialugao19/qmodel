
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from qmodel.data.sampler import CustomSampler
from qmodel.config import QConfig

def custom_collate_fn(batch):
    return batch

def setup_train_dataloader(config: QConfig, group: str, shuffle: bool = True) -> Tuple[DataLoader, CustomSampler]:
    dtype     = config.train_dtype
    dataset   = config.dataset_class(group, dtype)
    seed      = config.seed
    bs        = config.batch_size
    n_workers = config.num_workers

    sampler = CustomSampler(dataset, seed=seed, shuffle=shuffle, infinite=True)
    return DataLoader(
        dataset,
        batch_size=bs,
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=n_workers,
        pin_memory=True
    ), sampler


def setup_eval_dataloader(config: QConfig, group, shuffle=False):
    dtype = config.eval_dtype
    dataset = config.dataset_class(group, dtype)
    seed = config.seed
    sampler = CustomSampler(dataset, seed=seed, shuffle=shuffle, infinite=False)
    return DataLoader(
        dataset,
        batch_size=config.evaluator.eval_batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
