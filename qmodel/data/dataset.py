import torch

from typing import Protocol, Any

class QDataset(Protocol):

    def __init__(self, group: str, dtype: torch.dtype, *args, **kwargs):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index) -> Any:
        ...

    def __getitems__(self, indices) -> Any:
        ...

