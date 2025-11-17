import torch.nn as nn
from typing import Protocol, Type, TypeVar, cast
from qmodel.config import ModelConfig

from typing_extensions import TypeAlias

class InitWithConf(Protocol):
    def __init__(self, config: ModelConfig) -> None:
        ...



T = TypeVar("T", bound=nn.Module)

def build_model(model_cls: Type[T], cfg: ModelConfig) -> T:
    # check model_cls's __init__(cfg)
    checked = cast(Type[InitWithConf], model_cls)

    # cast back to T（nn.Module）
    return cast(T, checked(cfg))
