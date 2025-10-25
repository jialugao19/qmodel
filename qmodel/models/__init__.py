import torch.nn as nn
from typing import Protocol
from qmodel.config import ModelConfig

class QModel(Protocol):
    def __init__(self, config: ModelConfig):
        pass

    def forward(self, x):
        pass
