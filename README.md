# qmodel

`qmodel` is a lightweight Python package that factors out the reusable pieces of the
`AQ2/modeling` module into a standalone training and evaluation toolkit. It offers:

- a configuration-driven training loop with optional checkpointing and TensorBoard logging,
- evaluation helpers that accept user-specified metric functions,
- utilities such as gradient scaling wrappers, simple profilers, and deterministic samplers.

## Quick start

```python
import torch
from torch import nn
from torch.utils.data import Dataset

from qmodel import TrainEvalConfig, Trainer
from qmodel.lr_scheduler import build_warmup_cosine_scheduler


class ToyDataset(Dataset):
    def __init__(self, group: str):
        self.group = group
        torch.manual_seed(0)
        self.x = torch.randn(4096, 10)
        self.w = torch.randn(10, 1)
        noise = 0.1 * torch.randn(4096, 1)
        self.y = self.x @ self.w + noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def dataset_factory(group: str):
    return ToyDataset(group)


def model_builder(cfg: TrainEvalConfig):
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))


def optimizer_builder(params):
    return torch.optim.Adam(params, lr=1e-3)


def scheduler_builder(optimizer):
    return build_warmup_cosine_scheduler(optimizer)


config = TrainEvalConfig(
    dataset_factory=dataset_factory,
    model_builder=model_builder,
    loss_fn=nn.MSELoss(),
    optimizer_builder=optimizer_builder,
    scheduler_builder=scheduler_builder,
    batch_size=64,
    num_iters=200,
    eval_every=50,
    save_every=100,
    extra={
        "metrics": {
            "mse": lambda pred, target: torch.nn.functional.mse_loss(pred, target).item(),
        },
        "eval_group": "test",
    },
)

trainer = Trainer(config)
trainer.train()
```

Check `examples/` and `tests/` for additional reference implementations using synthetic data.

## Running tests

```
cd qmodel
python -m pytest
```
