import pytest
import torch
import torch.nn as nn

from qmodel.components.amp_scaler import MyScaler
from qmodel.components.lr_scheduler import GraphLRScheduler
from qmodel.components.profiler import MyProfiler
from qmodel.config import ProfilerConfig
from qmodel.core.eval_trainer import AsyncInference, AsyncTrainer, SyncInference, SyncTrainer


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="test_multi_eval_trainer requires at least two visible CUDA devices",
)


class Result(nn.Module):
    def forward(self, x):
        """Return the final GRU hidden state."""
        # Select the last layer and last feature slice.
        return x[1][-1, :, -1:]


def get_model(dtype: torch.dtype, device: torch.device) -> nn.Module:
    """Build one deterministic GRU test model."""
    # Reset model initialization and place the model on the requested GPU.
    torch.random.manual_seed(42)
    return nn.Sequential(
        *[nn.GRU(8, 64, 5, batch_first=True), Result()]
    ).to(dtype=dtype, device=device)


def get_data(num_samples: int, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic CPU training tensors."""
    # Reset data generation and create fixed CPU tensors.
    torch.random.manual_seed(seed)
    data_tensor = torch.randn(num_samples, 32, 8).to(dtype=dtype)
    target_tensor = torch.randn(num_samples, 1).to(dtype=dtype)
    return data_tensor, target_tensor


def get_dataloader(num_samples: int, bs: int, dtype: torch.dtype, seed: int):
    """Yield pinned batches for multi-GPU trainer tests."""
    # Slice fixed tensors into pinned CPU batches.
    data_tensor, target_tensor = get_data(num_samples, dtype, seed)
    n_batch = num_samples // bs
    return (
        (
            data_tensor[batch_idx*bs:(batch_idx+1)*bs].pin_memory(),
            target_tensor[batch_idx*bs:(batch_idx+1)*bs].pin_memory(),
            target_tensor[batch_idx*bs:(batch_idx+1)*bs],
        )
        for batch_idx in range(n_batch)
    )


def _post_step_eval(res, buffer, target, other_meta, curr_it):
    """Collect one inference result."""
    # Store prediction, target, and metadata for later comparison.
    res.append([buffer, target.ravel(), other_meta])


def _post_step_train(res: list, buffer: torch.Tensor, target, other_meta, curr_it):
    """Collect one training loss."""
    # Store detached CPU loss from the trainer post-step.
    res.append(buffer)


def get_graph_scheduler(optimizer, lr: float, device: torch.device) -> GraphLRScheduler:
    """Build the graph-compatible scheduler used by the sync trainer."""
    # Match the scheduler shape from the strict single-GPU test.
    return GraphLRScheduler(
        optimizer=optimizer,
        warmup_iters=19,
        total_iters=160,
        start_factor=0.01,
        end_factor=0.95,
        eta_min=1e-4,
        base_lr=lr,
        device=device,
    )


def _assert_zero_error(left: torch.Tensor, right: torch.Tensor):
    """Assert tensors are exactly equal."""
    # Compare CPU copies so tensors from different GPUs can be checked directly.
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    if not torch.equal(left_cpu, right_cpu):
        max_abs = (left_cpu - right_cpu).abs().max().item()
        assert False, max_abs


def test_multi_gpu_eval_trainer_has_zero_error():
    """Compare sync and async trainers across two GPUs."""
    # Configure the deterministic two-GPU workload.
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
    n_sample = 5120
    bs = 32
    seed = 42
    dtype = torch.float64
    lr = 1e-2

    # Build identical models and optimizers on separate GPUs.
    model1 = get_model(dtype, device1)
    model2 = get_model(dtype, device2)
    optim1 = torch.optim.Adam(model1.parameters(), lr)
    optim2 = torch.optim.Adam(model2.parameters(), lr)
    sched1 = get_graph_scheduler(optim1, lr, device1)
    sched2 = get_graph_scheduler(optim2, lr, device2)

    # Build scalers and profilers for train and eval paths.
    amp_dtype = torch.float32
    torch.cuda.set_device(device1)
    scaler1 = MyScaler("none", amp_dtype)
    torch.cuda.set_device(device2)
    scaler2 = MyScaler("none", amp_dtype)
    profiler1 = MyProfiler(False, ProfilerConfig("none", ""))
    profiler2 = MyProfiler(False, ProfilerConfig("none", ""))
    profiler3 = MyProfiler(False, ProfilerConfig("none", ""))
    profiler4 = MyProfiler(False, ProfilerConfig("none", ""))

    # Run sync training on GPU0.
    dataloader1 = get_dataloader(n_sample, bs, dtype, seed)
    trainer1 = SyncTrainer(
        model1, dataloader1, nn.MSELoss(), optim1, sched1, scaler1, profiler1, device1, 2000)
    trainer1._post_step = _post_step_train
    trainer1._pre_step = lambda curr_it: curr_it  # type: ignore
    res1 = trainer1.run(n_iter=n_sample//bs)

    # Run async training on GPU1.
    dataloader2 = get_dataloader(n_sample, bs, dtype, seed)
    trainer2 = AsyncTrainer(
        model2, dataloader2, nn.MSELoss(), optim2, sched2, scaler2, profiler2, device2, 2000)
    trainer2._post_step = _post_step_train
    trainer2._pre_step = lambda curr_it: curr_it  # type: ignore
    res2 = trainer2.run(n_iter=n_sample//bs)

    # Require zero drift in train losses and final parameters.
    train_loss1 = torch.utils.data.default_collate(res1)
    train_loss2 = torch.utils.data.default_collate(res2)
    _assert_zero_error(train_loss1, train_loss2)
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        _assert_zero_error(param1, param2)

    # Run inference with both trained models on their own GPUs.
    dataloader1 = get_dataloader(n_sample, bs, dtype, seed)
    eval1 = SyncInference(model1, dataloader1, scaler1, profiler3, device1, 2000)
    eval1._post_step = _post_step_eval
    eval1._pre_step = lambda curr_it: curr_it  # type: ignore
    eval_res1 = eval1.run(n_iter=n_sample//bs)

    dataloader2 = get_dataloader(n_sample, bs, dtype, seed)
    eval2 = AsyncInference(model2, dataloader2, scaler2, profiler4, device2, 2000)
    eval2._post_step = _post_step_eval
    eval2._pre_step = lambda curr_it: curr_it  # type: ignore
    eval_res2 = eval2.run(n_iter=n_sample//bs)

    # Require zero drift in inference outputs.
    pred1 = torch.utils.data.default_collate(eval_res1)[0]
    pred2 = torch.utils.data.default_collate(eval_res2)[0]
    _assert_zero_error(pred1, pred2)
