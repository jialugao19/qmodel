import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from qmodel import Trainer
from qmodel.config import (
    DistributedConfig,
    EvalConfig,
    LRSchedulerConfig,
    ModelConfig,
    ProfilerConfig,
    QConfig,
)


PARITY_WORLD_SIZE = 3
PARITY_LOCAL_BATCH_SIZE = 8
PARITY_GLOBAL_BATCH_SIZE = PARITY_WORLD_SIZE * PARITY_LOCAL_BATCH_SIZE
PARITY_NUM_ITERS = 5
PARITY_TOLERANCE = 1e-10
PARITY_SINGLE_ROOT = Path("train_logs/ddp_parity_single")
PARITY_DDP_ROOT = Path("train_logs/ddp_parity_3gpu")
PARITY_FINAL_STATE = "final_model.pt"
CHECKPOINT_FULL_ROOT = Path("train_logs/ddp_checkpoint_full")
CHECKPOINT_SPLIT_ROOT = Path("train_logs/ddp_checkpoint_split")
CHECKPOINT_RESUME_ITER = 2
CHECKPOINT_NUM_ITERS = 5
CHECKPOINT_TOLERANCE = 1e-5


class RuntimeDataset(Dataset):
    def __init__(self, group: str, dtype: torch.dtype):
        """Build a deterministic map-style dataset."""
        # Store group and dtype for batched reads.
        assert group in ["train", "test", "predict"]
        self.group = group
        self.dtype = dtype
        self.n_items = 128

    def __len__(self) -> int:
        """Return dataset length."""
        # Expose a fixed number of global rows.
        return self.n_items

    def __getitems__(self, indices) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one batched sample tuple."""
        # Build deterministic features from global indices.
        idx = torch.as_tensor(indices, dtype=torch.float32)
        data = torch.stack([
            idx / 100.0,
            torch.sin(idx / 10.0),
            torch.cos(idx / 10.0),
            torch.ones_like(idx),
        ], dim=1).to(dtype=self.dtype)

        # Build target and metadata tensors.
        target = (data[:, :1].to(dtype=torch.float32) * 0.3 + data[:, 1:2].to(dtype=torch.float32) * 0.1)
        meta = torch.stack([idx, idx + 1, idx + 2], dim=1).to(dtype=torch.int32)
        return data, target, meta


@dataclass
class RuntimeModelConfig(ModelConfig):
    input_dim: int = 4
    hidden_dim: int = 8


class RuntimeModel(nn.Module):
    def __init__(self, config: RuntimeModelConfig):
        """Build a tiny regression model."""
        super().__init__()

        # Create a small network for DDP smoke training.
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the tiny regression model."""
        # Apply the model to one local batch.
        return self.net(data)


@dataclass
class RuntimeEvalConfig(EvalConfig):
    eval_checkpoint_iter: Sequence[int] = field(default_factory=lambda: [0])
    eval_all_num_iters: int = 2
    eval_batch_size: int = 8
    eval_use_cuda_graph: bool = False


@dataclass
class RuntimeConfig(QConfig):
    window_size: int = 1
    ret_col_name: str = "target"
    dataset_class: Callable = RuntimeDataset

    model: ModelConfig = field(default_factory=RuntimeModelConfig)
    model_class: type[nn.Module] = RuntimeModel
    seed: int = 123

    amp_dtype: torch.dtype = torch.float32
    eval_dtype: torch.dtype = torch.float32
    train_dtype: torch.dtype = torch.float32
    criterion: Callable = nn.MSELoss()
    optimizer_class: Callable = partial(optim.Adam, capturable=True)
    learning_rate: float = 1e-3
    use_amp: str = "none"
    use_lr_sched: str = "custom"

    train_use_cuda_graph: bool = False
    num_normal_steps: int = 12

    expr_name: str = "ddp_smoke"

    load_from_iter = None

    # Global training batch size; each rank uses batch_size // world_size.
    batch_size: int = 8
    num_workers: int = 0
    num_iters: int = 4
    save_every: int = 100000
    eval_every: int = 100000
    eval_during: bool = False
    eval_during_num_iters: int = 2

    log_every: int = 100000
    mean_loss_length: int = 4

    root_dir: str = "train_logs/ddp_smoke"
    tensorboard_dir: str = "runs/ddp_smoke"

    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig(
        warmup_iters=2,
        finish_decay_iter=64,
        eta_min=1e-5,
    ))
    profiler: ProfilerConfig = field(default_factory=lambda: ProfilerConfig("none", ""))
    evaluator: EvalConfig = field(default_factory=RuntimeEvalConfig)
    distributed: DistributedConfig = field(default_factory=lambda: DistributedConfig(enabled=True))


def _make_config(use_cuda_graph: bool) -> RuntimeConfig:
    """Create a DDP smoke config for one execution mode."""
    # Specialize runtime paths and iteration count by execution mode.
    mode = "graph" if use_cuda_graph else "eager"
    config = RuntimeConfig()
    config.train_use_cuda_graph = use_cuda_graph
    config.num_iters = 14 if use_cuda_graph else 4
    if use_cuda_graph:
        config.use_amp = "custom"
        config.amp_dtype = torch.float16
    config.root_dir = f"train_logs/ddp_smoke_{mode}"
    config.tensorboard_dir = f"runs/ddp_smoke_{mode}"
    config.profiler = ProfilerConfig("none", "")
    return config


def _make_parity_config(distributed_enabled: bool, root_dir: Path) -> RuntimeConfig:
    """Create a config for single-vs-three-GPU parity."""
    # Use the same global batch for single and distributed training.
    config = RuntimeConfig()
    config.distributed = DistributedConfig(enabled=distributed_enabled)
    config.batch_size = PARITY_GLOBAL_BATCH_SIZE
    config.num_iters = PARITY_NUM_ITERS
    config.root_dir = root_dir.as_posix()
    config.tensorboard_dir = Path("runs", root_dir.name).as_posix()

    # Use fp64 and plain SGD to make parity differences easy to interpret.
    config.train_dtype = torch.float64
    config.eval_dtype = torch.float64
    config.amp_dtype = torch.float32
    config.use_amp = "none"
    config.optimizer_class = optim.SGD
    config.learning_rate = 1e-2
    config.profiler = ProfilerConfig("none", "")
    return config


def _make_checkpoint_config(
    root_dir: Path,
    num_iters: int,
    load_from_iter: int | None,
    save_every: int,
) -> RuntimeConfig:
    """Create a three-GPU checkpoint/resume parity config."""
    # Reuse the deterministic three-GPU parity workload.
    config = _make_parity_config(distributed_enabled=True, root_dir=root_dir)
    config.num_iters = num_iters
    config.load_from_iter = load_from_iter
    config.save_every = save_every
    return config


def _save_final_state(trainer: Trainer, output_path: Path):
    """Save the trained raw model state on CPU."""
    # Copy tensors to CPU so the parent process can compare files cheaply.
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in trainer.raw_model.state_dict().items()
    }
    torch.save(state, output_path)


def _load_state_vector(path: Path) -> torch.Tensor:
    """Load a model state file into one fp64 vector."""
    # Flatten named tensors in sorted order for stable comparison.
    state = torch.load(path, map_location="cpu", weights_only=True)
    return torch.cat([
        state[name].to(dtype=torch.float64).reshape(-1)
        for name in sorted(state)
    ])


def _relative_l2(left_path: Path, right_path: Path) -> float:
    """Compute relative L2 between two saved model states."""
    # Compare the two flattened state vectors.
    left = _load_state_vector(left_path)
    right = _load_state_vector(right_path)
    return torch.linalg.vector_norm(left - right).item() / torch.linalg.vector_norm(left).item()


def _assert_state_file_equal(left_path: Path, right_path: Path):
    """Assert two saved model state files are exactly equal."""
    # Load state dictionaries from both training runs.
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    assert left.keys() == right.keys()

    # Require every tensor to match exactly.
    for name in sorted(left):
        left_tensor = left[name]
        right_tensor = right[name]
        assert left_tensor.dtype == right_tensor.dtype, name
        if not torch.equal(left_tensor, right_tensor):
            max_abs = (left_tensor.to(dtype=torch.float64) - right_tensor.to(dtype=torch.float64)).abs().max().item()
            assert False, f"{name}: {max_abs}"


def _checkpoint_model_relative_l2(left_path: Path, right_path: Path) -> float:
    """Compute relative L2 between model states inside two checkpoints."""
    # Load full checkpoint objects because the config is part of the payload.
    left_checkpoint = torch.load(left_path, map_location="cpu", weights_only=False)
    right_checkpoint = torch.load(right_path, map_location="cpu", weights_only=False)
    assert left_checkpoint["iteration"] == right_checkpoint["iteration"]

    # Compare checkpointed model tensors in sorted key order.
    left = torch.cat([
        left_checkpoint["model"][name].to(dtype=torch.float64).reshape(-1)
        for name in sorted(left_checkpoint["model"])
    ])
    right = torch.cat([
        right_checkpoint["model"][name].to(dtype=torch.float64).reshape(-1)
        for name in sorted(right_checkpoint["model"])
    ])
    return torch.linalg.vector_norm(left - right).item() / torch.linalg.vector_norm(left).item()


def _run_single_parity_reference():
    """Train the one-GPU reference with three times local batch size."""
    # Run the reference trainer with the same global batch seen by DDP.
    torch.manual_seed(123)
    config = _make_parity_config(distributed_enabled=False, root_dir=PARITY_SINGLE_ROOT)
    trainer = Trainer(config)
    trainer.train()
    _save_final_state(trainer, PARITY_SINGLE_ROOT / PARITY_FINAL_STATE)


def _run_ddp_three_card_parity():
    """Train the three-GPU DDP parity target."""
    # Run DDP with each rank reading its contiguous global-batch section.
    torch.manual_seed(123)
    config = _make_parity_config(distributed_enabled=True, root_dir=PARITY_DDP_ROOT)
    trainer = Trainer(config)
    trainer.train()

    # Persist rank-zero final parameters for parent-process comparison.
    trainer.distributed.barrier()
    if trainer.distributed.is_main:
        _save_final_state(trainer, PARITY_DDP_ROOT / PARITY_FINAL_STATE)
    trainer.distributed.barrier()
    torch.distributed.destroy_process_group()


def _run_ddp_checkpoint_full():
    """Train uninterrupted three-GPU DDP and save final state."""
    # Run the full training trajectory without checkpoint recovery.
    torch.manual_seed(123)
    config = _make_checkpoint_config(
        root_dir=CHECKPOINT_FULL_ROOT,
        num_iters=CHECKPOINT_NUM_ITERS,
        load_from_iter=None,
        save_every=CHECKPOINT_RESUME_ITER,
    )
    trainer = Trainer(config)
    trainer.train()

    # Save the rank-zero final model for parent-process comparison.
    trainer.distributed.barrier()
    if trainer.distributed.is_main:
        _save_final_state(trainer, CHECKPOINT_FULL_ROOT / PARITY_FINAL_STATE)
    trainer.distributed.barrier()
    torch.distributed.destroy_process_group()


def _run_ddp_checkpoint_first_segment():
    """Train until the resume checkpoint is written."""
    # Keep the full budget in config while interrupting at a completed boundary.
    torch.manual_seed(123)
    config = _make_checkpoint_config(
        root_dir=CHECKPOINT_SPLIT_ROOT,
        num_iters=CHECKPOINT_NUM_ITERS,
        load_from_iter=None,
        save_every=CHECKPOINT_RESUME_ITER,
    )
    trainer = Trainer(config)
    trainer.run(n_iter=CHECKPOINT_RESUME_ITER, start=0, use_tqdm=False)
    trainer.writer.close() if trainer.writer is not None else None

    # Ensure the checkpoint is visible before the next torchrun.
    trainer.distributed.barrier()
    if trainer.distributed.is_main:
        assert Path(config.root_dir, "ckpt", f"iter_{CHECKPOINT_RESUME_ITER}.pt").exists()
    trainer.distributed.barrier()
    torch.distributed.destroy_process_group()


def _run_ddp_checkpoint_resume():
    """Resume three-GPU DDP from checkpoint and save final state."""
    # Restore the checkpoint and continue to the same total iteration count.
    torch.manual_seed(123)
    config = _make_checkpoint_config(
        root_dir=CHECKPOINT_SPLIT_ROOT,
        num_iters=CHECKPOINT_NUM_ITERS,
        load_from_iter=CHECKPOINT_RESUME_ITER,
        save_every=100000,
    )
    trainer = Trainer(config)
    trainer.train()

    # Save the recovered final model for parent-process comparison.
    trainer.distributed.barrier()
    if trainer.distributed.is_main:
        _save_final_state(trainer, CHECKPOINT_SPLIT_ROOT / PARITY_FINAL_STATE)
    trainer.distributed.barrier()
    torch.distributed.destroy_process_group()


def _run_ddp_smoke(use_cuda_graph: bool, run_eval: bool):
    """Run one DDP trainer smoke case inside torchrun."""
    # Build and execute the trainer.
    torch.manual_seed(123)
    config = _make_config(use_cuda_graph)
    config.save_every = config.num_iters
    trainer = Trainer(config)
    res = trainer.train()

    # Check rank-zero training outputs.
    trainer.distributed.barrier()
    if trainer.distributed.is_main:
        assert len(res) > 0
        assert Path(config.root_dir, "ckpt", f"iter_{config.num_iters}.pt").exists()
    else:
        assert len(res) == 0

    # Check distributed eval gather on the eager smoke path.
    if run_eval:
        metrics = trainer.evaluator.eval_single(config.num_iters, n_iter=2)
        if trainer.distributed.is_main:
            assert "global IC" in metrics
            assert Path(config.root_dir, "eval", f"eval_{config.num_iters}.feather").exists()

    # Tear down this torchrun process group.
    torch.distributed.destroy_process_group()


def test_ddp_smoke_eager():
    """Run eager DDP smoke under torchrun."""
    # This test body is selected by the launch test below.
    if "RANK" not in os.environ:
        pytest.skip("torchrun-only smoke body")
    _run_ddp_smoke(use_cuda_graph=False, run_eval=True)


def test_ddp_smoke_cuda_graph():
    """Run CUDA graph DDP smoke under torchrun."""
    # This test body is selected by the launch test below.
    if "RANK" not in os.environ:
        pytest.skip("torchrun-only smoke body")
    _run_ddp_smoke(use_cuda_graph=True, run_eval=False)


def test_ddp_three_card_parity():
    """Run the three-GPU DDP parity body under torchrun."""
    # This test body is selected by the launch test below.
    if "RANK" not in os.environ:
        pytest.skip("torchrun-only parity body")
    _run_ddp_three_card_parity()


def test_ddp_checkpoint_full():
    """Run the uninterrupted checkpoint parity body under torchrun."""
    # This test body is selected by the launch test below.
    if "RANK" not in os.environ:
        pytest.skip("torchrun-only checkpoint body")
    _run_ddp_checkpoint_full()


def test_ddp_checkpoint_first_segment():
    """Run the first checkpoint parity segment under torchrun."""
    # This test body is selected by the launch test below.
    if "RANK" not in os.environ:
        pytest.skip("torchrun-only checkpoint body")
    _run_ddp_checkpoint_first_segment()


def test_ddp_checkpoint_resume():
    """Run the resumed checkpoint parity body under torchrun."""
    # This test body is selected by the launch test below.
    if "RANK" not in os.environ:
        pytest.skip("torchrun-only checkpoint body")
    _run_ddp_checkpoint_resume()


def test_launch_ddp_smoke_eager():
    """Launch the eager DDP smoke test."""
    # Require two visible CUDA devices for the launch test.
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    # Remove only this test fixture before exercising fresh initialization.
    shutil.rmtree(_make_config(False).root_dir, ignore_errors=True)

    # Run this file under torchrun with two ranks.
    subprocess.run([
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "--module",
        "pytest",
        "-q",
        f"{__file__}::test_ddp_smoke_eager",
    ], check=True)


def test_launch_ddp_smoke_cuda_graph():
    """Launch the CUDA graph DDP smoke test."""
    # Require two visible CUDA devices for the launch test.
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    # Remove only this test fixture before exercising fresh initialization.
    shutil.rmtree(_make_config(True).root_dir, ignore_errors=True)

    # Run this file under torchrun with two ranks.
    subprocess.run([
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "--module",
        "pytest",
        "-q",
        f"{__file__}::test_ddp_smoke_cuda_graph",
    ], check=True)


def test_launch_ddp_three_card_parity():
    """Compare three-GPU DDP training with one-GPU 3x-batch training."""
    # Require three visible CUDA devices for this parity check.
    if torch.cuda.device_count() < PARITY_WORLD_SIZE:
        pytest.skip("requires at least three CUDA devices")

    # Clear prior parity artifacts before producing fresh model states.
    shutil.rmtree(PARITY_SINGLE_ROOT, ignore_errors=True)
    shutil.rmtree(PARITY_DDP_ROOT, ignore_errors=True)

    # Run the single-process reference and three-rank DDP target.
    _run_single_parity_reference()
    subprocess.run([
        "torchrun",
        "--standalone",
        f"--nproc_per_node={PARITY_WORLD_SIZE}",
        "--module",
        "pytest",
        "-q",
        f"{__file__}::test_ddp_three_card_parity",
    ], check=True)

    # Assert the final trained parameters are exactly identical.
    single_path = PARITY_SINGLE_ROOT / PARITY_FINAL_STATE
    ddp_path = PARITY_DDP_ROOT / PARITY_FINAL_STATE
    _assert_state_file_equal(single_path, ddp_path)


def test_launch_ddp_checkpoint_resume_parity():
    """Compare uninterrupted DDP training with checkpointed DDP recovery."""
    # Require three visible CUDA devices for this checkpoint parity check.
    if torch.cuda.device_count() < PARITY_WORLD_SIZE:
        pytest.skip("requires at least three CUDA devices")

    # Clear prior checkpoint artifacts before producing fresh model states.
    shutil.rmtree(CHECKPOINT_FULL_ROOT, ignore_errors=True)
    shutil.rmtree(CHECKPOINT_SPLIT_ROOT, ignore_errors=True)

    # Run uninterrupted, first segment, and resumed trajectories.
    for test_name in [
        "test_ddp_checkpoint_full",
        "test_ddp_checkpoint_first_segment",
        "test_ddp_checkpoint_resume",
    ]:
        subprocess.run([
            "torchrun",
            "--standalone",
            f"--nproc_per_node={PARITY_WORLD_SIZE}",
            "--module",
            "pytest",
            "-q",
            f"{__file__}::{test_name}",
        ], check=True)

    # Assert resumed training lands on the same final parameters.
    full_checkpoint = CHECKPOINT_FULL_ROOT / "ckpt" / f"iter_{CHECKPOINT_RESUME_ITER}.pt"
    split_checkpoint = CHECKPOINT_SPLIT_ROOT / "ckpt" / f"iter_{CHECKPOINT_RESUME_ITER}.pt"
    checkpoint_rel_l2 = _checkpoint_model_relative_l2(full_checkpoint, split_checkpoint)
    assert checkpoint_rel_l2 <= PARITY_TOLERANCE, checkpoint_rel_l2

    # Assert resumed training lands close to the uninterrupted trajectory.
    full_path = CHECKPOINT_FULL_ROOT / PARITY_FINAL_STATE
    resumed_path = CHECKPOINT_SPLIT_ROOT / PARITY_FINAL_STATE
    rel_l2 = _relative_l2(full_path, resumed_path)
    assert rel_l2 <= CHECKPOINT_TOLERANCE, rel_l2
