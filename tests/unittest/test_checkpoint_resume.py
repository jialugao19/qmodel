import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from qmodel import Trainer
from qmodel.components.checkpoint import CheckpointSaver
from qmodel.config import DistributedConfig
from test_distributed_runtime import RuntimeConfig, RuntimeModel, RuntimeDataset


class LongDataset(RuntimeDataset):
    def __init__(self, group, dtype):
        # Allow the resumed DDP runtime to recapture after warmup.
        super().__init__(group, dtype)
        self.n_items = 256


class DropoutModel(RuntimeModel):
    def forward(self, data):
        # Exercise CUDA RNG as well as optimizer state.
        return self.net(nn.functional.dropout(data, p=0.2, training=True))


class InterruptedTrainer(Trainer):
    def _completed_step(self, curr_it, post):
        # Request a checkpoint after graph capture or eager warmup.
        if curr_it == 13 and self.distributed.is_main:
            self.signal = 1
        super()._completed_step(curr_it, post)
        if curr_it == 13:
            assert self.signal == 0
            os._exit(0)


def run_segment(root, mode, graph, workers, amp, ddp):
    # Use identical initialization and a fixed total budget in every process.
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    config = RuntimeConfig()
    config.distributed = DistributedConfig(enabled=ddp)
    config.batch_size = 6 if ddp else 8
    config.root_dir = str(root)
    config.tensorboard_dir = str(root / "events")
    config.num_iters = 30
    config.save_every = 30
    config.dataset_class = LongDataset
    config.train_use_cuda_graph = graph
    config.num_workers = workers
    config.model_class = DropoutModel
    config.use_amp = amp
    config.amp_dtype = torch.float16 if amp != "none" else torch.float32
    if mode == "resume":
        config.num_iters = 1
        config.learning_rate = 99
    trainer = (InterruptedTrainer if mode == "stop" else Trainer)(config)
    trainer.train()
    assert trainer.global_step == 30
    if ddp:
        torch.distributed.destroy_process_group()


def assert_equal(left, right, allow_roundoff):
    # Recursively compare complete persistent training states.
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-6 if allow_roundoff and left.is_floating_point() else 0, atol=1e-9 if allow_roundoff and left.is_floating_point() else 0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key], allow_roundoff)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second, allow_roundoff)
    else:
        assert left == right


@pytest.mark.parametrize("graph,workers,amp,ddp", [
    (False, 0, "none", False), (False, 2, "torch", False),
    (True, 2, "custom", False), (False, 0, "none", True),
    (True, 0, "custom", True),
])
def test_restart_parity(tmp_path, graph, workers, amp, ddp):
    # Launch genuinely separate processes for interruption and restart.
    if not torch.cuda.is_available() or (ddp and torch.cuda.device_count() < 3):
        pytest.skip("CUDA topology unavailable")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent) + os.pathsep + os.getcwd())
    for mode, name in (("full", "full"), ("stop", "split"), ("resume", "split")):
        script = (
            "from pathlib import Path; from test_checkpoint_resume import run_segment; "
            f"run_segment(Path({str(tmp_path / name)!r}), {mode!r}, {graph}, {workers}, {amp!r}, {ddp})"
        )
        command = [sys.executable, "-c", script]
        if ddp:
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=3", "--no-python"] + command
        subprocess.run(command, check=True, env=env, timeout=180)
    # Check every saved state, including rank-local RNG and sampling position.
    full = CheckpointSaver.latest(tmp_path / "full")[2]
    split = CheckpointSaver.latest(tmp_path / "split")[2]
    for key in ("global_step", "model", "optimizer", "scheduler", "scaler", "progress"):
        assert_equal(full[key], split[key], ddp and key == "optimizer")


def test_corrupt_checkpoint_is_not_skipped(tmp_path):
    # A corrupt committed file must not silently restart training.
    directory = tmp_path / "ckpt"
    directory.mkdir()
    (directory / "iter_99.pt").write_bytes(b"broken")
    with pytest.raises(Exception):
        CheckpointSaver.latest(tmp_path)


def test_sampler_consumed_position():
    # Prefetch must not affect the next consumed batch, including epoch rollover.
    from qmodel.data.sampler import CustomSampler
    sampler = CustomSampler(range(10), 42, True, True, 4, 0, 1, True)
    iterator = iter(sampler)
    batches = [[next(iterator) for _ in range(4)] for _ in range(7)]
    state = sampler.state_dict(3)
    restored = CustomSampler(range(10), 42, True, True, 4, 0, 1, True)
    restored.load_state_dict(state)
    resumed = iter(restored)
    assert [next(resumed) for _ in range(4)] == batches[3]


def test_atomic_failure_and_legacy(tmp_path):
    # A serialization failure leaves the previous committed file unchanged.
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    from qmodel.components.amp_scaler import MyScaler
    scaler = MyScaler("none", torch.float32)
    saver = CheckpointSaver(tmp_path, {"total": 10}, "cpu")
    path = Path(saver.save(model, optimizer, scheduler, scaler, 2))
    original = path.read_bytes()
    saver.config = lambda: None
    with pytest.raises(AttributeError):
        saver.save(model, optimizer, scheduler, scaler, 2)
    assert path.read_bytes() == original
    assert CheckpointSaver.latest(tmp_path)[0] == 2
    assert saver.load(path, model, optimizer, scheduler, scaler) == 2


class OverflowTrainer(Trainer):
    def _normal_step(self, data, target, other_meta, curr_it):
        # Force the first update to be skipped by Torch GradScaler.
        if curr_it == 0:
            data = data * float("inf")
        return super()._normal_step(data, target, other_meta, curr_it)


def test_amp_skip_keeps_signal(tmp_path):
    # Consume two batches but save only after the first real update.
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = RuntimeConfig()
    config.distributed = DistributedConfig(enabled=False)
    config.root_dir = str(tmp_path)
    config.tensorboard_dir = str(tmp_path / "events")
    config.num_iters = 2
    config.use_amp = "torch"
    config.amp_dtype = torch.float32
    trainer = OverflowTrainer(config)
    trainer.signal = 1
    trainer.train()
    state = CheckpointSaver.latest(tmp_path)[2]
    assert state["global_step"] == 1
    assert state["progress"]["ranks"][0]["sampler"]["consumed_batches"] == 2
    assert trainer.signal == 0
