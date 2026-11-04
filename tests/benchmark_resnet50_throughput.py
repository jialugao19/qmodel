import logging
import os
import statistics
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore", message="Profiler won't be using warmup.*")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides.*")
warnings.filterwarnings(
    "ignore",
    message=(
        "Dynamo does not know how to trace the builtin "
        "`torch._C._distributed_c10d.PyCapsule._broadcast_coalesced.*"
    ),
)

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet50

from qmodel.components.amp_scaler import MyScaler
from qmodel.components.distributed import DistributedContext
from qmodel.components.profiler import MyProfiler
from qmodel.config import DistributedConfig, ProfilerConfig, QConfig
from qmodel.core.eval_trainer import AsyncInference, AsyncTrainer


SEED = 20260613
IMAGE_SIZE = 224
NUM_CLASSES = 1000
PROFILE_ROOT = "runs/resnet50_throughput_profile"

Batch = tuple[torch.Tensor, torch.Tensor, None]


@dataclass
class BenchSetting:
    name: str
    profile_steps: int
    throughput_start_step: int
    measure_steps: int
    pool_batches: int  # batches pinned in cpu for synthetic data
    repeat_runs: int
    train_batch_size: int
    infer_batch_size: int
    num_normal_steps: int
    learning_rate: float
    compile_model: bool
    profile_enabled: bool


SETTINGS = [
    BenchSetting("normal_path", 40, 50, 100, 64, 3, 16, 64, 100000, 1e-4, False, True),
    BenchSetting("torch_compile", 40, 50, 100, 64, 3, 16, 64, 100000, 1e-4, True, False),
    BenchSetting("cuda_graph", 40, 50, 100, 64, 3, 16, 64, 20, 1e-4, False, True),
]


class FixedScheduler:
    def step(self) -> None:
        """Keep learning rate fixed."""
        # qmodel expects a scheduler object with step().
        return None


def get_distributed() -> DistributedContext:
    """Create qmodel's distributed context."""
    # Let qmodel initialize either single-GPU mode or torchrun DDP mode.
    config = QConfig()
    config.device = torch.device("cuda:0")
    config.train_use_cuda_graph = any(
        setting.num_normal_steps + 1 < setting.throughput_start_step + setting.measure_steps
        for setting in SETTINGS
    )
    config.distributed = DistributedConfig(enabled=("RANK" in os.environ))
    distributed = DistributedContext.from_config(config)
    torch.cuda.set_device(distributed.device)
    return distributed


def make_model(setting: BenchSetting, distributed: DistributedContext) -> tuple[nn.Module, torch.cuda.Stream | None]:
    """Create ResNet50 on this rank."""
    # Build the same random model on every rank.
    torch.manual_seed(SEED)
    model = resnet50(weights=None, num_classes=NUM_CLASSES).to(distributed.device)

    # Keep single-process runs on the raw module.
    if not distributed.enabled:
        if setting.compile_model:
            model = torch.compile(model)
        return model, None

    # Build graph-captured DDP on the stream later used by qmodel capture.
    use_cuda_graph = setting.num_normal_steps + 1 < setting.throughput_start_step + setting.measure_steps
    if use_cuda_graph:
        stream = torch.cuda.Stream(device=distributed.device)
        stream.wait_stream(torch.cuda.current_stream(distributed.device))
        with torch.cuda.stream(stream):
            model = DistributedDataParallel(
                model, device_ids=[distributed.local_rank], output_device=distributed.local_rank, static_graph=True)
        torch.cuda.current_stream(distributed.device).wait_stream(stream)
        return model, stream

    # Build normal DDP on the current stream.
    model = DistributedDataParallel(model, device_ids=[distributed.local_rank], output_device=distributed.local_rank)
    if setting.compile_model:
        model = torch.compile(model)
    return model, None


def make_batches(setting: BenchSetting, batch_size: int, seed: int) -> list[Batch]:
    """Create fixed pinned random batches."""
    # Generate deterministic rank-local inputs and regression targets.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    data = torch.randn(
        setting.pool_batches, batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32, generator=generator)
    target = torch.randn(setting.pool_batches, batch_size, NUM_CLASSES, dtype=torch.float32, generator=generator)

    # Pin batches for qmodel's async host-to-device path.
    return [
        (data[batch_idx].pin_memory(), target[batch_idx].pin_memory(), None)
        for batch_idx in range(setting.pool_batches)
    ]


def repeat_batches(batch_pool: list[Batch], steps: int) -> Iterator[Batch]:
    """Yield batches for a fixed number of steps."""
    # Cycle through the small deterministic pool.
    for step_idx in range(steps):
        yield batch_pool[step_idx % len(batch_pool)]


def make_profiler(
    setting: BenchSetting,
    name: str,
    distributed: DistributedContext,
    enabled: bool,
) -> MyProfiler:
    """Create MyProfiler for the first repeat."""
    # Disable profiler for settings where profiler changes execution behavior.
    enabled = enabled and setting.profile_enabled

    # Save each rank's trace in its own directory.
    profile_dir = f"{PROFILE_ROOT}/{setting.name}/{name}/rank_{distributed.rank}"
    if enabled and os.path.isdir(profile_dir):
        # Remove stale profiler traces before writing the new one.
        for file_name in os.listdir(profile_dir):
            if file_name == "trace.pt" or file_name.endswith((".pt.trace.json", ".pt.trace.json.gz")):
                os.remove(os.path.join(profile_dir, file_name))

    # Build qmodel's profiler configuration.
    config = ProfilerConfig("benchmark", profile_dir, wait=0, warmup=0, active=setting.profile_steps + 1, repeat=1)
    return MyProfiler(enabled, config)


def empty_post_step(res: list, buffer: torch.Tensor, target, other_meta, curr_it: int) -> None:
    """Do no post-step work."""
    # Keep qmodel callback overhead out of the benchmark.
    return None


def timed_pre_step(
    setting_idx: int, total_settings: int, setting: BenchSetting, mode: str,
    repeat_idx: int, distributed: DistributedContext, start_time: list[float], total_steps: int,
):
    """Create the delayed timer callback."""
    # Keep the timing callback close to the benchmark constants.
    def pre_step(curr_it: int) -> None:
        """Start timing when the benchmark reaches the configured step."""
        # Print rank-zero progress every 100 steps.
        if distributed.is_main and curr_it % 100 == 0:
            now = time.time()
            timestamp = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
            print(
                f"[{timestamp}] setting {setting_idx + 1}/{total_settings}: {setting.name}, "
                f"{mode} repeat {repeat_idx + 1}/{setting.repeat_runs}, step {curr_it:03d}/{total_steps:03d}"
            )

        # Synchronize ranks before starting throughput timing.
        if curr_it == setting.throughput_start_step:
            distributed.barrier()
            torch.cuda.synchronize(distributed.device)
            start_time[0] = time.perf_counter()

    return pre_step


def finish_throughput(
    setting: BenchSetting,
    distributed: DistributedContext,
    start_time: list[float],
    local_batch_size: int,
) -> float:
    """Compute global images per second."""
    # Use the slowest rank as the DDP elapsed time.
    torch.cuda.synchronize(distributed.device)
    distributed.barrier()
    elapsed = time.perf_counter() - start_time[0]
    elapsed_values = distributed.gather_object(elapsed)
    if distributed.is_main:
        elapsed = max(elapsed_values)

    # Convert local per-rank work to global throughput.
    measured_steps = setting.measure_steps
    return measured_steps * local_batch_size * distributed.world_size / elapsed


def run_train(
    setting_idx: int, total_settings: int, setting: BenchSetting, distributed: DistributedContext,
) -> float:
    """Measure ResNet50 training throughput."""
    # Create rank-local data, model, optimizer, and fixed scheduler.
    batch_pool = make_batches(setting, setting.train_batch_size, SEED + distributed.rank)
    model, graph_stream = make_model(setting, distributed)
    optimizer = torch.optim.SGD(model.parameters(), lr=setting.learning_rate, momentum=0.9)
    scheduler = FixedScheduler()
    total_steps = setting.throughput_start_step + setting.measure_steps

    # Run repeated measurements and profile only the first repeat.
    throughputs = []
    for repeat_idx in range(setting.repeat_runs):
        if distributed.is_main:
            print(f"\nStart setting {setting_idx + 1}/{total_settings}: {setting.name}, train repeat {repeat_idx + 1}")

        start_time = [0.0]
        runner = AsyncTrainer(
            model, repeat_batches(batch_pool, total_steps), nn.MSELoss(), optimizer, scheduler,
            MyScaler("none", torch.float32), make_profiler(setting, "train", distributed, repeat_idx == 0),
            distributed.device, setting.num_normal_steps, distributed=distributed,
        )
        if graph_stream is not None:
            runner.model_stream = graph_stream
        runner._pre_step = timed_pre_step(
            setting_idx, total_settings, setting, "train", repeat_idx, distributed, start_time, total_steps)
        runner._post_step = empty_post_step

        runner.run(n_iter=total_steps, use_tqdm=False)

        throughputs.append(finish_throughput(setting, distributed, start_time, setting.train_batch_size))
    return statistics.median(throughputs)


def run_inference(
    setting_idx: int, total_settings: int, setting: BenchSetting, distributed: DistributedContext,
) -> float:
    """Measure ResNet50 inference throughput."""
    # Create rank-local data and an inference model.
    batch_pool = make_batches(setting, setting.infer_batch_size, SEED + 1000 + distributed.rank)
    model, graph_stream = make_model(setting, distributed)
    total_steps = setting.throughput_start_step + setting.measure_steps

    # Run repeated measurements and profile only the first repeat.
    throughputs = []
    for repeat_idx in range(setting.repeat_runs):
        if distributed.is_main:
            print(
                f"\nStart setting {setting_idx + 1}/{total_settings}: "
                f"{setting.name}, inference repeat {repeat_idx + 1}"
            )

        start_time = [0.0]
        runner = AsyncInference(
            model, repeat_batches(batch_pool, total_steps), MyScaler("none", torch.float32),
            make_profiler(setting, "inference", distributed, repeat_idx == 0),
            distributed.device, setting.num_normal_steps, distributed=distributed,
        )
        if graph_stream is not None:
            runner.model_stream = graph_stream
        runner._pre_step = timed_pre_step(
            setting_idx, total_settings, setting, "inference", repeat_idx, distributed, start_time, total_steps)
        runner._post_step = empty_post_step
        runner.run(n_iter=total_steps, use_tqdm=False)
        throughputs.append(finish_throughput(setting, distributed, start_time, setting.infer_batch_size))
    return statistics.median(throughputs)


def print_result(distributed: DistributedContext, rows: list[tuple[BenchSetting, float, float]]) -> None:
    """Print a readable benchmark summary."""
    # Print the run configuration first.
    print("ResNet50 throughput benchmark")
    print(f"world size       : {distributed.world_size}")
    print("")

    # Print the throughput table.
    print(
        f"{'setting':<12} {'normal_steps':>12} {'profile':>8} {'start':>7} {'steps':>7} "
        f"{'compile':>7} {'pool':>6} {'train bs':>8} {'infer bs':>8} {'train img/s':>12} {'infer img/s':>12}"
    )
    print(
        f"{'-' * 12} {'-' * 12:>12} {'-' * 8:>8} {'-' * 7:>7} {'-' * 7:>7} "
        f"{'-' * 7:>7} {'-' * 6:>6} {'-' * 8:>8} {'-' * 8:>8} {'-' * 12:>12} {'-' * 12:>12}"
    )
    for setting, train_throughput, infer_throughput in rows:
        # Show disabled profiler settings explicitly.
        profile_text = str(setting.profile_steps) if setting.profile_enabled else "off"
        print(
            f"{setting.name:<12} {setting.num_normal_steps:>12} {profile_text:>8} "
            f"{setting.throughput_start_step:>7} {setting.measure_steps:>7} "
            f"{str(setting.compile_model):>7} "
            f"{setting.pool_batches:>6} "
            f"{setting.train_batch_size * distributed.world_size:>8} "
            f"{setting.infer_batch_size * distributed.world_size:>8} "
            f"{train_throughput:>12.2f} {infer_throughput:>12.2f}"
        )
    print("")
    print("batch sizes are global; divide by world size for local per-rank batch size.")


def main() -> None:
    """Run the ResNet50 benchmark."""
    # Initialize qmodel distributed runtime and CUDA settings.
    distributed = get_distributed()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run train and inference measurements for each setting.
    rows = []
    for setting_idx, setting in enumerate(SETTINGS):
        train_throughput = run_train(setting_idx, len(SETTINGS), setting, distributed)
        infer_throughput = run_inference(setting_idx, len(SETTINGS), setting, distributed)
        rows.append((setting, train_throughput, infer_throughput))

    # Print one readable summary from rank zero.
    if distributed.is_main:
        print_result(distributed, rows)

    # Close DDP state created by qmodel.
    distributed.barrier()
    if distributed.enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
