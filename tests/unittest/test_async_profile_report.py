import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import matplotlib
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from qmodel.components.amp_scaler import MyScaler
from qmodel.components.distributed import DistributedContext
from qmodel.components.profiler import AsyncProfilingHelper, MyProfiler
from qmodel.config import ProfilerConfig
from qmodel.core.eval_trainer import AsyncTrainer

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib.pyplot as plt  # noqa: E402


PROFILE_STEP_T = 33
PROFILE_N_ITER = 50
PROFILE_BATCH_SIZE = 128
PROFILE_FEATURE_DIM = 256
PROFILE_HIDDEN_DIM = 512
PROFILE_WORLD_SIZE = 2
PROFILE_OUTPUT_DIR = Path("runs/async_profile_smoke")
PROFILE_CUDA_SPIN_CYCLES_PER_MS = 1_500_000

# Edit these values to stretch individual timeline stages in the smoke profile.
PROFILE_STAGE_SPIN_MS = {
    "h2d": 3.0,
    "model": 10.0,
    "stage_loss": 1.0,
    "d2h_loss": 5.0,
    "post_wait": 0.0,
    "post_step": 0.0,
}


@dataclass
class StreamMark:
    name: str
    stream_name: str
    event: torch.cuda.Event


@dataclass
class ResolvedMark:
    name: str
    lane: str
    timestamp_ms: float


@dataclass
class TimelineSpan:
    step: int
    part: str
    lane: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        """Return the span duration."""
        # Compute duration from normalized timestamps.
        return self.end_ms - self.start_ms


@dataclass
class P2PCommSummary:
    step: int
    count: int
    total_ms: float
    max_ms: float


class TinyRegressor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        """Build a small GPU-heavy regression model."""
        super().__init__()

        # Create enough work to make stream overlap visible.
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the regression network."""
        # Apply the model to one training batch.
        return self.net(data)


class ProfiledAsyncTrainer(AsyncTrainer):
    def _step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        """Run one train step inside a profiler step window."""
        # Add a CPU profiler range that owns DDP all-reduce launches for this step.
        with torch.profiler.record_function(f"train_step_{curr_it}"):
            return super()._step(data, target, other_meta, curr_it)


def _stage_name_from_start_marker(name: str) -> str | None:
    """Return the stage name represented by a start marker."""
    # Map marker prefixes to the report stage names.
    marker_prefixes = [
        ("h2d_start_", "h2d"),
        ("model_start_", "model"),
        ("stage_start_", "stage_loss"),
        ("d2h_start_", "d2h_loss"),
        ("post_wait_start_", "post_wait"),
        ("post_start_", "post_step"),
    ]

    # Find the first matching start marker prefix.
    for marker_prefix, stage_name in marker_prefixes:
        if name.startswith(marker_prefix):
            return stage_name
    return None


def _spin_cpu_ms(spin_ms: float):
    """Spin on CPU for the requested milliseconds."""
    # Keep the default zero-cost path cheap.
    if spin_ms <= 0.0:
        return

    # Busy-spin the post thread to make CPU stages visibly longer.
    end_time = time.perf_counter() + spin_ms / 1000.0
    while time.perf_counter() < end_time:
        pass


def _spin_cuda_ms(spin_ms: float):
    """Queue a CUDA spin kernel on the current stream."""
    # Keep the default zero-cost path cheap.
    if spin_ms <= 0.0:
        return

    # Convert user-facing milliseconds into CUDA spin cycles.
    spin_cycles = int(spin_ms * PROFILE_CUDA_SPIN_CYCLES_PER_MS)
    torch.cuda._sleep(spin_cycles)


class AsyncTimelineRecorder:
    def __init__(self):
        """Collect CPU and CUDA stream timestamps."""
        # Store timestamps behind a lock because trainer callbacks use threads.
        self._lock = Lock()
        self._stream_names: dict[int, str] = {}
        self._stream_marks: list[StreamMark] = []
        self._cpu_marks: list[ResolvedMark] = []

        # Create the origin after CUDA is initialized.
        self._cpu_origin = 0.0
        self._origin_event = torch.cuda.Event(enable_timing=True)

    def bind_stream(self, stream: torch.cuda.Stream, name: str):
        """Attach a stable lane name to a CUDA stream."""
        # Require every stream marker to use an explicit lane name.
        self._stream_names[id(stream)] = name

    def start(self):
        """Set the common CPU/GPU time origin."""
        # Finish previous CUDA work before defining the profile origin.
        torch.cuda.synchronize()

        # Record CPU and CUDA origins as close together as practical.
        self._cpu_origin = time.perf_counter()
        self._origin_event.record(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def mark(self, name: str, stream):
        """Record one CPU or CUDA stream timestamp."""
        # Resolve optional synthetic stage delay from the global knobs.
        stage_name = _stage_name_from_start_marker(name)
        spin_ms = PROFILE_STAGE_SPIN_MS[stage_name] if stage_name is not None else 0.0

        # Store CPU-only marks immediately.
        if stream is None:
            timestamp_ms = (time.perf_counter() - self._cpu_origin) * 1000.0
            with self._lock:
                self._cpu_marks.append(ResolvedMark(name, "cpu_post_thread", timestamp_ms))
            _spin_cpu_ms(spin_ms)
            return

        # Record CUDA marks on the requested stream.
        stream_name = self._stream_names[id(stream)]
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        _spin_cuda_ms(spin_ms)
        with self._lock:
            self._stream_marks.append(StreamMark(name, stream_name, event))

    def resolve(self) -> list[ResolvedMark]:
        """Convert recorded CUDA events into relative milliseconds."""
        # Complete all streams before reading event timestamps.
        torch.cuda.synchronize()

        # Snapshot marks before computing elapsed time.
        with self._lock:
            stream_marks = list(self._stream_marks)
            cpu_marks = list(self._cpu_marks)

        # Convert CUDA events onto the profile time axis.
        resolved = list(cpu_marks)
        for mark in stream_marks:
            timestamp_ms = self._origin_event.elapsed_time(mark.event)
            resolved.append(ResolvedMark(mark.name, mark.stream_name, timestamp_ms))
        return resolved


def _make_dataloader(n_batches: int, batch_size: int, feature_dim: int):
    """Create pinned CPU batches for async H2D transfer."""
    # Generate fixed random batches for the smoke profile.
    torch.manual_seed(42)
    batches = []
    for batch_idx in range(n_batches):
        data = torch.randn(batch_size, feature_dim).pin_memory()
        target = torch.randn(batch_size, 1).pin_memory()
        other_meta = torch.full((batch_size, 1), batch_idx)
        batches.append((data, target, other_meta))
    return batches


def _post_step(res: list, buffer: torch.Tensor, target, other_meta, curr_it: int):
    """Store one async post-step result."""
    # Keep a small CPU tensor so the callback has real work in the trace.
    res.append(buffer.detach().clone())


def _make_trainer(recorder: AsyncTimelineRecorder, profile_dir: Path) -> AsyncTrainer:
    """Build an instrumented DDP AsyncTrainer smoke workload."""
    # Initialize distributed placement for this torchrun rank.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    distributed = DistributedContext(True, rank, world_size, local_rank, device)

    # Build a DDP model and optimizer on the rank-local CUDA device.
    raw_model = TinyRegressor(PROFILE_FEATURE_DIM, PROFILE_HIDDEN_DIM).to(device)
    model = DistributedDataParallel(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    optimizer = torch.optim.SGD(raw_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step_idx: 1.0)

    # Build async trainer inputs.
    dataloader = _make_dataloader(PROFILE_N_ITER + 2, PROFILE_BATCH_SIZE, PROFILE_FEATURE_DIM)
    scaler = MyScaler("none", torch.float32)
    profiler = MyProfiler(True, ProfilerConfig(
        "train",
        profile_dir.as_posix(),
        wait=0,
        warmup=1,
        active=PROFILE_N_ITER + 4,
        repeat=1,
    ))
    trainer = ProfiledAsyncTrainer(
        model,
        dataloader,
        nn.MSELoss(),
        optimizer,
        scheduler,
        scaler,
        profiler,
        device,
        num_normal_steps=999999,
        distributed=distributed,
    )

    # Attach simple callbacks for the smoke profile.
    trainer._pre_step = lambda curr_it: curr_it  # type: ignore
    trainer._post_step = _post_step  # type: ignore

    # Bind stream names for the source-level async markers.
    recorder.bind_stream(trainer.stream1, "h2d_stream")
    recorder.bind_stream(trainer.model_stream, "model_stream")
    recorder.bind_stream(trainer.stream2, "d2h_stream")
    return trainer


def _marks_by_name(marks: list[ResolvedMark]) -> dict[str, ResolvedMark]:
    """Index resolved marks by marker name."""
    # Require every marker name to be unique in one run.
    return {mark.name: mark for mark in marks}


def _build_timeline_spans(marks: list[ResolvedMark], step_t: int) -> list[TimelineSpan]:
    """Build timeline spans for T-1, T, and T+1."""
    # Define the step window and the logical async parts.
    marks_by_name = _marks_by_name(marks)
    steps = [step_t - 1, step_t, step_t + 1]
    parts = [
        ("h2d", "h2d_start", "h2d_end"),
        ("model", "model_start", "model_end"),
        ("stage_loss", "stage_start", "stage_end"),
        ("d2h_loss", "d2h_start", "d2h_end"),
        ("post_wait", "post_wait_start", "post_wait_end"),
        ("post_step", "post_start", "post_end"),
    ]

    # Pair start and end markers into spans.
    spans: list[TimelineSpan] = []
    for step in steps:
        for part_name, start_prefix, end_prefix in parts:
            start = marks_by_name[f"{start_prefix}_{step}"]
            end = marks_by_name[f"{end_prefix}_{step}"]
            spans.append(
                TimelineSpan(
                    step=step,
                    part=part_name,
                    lane=start.lane,
                    start_ms=start.timestamp_ms,
                    end_ms=end.timestamp_ms,
                )
            )
    return spans


def _load_trace_events(trace_path: Path) -> list[dict]:
    """Load chrome trace events from profiler JSON."""
    # Read the trace emitted by torch.profiler.
    with trace_path.open("r") as trace_file:
        trace_data = json.load(trace_file)

    # Return the canonical trace event list.
    return trace_data["traceEvents"]


def _find_myprofiler_trace_path(profile_dir: Path) -> Path:
    """Return the trace JSON written by MyProfiler."""
    # MyProfiler writes one tensorboard chrome trace under profile_dir.
    trace_paths = sorted(profile_dir.glob("*.pt.trace.json"))
    assert len(trace_paths) == 1
    return trace_paths[0]


def _trace_event_summary(trace_path: Path, step_t: int) -> dict[str, int]:
    """Count source profiler ranges in the selected step window."""
    # Pick the source-level record_function names we expect in trace.
    events = _load_trace_events(trace_path)
    steps = [step_t - 1, step_t, step_t + 1]
    names = {
        f"_h2d_transfer_{step}"
        for step in steps
    } | {
        f"_post_step_{step}"
        for step in steps
    }

    # Count matching complete events.
    counts = {"total_trace_events": len(events)}
    counts.update({name: 0 for name in sorted(names)})
    for event in events:
        name = event.get("name")
        if event.get("ph") == "X" and name in counts:
            counts[name] += 1
    return counts


def _event_name(event: dict) -> str:
    """Return the trace event name as a string."""
    # Normalize missing names to an empty string.
    return str(event.get("name", ""))


def _event_category(event: dict) -> str:
    """Return the trace event category as a string."""
    # Normalize missing categories to an empty string.
    return str(event.get("cat", ""))


def _event_external_id(event: dict) -> int | None:
    """Return the profiler external id for one trace event."""
    # Read the profiler id stored in trace args.
    args = event.get("args", {})
    external_id = args.get("External id")
    if external_id is None:
        return None
    return int(external_id)


def _event_start_us(event: dict) -> float:
    """Return a trace event start timestamp in microseconds."""
    # Chrome trace timestamps are already stored in microseconds.
    return float(event["ts"])


def _event_end_us(event: dict) -> float:
    """Return a trace event end timestamp in microseconds."""
    # Add complete-event duration to the start timestamp.
    return _event_start_us(event) + float(event.get("dur", 0.0))


def _contains_time_us(event: dict, timestamp_us: float) -> bool:
    """Return whether an event contains a timestamp."""
    # Check the timestamp against the complete-event interval.
    return _event_start_us(event) <= timestamp_us <= _event_end_us(event)


def _is_cpu_train_step_event(event: dict, step: int) -> bool:
    """Return whether this event is the CPU train-step window."""
    # Match the explicit step range added by ProfiledAsyncTrainer.
    return (
        event.get("ph") == "X"
        and _event_category(event) == "user_annotation"
        and _event_name(event) == f"train_step_{step}"
    )


def _is_cpu_nccl_all_reduce_event(event: dict) -> bool:
    """Return whether this event is a CPU NCCL all-reduce annotation."""
    # CPU NCCL annotations carry the external id used by the GPU event.
    return (
        event.get("ph") == "X"
        and _event_category(event) == "user_annotation"
        and _event_name(event) == "nccl:all_reduce"
    )


def _is_gpu_nccl_all_reduce_event(event: dict) -> bool:
    """Return whether this event is a GPU NCCL all-reduce annotation."""
    # GPU NCCL annotation duration is the P2P communication time we report.
    return (
        event.get("ph") == "X"
        and _event_category(event) == "gpu_user_annotation"
        and _event_name(event) == "nccl:all_reduce"
    )


def _build_p2p_comm_summaries(trace_path: Path, step_t: int) -> list[P2PCommSummary]:
    """Build per-step GPU NCCL all-reduce timing summaries."""
    # Load the chrome trace and select the same step window as the timeline.
    events = _load_trace_events(trace_path)
    steps = [step_t - 1, step_t, step_t + 1]
    step_events = {
        step: next(event for event in events if _is_cpu_train_step_event(event, step))
        for step in steps
    }

    # Map CPU NCCL launch annotations to their owning train step.
    external_ids_by_step = {step: set() for step in steps}
    for event in events:
        if not _is_cpu_nccl_all_reduce_event(event):
            continue

        # Assign the CPU launch to the train-step range that contains it.
        start_us = _event_start_us(event)
        for step, step_event in step_events.items():
            if _contains_time_us(step_event, start_us):
                external_id = _event_external_id(event)
                assert external_id is not None
                external_ids_by_step[step].add(external_id)

    # Sum matching GPU NCCL annotation durations.
    summaries = []
    gpu_events = [event for event in events if _is_gpu_nccl_all_reduce_event(event)]
    for step in steps:
        durations_ms = [
            float(event.get("dur", 0.0)) / 1000.0
            for event in gpu_events
            if _event_external_id(event) in external_ids_by_step[step]
        ]
        summaries.append(
            P2PCommSummary(
                step=step,
                count=len(durations_ms),
                total_ms=sum(durations_ms),
                max_ms=max(durations_ms) if durations_ms else 0.0,
            )
        )
    return summaries


def _overlap_ms(left: TimelineSpan, right: TimelineSpan) -> float:
    """Measure overlap between two spans."""
    # Intersect the two time intervals.
    start_ms = max(left.start_ms, right.start_ms)
    end_ms = min(left.end_ms, right.end_ms)
    return max(0.0, end_ms - start_ms)


def _build_overlap_rows(spans: list[TimelineSpan], step_t: int) -> list[tuple[str, float]]:
    """Summarize expected async overlap relationships."""
    # Index spans by step and part.
    span_map = {(span.step, span.part): span for span in spans}

    # Compare adjacent pipeline stages around T.
    rows = []
    for step in [step_t - 1, step_t]:
        rows.append((
            f"d2h_loss_{step} vs h2d_{step + 1}",
            _overlap_ms(span_map[(step, "d2h_loss")], span_map[(step + 1, "h2d")]),
        ))
        rows.append((
            f"post_step_{step} vs model_{step + 1}",
            _overlap_ms(span_map[(step, "post_step")], span_map[(step + 1, "model")]),
        ))
    return rows


def _plot_timeline(
    spans: list[TimelineSpan],
    p2p_comm_summaries: list[P2PCommSummary],
    output_path: Path,
    step_t: int,
):
    """Plot async timeline spans as horizontal lines."""
    # Sort rows by step and pipeline order.
    order = {
        "h2d": 0,
        "model": 1,
        "stage_loss": 2,
        "d2h_loss": 3,
        "post_wait": 4,
        "post_step": 5,
    }
    sorted_spans = sorted(spans, key=lambda span: (span.step, order[span.part]))

    # Draw one horizontal interval per async part.
    fig_height = max(4.0, 0.38 * len(sorted_spans))
    fig, (ax_timeline, ax_p2p) = plt.subplots(
        nrows=2,
        figsize=(13, fig_height + 2.6),
        gridspec_kw={"height_ratios": [fig_height, 2.2]},
    )
    colors = {
        "h2d_stream": "#1f77b4",
        "model_stream": "#2ca02c",
        "d2h_stream": "#d62728",
        "cpu_post_thread": "#9467bd",
    }
    labels = []
    for row_idx, span in enumerate(sorted_spans):
        color = colors[span.lane]
        ax_timeline.hlines(row_idx, span.start_ms, span.end_ms, color=color, linewidth=5)
        ax_timeline.scatter([span.start_ms, span.end_ms], [row_idx, row_idx], color=color, s=12)
        labels.append(f"step {span.step} {span.part} ({span.lane})")

    # Format the timeline for quick visual inspection.
    min_ms = min(span.start_ms for span in sorted_spans)
    max_ms = max(span.end_ms for span in sorted_spans)
    pad_ms = max(0.5, (max_ms - min_ms) * 0.05)
    ax_timeline.set_xlim(min_ms - pad_ms, max_ms + pad_ms)
    ax_timeline.set_yticks(range(len(labels)))
    ax_timeline.set_yticklabels(labels)
    ax_timeline.set_xlabel("milliseconds from profile origin")
    ax_timeline.set_title(f"AsyncTrainer DDP timeline around step T={step_t}")
    ax_timeline.grid(axis="x", alpha=0.25)

    # Plot GPU-side NCCL all-reduce time by train step.
    p2p_steps = [summary.step for summary in p2p_comm_summaries]
    ax_p2p.bar(
        p2p_steps,
        [summary.total_ms for summary in p2p_comm_summaries],
        color="#17becf",
        label="gpu nccl all_reduce",
    )
    ax_p2p.set_xticks(p2p_steps)
    ax_p2p.set_xlabel("step")
    ax_p2p.set_ylabel("duration_ms")
    ax_p2p.set_title("P2P communication time from GPU NCCL all-reduce events")
    ax_p2p.grid(axis="y", alpha=0.25)
    ax_p2p.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(
    report_path: Path,
    trace_path: Path,
    plot_path: Path,
    spans: list[TimelineSpan],
    p2p_comm_summaries: list[P2PCommSummary],
    trace_counts: dict[str, int],
    overlap_rows: list[tuple[str, float]],
):
    """Write a compact Markdown profiling report."""
    # Build span table rows.
    lines = [
        "# AsyncTrainer profiling smoke report",
        "",
        f"- trace: `{trace_path}`",
        f"- plot: `{plot_path}`",
        "",
        "## Synthetic spin 设置",
        "",
        f"- cuda_spin_cycles_per_ms: `{PROFILE_CUDA_SPIN_CYCLES_PER_MS}`",
        "",
        "| stage | spin_ms |",
        "| --- | ---: |",
    ]
    for stage_name, spin_ms in PROFILE_STAGE_SPIN_MS.items():
        lines.append(f"| {stage_name} | {spin_ms:.3f} |")

    lines.extend([
        "",
        "## 时间线区间",
        "",
        "| step | part | lane | start_ms | end_ms | duration_ms |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for span in sorted(spans, key=lambda item: (item.step, item.start_ms)):
        lines.append(
            f"| {span.step} | {span.part} | {span.lane} | "
            f"{span.start_ms:.3f} | {span.end_ms:.3f} | {span.duration_ms:.3f} |"
        )

    # Add GPU-side DDP communication timing.
    lines.extend([
        "",
        "## P2P communication 时间",
        "",
        "这里统计 GPU-side `nccl:all_reduce` event duration; CPU `c10d::allreduce_` 只代表 launch 开销。",
        "",
        "| step | nccl_all_reduce_count | p2p_comm_ms | max_single_all_reduce_ms |",
        "| --- | ---: | ---: | ---: |",
    ])
    for summary in p2p_comm_summaries:
        lines.append(
            f"| {summary.step} | {summary.count} | "
            f"{summary.total_ms:.3f} | {summary.max_ms:.3f} |"
        )

    # Add overlap and trace summaries.
    lines.extend([
        "",
        "## Overlap 检查",
        "",
        "| pair | overlap_ms |",
        "| --- | ---: |",
    ])
    for pair_name, overlap in overlap_rows:
        lines.append(f"| {pair_name} | {overlap:.3f} |")

    lines.extend([
        "",
        "## Trace marker 计数",
        "",
        "这些计数来自 chrome trace JSON; stream 时间线以 CUDA event marker 为准。",
        "",
        "| marker | count |",
        "| --- | ---: |",
    ])
    for marker_name, count in trace_counts.items():
        lines.append(f"| {marker_name} | {count} |")

    # Persist the report for inspection.
    report_path.write_text("\n".join(lines) + "\n")


def test_async_trainer_trace_report():
    """Launch DDP async profiling and generate rank-local reports."""
    # Launch the DDP profiling body under torchrun from the parent process.
    if "RANK" not in os.environ:
        if torch.cuda.device_count() < PROFILE_WORLD_SIZE:
            pytest.skip("requires at least two CUDA devices")
        subprocess.run([
            "torchrun",
            "--standalone",
            f"--nproc_per_node={PROFILE_WORLD_SIZE}",
            "--module",
            "pytest",
            "-q",
            "-s",
            f"{__file__}::test_async_trainer_trace_report",
        ], check=True)
        return

    # Prepare output paths under ignored runs/.
    rank = int(os.environ["RANK"])
    PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile_dir = PROFILE_OUTPUT_DIR / f"rank_{rank}_myprofiler"
    plot_path = PROFILE_OUTPUT_DIR / f"async_timeline_step_{PROFILE_STEP_T}_rank{rank}.png"
    report_path = PROFILE_OUTPUT_DIR / f"async_timeline_step_{PROFILE_STEP_T}_rank{rank}.md"
    shutil.rmtree(profile_dir, ignore_errors=True)
    for output_path in [plot_path, report_path]:
        output_path.unlink(missing_ok=True)

    # Build the instrumented DDP async trainer.
    recorder = AsyncTimelineRecorder()
    trainer = _make_trainer(recorder, profile_dir)

    # Run MyProfiler through the trainer's normal profiler lifecycle.
    AsyncProfilingHelper.enable(recorder.mark)
    try:
        recorder.start()
        trainer.run(n_iter=PROFILE_N_ITER, use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        AsyncProfilingHelper.disable()

    # Resolve the chrome trace emitted by MyProfiler.
    trace_path = _find_myprofiler_trace_path(profile_dir)

    # Analyze markers and trace into local artifacts.
    marks = recorder.resolve()
    spans = _build_timeline_spans(marks, PROFILE_STEP_T)
    p2p_comm_summaries = _build_p2p_comm_summaries(trace_path, PROFILE_STEP_T)
    trace_counts = _trace_event_summary(trace_path, PROFILE_STEP_T)
    overlap_rows = _build_overlap_rows(spans, PROFILE_STEP_T)
    _plot_timeline(spans, p2p_comm_summaries, plot_path, PROFILE_STEP_T)
    _write_report(
        report_path,
        trace_path,
        plot_path,
        spans,
        p2p_comm_summaries,
        trace_counts,
        overlap_rows,
    )

    # Assert artifacts and expected trace markers exist.
    assert trace_path.exists()
    assert plot_path.exists()
    assert report_path.exists()
    assert len(spans) == 18
    assert len(p2p_comm_summaries) == 3
    assert all(summary.count > 0 for summary in p2p_comm_summaries)
    assert all(summary.total_ms > 0.0 for summary in p2p_comm_summaries)
    assert all(span.duration_ms >= 0.0 for span in spans)
    assert trace_counts["total_trace_events"] > 0
    for summary in p2p_comm_summaries:
        print(
            f"rank {rank} step {summary.step}: "
            f"p2p_comm={summary.total_ms:.3f} ms, "
            f"all_reduce_count={summary.count}, max={summary.max_ms:.3f} ms"
        )
    print(f"rank {rank} saved async png to: {plot_path}")

    # Release the DDP process group after all rank-local artifacts are written.
    dist.destroy_process_group()


if __name__ == "__main__":
    test_async_trainer_trace_report()
