import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import matplotlib
import torch
import torch.nn as nn

from qmodel.components.amp_scaler import MyScaler
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


def _make_trainer(recorder: AsyncTimelineRecorder) -> AsyncTrainer:
    """Build an instrumented AsyncTrainer smoke workload."""
    # Build model and optimizer on CUDA.
    device = torch.device("cuda")
    model = TinyRegressor(PROFILE_FEATURE_DIM, PROFILE_HIDDEN_DIM).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step_idx: 1.0)

    # Build async trainer inputs.
    dataloader = _make_dataloader(PROFILE_N_ITER + 2, PROFILE_BATCH_SIZE, PROFILE_FEATURE_DIM)
    scaler = MyScaler("none", torch.float32)
    profiler = MyProfiler(False, ProfilerConfig("none", ""))
    trainer = AsyncTrainer(
        model,
        dataloader,
        nn.MSELoss(),
        optimizer,
        scheduler,
        scaler,
        profiler,
        device,
        num_normal_steps=999999,
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


def _plot_timeline(spans: list[TimelineSpan], output_path: Path, step_t: int):
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
    fig, ax = plt.subplots(figsize=(13, fig_height))
    colors = {
        "h2d_stream": "#1f77b4",
        "model_stream": "#2ca02c",
        "d2h_stream": "#d62728",
        "cpu_post_thread": "#9467bd",
    }
    labels = []
    for row_idx, span in enumerate(sorted_spans):
        color = colors[span.lane]
        ax.hlines(row_idx, span.start_ms, span.end_ms, color=color, linewidth=5)
        ax.scatter([span.start_ms, span.end_ms], [row_idx, row_idx], color=color, s=12)
        labels.append(f"step {span.step} {span.part} ({span.lane})")

    # Format the timeline for quick visual inspection.
    min_ms = min(span.start_ms for span in sorted_spans)
    max_ms = max(span.end_ms for span in sorted_spans)
    pad_ms = max(0.5, (max_ms - min_ms) * 0.05)
    ax.set_xlim(min_ms - pad_ms, max_ms + pad_ms)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("milliseconds from profile origin")
    ax.set_title(f"AsyncTrainer timeline around step T={step_t}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(
    report_path: Path,
    trace_path: Path,
    plot_path: Path,
    spans: list[TimelineSpan],
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
    """Generate trace, timeline plot, and report for async training."""
    # Prepare output paths under ignored runs/.
    PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = PROFILE_OUTPUT_DIR / "async_training_trace.json"
    plot_path = PROFILE_OUTPUT_DIR / f"async_timeline_step_{PROFILE_STEP_T}.png"
    report_path = PROFILE_OUTPUT_DIR / f"async_timeline_step_{PROFILE_STEP_T}.md"
    for output_path in [trace_path, plot_path, report_path]:
        output_path.unlink(missing_ok=True)

    # Build the instrumented async trainer.
    recorder = AsyncTimelineRecorder()
    trainer = _make_trainer(recorder)

    # Run profiler and export a chrome trace.
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    AsyncProfilingHelper.enable(recorder.mark)
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
        ) as profiler:
            recorder.start()
            trainer.run(n_iter=PROFILE_N_ITER, use_tqdm=False)
            torch.cuda.synchronize()
    finally:
        AsyncProfilingHelper.disable()
    profiler.export_chrome_trace(str(trace_path))

    # Analyze markers and trace into local artifacts.
    marks = recorder.resolve()
    spans = _build_timeline_spans(marks, PROFILE_STEP_T)
    trace_counts = _trace_event_summary(trace_path, PROFILE_STEP_T)
    overlap_rows = _build_overlap_rows(spans, PROFILE_STEP_T)
    _plot_timeline(spans, plot_path, PROFILE_STEP_T)
    _write_report(report_path, trace_path, plot_path, spans, trace_counts, overlap_rows)

    # Assert artifacts and expected trace markers exist.
    assert trace_path.exists()
    assert plot_path.exists()
    assert report_path.exists()
    assert len(spans) == 18
    assert all(span.duration_ms >= 0.0 for span in spans)
    assert trace_counts["total_trace_events"] > 0
    print(f"saved async png to: {plot_path}")


if __name__ == "__main__":
    test_async_trainer_trace_report()
