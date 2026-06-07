import torch
import time
import os
from contextlib import contextmanager
from typing import Callable, TypeAlias

from qmodel.config import ProfilerConfig
from qmodel.logger import logger

Stream: TypeAlias = torch.cuda.Stream | torch._C._CudaStreamBase
AsyncProfilingMarker: TypeAlias = Callable[[str, Stream | None], None]

class AsyncProfilingHelper:
    enabled: bool = False
    marker: AsyncProfilingMarker | None = None

    @classmethod
    def enable(cls, marker: AsyncProfilingMarker):
        """Enable async profiling regions globally."""
        # Store the global marker and enable region recording.
        cls.marker = marker
        cls.enabled = True

    @classmethod
    def disable(cls):
        """Disable async profiling regions globally."""
        # Clear global profiling state after a diagnostic run.
        cls.enabled = False
        cls.marker = None

    @contextmanager
    def region(self, region_name: str, curr_it: int, stream: Stream | None):
        """Return a context manager for one async profiling region."""
        # Yield directly when async profiling is disabled.
        if not AsyncProfilingHelper.enabled:
            yield
            return

        # Emit start/end markers around the wrapped region.
        marker = AsyncProfilingHelper.marker
        assert marker is not None
        marker(f"{region_name}_start_{curr_it}", stream)
        try:
            yield
        finally:
            marker(f"{region_name}_end_{curr_it}", stream)


class MyProfiler:
    profiler: None | torch.profiler.profiler.profile
    finished: bool

    def __init__(self, real_run: bool, config: ProfilerConfig):
        profile_dir = config.profile_dir
        wait        = config.wait
        warmup      = config.warmup
        active      = config.active
        repeat      = config.repeat

        self.profiler = None
        self.finished = False  # prevent mutliple runs of the same profiler
        if not real_run:
            return

        os.makedirs(config.profile_dir, exist_ok=True)
        schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )
        tracer = torch.profiler.tensorboard_trace_handler(profile_dir)

        self.profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=tracer,
            record_shapes=True, profile_memory=True, with_stack=True,
            with_flops=True, with_modules=True
        )

    def start(self, sleep_sec=2):
        if self.profiler is not None and not self.finished:
            self.profiler.start()
            # sleep seconds to ensure correct running when no wait/warmup
            time.sleep(sleep_sec)
            self.profiler.step()

    def step(self):
        if self.profiler is not None and not self.finished:
            self.profiler.step()

    def stop(self):
        if self.profiler is not None and not self.finished:
            self.profiler.stop()
            self.finished = True
            logger.info("Profiler stopped and data saved.")
            # flops
