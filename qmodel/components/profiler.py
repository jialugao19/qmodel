import torch
import time
import os

from qmodel.config import ProfilerConfig
from qmodel.logger import logger

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
            
