import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Protocol, Iterable, TypeAlias
import time
from tqdm import tqdm

from qmodel.components.amp_scaler import MyScaler
from qmodel.components.profiler import AsyncProfilingHelper, MyProfiler
from qmodel.components.lr_scheduler import Scheduler

# alias torch cuda event and _cudaeventbase by the same one
Event: TypeAlias = torch.cuda.Event | torch._C._CudaEventBase
Stream: TypeAlias = torch.cuda.Stream | torch._C._CudaStreamBase


class _AsyncBase:
    device: torch.device
    move_target: bool
    model: nn.Module
    dataloader: Iterable | torch.utils.data.DataLoader
    profiler: MyProfiler
    amp_scaler: MyScaler
    async_profiling: AsyncProfilingHelper
    valid: bool

    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, move_target: bool,
        num_normal_steps: int
    ):
        self.device = torch.device(device)
        self.move_target = move_target

        self.model = model
        self.dataloader = dataloader
        self.it = iter(dataloader)
        self.amp_scaler = amp_scaler

        self.ex = ThreadPoolExecutor(max_workers=1)
        self.ex2 = ThreadPoolExecutor(max_workers=1)
        self.stream1: Stream = torch.cuda.Stream(device=self.device)   # host->device
        self.stream2: Stream = torch.cuda.Stream(device=self.device)   # device->host
        self.model_stream: Stream = torch.cuda.Stream(device=self.device)

        self.profiler = profiler
        self.async_profiling = AsyncProfilingHelper()
        self.valid = True

        assert num_normal_steps >= 12
        self.num_normal_steps = num_normal_steps

    def _h2d_transfer(self, curr_it: int):
        with torch.profiler.record_function(f"_h2d_transfer_{curr_it}"):
            try:
                data, target, other_meta = next(self.it)
            except StopIteration:
                return torch.Tensor(), torch.Tensor(), None, None

            with torch.cuda.stream(self.stream1):  # type: ignore
                with self.async_profiling.region("h2d", curr_it, self.stream1):
                    assert data.is_pinned()
                    data = data.to(self.device, non_blocking=True)
                    if self.move_target:
                        assert target.is_pinned()
                        target = target.to(self.device, non_blocking=True)

            event = torch.cuda.Event()
            event.record(self.stream1)

            return data, target, other_meta, event

    def _init(self):
        raise NotImplementedError

    def _normal_step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        raise NotImplementedError

    def _step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        return self._normal_step(data, target, other_meta, curr_it)

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError

    def run(self, n_iter=99999999, start=0, use_tqdm=True) -> list:
        assert self.valid, "Can only use once, re-init class"
        self.valid = False
        self._init()

        self.profiler.start(sleep_sec=2)
        torch.cuda.synchronize()

        res, gbuffer = [], None
        event2, event3 = torch.cuda.Event(), torch.cuda.Event()
        one_time_events = [torch.cuda.Event() for _ in range(3)]
        f1 = self.ex.submit(self._h2d_transfer, 0)

        if use_tqdm:
            iters = tqdm(range(start, n_iter), initial=start)
        else:
            iters = range(start, n_iter)
        for curr_it in iters:
            self._pre_step(curr_it)
            data, target, other_meta, event1 = f1.result()
            if event1 is None:  # means finished
                break

            event1.wait(self.model_stream)

            # prev round gbuffer must have completed since they are in same stream
            with torch.cuda.stream(self.model_stream):  # type: ignore
                with self.async_profiling.region("model", curr_it, self.model_stream):
                    output = self._step(data, target, other_meta, curr_it)

            bs = output.shape[0]
            if gbuffer is None:
                gbuffer = torch.empty_like(output)
            event3.wait(self.stream1)

            # loop idx+1: start data transfer
            f1 = self.ex.submit(self._h2d_transfer, curr_it + 1)

            # wait idx-1 to prevent to fast CPU submit of tasks
            event2.synchronize()

            # loop idx-1: wait for result transfer
            event2.wait(self.model_stream)
            with torch.cuda.stream(self.model_stream):  # type: ignore
                with self.async_profiling.region("stage", curr_it, self.model_stream):
                    gbuffer[:bs].copy_(output, non_blocking=True)
            event3.record(self.model_stream)  # event3: complete copy to gbuffer, can reuse output

            event3.wait(self.stream2)
            buffer = torch.empty_like(output, pin_memory=True, device="cpu")
            with torch.cuda.stream(self.stream2):  # type: ignore
                with self.async_profiling.region("d2h", curr_it, self.stream2):
                    buffer.copy_(gbuffer[:bs], non_blocking=True)
            event2.record(self.stream2)    # event2: complete copy to host, can reuse gbuffer

            # one-time event used in async post-step
            event4 = one_time_events[curr_it % len(one_time_events)]
            # it should have been synced, but for safety we sync again
            event4.synchronize()
            event4.record(self.stream2)

            self.ex2.submit(self._post_step_async, res, buffer, target, other_meta, curr_it, event4)
            self.profiler.step()

        torch.cuda.synchronize()
        self.profiler.stop()
        self.ex.shutdown(wait=True)
        self.ex2.shutdown(wait=True)

        return res

    def _post_step_async(self, res, buffer, target, other_meta, curr_it, event: Event):
        """(in sub-thread) ex2 will execute _post_step after event is recorded (i.e., buffer copy done)

        we need this method and the thread because we need to ensure data has arrived at buffer in cpu
        before we do anything with it
        """

        with self.async_profiling.region("post_wait", curr_it, None):
            event.synchronize()
        with torch.profiler.record_function(f"_post_step_{curr_it}"):
            with self.async_profiling.region("post", curr_it, None):
                self._post_step(res, buffer, target, other_meta, curr_it)

class _SyncBase:
    device: torch.device
    move_target: bool
    model: nn.Module
    dataloader: Iterable | torch.utils.data.DataLoader
    profiler: MyProfiler
    amp_scaler: MyScaler
    valid: bool

    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, move_target: bool,
        num_normal_steps: int
    ):
        self.device = torch.device(device)
        self.move_target = move_target
        self.model = model
        self.dataloader = dataloader
        self.amp_scaler = amp_scaler
        self.model_stream: Stream = torch.cuda.Stream(device=self.device)

        self.profiler = profiler
        self.valid = True

        assert num_normal_steps >= 12
        self.num_normal_steps = num_normal_steps

    def _init(self):
        raise NotImplementedError

    def _normal_step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        raise NotImplementedError

    def _step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        return self._normal_step(data, target, other_meta, curr_it)

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError

    def run(self, n_iter=99999999, start=0, use_tqdm=True) -> list:
        assert self.valid, "Can only use once, re-init class"
        self.valid = False
        self._init()

        self.profiler.start(sleep_sec=2)
        torch.cuda.synchronize()

        res = []
        it = iter(self.dataloader)

        if use_tqdm:
            iters = tqdm(range(start, n_iter), initial=start)
        else:
            iters = range(start, n_iter)
        for curr_it in iters:
            data, target, other_meta = next(it)
            # print(data.ravel()[:3], target.ravel()[:3], other_meta[:3])
            self._pre_step(curr_it)
            data = data.to(self.device)
            if self.move_target:
                target = target.to(self.device)

            with torch.cuda.stream(self.model_stream):  # type: ignore
                output = self._step(data, target, other_meta, curr_it)
            torch.cuda.synchronize()
            buffer = output.cpu()

            self._post_step(res, buffer, target, other_meta, curr_it)
            self.profiler.step()

        torch.cuda.synchronize()
        self.profiler.stop()

        return res


class InferenceProto(Protocol):
    model: nn.Module
    device: torch.device
    amp_scaler: MyScaler

    num_normal_steps: int

    model_stream: torch.cuda.Stream
    _g: torch.cuda.CUDAGraph | None
    _static_inputs: tuple[torch.Tensor, torch.Tensor] | None
    _static_outputs: torch.Tensor | None

    def _normal_step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        ...

class TrainProto(Protocol):
    model: nn.Module
    device: torch.device
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    lr_sched: Scheduler
    amp_scaler: MyScaler

    num_normal_steps: int

    model_stream: torch.cuda.Stream
    _g: torch.cuda.CUDAGraph | None
    _static_inputs: tuple[torch.Tensor, torch.Tensor] | None
    _static_outputs: torch.Tensor | None

    def _normal_step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        ...

# fix last batch size not match cuda graph problem
class InferenceMixin:
    def _init(self: InferenceProto):
        self.model.eval()
        self._g = None
        self._static_inputs = None
        self._static_outputs = None

    def _normal_step(self: InferenceProto, data, target, other_meta, curr_it) -> torch.Tensor:
        with torch.no_grad(), self.amp_scaler.autocast():
            output = self.model(data)
        return output

    def _step(self: InferenceProto, data: torch.Tensor, target: torch.Tensor, other_meta, curr_it: int) -> torch.Tensor:
        if curr_it <= self.num_normal_steps:
            v = self._normal_step(data, target, other_meta, curr_it)
            return v
        if self._g is None:
            # capture
            self._g = torch.cuda.CUDAGraph()
            self._static_inputs = (data.clone(), torch.tensor(0, device=self.device))
            self._static_outputs = torch.zeros_like(target, device=self.device)
            torch.cuda.synchronize()

            # it looks like everything is not updated but only recorded,
            # so we have to replay once
            with torch.cuda.graph(self._g):
                with torch.no_grad(), self.amp_scaler.autocast():
                    output = self.model(self._static_inputs[0])
                    self._static_outputs.copy_(output)

            torch.cuda.synchronize()
        else:
            # replay
            assert self._static_inputs is not None
            assert self._static_outputs is not None

            if self._static_inputs[0].shape[0] != data.shape[0]:
                # maybe last small batch, fall back to normal step
                return self._normal_step(data, target, other_meta, curr_it)

            self._static_inputs[0].copy_(data)

        self._g.replay()

        # it is not copied but only detached!
        result = self._static_outputs.detach()
        return result

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self: InferenceProto, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError

class TrainMixin:
    def _init(self: TrainProto):
        self.model.train()
        self._g = None
        self._static_inputs = None
        self._static_outputs = None

    def _normal_step(self: TrainProto, data, target, other_meta, curr_it) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)

        with self.amp_scaler.autocast():
            output = self.model(data)
            loss = self.loss_fn(output, target.to(dtype=output.dtype))

        self.amp_scaler.scale(loss).backward()
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()

        self.lr_sched.step()

        loss2 = loss.detach()
        if loss2.dim() == 0:
            loss2.unsqueeze_(0)
        return loss2

    def _step(self: TrainProto, data: torch.Tensor, target: torch.Tensor, other_meta, curr_it: int) -> torch.Tensor:
        if curr_it <= self.num_normal_steps:
            v = self._normal_step(data, target, other_meta, curr_it)
            return v

        if self._g is None:
            # capture
            self._g = torch.cuda.CUDAGraph()
            self._static_inputs = (data.clone(), target.to(dtype=data.dtype).clone())
            self._static_outputs = torch.empty(1, device=self.device)
            torch.cuda.synchronize()

            # copy data to static input
            self.optimizer.zero_grad(set_to_none=True)

            # it looks like everything is not updated but only recorded,
            # so we have to replay once
            with torch.cuda.graph(self._g):
                with self.amp_scaler.autocast():
                    output = self.model(self._static_inputs[0])
                    loss = self.loss_fn(output, self._static_inputs[1])
                    self._static_outputs.copy_(loss)

                self.amp_scaler.scale(self._static_outputs).backward()
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
                self.lr_sched.step()

            torch.cuda.synchronize()

        else:
            # replay
            assert self._static_inputs is not None
            assert self._static_outputs is not None

            if self._static_inputs[0].shape[0] != data.shape[0]:
                # maybe last small batch, fall back to normal step
                return self._normal_step(data, target, other_meta, curr_it)

            self._static_inputs[0].copy_(data)
            self._static_inputs[1].copy_(target.to(dtype=data.dtype))
            # self._static_inputs[1].copy_(target)

        self._g.replay()

        loss2 = self._static_outputs.detach()
        if loss2.dim() == 0:
            loss2.unsqueeze_(0)
        return loss2


    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self: TrainProto, res: list, buffer: torch.Tensor, target, other_meta, curr_it):
        raise NotImplementedError


class AsyncInference(InferenceMixin, _AsyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, num_normal_steps: int
    ):
        super().__init__(
            model=model, dataloader=dataloader,
            amp_scaler=amp_scaler, profiler=profiler, device=device,
            move_target=False, num_normal_steps=num_normal_steps
        )

class SyncInference(InferenceMixin, _SyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, num_normal_steps: int
    ):
        super().__init__(
            model=model, dataloader=dataloader,
            amp_scaler=amp_scaler, profiler=profiler, device=device,
            move_target=False, num_normal_steps=num_normal_steps
        )

class AsyncTrainer(TrainMixin, _AsyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        loss_fn: Callable, optimizer: torch.optim.Optimizer,
        lr_sched: Scheduler,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, num_normal_steps: int
    ):
        super().__init__(
            model=model, dataloader=dataloader,
            amp_scaler=amp_scaler, profiler=profiler, device=device,
            move_target=True, num_normal_steps=num_normal_steps
        )

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_sched = lr_sched

class SyncTrainer(TrainMixin, _SyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        loss_fn: Callable, optimizer: torch.optim.Optimizer,
        lr_sched: Scheduler,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, num_normal_steps: int
    ):
        super().__init__(
            model=model, dataloader=dataloader,
            amp_scaler=amp_scaler, profiler=profiler, device=device,
            move_target=True, num_normal_steps=num_normal_steps
        )

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_sched = lr_sched
