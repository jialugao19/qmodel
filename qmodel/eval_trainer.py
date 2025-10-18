import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Protocol, Iterable
import time
from tqdm import tqdm

from qmodel.amp_scaler import MyScaler
from qmodel.profiler import MyProfiler

class _AsyncBase:
    device: torch.device
    move_target: bool
    model: nn.Module
    dataloader: Iterable | torch.utils.data.DataLoader
    profiler: MyProfiler
    valid: bool

    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        device: torch.device, move_target: bool,
        do_profile: bool, profile_dir: str
    ):
        self.device = torch.device(device)
        self.move_target = move_target

        self.model = model
        self.dataloader = dataloader
        self.it = iter(dataloader)

        self.ex = ThreadPoolExecutor(max_workers=1)
        self.stream1 = torch.cuda.Stream(device=self.device)   # host->device
        self.stream2 = torch.cuda.Stream(device=self.device)   # device->host
        self.model_stream = torch.cuda.current_stream(device=self.device)

        self.profiler = MyProfiler(do_profile, profile_dir, wait=20, warmup=20, active=12)
        self.valid = True

    def _h2d_transfer(self):
        try:
            data, target, other_meta = next(self.it)
        except StopIteration:
            return torch.Tensor(), torch.Tensor(), None, True

        with torch.cuda.stream(self.stream1):  # type: ignore
            data = data.to(self.device, non_blocking=True)
            if self.move_target:
                target = target.to(self.device, non_blocking=True)

        event = torch.cuda.Event()
        event.record(self.stream1)
        event.wait(self.model_stream)

        return data, target, other_meta, False

    def _init(self):
        raise NotImplementedError

    def _step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        raise NotImplementedError

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError

    def run(self, n_iter=99999999, start=0):
        assert self.valid, "Can only use once, re-init class"
        self.valid = False
        self._init()

        self.profiler.start(sleep_sec=2)
        torch.cuda.synchronize()

        res, gbuffer = [], None
        event2, event3 = torch.cuda.Event(), torch.cuda.Event()
        f1 = self.ex.submit(self._h2d_transfer)

        for curr_it in tqdm(range(start, n_iter), initial=start):
            self._pre_step(curr_it)
            data, target, other_meta, finished = f1.result()
            if finished:
                break

            # prev round gbuffer must have completed since they are in same stream
            output = self._step(data, target, other_meta, curr_it)
            bs = output.shape[0]
            if gbuffer is None:
                gbuffer = torch.empty_like(output)
            event3.wait(self.stream1)

            # loop idx+1: start data transfer
            f1 = self.ex.submit(self._h2d_transfer)

            # loop idx-1: wait for result transfer
            event2.wait(self.model_stream)
            gbuffer[:bs].copy_(output, non_blocking=True)
            event3.record(self.model_stream)  # event3: complete copy to gbuffer, can reuse output

            event3.wait(self.stream2)
            buffer = torch.empty_like(output, pin_memory=True, device="cpu")
            with torch.cuda.stream(self.stream2):  # type: ignore
                buffer.copy_(gbuffer[:bs], non_blocking=True)
            event2.record(self.stream2)    # event2: complete copy to host, can reuse gbuffer

            self._post_step(res, buffer, target, other_meta, curr_it)
            self.profiler.step()

        torch.cuda.synchronize()
        self.profiler.stop()
        self.ex.shutdown(wait=True)

        return res

class _SyncBase:
    device: torch.device
    move_target: bool
    model: nn.Module
    dataloader: Iterable | torch.utils.data.DataLoader
    profiler: MyProfiler
    valid: bool

    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        device: torch.device, move_target: bool,
        do_profile: bool, profile_dir: str
    ):
        self.device = torch.device(device)
        self.move_target = move_target
        self.model = model
        self.dataloader = dataloader

        self.profiler = MyProfiler(do_profile, profile_dir)
        self.valid = True

    def _init(self):
        raise NotImplementedError

    def _step(self, data, target, other_meta, curr_it):
        raise NotImplementedError

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError

    def run(self, n_iter=99999999, start=0):
        assert self.valid, "Can only use once, re-init class"
        self.valid = False
        self._init()

        self.profiler.start(sleep_sec=2)
        torch.cuda.synchronize()

        res = []
        it = iter(self.dataloader)
        for curr_it in tqdm(range(start, n_iter), initial=start):
            data, target, other_meta = next(it)
            # print(data.ravel()[:3], target.ravel()[:3], other_meta[:3])
            self._pre_step(curr_it)
            data = data.to(self.device)
            if self.move_target:
                target = target.to(self.device)

            output = self._step(data, target, other_meta, curr_it)
            buffer = output.cpu()

            self._post_step(res, buffer, target, other_meta, curr_it)
            self.profiler.step()

        torch.cuda.synchronize()
        self.profiler.stop()

        return res


class InferenceProto(Protocol):
    model: nn.Module
    device: torch.device

class TrainProto(Protocol):
    model: nn.Module
    device: torch.device
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    lr_sched: torch.optim.lr_scheduler.LRScheduler
    scaler: MyScaler

class InferenceMixin:
    def _init(self: InferenceProto):
        self.model.eval()

    def _step(self: InferenceProto, data, target, other_meta, curr_it) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(data)
        return output

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self: InferenceProto, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError
        res.append([buffer, target.ravel(), other_meta])

class TrainMixin:
    def _init(self: TrainProto):
        self.model.train()

    def _step(self: TrainProto, data, target, other_meta, curr_it) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)

        with self.scaler.autocast():
            output = self.model(data)
            loss = self.loss_fn(output, target)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.lr_sched.step()

        loss2 = loss.detach()
        if loss2.dim() == 0:
            loss2.unsqueeze_(0)
        return loss2

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self: TrainProto, res: list, buffer: torch.Tensor, target, other_meta, curr_it):
        raise NotImplementedError
        # print(f"Iter {curr_it}, loss: {buffer}")
        res.append(buffer)


class AsyncInference(InferenceMixin, _AsyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        device: torch.device,
        do_profile=False, profile_dir="runs/profiler"
    ):
        super().__init__(
            model=model, dataloader=dataloader, device=device,
            move_target=False, do_profile=do_profile, profile_dir=profile_dir
        )

class SyncInference(InferenceMixin, _SyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        device: torch.device,
        do_profile=False, profile_dir="runs/profiler"
    ):
        super().__init__(
            model=model, dataloader=dataloader, device=device,
            move_target=False, do_profile=do_profile, profile_dir=profile_dir
        )

class AsyncTrainer(TrainMixin, _AsyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        loss_fn: Callable, optimizer: torch.optim.Optimizer,
        lr_sched: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device, amp_dtype: torch.dtype | None, do_profile=False, profile_dir="runs/profiler"
    ):
        super().__init__(
            model=model, dataloader=dataloader, device=device,
            move_target=True, do_profile=do_profile, profile_dir=profile_dir
        )

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_sched = lr_sched
        self.scaler = MyScaler(amp_dtype)

class SyncTrainer(TrainMixin, _SyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        loss_fn: Callable, optimizer: torch.optim.Optimizer,
        lr_sched: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device, amp_dtype: torch.dtype | None, do_profile=False, profile_dir="runs/profiler"
    ):
        super().__init__(
            model=model, dataloader=dataloader, device=device,
            move_target=True, do_profile=do_profile, profile_dir=profile_dir
        )

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_sched = lr_sched
        self.scaler = MyScaler(amp_dtype)
