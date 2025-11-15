# use pytest

import torch
import torch.nn as nn
import unittest

from qmodel.core.eval_trainer import AsyncInference, AsyncTrainer, SyncInference, SyncTrainer
from qmodel.components.amp_scaler import MyScaler
from qmodel.components.profiler import MyProfiler
from qmodel.config import ProfilerConfig

class Result(nn.Module):
    def forward(self, x):
        return x[1][-1, :, -1:]

def get_model(dtype):
    torch.random.manual_seed(42)
    model = nn.Sequential(*[nn.GRU(8, 64, 5, batch_first=True), Result()]).to(dtype=dtype).cuda()
    return model

def get_data(num_samples, dtype, seed):
    torch.random.manual_seed(seed)
    data_tensor = torch.randn(num_samples, 32, 8).to(dtype=dtype)
    target_tensor = torch.randn(num_samples, 1).to(dtype=dtype)
    return data_tensor, target_tensor

def get_dataloader(num_samples, bs, dtype, seed):
    data_tensor, target_tensor = get_data(num_samples, dtype, seed)
    n_batch = num_samples // bs
    return (
        (data_tensor[i*bs:(i+1)*bs].pin_memory(), target_tensor[i*bs:(i+1)*bs].pin_memory(), target_tensor[i*bs:(i+1)*bs])
        for i in range(n_batch)
    )

def _post_step_eval(res, buffer, target, other_meta, curr_it):
    res.append([buffer, target.ravel(), other_meta])

def _post_step_train(res: list, buffer: torch.Tensor, target, other_meta, curr_it):
    # print(f"Iter {curr_it}, loss: {buffer}")
    res.append(buffer)

def test_train():
    device = torch.device("cuda")
    n_sample = 5120
    bs = 32
    seed = 42
    dtype = torch.float32

    # model1
    model1 = get_model(dtype)
    optim1 = torch.optim.SGD(model1.parameters(), 1e-2)

    dataloader1 = get_dataloader(n_sample, bs, dtype, seed)
    sched1 = torch.optim.lr_scheduler.ConstantLR(optim1, 1/3, 1000)

    amp_dtype = torch.float16
    scaler1 = MyScaler(amp_dtype, dtype)
    scaler2 = MyScaler(amp_dtype, dtype)

    profiler1 = MyProfiler(False, ProfilerConfig("none", ""))
    profiler2 = MyProfiler(False, ProfilerConfig("none", ""))
    profiler3 = MyProfiler(False, ProfilerConfig("none", ""))
    profiler4 = MyProfiler(False, ProfilerConfig("none", ""))

    trainer1 = SyncTrainer(
        model1, dataloader1, nn.MSELoss(), optim1, sched1, scaler1, profiler1, device)
    trainer1._post_step = _post_step_train
    trainer1._pre_step = lambda a: a  # type: ignore
    res1 = trainer1.run(n_iter=n_sample//bs)

    # model2
    model2 = get_model(dtype)
    optim2 = torch.optim.SGD(model2.parameters(), 1e-2)

    dataloader2 = get_dataloader(n_sample, bs, dtype, seed)
    sched2 = torch.optim.lr_scheduler.ConstantLR(optim2, 1/3, 1000)

    trainer2 = AsyncTrainer(
        model2, dataloader2, nn.MSELoss(), optim2, sched2, scaler2, profiler2, device)
    trainer2._post_step = _post_step_train
    trainer2._pre_step = lambda a: a  # type: ignore
    res2 = trainer2.run(n_iter=n_sample//bs)

    r1 = torch.utils.data.default_collate(res1)
    r2 = torch.utils.data.default_collate(res2)
    assert torch.allclose(r1, r2, rtol=0, atol=0)

    assert all([
        torch.allclose(i, j, rtol=0, atol=0)
        for i, j in zip(model1.parameters(), model2.parameters())
    ])

    print("Train verification success")

    # === compare sync and async inference result ===

    dataloader1 = get_dataloader(n_sample, bs, dtype, seed)
    eval1 = SyncInference(model1, dataloader1, scaler1, profiler3, device)
    eval1._post_step = _post_step_eval
    eval1._pre_step = lambda a: a  # type: ignore
    res1 = eval1.run(n_iter=n_sample//bs)

    dataloader2 = get_dataloader(n_sample, bs, dtype, seed)
    eval2 = AsyncInference(model2, dataloader2, scaler2, profiler4, device)
    eval2._post_step = _post_step_eval
    eval2._pre_step = lambda a: a  # type: ignore
    res2 = eval2.run(n_iter=n_sample//bs)

    r1 = torch.utils.data.default_collate(res1)[0]
    r2 = torch.utils.data.default_collate(res2)[0]
    assert torch.allclose(r1, r2, rtol=0, atol=0)

    print("Eval verification success")

if __name__ == "__main__":
    test_train()
