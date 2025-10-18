import torch
from torch.utils.data import Sampler
from line_profiler import profile
import numpy as np

from qmodel.data.cpp.permute import Permute64

class CustomSampler(Sampler):
    def __init__(self, dataset, seed, shuffle=True, infinite=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.shuffler = Permute64(len(self.dataset), seed)
        self.not_yet_started = True
        self.start = 0

    # @profile
    # def __iter__(self):
    #     # indices = list(range(len(self.dataset)))
    #     while True:
    #         if self.shuffle:
    #             # np.random.shuffle(indices)
    #             for i in range(len(self.dataset)):
    #                 yield self.shuffler.permute(i)
    #         else:
    #             for i in range(len(self.dataset)):
    #                 yield i

    #         if not self.infinite:
    #             break

    def set_start_iter(self, iteration: int):
        self.start = iteration

    def __iter__(self):
        assert self.not_yet_started
        self.not_yet_started = False
        n = len(self.dataset)
        batch_size = 10_000  # 可调
        start = self.start

        while True:
            if self.shuffle:
                for st in range(start, n, batch_size):
                    end = min(st + batch_size, n)
                    batch = np.arange(st, end, dtype=np.int64)  # 不转 list，保持惰性
                    yield from self.shuffler.permute(batch).tolist()
            else:
                yield from range(start, n)

            if not self.infinite:
                break

    def __len__(self):
        return len(self.dataset)
