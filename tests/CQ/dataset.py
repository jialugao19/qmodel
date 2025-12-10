import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import os
import json
from numpy.lib import recfunctions as rfn

from line_profiler import profile

from qmodel.data.TmpFeather import TmpFeather

class DataStorage:
    data: np.memmap
    vidx: np.memmap

    def __init__(self, group: str):
        with open(f"/dev/shm/tmp/{group}.json", "r") as f:
            meta = json.load(f)
        shape = tuple(meta["shape"])
        shape2 = meta["valid_size"]

        f1 = f"/dev/shm/tmp/{group}.bin"
        f2 = f"/dev/shm/tmp/{group}_idx.bin"

        self.data = np.memmap(f1, mode="r", dtype=np.float32, shape=shape, order="F")
        self.vidx = np.memmap(f2, mode="r", dtype=np.int64, shape=shape2, order="F")

class CQ2MinDataset(Dataset):
    def __init__(self, group: str, dtype: torch.dtype, ret_col_name: str, window_size: int):
        assert group in ["train", "test", "predict"]
        self.group = group
        self.dtype = dtype
        self.ret_col_name = ret_col_name
        self.window_size = window_size

        self.d = DataStorage(group)

        self.total = self.d.vidx.size

    def __len__(self):
        return self.total

    @profile
    def __getitems__(self, raw_indices):
        indices = self.d.vidx[raw_indices]
        raw = np.array([self.d.data[idx+1-self.window_size:idx+1] for idx in indices])

        # assert raw[:, -1, -1].min() == 1
        # data = np.concatenate([raw[:, :, 3:4], raw[:, :, 5:9]], axis=-1)
        data = np.concatenate([raw[:, :, 3:9]], axis=-1)
        target = raw[:, -1, 13:14]

        v1 = torch.from_numpy(data).to(self.dtype)
        v2 = torch.from_numpy(target)
        v3 = torch.from_numpy(raw[:, -1, :3].astype(np.int32))

        return v1, v2, v3

# AQ2Dataset = AQ2DailyDataset
CQ2Dataset = CQ2MinDataset
