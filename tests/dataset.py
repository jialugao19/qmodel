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

class AQ2MinDataset(Dataset):
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

        assert raw[:, -1, -1].min() == 1
        data = raw[:, :, 3:11]
        target = raw[:, -1, 11:12]

        v1 = torch.from_numpy(data).to(self.dtype)
        v2 = torch.from_numpy(target)
        v3 = torch.from_numpy(raw[:, -1, :3].astype(np.int32))

        return v1, v2, v3

class AQ2DailyDataset(Dataset):

    @dataclass
    class StorageStat:
        filenames: np.ndarray
        indices: list
        lengths: np.ndarray
        cumsum: np.ndarray

        def get_code(self, idx: int):
            """compute from searchsorted"""
            i = np.searchsorted(self.cumsum, idx, side='right')
            return self.filenames[i], self.indices[i][idx - (self.cumsum[i - 1] if i > 0 else 0)]

    def __init__(self, group: str, ret_col_name: str, window_size: int):
        assert group in ["train", "test", "predict"]
        self.group = group
        self.ret_col_name = ret_col_name
        self.window_size = window_size

        self.storage = self.get_indices()
        self.ret = pd.read_feather(f"target_{self.group}.feather")

        self.total = self.storage.cumsum[-1]

    def get_indices(self) -> StorageStat:
        """read the length of each file"""
        indices = pd.read_parquet(f"indices_{self.group}.parquet")

        # indices dataframe contains col: filename length
        # compute cumsum
        indices["cumsum"] = indices["length"].cumsum()
        v = indices["code"].astype(str).str.pad(6, "left", "0").tolist()

        filenames = []
        # ex_map = {"0": "SZ", "3": "SZ", "4": "BJ", "6": "SH", "8": "BJ", "9": "BJ"}
        file_group = {
            "train": "data/stock1m",
            "test": "data/stock1m",
            "predict": "data/stock3s1min",
        }[self.group]
        for code in v:
            # ex = ex_map[code[0]]
            filenames.append(f"{file_group}/{code}.feather")

        return AQ2Dataset.StorageStat(
            filenames=np.array(filenames),
            indices=indices["indices"].tolist(),
            lengths=np.array(indices["length"]).flatten(),
            cumsum=np.array(indices["cumsum"]).flatten()
        )

    def __len__(self):
        return self.total


    def _get_ret_and_pre_close(self, date, code):
        """search sorted get the start of code/date and end"""
        c = self.ret["code"]
        left, right = np.searchsorted(c, code, side='left'), np.searchsorted(c, code, side='right')
        d = self.ret["date"][left:right]
        idx = np.searchsorted(d, date, side='left') + left
        row = self.ret.iloc[idx]

        assert int(row["code"]) == code and int(row["date"]) == date, \
            f'{int(row["code"])}, {code}, {int(row["date"])}, {date}'

        return row[self.ret_col_name], row["adj_base"]

    @profile
    def read_data(self, filename, idx):
        if TmpFeather.exists_in_shm(raw_filename=filename):
            tmp_df = TmpFeather.from_shm_file(TmpFeather.get_shm_filename(filename))
        else:
            df = pl.scan_ipc(filename, memory_map=False).select(
                pl.col(["open", "high", "low", "close", "amount"]),
                pl.col("trade_date").cast(pl.Int32)
            ).collect()
            tmp_df = TmpFeather.from_df(df, name=filename)

        ws = self.window_size - 1
        df = tmp_df.df[idx - ws:idx + 1]
        trade_date = int(df.select(pl.col("trade_date").last()).item())
        code = int(filename[-14:-8])

        arr = df[:, :5].to_numpy()
        arr = arr.copy().astype(np.float32)

        if self.group != "predict":
            assert trade_date < 20250207
        else:
            assert trade_date >= 20250207

        return arr, code, trade_date


    @profile
    def __getitem__(self, idx):
        filename, row_idx = self.storage.get_code(idx)

        data, code, date = self.read_data(filename, row_idx)

        ret, pre_close = self._get_ret_and_pre_close(date, code)

        data[:, :4] = (data[:, :4] / pre_close - 1) * 40  # just mul 40, a magic number to control std
        data[:, 4] = np.log10(data[:, 4] + 1e2) - 5  # log10 transform amount
        target = np.array([ret], dtype=np.float32) * 40  # scale target to match the data
        target = np.clip(target, -0.2 * 40, 0.2 * 40)
        return torch.from_numpy(data[:, :]), torch.from_numpy(target.reshape(-1)), torch.tensor([code, date], dtype=torch.int32)


# AQ2Dataset = AQ2DailyDataset
AQ2Dataset = AQ2MinDataset
