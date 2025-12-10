# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence
from tqdm import tqdm

import numpy as np
import polars as pl

OUTPUT_DIR = Path("/dev/shm/tmp")


def process_single(code, df):

    # 当天日期
    df = df.with_columns(pl.col("DateTime").dt.date().alias("Date"))

    assert not pl.select(pl.any_horizontal(pl.col("*").is_null() | pl.col("*").is_nan() | pl.col("*").is_infinite())).item()

    # 每日 00:00:00 的 Close
    day_open_close = (
        df.filter(pl.col("DateTime").dt.time() == pl.time(0, 0, 0))
        .select([pl.col("Date").alias("Date"), pl.col("Close").alias("Day0_Close")])
        # 显式向后移动一天（下一天用今天的 00:00:00)
        .with_columns((pl.col("Date") + pl.duration(days=1)).alias("NextDate"))
        .select(["NextDate", "Day0_Close"])
    )

    # 左连接，把 yesterday 00:00:00 close 对齐到今天
    df2 = (
        df.join(day_open_close, left_on="Date", right_on="NextDate", how="left")
        .with_columns(
            # 用昨天的 00:00:00 Close 做归一化因子
            pl.col("Day0_Close").forward_fill()
        )
        .select(
            [
                pl.col(["DateTime", "StockCode", "TransAmount", "TransCount"]),
                (pl.col("Open") / pl.col("Day0_Close") - 1).alias("NormOpen"),
                (pl.col("High") / pl.col("Day0_Close") - 1).alias("NormHigh"),
                (pl.col("Low") / pl.col("Day0_Close") - 1).alias("NormLow"),
                (pl.col("Close") / pl.col("Day0_Close") - 1).alias("NormClose"),
                (pl.col("Close").shift(-30) / pl.col("Close").shift(-1)).sub(1).alias(  "Ret1_30"  ),
                (pl.col("Close").shift(-30) / pl.col("Close").shift(-10)).sub(1).alias( "Ret10_30" ),
                (pl.col("Close").shift(-60) / pl.col("Close").shift(-1)).sub(1).alias(  "Ret1_60"  ),
                (pl.col("Close").shift(-60) / pl.col("Close").shift(-10)).sub(1).alias( "Ret10_60" ),
                (pl.col("Close").shift(-90) / pl.col("Close").shift(-1)).sub(1).alias(  "Ret1_90"  ),
                (pl.col("Close").shift(-90) / pl.col("Close").shift(-10)).sub(1).alias( "Ret10_90" ),
                (pl.col("Close").shift(-180) / pl.col("Close").shift(-10)).sub(1).alias("Ret10_180"),
                (pl.col("Close").shift(-300) / pl.col("Close").shift(-10)).sub(1).alias("Ret10_300"),
                (pl.col("Close").shift(-600) / pl.col("Close").shift(-10)).sub(1).alias("Ret10_600"),
                (pl.col("Close").shift(-900) / pl.col("Close").shift(-10)).sub(1).alias("Ret10_900"),
            ]
        ).filter(
            ~pl.any_horizontal(pl.col("*").is_null())
        )
    )


    return df2

# invalid data point exists: some leading rows of norm open does not have prev day close so they are none. recheck valid attribute
def process_all(data: pl.DataFrame, window_size: int) -> pl.DataFrame:
    dfs = []
    i = 0
    for (code,), df in tqdm(data.group_by(["StockCode"])):
        # if i >= 5:  break
        i += 1
        df2 = process_single(code, df)
        df2 = df2.with_columns(
            pl.lit(True).alias("valid"),
        ).with_columns((
            pl.col("valid").shift(240, fill_value=False) &
            pl.col("TransAmount").rolling_sum(5).gt(1.0) &
            pl.col("TransCount").rolling_sum(5).gt(10.0)
        ).alias("valid"))

        # doing normalization here
        # store date and second separately in seconds
        df3 = df2.select(
            pl.col("StockCode").cast(pl.Float32),
            pl.col("DateTime").dt.date().cast(pl.Int32).cast(pl.Float32).alias("Date"),
            (pl.col("DateTime").dt.timestamp("ms") - pl.col("DateTime").dt.date().cast(pl.Datetime).dt.timestamp("ms")).truediv(1000).cast(pl.Float32).alias("Time"),
            # pl.col("TransAmount", "TransCount"),
            pl.col("TransAmount").add(700).sqrt().log10().mul(3).sub(5).cast(pl.Float32),
            pl.col("TransCount").add(5).sqrt().log10().mul(7).sub(3).cast(pl.Float32),
            (pl.col("NormOpen" ).sign() * pl.col("NormOpen" ).abs().pow(0.5).mul(5)).cast(pl.Float32),
            (pl.col("NormHigh" ).sign() * pl.col("NormHigh" ).abs().pow(0.5).mul(5)).cast(pl.Float32),
            (pl.col("NormLow"  ).sign() * pl.col("NormLow"  ).abs().pow(0.5).mul(5)).cast(pl.Float32),
            (pl.col("NormClose").sign() * pl.col("NormClose").abs().pow(0.5).mul(5)).cast(pl.Float32),
            (pl.col("Ret1_30"  ).sign() * pl.col("Ret1_30"  ).abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret10_30" ).sign() * pl.col("Ret10_30" ).abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret1_60"  ).sign() * pl.col("Ret1_60"  ).abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret10_60" ).sign() * pl.col("Ret10_60" ).abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret1_90"  ).sign() * pl.col("Ret1_90"  ).abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret10_90" ).sign() * pl.col("Ret10_90" ).abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret10_180").sign() * pl.col("Ret10_180").abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret10_300").sign() * pl.col("Ret10_300").abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret10_600").sign() * pl.col("Ret10_600").abs().pow(0.5).mul(30)).cast(pl.Float32),
            (pl.col("Ret10_900").sign() * pl.col("Ret10_900").abs().pow(0.5).mul(30)).cast(pl.Float32),
            pl.col("valid")
        ).with_columns(
            pl.any_horizontal(pl.exclude([pl.Datetime, pl.Boolean]).is_null() | pl.exclude([pl.Datetime, pl.Boolean]).is_nan() | pl.exclude([pl.Datetime, pl.Boolean]).is_infinite()).not_().and_(pl.col("valid")).alias("valid")
        ).filter(
            pl.col("valid").rolling_max(window_size=window_size+1, ).shift(-window_size).forward_fill()
        )
        dfs.append(df3)

    res = pl.concat(dfs)
    return res


def stats(data: pl.DataFrame):
    print(f"data stats of shape {data.shape}:")
    # print kurt, std, mean
    for col in data.columns:
        print("  ", col, data.select([
            pl.col(col).kurtosis().alias("kurtosis"),
            pl.col(col).std().alias("std"),
            pl.col(col).mean().alias("mean"),
        ]).to_numpy().flatten().tolist())

def process(group, files):
    window_size = 240
    data = pl.scan_ipc(files, memory_map=False).collect()

    res = process_all(data, window_size)
    valid = res.select(pl.col("valid")).to_numpy().flatten()
    valid_idx = np.where(valid)[0]
    raw_data = res.drop("valid").to_numpy(order="fortran")
    stats(res.filter(pl.col("valid")))

    assert not np.any(np.isnan(raw_data))
    assert not np.any(np.isinf(raw_data))
    print("data shape:", raw_data.shape)
    print("valid size:", valid_idx.shape)

    meta = {
        "shape": raw_data.shape,
        "valid_size": valid_idx.shape[0],
    }
    with open(f"/dev/shm/tmp/{group}.json", "w+") as f:
        json.dump(meta, f)

    # open mmap to write
    print("start saving")
    f1 = f"/dev/shm/tmp/{group}.bin"
    f2 = f"/dev/shm/tmp/{group}_idx.bin"
    data_mm = np.memmap(f1, mode="w+", dtype=np.float32, shape=raw_data.shape, order="F")
    idx_mm = np.memmap(f2, mode="w+", dtype=np.int64, shape=valid_idx.shape, order="F")
    data_mm[:] = raw_data[:]
    idx_mm[:] = valid_idx[:]
    data_mm.flush()
    idx_mm.flush()
    return res

#%%
# DEFAULT_DATA_DIR = Path("/scratch/xli/findata/crypto/crypto1tresample/1s")
DEFAULT_DATA_DIR = Path("/mnt/findata/crypto/crypto1tresample/1m")

# DEFAULT_PATTERN = "1m-*.feather"

import os

train_files = []
test_files = []
for file in os.listdir(DEFAULT_DATA_DIR):
    date_str = file[3:13].replace("-", "")
    if date_str <= "20250715":
        train_files.append(DEFAULT_DATA_DIR / file)
    elif date_str <= "20250815":
        test_files.append(DEFAULT_DATA_DIR / file)

#%%

res1 = process("train", train_files)
process("test", test_files)

#%%

exit(0)
#%%
res = res1
res.to_pandas()[300:]

#%%

res2 = res.to_pandas()
res3 = res2[res2.valid]
res3.shape
#%%

t = res3.TransAmount
s = res3.TransCount

# %%

t2 = (np.log10((t+700)**0.5)) * 3 - 5
print(t2.std(), t2.kurt(), t2.mean())
t2.hist(bins=100, log=False)

#%%
s2 = (np.log10((s+5)**0.5)) * 7 - 3
print(s2.std(), s2.kurt(), s2.mean())
s2.hist(bins=100)

#%%

# v = res3.NormOpen
#%%
# import pandas as pd
# (v**1).kurt()
#%%
# (v**0.5).hist(bins=100, log=False)

#%%

# res3.describe()

#%%

o = res3.Ret10_300

# o = np.sign(o) * np.abs(o) ** 0.5 * 5
#%%

o.hist(bins=100, log=True)
(o.kurt(), o.std(), o.mean())


# too little trans amount / vol will cause wrong result, try to filter away more