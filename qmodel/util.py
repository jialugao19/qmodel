import numpy as np
import pandas as pd
import polars as pl

def merge_date_time_dataframe(df: pd.DataFrame, date_col="date", time_col="time") -> pd.Series:

    tmp_df = pl.from_pandas(df[[date_col, time_col]])
    tmp_df = tmp_df.lazy().select(
        pl.col(date_col).add(2*10**7).cast(pl.Int64).mul(10**6)
        .add(pl.col(time_col).cast(pl.Int64))
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime("ms"), "%Y%m%d%H%M%S")
        .alias("DateTime")
    ).collect()
    return tmp_df.to_pandas().DateTime
