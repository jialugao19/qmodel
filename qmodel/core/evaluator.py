import torch
import os

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import json
from line_profiler import profile
from pathlib import Path
import time

from torch.utils.tensorboard import SummaryWriter

from qmodel.logger import logger
from qmodel.util import merge_date_time_dataframe
from qmodel.data.dataloader import setup_eval_dataloader
from qmodel.core.eval_trainer import AsyncInference, SyncInference
from qmodel.components.profiler import MyProfiler
from qmodel.components.checkpoint import CheckpointSaver
from qmodel.components.amp_scaler import MyScaler
from qmodel.models import build_model

from qmodel.config import QConfig

class Inferencer(AsyncInference):
    def _post_step(self, res, buffer, target, other_meta, curr_it):
        res.append([buffer, target, other_meta])
        if len(res) >= 100:
            torch.cuda.synchronize()
            res2 = [torch.cat([r[i] for r in res]) for i in range(len(res[0]))]
            res.clear()
            res.append(res2)

    def _pre_step(self, curr_it: int):
        pass

class Evaluator:
    def __init__(self, config: QConfig, group: str):
        assert group in ["test", "predict"]
        self.config = config
        self.group  = group

        self.checkpointer = CheckpointSaver(config.root_dir, config, device=None)

        self.writer = SummaryWriter(log_dir=config.tensorboard_dir)

    def evaluate(self):
        files = [it for it in self.config.evaluator.eval_checkpoint_iter if it > 0]
        os.makedirs(os.path.join(self.config.root_dir, "eval"), exist_ok=True)

        for it in files:
            self.eval_single(it, n_iter=self.config.evaluator.eval_all_num_iters)

    def eval_partial(self, it: int, n_iter: int) -> dict:
        res = self.eval_single(it, n_iter=n_iter)
        return res

    def eval_single(self, it: int, n_iter=0) -> dict:
        config      = self.config
        device      = self.config.device
        amp_scaler  = MyScaler(config.use_amp, config.amp_dtype)
        model       = build_model(config.model_class, config.model).to(device=device, dtype=config.eval_dtype)
        do_profile  = config.profiler.profile_section == "eval"
        profiler    = MyProfiler(real_run=do_profile, config=config.profiler)

        dataloader  = setup_eval_dataloader(self.config, self.group, shuffle=False)


        ckpt = Path(self.config.root_dir) / "ckpt" / f"iter_{it}.pt"

        logger.info(f"Evaluating checkpoint: {ckpt}, iter {it}")
        self.checkpointer.load(ckpt, model, None, None, None)

        num_normal_steps = config.num_normal_steps if config.evaluator.eval_use_cuda_graph else 9999999999

        inferencer = Inferencer(
            model=model, dataloader=dataloader, device=device,
            amp_scaler=amp_scaler, profiler=profiler, num_normal_steps=num_normal_steps
        )
        if n_iter == 0:
            n_iter = len(dataloader)
        n_iter = min(n_iter, len(dataloader))
        res = inferencer.run(n_iter=n_iter)
        logger.info(f"finish running inference")

        res2 = []
        if len(res) > 0:
            for i in range(len(res[0])):
                res2.append(np.concatenate([r[i] for r in res], axis=0))
        logger.info(f"finish concat inference")

        cols = ['prediction', 'target', 'code', 'date', 'time']
        col_idx = 0
        res3 = {}
        for i in res2:
            assert len(i.shape) == 2
            assert col_idx + i.shape[1] <= len(cols), "no enough col labels"
            for k in range(i.shape[1]):
                res3[cols[col_idx]] = i[:, k]
                col_idx += 1
        res_df = pd.DataFrame(res3)

        logger.info(f"finish building dataframe")

        # convert date and time to datetime col
        res_df["StockCode"] = res_df["code"].astype(int)
        res_df["DateTime"] = merge_date_time_dataframe(res_df, "date", "time")
        logger.info(f"finish converting to datetime")

        # compute global IC
        start_time = time.time()
        global_ic = np.corrcoef(res_df['prediction'], res_df['target'])[0, 1]
        end_time = time.time()

        logger.info(f"global IC: {global_ic:.8f}, computed IC in {end_time-start_time:.3f}s")

        # save to feather
        os.makedirs(os.path.join(self.config.root_dir, "eval"), exist_ok=True)
        res_df.to_feather(os.path.join(self.config.root_dir, "eval", f"eval_{it}.feather"))

        metric_res = {
            "global IC": global_ic
        }

        return metric_res

    def eval_metrics(self, df):
        # todo: include this part
        res = []

        # df1 = pl.from_pandas(df)
        # assert df1.select(pl.col(["code", "date", "time"])).is_duplicated().max() == False
        # df2 = pl.scan_ipc("tmp_data/test.feather", memory_map=False).with_columns(pl.col("date") % 1000000).collect()
        # ret_col = self.config.ret_col_name

        # # df3 = pd.merge(df, df2, on=["code", "date", "time"], validate="1:1")
        # df3 = df1.join(df2, on=["code", "date", "time"], how="inner").to_pandas()

        # # todo: check uniqueness of df ["code", "date", "time"]: wrong eval?

        # # todo: accelerate below, too slow due to large feather
        # for d, grp in df3.groupby("date"):
        #     ic = grp["prediction"].corr(grp["target"])
        #     sic = grp["prediction"].corr(grp["target"], method="spearman")
        #     top = grp.nlargest(50, 'prediction')[ret_col].mean() * 10000
        #     res.append([d, ic, sic, top])
        # df4 = pd.DataFrame(res, columns=["date", "ic", "sic", "top50ret"])

        # ic = df["prediction"].corr(df["target"])
        # sic = df["prediction"].corr(df["target"], method="spearman")

        # loss = self.config.criterion(torch.from_numpy(df.prediction.values), torch.from_numpy(df.target.values)) / len(df)

        # return {
        #     "loss": loss.item(),
        #     "ic": ic,
        #     "sic": sic,
        #     "daily_ic": df4["ic"].mean(),
        #     "daily_sic": df4["sic"].mean(),
        #     "topret": df4["top50ret"].mean(),
        # }
