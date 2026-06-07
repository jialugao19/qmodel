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
from qmodel.components.distributed import DistributedContext
from qmodel.components.profiler import MyProfiler
from qmodel.components.checkpoint import CheckpointSaver
from qmodel.components.amp_scaler import MyScaler
from qmodel.models import build_model

from qmodel.config import QConfig

class Inferencer(AsyncInference):
    def _post_step(self, res, buffer, target, other_meta, curr_it):
        """Store one local inference batch."""
        # Accumulate local batches in the original inference result format.
        res.append([buffer, target, other_meta])
        if len(res) >= 100:
            torch.cuda.synchronize(self.device)
            res2 = [torch.cat([r[i] for r in res]) for i in range(len(res[0]))]
            res.clear()
            res.append(res2)

    def _pre_step(self, curr_it: int):
        pass

class Evaluator:
    def __init__(
        self,
        config: QConfig,
        group: str,
        distributed: DistributedContext | None = None,
    ):
        """Build an evaluator for one dataset group."""
        # Store group and distributed runtime context.
        assert group in ["test", "predict"]
        self.config = config
        self.group  = group
        self.distributed = distributed if distributed is not None else DistributedContext.from_config(config)
        self.config.device = self.distributed.device

        # Build checkpoint and optional writer handles.
        self.checkpointer = CheckpointSaver(config.root_dir, config, device=self.distributed.device)

        self.writer = SummaryWriter(log_dir=config.tensorboard_dir) if self.distributed.is_main else None

    def evaluate(self):
        """Evaluate configured checkpoints."""
        # Create output directory on rank zero.
        files = [it for it in self.config.evaluator.eval_checkpoint_iter if it > 0]
        if self.distributed.is_main:
            os.makedirs(os.path.join(self.config.root_dir, "eval"), exist_ok=True)
        self.distributed.barrier()

        # Evaluate each checkpoint on every rank.
        for it in files:
            self.eval_single(it, n_iter=self.config.evaluator.eval_all_num_iters)

    def eval_partial(self, it: int, n_iter: int) -> dict:
        res = self.eval_single(it, n_iter=n_iter)
        return res

    def eval_single(self, it: int, n_iter=0) -> dict:
        """Evaluate one checkpoint and return rank-zero metrics."""
        # Build rank-local evaluation runtime.
        config      = self.config
        device      = self.distributed.device
        amp_scaler  = MyScaler(config.use_amp, config.amp_dtype)
        model       = build_model(config.model_class, config.model).to(device=device, dtype=config.eval_dtype)
        do_profile  = config.profiler.profile_section == "eval"
        profiler    = MyProfiler(real_run=do_profile, config=config.profiler)

        dataloader  = setup_eval_dataloader(
            self.config,
            self.group,
            shuffle=False,
            distributed=self.distributed,
        )

        ckpt = Path(self.config.root_dir) / "ckpt" / f"iter_{it}.pt"

        logger.info(f"Evaluating checkpoint: {ckpt}, iter {it}")
        self.checkpointer.load(ckpt, model, None, None, None)

        num_normal_steps = config.num_normal_steps if config.evaluator.eval_use_cuda_graph else 9999999999

        inferencer = Inferencer(
            model=model, dataloader=dataloader, device=device,
            amp_scaler=amp_scaler, profiler=profiler, num_normal_steps=num_normal_steps,
            distributed=self.distributed,
        )
        if n_iter == 0:
            n_iter = len(dataloader)
        n_iter = min(n_iter, len(dataloader))
        res = inferencer.run(n_iter=n_iter)
        logger.info(f"finish running inference")

        # Gather rank-local batches and continue only on rank zero.
        local_rows = self._local_rows(res)
        gathered_rows = self.distributed.gather_object(local_rows)
        if not self.distributed.is_main:
            return {}

        # Reconstruct global eval row order from step and rank.
        rows = [row for rank_rows in gathered_rows for row in rank_rows]
        rows.sort(key=lambda row: (row[0], row[1]))
        res2 = self._concat_rows(rows)
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

        # Convert date and time to datetime columns.
        res_df["StockCode"] = res_df["code"].astype(int)
        res_df["DateTime"] = merge_date_time_dataframe(res_df, "date", "time")
        logger.info(f"finish converting to datetime")

        # Compute global IC on rank zero.
        start_time = time.time()
        global_ic = np.corrcoef(res_df['prediction'], res_df['target'])[0, 1]
        end_time = time.time()

        logger.info(f"global IC: {global_ic:.8f}, computed IC in {end_time-start_time:.3f}s")

        # Save evaluation output on rank zero.
        os.makedirs(os.path.join(self.config.root_dir, "eval"), exist_ok=True)
        res_df.to_feather(os.path.join(self.config.root_dir, "eval", f"eval_{it}.feather"))

        # Return scalar metrics to the caller.
        metric_res = {
            "global IC": global_ic
        }

        return metric_res

    def _local_rows(self, batches: list) -> list:
        """Convert local inference chunks into gatherable ordered rows."""
        # Merge local result chunks without changing Inferencer's output contract.
        local_arrays = [
            np.concatenate([
                batch[field_idx].detach().cpu().numpy()
                for batch in batches
            ], axis=0)
            for field_idx in range(len(batches[0]))
        ]

        # Split local arrays back into dataloader steps for global ordering.
        rows = []
        local_batch_size = self.config.evaluator.eval_batch_size // self.distributed.world_size
        for local_start in range(0, local_arrays[0].shape[0], local_batch_size):
            local_end = local_start + local_batch_size
            rows.append((
                local_start // local_batch_size,
                self.distributed.rank,
                local_arrays[0][local_start:local_end],
                local_arrays[1][local_start:local_end],
                local_arrays[2][local_start:local_end],
            ))
        return rows

    def _concat_rows(self, rows: list) -> list:
        """Concatenate globally ordered eval rows."""
        # Merge prediction, target, and metadata arrays independently.
        return [
            np.concatenate([row[2] for row in rows], axis=0),
            np.concatenate([row[3] for row in rows], axis=0),
            np.concatenate([row[4] for row in rows], axis=0),
        ]

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
