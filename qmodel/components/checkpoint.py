import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def capture_rng():
    # Capture all process-owned random generators.
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    # Restore generators after runtime construction.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class CheckpointSaver:
    def __init__(self, root_dir, config, device=None):
        # Keep the existing directory and evaluator interface.
        self.dir = Path(root_dir) / "ckpt"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.config = config
        self.ckpt = {}

    @staticmethod
    def latest(root_dir):
        # Read committed files; corrupt files must fail loudly.
        latest = None
        for path in (Path(root_dir) / "ckpt").glob("iter_*.pt"):
            state = torch.load(path, map_location="cpu", weights_only=False)
            for key in ("iteration", "model", "optimizer", "scheduler", "scaler", "config"):
                if key not in state:
                    raise ValueError(f"Incomplete checkpoint {path}: missing {key}")
            if "format_version" in state:
                for key in ("global_step", "progress"):
                    if key not in state:
                        raise ValueError(f"Incomplete checkpoint {path}: missing {key}")
                if state["format_version"] not in (1, 2):
                    raise ValueError(f"Unsupported checkpoint version: {path}")
                if state["format_version"] == 2:
                    for local in state["progress"]["ranks"]:
                        for key in ("rng", "sampler", "model", "statistics", "project"):
                            if key not in local:
                                raise ValueError(f"Incomplete checkpoint {path}: missing {key}")
            candidate = (state.get("global_step", state["iteration"]), path, state)
            if latest is None or candidate[0] > latest[0]:
                latest = candidate
        return latest

    def save(self, model, optim, sched, scaler, it, progress=None):
        # Materialize the complete snapshot before committing its filename.
        self.ckpt = {
            "iteration": it, "model": model.state_dict(),
            "optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(), "config": self.config,
        }
        if progress is not None:
            self.ckpt.update(format_version=2, global_step=it, progress=progress)
        path = self.dir / f"iter_{it}.pt"
        with tempfile.NamedTemporaryFile(dir=self.dir, suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            torch.save(self.ckpt, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(self.dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return path.absolute().as_posix()

    def load(self, path, model, optim, sched, scaler):
        # Preserve the existing evaluator and state-loading contract.
        self.ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for component, key in ((model, "model"), (optim, "optimizer"),
                               (sched, "scheduler"), (scaler, "scaler")):
            if component is not None:
                component.load_state_dict(self.ckpt[key])
        return self.ckpt["iteration"]
