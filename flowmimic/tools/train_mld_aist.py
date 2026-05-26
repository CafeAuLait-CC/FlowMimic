"""Workspace-local MLD launcher with Lightning compatibility fixes."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def pop_int_arg(name: str) -> int | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    try:
        value = int(sys.argv[index + 1])
    except (IndexError, ValueError) as exc:
        raise SystemExit(f"{name} requires an integer value") from exc
    del sys.argv[index:index + 2]
    return value


def pop_str_arg(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    try:
        value = sys.argv[index + 1]
    except IndexError as exc:
        raise SystemExit(f"{name} requires a value") from exc
    del sys.argv[index:index + 2]
    return value


def main() -> None:
    max_steps = pop_int_arg("--max-steps")
    resume_from_dir = pop_str_arg("--resume-from-dir")

    workspace = Path(__file__).resolve().parents[2]
    mld_root = (workspace / "motion-latent-diffusion").resolve()
    sys.path.insert(0, str(mld_root))
    os.chdir(mld_root)

    import torch
    import pytorch_lightning as pl

    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_compat

    original_trainer = pl.Trainer

    def trainer_compat(*args, **kwargs):
        if kwargs.get("strategy") is None:
            kwargs["strategy"] = "auto"
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        return original_trainer(*args, **kwargs)

    pl.Trainer = trainer_compat

    from mld import config as mld_config

    original_parse_args = mld_config.parse_args

    def parse_args_compat(*args, **kwargs):
        cfg = original_parse_args(*args, **kwargs)
        if resume_from_dir is not None:
            cfg.TRAIN.RESUME = resume_from_dir
        return cfg

    mld_config.parse_args = parse_args_compat

    from mld.models.modeltype import base as base_model

    def on_train_epoch_end(self):
        return self.allsplit_epoch_end("train", [])

    def on_validation_epoch_end(self):
        return self.allsplit_epoch_end("val", [])

    def on_test_epoch_end(self):
        return self.allsplit_epoch_end("test", [])

    base_model.BaseModel.on_train_epoch_end = on_train_epoch_end
    base_model.BaseModel.on_validation_epoch_end = on_validation_epoch_end
    base_model.BaseModel.on_test_epoch_end = on_test_epoch_end
    for hook_name in ("training_epoch_end", "validation_epoch_end", "test_epoch_end"):
        if hasattr(base_model.BaseModel, hook_name):
            delattr(base_model.BaseModel, hook_name)

    runpy.run_path(str(mld_root / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
