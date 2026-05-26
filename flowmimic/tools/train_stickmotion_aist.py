"""Workspace-local StickMotion trainer for the prepared AIST++ dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_devices(gpu: str) -> list[int]:
    if gpu in {"cpu", "-1"}:
        return []
    return [int(part) for part in gpu.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aist/stickmotion_remodiffuse_aist.py")
    parser.add_argument("--version", default="aist_remodiffuse")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[2]
    stick_root = (workspace / "stickmotion").resolve()
    config_path = (workspace / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    sys.path.insert(0, str(stick_root))
    os.chdir(stick_root)

    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-codex")

    import torch
    from lightning import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.strategies import DDPStrategy
    from mmcv import Config
    from mmcv.parallel import DataContainer
    from torch.utils.data import DataLoader

    from mogen.apis.lg_train import LgModel
    from mogen.datasets import build_dataset
    from mogen.models.attentions.semantics_modulated import SemanticsModulatedAttention
    from mogen.models.utils import gaussian_diffusion

    def collate_fn(batch):
        keys = batch[0].keys()
        final_batch = {}
        for key in keys:
            if isinstance(batch[0][key], DataContainer):
                final_batch[key] = [item[key]._data for item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                final_batch[key] = torch.stack([item[key] for item in batch], 0)
            else:
                raise NotImplementedError(f"Unsupported batch value for {key}: {type(batch[0][key])}")
        return final_batch

    def sample_timesteps(self, batch_size, device):
        w = self.weights()
        if not torch.is_tensor(w):
            w = torch.as_tensor(w, dtype=torch.float32, device=device)
        else:
            w = w.to(device=device, dtype=torch.float32)
        p = w / w.sum()
        indices = torch.multinomial(p, batch_size, replacement=True).long()
        weights = (1 / (len(p) * p[indices])).float()
        return indices, weights

    def extract_into_tensor(arr, timesteps, broadcast_shape):
        if not torch.is_tensor(arr):
            arr = torch.from_numpy(arr).to(device=timesteps.device)
        else:
            arr = arr.to(device=timesteps.device)
        res = arr[timesteps.long()].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    gaussian_diffusion.ScheduleSampler.sample = sample_timesteps
    gaussian_diffusion._extract_into_tensor = extract_into_tensor

    original_semantics_forward = SemanticsModulatedAttention.forward

    def semantics_forward_compat(
        self,
        x,
        text_emb,
        stick_emb,
        other_emb,
        src_mask,
        cond_type,
        stick_mask,
        locus_emb,
        mid_query=None,
    ):
        return original_semantics_forward(
            self,
            x,
            text_emb,
            stick_emb,
            other_emb,
            src_mask,
            cond_type,
            stick_mask,
            locus_emb,
            mid_query,
        )

    SemanticsModulatedAttention.forward = semantics_forward_compat

    cfg = Config.fromfile(str(config_path))
    cfg.version = args.version
    if args.batch_size is not None:
        cfg.data.samples_per_gpu = args.batch_size
    if args.max_epochs is not None:
        cfg.runner.max_epochs = args.max_epochs

    dataset = build_dataset(cfg.data.train)
    if args.smoke:
        sample = dataset[0]
        print("dataset_len", len(dataset))
        print("sample_keys", sorted(sample.keys()))
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(key, tuple(value.shape), value.dtype)
        return

    devices = parse_devices(args.gpu)
    if not devices or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to start full StickMotion training on CPU.")

    logger = TensorBoardLogger(
        save_dir=str(workspace / "runs" / "stickmotion"),
        name="human_ml3d",
        version=cfg.version,
    )
    model = LgModel(cfg)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=True,
        num_workers=cfg.data.workers_per_gpu,
        collate_fn=collate_fn,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor="all_loss_epoch",
        mode="min",
        save_top_k=10,
        every_n_epochs=4,
        save_last=True,
    )
    trainer = Trainer(
        accelerator="gpu",
        strategy=DDPStrategy() if len(devices) > 1 else "auto",
        devices=devices,
        max_epochs=cfg.runner.max_epochs,
        max_steps=args.max_steps,
        precision="16-mixed",
        gradient_clip_algorithm="norm",
        gradient_clip_val=2,
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(model, train_loader, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
