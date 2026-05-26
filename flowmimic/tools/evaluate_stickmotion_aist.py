"""Evaluate the workspace StickMotion AIST checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def parse_devices(gpu: str) -> list[int]:
    return [int(part) for part in gpu.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aist/stickmotion_remodiffuse_aist_eval.py")
    parser.add_argument("--ckpt", default="runs/stickmotion/human_ml3d/aist_remodiffuse/last.ckpt")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[2]
    stick_root = (workspace / "stickmotion").resolve()
    config_path = (workspace / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    ckpt_path = (workspace / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt)

    sys.path.insert(0, str(stick_root))
    os.chdir(stick_root)
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-codex")

    import torch
    from lightning import Trainer
    from lightning.fabric.fabric import Fabric
    from lightning.pytorch import seed_everything
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
                data = [item[key]._data for item in batch]
                final_batch[key] = data
                if key == "motion_metas" and data and isinstance(data[0], dict):
                    for meta_key in ("text", "token"):
                        if meta_key in data[0]:
                            final_batch[meta_key] = [item[meta_key] for item in data]
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

    def validation_epoch_end_aist(self) -> None:
        import os as _os
        import pickle as _pickle

        self.outputs = [item for batch_output in self.outputs for item in batch_output]
        tmp_file = f"/dev/shm/{self.unit}_{self.global_rank}.pkl"
        _pickle.dump(self.outputs, open(tmp_file, "wb"))
        self.trainer.strategy.barrier()
        part_list = []
        if self.global_rank == 0:
            for rank in range(self.trainer.num_devices):
                tmp_file = f"/dev/shm/{self.unit}_{rank}.pkl"
                outputs = _pickle.load(open(tmp_file, "rb"))
                _os.remove(tmp_file)
                part_list.append(outputs)
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            ordered_results = ordered_results[: len(self.dataset)]
            results = self.dataset.evaluate(ordered_results)
            for key, value in results.items():
                print(f"\n{key} : {value:.4f}")

    LgModel.on_validation_epoch_end = validation_epoch_end_aist

    seed_everything(123, workers=True)
    cfg = Config.fromfile(str(config_path))
    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    dataset.prepare_evaluation()

    if Fabric().global_rank == 0:
        pid_seed = os.getpid()
    else:
        pid_seed = os.getppid()
    model = LgModel(cfg, dataset, unit=hashlib.md5(str(pid_seed).encode()).hexdigest()[:8])
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=False,
        num_workers=cfg.data.workers_per_gpu,
        collate_fn=collate_fn,
    )
    devices = parse_devices(args.gpu)
    trainer = Trainer(
        accelerator="gpu",
        strategy=DDPStrategy() if len(devices) > 1 else "auto",
        devices=devices,
        logger=False,
        precision="16-mixed",
        inference_mode=False,
    )
    trainer.validate(model, loader, ckpt_path=str(ckpt_path))


if __name__ == "__main__":
    main()
