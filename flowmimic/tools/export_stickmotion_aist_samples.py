#!/usr/bin/env python3
"""Export AIST StickMotion generations as paired SMPL22 xyz clips."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _collate_fn(batch):
    from mmcv.parallel import DataContainer

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
            raise NotImplementedError(
                f"Unsupported batch value for {key}: {type(batch[0][key])}"
            )
    return final_batch


def _patch_stickmotion_runtime() -> None:
    from mogen.models.attentions.semantics_modulated import SemanticsModulatedAttention
    from mogen.models.utils import gaussian_diffusion

    def sample_timesteps(self, batch_size, device):
        weights = self.weights()
        if not torch.is_tensor(weights):
            weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        else:
            weights = weights.to(device=device, dtype=torch.float32)
        probs = weights / weights.sum()
        indices = torch.multinomial(probs, batch_size, replacement=True).long()
        inv_probs = (1 / (len(probs) * probs[indices])).float()
        return indices, inv_probs

    def extract_into_tensor(arr, timesteps, broadcast_shape):
        if not torch.is_tensor(arr):
            arr = torch.from_numpy(arr).to(device=timesteps.device)
        else:
            arr = arr.to(device=timesteps.device)
        result = arr[timesteps.long()].float()
        while len(result.shape) < len(broadcast_shape):
            result = result[..., None]
        return result.expand(broadcast_shape)

    gaussian_diffusion.ScheduleSampler.sample = sample_timesteps
    gaussian_diffusion._extract_into_tensor = extract_into_tensor

    original_forward = SemanticsModulatedAttention.forward

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
        return original_forward(
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


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=torch.float32)
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _tensor_length(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _recover_smpl22(raw_motion: torch.Tensor) -> np.ndarray:
    from mogen.utils.plot_utils import recover_from_ric

    joints = recover_from_ric(raw_motion, joints_num=22, ifnorm=False)
    joints = joints.detach().cpu().numpy().astype(np.float32)
    if joints.shape[-2:] != (22, 3):
        raise ValueError(f"Unexpected recovered joint shape: {joints.shape}")
    return joints


def _convert_output_space(joints: np.ndarray, output_space: str) -> np.ndarray:
    if output_space == "yup":
        return joints.astype(np.float32)
    if output_space == "blender":
        return np.stack(
            [joints[..., 0], -joints[..., 2], joints[..., 1]], axis=-1
        ).astype(np.float32)
    raise ValueError(f"Unsupported output space: {output_space}")


def _safe_name(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")[:80] or "sample"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/aist/stickmotion_remodiffuse_aist_eval.py"
    )
    parser.add_argument(
        "--ckpt", default="runs/stickmotion/human_ml3d/aist_remodiffuse/last.ckpt"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument(
        "--output-dir", default="output/aist_baseline_samples_20260629/stickmotion"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-space", choices=("blender", "yup"), default="blender")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[2]
    stick_root = workspace / "stickmotion"
    config_path = (
        (workspace / args.config).resolve()
        if not Path(args.config).is_absolute()
        else Path(args.config)
    )
    ckpt_path = (
        (workspace / args.ckpt).resolve()
        if not Path(args.ckpt).is_absolute()
        else Path(args.ckpt)
    )
    output_dir = (
        (workspace / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(stick_root))
    os.chdir(stick_root)
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-codex")

    from lightning.pytorch import seed_everything
    from mmcv import Config
    from torch.utils.data import DataLoader

    from mogen.apis.lg_train import LgModel
    from mogen.datasets import build_dataset

    _patch_stickmotion_runtime()
    seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    cfg = Config.fromfile(str(config_path))
    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = args.workers
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    mean = torch.as_tensor(dataset.mean, dtype=torch.float32)
    std = torch.as_tensor(dataset.std, dtype=torch.float32)

    model = LgModel(cfg, dataset)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.model.others_cuda()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate_fn,
    )

    manifest_rows = []
    exported = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model.model(return_loss=False, **batch)
            for result in outputs:
                if exported >= args.max_samples:
                    break
                length = _tensor_length(result["motion_length"])
                pred_norm = result["pred_motion"][:length].float().cpu()
                ref_norm = result["motion"][:length].float().cpu()
                pred_raw = pred_norm * (std + 1e-9) + mean
                ref_raw = ref_norm * (std + 1e-9) + mean
                pred_joints = _convert_output_space(
                    _recover_smpl22(pred_raw), args.output_space
                )
                ref_joints = _convert_output_space(
                    _recover_smpl22(ref_raw), args.output_space
                )

                text = str(result.get("text", ""))
                base = f"stickmotion_sample{exported:02d}_{_safe_name(text)}"
                pred_path = output_dir / f"{base}_gen.npy"
                ref_path = output_dir / f"{base}_ref.npy"
                np.save(pred_path, pred_joints)
                np.save(ref_path, ref_joints)
                manifest_rows.append(
                    {
                        "index": exported,
                        "length": length,
                        "text": text,
                        "generated": pred_path.name,
                        "reference": ref_path.name,
                        "space": args.output_space,
                    }
                )
                exported += 1
            if exported >= args.max_samples:
                break

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "length", "text", "generated", "reference", "space"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_rows, indent=2), encoding="utf-8"
    )
    print(f"Exported {exported} StickMotion samples to {output_dir}")


if __name__ == "__main__":
    main()
