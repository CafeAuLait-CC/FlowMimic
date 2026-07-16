#!/usr/bin/env python3
"""Generate one AIST++ text-conditioned motion with a trained MLD model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def _zup(joints: np.ndarray) -> np.ndarray:
    return np.stack(
        [joints[..., 0], -joints[..., 2], joints[..., 1]], axis=-1
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--length", type=int, default=196)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--mld-root", default="motion-latent-diffusion")
    parser.add_argument("--cfg", default="configs/aist/mld_eval_aist.yaml")
    parser.add_argument("--cfg-assets", default="configs/aist/mld_assets_aist.yaml")
    parser.add_argument(
        "--ckpt",
        default="runs/mld/mld/aist_ik263_mld_196/checkpoints/epoch=2999.ckpt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[2]
    mld_root = (workspace / args.mld_root).resolve()
    cfg_path = (workspace / args.cfg).resolve()
    assets_path = (workspace / args.cfg_assets).resolve()
    ckpt_path = (workspace / args.ckpt).resolve()
    output_path = (workspace / args.output).resolve()
    meta_path = (workspace / args.meta).resolve() if args.meta else output_path.with_suffix(".json")

    sys.path.insert(0, str(mld_root))
    old_cwd = Path.cwd()
    os.chdir(mld_root)
    try:
        from mld.config import get_module_config
        from mld.data.get_data import get_datasets
        from mld.models.get_model import get_model

        cfg_base = OmegaConf.load(mld_root / "configs" / "base.yaml")
        cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
        cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
        cfg = OmegaConf.merge(cfg_exp, cfg_model, OmegaConf.load(assets_path))
        cfg.TEST.CHECKPOINTS = str(ckpt_path)
        cfg.ACCELERATOR = "gpu" if args.device.startswith("cuda") else "cpu"
        cfg.DEVICE = [0]
        cfg.DEBUG = False

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        device = torch.device(args.device)
        dataset = get_datasets(cfg, phase="test")[0]
        model = get_model(cfg, dataset)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.sample_mean = cfg.TEST.MEAN
        model.fact = cfg.TEST.FACT
        model.to(device)
        model.eval()

        with torch.no_grad():
            generated = model({"length": [args.length], "text": [args.text]})
        if isinstance(generated, (list, tuple)):
            if not generated:
                raise ValueError("MLD returned no generated motions")
            joints = generated[0]
        elif torch.is_tensor(generated) or isinstance(generated, np.ndarray):
            joints = generated[0]
        else:
            raise TypeError(f"Unexpected MLD output type: {type(generated).__name__}")
        if torch.is_tensor(joints):
            joints = joints.detach().cpu().numpy()
        joints = np.asarray(joints[: args.length], dtype=np.float32)
        if joints.shape != (args.length, 22, 3):
            raise ValueError(f"Unexpected MLD output shape: {joints.shape}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, _zup(joints))
        metadata = {
            "method": "mld",
            "sample_id": args.sample_id,
            "text": args.text,
            "length": args.length,
            "seed": args.seed,
            "checkpoint": str(ckpt_path),
            "config": str(cfg_path),
            "output": str(output_path),
            "space": "blender_zup",
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
