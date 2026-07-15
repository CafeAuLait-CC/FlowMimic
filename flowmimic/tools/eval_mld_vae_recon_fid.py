#!/usr/bin/env python3
"""Evaluate the AIST-trained MLD VAE reconstruction distribution on 470 clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "motion-latent-diffusion"))

from flowmimic.scripts.eval_flow import _extract_t2m_features
from flowmimic.src.config.config import load_config
from flowmimic.src.metrics import T2MMotionFeatureExtractor, summarize_motion_feature_metrics
from mld.models.architectures.mld_vae import MldVae


def _load_model(path: str, device: torch.device) -> MldVae:
    model = MldVae(
        ablation=SimpleNamespace(MLP_DIST=False, PE_TYPE="mld"),
        nfeats=263,
        latent_dim=[1, 256],
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        arch="encoder_decoder",
        position_embedding="learned",
    ).to(device)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("vae."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("vae.")
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--data-root", default="prepared/aist_mld_humanml3d")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data_root = Path(args.data_root)
    names = [
        line.strip()
        for line in (data_root / f"{args.split}.txt").read_text().splitlines()
        if line.strip()
    ]
    mld_mean = np.load(data_root / "Mean.npy").astype(np.float32)
    mld_std = np.load(data_root / "Std.npy").astype(np.float32)
    t2m_mean = np.load(cfg["t2m_eval_mean_path"]).astype(np.float32)
    t2m_std = np.load(cfg["t2m_eval_std_path"]).astype(np.float32)
    mld_mean_t = torch.from_numpy(mld_mean).to(device)
    mld_std_t = torch.from_numpy(mld_std).to(device)

    model = _load_model(args.vae_ckpt, device)
    extractor = T2MMotionFeatureExtractor(input_size=cfg["d_in"]).to(device)
    extractor.load_pretrained(cfg["t2m_motion_encoder_ckpt"])
    extractor.eval()

    ref_features = []
    recon_features = []
    mse_sum = 0.0
    mse_count = 0
    motion_dir = data_root / "new_joint_vecs"
    for start in tqdm(range(0, len(names), args.batch_size), desc="MLD VAE recon FID"):
        batch_names = names[start : start + args.batch_size]
        raw_np = np.stack(
            [np.load(motion_dir / f"{name}.npy").astype(np.float32) for name in batch_names]
        )
        raw = torch.from_numpy(raw_np).to(device)
        norm = (raw - mld_mean_t) / (mld_std_t + 1e-6)
        lengths_list = [196] * len(batch_names)
        with torch.inference_mode():
            z, dist = model.encode(norm, lengths_list)
            latent = dist.loc if args.latent_mode == "mean" else z
            recon_norm = model.decode(latent, lengths_list)
        diff = recon_norm[..., :259] - norm[..., :259]
        mse_sum += float(diff.square().sum().item())
        mse_count += diff.numel()

        recon_raw = recon_norm.detach().cpu().numpy() * mld_std + mld_mean
        ref_eval = (raw_np - t2m_mean) / (t2m_std + 1e-6)
        recon_eval = (recon_raw - t2m_mean) / (t2m_std + 1e-6)
        lengths = torch.full((len(batch_names),), 196, dtype=torch.long, device=device)
        ref_features.append(_extract_t2m_features(extractor, ref_eval, lengths, device))
        recon_features.append(
            _extract_t2m_features(extractor, recon_eval, lengths, device)
        )

    ref_features_np = np.concatenate(ref_features, axis=0)
    recon_features_np = np.concatenate(recon_features, axis=0)
    np.random.seed(int(cfg.get("seed", 42)))
    metrics = summarize_motion_feature_metrics(
        recon_features_np,
        ref_features_np,
        diversity_times=300,
    )
    metrics.update(
        {
            "normalized_continuous_mse": mse_sum / max(mse_count, 1),
            "samples": len(names),
            "split": args.split,
            "clip": "first_196",
            "vae_ckpt": args.vae_ckpt,
            "t2m_motion_encoder_ckpt": cfg["t2m_motion_encoder_ckpt"],
            "latent_shape_per_sample": [1, 256],
            "latent_mode": args.latent_mode,
            "seed": args.seed,
        }
    )
    print(json.dumps(metrics, indent=2))
    if args.save_json:
        output = Path(args.save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
