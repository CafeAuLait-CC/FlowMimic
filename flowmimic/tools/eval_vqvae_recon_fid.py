#!/usr/bin/env python3
"""Evaluate a MotionVQVAE reconstruction ceiling with the AIST T2M encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flowmimic.scripts.eval_flow import _extract_t2m_features, _renorm_for_t2m
from flowmimic.src.config.config import load_config
from flowmimic.src.metrics import T2MMotionFeatureExtractor, summarize_motion_feature_metrics
from flowmimic.src.model.vae.backend import (
    decode_motion_latent,
    encode_motion_latent,
    load_vae_backend,
)
from flowmimic.src.model.vae.datasets.aist_filename_parser import get_genre_code
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id


CONT_END = 259


def _load_names(data_root: Path, split: str) -> list[str]:
    path = data_root / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--data-root", default="prepared/aist_mld_humanml3d")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    names = _load_names(data_root, args.split)
    vae_backend = load_vae_backend(
        args.vae_ckpt,
        cfg,
        device,
        seq_len=196,
        vae_type="motion_vqvae",
    )
    vae = vae_backend.model
    stats_path = args.stats_path or vae_backend.ckpt.get("stats_path")
    if not stats_path:
        raise ValueError("No motion stats path supplied or stored in the VQ-VAE checkpoint")
    stats = np.load(stats_path)
    flow_mean = np.asarray(stats["mean"], dtype=np.float32)
    flow_std = np.asarray(stats["std"], dtype=np.float32)
    mean_t = torch.from_numpy(flow_mean).to(device)
    std_t = torch.from_numpy(flow_std).to(device)

    t2m_mean_path = cfg["t2m_eval_mean_path"]
    t2m_std_path = cfg["t2m_eval_std_path"]
    t2m_mean = np.load(t2m_mean_path).astype(np.float32)
    t2m_std = np.load(t2m_std_path).astype(np.float32)
    extractor = T2MMotionFeatureExtractor(input_size=cfg["d_in"]).to(device)
    extractor.load_pretrained(cfg["t2m_motion_encoder_ckpt"])
    extractor.eval()

    genre_to_id = build_genre_to_id(cfg.get("aist_genres", []))
    ref_features = []
    recon_features = []
    mse_sum = 0.0
    mse_count = 0
    latent_shape = None

    motion_dir = data_root / "new_joint_vecs"
    for start in tqdm(range(0, len(names), args.batch_size), desc="VQ-VAE recon FID"):
        batch_names = names[start : start + args.batch_size]
        raw_np = np.stack(
            [np.load(motion_dir / f"{name}.npy").astype(np.float32) for name in batch_names]
        )
        if raw_np.shape[1:] != (196, 263):
            raise ValueError(f"Expected [B,196,263], got {raw_np.shape}")
        raw = torch.from_numpy(raw_np).to(device)
        motion = raw.clone()
        motion[..., :CONT_END] = (motion[..., :CONT_END] - mean_t) / (std_t + 1e-6)
        domain_id = torch.ones(len(batch_names), dtype=torch.long, device=device)
        style_id = torch.tensor(
            [genre_to_id.get(get_genre_code(name), 0) for name in batch_names],
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones(motion.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode():
            z_q = encode_motion_latent(
                vae, motion, domain_id, style_id, mask=mask
            )
            recon = decode_motion_latent(
                vae, z_q, domain_id, style_id, mask=mask, out_len=196
            )
        latent_shape = list(z_q.shape[1:])
        diff = recon[..., :CONT_END] - motion[..., :CONT_END]
        mse_sum += float(diff.square().sum().item())
        mse_count += diff.numel()

        ref_eval = _renorm_for_t2m(
            motion.detach().cpu().numpy(), flow_mean, flow_std, t2m_mean, t2m_std
        )
        recon_eval = _renorm_for_t2m(
            recon.detach().cpu().numpy(), flow_mean, flow_std, t2m_mean, t2m_std
        )
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
            "stats_path": str(stats_path),
            "t2m_motion_encoder_ckpt": cfg["t2m_motion_encoder_ckpt"],
            "latent_shape_per_sample": latent_shape,
        }
    )
    print(json.dumps(metrics, indent=2))
    if args.save_json:
        output = Path(args.save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
