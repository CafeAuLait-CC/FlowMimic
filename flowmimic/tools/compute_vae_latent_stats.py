import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id
from flowmimic.src.model.vae.backend import (
    VAE_TYPE_VQ,
    encode_motion_latent,
    load_vae_backend,
)
from flowmimic.src.model.vae.stats import load_mean_std


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _aist_split_paths(aist_dir, split_path):
    return [os.path.join(aist_dir, f"{name}.pkl") for name in _read_lines(split_path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument(
        "--vae-type",
        choices=("auto", "motion_vae", "motion_vqvae"),
        default="auto",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--aist-crop-mode", choices=("first", "random", "uniform"), default="first"
    )
    parser.add_argument("--aist-clip-repeat", type=int, default=1)
    parser.add_argument("--latent-len", type=int, default=None)
    parser.add_argument(
        "--latent-token-mode",
        choices=("pool", "query"),
        default=None,
    )
    parser.add_argument("--genre-map", type=str, default="flowmimic/src/config/genre_to_id.json")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    config = load_config()
    device = torch.device(args.device)
    seq_len = args.seq_len or config["seq_len"]
    stats_path = args.stats_path or config["stats_path"]
    mean, std = load_mean_std(stats_path)

    if os.path.exists(args.genre_map):
        with open(args.genre_map, "r", encoding="utf-8") as f:
            genre_to_id = json.load(f)
    else:
        genre_to_id = build_genre_to_id(config.get("aist_genres", []))

    split_key = {
        "train": "aist_split_train",
        "val": "aist_split_val",
        "test": "aist_split_test",
    }[args.split]
    split_path = config.get(split_key)
    if split_path is None:
        split_path = f"data/AIST++/Annotations/splits/pose_{args.split}.txt"

    files = _aist_split_paths(config["aist_motions_dir"], split_path)
    dataset = AISTDataset(
        config["aist_motions_dir"],
        genre_to_id,
        seq_len,
        mean=mean,
        std=std,
        files=files,
        cache_root=config["cache_root"],
        target_fps=config.get("target_fps", 30),
        src_fps=config.get("aist_fps", 60),
        crop_mode=args.aist_crop_mode,
        clip_repeat=args.aist_clip_repeat,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    loaded = load_vae_backend(
        args.checkpoint,
        config,
        device,
        seq_len=seq_len,
        vae_type=args.vae_type,
        latent_len=args.latent_len,
        latent_token_mode=args.latent_token_mode,
    )
    model = loaded.model

    sum_vec = None
    sum_sq = None
    count = 0
    logvar_sum = None
    logvar_count = 0

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            domain_id = batch["domain_id"].to(device)
            style_id = batch["style_id"].to(device)
            mask = batch["mask"].to(device)
            if loaded.vae_type == VAE_TYPE_VQ:
                z = encode_motion_latent(
                    model, motion, domain_id, style_id, mask=mask
                )
                values = z.detach().cpu().numpy().astype(np.float64)
                batch_count = values.shape[0]
            else:
                _h, mu, logvar = model.encode(
                    motion, model.cond(domain_id, style_id), mask=mask
                )
                if mu.shape[1] == mask.shape[1]:
                    values = mu[mask].detach().cpu().numpy().astype(np.float64)
                    valid_logvar = logvar[mask].detach().cpu().numpy().astype(np.float64)
                else:
                    values = (
                        mu.reshape(-1, mu.shape[-1])
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    valid_logvar = (
                        logvar.reshape(-1, logvar.shape[-1])
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                batch_count = values.shape[0]
                if logvar_sum is None:
                    logvar_sum = np.zeros(values.shape[1:], dtype=np.float64)
                logvar_sum += valid_logvar.sum(axis=0)
                logvar_count += valid_logvar.shape[0]
            if values.size == 0:
                continue
            if sum_vec is None:
                sum_vec = np.zeros(values.shape[1:], dtype=np.float64)
                sum_sq = np.zeros(values.shape[1:], dtype=np.float64)
            sum_vec += values.sum(axis=0)
            sum_sq += (values**2).sum(axis=0)
            count += batch_count

    if count == 0 or sum_vec is None or sum_sq is None:
        raise ValueError("No valid latent frames found")

    latent_mean = sum_vec / count
    latent_var = np.maximum(sum_sq / count - latent_mean**2, 1e-6)
    latent_std = np.sqrt(latent_var)
    posterior_sigma = None
    if logvar_sum is not None and logvar_count > 0:
        posterior_sigma = np.exp(0.5 * (logvar_sum / logvar_count))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    payload = {
        "mean": latent_mean.astype(np.float32),
        "std": latent_std.astype(np.float32),
        "count": np.array(count, dtype=np.int64),
        "vae_type": np.array(loaded.vae_type),
        "latent_len": np.array(loaded.latent_len, dtype=np.int64),
    }
    if posterior_sigma is not None:
        payload["posterior_sigma_mean"] = posterior_sigma.astype(np.float32)
    np.savez(args.out_path, **payload)
    print(f"Saved latent stats: {args.out_path}")
    posterior_text = (
        f" posterior_sigma_mean={float(np.mean(posterior_sigma)):.6f}"
        if posterior_sigma is not None
        else ""
    )
    print(
        "vae_type={} latent_shape={} mean_abs_mean={:.6f} std_mean={:.6f} "
        "std_min={:.6f} std_max={:.6f}{} count={}".format(
            loaded.vae_type,
            tuple(latent_mean.shape),
            float(np.mean(np.abs(latent_mean))),
            float(np.mean(latent_std)),
            float(np.min(latent_std)),
            float(np.max(latent_std)),
            posterior_text,
            count,
        )
    )


if __name__ == "__main__":
    main()
