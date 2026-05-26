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
from flowmimic.src.model.vae.motion_vae import MotionVAE
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

    state = torch.load(args.checkpoint, map_location=device)
    ckpt_config = state.get("config", {})
    latent_len = args.latent_len
    if latent_len is None:
        latent_len = ckpt_config.get("latent_len")
    latent_token_mode = args.latent_token_mode or ckpt_config.get("latent_token_mode", "pool")
    model = MotionVAE(
        d_in=config["d_in"],
        d_z=config["d_z"],
        num_styles=config["num_styles"],
        max_len=seq_len,
        latent_len=latent_len,
        latent_token_mode=latent_token_mode,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    sum_vec = np.zeros(config["d_z"], dtype=np.float64)
    sum_sq = np.zeros(config["d_z"], dtype=np.float64)
    count = 0
    logvar_sum = np.zeros(config["d_z"], dtype=np.float64)

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            domain_id = batch["domain_id"].to(device)
            style_id = batch["style_id"].to(device)
            mask = batch["mask"].to(device)
            _h, mu, logvar = model.encode(
                motion, model.cond(domain_id, style_id), mask=mask
            )
            if mu.shape[1] == mask.shape[1]:
                valid_mu = mu[mask].detach().cpu().numpy().astype(np.float64)
                valid_logvar = logvar[mask].detach().cpu().numpy().astype(np.float64)
            else:
                valid_mu = mu.reshape(-1, mu.shape[-1]).detach().cpu().numpy().astype(np.float64)
                valid_logvar = logvar.reshape(-1, logvar.shape[-1]).detach().cpu().numpy().astype(np.float64)
            if valid_mu.size == 0:
                continue
            sum_vec += valid_mu.sum(axis=0)
            sum_sq += (valid_mu**2).sum(axis=0)
            logvar_sum += valid_logvar.sum(axis=0)
            count += valid_mu.shape[0]

    if count == 0:
        raise ValueError("No valid latent frames found")

    latent_mean = sum_vec / count
    latent_var = np.maximum(sum_sq / count - latent_mean**2, 1e-6)
    latent_std = np.sqrt(latent_var)
    posterior_sigma = np.exp(0.5 * (logvar_sum / count))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.savez(
        args.out_path,
        mean=latent_mean.astype(np.float32),
        std=latent_std.astype(np.float32),
        posterior_sigma_mean=posterior_sigma.astype(np.float32),
        count=np.array(count, dtype=np.int64),
    )
    print(f"Saved latent stats: {args.out_path}")
    print(
        "mu_abs_mean={:.6f} mu_std_mean={:.6f} mu_std_min={:.6f} "
        "mu_std_max={:.6f} posterior_sigma_mean={:.6f} count={}".format(
            float(np.mean(np.abs(latent_mean))),
            float(np.mean(latent_std)),
            float(np.min(latent_std)),
            float(np.max(latent_std)),
            float(np.mean(posterior_sigma)),
            count,
        )
    )


if __name__ == "__main__":
    main()
