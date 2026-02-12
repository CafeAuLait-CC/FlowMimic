import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="output/latent_stats.txt")
    parser.add_argument("--num-passes", type=int, default=2)
    args = parser.parse_args()

    config = load_config()
    device = torch.device(args.device)
    seq_len = config["seq_len"]
    d_z = config["d_z"]
    mean, std = load_mean_std(config["stats_path"])

    aist_paths = [
        os.path.join(config["aist_motions_dir"], f"{n}.pkl")
        for n in _read_lines(config["aist_split_train"])
    ]
    mvh_dirs = _read_lines(config["mvh_split_train"])

    dataset_a = AISTDataset(
        config["aist_motions_dir"],
        genre_to_id={},
        seq_len=seq_len,
        mean=mean,
        std=std,
        files=aist_paths,
        cache_root=config["cache_root"],
        target_fps=config.get("target_fps", 30),
        src_fps=config.get("aist_fps", 60),
    )
    dataset_b = MVHumanNetDataset(
        config["mvhumannet_root"],
        seq_len=seq_len,
        mean=mean,
        std=std,
        sequence_dirs=mvh_dirs,
        cache_root=config["cache_root"],
        target_fps=config.get("target_fps", 30),
        src_fps=config.get("mvh_fps", 5),
    )

    loader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=False)
    loader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=False)

    vae_ckpt = args.vae_ckpt or config.get("vae_ckpt", "checkpoints/motion_vae_best.pt")
    vae = MotionVAE(
        d_in=config["d_in"],
        d_z=d_z,
        num_styles=config["num_styles"],
        max_len=seq_len,
    )
    state = torch.load(vae_ckpt, map_location=device)
    vae.load_state_dict(state["model"])
    vae.to(device)
    vae.eval()

    sum_z = 0.0
    sum_sq = 0.0
    count = 0
    sum_dim = np.zeros((d_z,), dtype=np.float64)
    sum_sq_dim = np.zeros((d_z,), dtype=np.float64)
    count_dim = 0

    def _accumulate(loader):
        nonlocal sum_z, sum_sq, count, sum_dim, sum_sq_dim, count_dim
        for pass_idx in range(args.num_passes):
            for batch in tqdm(loader, desc=f"Encoding pass {pass_idx + 1}"):
                motion = batch["motion"].to(device)
                domain_id = batch["domain_id"].to(device)
                style_id = batch["style_id"].to(device)
                with torch.no_grad():
                    _h, mu, _logvar = vae.encode(motion, vae.cond(domain_id, style_id))
                flat = mu.reshape(-1)
                sum_z += flat.sum().item()
                sum_sq += (flat ** 2).sum().item()
                count += flat.numel()
                mu_np = mu.detach().cpu().numpy()
                sum_dim += mu_np.sum(axis=(0, 1))
                sum_sq_dim += (mu_np ** 2).sum(axis=(0, 1))
                count_dim += mu_np.shape[0] * mu_np.shape[1]

    _accumulate(loader_a)
    _accumulate(loader_b)

    mean_z = sum_z / max(count, 1)
    var_z = sum_sq / max(count, 1) - mean_z ** 2
    std_z = np.sqrt(max(var_z, 1e-8))
    print(f"mean(z)={mean_z:.6f}, std(z)={std_z:.6f}")

    mean_dim = sum_dim / max(count_dim, 1)
    var_dim = sum_sq_dim / max(count_dim, 1) - mean_dim ** 2
    std_dim = np.sqrt(np.maximum(var_dim, 1e-8))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("mean_dim\n")
        f.write(" ".join(f"{v:.6f}" for v in mean_dim.tolist()) + "\n")
        f.write("std_dim\n")
        f.write(" ".join(f"{v:.6f}" for v in std_dim.tolist()) + "\n")
    print(f"Saved per-dim stats to {args.out}")


if __name__ == "__main__":
    main()
