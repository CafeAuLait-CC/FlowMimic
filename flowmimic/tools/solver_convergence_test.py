"""Solver convergence test for flow models.

Example:
  python flowmimic/tools/solver_convergence_test.py \
    --flow-ckpt checkpoints/flow/reflow_0/flow_round0_last.pt \
    --vae-ckpt checkpoints/vae/len200/motion_vae_best.pt \
    --max-step 800 \
    --device cuda
"""

import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
    category=UserWarning,
)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_datasets(config):
    aist_paths = []
    if os.path.exists(config.get("aist_split_train", "")):
        with open(config["aist_split_train"], "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        aist_paths = [
            os.path.join(config["aist_motions_dir"], f"{n}.pkl") for n in names
        ]

    mvh_dirs = []
    if os.path.exists(config.get("mvh_split_train", "")):
        with open(config["mvh_split_train"], "r", encoding="utf-8") as f:
            mvh_dirs = [line.strip() for line in f if line.strip()]

    mean, std = load_mean_std(config["stats_path"])
    seq_len = config["seq_len"]
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)
    cond_cfg = config.get("flow", {})

    datasets = []
    if aist_paths:
        datasets.append(
            AISTDataset(
                config["aist_motions_dir"],
                genre_to_id={},
                seq_len=seq_len,
                mean=mean,
                std=std,
                files=aist_paths,
                cache_root=config["cache_root"],
                target_fps=target_fps,
                src_fps=aist_fps,
                camera_ids=config.get("aist_cameras", ["01", "02", "08", "09"]),
                expand_cameras=True,
                include_cond=True,
                openpose_dir=config.get(
                    "aist_openpose_dir", "data/AIST++/Annotations/openpose"
                ),
                cond_cache_root=config.get("cond_cache_root", "data/cached_cond"),
                cond_frames_min=cond_cfg.get("cond_frames_min", 7),
                cond_frames_max=cond_cfg.get("cond_frames_max", 7),
                cond_drop_prob=cond_cfg.get("cond_drop_prob", 0.0),
            )
        )
    if mvh_dirs:
        datasets.append(
            MVHumanNetDataset(
                config["mvhumannet_root"],
                seq_len=seq_len,
                mean=mean,
                std=std,
                sequence_dirs=mvh_dirs,
                cache_root=config["cache_root"],
                target_fps=target_fps,
                src_fps=mvh_fps,
                camera_ids=config.get(
                    "mvh_cameras", ["22327091", "22327113", "22327084"]
                ),
                expand_cameras=True,
                include_cond=True,
                openpose_root=config.get("mvh_openpose_root", "data/MVHumanNet"),
                cond_cache_root=config.get("cond_cache_root", "data/cached_cond"),
                cond_frames_min=cond_cfg.get("cond_frames_min", 7),
                cond_frames_max=cond_cfg.get("cond_frames_max", 7),
                cond_drop_prob=cond_cfg.get("cond_drop_prob", 0.0),
            )
        )
    return datasets


def _stack_batch(samples):
    def _stack(name):
        return torch.stack([s[name] for s in samples], dim=0)

    batch = {
        "style_id": _stack("style_id"),
        "domain_id": _stack("domain_id"),
        "k2d": _stack("k2d"),
        "vis": _stack("vis"),
        "tau_cond": _stack("tau_cond"),
        "mask_cond": _stack("mask_cond"),
    }
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--solver", type=str, default="heun")
    parser.add_argument("--max-step", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="output/flow/solver_convergence")
    args = parser.parse_args()

    _seed_all(args.seed)
    config = load_config()
    device = torch.device(args.device)
    seq_len = config["seq_len"]
    d_z = config["d_z"]
    latent_stats_path = config.get("latent_stats_path", "data/latent_stats.npz")
    openpose_stats_path = config.get("openpose_stats_path", "data/openpose_stats.npz")
    stats_path = config.get("stats_path", "data/mean_std_263_train.npz")

    datasets = _build_datasets(config)
    if not datasets:
        raise ValueError("No datasets available for sampling")

    flow_cfg = config.get("flow", {})
    flow = ConditionalRectFlow(
        d_z=d_z,
        d_model=flow_cfg.get("d_model", 512),
        n_layers=flow_cfg.get("n_layers", 8),
        n_heads=flow_cfg.get("n_heads", 8),
        ffn_dim=flow_cfg.get("ffn_dim", 2048),
        dropout=flow_cfg.get("dropout", 0.1),
        num_styles=config["num_styles"],
        style_dim=flow_cfg.get("style_dim", 32),
        cond_dim=flow_cfg.get("cond_dim", 256),
        cond_layers=flow_cfg.get("cond_layers", 4),
        cond_heads=flow_cfg.get("cond_heads", 4),
        p_style_drop=flow_cfg.get("p_style_drop", 0.5),
    )
    state = torch.load(args.flow_ckpt, map_location=device)
    flow.load_state_dict(state.get("ema", state.get("model", state)))
    flow.to(device)
    flow.eval()

    vae_ckpt = args.vae_ckpt or config.get(
        "vae_ckpt", "checkpoints/vae/len200/motion_vae_best.pt"
    )
    vae = MotionVAE(
        d_in=config["d_in"],
        d_z=d_z,
        num_styles=config["num_styles"],
        max_len=seq_len,
    )
    vae_state = torch.load(vae_ckpt, map_location=device)
    vae.load_state_dict(vae_state["model"])
    vae.to(device)
    vae.eval()

    k2d_mean = None
    k2d_std = None
    if os.path.exists(openpose_stats_path):
        stats = np.load(openpose_stats_path)
        k2d_mean = stats["mean"]
        k2d_std = stats["std"]

    latent_mean = None
    latent_std = None
    if os.path.exists(latent_stats_path):
        stats = np.load(latent_stats_path)
        latent_mean = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
        latent_std = torch.tensor(stats["std"], device=device, dtype=torch.float32)

    mean_263, std_263 = load_mean_std(stats_path)
    mean_263 = torch.tensor(mean_263, device=device, dtype=torch.float32)
    std_263 = torch.tensor(std_263, device=device, dtype=torch.float32)
    cont_end = LAYOUT_SLICES["feet_contact"][0]

    samples = []
    for _ in range(args.num_samples):
        dataset = random.choice(datasets)
        samples.append(dataset[random.randint(0, len(dataset) - 1)])

    batch = _stack_batch(samples)
    k2d = batch["k2d"].to(device)
    vis = batch["vis"].to(device)
    tau_cond = batch["tau_cond"].to(device)
    mask_cond = batch["mask_cond"].to(device)
    style_id = batch["style_id"].to(device)
    domain_id = batch["domain_id"].to(device)

    with torch.inference_mode():
        g2d, mem, _vis = flow.cond_encoder(
            k2d,
            tau_cond,
            vis_mask=vis,
            mask_cond=mask_cond,
            mean=k2d_mean,
            std=k2d_std,
        )
        style = flow.style_emb(style_id, domain_id, apply_dropout=False)
        g = flow.cond_mlp(torch.cat([g2d, style], dim=-1))
        tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
        cond_batch = {
            "tau_out": tau_out,
            "mem": mem,
            "g": g,
            "mem_mask": ~mask_cond,
        }

        x0 = torch.randn((k2d.shape[0], seq_len, d_z), device=device)
        base_steps = [1, 2, 4, 8, 16, 32]
        steps_list = [s for s in base_steps if s <= args.max_step]
        if args.max_step > 32:
            extra = list(range(50, args.max_step + 1, 50))
            steps_list.extend(extra)
        steps_list = sorted(set(steps_list))
        x_ref = solve_flow(
            flow.flow, x0, cond_batch, num_steps=args.max_step, method=args.solver
        )
        if latent_mean is not None and latent_std is not None:
            x_ref_un = x_ref * (latent_std + 1e-6) + latent_mean
        else:
            x_ref_un = x_ref
        y_ref = vae.decode(x_ref_un, vae.cond(domain_id, style_id))
        y_ref = y_ref.clone()
        y_ref[..., :cont_end] = y_ref[..., :cont_end] * std_263 + mean_263

        latent_errors = []
        motion_errors = []
        for steps in steps_list:
            x_hat = solve_flow(
                flow.flow, x0, cond_batch, num_steps=steps, method=args.solver
            )
            diff = x_hat - x_ref
            lat_err = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1).mean()
            latent_errors.append(lat_err.item())

            if latent_mean is not None and latent_std is not None:
                x_hat_un = x_hat * (latent_std + 1e-6) + latent_mean
            else:
                x_hat_un = x_hat
            y_hat = vae.decode(x_hat_un, vae.cond(domain_id, style_id))
            y_hat = y_hat.clone()
            y_hat[..., :cont_end] = y_hat[..., :cont_end] * std_263 + mean_263
            diff_y = y_hat - y_ref
            mot_err = torch.linalg.norm(
                diff_y.reshape(diff_y.shape[0], -1), dim=1
            ).mean()
            motion_errors.append(mot_err.item())
            print(
                f"steps={steps:>3d} latent_err={lat_err.item():.6f} motion_err={mot_err.item():.6f}"
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(steps_list, latent_errors, marker="o")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Latent error (L2)")
    ax1.grid(True, alpha=0.3)
    latent_path = os.path.join(args.out_dir, "solver_latent_error.png")
    fig1.savefig(latent_path, dpi=200, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(steps_list, motion_errors, marker="o")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("263D error (L2)")
    ax2.grid(True, alpha=0.3)
    motion_path = os.path.join(args.out_dir, "solver_263_error.png")
    fig2.savefig(motion_path, dpi=200, bbox_inches="tight")

    print(f"Saved: {latent_path}")
    print(f"Saved: {motion_path}")


if __name__ == "__main__":
    main()
