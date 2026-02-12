import argparse
import os
import random
import sys

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.cond_api import build_cond_inputs, build_dummy_cond
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.motion.process_motion import ik263_to_smpl22
from flowmimic.src.data.openpose import load_aist_openpose, load_mvh_openpose
from flowmimic.src.model.vae.stats import load_mean_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--solver", type=str, default="heun")
    parser.add_argument("--style-id", type=int, default=0)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--k2d-npy", type=str, default=None)
    parser.add_argument("--tau-cond-npy", type=str, default=None)
    parser.add_argument("--sample-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["auto", "aist", "mvh"], default="auto")
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="result_smpl22.npy")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--src-fps", type=int, default=None)
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="output/flow")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    config = load_config()
    seq_len = config["seq_len"]
    d_z = config["d_z"]
    openpose_stats_path = config.get("openpose_stats_path", "data/openpose_stats.npz")
    target_fps = args.target_fps or config.get("target_fps", None)
    vae_ckpt_path = args.vae_checkpoint or config.get(
        "vae_ckpt", "checkpoints/motion_vae_best.pt"
    )
    stats_path = config.get("stats_path", "data/mean_std_263_train.npz")
    latent_stats_path = config.get("latent_stats_path", "data/latent_stats.npz")
    cond_frames_min = config.get("flow", {}).get("cond_frames_min", 2)
    cond_frames_max = config.get("flow", {}).get("cond_frames_max", 10)
    cond_cache_root = config.get("cond_cache_root", "data/cached_cond")
    aist_cameras = config.get("aist_cameras", ["01", "02", "08", "09"])
    mvh_cameras = config.get("mvh_cameras", ["22327091", "22327113", "22327084"])
    aist_openpose_dir = config.get(
        "aist_openpose_dir", "data/AIST++/Annotations/openpose"
    )
    mvh_openpose_root = config.get("mvh_openpose_root", "data/MVHumanNet")
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    aist_split_val = config.get("aist_split_val")
    mvh_split_val = config.get("mvh_split_val")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

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
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    if args.use_ema and "ema" in state:
        flow.load_state_dict(state["ema"])
    else:
        flow.load_state_dict(state["model"])
    flow.to(device)
    flow.eval()

    vae = MotionVAE(d_in=config["d_in"], d_z=d_z, num_styles=config["num_styles"], max_len=seq_len)
    vae_state = torch.load(vae_ckpt_path, map_location=device)
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

    meta = {}
    k2d = None
    vis = None
    if args.k2d_npy:
        k2d = np.load(args.k2d_npy)
        vis = None
        if k2d.ndim == 3 and k2d.shape[-1] == 3:
            vis = k2d[..., 2] > 0.0
            k2d = k2d[..., :2]
    elif args.sample_path or args.dataset in ("aist", "mvh", "auto"):
        dataset = args.dataset
        if dataset == "auto":
            dataset = "aist" if random.random() < 0.5 else "mvh"
        if dataset == "aist":
            if args.sample_path:
                pkl_path = args.sample_path
            else:
                if not aist_split_val:
                    raise ValueError("aist_split_val is required for random sampling")
                with open(aist_split_val, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f if line.strip()]
                name = random.choice(names)
                pkl_path = os.path.join(aist_dir, f"{name}.pkl")
            cam = args.camera or random.choice(aist_cameras)
            k2d, vis = load_aist_openpose(
                pkl_path,
                aist_openpose_dir,
                src_fps=args.src_fps,
                target_fps=target_fps,
                cache_root=cond_cache_root,
                camera=cam,
            )
            meta = {"dataset": "aist", "path": pkl_path, "camera": cam}
        else:
            if args.sample_path:
                seq_dir = args.sample_path
            else:
                if not mvh_split_val:
                    raise ValueError("mvh_split_val is required for random sampling")
                with open(mvh_split_val, "r", encoding="utf-8") as f:
                    seqs = [line.strip() for line in f if line.strip()]
                seq_dir = random.choice(seqs)
            cam = args.camera or random.choice(mvh_cameras)
            k2d, vis = load_mvh_openpose(
                seq_dir,
                mv_root,
                mvh_openpose_root,
                mvh_cameras,
                src_fps=args.src_fps,
                target_fps=target_fps,
                cache_root=cond_cache_root,
                camera=cam,
            )
            meta = {"dataset": "mvh", "path": seq_dir, "camera": cam}

    if k2d is None:
        cond = build_dummy_cond(1, device=device)
        tau_cond = cond["tau_cond"].squeeze(0).cpu().numpy()
        sample_idx = list(range(tau_cond.shape[0]))
    else:
        orig_len = k2d.shape[0]
        start = 0
        if orig_len >= seq_len:
            start = random.randint(0, orig_len - seq_len)
            k2d = k2d[start : start + seq_len]
            vis = vis[start : start + seq_len] if vis is not None else None
        else:
            pad_len = seq_len - orig_len
            k2d = np.concatenate(
                [k2d, np.zeros((pad_len, 25, 2), dtype=np.float32)], axis=0
            )
            if vis is not None:
                vis = np.concatenate(
                    [vis, np.zeros((pad_len, 25), dtype=np.float32)], axis=0
                )
        meta["orig_len"] = orig_len
        meta["start"] = start
        t_len = k2d.shape[0]
        k_frames = int(np.random.randint(cond_frames_min, cond_frames_max + 1))
        if t_len <= k_frames:
            sample_idx = np.arange(t_len)
        else:
            sample_idx = np.sort(np.random.choice(t_len, size=k_frames, replace=False))
        tau_cond = sample_idx.astype(np.float32) / max(t_len - 1, 1)
        k2d_sparse = k2d[sample_idx]
        vis_sparse = vis[sample_idx] if vis is not None else None
        cond = build_cond_inputs(k2d_sparse, tau_cond, device, vis_mask=vis_sparse)

    tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
    x0 = torch.randn(1, seq_len, d_z, device=device)

    g, mem, _vis = flow.cond_encoder(
        cond["k2d"],
        cond["tau_cond"],
        vis_mask=cond.get("vis_mask"),
        mean=k2d_mean,
        std=k2d_std,
    )
    style_id = torch.tensor([args.style_id], device=device)
    domain_id = torch.tensor([args.domain_id], device=device)
    style = flow.style_emb(style_id, domain_id, apply_dropout=False)
    g = flow.cond_mlp(torch.cat([g, style], dim=-1))

    cond_batch = {"tau_out": tau_out, "mem": mem, "g": g}
    z_hat = solve_flow(flow.flow, x0, cond_batch, num_steps=args.steps, method=args.solver)
    if latent_mean is not None and latent_std is not None:
        z_hat = z_hat * (latent_std + 1e-6) + latent_mean

    with torch.no_grad():
        x_hat = vae.decode(z_hat, vae.cond(domain_id, style_id))
    ik263 = x_hat.squeeze(0).cpu().numpy()
    mean, std = load_mean_std(stats_path)
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    ik263[:, :cont_end] = ik263[:, :cont_end] * std + mean
    joints = ik263_to_smpl22(ik263)

    os.makedirs(args.out_dir, exist_ok=True)
    out_npy = os.path.join(args.out_dir, args.out)
    np.save(out_npy, joints)

    meta_path = os.path.join(args.out_dir, "result_meta.json")
    meta_out = {
        "dataset": meta.get("dataset", "unknown"),
        "path": meta.get("path", ""),
        "camera": meta.get("camera", ""),
        "orig_len": meta.get("orig_len", ""),
        "start": meta.get("start", ""),
        "seq_len": seq_len,
        "sparse_indices": sample_idx.tolist(),
        "tau_cond": tau_cond.tolist(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        import json

        json.dump(meta_out, f, indent=2)


if __name__ == "__main__":
    main()
