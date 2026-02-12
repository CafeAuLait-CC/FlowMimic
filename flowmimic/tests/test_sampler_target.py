import argparse
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.data.openpose import load_aist_openpose, load_mvh_openpose
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std


def _load_cond_for_meta(meta, config, target_fps, aist_fps, mvh_fps):
    seq_len = config["seq_len"]
    cond_frames_min = config.get("flow", {}).get("cond_frames_min", 2)
    cond_frames_max = config.get("flow", {}).get("cond_frames_max", 10)
    cond_drop_prob = config.get("flow", {}).get("cond_drop_prob", 0.2)
    aist_openpose_dir = config.get(
        "aist_openpose_dir", "data/AIST++/Annotations/openpose"
    )
    mvh_openpose_root = config.get("mvh_openpose_root", "data/MVHumanNet")
    mv_root = config["mvhumannet_root"]
    mvh_cameras = config.get("mvh_cameras", ["22327091", "22327113", "22327084"])
    cond_cache_root = config.get("cond_cache_root", "data/cached_cond")

    path = meta["path"]
    if path.endswith(".pkl"):
        k2d, vis = load_aist_openpose(
            path,
            aist_openpose_dir,
            src_fps=aist_fps,
            target_fps=target_fps,
            cache_root=cond_cache_root,
            camera=meta.get("camera"),
        )
    else:
        k2d, vis = load_mvh_openpose(
            path,
            mv_root,
            mvh_openpose_root,
            mvh_cameras,
            src_fps=mvh_fps,
            target_fps=target_fps,
            cache_root=cond_cache_root,
            camera=meta.get("camera"),
        )

    if k2d is None:
        k2d = np.zeros((seq_len, 25, 2), dtype=np.float32)
        vis = np.zeros((seq_len, 25), dtype=np.float32)

    start = int(meta.get("start", 0))
    orig_len = int(meta.get("orig_len", k2d.shape[0]))
    if orig_len >= seq_len:
        k2d = k2d[start : start + seq_len]
        vis = vis[start : start + seq_len]
    else:
        pad_len = seq_len - orig_len
        k2d = np.concatenate(
            [k2d, np.zeros((pad_len, 25, 2), dtype=np.float32)], axis=0
        )
        vis = np.concatenate(
            [vis, np.zeros((pad_len, 25), dtype=np.float32)], axis=0
        )

    t_len = k2d.shape[0]
    k_frames = int(np.random.randint(cond_frames_min, cond_frames_max + 1))
    if t_len <= k_frames:
        idx = np.arange(t_len)
    else:
        idx = np.sort(np.random.choice(t_len, size=k_frames, replace=False))
    k2d = k2d[idx]
    vis = vis[idx]
    if cond_drop_prob > 0:
        drop = np.random.rand(*vis.shape) < cond_drop_prob
        vis = vis * (~drop)
        k2d = k2d * vis[..., None]
    tau_cond = idx.astype(np.float32) / max(t_len - 1, 1)
    return k2d, vis, tau_cond


def _distance(x, y):
    return torch.mean((x - y) ** 2).item()


def _print_stats(label, tensor):
    t = tensor.detach().float()
    finite = torch.isfinite(t)
    if not finite.all():
        t = t[finite]
    if t.numel() == 0:
        print(f"{label}: all non-finite")
        return
    print(
        f"{label}: min={t.min().item():.6f} max={t.max().item():.6f} mean={t.mean().item():.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--dataset", choices=["aist", "mvh", "both"], default="both")
    args = parser.parse_args()

    config = load_config()
    device = torch.device(args.device)
    seq_len = config["seq_len"]
    d_z = config["d_z"]
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)

    vae_ckpt = args.vae_ckpt or config.get("vae_ckpt", "checkpoints/motion_vae_best.pt")
    mean, std = load_mean_std(config["stats_path"])

    aist_train = config.get("aist_split_train")
    mvh_train = config.get("mvh_split_train")

    datasets = []
    if args.dataset in ("aist", "both"):
        with open(aist_train, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        aist_paths = [os.path.join(config["aist_motions_dir"], f"{n}.pkl") for n in names]
        dataset_a = AISTDataset(
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
        )
        datasets.append(dataset_a)

    if args.dataset in ("mvh", "both"):
        with open(mvh_train, "r", encoding="utf-8") as f:
            mvh_dirs = [line.strip() for line in f if line.strip()]
        dataset_b = MVHumanNetDataset(
            config["mvhumannet_root"],
            seq_len=seq_len,
            mean=mean,
            std=std,
            sequence_dirs=mvh_dirs,
            cache_root=config["cache_root"],
            target_fps=target_fps,
            src_fps=mvh_fps,
            camera_ids=config.get("mvh_cameras", ["22327091", "22327113", "22327084"]),
            expand_cameras=True,
        )
        datasets.append(dataset_b)

    if not datasets:
        raise ValueError("No datasets selected")

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
    flow_state = torch.load(args.flow_ckpt, map_location=device)
    flow.load_state_dict(flow_state["model"])
    flow.to(device)
    flow.eval()

    stats = np.load(config.get("openpose_stats_path", "data/openpose_stats.npz"))
    k2d_mean = stats["mean"]
    k2d_std = stats["std"]

    dt = 0.1
    x_t_dist = []
    x_t_step_dist = []
    pred_step_dist = []

    for sample_idx in range(args.num_samples):
        dataset = random.choice(datasets)
        sample = dataset[random.randint(0, len(dataset) - 1)]
        motion = sample["motion"].unsqueeze(0).to(device)
        style_id = sample["style_id"].unsqueeze(0).to(device)
        domain_id = sample["domain_id"].unsqueeze(0).to(device)

        with torch.no_grad():
            _h, mu, _logvar = vae.encode(motion, vae.cond(domain_id, style_id))
        x1 = mu
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=device)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1

        # Target direction step
        v_target = x1 - x0
        x_t_next = x_t + dt * v_target
        x_t_dist.append(_distance(x_t, x1))
        x_t_step_dist.append(_distance(x_t_next, x1))

        # Predicted direction step
        k2d, vis, tau_cond = _load_cond_for_meta(
            sample["meta"], config, target_fps, aist_fps, mvh_fps
        )
        k2d = torch.tensor(k2d, dtype=torch.float32, device=device).unsqueeze(0)
        vis = torch.tensor(vis, dtype=torch.float32, device=device).unsqueeze(0)
        tau_cond = torch.tensor(tau_cond, dtype=torch.float32, device=device).unsqueeze(0)

        g2d, mem, _vis = flow.cond_encoder(
            k2d, tau_cond, vis_mask=vis, mean=k2d_mean, std=k2d_std
        )
        style = flow.style_emb(style_id, domain_id, apply_dropout=False)
        g = flow.cond_mlp(torch.cat([g2d, style], dim=-1))
        tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
        v_pred = flow.flow(x_t, t, tau_out, mem, g)
        x_t_pred = x_t + dt * v_pred
        dist_pred = _distance(x_t_pred, x1)
        pred_step_dist.append(dist_pred)

        if not np.isfinite(dist_pred):
            print("Non-finite dist_after detected")
            print(f"sample_idx={sample_idx}")
            meta = sample["meta"]
            print(f"path={meta.get('path')}")
            if "camera" in meta:
                print(f"camera={meta.get('camera')}")
            _print_stats("k2d", k2d)
            _print_stats("vis", vis)
            _print_stats("g2d", g2d)
            _print_stats("mem", mem)
            _print_stats("v_pred", v_pred)
            _print_stats("x_t", x_t)
            _print_stats("x1", x1)
            break

    print("v_target step:")
    print(f"  dist_before={np.mean(x_t_dist):.6f}, dist_after={np.mean(x_t_step_dist):.6f}")
    print("v_pred step (avg over samples):")
    print(f"  dist_after={np.mean(pred_step_dist):.6f}")


if __name__ == "__main__":
    import random

    main()
