"""Visualize flow path straightness with PCA-projected trajectories.

Example:
  python flowmimic/tools/vis_path.py --flow-ckpt checkpoints/flow/flow_round0_last.pt
"""

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
from flowmimic.src.data.openpose import load_aist_openpose, load_mvh_openpose
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        vis = np.concatenate([vis, np.zeros((pad_len, 25), dtype=np.float32)], axis=0)

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
    mask_cond = np.ones((k2d.shape[0],), dtype=bool)
    return k2d, vis, tau_cond, mask_cond


def _pad_batch(cond_list):
    max_len = max(item[0].shape[0] for item in cond_list)
    k2d_pad, vis_pad, tau_pad, mask_pad = [], [], [], []
    for k2d, vis, tau, mask in cond_list:
        pad = max_len - k2d.shape[0]
        if pad > 0:
            k2d = np.concatenate(
                [k2d, np.zeros((pad, 25, 2), dtype=np.float32)], axis=0
            )
            vis = np.concatenate([vis, np.zeros((pad, 25), dtype=np.float32)], axis=0)
            tau = np.concatenate([tau, np.zeros((pad,), dtype=np.float32)], axis=0)
            mask = np.concatenate([mask, np.zeros((pad,), dtype=bool)], axis=0)
        k2d_pad.append(k2d)
        vis_pad.append(vis)
        tau_pad.append(tau)
        mask_pad.append(mask)
    return (
        np.stack(k2d_pad, axis=0),
        np.stack(vis_pad, axis=0),
        np.stack(tau_pad, axis=0),
        np.stack(mask_pad, axis=0),
    )


def _rollout(flow_model, x0, tau_out, mem, g, mem_mask, steps, method):
    t_grid = torch.linspace(0.0, 1.0, steps=steps, device=x0.device)
    traj = []
    x = x0
    for i in range(steps):
        t = t_grid[i].expand(x0.shape[0])
        traj.append(x)
        if i == steps - 1:
            break
        t_next = t_grid[i + 1]
        dt = t_next - t_grid[i]
        if method == "heun":
            v1 = flow_model.flow(x, t, tau_out, mem, g, mem_mask=mem_mask)
            x1 = x + v1 * dt
            v2 = flow_model.flow(
                x1, t_next.expand(x0.shape[0]), tau_out, mem, g, mem_mask=mem_mask
            )
            x = x + 0.5 * (v1 + v2) * dt
        else:
            v = flow_model.flow(x, t, tau_out, mem, g, mem_mask=mem_mask)
            x = x + v * dt
    return torch.stack(traj, dim=1)


def _project_pca(traj_list, shared_basis=True):
    from sklearn.decomposition import PCA

    if shared_basis:
        all_points = np.concatenate(
            [t.reshape(t.shape[0] * t.shape[1], -1) for t in traj_list], axis=0
        )
        pca = PCA(n_components=2)
        pca.fit(all_points)
        out = []
        for traj in traj_list:
            flat = traj.reshape(traj.shape[0] * traj.shape[1], -1)
            proj = pca.transform(flat).reshape(traj.shape[0], traj.shape[1], 2)
            out.append(proj)
        return out
    out = []
    for traj in traj_list:
        flat = traj.reshape(traj.shape[0] * traj.shape[1], -1)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(flat).reshape(traj.shape[0], traj.shape[1], 2)
        out.append(proj)
    return out


def _plot_traj(ax, traj2, title):
    for i in range(traj2.shape[0]):
        ax.plot(traj2[i, :, 0], traj2[i, :, 1], linewidth=0.8, alpha=0.6)
    ax.scatter(traj2[:, 0, 0], traj2[:, 0, 1], s=10, alpha=0.7)
    ax.scatter(traj2[:, -1, 0], traj2[:, -1, 1], s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def _path_deviation(traj, t_grid):
    # traj: [N, K, T, D], t_grid: [K]
    x0 = traj[:, 0:1]
    x1 = traj[:, -1:]
    t = t_grid[None, :, None, None]
    line = (1 - t) * x0 + t * x1
    diff = traj - line
    dev = np.linalg.norm(diff, axis=(-1, -2))
    avg_dev = dev.mean(axis=1)
    max_dev = dev.max(axis=1)
    return avg_dev, max_dev


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
            )
        )
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--reflow-ckpt", default=None)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--solver", type=str, default="heun")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="output/flow/path_vis.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _seed_all(args.seed)
    config = load_config()
    device = torch.device(args.device)
    seq_len = config["seq_len"]
    d_z = config["d_z"]
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)
    latent_stats_path = config.get("latent_stats_path", "data/latent_stats.npz")

    datasets = _build_datasets(config)
    if not datasets:
        raise ValueError("No datasets available for sampling")

    vae_ckpt = args.vae_ckpt or config.get("vae_ckpt", "checkpoints/motion_vae_best.pt")
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

    flow2 = None
    if args.reflow_ckpt:
        flow2 = ConditionalRectFlow(
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
        flow2_state = torch.load(args.reflow_ckpt, map_location=device)
        flow2.load_state_dict(flow2_state["model"])
        flow2.to(device)
        flow2.eval()

    k2d_mean = None
    k2d_std = None
    stats_path = config.get("openpose_stats_path", "data/openpose_stats.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        k2d_mean = stats["mean"]
        k2d_std = stats["std"]

    latent_mean = None
    latent_std = None
    if os.path.exists(latent_stats_path):
        stats = np.load(latent_stats_path)
        latent_mean = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
        latent_std = torch.tensor(stats["std"], device=device, dtype=torch.float32)

    samples = []
    for _ in range(args.num_samples):
        dataset = random.choice(datasets)
        sample = dataset[random.randint(0, len(dataset) - 1)]
        samples.append(sample)

    motions = torch.stack([s["motion"] for s in samples], dim=0).to(device)
    style_id = torch.stack([s["style_id"] for s in samples], dim=0).to(device)
    domain_id = torch.stack([s["domain_id"] for s in samples], dim=0).to(device)
    metas = [s["meta"] for s in samples]

    with torch.no_grad():
        _h, mu, _logvar = vae.encode(motions, vae.cond(domain_id, style_id))
    x1 = mu
    if latent_mean is not None and latent_std is not None:
        x1 = (x1 - latent_mean) / (latent_std + 1e-6)

    x0 = torch.randn_like(x1)

    cond_list = [
        _load_cond_for_meta(m, config, target_fps, aist_fps, mvh_fps) for m in metas
    ]
    k2d_np, vis_np, tau_np, mask_np = _pad_batch(cond_list)
    k2d = torch.tensor(k2d_np, device=device)
    vis = torch.tensor(vis_np, device=device)
    tau_cond = torch.tensor(tau_np, device=device)
    mask_cond = torch.tensor(mask_np, device=device)

    g2d, mem, _vis = flow.cond_encoder(
        k2d, tau_cond, vis_mask=vis, mask_cond=mask_cond, mean=k2d_mean, std=k2d_std
    )
    style = flow.style_emb(style_id, domain_id, apply_dropout=False)
    g = flow.cond_mlp(torch.cat([g2d, style], dim=-1))
    mem_mask = ~mask_cond
    tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=device)

    traj1 = _rollout(flow, x0, tau_out, mem, g, mem_mask, args.steps, args.solver)
    traj_list = [traj1.detach().cpu().numpy()]

    if flow2 is not None:
        g2d2, mem2, _vis2 = flow2.cond_encoder(
            k2d, tau_cond, vis_mask=vis, mask_cond=mask_cond, mean=k2d_mean, std=k2d_std
        )
        style2 = flow2.style_emb(style_id, domain_id, apply_dropout=False)
        g2 = flow2.cond_mlp(torch.cat([g2d2, style2], dim=-1))
        traj2 = _rollout(
            flow2, x0, tau_out, mem2, g2, mem_mask, args.steps, args.solver
        )
        traj_list.append(traj2.detach().cpu().numpy())

    proj = _project_pca(traj_list, shared_basis=True)

    t_grid = np.linspace(0.0, 1.0, args.steps, dtype=np.float32)
    avg_dev1, max_dev1 = _path_deviation(traj_list[0], t_grid)
    print(
        f"Z1 avg_dev mean={avg_dev1.mean():.6f} std={avg_dev1.std():.6f} max={max_dev1.mean():.6f}"
    )
    if len(traj_list) > 1:
        avg_dev2, max_dev2 = _path_deviation(traj_list[1], t_grid)
        print(
            f"Z2 avg_dev mean={avg_dev2.mean():.6f} std={avg_dev2.std():.6f} max={max_dev2.mean():.6f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_traj(axes[0], proj[0], "(a) The 1st rectified flow Z1")
    if len(proj) > 1:
        _plot_traj(axes[1], proj[1], "(b) Reflow Z2")
    else:
        axes[1].set_title("(b) Reflow Z2 (placeholder)")
        axes[1].axis("off")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
