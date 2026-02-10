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
from flowmimic.src.model.flow.cond_api import build_dummy_cond
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.teacher import EMA, Teacher
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std
from flowmimic.src.model.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id
from flowmimic.src.data.openpose import (
    compute_openpose_stats,
    load_aist_openpose,
    load_mvh_openpose,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reflow-round", type=int, default=0)
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--teacher-solver", type=str, default=None)
    parser.add_argument("--use-ema-teacher", action="store_true")
    parser.add_argument("--teacher-mode", type=str, choices=["strict", "mixed"], default=None)
    parser.add_argument("--p-teacher", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_flow")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    seq_len = config["seq_len"]
    stats_path = config["stats_path"]
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)
    openpose_aist_dir = config.get("aist_openpose_dir", "data/AIST++/Annotations/openpose")
    openpose_mvh_root = config.get("mvh_openpose_root", "data/MVHumanNet")
    mvh_cameras = config.get("mvh_cameras", ["22327091", "22327113", "22327084"])
    openpose_stats_path = config.get("openpose_stats_path", "data/openpose_stats.npz")

    mean, std = load_mean_std(stats_path)

    genre_to_id = build_genre_to_id(config.get("aist_genres", []))

    flow_cfg = config.get("flow", {})
    lr = args.lr or flow_cfg.get("lr", 2e-4)
    weight_decay = flow_cfg.get("weight_decay", 1e-2)
    teacher_steps = args.teacher_steps or flow_cfg.get("teacher_steps", 16)
    teacher_solver = args.teacher_solver or flow_cfg.get("teacher_solver", "heun")
    teacher_mode = args.teacher_mode or flow_cfg.get("teacher_mode", "strict")
    p_teacher = args.p_teacher if args.p_teacher is not None else flow_cfg.get("p_teacher", 1.0)
    ema_decay = flow_cfg.get("ema_decay", 0.999)
    cond_frames_min = flow_cfg.get("cond_frames_min", 2)
    cond_frames_max = flow_cfg.get("cond_frames_max", 10)
    cond_drop_prob = flow_cfg.get("cond_drop_prob", 0.2)

    dataset_a = AISTDataset(
        aist_dir,
        genre_to_id=genre_to_id,
        seq_len=seq_len,
        mean=mean,
        std=std,
        cache_root=config["cache_root"],
        target_fps=target_fps,
        src_fps=aist_fps,
    )
    dataset_b = MVHumanNetDataset(
        mv_root,
        seq_len=seq_len,
        mean=mean,
        std=std,
        cache_root=config["cache_root"],
        target_fps=target_fps,
        src_fps=mvh_fps,
    )

    loader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=True, drop_last=True)
    batch_iter = balanced_batch_iter(loader_a, loader_b, 1, 1)

    vae = MotionVAE(d_in=config["d_in"], d_z=config["d_z"], num_styles=config["num_styles"], max_len=seq_len)
    vae_ckpt = torch.load(config.get("vae_ckpt", "checkpoints/motion_vae_best.pt"), map_location=args.device)
    vae.load_state_dict(vae_ckpt["model"])
    vae.to(args.device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    flow = ConditionalRectFlow(
        d_z=config["d_z"],
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
    flow.to(args.device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)

    if not os.path.exists(openpose_stats_path):
        compute_openpose_stats(
            aist_paths=_aist_split_paths(aist_dir, config["aist_split_train"]),
            mvh_dirs=_read_lines(config["mvh_split_train"]),
            aist_openpose_dir=openpose_aist_dir,
            mvh_openpose_root=openpose_mvh_root,
            mv_root=mv_root,
            cameras=mvh_cameras,
            target_fps=target_fps,
            aist_fps=aist_fps,
            mvh_fps=mvh_fps,
            out_path=openpose_stats_path,
        )
    stats = np.load(openpose_stats_path)
    k2d_mean = stats["mean"]
    k2d_std = stats["std"]

    teacher = None
    if args.reflow_round >= 1:
        if not args.teacher_ckpt:
            raise ValueError("teacher_ckpt is required for reflow_round >= 1")
        teacher_flow = ConditionalRectFlow(d_z=config["d_z"], num_styles=config["num_styles"])
        state = torch.load(args.teacher_ckpt, map_location=args.device)
        if "ema" in state:
            teacher_flow.load_state_dict(state["ema"])
        else:
            teacher_flow.load_state_dict(state["model"])
        teacher_flow.to(args.device)
        teacher = Teacher(
            teacher_flow,
            solver_cfg={"num_steps": teacher_steps, "method": teacher_solver},
        )

    ema = EMA(flow, decay=ema_decay) if args.use_ema_teacher else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=args.device)

    for epoch in range(args.epochs):
        flow.train()
        total_loss = 0.0
        total_count = 0
        for _ in tqdm(range(min(len(loader_a), len(loader_b))), desc=f"Flow Epoch {epoch+1}", leave=False):
            batch = next(batch_iter)
            motion, domain_id, style_id, mask, metas = _merge_batches(batch)
            motion = motion.to(args.device)
            domain_id = domain_id.to(args.device)
            style_id = style_id.to(args.device)

            with torch.no_grad():
                enc_h, mu, _logvar = vae.encode(motion, vae.cond(domain_id, style_id), mask=mask.to(args.device))
                z_data = mu

            x0 = torch.randn_like(z_data)
            t = torch.rand(z_data.shape[0], device=args.device)
            x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * z_data

            k2d_batch, vis_batch, tau_cond, mask_cond = _load_cond_batch(
                metas,
                openpose_aist_dir,
                openpose_mvh_root,
                mv_root,
                mvh_cameras,
                seq_len,
                cond_frames_min,
                cond_frames_max,
                cond_drop_prob,
                aist_fps,
                mvh_fps,
                target_fps,
            )
            k2d_batch = k2d_batch.to(args.device)
            vis_batch = vis_batch.to(args.device)
            tau_cond = tau_cond.to(args.device)
            mask_cond = mask_cond.to(args.device)
            g2d, mem, _vis = flow.cond_encoder(
                k2d_batch,
                tau_cond,
                vis_mask=vis_batch,
                mask_cond=mask_cond,
                mean=k2d_mean,
                std=k2d_std,
            )
            style = flow.style_emb(style_id, domain_id, apply_dropout=True)
            g = flow.cond_mlp(torch.cat([g2d, style], dim=-1))
            v_pred = flow.flow(x_t, t, tau_out, mem, g)
            target = z_data - x0

            if teacher is not None and args.reflow_round >= 1:
                use_teacher = True
                if teacher_mode == "mixed":
                    use_teacher = torch.rand(1).item() < p_teacher
                if use_teacher:
                    with torch.no_grad():
                        style_t = flow.style_emb(style_id, domain_id, apply_dropout=False)
                        g_t = flow.cond_mlp(torch.cat([g2d, style_t], dim=-1))
                        cond_batch = {"tau_out": tau_out, "mem": mem, "g": g_t}
                        z_data = teacher.generate_x1_hat(x0, cond_batch)
                        target = z_data - x0

            loss = torch.mean((v_pred - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(flow)
            total_loss += loss.item()
            total_count += 1

        if args.debug:
            print(f"Epoch {epoch+1} avg_flow_loss={total_loss / max(total_count, 1):.6f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"flow_round{args.reflow_round}_epoch{epoch+1}.pt")
        state = {"model": flow.state_dict()}
        if ema is not None:
            state["ema"] = ema.state_dict()
        torch.save(state, ckpt_path)


def _merge_batches(batches):
    motions = []
    domain_ids = []
    style_ids = []
    masks = []
    metas = []
    for batch in batches:
        motions.append(batch["motion"])
        domain_ids.append(batch["domain_id"])
        style_ids.append(batch["style_id"])
        masks.append(batch["mask"])
        metas.extend(batch["meta"])
    motion = torch.cat(motions, dim=0)
    domain_id = torch.cat(domain_ids, dim=0)
    style_id = torch.cat(style_ids, dim=0)
    mask = torch.cat(masks, dim=0)
    return motion, domain_id, style_id, mask, metas


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _aist_split_paths(aist_dir, split_path):
    names = _read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def _load_cond_batch(
    metas,
    aist_openpose_dir,
    mvh_openpose_root,
    mv_root,
    cameras,
    seq_len,
    cond_frames_min,
    cond_frames_max,
    cond_drop_prob,
    aist_fps,
    mvh_fps,
    target_fps,
):
    k2d_list = []
    vis_list = []
    mask_list = []
    tau_list = []
    for meta in metas:
        path = meta["path"]
        if path.endswith(".pkl"):
            k2d, vis = load_aist_openpose(
                path, aist_openpose_dir, src_fps=aist_fps, target_fps=target_fps
            )
        else:
            k2d, vis = load_mvh_openpose(
                path,
                mv_root,
                mvh_openpose_root,
                cameras,
                src_fps=mvh_fps,
                target_fps=target_fps,
            )
        if k2d is None:
            k2d = np.zeros((seq_len, 25, 2), dtype=np.float32)
            vis = np.zeros((seq_len, 25), dtype=np.float32)
        start = meta.get("start", 0)
        orig_len = meta.get("orig_len", k2d.shape[0])
        if orig_len >= seq_len:
            k2d = k2d[start : start + seq_len]
            vis = vis[start : start + seq_len]
        else:
            pad_len = seq_len - orig_len
            k2d = np.concatenate([k2d, np.zeros((pad_len, 25, 2), dtype=np.float32)], axis=0)
            vis = np.concatenate([vis, np.zeros((pad_len, 25), dtype=np.float32)], axis=0)
        t_len = k2d.shape[0]
        k_frames = int(np.random.randint(cond_frames_min, cond_frames_max + 1))
        if t_len <= k_frames:
            idx = np.arange(t_len)
        else:
            idx = np.sort(np.random.choice(t_len, size=k_frames, replace=False))
        k2d_sparse = k2d[idx]
        vis_sparse = vis[idx]
        if cond_drop_prob > 0:
            drop = np.random.rand(*vis_sparse.shape) < cond_drop_prob
            vis_sparse = vis_sparse * (~drop)
            k2d_sparse = k2d_sparse * vis_sparse[..., None]
        mask_cond = np.ones((k2d_sparse.shape[0],), dtype=bool)
        tau_cond = idx.astype(np.float32) / max(t_len - 1, 1)

        k2d_list.append(k2d_sparse)
        vis_list.append(vis_sparse)
        mask_list.append(mask_cond)
        tau_list.append(tau_cond)

    max_len = max(k.shape[0] for k in k2d_list)
    k2d_pad = []
    vis_pad = []
    mask_pad = []
    tau_pad = []
    for k2d, vis, mask, tau in zip(k2d_list, vis_list, mask_list, tau_list):
        pad = max_len - k2d.shape[0]
        if pad > 0:
            k2d = np.concatenate([k2d, np.zeros((pad, 25, 2), dtype=np.float32)], axis=0)
            vis = np.concatenate([vis, np.zeros((pad, 25), dtype=np.float32)], axis=0)
            mask = np.concatenate([mask, np.zeros((pad,), dtype=bool)], axis=0)
            tau = np.concatenate([tau, np.zeros((pad,), dtype=np.float32)], axis=0)
        k2d_pad.append(k2d)
        vis_pad.append(vis)
        mask_pad.append(mask)
        tau_pad.append(tau)

    k2d_batch = torch.from_numpy(np.stack(k2d_pad, axis=0))
    vis_batch = torch.from_numpy(np.stack(vis_pad, axis=0))
    mask_batch = torch.from_numpy(np.stack(mask_pad, axis=0))
    tau_batch = torch.from_numpy(np.stack(tau_pad, axis=0))
    return k2d_batch, vis_batch, tau_batch, mask_batch


if __name__ == "__main__":
    main()
