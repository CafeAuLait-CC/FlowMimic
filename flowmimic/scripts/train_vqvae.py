import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
torch.multiprocessing.set_sharing_strategy("file_system")

from flowmimic.scripts.train_vae import (
    _ddp_any,
    _denormalize_ik263,
    _parse_dataset_names,
    _ramp_weight,
    _seed_worker,
    _single_loader_iter,
    aist_split_paths,
    apply_style_dropout,
    merge_batches,
    read_lines,
)
from flowmimic.src.config.config import load_config
from flowmimic.src.model.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.datasets.label_map_builder import (
    build_genre_to_id,
    save_genre_to_id,
)
from flowmimic.src.model.vae.losses import (
    continuous_smoothness_loss,
    grouped_recon_loss,
    style_ce_loss,
)
from flowmimic.src.model.vae.motion_vqvae import MotionVQVAE
from flowmimic.src.model.vae.stats import compute_mean_std_from_splits, load_mean_std
from flowmimic.src.motion.ik.common.quaternion import qinv, qrot
from flowmimic.src.motion.process_motion import ik263_to_smpl22

CONT_END = 259
DISTAL_JOINTS = (10, 11, 20, 21)


def _denormalize_ik263_torch(motion, mean_t, std_t):
    out = motion.clone()
    out[..., :CONT_END] = out[..., :CONT_END] * std_t + mean_t
    return out


def _ik263_to_smpl22_torch(features):
    rot_vel = features[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(
        rot_vel.shape + (4,), dtype=features.dtype, device=features.device
    )
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(
        rot_vel.shape + (3,), dtype=features.dtype, device=features.device
    )
    r_pos[..., 1:, 0] = features[..., :-1, 1]
    r_pos[..., 1:, 2] = features[..., :-1, 2]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = features[..., 3]

    start = 4
    end = start + (22 - 1) * 3
    positions = features[..., start:end].reshape(features.shape[:-1] + (22 - 1, 3))
    r_rot_rep = r_rot_quat.unsqueeze(-2).expand(
        *r_rot_quat.shape[:-1], positions.shape[-2], 4
    )
    positions = qrot(qinv(r_rot_rep), positions)
    positions[..., 0] = positions[..., 0] + r_pos[..., 0:1]
    positions[..., 2] = positions[..., 2] + r_pos[..., 2:3]
    return torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)


def _masked_smooth_l1_nd(pred, target, mask=None):
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    if mask is None:
        return loss.mean()
    mask_f = mask.float()
    while mask_f.ndim < loss.ndim:
        mask_f = mask_f.unsqueeze(-1)
    denom = mask.float().sum().clamp_min(1.0)
    for dim_size in loss.shape[2:]:
        denom = denom * dim_size
    return (loss * mask_f).sum() / denom


def _last_valid_root_xz(joints, mask):
    lengths = mask.long().sum(dim=1).clamp_min(1)
    idx = lengths - 1
    batch = torch.arange(joints.shape[0], device=joints.device)
    return joints[batch, idx, 0][:, [0, 2]]


def visible_recon_losses(x_hat, x, mask, mean_t, std_t):
    pred_ik = _denormalize_ik263_torch(x_hat, mean_t, std_t)
    target_ik = _denormalize_ik263_torch(x, mean_t, std_t)
    pred_joints = _ik263_to_smpl22_torch(pred_ik)
    target_joints = _ik263_to_smpl22_torch(target_ik)

    joint = _masked_smooth_l1_nd(pred_joints, target_joints, mask)
    distal_idx = torch.tensor(DISTAL_JOINTS, device=x_hat.device, dtype=torch.long)
    distal = _masked_smooth_l1_nd(
        pred_joints.index_select(-2, distal_idx),
        target_joints.index_select(-2, distal_idx),
        mask,
    )
    root_xz = _masked_smooth_l1_nd(
        pred_joints[:, :, 0, [0, 2]],
        target_joints[:, :, 0, [0, 2]],
        mask,
    )
    pred_drift = _last_valid_root_xz(pred_joints, mask) - pred_joints[:, 0, 0, [0, 2]]
    target_drift = _last_valid_root_xz(target_joints, mask) - target_joints[:, 0, 0, [0, 2]]
    root_drift = torch.nn.functional.smooth_l1_loss(pred_drift, target_drift)

    vel_mask = mask[:, 1:] & mask[:, :-1]
    pred_distal_vel = (
        pred_joints[:, 1:].index_select(-2, distal_idx)
        - pred_joints[:, :-1].index_select(-2, distal_idx)
    )
    target_distal_vel = (
        target_joints[:, 1:].index_select(-2, distal_idx)
        - target_joints[:, :-1].index_select(-2, distal_idx)
    )
    distal_vel = _masked_smooth_l1_nd(pred_distal_vel, target_distal_vel, vel_mask)
    return {
        "joint": joint,
        "distal": distal,
        "root_xz": root_xz,
        "root_drift": root_drift,
        "distal_vel": distal_vel,
    }


def _masked_distal_path_lengths(joints, mask):
    distal = joints[:, :, list(DISTAL_JOINTS)]
    vel = distal[:, 1:] - distal[:, :-1]
    valid = (mask[:, 1:] & mask[:, :-1]).astype(bool)
    lengths = []
    for b in range(joints.shape[0]):
        if not valid[b].any():
            continue
        lengths.append(np.linalg.norm(vel[b, valid[b]], axis=-1).sum(axis=0))
    if not lengths:
        return np.zeros((0, len(DISTAL_JOINTS)), dtype=np.float32)
    return np.stack(lengths, axis=0).astype(np.float32)


def _joint_quality_from_arrays(pred_joints, target_joints, mask_np, prefix, fps=30.0):
    diff = pred_joints - target_joints
    valid_diff = diff[mask_np]
    if valid_diff.size == 0:
        return {}, 0

    joint_l2 = np.linalg.norm(valid_diff, axis=-1)
    root_diff = diff[:, :, 0, [0, 2]]
    valid_root_diff = root_diff[mask_np]
    root_xz_l2 = np.linalg.norm(valid_root_diff, axis=-1)
    root_full_l2 = np.linalg.norm(diff[:, :, 0][mask_np], axis=-1)
    distal_diff = diff[:, :, list(DISTAL_JOINTS)]
    distal_l2 = np.linalg.norm(distal_diff[mask_np], axis=-1)

    vel_mask = mask_np[:, 1:] & mask_np[:, :-1]
    pred_velocity = np.diff(pred_joints, axis=1) * fps
    target_velocity = np.diff(target_joints, axis=1) * fps
    velocity_error = np.linalg.norm(pred_velocity - target_velocity, axis=-1)
    distal_velocity_error = velocity_error[..., list(DISTAL_JOINTS)]
    joint_velocity_l2 = float(velocity_error[vel_mask].mean())
    distal_velocity_l2 = float(distal_velocity_error[vel_mask].mean())

    full_clips = mask_np.all(axis=1)
    if full_clips.any():
        pred_distal_velocity = pred_velocity[full_clips][..., list(DISTAL_JOINTS), :]
        target_distal_velocity = target_velocity[full_clips][..., list(DISTAL_JOINTS), :]
        pred_spectrum = np.abs(np.fft.rfft(pred_distal_velocity, axis=1))
        target_spectrum = np.abs(np.fft.rfft(target_distal_velocity, axis=1))
        frequencies = np.fft.rfftfreq(pred_distal_velocity.shape[1], d=1.0 / fps)
        high_frequency_mask = frequencies >= 3.0
        high_frequency_spectrum_l1 = float(
            np.abs(
                pred_spectrum[:, high_frequency_mask]
                - target_spectrum[:, high_frequency_mask]
            ).sum()
            / max(float(target_spectrum[:, high_frequency_mask].sum()), 1e-8)
        )
        high_frequency_energy_ratio = float(
            np.square(pred_spectrum[:, high_frequency_mask]).sum()
            / max(
                float(np.square(target_spectrum[:, high_frequency_mask]).sum()),
                1e-8,
            )
        )
    else:
        high_frequency_spectrum_l1 = 0.0
        high_frequency_energy_ratio = 0.0

    pred_path = _masked_distal_path_lengths(pred_joints, mask_np)
    target_path = _masked_distal_path_lengths(target_joints, mask_np)
    if len(pred_path) and len(target_path):
        path_ratio = pred_path / np.maximum(target_path, 1e-6)
        path_ratio_mean = float(np.mean(path_ratio))
        path_ratio_abs_err = float(np.mean(np.abs(path_ratio - 1.0)))
    else:
        path_ratio_mean = 0.0
        path_ratio_abs_err = 1.0

    drift_errors = []
    for b in range(mask_np.shape[0]):
        valid_idx = np.flatnonzero(mask_np[b])
        if len(valid_idx) == 0:
            continue
        first = valid_idx[0]
        last = valid_idx[-1]
        pred_drift = pred_joints[b, last, 0, [0, 2]] - pred_joints[b, first, 0, [0, 2]]
        target_drift = target_joints[b, last, 0, [0, 2]] - target_joints[b, first, 0, [0, 2]]
        drift_errors.append(float(np.linalg.norm(pred_drift - target_drift)))
    drift_l2 = float(np.mean(drift_errors)) if drift_errors else 0.0

    score = (
        float(joint_l2.mean())
        + float(distal_l2.mean())
        + float(root_xz_l2.mean())
        + drift_l2
        + path_ratio_abs_err
    )
    metrics = {
        f"{prefix}/joint_mse": float(np.square(valid_diff).mean()),
        f"{prefix}/joint_l2": float(joint_l2.mean()),
        f"{prefix}/root_l2": float(root_full_l2.mean()),
        f"{prefix}/root_xz_l2": float(root_xz_l2.mean()),
        f"{prefix}/root_drift_l2": drift_l2,
        f"{prefix}/distal_l2": float(distal_l2.mean()),
        f"{prefix}/joint_velocity_l2": joint_velocity_l2,
        f"{prefix}/distal_velocity_l2": distal_velocity_l2,
        f"{prefix}/distal_velocity_high_frequency_spectrum_l1_relative": (
            high_frequency_spectrum_l1
        ),
        f"{prefix}/distal_velocity_high_frequency_energy_ratio": (
            high_frequency_energy_ratio
        ),
        f"{prefix}/distal_path_ratio": path_ratio_mean,
        f"{prefix}/distal_path_ratio_abs_err": path_ratio_abs_err,
        f"{prefix}/score": score,
    }
    return metrics, int(mask_np.shape[0])


def _fixed_aist_joint_metrics(model, loader, device, mean, std, max_samples, fps):
    if loader is None or max_samples <= 0:
        return {}

    pred_chunks = []
    target_chunks = []
    mask_chunks = []
    used = 0
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            domain_id = batch["domain_id"].to(device)
            style_id = batch["style_id"].to(device)
            mask = batch["mask"].to(device)
            if used + motion.shape[0] > max_samples:
                keep = max_samples - used
                motion = motion[:keep]
                domain_id = domain_id[:keep]
                style_id = style_id[:keep]
                mask = mask[:keep]

            outputs = model(
                motion, domain_id, style_id, mask=mask, update_codebook=False
            )
            pred = _denormalize_ik263(outputs["x_hat"], mean, std)
            target = _denormalize_ik263(motion, mean, std)
            pred_joints = ik263_to_smpl22(pred)
            target_joints = ik263_to_smpl22(target)
            mask_np = mask.detach().cpu().numpy().astype(bool)
            pred_chunks.append(pred_joints)
            target_chunks.append(target_joints)
            mask_chunks.append(mask_np)
            used += int(motion.shape[0])
            if used >= max_samples:
                break

    if used == 0:
        return {}
    pred_all = np.concatenate(pred_chunks, axis=0)
    target_all = np.concatenate(target_chunks, axis=0)
    mask_all = np.concatenate(mask_chunks, axis=0)
    metrics, samples = _joint_quality_from_arrays(
        pred_all, target_all, mask_all, "val_quality/aist", fps=fps
    )
    legacy = {
        "val_fixed/aist_joint_mse": metrics["val_quality/aist/joint_mse"],
        "val_fixed/aist_joint_l2": metrics["val_quality/aist/joint_l2"],
        "val_fixed/aist_root_l2": metrics["val_quality/aist/root_l2"],
        "val_fixed/aist_joint_samples": samples,
    }
    return {**legacy, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="AIST,MVH")
    parser.add_argument("--val-datasets", type=str, default="AIST")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-len", type=int, default=16)
    parser.add_argument(
        "--latent-token-mode", choices=("pool", "query"), default="query"
    )
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--codebook-decay", type=float, default=0.99)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--ratio-aist", type=int, default=1)
    parser.add_argument("--ratio-mvh", type=int, default=1)
    parser.add_argument(
        "--aist-crop-mode", choices=("first", "random", "uniform"), default="random"
    )
    parser.add_argument(
        "--aist-val-crop-mode",
        choices=("first", "random", "uniform"),
        default="uniform",
    )
    parser.add_argument("--aist-clip-repeat", type=int, default=32)
    parser.add_argument("--aist-val-clip-repeat", type=int, default=4)
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--val-every-epochs", type=int, default=None)
    parser.add_argument("--val-joint-metric-samples", type=int, default=64)
    parser.add_argument("--w-vel", type=float, default=None)
    parser.add_argument("--w-acc", type=float, default=None)
    parser.add_argument("--w-joint", type=float, default=1.0)
    parser.add_argument("--w-distal", type=float, default=2.0)
    parser.add_argument("--w-root-traj", type=float, default=2.0)
    parser.add_argument("--w-root-drift", type=float, default=2.0)
    parser.add_argument("--w-distal-vel", type=float, default=1.0)
    parser.add_argument("--smooth-warmup-frac", type=float, default=0.2)
    parser.add_argument("--w-style", type=float, default=None)
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--w-root", type=float, default=None)
    parser.add_argument("--w-root-late-start", type=float, default=None)
    parser.add_argument("--w-root-late-factor", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/vqvae")
    parser.add_argument(
        "--archive-every-epochs",
        type=int,
        default=0,
        help="Keep an immutable epoch checkpoint at this interval; 0 disables it.",
    )
    parser.add_argument(
        "--genre-map", type=str, default="flowmimic/src/config/genre_to_id.json"
    )
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--reset-best-val", action="store_true")
    parser.add_argument(
        "--best-metric",
        type=str,
        default="val_quality/aist/score",
        help="Metric to minimize for motion_vqvae_best.pt. Use val/recon for old behavior.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-epochs", type=int, default=0)
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument("--debug-every", type=int, default=50)
    parser.add_argument("--wandb-project", type=str, default="FlowMimic")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="FlowMimic-VQVAE")
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=("allow", "must", "never"),
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
    )
    args = parser.parse_args()

    ddp = args.ddp
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("--ddp requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device(args.device)
        rank = 0
        world_size = 1
    is_main = rank == 0
    args.device = str(device)

    try:
        config = load_config()
        datasets = _parse_dataset_names(args.datasets)
        val_datasets = _parse_dataset_names(args.val_datasets)
        use_aist = "AIST" in datasets
        use_mvh = "MVH" in datasets
        val_use_aist = "AIST" in val_datasets
        val_use_mvh = "MVH" in val_datasets

        seed = config.get("seed", 42) + rank
        random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        wandb_run = None
        if is_main:
            print("Config loaded")
            try:
                import wandb  # type: ignore
            except ModuleNotFoundError:
                wandb = None
                print("wandb not installed; continuing without logging.")
            if wandb is not None:
                tags = [t for t in (args.wandb_tags or "").split(",") if t]
                run_name = args.wandb_name
                if run_name is None:
                    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
                    run_name = f"vqvae-{stamp}"
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    group=args.wandb_group,
                    id=args.wandb_id,
                    resume=args.wandb_resume if args.wandb_id else None,
                    tags=tags or None,
                    mode=args.wandb_mode,
                    config={
                        **vars(args),
                        "world_size": world_size,
                        "selected_datasets": sorted(datasets),
                        "selected_val_datasets": sorted(val_datasets),
                    },
                )
                wandb_run.define_metric("epoch")
                for metric_pattern in (
                    "loss/*",
                    "vq/*",
                    "val/*",
                    "val_fixed/*",
                    "val_quality/*",
                ):
                    wandb_run.define_metric(metric_pattern, step_metric="epoch")

        aist_dir = config["aist_motions_dir"]
        mv_root = config["mvhumannet_root"]
        aist_genres = config["aist_genres"]
        num_styles = config["num_styles"]
        d_in = config["d_in"]
        d_z = config["d_z"]
        seq_len = args.seq_len or config["seq_len"]
        batch_size = args.batch_size or config["train_batch_size"]
        num_workers = config["num_workers"]
        pin_memory = config["pin_memory"]
        prefetch_factor = config["prefetch_factor"]
        persistent_workers = config["persistent_workers"]
        val_every_epochs = args.val_every_epochs or config["val_every_epochs"]
        eval_batch_size = config["eval_batch_size"]
        w_vel = args.w_vel if args.w_vel is not None else config["w_vel"]
        w_acc = args.w_acc if args.w_acc is not None else config["w_acc"]
        w_style = args.w_style if args.w_style is not None else config["w_style"]
        w_contact = (
            args.w_contact if args.w_contact is not None else config["w_contact"]
        )
        w_root = args.w_root if args.w_root is not None else config.get("w_root", 1.0)
        w_root_late_start = (
            args.w_root_late_start
            if args.w_root_late_start is not None
            else config.get("w_root_late_start", 1.0)
        )
        w_root_late_factor = (
            args.w_root_late_factor
            if args.w_root_late_factor is not None
            else config.get("w_root_late_factor", 1.0)
        )
        style_dropout_p = config["style_dropout_p"]
        stats_path = args.stats_path or config["stats_path"]
        smooth_warmup_frac = args.smooth_warmup_frac
        target_fps = config.get("target_fps", 30)
        aist_fps = config.get("aist_fps", 60)
        mvh_fps = config.get("mvh_fps", 5)
        cache_root = config["cache_root"]
        aist_split_train = config["aist_split_train"]
        mvh_split_train = config["mvh_split_train"]
        grad_clip_norm = config["grad_clip_norm"]

        if use_aist and not os.path.exists(aist_split_train):
            raise FileNotFoundError(f"AIST split file not found: {aist_split_train}")
        if use_mvh and not os.path.exists(mvh_split_train):
            raise FileNotFoundError(f"MVHumanNet split file not found: {mvh_split_train}")

        if is_main and not os.path.exists(args.genre_map):
            save_genre_to_id(build_genre_to_id(aist_genres), args.genre_map)
        if ddp:
            dist.barrier()
        with open(args.genre_map, "r", encoding="utf-8") as f:
            genre_to_id = json.load(f)

        aist_train_paths = aist_split_paths(aist_dir, aist_split_train) if use_aist else []
        mvh_train_dirs = read_lines(mvh_split_train) if use_mvh else []
        if not os.path.exists(stats_path):
            if is_main:
                print(f"Computing mean/std: {stats_path}")
                compute_mean_std_from_splits(
                    aist_train_paths,
                    mvh_train_dirs,
                    stats_path,
                    workers=10,
                    target_fps=target_fps,
                    aist_fps=aist_fps,
                    mvh_fps=mvh_fps,
                )
            if ddp:
                dist.barrier()
        mean, std = load_mean_std(stats_path)
        mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
        std_t = torch.from_numpy(std.astype(np.float32)).to(device)

        loader_a = sampler_a = None
        if use_aist:
            if is_main:
                print(f"Building AIST++ train dataset ({len(aist_train_paths)} files)")
            dataset_a = AISTDataset(
                aist_dir,
                genre_to_id,
                seq_len,
                mean=mean,
                std=std,
                files=aist_train_paths,
                cache_root=cache_root,
                target_fps=target_fps,
                src_fps=aist_fps,
                crop_mode=args.aist_crop_mode,
                clip_repeat=args.aist_clip_repeat,
            )
            sampler_a = (
                DistributedSampler(dataset_a, shuffle=True, drop_last=True)
                if ddp
                else None
            )
            loader_a = DataLoader(
                dataset_a,
                batch_size=batch_size,
                shuffle=sampler_a is None,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                sampler=sampler_a,
                worker_init_fn=_seed_worker(seed),
            )

        loader_b = sampler_b = None
        if use_mvh:
            if is_main:
                print(f"Building MVHumanNet train dataset ({len(mvh_train_dirs)} dirs)")
            dataset_b = MVHumanNetDataset(
                mv_root,
                seq_len,
                mean=mean,
                std=std,
                sequence_dirs=mvh_train_dirs,
                cache_root=cache_root,
                target_fps=target_fps,
                src_fps=mvh_fps,
            )
            sampler_b = (
                DistributedSampler(dataset_b, shuffle=True, drop_last=True)
                if ddp
                else None
            )
            loader_b = DataLoader(
                dataset_b,
                batch_size=batch_size,
                shuffle=sampler_b is None,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                sampler=sampler_b,
                worker_init_fn=_seed_worker(seed + 10000),
            )

        if is_main:
            print(
                f"Training VQ-VAE datasets={sorted(datasets)} seq_len={seq_len} "
                f"latent_len={args.latent_len} codebook={args.codebook_size} "
                f"per_gpu_batch={batch_size} global_batch={batch_size * world_size}"
            )
        model = MotionVQVAE(
            d_in=d_in,
            d_z=d_z,
            num_styles=num_styles,
            max_len=seq_len,
            latent_len=args.latent_len,
            latent_token_mode=args.latent_token_mode,
            codebook_size=args.codebook_size,
            commitment_weight=args.commitment_weight,
            codebook_decay=args.codebook_decay,
        )
        resume_state = None
        if args.resume_ckpt:
            resume_state = torch.load(args.resume_ckpt, map_location=device)
            model.load_state_dict(resume_state["model"])
        model.to(device)
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        model_for_state = model.module if ddp else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        if resume_state is not None and "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])

        if is_main:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if ddp:
            dist.barrier()

        loader_lens = [len(x) for x in (loader_a, loader_b) if x is not None]
        num_steps_base = max(loader_lens) if len(loader_lens) > 1 else loader_lens[0]
        total_steps_est = max(1, args.epochs * num_steps_base)
        start_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
        step = start_epoch * num_steps_base
        best_val = resume_state.get("best_val") if resume_state else None
        best_epoch = resume_state.get("best_epoch") if resume_state else None
        if args.reset_best_val:
            best_val = None
            best_epoch = None
        stale_val_checks = 0

        for epoch in range(start_epoch, args.epochs):
            if is_main:
                print(f"Epoch {epoch + 1}/{args.epochs}")
            model.train()
            if sampler_a is not None:
                sampler_a.set_epoch(epoch)
            if sampler_b is not None:
                sampler_b.set_epoch(epoch)
            if loader_a is not None and loader_b is not None:
                batch_iter = balanced_batch_iter(
                    loader_a, loader_b, args.ratio_aist, args.ratio_mvh
                )
                num_steps = max(len(loader_a), len(loader_b))
            elif loader_a is not None:
                batch_iter = _single_loader_iter(loader_a)
                num_steps = len(loader_a)
            elif loader_b is not None:
                batch_iter = _single_loader_iter(loader_b)
                num_steps = len(loader_b)
            else:
                raise ValueError("No train loaders were created")
            if num_steps == 0:
                raise ValueError("No training steps available; lower batch size.")

            epoch_frac = (epoch + 1) / max(args.epochs, 1)
            w_root_epoch = w_root * (
                w_root_late_factor if epoch_frac >= w_root_late_start else 1.0
            )
            sums = {k: 0.0 for k in (
                "recon", "cont", "contact", "root", "vq", "vel", "acc",
                "joint", "distal", "root_traj", "root_drift", "distal_vel",
                "visible", "style", "total", "perplexity", "codes_used"
            )}
            count = 0
            iter_range = tqdm(range(num_steps), desc="Training", leave=False) if is_main else range(num_steps)
            for step_idx in iter_range:
                t0 = time.perf_counter()
                batches = next(batch_iter)
                t1 = time.perf_counter()
                motion, domain_id, style_id, mask = merge_batches(batches)
                motion = motion.to(device)
                domain_id = domain_id.to(device)
                style_id = style_id.to(device)
                mask = mask.to(device)
                if _ddp_any(ddp, not torch.isfinite(motion).all().item(), device):
                    continue
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                style_id_in = apply_style_dropout(style_id, domain_id, style_dropout_p)
                outputs = model(motion, domain_id, style_id_in, mask=mask)
                x_hat = outputs["x_hat"]
                if _ddp_any(ddp, not torch.isfinite(x_hat).all().item(), device):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                recon, cont_loss, contact_loss, root_loss = grouped_recon_loss(
                    x_hat, motion, mask, w_contact=w_contact, w_root=w_root_epoch
                )
                visible_losses = visible_recon_losses(x_hat, motion, mask, mean_t, std_t)
                visible_loss = (
                    args.w_joint * visible_losses["joint"]
                    + args.w_distal * visible_losses["distal"]
                    + args.w_root_traj * visible_losses["root_xz"]
                    + args.w_root_drift * visible_losses["root_drift"]
                    + args.w_distal_vel * visible_losses["distal_vel"]
                )
                vq_loss = outputs["vq_loss"]
                vel, acc = continuous_smoothness_loss(x_hat, motion, mask)
                style_loss = style_ce_loss(
                    outputs.get("style_logits"), style_id_in, domain_id
                )
                vel_w = _ramp_weight(step, total_steps_est, w_vel, smooth_warmup_frac)
                acc_w = _ramp_weight(step, total_steps_est, w_acc, smooth_warmup_frac)
                loss = recon + visible_loss + vq_loss + vel_w * vel + acc_w * acc
                if style_loss is not None:
                    loss = loss + w_style * style_loss
                if _ddp_any(ddp, not torch.isfinite(loss).item(), device):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t5 = time.perf_counter()

                step += 1
                sums["recon"] += recon.item()
                sums["cont"] += cont_loss.item()
                sums["contact"] += contact_loss.item()
                sums["root"] += root_loss.item()
                sums["vq"] += vq_loss.item()
                sums["vel"] += vel.item()
                sums["acc"] += acc.item()
                sums["joint"] += visible_losses["joint"].item()
                sums["distal"] += visible_losses["distal"].item()
                sums["root_traj"] += visible_losses["root_xz"].item()
                sums["root_drift"] += visible_losses["root_drift"].item()
                sums["distal_vel"] += visible_losses["distal_vel"].item()
                sums["visible"] += visible_loss.item()
                sums["style"] += style_loss.item() if style_loss is not None else 0.0
                sums["total"] += loss.item()
                sums["perplexity"] += outputs["perplexity"].item()
                sums["codes_used"] += outputs["codes_used"].item()
                count += 1

                if args.debug_timing and is_main and step_idx % args.debug_every == 0:
                    print(
                        "timing (s) load={:.4f} to_gpu={:.4f} step={:.4f}".format(
                            t1 - t0, t2 - t1, t5 - t2
                        )
                    )

            if ddp:
                tensor = torch.tensor(
                    [sums[k] for k in sums] + [count],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                for i, k in enumerate(sums):
                    sums[k] = tensor[i].item()
                count = int(tensor[-1].item())

            if is_main:
                denom = max(float(count), 1.0)
                avg = {k: v / denom for k, v in sums.items()}
                print(
                    "Epoch {} loss_total={:.6f} recon={:.6f} cont={:.6f} "
                    "contact={:.6f} root={:.6f} vq={:.6f} vel={:.6f} "
                    "acc={:.6f} visible={:.6f} joint={:.6f} distal={:.6f} "
                    "root_traj={:.6f} root_drift={:.6f} distal_vel={:.6f} "
                    "style={:.6f} perplexity={:.2f} codes_used={:.1f}".format(
                        epoch + 1,
                        avg["total"],
                        avg["recon"],
                        avg["cont"],
                        avg["contact"],
                        avg["root"],
                        avg["vq"],
                        avg["vel"],
                        avg["acc"],
                        avg["visible"],
                        avg["joint"],
                        avg["distal"],
                        avg["root_traj"],
                        avg["root_drift"],
                        avg["distal_vel"],
                        avg["style"],
                        avg["perplexity"],
                        avg["codes_used"],
                    )
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "epoch": epoch + 1,
                        "loss/total": avg["total"],
                        "loss/recon": avg["recon"],
                        "loss/cont": avg["cont"],
                        "loss/contact": avg["contact"],
                        "loss/root": avg["root"],
                        "loss/vq": avg["vq"],
                        "loss/vel": avg["vel"],
                        "loss/acc": avg["acc"],
                        "loss/visible": avg["visible"],
                        "loss/joint": avg["joint"],
                        "loss/distal": avg["distal"],
                        "loss/root_traj": avg["root_traj"],
                        "loss/root_drift": avg["root_drift"],
                        "loss/distal_vel": avg["distal_vel"],
                        "loss/style": avg["style"],
                        "vq/perplexity": avg["perplexity"],
                        "vq/codes_used": avg["codes_used"],
                    }, commit=True)

            save_ckpt = val_every_epochs and (epoch + 1) % val_every_epochs == 0
            stop_training = False
            if save_ckpt and is_main:
                print("Running validation")
                model_for_state.eval()
                val_loaders = []
                fixed_aist_loader = None
                if val_use_aist:
                    aist_val_paths = aist_split_paths(aist_dir, config["aist_split_val"])
                    val_a = AISTDataset(
                        aist_dir,
                        genre_to_id,
                        seq_len,
                        mean=mean,
                        std=std,
                        files=aist_val_paths,
                        cache_root=cache_root,
                        target_fps=target_fps,
                        src_fps=aist_fps,
                        crop_mode=args.aist_val_crop_mode,
                        clip_repeat=args.aist_val_clip_repeat,
                    )
                    val_loader_a = DataLoader(
                        val_a,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=persistent_workers,
                        prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    )
                    val_loaders.append(val_loader_a)
                    if args.val_joint_metric_samples > 0:
                        fixed_aist_loader = DataLoader(
                            val_a,
                            batch_size=min(eval_batch_size, args.val_joint_metric_samples),
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False,
                        )
                if val_use_mvh:
                    mvh_val_dirs = read_lines(config["mvh_split_val"])
                    val_b = MVHumanNetDataset(
                        mv_root,
                        seq_len,
                        mean=mean,
                        std=std,
                        sequence_dirs=mvh_val_dirs,
                        cache_root=cache_root,
                        target_fps=target_fps,
                        src_fps=mvh_fps,
                    )
                    val_loaders.append(
                        DataLoader(
                            val_b,
                            batch_size=eval_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor if num_workers > 0 else None,
                        )
                    )
                val_sums = {k: 0.0 for k in ("recon", "vq", "perplexity", "codes_used")}
                val_count = 0
                with torch.no_grad():
                    for loader in val_loaders:
                        for batch in loader:
                            motion = batch["motion"].to(device)
                            domain_id = batch["domain_id"].to(device)
                            style_id = batch["style_id"].to(device)
                            mask = batch["mask"].to(device)
                            outputs = model_for_state(
                                motion,
                                domain_id,
                                style_id,
                                mask=mask,
                                update_codebook=False,
                            )
                            v_recon, _, _, _ = grouped_recon_loss(
                                outputs["x_hat"],
                                motion,
                                mask,
                                w_contact=w_contact,
                                w_root=w_root,
                            )
                            val_sums["recon"] += v_recon.item()
                            val_sums["vq"] += outputs["vq_loss"].item()
                            val_sums["perplexity"] += outputs["perplexity"].item()
                            val_sums["codes_used"] += outputs["codes_used"].item()
                            val_count += 1
                val_avg = {
                    f"val/{k}": v / max(val_count, 1) for k, v in val_sums.items()
                }
                print(
                    "Validation recon={:.6f} vq={:.6f} perplexity={:.2f} codes_used={:.1f}".format(
                        val_avg["val/recon"],
                        val_avg["val/vq"],
                        val_avg["val/perplexity"],
                        val_avg["val/codes_used"],
                    )
                )
                joint_metrics = _fixed_aist_joint_metrics(
                    model_for_state,
                    fixed_aist_loader,
                    device,
                    mean,
                    std,
                    args.val_joint_metric_samples,
                    target_fps,
                )
                if joint_metrics:
                    print(
                        "Fixed AIST joint recon "
                        + " ".join(
                            f"{key}={value:.6f}"
                            if isinstance(value, float)
                            else f"{key}={value}"
                            for key, value in joint_metrics.items()
                        )
                    )
                all_val_metrics = {**val_avg, **joint_metrics}
                current_best_metric_name = args.best_metric
                current_best_metric = all_val_metrics.get(current_best_metric_name)
                if current_best_metric is None:
                    current_best_metric_name = "val/recon"
                    current_best_metric = val_avg["val/recon"]
                    print(
                        f"Requested best metric {args.best_metric!r} was not available; "
                        "falling back to val/recon."
                    )
                if isinstance(current_best_metric, (int, float)):
                    print(
                        f"Checkpoint metric {current_best_metric_name}="
                        f"{float(current_best_metric):.6f}"
                    )
                if wandb_run is not None:
                    wandb_run.log({
                        "epoch": epoch + 1,
                        **val_avg,
                        **joint_metrics,
                    }, commit=True)

                latest_path = os.path.join(args.checkpoint_dir, "motion_vqvae_latest.pt")
                ckpt_state = {
                    "model": model_for_state.state_dict(),
                    "genre_to_id": genre_to_id,
                    "config": vars(args),
                    "epoch": epoch + 1,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "best_metric": current_best_metric_name,
                    "best_metric_value": best_val,
                    "optimizer": optimizer.state_dict(),
                    "stats_path": stats_path,
                    "selected_datasets": sorted(datasets),
                    "selected_val_datasets": sorted(val_datasets),
                }
                torch.save(ckpt_state, latest_path)
                print(f"Saved checkpoint: {latest_path}")
                improved = val_count > 0 and (
                    best_val is None or float(current_best_metric) < best_val
                )
                if improved:
                    best_val = float(current_best_metric)
                    best_epoch = epoch + 1
                    stale_val_checks = 0
                    best_path = os.path.join(
                        args.checkpoint_dir, "motion_vqvae_best.pt"
                    )
                    best_state = dict(ckpt_state)
                    best_state["best_val"] = best_val
                    best_state["best_epoch"] = best_epoch
                    best_state["best_metric"] = current_best_metric_name
                    best_state["best_metric_value"] = best_val
                    best_state["epoch"] = best_epoch
                    torch.save(best_state, best_path)
                    print(
                        f"Saved best checkpoint: {best_path} "
                        f"(epoch {best_epoch}, {current_best_metric_name}={best_val:.6f})"
                    )
                else:
                    stale_val_checks += 1
                    if args.early_stop_patience > 0:
                        print(
                            "Validation did not improve "
                            f"({stale_val_checks}/{args.early_stop_patience}); "
                            f"best {current_best_metric_name}={best_val:.6f} "
                            f"at epoch {best_epoch}"
                        )
                if (
                    args.archive_every_epochs > 0
                    and (epoch + 1) % args.archive_every_epochs == 0
                ):
                    archive_state = dict(ckpt_state)
                    archive_state["best_val"] = best_val
                    archive_state["best_epoch"] = best_epoch
                    archive_state["best_metric"] = current_best_metric_name
                    archive_state["best_metric_value"] = best_val
                    archive_path = os.path.join(
                        args.checkpoint_dir,
                        f"motion_vqvae_epoch{epoch + 1}.pt",
                    )
                    torch.save(archive_state, archive_path)
                    print(f"Saved archive checkpoint: {archive_path}")
                model_for_state.train()
                if (
                    args.early_stop_patience > 0
                    and stale_val_checks >= args.early_stop_patience
                    and epoch + 1 >= args.early_stop_min_epochs
                ):
                    print(
                        "Early stopping after "
                        f"{stale_val_checks} stale validation checks."
                    )
                    stop_training = True
            if is_main and not save_ckpt:
                latest_path = os.path.join(args.checkpoint_dir, "motion_vqvae_latest.pt")
                ckpt_state = {
                    "model": model_for_state.state_dict(),
                    "genre_to_id": genre_to_id,
                    "config": vars(args),
                    "epoch": epoch + 1,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_val,
                    "optimizer": optimizer.state_dict(),
                    "stats_path": stats_path,
                    "selected_datasets": sorted(datasets),
                    "selected_val_datasets": sorted(val_datasets),
                }
                torch.save(ckpt_state, latest_path)
                print(f"Saved checkpoint: {latest_path}")
            if ddp:
                dist.barrier()
            if _ddp_any(ddp, stop_training, device):
                break

        if is_main and wandb_run is not None:
            wandb_run.finish()
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
