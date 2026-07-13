import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.teacher import EMA, Teacher
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.backend import (
    decode_motion_latent,
    encode_motion_latent,
    load_vae_backend,
)
from flowmimic.src.model.vae.stats import load_mean_std
from flowmimic.src.model.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.motion.ik.common.quaternion import qinv, qrot
from flowmimic.src.data.openpose import (
    compute_openpose_stats,
    load_aist_openpose,
    load_mvh_openpose,
)
from flowmimic.src.metrics import T2MMotionFeatureExtractor
from flowmimic.scripts.eval_flow import _build_smpl22_to_body25, evaluate_dataset


WANDB_EVAL_METRIC_KEYS = (
    "fid",
    "mmdist",
    "diversity",
    "accel_p90",
    "jerk_p90",
    "e2d_slack",
    "skate_mean",
)


def _apply_train_flow_config_defaults(args, config):
    flow_cfg = config.get("flow", {})
    flow_eval_cfg = flow_cfg.get("eval", {})
    flow_reg_cfg = flow_cfg.get("regularization", {})
    flow_ckpt_cfg = flow_cfg.get("checkpointing", {})

    args.stats_path = config.get("stats_path")
    args.openpose_stats_path = config.get(
        "openpose_stats_path", "data/openpose_stats.npz"
    )
    if args.latent_stats_path is None:
        args.latent_stats_path = config.get(
            "latent_stats_path", "data/latent_stats.npz"
        )

    args.vae_latent_len = flow_cfg.get("vae_latent_len")
    args.vae_latent_token_mode = flow_cfg.get("vae_latent_token_mode")
    args.eval_cond_frames = flow_eval_cfg.get("cond_frames")

    if args.cond_frames_min is None:
        args.cond_frames_min = flow_cfg.get("cond_frames_min")
    if args.cond_frames_max is None:
        args.cond_frames_max = flow_cfg.get("cond_frames_max")
    if args.cond_drop_prob is None:
        args.cond_drop_prob = flow_cfg.get("cond_drop_prob")
    if args.cond_frame_drop_prob is None:
        args.cond_frame_drop_prob = flow_cfg.get("cond_frame_drop_prob", 0.0)
    if args.cond_frame_drop_ramp_epochs is None:
        args.cond_frame_drop_ramp_epochs = int(
            flow_cfg.get("cond_frame_drop_ramp_epochs", 0) or 0
        )
    if args.cond_frame_drop_mode is None:
        args.cond_frame_drop_mode = flow_cfg.get("cond_frame_drop_mode", "random")
    args.cond_frame_drop_max_block_frac = float(
        flow_cfg.get("cond_frame_drop_max_block_frac", 0.25)
    )
    if args.ema_decay is None:
        args.ema_decay = flow_cfg.get("ema_decay")
    if args.lr is None:
        args.lr = flow_cfg.get("lr")
    if args.num_workers is None:
        args.num_workers = int(config.get("num_workers", 10))
    if args.eval_guidance_scale is None:
        args.eval_guidance_scale = float(flow_eval_cfg.get("guidance_scale", 1.0))
    if args.cfg_drop_prob is None:
        args.cfg_drop_prob = float(flow_cfg.get("cfg_drop_prob", 0.0))
    if args.cfg_start_epoch is None:
        args.cfg_start_epoch = int(flow_cfg.get("cfg_start_epoch", 0))
    if args.cond_frame_drop_start_epoch is None:
        args.cond_frame_drop_start_epoch = int(
            flow_cfg.get("cond_frame_drop_start_epoch", 0)
        )
    if args.lr_decay_epoch is None:
        args.lr_decay_epoch = flow_cfg.get("lr_decay_epoch")
    if args.solver_cond_start_epoch is None:
        args.solver_cond_start_epoch = int(flow_cfg.get("solver_cond_start_epoch", 30))
    if args.solver_smooth_start_epoch is None:
        args.solver_smooth_start_epoch = int(
            flow_cfg.get("solver_smooth_start_epoch", 40)
        )
    if args.lambda_cond is None:
        args.lambda_cond = float(flow_cfg.get("lambda_cond", 1e-3))
    if args.lambda_acc is None:
        args.lambda_acc = float(flow_cfg.get("lambda_acc", 5e-2))
    if args.lambda_jerk is None:
        args.lambda_jerk = float(flow_cfg.get("lambda_jerk", 5e-4))
    if args.cond_match_camera_mode is None:
        args.cond_match_camera_mode = flow_reg_cfg.get(
            "cond_match_camera_mode",
            flow_cfg.get("cond_match_camera_mode", "per_frame"),
        )
    args.cond_match_min_conf = float(
        flow_reg_cfg.get(
            "cond_match_min_conf",
            flow_cfg.get("cond_match_min_conf", 0.4),
        )
    )
    args.cond_match_min_joints = int(
        flow_reg_cfg.get(
            "cond_match_min_joints",
            flow_cfg.get("cond_match_min_joints", 6),
        )
    )
    if args.solver_reg_subbatch_size is None:
        args.solver_reg_subbatch_size = int(
            flow_cfg.get("solver_reg_subbatch_size", 0) or 0
        )
    if args.smooth_every is None:
        args.smooth_every = flow_reg_cfg.get(
            "smooth_every", flow_cfg.get("smooth_every")
        )

    args.teacher_steps = int(flow_cfg.get("teacher_steps", 16))
    args.teacher_solver = flow_cfg.get("teacher_solver", "heun")
    args.teacher_mode = flow_cfg.get("teacher_mode", "strict")
    args.p_teacher = float(flow_cfg.get("p_teacher", 1.0))
    args.eval_use_ema = bool(flow_eval_cfg.get("use_ema", True))
    args.save_every_steps = int(flow_ckpt_cfg.get("save_every_steps", 100) or 0)
    args.lr_scale_mode = flow_cfg.get("lr_scale_mode", "none")
    args.max_bad_steps = int(flow_cfg.get("max_bad_steps", 50) or 0)
    args.cond_lr_scale = float(flow_cfg.get("cond_lr_scale", 0.1))
    args.save_every_epochs = int(flow_ckpt_cfg.get("save_every_epochs", 1) or 0)

    args.solver_every = int(flow_cfg.get("solver_every", 8))
    args.solver_method = flow_cfg.get("solver_method", "euler")
    args.solver_cond_ramp_epochs = int(flow_cfg.get("solver_cond_ramp_epochs", 10))
    args.solver_smooth_ramp_epochs = int(flow_cfg.get("solver_smooth_ramp_epochs", 10))
    args.smooth_loss_domain = flow_cfg.get("smooth_loss_domain", "joints")
    args.smooth_subbatch_size = int(flow_cfg.get("smooth_subbatch_size", 32))
    args.reg_decode_batch_size = int(flow_cfg.get("reg_decode_batch_size", 128))
    args.reg_decode_checkpoint = bool(flow_cfg.get("reg_decode_checkpoint", True))
    args.solver_checkpoint = bool(flow_cfg.get("solver_checkpoint", True))
    args.cond_every = flow_reg_cfg.get("cond_every", flow_cfg.get("cond_every"))
    args.solver_steps_early = flow_cfg.get("solver_steps_early", "16")
    args.solver_steps_mid = flow_cfg.get("solver_steps_mid", "8,16")
    args.solver_steps_late = flow_cfg.get("solver_steps_late", "4,8,2")
    args.solver_mid_epoch = int(flow_cfg.get("solver_mid_epoch", 80))
    args.solver_late_epoch = int(flow_cfg.get("solver_late_epoch", 160))

    args.eval_t2m_motion_encoder_ckpt = config.get("t2m_motion_encoder_ckpt")
    args.eval_t2m_mean_path = config.get("t2m_eval_mean_path")
    args.eval_t2m_std_path = config.get("t2m_eval_std_path")
    args.eval_num_samples = 0
    args.eval_batch_size = int(
        flow_eval_cfg.get("batch_size", config.get("eval_batch_size", 32))
    )
    args.eval_diversity_times = int(flow_eval_cfg.get("diversity_times", 300))
    args.eval_multimodality_repeats = int(flow_eval_cfg.get("multimodality_repeats", 1))
    args.eval_multimodality_times = int(flow_eval_cfg.get("multimodality_times", 20))
    args.async_eval_cpu_threads = int(flow_eval_cfg.get("async_eval_cpu_threads", 8))
    args.async_eval_nice = int(flow_eval_cfg.get("async_eval_nice", 10))


def main():
    config = load_config()
    flow_cfg = config.get("flow", {})
    flow_eval_cfg = flow_cfg.get("eval", {})
    flow_wandb_cfg = flow_cfg.get("wandb", {})

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--datasets", type=str, default="AIST")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--latent-stats-path", type=str, default=None)
    parser.add_argument("--vae-ckpt", type=str, default=None)
    parser.add_argument(
        "--vae-type",
        choices=("auto", "motion_vae", "motion_vqvae"),
        default="auto",
    )
    parser.add_argument(
        "--aist-crop-mode", choices=("first", "random", "uniform"), default="first"
    )
    parser.add_argument("--aist-clip-repeat", type=int, default=1)
    parser.add_argument("--cond-frames-min", type=int, default=None)
    parser.add_argument("--cond-frames-max", type=int, default=None)
    parser.add_argument("--cond-drop-prob", type=float, default=None)
    parser.add_argument("--cond-frame-drop-prob", type=float, default=None)
    parser.add_argument(
        "--cond-frame-drop-start-epoch",
        type=int,
        default=None,
        help="1-based epoch to enable sequence-frame condition masking. 0 enables it from the start.",
    )
    parser.add_argument(
        "--cond-frame-drop-ramp-epochs",
        type=int,
        default=None,
        help="Linearly ramp sequence-frame condition masking over this many epochs.",
    )
    parser.add_argument(
        "--cond-frame-drop-mode",
        choices=("random", "block", "mixed"),
        default=None,
        help="How to select sequence condition frames to mask.",
    )
    parser.add_argument("--cfg-drop-prob", type=float, default=None)
    parser.add_argument(
        "--cfg-start-epoch",
        type=int,
        default=None,
        help="1-based epoch to enable sample-level full-condition CFG dropout. 0 enables it from the start.",
    )
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--eval-guidance-scale", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--reflow-round", type=int, default=0)
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--use-ema-teacher", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/flow")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--lr-decay-epoch",
        type=int,
        default=None,
        help="Epoch at which to apply the one-time 0.5 LR decay; defaults to epochs//2.",
    )
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--solver-cond-start-epoch", type=int, default=None)
    parser.add_argument("--solver-smooth-start-epoch", type=int, default=None)
    parser.add_argument("--lambda-cond", type=float, default=None)
    parser.add_argument("--lambda-acc", type=float, default=None)
    parser.add_argument("--lambda-jerk", type=float, default=None)
    parser.add_argument(
        "--cond-match-camera-mode",
        choices=("per_frame", "shared"),
        default=None,
        help="Camera alignment used by the differentiable condition matching loss.",
    )
    parser.add_argument(
        "--solver-reg-subbatch-size",
        type=int,
        default=None,
        help=(
            "If >0, run differentiable solver/decode regularizers on a random "
            "subbatch of each full training batch. The base CFM loss still uses "
            "the full batch."
        ),
    )
    parser.add_argument("--smooth-every", type=int, default=None)
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=flow_wandb_cfg.get("project", "FlowMimic"),
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=flow_wandb_cfg.get("entity")
    )
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=flow_wandb_cfg.get("group"))
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default=flow_wandb_cfg.get("resume", "allow"),
        choices=("allow", "must", "never"),
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=flow_wandb_cfg.get("mode", "online"),
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument(
        "--eval-steps", type=str, default=flow_eval_cfg.get("steps", "16,50")
    )
    parser.add_argument(
        "--eval-every-epochs", type=int, default=flow_eval_cfg.get("every_epochs")
    )
    parser.add_argument(
        "--eval-aist-splits",
        type=str,
        default=flow_eval_cfg.get("aist_splits", "test"),
    )
    parser.add_argument(
        "--eval-aist-cameras",
        type=str,
        default=flow_eval_cfg.get("aist_cameras", "01"),
    )
    parser.add_argument(
        "--eval-aist-crop-mode",
        choices=("first", "random", "uniform"),
        default=flow_eval_cfg.get("aist_crop_mode", "first"),
    )
    parser.add_argument(
        "--eval-replications",
        type=int,
        default=flow_eval_cfg.get("replications", 3),
    )
    parser.add_argument("--eval-no-dist", action="store_true")
    parser.add_argument(
        "--async-cpu-eval",
        action="store_true",
        default=flow_eval_cfg.get("async_cpu_eval", False),
    )
    parser.add_argument("--async-eval-log-dir", type=str, default=None)
    args = parser.parse_args()
    _apply_train_flow_config_defaults(args, config)

    env_world_size = int(os.environ.get("WORLD_SIZE") or "1")
    ddp = args.ddp or env_world_size > 1
    if ddp:
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

    def _debug_log(msg, epoch_idx=None):
        if not args.debug or not is_main:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_val = epoch_idx if epoch_idx is not None else "-"
        with open("debug.log", "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] [epoch {epoch_val}] {msg}\n")

    if is_main:
        print("Loaded config")
    datasets = _parse_dataset_names(args.datasets)
    use_aist = "AIST" in datasets
    use_mvh = "MVH" in datasets
    wandb_run = None
    if is_main:
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
                run_name = f"flow-{stamp}"
            run_group = args.wandb_group or "Flow"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                group=run_group,
                id=args.wandb_id,
                resume=args.wandb_resume if args.wandb_id else None,
                tags=tags or None,
                mode=args.wandb_mode,
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len or config.get("seq_len"),
                    "lr": args.lr,
                    "lr_decay_epoch": args.lr_decay_epoch,
                    "datasets": sorted(datasets),
                    "stats_path": args.stats_path or config.get("stats_path"),
                    "latent_stats_path": (
                        args.latent_stats_path or config.get("latent_stats_path")
                    ),
                    "openpose_stats_path": (
                        args.openpose_stats_path or config.get("openpose_stats_path")
                    ),
                    "vae_ckpt": args.vae_ckpt or config.get("vae_ckpt"),
                    "vae_type": args.vae_type,
                    "eval_steps": args.eval_steps,
                    "eval_guidance_scale": args.eval_guidance_scale,
                    "cond_lr_scale": args.cond_lr_scale,
                    "reflow_round": args.reflow_round,
                    "teacher_mode": args.teacher_mode,
                    "p_teacher": args.p_teacher,
                    "cond_frames_min": args.cond_frames_min
                    if args.cond_frames_min is not None
                    else config.get("flow", {}).get("cond_frames_min"),
                    "cond_frames_max": args.cond_frames_max
                    if args.cond_frames_max is not None
                    else config.get("flow", {}).get("cond_frames_max"),
                    "cond_drop_prob": args.cond_drop_prob
                    if args.cond_drop_prob is not None
                    else config.get("flow", {}).get("cond_drop_prob"),
                    "cond_frame_drop_prob": args.cond_frame_drop_prob
                    if args.cond_frame_drop_prob is not None
                    else config.get("flow", {}).get("cond_frame_drop_prob", 0.0),
                    "cond_frame_drop_start_epoch": args.cond_frame_drop_start_epoch,
                    "cond_frame_drop_ramp_epochs": args.cond_frame_drop_ramp_epochs,
                    "cond_frame_drop_mode": args.cond_frame_drop_mode,
                    "cond_frame_drop_max_block_frac": (
                        args.cond_frame_drop_max_block_frac
                    ),
                    "cfg_drop_prob": args.cfg_drop_prob,
                    "cfg_start_epoch": args.cfg_start_epoch,
                    "ema_decay": args.ema_decay
                    if args.ema_decay is not None
                    else config.get("flow", {}).get("ema_decay"),
                    "vae_latent_len": args.vae_latent_len,
                    "aist_clip_repeat": args.aist_clip_repeat,
                    "solver_reg_subbatch_size": args.solver_reg_subbatch_size,
                    "cond_match_camera_mode": args.cond_match_camera_mode,
                    "cond_match_min_conf": args.cond_match_min_conf,
                    "cond_match_min_joints": args.cond_match_min_joints,
                },
            )
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    seq_len = args.seq_len or config["seq_len"]
    stats_path = args.stats_path or config["stats_path"]
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)
    openpose_aist_dir = config.get(
        "aist_openpose_dir", "data/AIST++/Annotations/openpose"
    )
    openpose_mvh_root = config.get("mvh_openpose_root", "data/MVHumanNet")
    mvh_cameras = config.get("mvh_cameras", ["22327091", "22327113", "22327084"])
    aist_cameras = config.get("aist_cameras", ["01", "02", "08", "09"])
    openpose_stats_path = args.openpose_stats_path or config.get(
        "openpose_stats_path", "data/openpose_stats.npz"
    )
    cond_cache_root = config.get("cond_cache_root", "data/cached_cond")
    latent_stats_path = args.latent_stats_path or config.get(
        "latent_stats_path", "data/latent_stats.npz"
    )

    if is_main:
        print("Loading 263D stats")
    mean, std = load_mean_std(stats_path)

    genre_to_id = build_genre_to_id(config.get("aist_genres", []))

    flow_cfg = config.get("flow", {})
    lr = args.lr or flow_cfg.get("lr", 2e-4)
    weight_decay = flow_cfg.get("weight_decay", 1e-2)
    cond_lr_scale = args.cond_lr_scale
    teacher_steps = args.teacher_steps or flow_cfg.get("teacher_steps", 16)
    teacher_solver = args.teacher_solver or flow_cfg.get("teacher_solver", "heun")
    teacher_mode = args.teacher_mode or flow_cfg.get("teacher_mode", "strict")
    p_teacher = (
        args.p_teacher if args.p_teacher is not None else flow_cfg.get("p_teacher", 1.0)
    )
    ema_decay = (
        args.ema_decay
        if args.ema_decay is not None
        else flow_cfg.get("ema_decay", 0.999)
    )
    cond_frames_min = (
        args.cond_frames_min
        if args.cond_frames_min is not None
        else flow_cfg.get("cond_frames_min", 2)
    )
    cond_frames_max = (
        args.cond_frames_max
        if args.cond_frames_max is not None
        else flow_cfg.get("cond_frames_max", 10)
    )
    eval_cond_frames = args.eval_cond_frames or cond_frames_min
    cond_drop_prob = (
        args.cond_drop_prob
        if args.cond_drop_prob is not None
        else flow_cfg.get("cond_drop_prob", 0.2)
    )
    cond_frame_drop_prob = (
        args.cond_frame_drop_prob
        if args.cond_frame_drop_prob is not None
        else flow_cfg.get("cond_frame_drop_prob", 0.0)
    )
    solver_every = max(1, args.solver_every)
    solver_method = args.solver_method
    solver_cond_start_epoch = args.solver_cond_start_epoch
    solver_cond_ramp_epochs = args.solver_cond_ramp_epochs
    solver_smooth_start_epoch = args.solver_smooth_start_epoch
    solver_smooth_ramp_epochs = args.solver_smooth_ramp_epochs
    lambda_cond = args.lambda_cond
    lambda_acc = args.lambda_acc
    lambda_jerk = args.lambda_jerk
    solver_reg_subbatch_size = max(0, int(args.solver_reg_subbatch_size or 0))
    cond_every = max(1, args.cond_every or solver_every)
    smooth_every = max(
        1,
        args.smooth_every
        or (1 if args.smooth_loss_domain == "latent_est" else solver_every),
    )
    solver_steps_early = _parse_steps(args.solver_steps_early)
    solver_steps_mid = _parse_steps(args.solver_steps_mid)
    solver_steps_late = _parse_steps(args.solver_steps_late)
    grad_clip_norm = config.get("grad_clip_norm", 1.0)
    save_every_steps = args.save_every_steps or flow_cfg.get("save_every_steps", 0)
    save_every_epochs = args.save_every_epochs
    max_bad_steps = args.max_bad_steps
    seed = config.get("seed", 42)
    seed_rank = seed + rank
    random.seed(seed_rank)
    np.random.seed(seed_rank)
    torch.manual_seed(seed_rank)
    torch.cuda.manual_seed_all(seed_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    aist_train_paths = (
        _aist_split_paths(aist_dir, config["aist_split_train"]) if use_aist else []
    )
    mvh_train_dirs = _read_lines(config["mvh_split_train"]) if use_mvh else []
    dataset_a = None
    dataset_b = None
    if use_aist:
        if is_main:
            print(f"Building datasets -- AIST++ (train split: {len(aist_train_paths)})")
        dataset_a = AISTDataset(
            aist_dir,
            genre_to_id=genre_to_id,
            seq_len=seq_len,
            mean=mean,
            std=std,
            files=aist_train_paths,
            cache_root=config["cache_root"],
            target_fps=target_fps,
            src_fps=aist_fps,
            camera_ids=aist_cameras,
            expand_cameras=True,
            include_cond=True,
            openpose_dir=openpose_aist_dir,
            cond_cache_root=cond_cache_root,
            cond_frames_min=cond_frames_min,
            cond_frames_max=cond_frames_max,
            cond_drop_prob=cond_drop_prob,
            cond_frame_drop_prob=cond_frame_drop_prob,
            cond_frame_drop_mode=args.cond_frame_drop_mode,
            cond_frame_drop_max_block_frac=args.cond_frame_drop_max_block_frac,
            crop_mode=args.aist_crop_mode,
            clip_repeat=args.aist_clip_repeat,
        )
    if use_mvh:
        if is_main:
            print(f"Building datasets -- MVH (train split: {len(mvh_train_dirs)})")
        dataset_b = MVHumanNetDataset(
            mv_root,
            seq_len=seq_len,
            mean=mean,
            std=std,
            sequence_dirs=mvh_train_dirs,
            cache_root=config["cache_root"],
            target_fps=target_fps,
            src_fps=mvh_fps,
            camera_ids=mvh_cameras,
            expand_cameras=True,
            include_cond=True,
            openpose_root=openpose_mvh_root,
            cond_cache_root=cond_cache_root,
            cond_frames_min=cond_frames_min,
            cond_frames_max=cond_frames_max,
            cond_drop_prob=cond_drop_prob,
            cond_frame_drop_prob=cond_frame_drop_prob,
            cond_frame_drop_mode=args.cond_frame_drop_mode,
            cond_frame_drop_max_block_frac=args.cond_frame_drop_max_block_frac,
        )

    if is_main:
        print("Building dataloaders")
    sampler_a = None
    sampler_b = None
    if ddp:
        if dataset_a is not None:
            sampler_a = DistributedSampler(dataset_a, shuffle=True, drop_last=True)
        if dataset_b is not None:
            sampler_b = DistributedSampler(dataset_b, shuffle=True, drop_last=True)

    def _seed_worker(worker_id):
        worker_seed = seed_rank + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_a = None
    loader_b = None
    if dataset_a is not None:
        loader_a = DataLoader(
            dataset_a,
            batch_size=args.batch_size,
            shuffle=(sampler_a is None),
            drop_last=True,
            num_workers=args.num_workers,
            sampler=sampler_a,
            worker_init_fn=_seed_worker,
        )
    if dataset_b is not None:
        loader_b = DataLoader(
            dataset_b,
            batch_size=args.batch_size,
            shuffle=(sampler_b is None),
            drop_last=True,
            num_workers=args.num_workers,
            sampler=sampler_b,
            worker_init_fn=_seed_worker,
        )

    if is_main:
        print("Loading VAE checkpoint")
    vae_ckpt_path = args.vae_ckpt or config.get(
        "vae_ckpt", "checkpoints/vae/len200/motion_vae_best.pt"
    )
    vae_backend = load_vae_backend(
        vae_ckpt_path,
        config,
        device,
        seq_len=seq_len,
        vae_type=args.vae_type,
        latent_len=args.vae_latent_len,
        latent_token_mode=args.vae_latent_token_mode,
    )
    vae = vae_backend.model
    for p in vae.parameters():
        p.requires_grad = False
    latent_len = vae_backend.latent_len
    flow_d_z = vae_backend.d_z
    if is_main:
        print(
            f"Loaded {vae_backend.vae_type}: latent_len={latent_len}, "
            f"d_z={flow_d_z}, max_len={vae_backend.max_len}"
        )

    if is_main:
        print("Building flow model")
    flow = ConditionalRectFlow(
        d_z=flow_d_z,
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
    flow.to(device)
    resume_state = None
    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        resume_state = torch.load(args.resume, map_location=device)
        flow.load_state_dict(resume_state["model"])

    if ddp:
        flow = torch.nn.parallel.DistributedDataParallel(
            flow,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    flow_model = flow.module if ddp else flow

    if ddp and args.lr_scale_mode == "linear":
        lr = lr * world_size
    lr_decay_epoch = max(1, args.lr_decay_epoch or (args.epochs // 2))
    if is_main:
        print(f"LR={lr} (mode={args.lr_scale_mode}, world_size={world_size})")
        print(
            f"Per-GPU batch={args.batch_size}, global batch={args.batch_size * world_size}"
        )
        print(
            f"Cond LR scale={cond_lr_scale}; cond_frames={cond_frames_min}-{cond_frames_max}; "
            f"eval_cond_frames={eval_cond_frames}; cond_frame_drop_prob={cond_frame_drop_prob}; "
            f"cond_frame_drop_start_epoch={args.cond_frame_drop_start_epoch}; "
            f"cond_frame_drop_ramp_epochs={args.cond_frame_drop_ramp_epochs}; "
            f"cond_frame_drop_mode={args.cond_frame_drop_mode}; "
            f"cfg_drop_prob={args.cfg_drop_prob}; cfg_start_epoch={args.cfg_start_epoch}; "
            f"ema_decay={ema_decay}; "
            f"lr_decay_epoch={lr_decay_epoch}; "
            f"solver_reg_subbatch_size={solver_reg_subbatch_size or 'full'}; "
            f"cond_match_camera_mode={args.cond_match_camera_mode}"
        )

    cond_params = list(flow_model.cond_encoder.parameters()) + list(
        flow_model.cond_mlp.parameters()
    )
    cond_param_ids = {id(p) for p in cond_params}
    other_params = [p for p in flow_model.parameters() if id(p) not in cond_param_ids]
    base_group_lrs = [lr, lr * cond_lr_scale]
    param_groups = [
        {"params": other_params, "lr": lr},
        {"params": cond_params, "lr": lr * cond_lr_scale},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    start_epoch = 0
    ema_state = None
    lr_halved = False
    if resume_state is not None:
        if not args.reset_optimizer and "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if args.reset_optimizer and is_main:
            print("Resetting optimizer state for resume")
        if args.use_ema_teacher and "ema" in resume_state:
            ema_state = resume_state["ema"]
        start_epoch = int(resume_state.get("epoch", 0))
        lr_halved = bool(resume_state.get("lr_halved", start_epoch >= lr_decay_epoch))
        scheduled_lrs = [
            group_lr * (0.5 if lr_halved else 1.0) for group_lr in base_group_lrs
        ]
        old_lrs = [group["lr"] for group in optimizer.param_groups]
        _set_optimizer_group_lrs(optimizer, scheduled_lrs)
        new_lrs = [group["lr"] for group in optimizer.param_groups]
        if is_main and any(abs(a - b) > 1e-12 for a, b in zip(old_lrs, new_lrs)):
            print(f"Adjusted resume LR to scheduled values: {new_lrs}")

    if is_main:
        print("Loading OpenPose stats")
    if not os.path.exists(openpose_stats_path):
        if is_main:
            compute_openpose_stats(
                aist_paths=aist_train_paths,
                mvh_dirs=mvh_train_dirs,
                aist_openpose_dir=openpose_aist_dir,
                mvh_openpose_root=openpose_mvh_root,
                mv_root=mv_root,
                cameras=mvh_cameras,
                target_fps=target_fps,
                aist_fps=aist_fps,
                mvh_fps=mvh_fps,
                out_path=openpose_stats_path,
                cache_root=cond_cache_root,
                aist_cameras=aist_cameras,
                mvh_cameras=mvh_cameras,
            )
        if ddp:
            dist.barrier()
    stats = np.load(openpose_stats_path)
    k2d_mean = stats["mean"]
    k2d_std = stats["std"]
    if not np.isfinite(k2d_mean).all() or not np.isfinite(k2d_std).all():
        raise ValueError("OpenPose mean/std contain non-finite values")

    latent_mean = None
    latent_std = None
    if os.path.exists(latent_stats_path):
        latent_stats = np.load(latent_stats_path)
        latent_mean = torch.tensor(
            latent_stats["mean"], device=device, dtype=torch.float32
        )
        latent_std = torch.tensor(
            latent_stats["std"], device=device, dtype=torch.float32
        )
    body25_to_smpl_idx, body25_to_smpl_valid = _build_body25_to_smpl22(
        config["smpl45_to_body25_def"], device
    )

    eval_compute_dist = False
    eval_t2m_extractor = None
    eval_t2m_stats = None
    eval_t2m_ckpt = args.eval_t2m_motion_encoder_ckpt or config.get(
        "t2m_motion_encoder_ckpt"
    )
    eval_t2m_mean_path = args.eval_t2m_mean_path or config.get("t2m_eval_mean_path")
    eval_t2m_std_path = args.eval_t2m_std_path or config.get("t2m_eval_std_path")
    if is_main and (not args.eval_no_dist) and eval_t2m_ckpt and args.async_cpu_eval:
        if not eval_t2m_mean_path or not eval_t2m_std_path:
            raise ValueError(
                "Async eval T2M metrics require mean/std paths. "
                "Set evaluator.t2m_eval_mean_path and evaluator.t2m_eval_std_path "
                "in config.json."
            )
        if not os.path.exists(eval_t2m_mean_path) or not os.path.exists(
            eval_t2m_std_path
        ):
            raise FileNotFoundError(
                "Eval T2M mean/std not found: "
                f"mean={eval_t2m_mean_path}, std={eval_t2m_std_path}"
            )
        eval_compute_dist = True
        print("Async CPU eval enabled; T2M evaluator will be loaded in child process.")
    elif is_main and (not args.eval_no_dist) and eval_t2m_ckpt:
        if not eval_t2m_mean_path or not eval_t2m_std_path:
            raise ValueError(
                "Eval T2M metrics require mean/std paths. "
                "Set evaluator.t2m_eval_mean_path and evaluator.t2m_eval_std_path "
                "in config.json."
            )
        if not os.path.exists(eval_t2m_mean_path) or not os.path.exists(
            eval_t2m_std_path
        ):
            raise FileNotFoundError(
                "Eval T2M mean/std not found: "
                f"mean={eval_t2m_mean_path}, std={eval_t2m_std_path}"
            )
        print(f"Loading eval T2M motion encoder: {eval_t2m_ckpt}")
        eval_t2m_extractor = T2MMotionFeatureExtractor(input_size=config["d_in"]).to(
            device
        )
        eval_t2m_extractor.load_pretrained(eval_t2m_ckpt)
        eval_t2m_extractor.eval()
        eval_t2m_mean = np.load(eval_t2m_mean_path).astype(np.float32)
        eval_t2m_std = np.load(eval_t2m_std_path).astype(np.float32)
        expected_t2m_dim = config["d_in"]
        if (
            eval_t2m_mean.shape[-1] != expected_t2m_dim
            or eval_t2m_std.shape[-1] != expected_t2m_dim
        ):
            raise ValueError(
                f"Eval T2M mean/std dims must be {expected_t2m_dim} (MLD convention), "
                f"got {eval_t2m_mean.shape}, {eval_t2m_std.shape}"
            )
        eval_t2m_stats = (eval_t2m_mean, eval_t2m_std)
        eval_compute_dist = True
    elif is_main and (not args.eval_no_dist):
        print(
            "Warning: eval distribution metrics disabled "
            "(no eval T2M checkpoint provided)."
        )

    teacher = None
    if args.reflow_round >= 1:
        if not args.teacher_ckpt:
            raise ValueError("teacher_ckpt is required for reflow_round >= 1")
        teacher_flow = ConditionalRectFlow(
            d_z=flow_d_z,
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
        state = torch.load(args.teacher_ckpt, map_location=device)
        if "ema" in state:
            teacher_flow.load_state_dict(state["ema"])
        else:
            teacher_flow.load_state_dict(state["model"])
        teacher_flow.to(device)
        teacher = Teacher(
            teacher_flow,
            solver_cfg={"num_steps": teacher_steps, "method": teacher_solver},
        )

    ema = EMA(flow_model, decay=ema_decay) if args.use_ema_teacher else None
    if ema is not None and args.resume and ema_state is not None:
        ema.load_state_dict(ema_state)
    checkpoint_metadata = _make_checkpoint_metadata(
        args=args,
        config=config,
        datasets=datasets,
        vae_backend=vae_backend,
        vae_ckpt_path=vae_ckpt_path,
        stats_path=stats_path,
        openpose_stats_path=openpose_stats_path,
        latent_stats_path=latent_stats_path,
        latent_stats_available=latent_mean is not None and latent_std is not None,
        seq_len=seq_len,
        flow_d_z=flow_d_z,
        latent_len=latent_len,
        cond_frames_min=cond_frames_min,
        cond_frames_max=cond_frames_max,
        eval_cond_frames=eval_cond_frames,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_path = os.path.join(
        args.checkpoint_dir, f"flow_round{args.reflow_round}_last.pt"
    )
    last_good_path = os.path.join(
        args.checkpoint_dir, f"flow_round{args.reflow_round}_last_good.pt"
    )
    if not args.resume and is_main:
        init_state = _flow_checkpoint_state(
            flow_model,
            optimizer,
            epoch=0,
            lr_halved=lr_halved,
            ema=ema,
            metadata=checkpoint_metadata,
        )
        torch.save(init_state, last_path)
        torch.save(init_state, last_good_path)
    if ddp:
        dist.barrier()
    tau_out = torch.linspace(0.0, 1.0, steps=latent_len, device=device)

    def _sync_restore_if_needed(bad_local):
        if not bad_local and not ddp:
            return False
        if not ddp:
            if bad_local:
                restore_path = (
                    last_good_path if os.path.exists(last_good_path) else last_path
                )
                if os.path.exists(restore_path):
                    _restore_checkpoint(
                        restore_path, flow_model, optimizer, ema, device
                    )
            return bad_local
        flag = torch.tensor([1 if bad_local else 0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            dist.barrier()
            restore_path = (
                last_good_path if os.path.exists(last_good_path) else last_path
            )
            if os.path.exists(restore_path):
                _restore_checkpoint(restore_path, flow_model, optimizer, ema, device)
            dist.barrier()
            return True
        return False

    if is_main:
        print("Starting training loop")
    global_step = 0
    async_eval_job = None
    cond_match_log_ema = None
    for epoch in range(start_epoch, args.epochs):
        flow.train()
        epoch_num = epoch + 1
        effective_cond_frame_drop_prob = _scheduled_probability(
            epoch_num,
            cond_frame_drop_prob
            if cond_frame_drop_prob is not None
            else 0.0,
            start_epoch=args.cond_frame_drop_start_epoch,
            ramp_epochs=args.cond_frame_drop_ramp_epochs,
        )
        effective_cfg_drop_prob = (
            args.cfg_drop_prob
            if args.cfg_start_epoch <= 0 or epoch_num >= args.cfg_start_epoch
            else 0.0
        )
        for dataset in (dataset_a, dataset_b):
            if dataset is not None and hasattr(dataset, "cond_frame_drop_prob"):
                dataset.cond_frame_drop_prob = effective_cond_frame_drop_prob
        if is_main and (
            epoch == start_epoch
            or epoch_num == args.cond_frame_drop_start_epoch
            or (
                args.cond_frame_drop_ramp_epochs
                and epoch_num
                == max(1, args.cond_frame_drop_start_epoch)
                + args.cond_frame_drop_ramp_epochs
            )
            or epoch_num == args.cfg_start_epoch
        ):
            print(
                f"Epoch {epoch_num}: effective_cond_frame_drop_prob="
                f"{effective_cond_frame_drop_prob}; "
                f"effective_cfg_drop_prob={effective_cfg_drop_prob}"
            )
        w_cond_epoch = _ramp_weight(
            epoch, solver_cond_start_epoch, solver_cond_ramp_epochs
        )
        w_smooth_epoch = _ramp_weight(
            epoch, solver_smooth_start_epoch, solver_smooth_ramp_epochs
        )
        total_loss = 0.0
        base_loss_sum = 0.0
        smooth_weighted_sum = 0.0
        cond_weighted_sum = 0.0
        total_count = 0
        smooth_acc_sum = 0.0
        smooth_jerk_sum = 0.0
        cond_match_sum = 0.0
        smooth_count = 0
        cond_count = 0
        bad_streak = 0
        t_load = 0.0
        t_encode = 0.0
        t_cond = 0.0
        t_forward = 0.0
        t_backward = 0.0
        if ddp:
            if sampler_a is not None:
                sampler_a.set_epoch(epoch)
            if sampler_b is not None:
                sampler_b.set_epoch(epoch)
        if loader_a is not None and loader_b is not None:
            batch_iter = balanced_batch_iter(loader_a, loader_b, 1, 1)
            steps_per_epoch = max(len(loader_a), len(loader_b))
        elif loader_a is not None:
            batch_iter = _single_loader_iter(loader_a)
            steps_per_epoch = len(loader_a)
        elif loader_b is not None:
            batch_iter = _single_loader_iter(loader_b)
            steps_per_epoch = len(loader_b)
        else:
            raise ValueError("No train loaders were created")
        if not lr_halved and (epoch + 1) >= lr_decay_epoch:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
            lr_halved = True
            if is_main:
                print(
                    f"Halved learning rate at epoch {epoch + 1}: {optimizer.param_groups[0]['lr']:.6g}"
                )
        iter_range = range(steps_per_epoch)
        if is_main:
            iter_range = tqdm(
                iter_range,
                desc=f"Flow Epoch {epoch + 1}",
                leave=False,
            )
        for step_idx in iter_range:
            global_step += 1
            bad_local = False
            loss = None
            drop_cond = None
            t0 = time.perf_counter()
            batch = next(batch_iter)
            (
                motion,
                domain_id,
                style_id,
                mask,
                metas,
                k2d_batch,
                vis_batch,
                conf_batch,
                tau_cond,
                mask_cond,
            ) = _merge_batches(batch)
            motion = motion.to(device)
            domain_id = domain_id.to(device)
            style_id = style_id.to(device)
            if not torch.isfinite(motion).all():
                if args.debug:
                    _debug_log("Warning: non-finite motion batch; skipping", epoch + 1)
                bad_local = True
            if not bad_local:
                t1 = time.perf_counter()
                t_load += t1 - t0

                with torch.no_grad():
                    z_data = encode_motion_latent(
                        vae,
                        motion,
                        domain_id,
                        style_id,
                        mask=mask.to(device),
                    )
                if latent_mean is not None and latent_std is not None:
                    z_data = (z_data - latent_mean) / (latent_std + 1e-6)
                if not torch.isfinite(z_data).all():
                    if args.debug:
                        _debug_log("Warning: non-finite z_data; skipping", epoch + 1)
                    bad_local = True
            if not bad_local:
                t2 = time.perf_counter()
                t_encode += t2 - t1

                x0 = torch.randn_like(z_data)
                t = torch.rand(z_data.shape[0], device=device)
                x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * z_data
                if k2d_batch is None:
                    if args.debug:
                        _debug_log("Warning: missing k2d batch; skipping", epoch + 1)
                    bad_local = True
                else:
                    k2d_batch = k2d_batch.to(device)
                    vis_batch = vis_batch.to(device)
                    conf_batch = (
                        conf_batch.to(device) if conf_batch is not None else vis_batch
                    )
                    tau_cond = tau_cond.to(device)
                    mask_cond = mask_cond.to(device)
                    if not torch.isfinite(k2d_batch).all():
                        if args.debug:
                            _debug_log(
                                "Warning: non-finite keypoints batch; skipping",
                                epoch + 1,
                            )
                        bad_local = True
                    elif effective_cfg_drop_prob > 0:
                        drop_cond = (
                            torch.rand(k2d_batch.shape[0], device=device)
                            < effective_cfg_drop_prob
                        )
                        if drop_cond.any():
                            k2d_batch = k2d_batch.clone()
                            vis_batch = vis_batch.clone()
                            conf_batch = conf_batch.clone()
                            k2d_batch[drop_cond] = 0.0
                            vis_batch[drop_cond] = 0.0
                            conf_batch[drop_cond] = 0.0
            if not bad_local:
                g2d, mem, _vis = flow_model.cond_encoder(
                    k2d_batch,
                    tau_cond,
                    vis_mask=vis_batch,
                    mask_cond=mask_cond,
                    mean=k2d_mean,
                    std=k2d_std,
                )
                if not torch.isfinite(g2d).all() or not torch.isfinite(mem).all():
                    if args.debug:
                        _debug_log(
                            "Warning: non-finite cond encoder output; skipping",
                            epoch + 1,
                        )
                    bad_local = True
            if not bad_local:
                use_teacher = False
                if teacher is not None and args.reflow_round >= 1:
                    use_teacher = True
                    if teacher_mode == "mixed":
                        use_teacher = torch.rand(1).item() < p_teacher
                    if use_teacher:
                        with torch.no_grad():
                            style_id_cond = style_id
                            if drop_cond is not None and drop_cond.any():
                                style_id_cond = style_id.clone()
                                style_id_cond[drop_cond] = 0
                            style_t = flow_model.style_emb(
                                style_id_cond, domain_id, apply_dropout=False
                            )
                            g_t = flow_model.cond_mlp(torch.cat([g2d, style_t], dim=-1))
                            cond_batch = {"tau_out": tau_out, "mem": mem, "g": g_t}
                        z_data = teacher.generate_x1_hat(x0, cond_batch)
                        if not torch.isfinite(z_data).all():
                            if args.debug:
                                _debug_log(
                                    "Warning: non-finite teacher target; skipping",
                                    epoch + 1,
                                )
                            bad_local = True

            if not bad_local:
                x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * z_data
                style_id_cond = style_id
                if drop_cond is not None and drop_cond.any():
                    style_id_cond = style_id.clone()
                    style_id_cond[drop_cond] = 0
                style = flow_model.style_emb(
                    style_id_cond, domain_id, apply_dropout=not use_teacher
                )
                g = flow_model.cond_mlp(torch.cat([g2d, style], dim=-1))
                t3 = time.perf_counter()
                t_cond += t3 - t2
                v_pred = flow_model.flow(x_t, t, tau_out, mem, g)
                target = z_data - x0
                if not torch.isfinite(v_pred).all() or not torch.isfinite(target).all():
                    if args.debug:
                        _debug_log(
                            "Warning: non-finite v_pred/target; skipping", epoch + 1
                        )
                    bad_local = True

            if not bad_local:
                t4 = time.perf_counter()
                t_forward += t4 - t3
                base_loss = torch.mean((v_pred - target) ** 2)
                loss = base_loss
                smooth_acc = None
                smooth_jerk = None
                cond_match_loss = None
                smooth_weighted_loss = None
                cond_weighted_loss = None
                do_smooth_reg = (
                    (global_step % smooth_every == 0)
                    and w_smooth_epoch > 0.0
                    and (lambda_acc > 0.0 or lambda_jerk > 0.0)
                )
                do_cond_reg = (
                    (global_step % cond_every == 0)
                    and w_cond_epoch > 0.0
                    and lambda_cond > 0.0
                )
                needs_solver_reg = do_cond_reg or (
                    do_smooth_reg
                    and args.smooth_loss_domain in ("joints", "latent_solver")
                )
                if do_smooth_reg and args.smooth_loss_domain in (
                    "latent_est",
                    "joints_est",
                ):
                    z_est = x_t + (1.0 - t[:, None, None]) * v_pred
                    if args.smooth_loss_domain == "latent_est":
                        smooth_target = z_est
                    else:
                        smooth_limit = int(args.smooth_subbatch_size)
                        if smooth_limit > 0 and z_est.shape[0] > smooth_limit:
                            smooth_idx = torch.randperm(
                                z_est.shape[0], device=z_est.device
                            )[:smooth_limit]
                            z_smooth = z_est.index_select(0, smooth_idx)
                            domain_smooth = domain_id.index_select(0, smooth_idx)
                            style_smooth = style_id.index_select(0, smooth_idx)
                        else:
                            z_smooth = z_est
                            domain_smooth = domain_id
                            style_smooth = style_id
                        z_decode = z_smooth
                        if latent_mean is not None and latent_std is not None:
                            z_decode = z_decode * (latent_std + 1e-6) + latent_mean
                        x_smooth = decode_motion_latent(
                            vae,
                            z_decode,
                            domain_smooth,
                            style_smooth,
                            out_len=seq_len,
                        )
                        cont_end = LAYOUT_SLICES["feet_contact"][0]
                        mean_t = torch.as_tensor(
                            mean, device=device, dtype=x_smooth.dtype
                        )
                        std_t = torch.as_tensor(
                            std, device=device, dtype=x_smooth.dtype
                        )
                        x_smooth = x_smooth.clone()
                        x_smooth[..., :cont_end] = (
                            x_smooth[..., :cont_end] * std_t + mean_t
                        )
                        smooth_target = _ik263_to_smpl22_torch(x_smooth)
                    smooth_acc, smooth_jerk = _temporal_smoothness_loss(
                        smooth_target, compute_jerk=lambda_jerk > 0.0
                    )
                    smooth_weighted_loss = w_smooth_epoch * (
                        lambda_acc * smooth_acc + lambda_jerk * smooth_jerk
                    )
                    loss = loss + smooth_weighted_loss
                if needs_solver_reg:
                    solver_steps = _pick_solver_steps(
                        epoch,
                        solver_steps_early,
                        solver_steps_mid,
                        solver_steps_late,
                        args.solver_mid_epoch,
                        args.solver_late_epoch,
                    )
                    solver_reg_idx = None
                    if (
                        solver_reg_subbatch_size > 0
                        and x0.shape[0] > solver_reg_subbatch_size
                    ):
                        solver_reg_idx = torch.randperm(x0.shape[0], device=x0.device)[
                            :solver_reg_subbatch_size
                        ]

                    def _reg_select(tensor):
                        if solver_reg_idx is None:
                            return tensor
                        return tensor.index_select(0, solver_reg_idx)

                    x0_reg = _reg_select(x0)
                    mem_reg = _reg_select(mem)
                    g_reg = _reg_select(g)
                    mask_cond_reg = _reg_select(mask_cond)
                    domain_reg = _reg_select(domain_id)
                    style_reg = _reg_select(style_id)
                    k2d_reg = _reg_select(k2d_batch)
                    conf_reg = _reg_select(conf_batch)
                    tau_cond_reg = _reg_select(tau_cond)
                    cond_batch = {
                        "tau_out": tau_out,
                        "mem": mem_reg,
                        "g": g_reg,
                        "mem_mask": ~mask_cond_reg,
                    }
                    z_reg = solve_flow(
                        flow_model.flow,
                        x0_reg,
                        cond_batch,
                        num_steps=solver_steps,
                        method=solver_method,
                        use_activation_checkpoint=args.solver_checkpoint,
                    )
                    joints_reg = None
                    if do_smooth_reg and args.smooth_loss_domain == "latent_solver":
                        smooth_acc, smooth_jerk = _temporal_smoothness_loss(
                            z_reg, compute_jerk=lambda_jerk > 0.0
                        )
                        smooth_weighted_loss = w_smooth_epoch * (
                            lambda_acc * smooth_acc + lambda_jerk * smooth_jerk
                        )
                        loss = loss + smooth_weighted_loss
                    if do_cond_reg or (
                        do_smooth_reg and args.smooth_loss_domain == "joints"
                    ):
                        z_decode = z_reg
                        if latent_mean is not None and latent_std is not None:
                            z_decode = z_decode * (latent_std + 1e-6) + latent_mean
                        cont_end = LAYOUT_SLICES["feet_contact"][0]
                        mean_t = torch.as_tensor(mean, device=device)
                        std_t = torch.as_tensor(std, device=device)
                        reg_decode_batch = int(args.reg_decode_batch_size or 0)
                        if reg_decode_batch <= 0:
                            reg_decode_batch = z_decode.shape[0]
                        reg_decode_batch = max(
                            1, min(reg_decode_batch, z_decode.shape[0])
                        )
                        cond_loss_sum = z_decode.new_zeros(())
                        smooth_acc_sum_chunk = z_decode.new_zeros(())
                        smooth_jerk_sum_chunk = z_decode.new_zeros(())
                        cond_loss_weight = 0
                        smooth_loss_weight = 0
                        for reg_start in range(0, z_decode.shape[0], reg_decode_batch):
                            reg_end = min(
                                reg_start + reg_decode_batch, z_decode.shape[0]
                            )
                            reg_slice = slice(reg_start, reg_end)
                            z_chunk = z_decode[reg_slice]
                            domain_chunk = domain_reg[reg_slice]
                            style_chunk = style_reg[reg_slice]
                            if args.reg_decode_checkpoint and z_chunk.requires_grad:

                                def _decode_chunk(z_in):
                                    return decode_motion_latent(
                                        vae,
                                        z_in,
                                        domain_chunk,
                                        style_chunk,
                                        out_len=seq_len,
                                    )

                                x_reg = checkpoint(
                                    _decode_chunk, z_chunk, use_reentrant=False
                                )
                            else:
                                x_reg = decode_motion_latent(
                                    vae,
                                    z_chunk,
                                    domain_chunk,
                                    style_chunk,
                                    out_len=seq_len,
                                )
                            mean_t = mean_t.to(dtype=x_reg.dtype)
                            std_t = std_t.to(dtype=x_reg.dtype)
                            x_reg = x_reg.clone()
                            x_reg[..., :cont_end] = (
                                x_reg[..., :cont_end] * std_t + mean_t
                            )
                            joints_chunk = _ik263_to_smpl22_torch(x_reg)
                            chunk_size = reg_end - reg_start
                            if do_cond_reg:
                                cond_chunk = _condition_match_loss(
                                    joints_chunk,
                                    k2d_reg[reg_slice],
                                    conf_reg[reg_slice],
                                    tau_cond_reg[reg_slice],
                                    mask_cond_reg[reg_slice],
                                    body25_to_smpl_idx,
                                    body25_to_smpl_valid,
                                    camera_mode=args.cond_match_camera_mode,
                                    min_conf=args.cond_match_min_conf,
                                    min_joints=args.cond_match_min_joints,
                                )
                                cond_loss_sum = cond_loss_sum + cond_chunk * chunk_size
                                cond_loss_weight += chunk_size
                            if do_smooth_reg and args.smooth_loss_domain == "joints":
                                smooth_acc_chunk, smooth_jerk_chunk = (
                                    _temporal_smoothness_loss(
                                        joints_chunk, compute_jerk=lambda_jerk > 0.0
                                    )
                                )
                                smooth_acc_sum_chunk = (
                                    smooth_acc_sum_chunk + smooth_acc_chunk * chunk_size
                                )
                                smooth_jerk_sum_chunk = (
                                    smooth_jerk_sum_chunk
                                    + smooth_jerk_chunk * chunk_size
                                )
                                smooth_loss_weight += chunk_size

                        if do_cond_reg and cond_loss_weight > 0:
                            cond_match_loss = cond_loss_sum / cond_loss_weight
                            cond_weighted_loss = (
                                lambda_cond * w_cond_epoch
                            ) * cond_match_loss
                            loss = loss + cond_weighted_loss

                    if do_smooth_reg and args.smooth_loss_domain == "joints":
                        smooth_acc = smooth_acc_sum_chunk / max(smooth_loss_weight, 1)
                        smooth_jerk = smooth_jerk_sum_chunk / max(smooth_loss_weight, 1)
                        smooth_weighted_loss = w_smooth_epoch * (
                            lambda_acc * smooth_acc + lambda_jerk * smooth_jerk
                        )
                        loss = loss + smooth_weighted_loss
                if not torch.isfinite(loss):
                    if args.debug:
                        _debug_log("Warning: non-finite loss; skipping", epoch + 1)
                    bad_local = True

            if _sync_restore_if_needed(bad_local):
                if args.debug:
                    _debug_log("Recover triggered; skipping step", epoch + 1)
                bad_streak += 1
                if max_bad_steps and bad_streak >= max_bad_steps:
                    for group in optimizer.param_groups:
                        group["lr"] *= 0.5
                    if is_main:
                        print(
                            f"Warning: too many bad steps; reducing lr to {optimizer.param_groups[0]['lr']:.6g}"
                        )
                        _debug_log(
                            f"Too many bad steps; reducing lr to {optimizer.param_groups[0]['lr']:.6g}",
                            epoch + 1,
                        )
                    bad_streak = 0
                continue
            if loss is None:
                continue
            bad_streak = 0
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), grad_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(flow_model)
            t5 = time.perf_counter()
            t_backward += t5 - t4
            total_loss += loss.item()
            base_loss_sum += base_loss.detach().item()
            total_count += 1
            if smooth_weighted_loss is not None:
                smooth_weighted_sum += smooth_weighted_loss.detach().item()
            if cond_weighted_loss is not None:
                cond_weighted_sum += cond_weighted_loss.detach().item()
            if smooth_acc is not None:
                smooth_acc_sum += smooth_acc.detach().item()
                smooth_count += 1
            if smooth_jerk is not None:
                smooth_jerk_sum += smooth_jerk.detach().item()
            if cond_match_loss is not None:
                cond_match_sum += cond_match_loss.detach().item()
                cond_count += 1
            if save_every_steps and (step_idx + 1) % save_every_steps == 0:
                state = _flow_checkpoint_state(
                    flow_model,
                    optimizer,
                    epoch=epoch + 1,
                    lr_halved=lr_halved,
                    ema=ema,
                    metadata=checkpoint_metadata,
                )
                if is_main:
                    torch.save(state, last_path)
                    torch.save(state, last_good_path)

        if ddp:
            stats = torch.tensor(
                [
                    total_loss,
                    base_loss_sum,
                    smooth_weighted_sum,
                    cond_weighted_sum,
                    total_count,
                    smooth_acc_sum,
                    smooth_jerk_sum,
                    cond_match_sum,
                    t_load,
                    t_encode,
                    t_cond,
                    t_forward,
                    t_backward,
                    smooth_count,
                    cond_count,
                ],
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss = stats[0].item()
            base_loss_sum = stats[1].item()
            smooth_weighted_sum = stats[2].item()
            cond_weighted_sum = stats[3].item()
            total_count = int(stats[4].item())
            smooth_acc_sum = stats[5].item()
            smooth_jerk_sum = stats[6].item()
            cond_match_sum = stats[7].item()
            t_load, t_encode, t_cond, t_forward, t_backward = [
                s.item() for s in stats[8:13]
            ]
            smooth_count = int(stats[13].item())
            cond_count = int(stats[14].item())
        if is_main:
            total_loss_avg = total_loss / max(total_count, 1)
            base_loss_avg = base_loss_sum / max(total_count, 1)
            smooth_weighted_avg = smooth_weighted_sum / max(total_count, 1)
            cond_weighted_avg = cond_weighted_sum / max(total_count, 1)
            smooth_acc_avg = smooth_acc_sum / smooth_count if smooth_count > 0 else 0.0
            smooth_jerk_avg = (
                smooth_jerk_sum / smooth_count if smooth_count > 0 else 0.0
            )
            cond_match_avg = cond_match_sum / cond_count if cond_count > 0 else 0.0
            cond_match_effective = cond_match_sum / max(total_count, 1)
            cond_frequency = cond_count / max(total_count, 1)
            if cond_count > 0:
                cond_match_log_ema = (
                    cond_match_avg
                    if cond_match_log_ema is None
                    else 0.8 * cond_match_log_ema + 0.2 * cond_match_avg
                )
            cond_match_ema_value = (
                cond_match_log_ema if cond_match_log_ema is not None else 0.0
            )
            cond_updates = cond_count / max(world_size, 1)
            print(f"Epoch {epoch + 1} avg_loss={total_loss_avg:.6f}")
            print(
                (
                    "Epoch {} base_velocity_mse={:.6f} smooth_weighted={:.6f} "
                    "cond_weighted={:.6f}"
                ).format(
                    epoch + 1,
                    base_loss_avg,
                    smooth_weighted_avg,
                    cond_weighted_avg,
                )
            )
            print(
                (
                    "Epoch {} smooth_acc={:.6f} smooth_jerk={:.6f} "
                    "cond_match_active={:.6f} cond_match_ema={:.6f} "
                    "cond_updates={:.1f}"
                ).format(
                    epoch + 1,
                    smooth_acc_avg,
                    smooth_jerk_avg,
                    cond_match_avg,
                    cond_match_ema_value,
                    cond_updates,
                )
            )
            async_eval_job = _poll_async_eval_job(
                async_eval_job, wandb_run, log_step=epoch + 1
            )
            eval_every_epochs = (
                args.eval_every_epochs
                if args.eval_every_epochs is not None
                else save_every_epochs
            )
            do_eval = (
                eval_every_epochs and (epoch + 1) % eval_every_epochs == 0 and use_aist
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss/avg_velocity_mse": total_loss / max(total_count, 1),
                        "loss/avg_total": total_loss_avg,
                        "loss/base_velocity_mse": base_loss_avg,
                        "loss/smooth_weighted": smooth_weighted_avg,
                        "loss/cond_weighted": cond_weighted_avg,
                        "timing/load": t_load / max(total_count, 1),
                        "timing/encode": t_encode / max(total_count, 1),
                        "timing/cond": t_cond / max(total_count, 1),
                        "timing/forward": t_forward / max(total_count, 1),
                        "timing/backward": t_backward / max(total_count, 1),
                        "loss/smooth_acc": smooth_acc_avg,
                        "loss/smooth_jerk": smooth_jerk_avg,
                        "loss/cond_match": cond_match_avg,
                        "loss/cond_match_active": cond_match_avg,
                        "loss/cond_match_ema": cond_match_ema_value,
                        "loss/cond_match_effective_per_step": cond_match_effective,
                        "schedule/cond_fraction": cond_frequency,
                        "schedule/cond_updates": cond_updates,
                        "schedule/w_cond_epoch": w_cond_epoch,
                        "schedule/w_smooth_epoch": w_smooth_epoch,
                        "schedule/effective_cond_frame_drop_prob": effective_cond_frame_drop_prob,
                        "schedule/effective_cfg_drop_prob": effective_cfg_drop_prob,
                    },
                    step=epoch + 1,
                    commit=not do_eval,
                )
            if total_count == 0:
                print(
                    "Warning: no valid batches this epoch; enable --debug to inspect."
                )
            if args.debug and total_count > 0:
                _debug_log(
                    "Timing (s) "
                    f"load={t_load / total_count:.4f} "
                    f"encode={t_encode / total_count:.4f} "
                    f"cond={t_cond / total_count:.4f} "
                    f"forward={t_forward / total_count:.4f} "
                    f"backward={t_backward / total_count:.4f}",
                    epoch + 1,
                )

            eval_ckpt_path = last_path
            if total_count > 0:
                state = _flow_checkpoint_state(
                    flow_model,
                    optimizer,
                    epoch=epoch + 1,
                    lr_halved=lr_halved,
                    ema=ema,
                    metadata=checkpoint_metadata,
                )
                torch.save(state, last_path)
                torch.save(state, last_good_path)
                if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
                    ckpt_path = os.path.join(
                        args.checkpoint_dir,
                        f"flow_round{args.reflow_round}_epoch{epoch + 1}.pt",
                    )
                    torch.save(state, ckpt_path)
                    eval_ckpt_path = ckpt_path
            if do_eval:
                if args.async_cpu_eval:
                    if async_eval_job is None:
                        async_eval_job = _launch_async_cpu_eval(
                            args=args,
                            config=config,
                            epoch=epoch + 1,
                            flow_ckpt=eval_ckpt_path,
                            vae_ckpt=vae_ckpt_path,
                            seq_len=seq_len,
                        )
                    else:
                        print(
                            "Skipping async CPU eval launch because previous eval "
                            f"for epoch {async_eval_job['epoch']} is still running."
                        )
                else:
                    print(
                        f"Running AIST eval (steps={args.eval_steps}, "
                        f"splits={args.eval_aist_splits}, "
                        f"cameras={args.eval_aist_cameras}, "
                        f"replications={args.eval_replications})"
                    )
                    flow_model.eval()
                    eval_state = flow_model.state_dict()
                    if args.eval_use_ema and ema is not None:
                        flow_model.load_state_dict(ema.state_dict())
                    vae.eval()
                    mapping, computed = _build_smpl22_to_body25(
                        config["smpl45_to_body25_def"]
                    )
                    aist_val_paths = _aist_paths_for_splits(
                        config, aist_dir, args.eval_aist_splits
                    )
                    aist_val = AISTDataset(
                        aist_dir,
                        genre_to_id,
                        seq_len,
                        mean=mean,
                        std=std,
                        files=aist_val_paths,
                        cache_root=config["cache_root"],
                        target_fps=target_fps,
                        src_fps=aist_fps,
                        camera_ids=[
                            cam.strip()
                            for cam in str(args.eval_aist_cameras).split(",")
                            if cam.strip()
                        ],
                        expand_cameras=True,
                        crop_mode=args.eval_aist_crop_mode,
                    )
                    aist_loader = DataLoader(
                        aist_val,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                    )
                    eval_cfg = type(
                        "EvalCfg",
                        (),
                        {
                            "seq_len": seq_len,
                            "d_z": flow_d_z,
                            "latent_len": latent_len,
                            "fps": target_fps,
                            "slack_seconds": 0.1,
                            "cam_mode": "fixed",
                        },
                    )()
                    openpose_cfg = {
                        "aist_dir": openpose_aist_dir,
                        "mvh_root": openpose_mvh_root,
                        "mv_root": mv_root,
                        "mvh_cameras": mvh_cameras,
                        "cond_frames_min": eval_cond_frames,
                        "cond_frames_max": eval_cond_frames,
                        "cond_drop_prob": 0.0,
                        "aist_fps": aist_fps,
                        "mvh_fps": mvh_fps,
                        "target_fps": target_fps,
                        "cond_cache_root": cond_cache_root,
                        "mean": k2d_mean,
                        "std": k2d_std,
                    }
                    for eval_steps in _parse_steps(args.eval_steps):
                        rep_summaries = []
                        for rep_idx in range(max(1, int(args.eval_replications))):
                            rep_seed = seed + rep_idx
                            random.seed(rep_seed)
                            np.random.seed(rep_seed)
                            torch.manual_seed(rep_seed)
                            torch.cuda.manual_seed_all(rep_seed)
                            eval_summary, _ = evaluate_dataset(
                                "AIST",
                                aist_loader,
                                flow_model,
                                vae,
                                eval_cfg,
                                mapping,
                                computed,
                                openpose_cfg,
                                (mean, std),
                                (latent_mean, latent_std),
                                steps=eval_steps,
                                solver="heun",
                                num_samples=0,
                                seed=rep_seed,
                                device=device,
                                compute_dist=eval_compute_dist,
                                save_per_sample=False,
                                metric_extractor=eval_t2m_extractor,
                                t2m_stats=eval_t2m_stats,
                                diversity_times=args.eval_diversity_times,
                                multimodality_repeats=args.eval_multimodality_repeats,
                                multimodality_times=args.eval_multimodality_times,
                                guidance_scale=args.eval_guidance_scale,
                            )
                            if eval_summary:
                                rep_summaries.append(eval_summary)
                        eval_summary = _aggregate_eval_summaries(rep_summaries)
                        if eval_summary:
                            print(
                                f"AIST eval summary steps={eval_steps} "
                                + json.dumps(eval_summary, sort_keys=True)
                            )
                            if wandb_run is not None:
                                wandb_run.log(
                                    _select_wandb_eval_metrics(
                                        eval_summary, f"eval/aist/steps{eval_steps}"
                                    ),
                                    step=epoch + 1,
                                    commit=True,
                                )
                    flow_model.load_state_dict(eval_state)
                    flow_model.train()
        if ddp:
            dist.barrier()

    if is_main:
        _poll_async_eval_job(async_eval_job, wandb_run)
    if ddp:
        dist.destroy_process_group()


def _make_checkpoint_metadata(
    *,
    args,
    config,
    datasets,
    vae_backend,
    vae_ckpt_path,
    stats_path,
    openpose_stats_path,
    latent_stats_path,
    latent_stats_available,
    seq_len,
    flow_d_z,
    latent_len,
    cond_frames_min,
    cond_frames_max,
    eval_cond_frames,
):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "vae_ckpt": vae_ckpt_path,
        "vae_type": vae_backend.vae_type,
        "vae_requested_type": args.vae_type,
        "vae_max_len": vae_backend.max_len,
        "stats_path": stats_path,
        "openpose_stats_path": openpose_stats_path,
        "latent_stats_path": latent_stats_path,
        "latent_stats_available": bool(latent_stats_available),
        "seq_len": int(seq_len),
        "latent_len": int(latent_len),
        "d_z": int(flow_d_z),
        "datasets": sorted(datasets),
        "aist_crop_mode": args.aist_crop_mode,
        "aist_clip_repeat": int(args.aist_clip_repeat),
        "conditioning": {
            "cond_frames_min": int(cond_frames_min),
            "cond_frames_max": int(cond_frames_max),
            "eval_cond_frames": int(eval_cond_frames),
            "cond_drop_prob": args.cond_drop_prob,
            "cond_frame_drop_prob": args.cond_frame_drop_prob,
            "cond_frame_drop_start_epoch": args.cond_frame_drop_start_epoch,
            "cond_frame_drop_ramp_epochs": args.cond_frame_drop_ramp_epochs,
            "cond_frame_drop_mode": args.cond_frame_drop_mode,
            "cond_frame_drop_max_block_frac": args.cond_frame_drop_max_block_frac,
            "cfg_drop_prob": args.cfg_drop_prob,
            "cfg_start_epoch": args.cfg_start_epoch,
        },
        "regularization": {
            "solver_cond_start_epoch": args.solver_cond_start_epoch,
            "solver_smooth_start_epoch": args.solver_smooth_start_epoch,
            "lambda_cond": args.lambda_cond,
            "lambda_acc": args.lambda_acc,
            "lambda_jerk": args.lambda_jerk,
            "solver_reg_subbatch_size": args.solver_reg_subbatch_size,
            "smooth_loss_domain": args.smooth_loss_domain,
            "cond_match_camera_mode": args.cond_match_camera_mode,
            "cond_match_min_conf": args.cond_match_min_conf,
            "cond_match_min_joints": args.cond_match_min_joints,
        },
        "eval": {
            "steps": args.eval_steps,
            "aist_splits": args.eval_aist_splits,
            "aist_cameras": args.eval_aist_cameras,
            "aist_crop_mode": args.eval_aist_crop_mode,
            "replications": args.eval_replications,
            "t2m_motion_encoder_ckpt": config.get("t2m_motion_encoder_ckpt"),
            "t2m_eval_mean_path": config.get("t2m_eval_mean_path"),
            "t2m_eval_std_path": config.get("t2m_eval_std_path"),
        },
        "flow_config": config.get("flow", {}),
    }


def _flow_checkpoint_state(flow_model, optimizer, *, epoch, lr_halved, ema, metadata):
    state = {
        "model": flow_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "lr_halved": lr_halved,
        "metadata": dict(metadata),
        "vae_ckpt": metadata.get("vae_ckpt"),
        "vae_type": metadata.get("vae_type"),
        "latent_stats_path": metadata.get("latent_stats_path"),
        "stats_path": metadata.get("stats_path"),
        "openpose_stats_path": metadata.get("openpose_stats_path"),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


def _poll_async_eval_job(job, wandb_run, log_step=None):
    if job is None:
        return None
    proc = job["proc"]
    code = proc.poll()
    if code is None:
        return job
    print(
        "Async CPU eval for epoch {} finished with code {}. Log: {}".format(
            job["epoch"], code, job["log_path"]
        )
    )
    if code == 0 and wandb_run is not None and os.path.exists(job["json_path"]):
        try:
            with open(job["json_path"], "r", encoding="utf-8") as f:
                payload = json.load(f)
            metrics = {}
            for row in payload.get("summary", []):
                dataset = str(row.get("dataset", "dataset")).lower()
                steps = row.get("steps", "unknown")
                try:
                    steps_label = f"steps{int(steps)}"
                except (TypeError, ValueError):
                    steps_label = f"steps{steps}"
                prefix = f"eval_async/{dataset}/{steps_label}"
                metrics.update(_select_wandb_eval_metrics(row, prefix))
                metrics[f"{prefix}/source_epoch"] = job["epoch"]
            if metrics:
                if log_step is None:
                    wandb_run.log(metrics, commit=True)
                else:
                    wandb_run.log(metrics, step=log_step, commit=False)
        except Exception as exc:
            print(f"Warning: failed to log async eval metrics to wandb: {exc}")
    return None


def _launch_async_cpu_eval(
    *,
    args,
    config,
    epoch,
    flow_ckpt,
    vae_ckpt,
    seq_len,
):
    log_dir = args.async_eval_log_dir or os.path.join(args.checkpoint_dir, "async_eval")
    os.makedirs(log_dir, exist_ok=True)
    steps_tag = str(args.eval_steps).replace(",", "-")
    base = f"epoch{epoch:04d}_steps{steps_tag}_rep{args.eval_replications}"
    json_path = os.path.join(log_dir, f"{base}.json")
    csv_path = os.path.join(log_dir, f"{base}.csv")
    plot_path = os.path.join(log_dir, f"{base}.png")
    log_path = os.path.join(log_dir, f"{base}.log")
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "flowmimic/scripts/eval_flow.py"),
        "--flow-ckpt",
        flow_ckpt,
        "--vae-ckpt",
        vae_ckpt,
        "--vae-type",
        args.vae_type,
        "--device",
        "cpu",
        "--batch-size",
        str(args.eval_batch_size),
        "--num-samples",
        "0",
        "--seq-len",
        str(seq_len),
        "--datasets",
        "AIST",
        "--aist-splits",
        args.eval_aist_splits,
        "--aist-cameras",
        args.eval_aist_cameras,
        "--cond-frames",
        str(
            args.eval_cond_frames
            or args.cond_frames_min
            or config.get("flow", {}).get("cond_frames_min", 7)
        ),
        "--guidance-scale",
        str(args.eval_guidance_scale),
        "--steps",
        str(args.eval_steps),
        "--solver",
        "heun",
        "--aist-crop-mode",
        args.eval_aist_crop_mode,
        "--replications",
        str(args.eval_replications),
        "--save-json",
        json_path,
        "--save-csv",
        csv_path,
        "--save-plot",
        plot_path,
    ]
    if args.eval_use_ema:
        cmd.append("--use-ema")
    else:
        cmd.append("--no-use-ema")
    if args.eval_no_dist:
        cmd.append("--no-dist")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    cpu_eval_threads = str(max(1, int(args.async_eval_cpu_threads)))
    env["OMP_NUM_THREADS"] = env.get("FLOWMIMIC_CPU_EVAL_THREADS", cpu_eval_threads)
    env["MKL_NUM_THREADS"] = env.get("FLOWMIMIC_CPU_EVAL_THREADS", cpu_eval_threads)
    env["OPENBLAS_NUM_THREADS"] = env.get(
        "FLOWMIMIC_CPU_EVAL_THREADS", cpu_eval_threads
    )
    env["NUMEXPR_NUM_THREADS"] = env.get("FLOWMIMIC_CPU_EVAL_THREADS", cpu_eval_threads)
    env["PYTHONUNBUFFERED"] = "1"

    def _preexec_child():
        if args.async_eval_nice:
            try:
                os.nice(int(args.async_eval_nice))
            except OSError:
                pass

    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(
            "[{}] Launching async CPU eval for epoch {}\n{}\n".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                shlex.join(cmd),
            )
        )
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_preexec_child,
        )
    print(
        "Launched async CPU eval for epoch {}: pid={} json={} log={}".format(
            epoch, proc.pid, json_path, log_path
        )
    )
    return {
        "proc": proc,
        "epoch": epoch,
        "json_path": json_path,
        "log_path": log_path,
    }


def _parse_dataset_names(value):
    names = {name.strip().upper() for name in value.split(",") if name.strip()}
    valid = {"AIST", "MVH"}
    unknown = names - valid
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")
    if not names:
        raise ValueError("At least one dataset must be selected")
    return names


def _single_loader_iter(loader):
    while True:
        for batch in loader:
            yield [batch]


def _merge_batches(batches):
    motions = []
    domain_ids = []
    style_ids = []
    masks = []
    metas = []
    k2d_list = []
    vis_list = []
    conf_list = []
    tau_list = []
    mask_cond_list = []
    for batch in batches:
        motions.append(batch["motion"])
        domain_ids.append(batch["domain_id"])
        style_ids.append(batch["style_id"])
        masks.append(batch["mask"])
        metas_raw = batch["meta"]
        metas.extend(_normalize_meta(metas_raw))
        if "k2d" in batch:
            k2d_list.append(batch["k2d"])
            vis_list.append(batch["vis"])
            conf_list.append(batch.get("conf", batch["vis"]))
            tau_list.append(batch["tau_cond"])
            mask_cond_list.append(batch["mask_cond"])
    motion = torch.cat(motions, dim=0)
    domain_id = torch.cat(domain_ids, dim=0)
    style_id = torch.cat(style_ids, dim=0)
    mask = torch.cat(masks, dim=0)
    k2d = torch.cat(k2d_list, dim=0) if k2d_list else None
    vis = torch.cat(vis_list, dim=0) if vis_list else None
    conf = torch.cat(conf_list, dim=0) if conf_list else None
    tau_cond = torch.cat(tau_list, dim=0) if tau_list else None
    mask_cond = torch.cat(mask_cond_list, dim=0) if mask_cond_list else None
    return motion, domain_id, style_id, mask, metas, k2d, vis, conf, tau_cond, mask_cond


def _normalize_meta(metas_raw):
    if isinstance(metas_raw, dict):
        keys = list(metas_raw.keys())
        n = len(metas_raw[keys[0]])
        items = []
        for i in range(n):
            item = {k: metas_raw[k][i] for k in keys}
            items.append(item)
        return items
    if isinstance(metas_raw, list):
        return metas_raw
    return [metas_raw]


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _aist_split_paths(aist_dir, split_path):
    names = _read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def _aist_paths_for_splits(config, aist_dir, split_names):
    paths = []
    for split in str(split_names).split(","):
        split = split.strip().lower()
        if not split:
            continue
        split_path = config.get(f"aist_split_{split}")
        if split_path is None:
            split_path = f"data/AIST++/Annotations/splits/pose_{split}.txt"
        paths.extend(_aist_split_paths(aist_dir, split_path))
    seen = set()
    unique = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


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
    cache_root=None,
):
    k2d_list = []
    vis_list = []
    conf_list = []
    mask_list = []
    tau_list = []
    for meta in metas:
        path = meta["path"]
        if path.endswith(".pkl"):
            k2d, vis, conf = load_aist_openpose(
                path,
                aist_openpose_dir,
                src_fps=aist_fps,
                target_fps=target_fps,
                cache_root=cache_root,
                camera=meta.get("camera"),
                return_conf=True,
            )
        else:
            k2d, vis, conf = load_mvh_openpose(
                path,
                mv_root,
                mvh_openpose_root,
                cameras,
                src_fps=mvh_fps,
                target_fps=target_fps,
                cache_root=cache_root,
                camera=meta.get("camera"),
                return_conf=True,
            )
        if k2d is None:
            k2d = np.zeros((seq_len, 25, 2), dtype=np.float32)
            vis = np.zeros((seq_len, 25), dtype=np.float32)
            conf = np.zeros((seq_len, 25), dtype=np.float32)
        start = meta.get("start", 0)
        orig_len = meta.get("orig_len", k2d.shape[0])
        if orig_len >= seq_len:
            k2d = k2d[start : start + seq_len]
            vis = vis[start : start + seq_len]
            conf = conf[start : start + seq_len]
        else:
            pad_len = seq_len - orig_len
            k2d = np.concatenate(
                [k2d, np.zeros((pad_len, 25, 2), dtype=np.float32)], axis=0
            )
            vis = np.concatenate(
                [vis, np.zeros((pad_len, 25), dtype=np.float32)], axis=0
            )
            conf = np.concatenate(
                [conf, np.zeros((pad_len, 25), dtype=np.float32)], axis=0
            )
        t_len = k2d.shape[0]
        min_frames = max(1, min(int(cond_frames_min or 1), seq_len))
        max_frames = max(
            min_frames,
            min(int(cond_frames_max or min_frames), seq_len),
        )
        k_frames = (
            random.randint(min_frames, max_frames)
            if max_frames > min_frames
            else min_frames
        )
        if t_len <= k_frames:
            idx = np.arange(t_len)
        else:
            idx = np.linspace(0, t_len - 1, k_frames)
            idx = np.unique(np.round(idx).astype(int))
        k2d_sparse = k2d[idx]
        vis_sparse = vis[idx]
        conf_sparse = conf[idx]
        if cond_drop_prob > 0:
            drop = np.random.rand(*vis_sparse.shape) < cond_drop_prob
            vis_sparse = vis_sparse * (~drop)
            conf_sparse = conf_sparse * (~drop)
            k2d_sparse = k2d_sparse * vis_sparse[..., None]
        mask_cond = np.ones((k2d_sparse.shape[0],), dtype=bool)
        tau_cond = idx.astype(np.float32) / max(t_len - 1, 1)

        k2d_list.append(k2d_sparse)
        vis_list.append(vis_sparse)
        conf_list.append(conf_sparse)
        mask_list.append(mask_cond)
        tau_list.append(tau_cond)

    max_len = max(k.shape[0] for k in k2d_list)
    k2d_pad = []
    vis_pad = []
    conf_pad = []
    mask_pad = []
    tau_pad = []
    for k2d, vis, conf, mask, tau in zip(
        k2d_list, vis_list, conf_list, mask_list, tau_list
    ):
        pad = max_len - k2d.shape[0]
        if pad > 0:
            k2d = np.concatenate(
                [k2d, np.zeros((pad, 25, 2), dtype=np.float32)], axis=0
            )
            vis = np.concatenate([vis, np.zeros((pad, 25), dtype=np.float32)], axis=0)
            conf = np.concatenate([conf, np.zeros((pad, 25), dtype=np.float32)], axis=0)
            mask = np.concatenate([mask, np.zeros((pad,), dtype=bool)], axis=0)
            tau = np.concatenate([tau, np.zeros((pad,), dtype=np.float32)], axis=0)
        k2d_pad.append(k2d)
        vis_pad.append(vis)
        conf_pad.append(conf)
        mask_pad.append(mask)
        tau_pad.append(tau)

    k2d_batch = torch.from_numpy(np.stack(k2d_pad, axis=0)).float()
    vis_batch = torch.from_numpy(np.stack(vis_pad, axis=0)).float()
    conf_batch = torch.from_numpy(np.stack(conf_pad, axis=0)).float()
    mask_batch = torch.from_numpy(np.stack(mask_pad, axis=0))
    tau_batch = torch.from_numpy(np.stack(tau_pad, axis=0)).float()
    return k2d_batch, vis_batch, conf_batch, tau_batch, mask_batch


def _parse_steps(raw):
    vals = [int(v.strip()) for v in str(raw).split(",") if v.strip()]
    vals = [v for v in vals if v > 0]
    if not vals:
        return [8]
    return vals


def _select_wandb_eval_metrics(row, prefix):
    metrics = {}
    for key in WANDB_EVAL_METRIC_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            metrics[f"{prefix}/{key}"] = value
    return metrics


def _aggregate_eval_summaries(rows):
    if not rows:
        return {}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
    )
    return {
        key: float(
            np.mean(
                [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            )
        )
        for key in keys
    }


def _ramp_weight(epoch, start_epoch, ramp_epochs):
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    if epoch >= start_epoch + ramp_epochs:
        return 1.0
    return (epoch - start_epoch) / float(ramp_epochs)


def _scheduled_probability(epoch_num, target_prob, start_epoch=0, ramp_epochs=0):
    target_prob = float(target_prob or 0.0)
    if target_prob <= 0.0:
        return 0.0
    start_epoch = int(start_epoch or 0)
    ramp_epochs = int(ramp_epochs or 0)
    if start_epoch > 0 and epoch_num < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return target_prob

    ramp_start = max(1, start_epoch)
    progress = (int(epoch_num) - ramp_start) / float(ramp_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return target_prob * progress


def _pick_solver_steps(epoch, early, mid, late, mid_epoch, late_epoch):
    if epoch < mid_epoch:
        pool = early
    elif epoch < late_epoch:
        pool = mid
    else:
        pool = late
    return random.choice(pool)


def _build_body25_to_smpl22(def_path, device):
    with open(def_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    smpl_name_to_idx = {}
    body_to_smpl = np.full((22,), -1, dtype=np.int64)
    for joint in cfg.get("smpl_joints", []):
        smpl_idx = joint.get("smpl_idx")
        body_idx = joint.get("body25_idx")
        name = joint.get("name")
        if name is not None and smpl_idx is not None:
            smpl_name_to_idx[name] = smpl_idx
        if smpl_idx is None or smpl_idx >= 22 or body_idx is None:
            continue
        body_to_smpl[smpl_idx] = int(body_idx)
    for rule in cfg.get("computed_body25", []):
        name = rule.get("name")
        body_idx = rule.get("body25_idx")
        if name is None or body_idx is None:
            continue
        smpl_idx = smpl_name_to_idx.get(name)
        if smpl_idx is not None and smpl_idx < 22:
            body_to_smpl[smpl_idx] = int(body_idx)
    idx = torch.tensor(body_to_smpl, device=device, dtype=torch.long)
    valid = idx >= 0
    return idx.clamp_min(0), valid


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


def _condition_match_loss(
    joints22,
    k2d_body25,
    conf_body25,
    tau_cond,
    mask_cond,
    body25_to_smpl_idx,
    body25_to_smpl_valid,
    camera_mode="per_frame",
    min_conf=0.4,
    min_joints=6,
    eps=1e-6,
):
    bsz, _, _, _ = joints22.shape
    t_len = joints22.shape[1]
    num_cond = tau_cond.shape[1]
    idx = body25_to_smpl_idx.view(1, 1, 22, 1).expand(bsz, num_cond, 22, 1)
    idx2 = body25_to_smpl_idx.view(1, 1, 22).expand(bsz, num_cond, 22)
    k2d_smpl = torch.gather(k2d_body25, 2, idx.expand(-1, -1, -1, 2))
    conf_smpl = torch.gather(conf_body25, 2, idx2)
    valid_joint = body25_to_smpl_valid.view(1, 1, 22).to(conf_smpl.dtype)
    valid_frame = mask_cond.unsqueeze(-1).to(conf_smpl.dtype)
    conf_smpl = conf_smpl * valid_joint * valid_frame

    t_idx = torch.clamp(
        torch.round(tau_cond * float(t_len - 1)).long(), min=0, max=t_len - 1
    )
    gather_idx = t_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 22, 3)
    joints_cond = torch.gather(joints22, 1, gather_idx)
    xy = joints_cond[..., :2]

    if camera_mode == "per_frame":
        s, tx, ty, _ = _fit_weak_persp_torch(xy, k2d_smpl, conf_smpl, eps=eps)
        pred = s.unsqueeze(-1).unsqueeze(-1) * xy
        pred[..., 0] = pred[..., 0] + tx.unsqueeze(-1)
        pred[..., 1] = pred[..., 1] + ty.unsqueeze(-1)
    elif camera_mode == "shared":
        fit_conf = conf_smpl * (conf_smpl >= float(min_conf)).to(conf_smpl.dtype)
        enough_fit_joints = (fit_conf > eps).sum(dim=(1, 2), keepdim=True) >= max(
            int(min_joints), 1
        )
        fit_conf = torch.where(enough_fit_joints, fit_conf, conf_smpl)
        flat_joints = xy.reshape(bsz, 1, num_cond * 22, 2)
        flat_k2d = k2d_smpl.reshape(bsz, 1, num_cond * 22, 2)
        flat_conf = fit_conf.reshape(bsz, 1, num_cond * 22)
        s, tx, ty, _ = _fit_weak_persp_torch(
            flat_joints,
            flat_k2d,
            flat_conf,
            eps=eps,
        )
        pred = s.view(bsz, 1, 1, 1) * xy
        pred[..., 0] = pred[..., 0] + tx.view(bsz, 1, 1)
        pred[..., 1] = pred[..., 1] + ty.view(bsz, 1, 1)
    else:
        raise ValueError(f"Unsupported condition match camera mode: {camera_mode}")

    wsum = conf_smpl.sum(dim=-1)
    err = torch.linalg.norm(pred - k2d_smpl, dim=-1)
    frame_err = (err * conf_smpl).sum(dim=-1) / (wsum + eps)
    valid = wsum > eps
    if valid.any():
        return frame_err[valid].mean()
    return joints22.new_zeros(())


def _temporal_smoothness_loss(sequence, compute_jerk=True):
    if sequence.shape[1] < 3:
        zero = sequence.new_zeros(())
        return zero, zero
    vel = sequence[:, 1:] - sequence[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    smooth_acc = torch.mean(acc**2) if acc.numel() else sequence.new_zeros(())
    if compute_jerk and acc.shape[1] >= 2:
        jerk = acc[:, 1:] - acc[:, :-1]
        smooth_jerk = torch.mean(jerk**2) if jerk.numel() else sequence.new_zeros(())
    else:
        smooth_jerk = sequence.new_zeros(())
    return smooth_acc, smooth_jerk


def _fit_weak_persp_torch(xy, uv, w, eps=1e-6):
    x = xy[..., 0]
    y = xy[..., 1]
    u = uv[..., 0]
    v = uv[..., 1]
    w = w.clamp_min(0.0)
    sw = w.sum(dim=-1)
    sxx = (w * (x * x + y * y)).sum(dim=-1)
    sx = (w * x).sum(dim=-1)
    sy = (w * y).sum(dim=-1)
    su = (w * u).sum(dim=-1)
    sv = (w * v).sum(dim=-1)
    sxu = (w * (x * u + y * v)).sum(dim=-1)
    n = x.shape[0] * x.shape[1]
    a = torch.zeros((n, 3, 3), dtype=xy.dtype, device=xy.device)
    b = torch.zeros((n, 3), dtype=xy.dtype, device=xy.device)
    a[:, 0, 0] = sxx.reshape(-1)
    a[:, 0, 1] = sx.reshape(-1)
    a[:, 0, 2] = sy.reshape(-1)
    a[:, 1, 0] = sx.reshape(-1)
    a[:, 1, 1] = sw.reshape(-1)
    a[:, 2, 0] = sy.reshape(-1)
    a[:, 2, 2] = sw.reshape(-1)
    b[:, 0] = sxu.reshape(-1)
    b[:, 1] = su.reshape(-1)
    b[:, 2] = sv.reshape(-1)
    eye = torch.eye(3, dtype=xy.dtype, device=xy.device).unsqueeze(0)
    theta = torch.linalg.solve(a + eps * eye, b)
    theta = theta.view(x.shape[0], x.shape[1], 3)
    s = theta[..., 0]
    tx = theta[..., 1]
    ty = theta[..., 2]
    return s, tx, ty, sw


def _restore_checkpoint(path, flow, optimizer, ema, device):
    state = torch.load(path, map_location=device)
    flow.load_state_dict(state["model"])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])


def _set_optimizer_group_lrs(optimizer, lrs):
    if not lrs:
        return
    for idx, group in enumerate(optimizer.param_groups):
        group["lr"] = lrs[min(idx, len(lrs) - 1)]


if __name__ == "__main__":
    main()
