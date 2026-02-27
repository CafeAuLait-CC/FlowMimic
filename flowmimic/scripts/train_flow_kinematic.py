import argparse
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

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.teacher import EMA, Teacher
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.stats import load_mean_std
from flowmimic.src.model.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.data.openpose import compute_openpose_stats
from flowmimic.scripts.eval_flow_kinematic import _build_smpl22_to_body25, evaluate_dataset


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _aist_split_paths(aist_dir, split_path):
    names = _read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


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


def _merge_batches(batches):
    motions = []
    domain_ids = []
    style_ids = []
    masks = []
    metas = []
    k2d_list = []
    vis_list = []
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
            tau_list.append(batch["tau_cond"])
            mask_cond_list.append(batch["mask_cond"])
    motion = torch.cat(motions, dim=0)
    domain_id = torch.cat(domain_ids, dim=0)
    style_id = torch.cat(style_ids, dim=0)
    mask = torch.cat(masks, dim=0)
    k2d = torch.cat(k2d_list, dim=0) if k2d_list else None
    vis = torch.cat(vis_list, dim=0) if vis_list else None
    tau_cond = torch.cat(tau_list, dim=0) if tau_list else None
    mask_cond = torch.cat(mask_cond_list, dim=0) if mask_cond_list else None
    return motion, domain_id, style_id, mask, metas, k2d, vis, tau_cond, mask_cond


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
    from flowmimic.src.data.openpose import load_aist_openpose, load_mvh_openpose

    k2d_list = []
    vis_list = []
    mask_list = []
    tau_list = []
    for meta in metas:
        path = meta["path"]
        if path.endswith(".pkl"):
            k2d, vis = load_aist_openpose(
                path,
                aist_openpose_dir,
                src_fps=aist_fps,
                target_fps=target_fps,
                cache_root=cache_root,
                camera=meta.get("camera"),
            )
        else:
            k2d, vis = load_mvh_openpose(
                path,
                mv_root,
                mvh_openpose_root,
                cameras,
                src_fps=mvh_fps,
                target_fps=target_fps,
                cache_root=cache_root,
                camera=meta.get("camera"),
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
            k2d = np.concatenate(
                [k2d, np.zeros((pad_len, 25, 2), dtype=np.float32)], axis=0
            )
            vis = np.concatenate(
                [vis, np.zeros((pad_len, 25), dtype=np.float32)], axis=0
            )
        t_len = k2d.shape[0]
        k_frames = cond_frames_min
        if t_len <= k_frames:
            idx = np.arange(t_len)
        else:
            idx = np.linspace(0, t_len - 1, k_frames)
            idx = np.unique(np.round(idx).astype(int))
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
            k2d = np.concatenate(
                [k2d, np.zeros((pad, 25, 2), dtype=np.float32)], axis=0
            )
            vis = np.concatenate([vis, np.zeros((pad, 25), dtype=np.float32)], axis=0)
            mask = np.concatenate([mask, np.zeros((pad,), dtype=bool)], axis=0)
            tau = np.concatenate([tau, np.zeros((pad,), dtype=np.float32)], axis=0)
        k2d_pad.append(k2d)
        vis_pad.append(vis)
        mask_pad.append(mask)
        tau_pad.append(tau)

    k2d_batch = torch.from_numpy(np.stack(k2d_pad, axis=0)).float()
    vis_batch = torch.from_numpy(np.stack(vis_pad, axis=0)).float()
    mask_batch = torch.from_numpy(np.stack(mask_pad, axis=0))
    tau_batch = torch.from_numpy(np.stack(tau_pad, axis=0)).float()
    return k2d_batch, vis_batch, tau_batch, mask_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--reflow-round", type=int, default=0)
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--teacher-solver", type=str, default=None)
    parser.add_argument("--use-ema-teacher", action="store_true")
    parser.add_argument(
        "--teacher-mode", type=str, choices=["strict", "mixed"], default=None
    )
    parser.add_argument("--p-teacher", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/flow_263")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument(
        "--lr-scale-mode", type=str, choices=["none", "linear"], default="none"
    )
    parser.add_argument("--max-bad-steps", type=int, default=50)
    parser.add_argument("--cond-lr-scale", type=float, default=0.1)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smooth-warmup-epochs", type=int, default=0)
    parser.add_argument("--smooth-ramp-epochs", type=int, default=0)
    parser.add_argument("--smooth-lambda-acc", type=float, default=0.0)
    parser.add_argument("--smooth-lambda-jerk", type=float, default=0.0)
    parser.add_argument("--wandb-project", type=str, default="FlowMimic")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
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
    parser.add_argument("--eval-use-ema", action="store_true", default=True)
    parser.add_argument("--no-eval-use-ema", dest="eval_use_ema", action="store_false")
    args = parser.parse_args()

    ddp = args.ddp
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=local_rank)
        device = torch.device("cuda", local_rank)
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
        print("Loading config")
    config = load_config()
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
                run_name = f"flow-kinematic-{stamp}"
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
                    "seq_len": config.get("seq_len"),
                    "lr": args.lr,
                    "cond_lr_scale": args.cond_lr_scale,
                    "reflow_round": args.reflow_round,
                    "teacher_mode": args.teacher_mode,
                    "p_teacher": args.p_teacher,
                    "cond_frames_min": config.get("flow", {}).get("cond_frames_min", 7),
                    "cond_frames_max": config.get("flow", {}).get("cond_frames_max", 7),
                },
            )
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    seq_len = config["seq_len"]
    stats_path = config["stats_path"]
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)
    openpose_aist_dir = config.get(
        "aist_openpose_dir", "data/AIST++/Annotations/openpose"
    )
    openpose_mvh_root = config.get("mvh_openpose_root", "data/MVHumanNet")
    mvh_cameras = config.get("mvh_cameras", ["22327091", "22327113", "22327084"])
    aist_cameras = config.get("aist_cameras", ["01", "02", "08", "09"])
    openpose_stats_path = config.get("openpose_stats_path", "data/openpose_stats.npz")
    cond_cache_root = config.get("cond_cache_root", "data/cached_cond")

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
    ema_decay = flow_cfg.get("ema_decay", 0.999)
    cond_frames_min = flow_cfg.get("cond_frames_min", 7)
    cond_frames_max = flow_cfg.get("cond_frames_max", 7)
    cond_drop_prob = flow_cfg.get("cond_drop_prob", 0.2)
    grad_clip_norm = config.get("grad_clip_norm", 1.0)
    save_every_steps = args.save_every_steps or flow_cfg.get("save_every_steps", 0)
    save_every_epochs = args.save_every_epochs
    max_bad_steps = args.max_bad_steps
    smooth_warmup_epochs = args.smooth_warmup_epochs
    smooth_ramp_epochs = args.smooth_ramp_epochs
    smooth_lambda_acc = args.smooth_lambda_acc
    smooth_lambda_jerk = args.smooth_lambda_jerk
    smooth_slices = [
        ("root_yaw_vel", 0, 1, 0.5),
        ("root_xz_vel", 1, 3, 0.5),
        ("root_y", 3, 4, 0.5),
        ("ric", 4, 67, 1.0),
        ("local_vel", 193, 259, 1.0),
    ]
    seed = config.get("seed", 42)
    seed_rank = seed + rank
    random.seed(seed_rank)
    np.random.seed(seed_rank)
    torch.manual_seed(seed_rank)
    torch.cuda.manual_seed_all(seed_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    aist_train_paths = _aist_split_paths(aist_dir, config["aist_split_train"])
    mvh_train_dirs = _read_lines(config["mvh_split_train"])
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
    )
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
    )

    sampler_a = (
        DistributedSampler(dataset_a, num_replicas=world_size, rank=rank)
        if ddp
        else None
    )
    sampler_b = (
        DistributedSampler(dataset_b, num_replicas=world_size, rank=rank)
        if ddp
        else None
    )

    def _seed_worker(worker_id):
        worker_seed = seed_rank + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    if is_main:
        print("Building dataloaders")
    loader_a = DataLoader(
        dataset_a,
        batch_size=args.batch_size,
        shuffle=(sampler_a is None),
        drop_last=True,
        num_workers=args.num_workers,
        sampler=sampler_a,
        worker_init_fn=_seed_worker,
    )
    loader_b = DataLoader(
        dataset_b,
        batch_size=args.batch_size,
        shuffle=(sampler_b is None),
        drop_last=True,
        num_workers=args.num_workers,
        sampler=sampler_b,
        worker_init_fn=_seed_worker,
    )
    batch_iter = balanced_batch_iter(loader_a, loader_b, 1, 1)

    if is_main:
        print("Building flow model")
    d_z = 263
    flow_model = ConditionalRectFlow(
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
    ).to(device)

    last_path = os.path.join(args.checkpoint_dir, "flow_last.pt")
    last_good_path = os.path.join(args.checkpoint_dir, "flow_last_good.pt")
    if is_main:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device)
        flow_model.load_state_dict(resume_state["model"])
        if is_main:
            print(f"Resuming from {args.resume}")

    if ddp:
        flow_model = torch.nn.parallel.DistributedDataParallel(
            flow_model, device_ids=[local_rank]
        )
    flow_core = flow_model.module if ddp else flow_model
    if ddp and args.lr_scale_mode == "linear":
        lr = lr * world_size
    if is_main:
        print(f"LR={lr} (mode={args.lr_scale_mode}, world_size={world_size})")
        print(
            f"Per-GPU batch={args.batch_size}, global batch={args.batch_size * world_size}"
        )
        print(f"Cond LR scale={cond_lr_scale}")

    cond_params = list(flow_core.cond_encoder.parameters()) + list(
        flow_core.cond_mlp.parameters()
    )
    cond_param_ids = {id(p) for p in cond_params}
    other_params = [p for p in flow_model.parameters() if id(p) not in cond_param_ids]
    param_groups = [
        {"params": other_params, "lr": lr},
        {"params": cond_params, "lr": lr * cond_lr_scale},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    start_epoch = 0
    ema_state = None
    if resume_state is not None:
        if not args.reset_optimizer and "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if args.reset_optimizer and is_main:
            print("Resetting optimizer state for resume")
        if args.use_ema_teacher and "ema" in resume_state:
            ema_state = resume_state["ema"]
        start_epoch = int(resume_state.get("epoch", 0))

    if is_main:
        print("Loading OpenPose stats")
    if not os.path.exists(openpose_stats_path):
        if is_main:
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

    teacher = None
    ema = None
    if args.use_ema_teacher:
        ema = EMA(flow_core, decay=ema_decay)
        if ema_state is not None:
            ema.load_state_dict(ema_state)
        teacher = Teacher(
            ema, solver_cfg={"num_steps": teacher_steps, "method": teacher_solver}
        )
    elif args.reflow_round >= 1:
        if not args.teacher_ckpt:
            raise ValueError("teacher_ckpt is required for reflow_round >= 1")
        teacher_flow = ConditionalRectFlow(
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
        ).to(device)
        state = torch.load(args.teacher_ckpt, map_location=device)
        teacher_flow.load_state_dict(state["model"])
        teacher_flow.eval()
        teacher = Teacher(
            teacher_flow,
            solver_cfg={"num_steps": teacher_steps, "method": teacher_solver},
        )

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
    lr_halved = False
    for epoch in range(start_epoch, args.epochs):
        flow_model.train()
        if smooth_warmup_epochs > 0 and epoch < smooth_warmup_epochs:
            w_smooth_epoch = 0.0
        elif (
            smooth_ramp_epochs > 0 and epoch < smooth_warmup_epochs + smooth_ramp_epochs
        ):
            w_smooth_epoch = (epoch - smooth_warmup_epochs) / float(smooth_ramp_epochs)
        else:
            w_smooth_epoch = 1.0
        total_loss = 0.0
        total_count = 0
        smooth_acc_sum = 0.0
        smooth_jerk_sum = 0.0
        bad_streak = 0
        t_load = 0.0
        t_encode = 0.0
        t_cond = 0.0
        t_forward = 0.0
        t_backward = 0.0
        if ddp:
            sampler_a.set_epoch(epoch)
            sampler_b.set_epoch(epoch)
        steps_per_epoch = max(len(loader_a), len(loader_b))
        if not lr_halved and (epoch + 1) >= max(1, args.epochs // 2):
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
            bad_local = False
            loss = None
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
                z_data = motion
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
                tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
                if k2d_batch is None:
                    if args.debug:
                        _debug_log("Warning: missing k2d batch; skipping", epoch + 1)
                    bad_local = True
                else:
                    k2d_batch = k2d_batch.to(device)
                    vis_batch = vis_batch.to(device)
                    tau_cond = tau_cond.to(device)
                    mask_cond = mask_cond.to(device)
                    if not torch.isfinite(k2d_batch).all():
                        if args.debug:
                            _debug_log(
                                "Warning: non-finite keypoints batch; skipping",
                                epoch + 1,
                            )
                        bad_local = True
            if not bad_local:
                g2d, mem, _vis = flow_core.cond_encoder(
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
                            style_t = flow_core.style_emb(
                                style_id, domain_id, apply_dropout=False
                            )
                            g_t = flow_core.cond_mlp(torch.cat([g2d, style_t], dim=-1))
                            tau_out = torch.linspace(
                                0.0, 1.0, steps=seq_len, device=device
                            )
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
                style = flow_core.style_emb(
                    style_id, domain_id, apply_dropout=not use_teacher
                )
                g = flow_core.cond_mlp(torch.cat([g2d, style], dim=-1))
                t3 = time.perf_counter()
                t_cond += t3 - t2
                v_pred = flow_core.flow(x_t, t, tau_out, mem, g)
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
                loss = torch.mean((v_pred - target) ** 2)
                # Smoothness regularizer disabled for now (velocity MSE only).
                smooth_acc = None
                smooth_jerk = None
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
                ema.update(flow_core)
            t5 = time.perf_counter()
            t_backward += t5 - t4
            total_loss += loss.item()
            total_count += 1
            if smooth_acc is not None:
                smooth_acc_sum += smooth_acc.detach().item()
            if smooth_jerk is not None:
                smooth_jerk_sum += smooth_jerk.detach().item()
            if save_every_steps and (step_idx + 1) % save_every_steps == 0:
                state = {
                    "model": flow_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                }
                if ema is not None:
                    state["ema"] = ema.state_dict()
                if is_main:
                    torch.save(state, last_path)
                    torch.save(state, last_good_path)

        if ddp:
            stats = torch.tensor(
                [
                    total_loss,
                    total_count,
                    t_load,
                    t_encode,
                    t_cond,
                    t_forward,
                    t_backward,
                ],
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_count = stats[0].item(), int(stats[1].item())
            t_load, t_encode, t_cond, t_forward, t_backward = [
                s.item() for s in stats[2:]
            ]
        if is_main:
            print(
                f"Epoch {epoch + 1} avg_velocity_mse={total_loss / max(total_count, 1):.6f}"
            )
            do_eval = (
                wandb_run is not None
                and save_every_epochs
                and (epoch + 1) % save_every_epochs == 0
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss/avg_velocity_mse": total_loss / max(total_count, 1),
                        "timing/load": t_load / max(total_count, 1),
                        "timing/encode": t_encode / max(total_count, 1),
                        "timing/cond": t_cond / max(total_count, 1),
                        "timing/forward": t_forward / max(total_count, 1),
                        "timing/backward": t_backward / max(total_count, 1),
                    },
                    step=epoch + 1,
                    commit=not do_eval,
                )
            # print(
            #     "Epoch {} smooth_acc={:.6f} smooth_jerk={:.6f}".format(
            #         epoch + 1,
            #         smooth_acc_sum / max(total_count, 1),
            #         smooth_jerk_sum / max(total_count, 1),
            #     )
            # )
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
            if total_count > 0:
                state = {
                    "model": flow_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                }
                if ema is not None:
                    state["ema"] = ema.state_dict()
                torch.save(state, last_path)
                torch.save(state, last_good_path)
                if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
                    ckpt_path = os.path.join(
                        args.checkpoint_dir,
                        f"flow_round{args.reflow_round}_epoch{epoch + 1}.pt",
                    )
                    torch.save(state, ckpt_path)
            if do_eval:
                print("Running AIST eval (steps=4)")
                flow_core.eval()
                eval_state = flow_core.state_dict()
                if args.eval_use_ema and ema is not None:
                    flow_core.load_state_dict(ema.state_dict())
                mapping, computed = _build_smpl22_to_body25(
                    config["smpl45_to_body25_def"]
                )
                aist_val_paths = _aist_split_paths(
                    aist_dir, config["aist_split_val"]
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
                    camera_ids=aist_cameras,
                    expand_cameras=True,
                    include_cond=True,
                    openpose_dir=openpose_aist_dir,
                    cond_cache_root=cond_cache_root,
                    cond_frames_min=cond_frames_min,
                    cond_frames_max=cond_frames_max,
                    cond_drop_prob=cond_drop_prob,
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
                        "d_z": d_z,
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
                    "cond_frames_min": cond_frames_min,
                    "cond_frames_max": cond_frames_max,
                    "cond_drop_prob": cond_drop_prob,
                    "aist_fps": aist_fps,
                    "mvh_fps": mvh_fps,
                    "target_fps": target_fps,
                    "cond_cache_root": cond_cache_root,
                    "mean": k2d_mean,
                    "std": k2d_std,
                }
                eval_summary, _ = evaluate_dataset(
                    "AIST",
                    aist_loader,
                    flow_core,
                    eval_cfg,
                    mapping,
                    computed,
                    openpose_cfg,
                    (mean, std),
                    steps=4,
                    solver="heun",
                    num_samples=0,
                    seed=seed,
                    device=device,
                    compute_dist=True,
                    save_per_sample=False,
                )
                if eval_summary:
                    wandb_run.log(
                        {f"eval/aist/{k}": v for k, v in eval_summary.items()},
                        step=epoch + 1,
                        commit=True,
                    )
                flow_core.load_state_dict(eval_state)
                flow_core.train()
        if ddp:
            dist.barrier()

    if ddp:
        dist.destroy_process_group()


def _restore_checkpoint(path, flow, optimizer, ema, device):
    state = torch.load(path, map_location=device)
    flow.load_state_dict(state["model"])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])


if __name__ == "__main__":
    main()
