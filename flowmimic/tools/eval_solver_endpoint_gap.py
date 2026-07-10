"""Measure one-step rectified-flow endpoint gap against solver endpoints.

This diagnostic compares:

    z_est = x_t + (1 - t) * v_pred

with:

    z_solved = solve_flow(flow, x0, condition, num_steps=K)

for saved flow checkpoints. It is intended to test whether the cheap one-step
endpoint can safely replace the expensive multi-step solver endpoint for
decoded condition/smoothness regularization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.backend import encode_motion_latent, load_vae_backend
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id
from flowmimic.src.model.vae.stats import load_mean_std


@dataclass(frozen=True)
class CheckpointSpec:
    run_id: str
    epoch: int
    path: str


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _aist_split_paths(aist_dir: str, split_path: str) -> list[str]:
    names = _read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def _aist_paths_for_splits(config: dict, split_names: str) -> list[str]:
    paths: list[str] = []
    aist_dir = config["aist_motions_dir"]
    for split in str(split_names).split(","):
        split = split.strip().lower()
        if not split:
            continue
        split_path = config.get(f"aist_split_{split}")
        if split_path is None:
            split_path = f"data/AIST++/Annotations/splits/pose_{split}.txt"
        paths.extend(_aist_split_paths(aist_dir, split_path))
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def _parse_checkpoint_spec(raw: str) -> CheckpointSpec:
    parts = raw.split(":", 2)
    if len(parts) == 3:
        run_id, epoch_raw, path = parts
        return CheckpointSpec(run_id=run_id, epoch=int(epoch_raw), path=path)
    path = raw
    name = os.path.basename(path)
    match = re.search(r"epoch(\d+)", name)
    epoch = int(match.group(1)) if match else -1
    run_id = Path(path).parent.name
    return CheckpointSpec(run_id=run_id, epoch=epoch, path=path)


def _make_flow(config: dict, d_z: int, device: torch.device) -> ConditionalRectFlow:
    flow_cfg = config.get("flow", {})
    model = ConditionalRectFlow(
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
    return model.to(device)


def _load_flow_state(
    flow: ConditionalRectFlow,
    ckpt_path: str,
    device: torch.device,
    state_key: str,
) -> tuple[str, int | None]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    key_used = state_key
    if state_key == "auto":
        key_used = "ema" if isinstance(state, dict) and "ema" in state else "model"
    if isinstance(state, dict) and key_used in state:
        model_state = state[key_used]
    elif isinstance(state, dict) and "model" in state:
        key_used = "model"
        model_state = state["model"]
    else:
        key_used = "raw"
        model_state = state
    flow.load_state_dict(_strip_module_prefix(model_state))
    flow.eval()
    epoch = int(state.get("epoch")) if isinstance(state, dict) and "epoch" in state else None
    return key_used, epoch


def _collate_optional(batch: dict, name: str) -> torch.Tensor | None:
    value = batch.get(name)
    if value is None:
        return None
    return value


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _tensor_stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _evaluate_checkpoint(
    spec: CheckpointSpec,
    flow: ConditionalRectFlow,
    vae: torch.nn.Module,
    loader: DataLoader,
    config: dict,
    device: torch.device,
    *,
    latent_mean: torch.Tensor | None,
    latent_std: torch.Tensor | None,
    k2d_mean: np.ndarray,
    k2d_std: np.ndarray,
    latent_len: int,
    solver_steps: int,
    solver_method: str,
    state_key: str,
    seed: int,
) -> dict[str, float | int | str | None]:
    _seed_all(seed, device)
    key_used, saved_epoch = _load_flow_state(flow, spec.path, device, state_key)
    tau_out = torch.linspace(0.0, 1.0, steps=latent_len, device=device)

    gap_l2: list[float] = []
    gap_rmse: list[float] = []
    rel_gap_solved: list[float] = []
    rel_gap_target: list[float] = []
    est_target_l2: list[float] = []
    solved_target_l2: list[float] = []
    base_mse: list[float] = []
    endpoint_cos: list[float] = []

    eps = 1e-8
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{spec.run_id} epoch {spec.epoch}", leave=False):
            motion = batch["motion"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)
            style_id = batch["style_id"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            k2d = batch["k2d"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            tau_cond = batch["tau_cond"].to(device, non_blocking=True)
            mask_cond = batch["mask_cond"].to(device, non_blocking=True)

            z_data = encode_motion_latent(
                vae,
                motion,
                domain_id,
                style_id,
                mask=mask,
            )
            if latent_mean is not None and latent_std is not None:
                z_data = (z_data - latent_mean) / (latent_std + 1e-6)

            x0 = torch.randn_like(z_data)
            t = torch.rand(z_data.shape[0], device=device)
            x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * z_data

            g2d, mem, _ = flow.cond_encoder(
                k2d,
                tau_cond,
                vis_mask=vis,
                mask_cond=mask_cond,
                mean=k2d_mean,
                std=k2d_std,
            )
            style = flow.style_emb(style_id, domain_id, apply_dropout=False)
            g = flow.cond_mlp(torch.cat([g2d, style], dim=-1))
            v_pred = flow.flow(x_t, t, tau_out, mem, g, ~mask_cond)
            target = z_data - x0
            z_est = x_t + (1.0 - t[:, None, None]) * v_pred

            cond_batch = {
                "tau_out": tau_out,
                "mem": mem,
                "g": g,
                "mem_mask": ~mask_cond,
            }
            z_solved = solve_flow(
                flow.flow,
                x0,
                cond_batch,
                num_steps=solver_steps,
                method=solver_method,
            )

            diff = _flatten(z_est - z_solved)
            z_solved_flat = _flatten(z_solved)
            z_data_flat = _flatten(z_data)
            z_est_flat = _flatten(z_est)
            x0_flat = _flatten(x0)
            solved_delta = z_solved_flat - x0_flat
            est_delta = z_est_flat - x0_flat

            l2 = torch.linalg.norm(diff, dim=1)
            rmse = torch.sqrt(torch.mean(diff**2, dim=1))
            solved_norm = torch.linalg.norm(z_solved_flat, dim=1).clamp_min(eps)
            target_norm = torch.linalg.norm(z_data_flat, dim=1).clamp_min(eps)
            est_to_target = torch.linalg.norm(z_est_flat - z_data_flat, dim=1)
            solved_to_target = torch.linalg.norm(z_solved_flat - z_data_flat, dim=1)
            cos = torch.nn.functional.cosine_similarity(est_delta, solved_delta, dim=1)
            mse = torch.mean((v_pred - target) ** 2, dim=(1, 2))

            gap_l2.extend(l2.detach().cpu().tolist())
            gap_rmse.extend(rmse.detach().cpu().tolist())
            rel_gap_solved.extend((l2 / solved_norm).detach().cpu().tolist())
            rel_gap_target.extend((l2 / target_norm).detach().cpu().tolist())
            est_target_l2.extend(est_to_target.detach().cpu().tolist())
            solved_target_l2.extend(solved_to_target.detach().cpu().tolist())
            base_mse.extend(mse.detach().cpu().tolist())
            endpoint_cos.extend(cos.detach().cpu().tolist())

    row: dict[str, float | int | str | None] = {
        "run_id": spec.run_id,
        "epoch": spec.epoch,
        "saved_epoch": saved_epoch,
        "checkpoint": spec.path,
        "state_key": key_used,
        "solver_steps": solver_steps,
        "solver_method": solver_method,
        "num_samples": len(gap_l2),
    }
    for prefix, values in (
        ("gap_l2", gap_l2),
        ("gap_rmse", gap_rmse),
        ("rel_gap_solved", rel_gap_solved),
        ("rel_gap_target", rel_gap_target),
        ("est_target_l2", est_target_l2),
        ("solved_target_l2", solved_target_l2),
        ("base_velocity_mse", base_mse),
        ("endpoint_cos", endpoint_cos),
    ):
        stats = _tensor_stats(values)
        for key, value in stats.items():
            row[f"{prefix}_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--state-key", choices=("auto", "ema", "model"), default="auto")
    parser.add_argument("--seq-len", type=int, default=196)
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--aist-cameras", type=str, default="01")
    parser.add_argument("--aist-crop-mode", choices=("first", "random", "uniform"), default="first")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cond-frames", type=int, default=196)
    parser.add_argument("--cond-drop-prob", type=float, default=0.0)
    parser.add_argument("--solver-steps", type=int, default=16)
    parser.add_argument("--solver-method", choices=("euler", "heun"), default="euler")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stats-path", type=str, default="data/mean_std_263_train.npz")
    parser.add_argument("--openpose-stats-path", type=str, default="data/openpose_stats.npz")
    parser.add_argument(
        "--latent-stats-path",
        type=str,
        default="data/vqvae_latent_stats_aist_train_latent16_epoch800.npz",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default="checkpoints/vqvae/aist_mvh_len196_latent16_code1024_visible_ddp_260618-153911/motion_vqvae_latest.pt",
    )
    parser.add_argument("--vae-type", choices=("auto", "motion_vae", "motion_vqvae"), default="motion_vqvae")
    parser.add_argument("--vae-latent-len", type=int, default=None)
    parser.add_argument("--vae-latent-token-mode", choices=("pool", "query"), default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Defaults to training_logs/solver_endpoint_gap_<timestamp>.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    _seed_all(args.seed, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    config = load_config()
    config["seq_len"] = args.seq_len
    mean, std = load_mean_std(args.stats_path)
    genre_to_id = build_genre_to_id(config.get("aist_genres", []))
    aist_paths = _aist_paths_for_splits(config, args.splits)
    cameras = [item.strip() for item in args.aist_cameras.split(",") if item.strip()]
    dataset = AISTDataset(
        config["aist_motions_dir"],
        genre_to_id=genre_to_id,
        seq_len=args.seq_len,
        mean=mean,
        std=std,
        files=aist_paths,
        cache_root=config["cache_root"],
        target_fps=config.get("target_fps", 30),
        src_fps=config.get("aist_fps", 60),
        camera_ids=cameras,
        expand_cameras=True,
        include_cond=True,
        openpose_dir=config.get("aist_openpose_dir", "data/AIST++/Annotations/openpose"),
        cond_cache_root=config.get("cond_cache_root", "data/cached_cond"),
        cond_frames_min=args.cond_frames,
        cond_frames_max=args.cond_frames,
        cond_drop_prob=args.cond_drop_prob,
        cond_frame_drop_prob=0.0,
        crop_mode=args.aist_crop_mode,
    )

    if args.num_samples > 0 and args.num_samples < len(dataset):
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(len(dataset), size=args.num_samples, replace=False))
        eval_dataset = Subset(dataset, indices.tolist())
    else:
        indices = np.arange(len(dataset))
        eval_dataset = dataset

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    op_stats = np.load(args.openpose_stats_path)
    k2d_mean = op_stats["mean"]
    k2d_std = op_stats["std"]

    latent_mean = None
    latent_std = None
    if args.latent_stats_path and os.path.exists(args.latent_stats_path):
        z_stats = np.load(args.latent_stats_path)
        latent_mean = torch.tensor(z_stats["mean"], dtype=torch.float32, device=device)
        latent_std = torch.tensor(z_stats["std"], dtype=torch.float32, device=device)

    loaded_vae = load_vae_backend(
        args.vae_ckpt,
        config,
        device,
        seq_len=args.seq_len,
        vae_type=args.vae_type,
        latent_len=args.vae_latent_len,
        latent_token_mode=args.vae_latent_token_mode,
    )
    vae = loaded_vae.model
    for param in vae.parameters():
        param.requires_grad = False
    flow = _make_flow(config, loaded_vae.d_z, device)

    specs = [_parse_checkpoint_spec(raw) for raw in args.checkpoint]
    rows = []
    for spec in specs:
        row = _evaluate_checkpoint(
            spec,
            flow,
            vae,
            loader,
            config,
            device,
            latent_mean=latent_mean,
            latent_std=latent_std,
            k2d_mean=k2d_mean,
            k2d_std=k2d_std,
            latent_len=loaded_vae.latent_len,
            solver_steps=args.solver_steps,
            solver_method=args.solver_method,
            state_key=args.state_key,
            seed=args.seed,
        )
        rows.append(row)
        print(
            "epoch={epoch:>4} run={run_id} gap_rmse={gap_rmse_mean:.6f} "
            "rel={rel_gap_solved_mean:.4f} est_to_target={est_target_l2_mean:.4f} "
            "solved_to_target={solved_target_l2_mean:.4f} cos={endpoint_cos_mean:.4f}".format(
                **row
            )
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"training_logs/solver_endpoint_gap_{stamp}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "endpoint_gap.csv")
    json_path = os.path.join(out_dir, "endpoint_gap.json")
    meta_path = os.path.join(out_dir, "metadata.json")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    metadata = {
        "args": vars(args),
        "device": str(device),
        "dataset_size": len(dataset),
        "selected_indices": indices.tolist(),
        "vae": {
            "ckpt": args.vae_ckpt,
            "type": loaded_vae.vae_type,
            "latent_len": loaded_vae.latent_len,
            "d_z": loaded_vae.d_z,
            "max_len": loaded_vae.max_len,
        },
        "checkpoints": [asdict(spec) for spec in specs],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
