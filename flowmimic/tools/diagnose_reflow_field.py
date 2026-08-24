"""Diagnose reflow errors on straight teacher paths and student rollout states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.checkpoint import (
    flow_state_uses_latent_slot_adapter,
    flow_state_uses_relative_time_bias,
    flow_state_uses_true_null_condition,
    infer_latent_slot_adapter_config,
    infer_relative_time_hidden_dim,
    load_flow_state_dict,
)
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.solver import combine_cfg_velocities, solve_flow
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id
from flowmimic.src.model.vae.stats import load_mean_std


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: str


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _parse_model_spec(raw: str) -> ModelSpec:
    if "=" in raw:
        label, path = raw.split("=", 1)
        if not label or not path:
            raise ValueError(f"Invalid model specification: {raw}")
        return ModelSpec(label=label, path=path)
    path = raw
    return ModelSpec(label=Path(path).parent.name, path=path)


def _parse_times(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("Flow times must start at 0 and end at 1")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Flow times must lie in [0, 1]")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("Flow times must be strictly increasing")
    return values


def _read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _aist_paths(config: dict, splits: str) -> list[str]:
    paths: list[str] = []
    for split in splits.split(","):
        split = split.strip().lower()
        if not split:
            continue
        split_path = config.get(
            f"aist_split_{split}",
            f"data/AIST++/Annotations/splits/pose_{split}.txt",
        )
        paths.extend(
            os.path.join(config["aist_motions_dir"], f"{name}.pkl")
            for name in _read_lines(split_path)
        )
    return list(dict.fromkeys(paths))


def _load_manifest_entries(path: str | None) -> dict[str, list[int]] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Condition manifest has no entries mapping: {path}")
    return entries


def _checkpoint_state(
    path: str,
    device: torch.device,
    state_key: str,
) -> tuple[dict, dict[str, torch.Tensor], str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a checkpoint mapping: {path}")
    selected_key = state_key
    if selected_key == "auto":
        selected_key = "ema" if "ema" in checkpoint else "model"
    if selected_key not in checkpoint:
        raise KeyError(f"Checkpoint {path} has no '{selected_key}' state")
    return checkpoint, checkpoint[selected_key], selected_key


def _build_flow(
    config: dict,
    checkpoint: dict,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> ConditionalRectFlow:
    metadata = checkpoint.get("metadata", {})
    flow_cfg = config.get("flow", {})
    architecture = metadata.get("flow_architecture", {})
    latent_len = int(metadata.get("latent_len", 16))
    d_z = int(metadata.get("d_z", 256))
    slot_config = infer_latent_slot_adapter_config(
        state,
        default_latent_len=latent_len,
        default_ffn_dim=int(flow_cfg.get("latent_slot_adapter_ffn_dim", 1024)),
    )
    model = ConditionalRectFlow(
        d_z=d_z,
        d_model=int(flow_cfg.get("d_model", 512)),
        n_layers=int(flow_cfg.get("n_layers", 8)),
        n_heads=int(flow_cfg.get("n_heads", 8)),
        ffn_dim=int(flow_cfg.get("ffn_dim", 2048)),
        dropout=float(flow_cfg.get("dropout", 0.1)),
        num_styles=int(config["num_styles"]),
        style_dim=int(flow_cfg.get("style_dim", 32)),
        cond_dim=int(flow_cfg.get("cond_dim", 256)),
        cond_layers=int(flow_cfg.get("cond_layers", 4)),
        cond_heads=int(flow_cfg.get("cond_heads", 4)),
        p_style_drop=float(flow_cfg.get("p_style_drop", 0.5)),
        relative_time_bias=flow_state_uses_relative_time_bias(state),
        relative_time_hidden_dim=infer_relative_time_hidden_dim(state),
        latent_len=slot_config["latent_len"],
        latent_slot_adapter=flow_state_uses_latent_slot_adapter(state),
        latent_slot_adapter_heads=int(
            architecture.get(
                "latent_slot_adapter_heads",
                flow_cfg.get("latent_slot_adapter_heads", 8),
            )
        ),
        latent_slot_adapter_ffn_dim=slot_config["ffn_dim"],
        true_null_condition=flow_state_uses_true_null_condition(state),
    ).to(device)
    load_flow_state_dict(model, state)
    model.eval().requires_grad_(False)
    return model


def _build_guided_condition(
    model: ConditionalRectFlow,
    batch: dict[str, torch.Tensor],
    tau_out: torch.Tensor,
    k2d_mean: np.ndarray,
    k2d_std: np.ndarray,
    guidance_scale: float,
) -> dict[str, torch.Tensor | float]:
    k2d = batch["k2d"]
    vis = batch["vis"]
    tau_cond = batch["tau_cond"]
    mask_cond = batch["mask_cond"]
    style_id = batch["style_id"]
    domain_id = batch["domain_id"]
    g2d, memory, _ = model.cond_encoder(
        k2d,
        tau_cond,
        vis_mask=vis,
        mask_cond=mask_cond,
        mean=k2d_mean,
        std=k2d_std,
    )
    style = model.style_emb(style_id, domain_id, apply_dropout=False)
    global_context = model.cond_mlp(torch.cat([g2d, style], dim=-1))
    if not model.true_null_condition:
        raise RuntimeError("Reflow diagnostics require learned true-null conditioning")
    null_g, null_memory, null_mask, null_tau = model.encode_null_condition(
        style_id,
        domain_id,
    )
    return {
        "tau_out": tau_out,
        "tau_cond": tau_cond,
        "mem": memory,
        "g": global_context,
        "mem_mask": ~mask_cond,
        "mem_uncond": null_memory,
        "g_uncond": null_g,
        "mem_mask_uncond": null_mask,
        "tau_cond_uncond": null_tau,
        "guidance_scale": guidance_scale,
    }


def _guided_velocity(
    model: ConditionalRectFlow,
    x: torch.Tensor,
    t: torch.Tensor,
    condition: dict[str, torch.Tensor | float],
) -> torch.Tensor:
    v_cond = model.flow(
        x,
        t,
        condition["tau_out"],
        condition["mem"],
        condition["g"],
        mem_mask=condition["mem_mask"],
        tau_cond=condition["tau_cond"],
    )
    v_null = model.flow(
        x,
        t,
        condition["tau_out"],
        condition["mem_uncond"],
        condition["g_uncond"],
        mem_mask=condition["mem_mask_uncond"],
        tau_cond=condition["tau_cond_uncond"],
    )
    return combine_cfg_velocities(
        v_cond,
        v_null,
        condition["guidance_scale"],
    )


def heun_advance(
    velocity_fn,
    x: torch.Tensor,
    t_start: float,
    t_end: float,
) -> torch.Tensor:
    """Advance a state over one possibly nonuniform Heun interval."""
    dt = float(t_end - t_start)
    if dt <= 0.0:
        raise ValueError("Heun interval must have positive width")
    t0 = torch.full((x.shape[0],), t_start, dtype=x.dtype, device=x.device)
    t1 = torch.full((x.shape[0],), t_end, dtype=x.dtype, device=x.device)
    v0 = velocity_fn(x, t0)
    proposal = x + dt * v0
    v1 = velocity_fn(proposal, t1)
    return x + 0.5 * dt * (v0 + v1)


def _per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.mean((left - right) ** 2, dim=tuple(range(1, left.ndim)))


def _per_sample_rmse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_per_sample_mse(left, right))


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": math.nan, "std": math.nan, "conf95": math.nan}
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": std,
        "conf95": float(1.96 * std / math.sqrt(array.size)),
    }


def _move_condition_batch(batch: dict, device: torch.device) -> dict:
    required = ("k2d", "vis", "tau_cond", "mask_cond", "style_id", "domain_id")
    return {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in required
    }


def _evaluate(
    teacher: ConditionalRectFlow,
    students: list[tuple[ModelSpec, ConditionalRectFlow, str]],
    loader: DataLoader,
    device: torch.device,
    *,
    latent_len: int,
    d_z: int,
    k2d_mean: np.ndarray,
    k2d_std: np.ndarray,
    guidance_scale: float,
    teacher_steps: int,
    teacher_solver: str,
    flow_times: list[float],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    _seed_all(seed, device)
    tau_out = torch.linspace(0.0, 1.0, latent_len, device=device)
    metric_names = (
        "straight_target_mse",
        "straight_target_rmse",
        "teacher_straight_target_mse",
        "straight_student_teacher_mse",
        "rollout_target_mse",
        "rollout_target_rmse",
        "rollout_student_teacher_mse",
        "rollout_state_rmse",
    )
    samples: dict[str, dict[float, dict[str, list[float]]]] = {
        spec.label: {
            time_value: {name: [] for name in metric_names}
            for time_value in flow_times
        }
        for spec, _, _ in students
    }

    with torch.inference_mode():
        for raw_batch in tqdm(loader, desc=f"CFG {guidance_scale:g} reflow diagnostics"):
            batch = _move_condition_batch(raw_batch, device)
            batch_size = batch["style_id"].shape[0]
            x0 = torch.randn(batch_size, latent_len, d_z, device=device)
            teacher_condition = _build_guided_condition(
                teacher,
                batch,
                tau_out,
                k2d_mean,
                k2d_std,
                guidance_scale,
            )
            teacher_endpoint = solve_flow(
                teacher.flow,
                x0,
                teacher_condition,
                num_steps=teacher_steps,
                method=teacher_solver,
            )
            target_velocity = teacher_endpoint - x0

            for spec, student, _ in students:
                student_condition = _build_guided_condition(
                    student,
                    batch,
                    tau_out,
                    k2d_mean,
                    k2d_std,
                    guidance_scale,
                )
                rollout = x0.clone()
                previous_time = flow_times[0]
                for time_index, time_value in enumerate(flow_times):
                    if time_index > 0:
                        rollout = heun_advance(
                            lambda state, time: _guided_velocity(
                                student,
                                state,
                                time,
                                student_condition,
                            ),
                            rollout,
                            previous_time,
                            time_value,
                        )
                    t = torch.full(
                        (batch_size,),
                        time_value,
                        dtype=x0.dtype,
                        device=device,
                    )
                    straight = x0 + time_value * target_velocity
                    student_straight = _guided_velocity(
                        student,
                        straight,
                        t,
                        student_condition,
                    )
                    teacher_straight = _guided_velocity(
                        teacher,
                        straight,
                        t,
                        teacher_condition,
                    )
                    student_rollout = _guided_velocity(
                        student,
                        rollout,
                        t,
                        student_condition,
                    )
                    teacher_rollout = _guided_velocity(
                        teacher,
                        rollout,
                        t,
                        teacher_condition,
                    )
                    values = {
                        "straight_target_mse": _per_sample_mse(
                            student_straight, target_velocity
                        ),
                        "straight_target_rmse": _per_sample_rmse(
                            student_straight, target_velocity
                        ),
                        "teacher_straight_target_mse": _per_sample_mse(
                            teacher_straight, target_velocity
                        ),
                        "straight_student_teacher_mse": _per_sample_mse(
                            student_straight, teacher_straight
                        ),
                        "rollout_target_mse": _per_sample_mse(
                            student_rollout, target_velocity
                        ),
                        "rollout_target_rmse": _per_sample_rmse(
                            student_rollout, target_velocity
                        ),
                        "rollout_student_teacher_mse": _per_sample_mse(
                            student_rollout, teacher_rollout
                        ),
                        "rollout_state_rmse": _per_sample_rmse(rollout, straight),
                    }
                    target = samples[spec.label][time_value]
                    for name, tensor in values.items():
                        target[name].extend(tensor.cpu().tolist())
                    previous_time = time_value

    rows: list[dict] = []
    conclusions: list[dict] = []
    for spec, _, state_key in students:
        for time_value in flow_times:
            row = {
                "label": spec.label,
                "checkpoint": spec.path,
                "state_key": state_key,
                "guidance_scale": guidance_scale,
                "flow_time": time_value,
                "num_samples": len(
                    samples[spec.label][time_value]["straight_target_mse"]
                ),
            }
            for metric_name, metric_values in samples[spec.label][time_value].items():
                for stat_name, stat_value in _summarize(metric_values).items():
                    row[f"{metric_name}_{stat_name}"] = stat_value
            straight = row["straight_target_mse_mean"]
            rollout = row["rollout_target_mse_mean"]
            row["rollout_to_straight_mse_ratio"] = rollout / max(straight, 1e-12)
            rows.append(row)

        selected = [row for row in rows if row["label"] == spec.label]
        endpoint = selected[-1]
        interior = [row for row in selected if 0.0 < row["flow_time"] < 1.0]
        conclusions.append(
            {
                "label": spec.label,
                "checkpoint": spec.path,
                "guidance_scale": guidance_scale,
                "max_straight_target_mse": max(
                    row["straight_target_mse_mean"] for row in selected
                ),
                "max_straight_target_mse_time": max(
                    selected,
                    key=lambda row: row["straight_target_mse_mean"],
                )["flow_time"],
                "endpoint_straight_target_mse_t0": selected[0][
                    "straight_target_mse_mean"
                ],
                "endpoint_straight_target_mse_t1": endpoint[
                    "straight_target_mse_mean"
                ],
                "mean_interior_straight_target_mse": float(
                    np.mean([row["straight_target_mse_mean"] for row in interior])
                ),
                "mean_interior_rollout_target_mse": float(
                    np.mean([row["rollout_target_mse_mean"] for row in interior])
                ),
                "mean_interior_rollout_to_straight_ratio": float(
                    np.mean([row["rollout_to_straight_mse_ratio"] for row in interior])
                ),
                "final_rollout_state_rmse": endpoint["rollout_state_rmse_mean"],
            }
        )
    return rows, conclusions


def _write_plot(rows: list[dict], path: str) -> None:
    import matplotlib.pyplot as plt

    panels = (
        ("straight_target_mse_mean", "Straight-state target MSE"),
        ("rollout_target_mse_mean", "Rollout-state target MSE"),
        ("rollout_to_straight_mse_ratio", "Rollout / straight MSE"),
        ("rollout_state_rmse_mean", "Rollout state RMSE"),
    )
    labels = list(dict.fromkeys(row["label"] for row in rows))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        for label in labels:
            selected = [row for row in rows if row["label"] == label]
            axis.plot(
                [row["flow_time"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Flow time")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--student-checkpoint", action="append", required=True)
    parser.add_argument("--teacher-state-key", choices=("auto", "ema", "model"), default="ema")
    parser.add_argument("--student-state-key", choices=("auto", "ema", "model"), default="ema")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument("--teacher-solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--flow-times", default="0,0.05,0.15,0.3,0.5,0.7,0.85,0.95,1")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--camera", default="01")
    parser.add_argument("--crop-mode", choices=("first", "random", "uniform"), default="first")
    parser.add_argument("--cond-frames", type=int, default=7)
    parser.add_argument("--cond-pattern", default="boundary_gap")
    parser.add_argument("--cond-pattern-seed", type=int, default=20260720)
    parser.add_argument("--condition-manifest", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stats-path", default="data/mean_std_263_train.npz")
    parser.add_argument("--openpose-stats-path", default="data/openpose_stats.npz")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    _seed_all(args.seed, device)
    flow_times = _parse_times(args.flow_times)
    student_specs = [_parse_model_spec(raw) for raw in args.student_checkpoint]

    config = load_config()
    teacher_checkpoint, teacher_state, teacher_key = _checkpoint_state(
        args.teacher_ckpt,
        device,
        args.teacher_state_key,
    )
    teacher = _build_flow(config, teacher_checkpoint, teacher_state, device)
    teacher_metadata = teacher_checkpoint.get("metadata", {})
    latent_len = int(teacher_metadata.get("latent_len", 16))
    d_z = int(teacher_metadata.get("d_z", 256))

    students: list[tuple[ModelSpec, ConditionalRectFlow, str]] = []
    for spec in student_specs:
        checkpoint, state, state_key = _checkpoint_state(
            spec.path,
            device,
            args.student_state_key,
        )
        metadata = checkpoint.get("metadata", {})
        if int(metadata.get("latent_len", latent_len)) != latent_len:
            raise ValueError(f"Latent length mismatch for {spec.path}")
        if int(metadata.get("d_z", d_z)) != d_z:
            raise ValueError(f"Latent width mismatch for {spec.path}")
        students.append((spec, _build_flow(config, checkpoint, state, device), state_key))

    motion_mean, motion_std = load_mean_std(args.stats_path)
    op_stats = np.load(args.openpose_stats_path)
    manifest_entries = _load_manifest_entries(args.condition_manifest)
    dataset = AISTDataset(
        config["aist_motions_dir"],
        genre_to_id=build_genre_to_id(config.get("aist_genres", [])),
        seq_len=int(teacher_metadata.get("seq_len", 196)),
        mean=motion_mean,
        std=motion_std,
        files=_aist_paths(config, args.splits),
        cache_root=config["cache_root"],
        target_fps=config.get("target_fps", 30),
        src_fps=config.get("aist_fps", 60),
        camera_ids=[args.camera],
        expand_cameras=True,
        include_cond=True,
        openpose_dir=config.get("aist_openpose_dir"),
        cond_cache_root=config.get("cond_cache_root"),
        cond_frames_min=args.cond_frames,
        cond_frames_max=args.cond_frames,
        cond_pattern=args.cond_pattern,
        cond_pattern_seed=args.cond_pattern_seed,
        cond_index_manifest=manifest_entries,
        crop_mode=args.crop_mode,
    )
    selected_indices = np.arange(len(dataset))
    if 0 < args.num_samples < len(dataset):
        rng = np.random.default_rng(args.seed)
        selected_indices = np.sort(
            rng.choice(len(dataset), size=args.num_samples, replace=False)
        )
        eval_dataset = Subset(dataset, selected_indices.tolist())
    else:
        eval_dataset = dataset
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    rows, conclusions = _evaluate(
        teacher,
        students,
        loader,
        device,
        latent_len=latent_len,
        d_z=d_z,
        k2d_mean=op_stats["mean"],
        k2d_std=op_stats["std"],
        guidance_scale=args.guidance_scale,
        teacher_steps=args.teacher_steps,
        teacher_solver=args.teacher_solver,
        flow_times=flow_times,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "reflow_field_diagnostics.csv")
    json_path = os.path.join(args.out_dir, "reflow_field_diagnostics.json")
    plot_path = os.path.join(args.out_dir, "reflow_field_diagnostics.png")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": {
            "teacher_checkpoint": args.teacher_ckpt,
            "teacher_state_key": teacher_key,
            "teacher_steps": args.teacher_steps,
            "teacher_solver": args.teacher_solver,
            "guidance_scale": args.guidance_scale,
            "flow_times": flow_times,
            "splits": args.splits,
            "camera": args.camera,
            "crop_mode": args.crop_mode,
            "cond_frames": args.cond_frames,
            "cond_pattern": args.cond_pattern,
            "cond_pattern_seed": args.cond_pattern_seed,
            "condition_manifest": args.condition_manifest,
            "num_samples": len(eval_dataset),
            "selected_indices": selected_indices.tolist(),
            "seed": args.seed,
            "device": str(device),
        },
        "conclusions": conclusions,
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    _write_plot(rows, plot_path)
    for conclusion in conclusions:
        print(
            "{label}: t0_mse={endpoint_straight_target_mse_t0:.6f} "
            "t1_mse={endpoint_straight_target_mse_t1:.6f} "
            "interior_straight={mean_interior_straight_target_mse:.6f} "
            "interior_rollout={mean_interior_rollout_target_mse:.6f} "
            "rollout_ratio={mean_interior_rollout_to_straight_ratio:.3f} "
            "endpoint_state_rmse={final_rollout_state_rmse:.6f}".format(**conclusion)
        )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
