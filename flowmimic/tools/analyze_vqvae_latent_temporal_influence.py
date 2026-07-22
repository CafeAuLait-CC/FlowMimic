#!/usr/bin/env python3
"""Measure how each VQ-VAE latent slot influences reconstructed motion frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flowmimic.scripts.train_vqvae import _ik263_to_smpl22_torch
from flowmimic.src.config.config import load_config
from flowmimic.src.model.vae.backend import load_vae_backend
from flowmimic.src.model.vae.datasets.aist_filename_parser import get_genre_code
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id


CONT_END = 259


def _load_names(data_root: Path, splits: str) -> list[str]:
    names: list[str] = []
    for split in splits.split(","):
        split = split.strip()
        if not split:
            continue
        split_path = data_root / f"{split}.txt"
        names.extend(
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return list(dict.fromkeys(names))


def _pick_names(names: list[str], count: int, seed: int) -> list[str]:
    if count <= 0 or count >= len(names):
        return names
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(names), size=count, replace=False))
    return [names[int(index)] for index in indices]


def _to_physical_joints(
    decoded: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    physical = decoded.clone()
    physical[..., :CONT_END] = physical[..., :CONT_END] * std + mean
    return _ik263_to_smpl22_torch(physical)


def _row_normalize(values: np.ndarray) -> np.ndarray:
    denom = values.sum(axis=-1, keepdims=True)
    return values / np.maximum(denom, 1e-12)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for group, count in enumerate(counts):
        if count > 1:
            members = np.flatnonzero(inverse == group)
            ranks[members] = ranks[members].mean()
    return ranks


def _find_local_peaks(
    values: np.ndarray,
    *,
    max_peaks: int = 4,
    sigma: float = 2.0,
    min_distance: int = 10,
) -> list[int]:
    smoothed = gaussian_filter1d(
        np.asarray(values, dtype=np.float64), sigma=sigma, mode="nearest"
    )
    dynamic_range = float(smoothed.max() - smoothed.min())
    if dynamic_range <= 1e-12:
        return [int(np.argmax(smoothed))]
    min_prominence = max(0.08 * dynamic_range, 1e-12)
    indices, properties = find_peaks(
        smoothed,
        distance=min_distance,
        prominence=min_prominence,
    )
    candidates = [
        (int(index), float(prominence))
        for index, prominence in zip(indices, properties["prominences"])
    ]
    edge_width = min(min_distance + 1, smoothed.shape[0])
    for index, window in ((0, smoothed[:edge_width]), (-1, smoothed[-edge_width:])):
        resolved = index % smoothed.shape[0]
        edge_prominence = float(smoothed[resolved] - window.min())
        if smoothed[resolved] >= window.max() and edge_prominence >= min_prominence:
            candidates.append((resolved, edge_prominence))
    if not candidates:
        return [int(np.argmax(smoothed))]
    candidates = sorted(
        candidates,
        key=lambda item: (item[1], float(smoothed[item[0]])),
        reverse=True,
    )[:max_peaks]
    return sorted(index for index, _prominence in candidates)


def _local_peak_rows(values: np.ndarray) -> list[list[int]]:
    return [_find_local_peaks(row) for row in values]


def _summarize_influence(values: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    per_sample_normalized = _row_normalize(values)
    aggregate = per_sample_normalized.mean(axis=0)
    aggregate = _row_normalize(aggregate)
    frames = np.arange(values.shape[-1], dtype=np.float64)
    centers = (aggregate * frames[None]).sum(axis=-1)
    widths = np.sqrt(
        (aggregate * np.square(frames[None] - centers[:, None])).sum(axis=-1)
    )
    entropy = -(
        aggregate * np.log(np.maximum(aggregate, 1e-12))
    ).sum(axis=-1) / np.log(values.shape[-1])
    nominal = np.linspace(0.0, values.shape[-1] - 1.0, values.shape[-2])
    rank_corr = float(
        np.corrcoef(_rankdata(np.arange(values.shape[-2])), _rankdata(centers))[0, 1]
    )
    local_peaks = _local_peak_rows(aggregate)
    dominant_peaks = np.asarray(
        [max(peaks, key=lambda index: aggregate[slot, index]) for slot, peaks in enumerate(local_peaks)],
        dtype=np.float64,
    )
    dominant_peak_rank_corr = float(
        np.corrcoef(
            _rankdata(np.arange(values.shape[-2])),
            _rankdata(dominant_peaks),
        )[0, 1]
    )
    per_sample_dominant = gaussian_filter1d(
        values.astype(np.float64), sigma=2.0, axis=-1, mode="nearest"
    ).argmax(axis=-1)
    per_sample_peak_stats = []
    for slot in range(values.shape[-2]):
        slot_peaks = per_sample_dominant[:, slot]
        q25, median, q75 = np.percentile(slot_peaks, [25, 50, 75])
        per_sample_peak_stats.append(
            {
                "slot": slot,
                "q25_frame": float(q25),
                "median_frame": float(median),
                "q75_frame": float(q75),
                "fraction_at_or_after_150": float((slot_peaks >= 150).mean()),
                "fraction_at_or_after_180": float((slot_peaks >= 180).mean()),
            }
        )
    half_bin = (values.shape[-1] - 1.0) / max(2.0 * (values.shape[-2] - 1.0), 1.0)
    nominal_mass = []
    for slot, center in enumerate(nominal):
        mask = np.abs(frames - center) <= half_bin
        nominal_mass.append(float(aggregate[slot, mask].sum()))
    summary = {
        "centers": centers.tolist(),
        "widths": widths.tolist(),
        "normalized_entropy": entropy.tolist(),
        "nominal_centers": nominal.tolist(),
        "center_mae_from_nominal": float(np.abs(centers - nominal).mean()),
        "center_spearman_vs_slot": rank_corr,
        "local_peak_frames": local_peaks,
        "dominant_peak_frames": dominant_peaks.astype(int).tolist(),
        "dominant_peak_spearman_vs_slot": dominant_peak_rank_corr,
        "per_sample_dominant_peak_stats": per_sample_peak_stats,
        "mean_temporal_width_frames": float(widths.mean()),
        "mean_normalized_entropy": float(entropy.mean()),
        "mean_mass_near_nominal_center": float(np.mean(nominal_mass)),
        "mass_near_nominal_center": nominal_mass,
    }
    return aggregate, summary


def _plot_heatmap(
    axis,
    values: np.ndarray,
    title: str,
    nominal: np.ndarray,
    local_peaks: list[list[int]],
    *,
    colorbar: bool = True,
) -> None:
    image = axis.imshow(values, aspect="auto", origin="lower", cmap="magma")
    slots = np.arange(values.shape[0])
    axis.scatter(nominal, slots, marker="x", s=26, linewidths=1.2, color="white", label="linspace")
    peak_x = []
    peak_y = []
    for slot, peaks in enumerate(local_peaks):
        peak_x.extend(peaks)
        peak_y.extend([slot] * len(peaks))
    axis.scatter(peak_x, peak_y, marker="o", s=15, color="#55d6be", label="local peaks")
    axis.set_title(title)
    axis.set_xlabel("Output frame")
    axis.set_ylabel("VQ latent slot")
    axis.set_yticks(slots)
    if colorbar:
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def _save_aggregate_plot(
    output_path: Path,
    aggregates: dict[str, np.ndarray],
    summaries: dict[str, dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    labels = {
        "full_body": "Full-body joint influence",
        "root_relative": "Root-relative pose influence",
        "root_trajectory": "Root-trajectory influence",
    }
    for axis, key in zip(axes, labels):
        summary = summaries[key]
        _plot_heatmap(
            axis,
            aggregates[key],
            labels[key],
            np.asarray(summary["nominal_centers"]),
            summary["local_peak_frames"],
        )
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.suptitle("VQ-VAE latent-slot decoder influence (row-normalized)", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_sample_plot(
    output_path: Path,
    names: list[str],
    values: np.ndarray,
    count: int,
) -> None:
    count = min(count, len(names))
    if count <= 0:
        return
    selected = np.linspace(0, len(names) - 1, count, dtype=int)
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.2 * columns, 4.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    nominal = np.linspace(0.0, values.shape[-1] - 1.0, values.shape[-2])
    for axis, sample_index in zip(axes.flat, selected):
        normalized = _row_normalize(values[sample_index])
        _plot_heatmap(
            axis,
            normalized,
            names[sample_index],
            nominal,
            _local_peak_rows(normalized),
            colorbar=False,
        )
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    axes.flat[0].legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.suptitle("Per-sample full-body latent influence", fontsize=14)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _save_sample_absolute_plot(
    output_path: Path,
    names: list[str],
    values: np.ndarray,
    count: int,
    *,
    title: str,
    colorbar_label: str,
) -> None:
    count = min(count, len(names))
    if count <= 0:
        return
    selected = np.linspace(0, len(names) - 1, count, dtype=int)
    shown = values[selected]
    vmax = float(np.percentile(shown, 99.5))
    vmax = max(vmax, 1e-12)
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.2 * columns, 4.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for axis, sample_index in zip(axes.flat, selected):
        sample = values[sample_index]
        image = axis.imshow(
            sample,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(
            f"{names[sample_index]}\nmean={sample.mean():.4f}, max={sample.max():.4f}"
        )
        axis.set_xlabel("Output frame")
        axis.set_ylabel("VQ latent slot")
        axis.set_yticks(np.arange(values.shape[1]))
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    if image is not None:
        fig.colorbar(
            image,
            ax=[axis for axis in axes.flat[:count]],
            fraction=0.02,
            pad=0.02,
            label=colorbar_label,
        )
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _code_entropy(code_ids: np.ndarray) -> tuple[int, float]:
    _codes, counts = np.unique(code_ids, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    max_entropy = np.log(max(len(counts), 2))
    return len(counts), entropy / max_entropy


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vae-ckpt",
        default="checkpoints/vqvae/aist_mvh_len196_latent16_code1024_visible_retrain_to200_ddp2_retry_260717/motion_vqvae_epoch200.pt",
    )
    parser.add_argument(
        "--latent-stats-path",
        default="data/vqvae_latent_stats_aist_train_latent16_epoch200_retry.npz",
    )
    parser.add_argument("--motion-stats-path", default=None)
    parser.add_argument("--data-root", default="prepared/aist_mld_humanml3d")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--individual-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--replacement",
        choices=("mean", "permutation"),
        default="permutation",
        help="Use an off-manifold per-slot mean or a quantized slot from another sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/diagnostics/vqvae_latent_temporal_influence_epoch200",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = _pick_names(_load_names(data_root, args.splits), args.num_samples, args.seed)
    if not names:
        raise ValueError("No samples selected")

    backend = load_vae_backend(
        args.vae_ckpt,
        cfg,
        device,
        seq_len=196,
        vae_type="motion_vqvae",
    )
    vae = backend.model
    if getattr(vae, "latent_token_mode", None) != "query":
        raise ValueError("This diagnostic expects query-mode VQ latent tokens")
    motion_stats_path = (
        args.motion_stats_path
        or backend.ckpt.get("stats_path")
        or backend.ckpt.get("config", {}).get("stats_path")
        or cfg.get("stats_path")
    )
    if not motion_stats_path:
        raise ValueError("Motion mean/std path is unavailable")
    motion_stats = np.load(motion_stats_path)
    motion_mean = torch.from_numpy(motion_stats["mean"].astype(np.float32)).to(device)
    motion_std = torch.from_numpy(motion_stats["std"].astype(np.float32)).to(device)
    latent_stats = np.load(args.latent_stats_path)
    latent_mean = torch.from_numpy(latent_stats["mean"].astype(np.float32)).to(device)
    if latent_mean.shape != (backend.latent_len, backend.d_z):
        raise ValueError(
            f"Expected latent mean {(backend.latent_len, backend.d_z)}, got {tuple(latent_mean.shape)}"
        )

    genre_to_id = build_genre_to_id(cfg.get("aist_genres", []))
    motion_dir = data_root / "new_joint_vecs"
    influence_chunks = {key: [] for key in ("full_body", "root_relative", "root_trajectory")}
    latent_delta_chunks = []
    code_id_chunks = []

    for start in tqdm(range(0, len(names), args.batch_size), desc="Latent influence"):
        batch_names = names[start : start + args.batch_size]
        raw_np = np.stack(
            [np.load(motion_dir / f"{name}.npy").astype(np.float32) for name in batch_names]
        )
        if raw_np.shape[1:] != (196, 263):
            raise ValueError(f"Expected [B,196,263], got {raw_np.shape}")
        raw = torch.from_numpy(raw_np).to(device)
        motion = raw.clone()
        motion[..., :CONT_END] = (
            motion[..., :CONT_END] - motion_mean
        ) / (motion_std + 1e-6)
        domain_id = torch.ones(len(batch_names), dtype=torch.long, device=device)
        style_id = torch.tensor(
            [genre_to_id.get(get_genre_code(name), 0) for name in batch_names],
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones(motion.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode():
            cond = vae.cond(domain_id, style_id)
            _enc_h, _z_e, z_q, _code_ids, _vq_loss, _perplexity, _codes_used = vae.encode(
                motion,
                cond,
                mask=mask,
                update_codebook=False,
            )
            baseline = vae.decode(z_q, cond, mask=mask, out_len=196)
            baseline_joints = _to_physical_joints(baseline, motion_mean, motion_std)
            if args.replacement == "mean":
                replacement = latent_mean.unsqueeze(0).expand_as(z_q)
            else:
                replacement = z_q.roll(shifts=1, dims=0)
            latent_delta = torch.linalg.vector_norm(z_q - replacement, dim=-1)
            latent_delta_chunks.append(latent_delta.cpu().numpy().astype(np.float32))
            code_id_chunks.append(_code_ids.cpu().numpy())
            batch_influence = {
                key: torch.empty(
                    len(batch_names), backend.latent_len, 196, device="cpu"
                )
                for key in influence_chunks
            }
            for slot in range(backend.latent_len):
                ablated = z_q.clone()
                ablated[:, slot] = replacement[:, slot]
                decoded = vae.decode(ablated, cond, mask=mask, out_len=196)
                joints = _to_physical_joints(decoded, motion_mean, motion_std)
                joint_delta = joints - baseline_joints
                full_body = torch.linalg.vector_norm(joint_delta, dim=-1).mean(dim=-1)
                root_delta = joint_delta[:, :, :1]
                root_relative_delta = joint_delta - root_delta
                root_relative = torch.linalg.vector_norm(
                    root_relative_delta, dim=-1
                ).mean(dim=-1)
                root_trajectory = torch.linalg.vector_norm(
                    root_delta.squeeze(-2), dim=-1
                )
                batch_influence["full_body"][:, slot] = full_body.cpu()
                batch_influence["root_relative"][:, slot] = root_relative.cpu()
                batch_influence["root_trajectory"][:, slot] = root_trajectory.cpu()
        for key in influence_chunks:
            influence_chunks[key].append(batch_influence[key].numpy().astype(np.float32))

    influence = {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in influence_chunks.items()
    }
    latent_delta = np.concatenate(latent_delta_chunks, axis=0)
    code_ids = np.concatenate(code_id_chunks, axis=0)
    sensitivity = {
        key: values / np.maximum(latent_delta[..., None], 1e-6)
        for key, values in influence.items()
    }
    aggregates: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, object]] = {}
    for key, values in influence.items():
        aggregate, summary = _summarize_influence(values)
        aggregates[key] = aggregate
        summaries[key] = summary
        np.save(output_dir / f"{key}_influence.npy", values)
        np.save(output_dir / f"{key}_aggregate_row_normalized.npy", aggregate)
        np.save(output_dir / f"{key}_sensitivity.npy", sensitivity[key])
    np.save(output_dir / "latent_replacement_l2.npy", latent_delta)
    np.save(output_dir / "code_ids.npy", code_ids)

    _save_aggregate_plot(
        output_dir / "aggregate_latent_temporal_influence.png",
        aggregates,
        summaries,
    )
    _save_sample_plot(
        output_dir / "sample_full_body_latent_temporal_influence.png",
        names,
        influence["full_body"],
        args.individual_count,
    )
    _save_sample_absolute_plot(
        output_dir / "sample_full_body_latent_influence_absolute.png",
        names,
        influence["full_body"],
        args.individual_count,
        title="Per-sample full-body latent influence (shared absolute scale)",
        colorbar_label="mean SMPL22 displacement",
    )
    _save_sample_absolute_plot(
        output_dir / "sample_full_body_latent_sensitivity.png",
        names,
        sensitivity["full_body"],
        args.individual_count,
        title="Per-sample full-body decoder sensitivity (shared scale)",
        colorbar_label="displacement / latent replacement L2",
    )

    slot_utilization = []
    for slot in range(backend.latent_len):
        unique_codes, normalized_code_entropy = _code_entropy(code_ids[:, slot])
        slot_utilization.append(
            {
                "slot": slot,
                "unique_codes": unique_codes,
                "normalized_code_entropy": normalized_code_entropy,
                "mean_replacement_l2": float(latent_delta[:, slot].mean()),
                "mean_full_body_influence": float(influence["full_body"][:, slot].mean()),
                "mean_full_body_sensitivity": float(sensitivity["full_body"][:, slot].mean()),
            }
        )

    summary = {
        "protocol": {
            "vae_ckpt": args.vae_ckpt,
            "latent_stats_path": args.latent_stats_path,
            "motion_stats_path": str(motion_stats_path),
            "data_root": str(data_root),
            "splits": args.splits,
            "samples": len(names),
            "seed": args.seed,
            "replacement": args.replacement,
            "sequence_length": 196,
            "latent_length": backend.latent_len,
        },
        "metrics": summaries,
        "slot_utilization": slot_utilization,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "sample"])
        writer.writerows(enumerate(names))

    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostic to {output_dir}")


if __name__ == "__main__":
    main()
