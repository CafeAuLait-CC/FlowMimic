#!/usr/bin/env python3
"""Evaluate a MotionVQVAE reconstruction ceiling with the AIST T2M encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flowmimic.scripts.eval_flow import _extract_t2m_features, _renorm_for_t2m
from flowmimic.src.config.config import load_config
from flowmimic.src.metrics import T2MMotionFeatureExtractor, summarize_motion_feature_metrics
from flowmimic.src.model.vae.backend import load_vae_backend
from flowmimic.src.model.vae.datasets.aist_filename_parser import get_genre_code
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id
from flowmimic.src.motion.process_motion import ik263_to_smpl22


CONT_END = 259
DISTAL_JOINTS = (10, 11, 20, 21)


def _load_names(data_root: Path, split: str) -> list[str]:
    path = data_root / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-8))


def _physical_reconstruction_metrics(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    fps: float,
    fast_quantile: float,
) -> dict[str, float | int]:
    diff = reconstructed - reference
    joint_l2 = np.linalg.norm(diff, axis=-1)
    distal_l2 = joint_l2[..., list(DISTAL_JOINTS)]
    reconstructed_root_relative = reconstructed - reconstructed[:, :, :1]
    reference_root_relative = reference - reference[:, :, :1]
    root_relative_l2 = np.linalg.norm(
        reconstructed_root_relative - reference_root_relative, axis=-1
    )
    root_relative_distal_l2 = root_relative_l2[..., list(DISTAL_JOINTS)]

    reconstructed_velocity = np.diff(reconstructed, axis=1) * fps
    reference_velocity = np.diff(reference, axis=1) * fps
    velocity_error = np.linalg.norm(
        reconstructed_velocity - reference_velocity, axis=-1
    )
    root_velocity_error = velocity_error[..., 0]
    reconstructed_distal_velocity = reconstructed_velocity[
        ..., list(DISTAL_JOINTS), :
    ]
    reference_distal_velocity = reference_velocity[..., list(DISTAL_JOINTS), :]
    distal_velocity_error = np.linalg.norm(
        reconstructed_distal_velocity - reference_distal_velocity, axis=-1
    )
    reconstructed_distal_speed = np.linalg.norm(
        reconstructed_distal_velocity, axis=-1
    )
    reference_distal_speed = np.linalg.norm(
        reference_distal_velocity, axis=-1
    )
    fast_threshold = float(np.quantile(reference_distal_speed, fast_quantile))
    fast_events = reference_distal_speed >= fast_threshold

    per_clip_fast_score = np.quantile(reference_distal_speed, 0.9, axis=(1, 2))
    fast_clip_threshold = float(np.quantile(per_clip_fast_score, fast_quantile))
    fast_clips = per_clip_fast_score >= fast_clip_threshold

    reconstructed_root_xz = reconstructed[:, :, 0][:, :, [0, 2]]
    reference_root_xz = reference[:, :, 0][:, :, [0, 2]]
    reconstructed_root_drift = reconstructed_root_xz[:, -1] - reconstructed_root_xz[:, 0]
    reference_root_drift = reference_root_xz[:, -1] - reference_root_xz[:, 0]
    root_drift_l2 = np.linalg.norm(
        reconstructed_root_drift - reference_root_drift, axis=-1
    )
    reconstructed_root_path = np.linalg.norm(
        np.diff(reconstructed_root_xz, axis=1), axis=-1
    ).sum(axis=1)
    reference_root_path = np.linalg.norm(
        np.diff(reference_root_xz, axis=1), axis=-1
    ).sum(axis=1)

    reconstructed_accel = np.diff(reconstructed_velocity, axis=1) * fps
    reference_accel = np.diff(reference_velocity, axis=1) * fps
    reconstructed_jerk = np.diff(reconstructed_accel, axis=1) * fps
    reference_jerk = np.diff(reference_accel, axis=1) * fps
    reconstructed_accel_norm = np.linalg.norm(reconstructed_accel, axis=-1)
    reference_accel_norm = np.linalg.norm(reference_accel, axis=-1)
    reconstructed_jerk_norm = np.linalg.norm(reconstructed_jerk, axis=-1)
    reference_jerk_norm = np.linalg.norm(reference_jerk, axis=-1)
    accel_error = np.linalg.norm(reconstructed_accel - reference_accel, axis=-1)
    jerk_error = np.linalg.norm(reconstructed_jerk - reference_jerk, axis=-1)
    accel_p90 = float(np.percentile(reconstructed_accel_norm, 90))
    accel_p90_ref = float(np.percentile(reference_accel_norm, 90))
    jerk_p90 = float(np.percentile(reconstructed_jerk_norm, 90))
    jerk_p90_ref = float(np.percentile(reference_jerk_norm, 90))

    reconstructed_spectrum = np.abs(
        np.fft.rfft(reconstructed_distal_velocity, axis=1)
    )
    reference_spectrum = np.abs(np.fft.rfft(reference_distal_velocity, axis=1))
    spectrum_error = np.abs(reconstructed_spectrum - reference_spectrum)
    frequencies = np.fft.rfftfreq(reconstructed_distal_velocity.shape[1], d=1.0 / fps)
    high_frequency_mask = frequencies >= 3.0
    reconstructed_high_frequency_energy = np.square(
        reconstructed_spectrum[:, high_frequency_mask]
    ).sum()
    reference_high_frequency_energy = np.square(
        reference_spectrum[:, high_frequency_mask]
    ).sum()

    return {
        "joint_l2": float(joint_l2.mean()),
        "distal_l2": float(distal_l2.mean()),
        "root_relative_joint_l2": float(root_relative_l2.mean()),
        "root_relative_distal_l2": float(root_relative_distal_l2.mean()),
        "joint_velocity_l2": float(velocity_error.mean()),
        "root_velocity_l2": float(root_velocity_error.mean()),
        "distal_velocity_l2": float(distal_velocity_error.mean()),
        "root_endpoint_drift_l2": float(root_drift_l2.mean()),
        "root_path_ratio": _safe_ratio(
            reconstructed_root_path.sum(), reference_root_path.sum()
        ),
        "distal_speed_ratio": _safe_ratio(
            reconstructed_distal_speed.sum(), reference_distal_speed.sum()
        ),
        "fast_event_quantile": float(fast_quantile),
        "fast_event_speed_threshold": fast_threshold,
        "fast_event_count": int(fast_events.sum()),
        "fast_event_distal_speed_ratio": _safe_ratio(
            reconstructed_distal_speed[fast_events].sum(),
            reference_distal_speed[fast_events].sum(),
        ),
        "fast_event_distal_velocity_l2": float(
            distal_velocity_error[fast_events].mean()
        ),
        "fast_clip_count": int(fast_clips.sum()),
        "fast_clip_distal_speed_ratio": _safe_ratio(
            reconstructed_distal_speed[fast_clips].sum(),
            reference_distal_speed[fast_clips].sum(),
        ),
        "fast_clip_distal_velocity_l2": float(
            distal_velocity_error[fast_clips].mean()
        ),
        "accel_l2": float(accel_error.mean()),
        "jerk_l2": float(jerk_error.mean()),
        "accel_p90": accel_p90,
        "accel_p90_ref": accel_p90_ref,
        "accel_p90_ratio": _safe_ratio(accel_p90, accel_p90_ref),
        "jerk_p90": jerk_p90,
        "jerk_p90_ref": jerk_p90_ref,
        "jerk_p90_ratio": _safe_ratio(jerk_p90, jerk_p90_ref),
        "distal_velocity_spectrum_l1_relative": _safe_ratio(
            spectrum_error.sum(), reference_spectrum.sum()
        ),
        "distal_velocity_high_frequency_hz": 3.0,
        "distal_velocity_high_frequency_spectrum_l1_relative": _safe_ratio(
            spectrum_error[:, high_frequency_mask].sum(),
            reference_spectrum[:, high_frequency_mask].sum(),
        ),
        "distal_velocity_high_frequency_energy_ratio": _safe_ratio(
            reconstructed_high_frequency_energy,
            reference_high_frequency_energy,
        ),
    }


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--data-root", default="prepared/aist_mld_humanml3d")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fast-quantile", type=float, default=0.9)
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()
    if not 0.5 <= args.fast_quantile < 1.0:
        raise ValueError("--fast-quantile must be in [0.5, 1.0)")

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    names = _load_names(data_root, args.split)
    vae_backend = load_vae_backend(
        args.vae_ckpt,
        cfg,
        device,
        seq_len=196,
        vae_type="motion_vqvae",
    )
    vae = vae_backend.model
    stats_path = args.stats_path or vae_backend.ckpt.get("stats_path")
    if not stats_path:
        raise ValueError("No motion stats path supplied or stored in the VQ-VAE checkpoint")
    stats = np.load(stats_path)
    flow_mean = np.asarray(stats["mean"], dtype=np.float32)
    flow_std = np.asarray(stats["std"], dtype=np.float32)
    mean_t = torch.from_numpy(flow_mean).to(device)
    std_t = torch.from_numpy(flow_std).to(device)

    t2m_mean_path = cfg["t2m_eval_mean_path"]
    t2m_std_path = cfg["t2m_eval_std_path"]
    t2m_mean = np.load(t2m_mean_path).astype(np.float32)
    t2m_std = np.load(t2m_std_path).astype(np.float32)
    extractor = T2MMotionFeatureExtractor(input_size=cfg["d_in"]).to(device)
    extractor.load_pretrained(cfg["t2m_motion_encoder_ckpt"])
    extractor.eval()

    genre_to_id = build_genre_to_id(cfg.get("aist_genres", []))
    ref_features = []
    recon_features = []
    mse_sum = 0.0
    mse_count = 0
    latent_shape = None
    code_histogram = np.zeros(vae.quantizer.num_codes, dtype=np.int64)
    reference_joint_chunks = []
    reconstructed_joint_chunks = []

    motion_dir = data_root / "new_joint_vecs"
    for start in tqdm(range(0, len(names), args.batch_size), desc="VQ-VAE recon FID"):
        batch_names = names[start : start + args.batch_size]
        raw_np = np.stack(
            [np.load(motion_dir / f"{name}.npy").astype(np.float32) for name in batch_names]
        )
        if raw_np.shape[1:] != (196, 263):
            raise ValueError(f"Expected [B,196,263], got {raw_np.shape}")
        raw = torch.from_numpy(raw_np).to(device)
        motion = raw.clone()
        motion[..., :CONT_END] = (motion[..., :CONT_END] - mean_t) / (std_t + 1e-6)
        domain_id = torch.ones(len(batch_names), dtype=torch.long, device=device)
        style_id = torch.tensor(
            [genre_to_id.get(get_genre_code(name), 0) for name in batch_names],
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones(motion.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode():
            cond = vae.cond(domain_id, style_id)
            _enc_h, _z_e, z_q, code_ids, _vq_loss, _perplexity, _codes_used = vae.encode(
                motion, cond, mask=mask, update_codebook=False
            )
            recon = vae.decode(z_q, cond, mask=mask, out_len=196)
        latent_shape = list(z_q.shape[1:])
        code_histogram += np.bincount(
            code_ids.detach().cpu().numpy().reshape(-1),
            minlength=vae.quantizer.num_codes,
        )
        diff = recon[..., :CONT_END] - motion[..., :CONT_END]
        mse_sum += float(diff.square().sum().item())
        mse_count += diff.numel()

        # NumPy shares storage with a CPU tensor. Keep T2M's normalized `recon`
        # intact while denormalizing a separate array for joint-space metrics.
        reconstructed_raw = recon.detach().cpu().numpy().copy()
        reconstructed_raw[..., :CONT_END] = (
            reconstructed_raw[..., :CONT_END] * flow_std + flow_mean
        )
        reference_joint_chunks.append(ik263_to_smpl22(raw_np))
        reconstructed_joint_chunks.append(ik263_to_smpl22(reconstructed_raw))

        ref_eval = _renorm_for_t2m(
            motion.detach().cpu().numpy(), flow_mean, flow_std, t2m_mean, t2m_std
        )
        recon_eval = _renorm_for_t2m(
            recon.detach().cpu().numpy(), flow_mean, flow_std, t2m_mean, t2m_std
        )
        lengths = torch.full((len(batch_names),), 196, dtype=torch.long, device=device)
        ref_features.append(_extract_t2m_features(extractor, ref_eval, lengths, device))
        recon_features.append(
            _extract_t2m_features(extractor, recon_eval, lengths, device)
        )

    ref_features_np = np.concatenate(ref_features, axis=0)
    recon_features_np = np.concatenate(recon_features, axis=0)
    np.random.seed(int(cfg.get("seed", 42)))
    metrics = summarize_motion_feature_metrics(
        recon_features_np,
        ref_features_np,
        diversity_times=300,
    )
    reference_joints = np.concatenate(reference_joint_chunks, axis=0)
    reconstructed_joints = np.concatenate(reconstructed_joint_chunks, axis=0)
    metrics.update(
        _physical_reconstruction_metrics(
            reconstructed_joints,
            reference_joints,
            args.fps,
            args.fast_quantile,
        )
    )
    code_probabilities = code_histogram.astype(np.float64)
    code_probabilities /= max(float(code_probabilities.sum()), 1.0)
    nonzero_probabilities = code_probabilities[code_probabilities > 0]
    aggregate_perplexity = float(
        np.exp(-(nonzero_probabilities * np.log(nonzero_probabilities)).sum())
    )
    aggregate_codes_used = int(np.count_nonzero(code_histogram))
    metrics.update(
        {
            "normalized_continuous_mse": mse_sum / max(mse_count, 1),
            "samples": len(names),
            "split": args.split,
            "clip": "first_196",
            "vae_ckpt": args.vae_ckpt,
            "stats_path": str(stats_path),
            "t2m_motion_encoder_ckpt": cfg["t2m_motion_encoder_ckpt"],
            "latent_shape_per_sample": latent_shape,
            "checkpoint_epoch": int(vae_backend.ckpt.get("epoch", -1)),
            "checkpoint_best_epoch": int(vae_backend.ckpt.get("best_epoch", -1)),
            "codebook_size": int(vae.quantizer.num_codes),
            "aggregate_perplexity": aggregate_perplexity,
            "aggregate_codes_used": aggregate_codes_used,
            "aggregate_code_utilization": _safe_ratio(
                aggregate_codes_used, vae.quantizer.num_codes
            ),
        }
    )
    print(json.dumps(metrics, indent=2))
    if args.save_json:
        output = Path(args.save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
