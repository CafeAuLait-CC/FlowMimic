#!/usr/bin/env python3
"""Evaluate AIST VAE reconstruction for FlowMimic and MLD checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


CONT_END = 259


def _load_ids(root: Path, split: str) -> list[str]:
    split_path = root / f"{split}.txt"
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def _batches(names: list[str], root: Path, batch_size: int):
    motion_dir = root / "new_joint_vecs"
    for start in range(0, len(names), batch_size):
        chunk = names[start : start + batch_size]
        motions = [np.load(motion_dir / f"{name}.npy").astype(np.float32) for name in chunk]
        lengths = [motion.shape[0] for motion in motions]
        max_len = max(lengths)
        padded = np.zeros((len(motions), max_len, motions[0].shape[-1]), dtype=np.float32)
        mask = np.zeros((len(motions), max_len), dtype=bool)
        for i, motion in enumerate(motions):
            padded[i, : motion.shape[0]] = motion
            mask[i, : motion.shape[0]] = True
        yield chunk, torch.from_numpy(padded), torch.tensor(lengths), torch.from_numpy(mask)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None, dims: int = 1) -> torch.Tensor:
    if mask is None:
        return value.mean()
    frame_mask = mask
    feature_elems = int(np.prod(value.shape[frame_mask.ndim :])) if value.ndim > frame_mask.ndim else 1
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    denom = frame_mask.float().sum() * feature_elems
    return (value * mask.float()).sum() / denom.clamp_min(1.0)


def _masked_feature_metrics(pred_raw: torch.Tensor, target_raw: torch.Tensor, mask: torch.Tensor):
    cont_pred = pred_raw[..., :CONT_END]
    cont_target = target_raw[..., :CONT_END]
    diff = cont_pred - cont_target
    mse = _masked_mean(diff.square(), mask)
    mae = _masked_mean(diff.abs(), mask)
    smooth_l1 = _masked_mean(F.smooth_l1_loss(cont_pred, cont_target, reduction="none"), mask)

    contact_pred = pred_raw[..., CONT_END:] > 0.5
    contact_target = target_raw[..., CONT_END:] > 0.5
    contact_mask = mask.unsqueeze(-1).expand_as(contact_pred)
    contact_acc = (contact_pred[contact_mask] == contact_target[contact_mask]).float().mean()
    return mse, mae, smooth_l1, contact_acc


def _joint_metrics(pred_raw: torch.Tensor, target_raw: torch.Tensor, mask: torch.Tensor):
    from mld.data.humanml.scripts.motion_process import recover_from_ric

    pred_j = recover_from_ric(pred_raw, 22)
    target_j = recover_from_ric(target_raw, 22)
    l2 = torch.linalg.norm(pred_j - target_j, dim=-1)
    joint = _masked_mean(l2, mask, dims=1)
    root = _masked_mean(l2[..., :1], mask, dims=1)
    return joint, root


def _evaluate_mld(args, root: Path, names: list[str], device: torch.device):
    workspace = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(workspace / "motion-latent-diffusion"))
    from mld.models.architectures.mld_vae import MldVae

    mean = torch.from_numpy(np.load(root / "Mean.npy").astype(np.float32)).to(device)
    std = torch.from_numpy(np.load(root / "Std.npy").astype(np.float32)).to(device)
    model = MldVae(
        ablation=SimpleNamespace(MLP_DIST=False, PE_TYPE="mld"),
        nfeats=263,
        latent_dim=[1, 256],
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        arch="encoder_decoder",
        position_embedding="learned",
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("vae."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("vae.")
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    return _run_batches(
        args,
        root,
        names,
        device,
        lambda raw, mask, lengths: _mld_reconstruct(model, raw, mask, lengths, mean, std),
    )


def _mld_reconstruct(model, raw, mask, lengths, mean, std):
    norm = (raw - mean) / std
    z, dist = model.encode(norm, lengths.tolist())
    pred_norm = model.decode(dist.loc, lengths.tolist())
    pred_raw = pred_norm * std + mean
    norm_loss = F.smooth_l1_loss(pred_norm, norm, reduction="none")
    return pred_raw, norm_loss


def _evaluate_flowmimic(args, root: Path, names: list[str], device: torch.device):
    from flowmimic.src.model.vae.motion_vae import MotionVAE
    from flowmimic.src.model.vae.datasets.aist_filename_parser import get_genre_code

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]
    latent_len = ckpt.get("config", {}).get("latent_len")
    latent_token_mode = ckpt.get("config", {}).get("latent_token_mode", "pool")
    max_len = model_state["enc_pos"].shape[1]
    num_styles = model_state["cond.style_emb.weight"].shape[0]
    model = MotionVAE(
        max_len=max_len,
        num_styles=num_styles,
        latent_len=latent_len,
        latent_token_mode=latent_token_mode,
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()

    stats_path = Path(args.stats_path or ckpt.get("stats_path") or "prepared/flowmimic_aist_mean_std_259_train.npz")
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(device)
    std = torch.from_numpy(stats["std"].astype(np.float32)).to(device)
    with open(args.genre_map, "r", encoding="utf-8") as f:
        genre_to_id = json.load(f)

    def reconstruct(raw, mask, lengths, batch_names):
        norm = raw.clone()
        norm[..., :CONT_END] = (norm[..., :CONT_END] - mean) / std
        domain = torch.ones(raw.shape[0], dtype=torch.long, device=device)
        style = torch.zeros(raw.shape[0], dtype=torch.long, device=device)
        for i, name in enumerate(batch_names):
            genre = get_genre_code(name)
            style[i] = int(genre_to_id.get(genre, 0))
        cond = model.cond(domain, style)
        _, mu, _ = model.encode(norm, cond, mask=mask)
        pred_norm = model.decode(mu, cond, mask=mask, out_len=raw.shape[1])
        pred_raw = pred_norm.clone()
        pred_raw[..., :CONT_END] = pred_raw[..., :CONT_END] * std + mean
        pred_raw[..., CONT_END:] = torch.sigmoid(pred_raw[..., CONT_END:])
        norm_loss = F.smooth_l1_loss(pred_norm[..., :CONT_END], norm[..., :CONT_END], reduction="none")
        return pred_raw, norm_loss

    return _run_batches(args, root, names, device, reconstruct)


def _run_batches(args, root: Path, names: list[str], device: torch.device, reconstruct):
    totals = {
        "norm_smooth_l1": 0.0,
        "raw_cont_mse": 0.0,
        "raw_cont_mae": 0.0,
        "raw_cont_smooth_l1": 0.0,
        "contact_acc": 0.0,
        "joint_l2": 0.0,
        "root_l2": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch_names, raw_cpu, lengths_cpu, mask_cpu in _batches(names, root, args.batch_size):
            raw = raw_cpu.to(device)
            mask = mask_cpu.to(device)
            lengths = lengths_cpu.to(device)
            try:
                pred_raw, norm_loss = reconstruct(raw, mask, lengths, batch_names)
            except TypeError:
                pred_raw, norm_loss = reconstruct(raw, mask, lengths)
            norm = _masked_mean(norm_loss, mask)
            mse, mae, smooth, contact = _masked_feature_metrics(pred_raw, raw, mask)
            joint, root_l2 = _joint_metrics(pred_raw, raw, mask)
            batch_n = raw.shape[0]
            totals["norm_smooth_l1"] += float(norm.item()) * batch_n
            totals["raw_cont_mse"] += float(mse.item()) * batch_n
            totals["raw_cont_mae"] += float(mae.item()) * batch_n
            totals["raw_cont_smooth_l1"] += float(smooth.item()) * batch_n
            totals["contact_acc"] += float(contact.item()) * batch_n
            totals["joint_l2"] += float(joint.item()) * batch_n
            totals["root_l2"] += float(root_l2.item()) * batch_n
            count += batch_n
    return {key: value / max(count, 1) for key, value in totals.items()} | {"samples": count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mld", "flowmimic"), required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="prepared/aist_mld_humanml3d")
    parser.add_argument("--split", default="val_test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--genre-map", default="flowmimic/src/config/genre_to_id.json")
    args = parser.parse_args()

    root = Path(args.data_root)
    names = _load_ids(root, args.split)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.model == "mld":
        metrics = _evaluate_mld(args, root, names, device)
    else:
        metrics = _evaluate_flowmimic(args, root, names, device)
    print(f"model={args.model}")
    print(f"ckpt={args.ckpt}")
    print(f"split={args.split}")
    for key, value in metrics.items():
        print(f"{key}={value:.8f}" if isinstance(value, float) else f"{key}={value}")


if __name__ == "__main__":
    main()
