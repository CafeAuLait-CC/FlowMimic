import argparse
import csv
import json
import os
import random
import sys
import warnings
import time
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

warnings.filterwarnings(
    "ignore",
    message=".*nested tensors is in prototype stage.*",
    category=UserWarning,
)

from flowmimic.src.config.config import load_config
from flowmimic.src.data.openpose import load_aist_openpose, load_mvh_openpose
from flowmimic.src.metrics import (
    T2MMotionFeatureExtractor,
    summarize_motion_feature_metrics,
)
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id
from flowmimic.src.motion.process_motion import ik263_to_smpl22
from flowmimic.src.motion.ik.utils.paramUtil import t2m_kinematic_chain


@dataclass
class EvalConfig:
    seq_len: int
    d_z: int
    latent_len: int
    fps: int
    slack_seconds: float
    cam_mode: str


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _aist_split_paths(aist_dir, split_path):
    names = _read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def _aist_paths_for_splits(cfg, split_names):
    paths = []
    for split in split_names:
        split = split.strip().lower()
        if not split:
            continue
        split_path = cfg.get(f"aist_split_{split}")
        if split_path is None:
            split_path = f"data/AIST++/Annotations/splits/pose_{split}.txt"
        paths.extend(_aist_split_paths(cfg["aist_motions_dir"], split_path))
    seen = set()
    unique = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


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


def _edges_from_chain(chain):
    edges = []
    for ch in chain:
        for i in range(len(ch) - 1):
            edges.append((ch[i], ch[i + 1]))
    return edges


def _build_smpl22_to_body25(def_path):
    with open(def_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mapping = []
    for joint in cfg.get("smpl_joints", []):
        smpl_idx = joint.get("smpl_idx")
        body_idx = joint.get("body25_idx")
        if body_idx is None or smpl_idx is None:
            continue
        if smpl_idx < 22:
            mapping.append((body_idx, smpl_idx))
    computed = []
    for rule in cfg.get("computed_body25", []):
        body_idx = rule.get("body25_idx")
        smpl_indices = rule.get("smpl_indices", [])
        if body_idx is None or not smpl_indices:
            continue
        if all(idx < 22 for idx in smpl_indices):
            computed.append((body_idx, smpl_indices))
    return mapping, computed


def _smpl22_to_body25(joints, mapping, computed):
    t = joints.shape[0]
    body = np.full((t, 25, 3), np.nan, dtype=joints.dtype)
    for body_idx, smpl_idx in mapping:
        body[:, body_idx] = joints[:, smpl_idx]
    for body_idx, smpl_indices in computed:
        body[:, body_idx] = joints[:, smpl_indices].mean(axis=1)
    return body


def _fit_weak_persp(xy, uv, w):
    if xy.shape[0] < 2:
        return None
    w = np.clip(w, 0.0, 1.0)
    w_sqrt = np.sqrt(w)[:, None]
    a = np.zeros((xy.shape[0] * 2, 3), dtype=np.float64)
    b = np.zeros((xy.shape[0] * 2,), dtype=np.float64)
    a[0::2, 0] = xy[:, 0]
    a[0::2, 1] = 1.0
    b[0::2] = uv[:, 0]
    a[1::2, 0] = xy[:, 1]
    a[1::2, 2] = 1.0
    b[1::2] = uv[:, 1]
    a = a * np.repeat(w_sqrt, 2, axis=0)
    b = b * np.repeat(w_sqrt[:, 0], 2, axis=0)
    try:
        sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return sol[0], sol[1], sol[2]


def _project_points(xy, cam):
    s, tx, ty = cam
    return s * xy + np.array([tx, ty], dtype=xy.dtype)


def _weighted_l2_error(pred, target, w, eps=1e-6):
    w = np.clip(w, 0.0, 1.0)
    denom = w.sum() + eps
    err = np.linalg.norm(pred - target, axis=-1)
    return float((err * w).sum() / denom)


def _condition_error(
    joints3d,
    k2d,
    conf,
    tau_cond,
    fps,
    slack_seconds,
    cam_mode,
    mapping,
    computed,
):
    t_len = joints3d.shape[0]
    body25 = _smpl22_to_body25(joints3d, mapping, computed)
    if tau_cond.size == 0:
        return None
    t_idx = np.clip(np.round(tau_cond * (t_len - 1)).astype(int), 0, t_len - 1)
    slack = int(round(slack_seconds * fps))
    strict_err = []
    slack_err = []
    slack_offsets = []

    cam_fixed = None
    if cam_mode == "fixed":
        first = t_idx[0]
        xy = body25[first, :, :2]
        mask = np.isfinite(xy).all(axis=1) & (conf[0] > 0)
        cam_fixed = _fit_weak_persp(xy[mask], k2d[0][mask], conf[0][mask])
        if cam_fixed is None:
            cam_fixed = (1.0, 0.0, 0.0)

    for i, t0 in enumerate(t_idx):
        xy0 = body25[t0, :, :2]
        mask0 = np.isfinite(xy0).all(axis=1) & (conf[i] > 0)
        if mask0.any():
            if cam_mode == "fixed":
                cam0 = cam_fixed
            else:
                cam0 = _fit_weak_persp(xy0[mask0], k2d[i][mask0], conf[i][mask0])
                if cam0 is None:
                    cam0 = (1.0, 0.0, 0.0)
            proj0 = _project_points(xy0[mask0], cam0)
            strict_err.append(_weighted_l2_error(proj0, k2d[i][mask0], conf[i][mask0]))

        if slack <= 0:
            candidates = [t0]
        else:
            t_start = max(0, t0 - slack)
            t_end = min(t_len - 1, t0 + slack)
            candidates = list(range(t_start, t_end + 1))
        best_err = None
        best_dt = 0
        for t in candidates:
            xy = body25[t, :, :2]
            mask = np.isfinite(xy).all(axis=1) & (conf[i] > 0)
            if not mask.any():
                continue
            if cam_mode == "fixed":
                cam = cam_fixed
            else:
                cam = _fit_weak_persp(xy[mask], k2d[i][mask], conf[i][mask])
                if cam is None:
                    cam = (1.0, 0.0, 0.0)
            proj = _project_points(xy[mask], cam)
            err = _weighted_l2_error(proj, k2d[i][mask], conf[i][mask])
            if best_err is None or err < best_err:
                best_err = err
                best_dt = t - t0
        if best_err is None:
            continue
        slack_err.append(best_err)
        slack_offsets.append(abs(best_dt))
    if not strict_err:
        return None
    return {
        "e2d_strict": float(np.mean(strict_err)),
        "e2d_slack": float(np.mean(slack_err)),
        "e2d_strict_median": float(np.median(strict_err)),
        "e2d_strict_p90": float(np.percentile(strict_err, 90)),
        "e2d_slack_median": float(np.median(slack_err)),
        "e2d_slack_p90": float(np.percentile(slack_err, 90)),
        "slack_dt_mean": float(np.mean(slack_offsets)) if slack_offsets else 0.0,
    }


def _foot_skate(joints, contact_logits, fps, threshold=0.5):
    left_idx = 7
    right_idx = 8
    if isinstance(contact_logits, torch.Tensor):
        contact = torch.sigmoid(contact_logits).cpu().numpy()
    else:
        contact = 1.0 / (1.0 + np.exp(-contact_logits))
    left_contact = np.maximum(contact[..., 0], contact[..., 1]) > threshold
    right_contact = np.maximum(contact[..., 2], contact[..., 3]) > threshold
    v = np.linalg.norm(joints[1:] - joints[:-1], axis=-1) * fps
    left_v = v[:, left_idx]
    right_v = v[:, right_idx]
    left_vals = left_v[left_contact[: left_v.shape[0]]]
    right_vals = right_v[right_contact[: right_v.shape[0]]]
    return {
        "skate_left": float(left_vals.mean()) if left_vals.size else 0.0,
        "skate_right": float(right_vals.mean()) if right_vals.size else 0.0,
    }


def _bone_var(joints, edges):
    lengths = []
    for p, c in edges:
        seg = np.linalg.norm(joints[:, c] - joints[:, p], axis=-1)
        lengths.append(np.std(seg))
    return float(np.mean(lengths)) if lengths else 0.0


def _smoothness(joints, fps):
    v = (joints[1:] - joints[:-1]) * fps
    a = (v[1:] - v[:-1]) * fps
    j = (a[1:] - a[:-1]) * fps
    a_norm = np.linalg.norm(a, axis=-1).reshape(-1)
    j_norm = np.linalg.norm(j, axis=-1).reshape(-1)
    return {
        "accel_median": float(np.median(a_norm)) if a_norm.size else 0.0,
        "accel_p90": float(np.percentile(a_norm, 90)) if a_norm.size else 0.0,
        "jerk_median": float(np.median(j_norm)) if j_norm.size else 0.0,
        "jerk_p90": float(np.percentile(j_norm, 90)) if j_norm.size else 0.0,
    }


def _extract_t2m_features(extractor, motion_np, lengths, device, chunk=64):
    feats = []
    total = motion_np.shape[0]
    start = 0
    if not isinstance(device, torch.device):
        device = torch.device(device)
    while start < total:
        end = min(total, start + chunk)
        if isinstance(motion_np, np.ndarray):
            motion_t = torch.from_numpy(motion_np[start:end]).to(device)
        else:
            motion_t = motion_np[start:end].to(device)
        if isinstance(lengths, np.ndarray):
            lengths_t = torch.from_numpy(lengths[start:end]).to(device=device, dtype=torch.long)
        else:
            lengths_t = lengths[start:end].to(device=device, dtype=torch.long)
        with torch.inference_mode():
            feat = extractor.encode(motion_t, lengths_t)
        feats.append(feat.cpu().numpy())
        start = end
    return np.concatenate(feats, axis=0)


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
    rng=None,
):
    rng = rng or np.random
    k2d_list = []
    vis_list = []
    conf_list = []
    mask_list = []
    tau_list = []
    idx_list = []
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
        k_frames = cond_frames_min
        if t_len <= k_frames:
            idx = np.arange(t_len)
        else:
            idx = np.linspace(0, t_len - 1, k_frames)
            idx = np.unique(np.round(idx).astype(int))
        k2d_sparse = k2d[idx]
        vis_sparse = vis[idx]
        conf_sparse = conf[idx]
        if cond_drop_prob > 0:
            drop = rng.rand(*vis_sparse.shape) < cond_drop_prob
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
        idx_list.append(idx)

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
            conf = np.concatenate(
                [conf, np.zeros((pad, 25), dtype=np.float32)], axis=0
            )
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
    return (
        k2d_batch,
        vis_batch,
        conf_batch,
        tau_batch,
        mask_batch,
        idx_list,
        k2d_list,
        vis_list,
    )


def _generate_batch(
    flow,
    vae,
    mean,
    std,
    latent_mean,
    latent_std,
    motion,
    domain_id,
    style_id,
    k2d_batch,
    vis_batch,
    tau_cond,
    mask_cond,
    steps,
    solver,
    device,
    k2d_mean,
    k2d_std,
    d_z,
    latent_len,
    guidance_scale=1.0,
):
    t_start = time.perf_counter()
    t_flow_start = time.perf_counter()
    with torch.inference_mode():
        if mask_cond is not None:
            empty = mask_cond.sum(dim=1) == 0
            if empty.any() and k2d_batch.shape[1] > 0:
                k2d_batch = k2d_batch.clone()
                vis_batch = vis_batch.clone()
                tau_cond = tau_cond.clone()
                mask_cond = mask_cond.clone()
                k2d_batch[empty, 0] = 0.0
                vis_batch[empty, 0] = 0.0
                tau_cond[empty, 0] = 0.0
                mask_cond[empty, 0] = True
        tau_out = torch.linspace(0.0, 1.0, steps=latent_len, device=device)
        x0 = torch.randn(motion.shape[0], latent_len, d_z, device=device)
        g2d, mem, _vis = flow.cond_encoder(
            k2d_batch,
            tau_cond,
            vis_mask=vis_batch,
            mask_cond=mask_cond,
            mean=k2d_mean,
            std=k2d_std,
        )
        style = flow.style_emb(style_id, domain_id, apply_dropout=False)
        g = flow.cond_mlp(torch.cat([g2d, style], dim=-1))
        cond_batch = {"tau_out": tau_out, "mem": mem, "g": g}
        if guidance_scale != 1.0:
            k2d_uncond = torch.zeros_like(k2d_batch)
            vis_uncond = torch.zeros_like(vis_batch)
            g2d_uncond, mem_uncond, _ = flow.cond_encoder(
                k2d_uncond,
                tau_cond,
                vis_mask=vis_uncond,
                mask_cond=mask_cond,
                mean=k2d_mean,
                std=k2d_std,
            )
            style_uncond = flow.style_emb(
                torch.zeros_like(style_id), domain_id, apply_dropout=False
            )
            g_uncond = flow.cond_mlp(torch.cat([g2d_uncond, style_uncond], dim=-1))
            cond_batch.update(
                {
                    "mem_uncond": mem_uncond,
                    "g_uncond": g_uncond,
                    "mem_mask_uncond": ~mask_cond,
                    "guidance_scale": guidance_scale,
                }
            )
        z_hat = solve_flow(flow.flow, x0, cond_batch, num_steps=steps, method=solver)
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.synchronize()
        t_flow = time.perf_counter() - t_flow_start
        t_vae_start = time.perf_counter()
        if latent_mean is not None and latent_std is not None:
            z_hat = z_hat * (latent_std + 1e-6) + latent_mean
        x_hat = vae.decode(z_hat, vae.cond(domain_id, style_id))
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.synchronize()
        t_vae = time.perf_counter() - t_vae_start
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()
    net_time = time.perf_counter() - t_start
    x_hat_norm_np = x_hat.cpu().numpy()
    x_hat_np = x_hat_norm_np.copy()
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    x_hat_np[..., :cont_end] = x_hat_np[..., :cont_end] * std + mean
    contact_logits = x_hat_np[..., LAYOUT_SLICES["feet_contact"][0] : LAYOUT_SLICES["feet_contact"][1]]
    joints = ik263_to_smpl22(x_hat_np)
    full_time = time.perf_counter() - t_start
    return (
        x_hat_norm_np,
        x_hat_np,
        joints,
        contact_logits,
        net_time,
        full_time,
        t_flow,
        t_vae,
    )


def _infer_motion_lengths(batch, batch_size, seq_len, device):
    mask = batch.get("mask")
    if mask is None:
        return torch.full((batch_size,), int(seq_len), dtype=torch.long, device=device)
    lengths = mask[:batch_size].to(device=device)
    if lengths.dtype != torch.bool:
        lengths = lengths > 0
    lengths = lengths.sum(dim=1).long()
    return torch.clamp(lengths, min=1, max=int(seq_len))


def _renorm_for_t2m(features_norm, flow_mean, flow_std, t2m_mean, t2m_std):
    x = np.asarray(features_norm, dtype=np.float32)
    flow_mean = np.asarray(flow_mean, dtype=np.float32).reshape(-1)
    flow_std = np.asarray(flow_std, dtype=np.float32).reshape(-1)
    t2m_mean = np.asarray(t2m_mean, dtype=np.float32).reshape(-1)
    t2m_std = np.asarray(t2m_std, dtype=np.float32).reshape(-1)

    d_x = x.shape[-1]
    d_eval = t2m_mean.shape[0]
    d_flow = flow_mean.shape[0]
    if flow_std.shape[0] != d_flow:
        raise ValueError(
            f"Flow std dim mismatch: mean={d_flow}, std={flow_std.shape[0]}"
        )
    if t2m_std.shape[0] != d_eval:
        raise ValueError(
            f"T2M std dim mismatch: mean={d_eval}, std={t2m_std.shape[0]}"
        )
    d_motion = d_x - 4
    if d_flow < d_motion:
        raise ValueError(
            f"Flow mean/std dims too small for renorm: flow_mean={flow_mean.shape[0]}, "
            f"flow_std={flow_std.shape[0]}, required_at_least={d_motion}"
        )
    if d_eval < d_motion:
        raise ValueError(
            f"T2M mean/std dims too small for renorm: t2m_mean={d_eval}, "
            f"t2m_std={t2m_std.shape[0]}, required_at_least={d_motion}"
        )

    y = x.copy()
    # The T2M motion encoder consumes only [..., :-4], so renorm that slice.
    y_motion = y[..., :d_motion]
    y[..., :d_motion] = (
        (y_motion * flow_std[:d_motion] + flow_mean[:d_motion]) - t2m_mean[:d_motion]
    ) / (
        t2m_std[:d_motion] + 1e-6
    )
    return y


def evaluate_dataset(
    name,
    dataloader,
    flow,
    vae,
    cfg,
    mapping,
    computed,
    openpose_cfg,
    stats,
    latent_stats,
    steps,
    solver,
    num_samples,
    seed,
    device,
    compute_dist,
    save_per_sample,
    metric_extractor=None,
    t2m_stats=None,
    diversity_times=300,
    multimodality_repeats=1,
    multimodality_times=20,
    skip_empty_cond=True,
    guidance_scale=1.0,
):
    print(f"Evaluating {name} (steps={steps})")
    mean, std = stats
    latent_mean, latent_std = latent_stats
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    t2m_mean = None
    t2m_std = None
    if t2m_stats is not None:
        t2m_mean = np.asarray(t2m_stats[0], dtype=np.float32)
        t2m_std = np.asarray(t2m_stats[1], dtype=np.float32)
    if compute_dist and metric_extractor is not None and (t2m_mean is None or t2m_std is None):
        raise ValueError("t2m_stats is required for distribution metrics with T2M extractor")
    k2d_mean, k2d_std = openpose_cfg["mean"], openpose_cfg["std"]
    results = []
    feats_gen_list = []
    feats_ref = []
    feats_mm = []
    total_time_net = 0.0
    total_time_full = 0.0
    total_time_flow = 0.0
    total_time_vae = 0.0
    seq_count = 0
    rng = np.random.RandomState(seed)
    edges = _edges_from_chain(t2m_kinematic_chain)

    seen = 0
    for batch in tqdm(dataloader, desc=f"{name} eval", leave=False):
        if num_samples and seen >= num_samples:
            break
        motion = batch["motion"].to(device)
        domain_id = batch["domain_id"].to(device)
        style_id = batch["style_id"].to(device)
        metas = _normalize_meta(batch["meta"])
        batch_size = motion.shape[0]
        if num_samples:
            keep = min(batch_size, num_samples - seen)
            motion = motion[:keep]
            domain_id = domain_id[:keep]
            style_id = style_id[:keep]
            metas = metas[:keep]
            batch_size = keep

        if "k2d" in batch:
            k2d_batch = batch["k2d"][:batch_size].to(device)
            vis_batch = batch["vis"][:batch_size].to(device)
            conf_batch = batch.get("conf", batch["vis"])[:batch_size].to(device)
            tau_cond = batch["tau_cond"][:batch_size].to(device)
            mask_cond = batch["mask_cond"][:batch_size].to(device)
        else:
            (
                k2d_batch,
                vis_batch,
                conf_batch,
                tau_cond,
                mask_cond,
                idx_list,
                k2d_list,
                vis_list,
            ) = _load_cond_batch(
                metas,
                openpose_cfg["aist_dir"],
                openpose_cfg["mvh_root"],
                openpose_cfg["mv_root"],
                openpose_cfg["mvh_cameras"],
                cfg.seq_len,
                openpose_cfg["cond_frames_min"],
                openpose_cfg["cond_frames_max"],
                openpose_cfg["cond_drop_prob"],
                openpose_cfg["aist_fps"],
                openpose_cfg["mvh_fps"],
                openpose_cfg["target_fps"],
                cache_root=openpose_cfg["cond_cache_root"],
                rng=rng,
            )
            k2d_batch = k2d_batch.to(device)
            vis_batch = vis_batch.to(device)
            conf_batch = conf_batch.to(device)
            tau_cond = tau_cond.to(device)
            mask_cond = mask_cond.to(device)
        if skip_empty_cond and k2d_batch.shape[1] == 0:
            continue

        (
            x_hat_norm_np,
            x_hat_np,
            joints,
            contact_logits,
            net_time,
            full_time,
            flow_time,
            vae_time,
        ) = _generate_batch(
            flow,
            vae,
            mean,
            std,
            latent_mean,
            latent_std,
            motion,
            domain_id,
            style_id,
            k2d_batch,
            vis_batch,
            tau_cond,
            mask_cond,
            steps,
            solver,
            device,
            k2d_mean,
            k2d_std,
            cfg.d_z,
            cfg.latent_len,
            guidance_scale=guidance_scale,
        )
        total_time_net += net_time
        total_time_full += full_time
        total_time_flow += flow_time
        total_time_vae += vae_time
        seq_count += batch_size
        joints = joints if isinstance(joints, np.ndarray) else joints

        k2d_np = k2d_batch.cpu().numpy()
        conf_np = conf_batch.cpu().numpy()
        tau_np = tau_cond.cpu().numpy()
        mask_np = mask_cond.cpu().numpy()
        for i in range(batch_size):
            valid_mask = mask_np[i].astype(bool)
            k2d_i = k2d_np[i][valid_mask]
            conf_i = conf_np[i][valid_mask]
            tau_i = tau_np[i][valid_mask]
            err = _condition_error(
                joints[i],
                k2d_i,
                conf_i,
                tau_i,
                cfg.fps,
                cfg.slack_seconds,
                cfg.cam_mode,
                mapping,
                computed,
            )
            if err is None:
                continue
            skate = _foot_skate(joints[i], contact_logits[i], cfg.fps)
            bone = _bone_var(joints[i], edges)
            smooth = _smoothness(joints[i], cfg.fps)
            record = {
                "dataset": name,
                "e2d_strict": err["e2d_strict"],
                "e2d_slack": err["e2d_slack"],
                "e2d_strict_median": err["e2d_strict_median"],
                "e2d_strict_p90": err["e2d_strict_p90"],
                "e2d_slack_median": err["e2d_slack_median"],
                "e2d_slack_p90": err["e2d_slack_p90"],
                "slack_dt_mean": err["slack_dt_mean"],
                "skate_left": skate["skate_left"],
                "skate_right": skate["skate_right"],
                "skate_mean": 0.5 * (skate["skate_left"] + skate["skate_right"]),
                "bone_var": bone,
                "accel_median": smooth["accel_median"],
                "accel_p90": smooth["accel_p90"],
                "jerk_median": smooth["jerk_median"],
                "jerk_p90": smooth["jerk_p90"],
            }
            results.append(record)

        if compute_dist and metric_extractor is not None:
            lengths = _infer_motion_lengths(batch, batch_size, cfg.seq_len, device)
            motion_eval_np = _renorm_for_t2m(
                motion.detach().cpu().numpy(), mean, std, t2m_mean, t2m_std
            )
            x_hat_eval_np = _renorm_for_t2m(
                x_hat_norm_np, mean, std, t2m_mean, t2m_std
            )
            feats_gen_list.append(
                _extract_t2m_features(
                    metric_extractor, x_hat_eval_np, lengths, device
                )
            )
            feats_ref.append(
                _extract_t2m_features(metric_extractor, motion_eval_np, lengths, device)
            )
            if multimodality_repeats > 1:
                mm_batch = [feats_gen_list[-1]]
                for _ in range(multimodality_repeats - 1):
                    (
                        x_hat_mm_norm_np,
                        _x_hat_mm_np,
                        _joints_mm,
                        _contact_mm,
                        _net_mm,
                        _full_mm,
                        _flow_mm,
                        _vae_mm,
                    ) = _generate_batch(
                        flow,
                        vae,
                        mean,
                        std,
                        latent_mean,
                        latent_std,
                        motion,
                        domain_id,
                        style_id,
                        k2d_batch,
                        vis_batch,
                        tau_cond,
                        mask_cond,
                        steps,
                        solver,
                        device,
                        k2d_mean,
                        k2d_std,
                        cfg.d_z,
                        cfg.latent_len,
                        guidance_scale=guidance_scale,
                    )
                    mm_batch.append(
                        _extract_t2m_features(
                            metric_extractor,
                            _renorm_for_t2m(
                                x_hat_mm_norm_np, mean, std, t2m_mean, t2m_std
                            ),
                            lengths,
                            device,
                        )
                    )
                feats_mm.append(np.stack(mm_batch, axis=1))

        seen += batch_size

    summary = {}
    if results:
        keys = [k for k in results[0].keys() if k not in ("dataset")]
        for k in keys:
            summary[k] = float(np.mean([r[k] for r in results]))
    if seq_count > 0:
        summary["aits_full"] = total_time_full / seq_count
        summary["aits_net"] = total_time_net / seq_count
        summary["aits_flow"] = total_time_flow / seq_count
        summary["aits_vae"] = total_time_vae / seq_count
    if compute_dist and metric_extractor is not None and feats_gen_list and feats_ref:
        f_gen = np.concatenate(feats_gen_list, axis=0)
        f_ref = np.concatenate(feats_ref, axis=0)
        mm_feats = np.concatenate(feats_mm, axis=0) if feats_mm else None
        metric_summary = summarize_motion_feature_metrics(
            f_gen,
            f_ref,
            diversity_times=diversity_times,
            multimodality_feats=mm_feats,
            multimodality_times=multimodality_times,
        )
        summary.update(metric_summary)
        # Keep old key for downstream compatibility.
        summary["mmd"] = summary["mmdist"]
    return summary, results if save_per_sample else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--t2m-motion-encoder-ckpt", default=None)
    parser.add_argument("--t2m-mean-path", default=None)
    parser.add_argument("--t2m-std-path", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--openpose-stats-path", default=None)
    parser.add_argument("--latent-stats-path", default=None)
    parser.add_argument(
        "--aist-crop-mode", choices=("first", "random", "uniform"), default="first"
    )
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vae-max-len", type=int, default=None)
    parser.add_argument("--vae-latent-len", type=int, default=None)
    parser.add_argument(
        "--vae-latent-token-mode",
        choices=("pool", "query"),
        default=None,
    )
    parser.add_argument("--cond-frames", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--aist-splits", type=str, default="val")
    parser.add_argument(
        "--datasets",
        type=str,
        default="AIST,MVH",
        help="Comma-separated datasets to evaluate: AIST,MVH.",
    )
    parser.add_argument("--steps", type=str, default="16,8,4,2,1")
    parser.add_argument("--solver", type=str, default="heun")
    parser.add_argument("--cam-mode", type=str, default="fixed", choices=["fixed", "per_frame"])
    parser.add_argument("--slack-seconds", type=float, default=0.1)
    parser.add_argument(
        "--cond-drop-prob",
        type=float,
        default=0.0,
        help="Drop probability for sparse conditioning keypoints during eval. Defaults to 0; training augmentation should not be applied to validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--diversity-times", type=int, default=300)
    parser.add_argument("--multimodality-repeats", type=int, default=1)
    parser.add_argument("--multimodality-times", type=int, default=20)
    parser.add_argument("--no-dist", action="store_true")
    parser.add_argument("--save-per-sample", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading config")
    cfg = load_config()
    seq_len = args.seq_len or cfg["seq_len"]
    vae_max_len = args.vae_max_len or cfg["seq_len"]
    if seq_len > vae_max_len:
        raise ValueError(f"--seq-len ({seq_len}) cannot exceed --vae-max-len ({vae_max_len})")
    d_z = cfg["d_z"]
    stats_path = args.stats_path or cfg["stats_path"]
    print("Loading 263D stats")
    mean, std = load_mean_std(stats_path)
    openpose_stats_path = args.openpose_stats_path or cfg.get(
        "openpose_stats_path", "data/openpose_stats.npz"
    )
    print("Loading OpenPose stats")
    if os.path.exists(openpose_stats_path):
        op_stats = np.load(openpose_stats_path)
        k2d_mean = op_stats["mean"]
        k2d_std = op_stats["std"]
    else:
        raise FileNotFoundError(f"OpenPose stats not found: {openpose_stats_path}")
    latent_stats_path = args.latent_stats_path or cfg.get(
        "latent_stats_path", "data/latent_stats.npz"
    )
    latent_mean = None
    latent_std = None
    print("Loading latent stats")
    if os.path.exists(latent_stats_path):
        z_stats = np.load(latent_stats_path)
        latent_mean = torch.tensor(z_stats["mean"], dtype=torch.float32, device=args.device)
        latent_std = torch.tensor(z_stats["std"], dtype=torch.float32, device=args.device)

    print("Building models")
    mapping, computed = _build_smpl22_to_body25(cfg["smpl45_to_body25_def"])

    flow_cfg = cfg.get("flow", {})
    flow = ConditionalRectFlow(
        d_z=d_z,
        d_model=flow_cfg.get("d_model", 512),
        n_layers=flow_cfg.get("n_layers", 8),
        n_heads=flow_cfg.get("n_heads", 8),
        ffn_dim=flow_cfg.get("ffn_dim", 2048),
        dropout=flow_cfg.get("dropout", 0.1),
        num_styles=cfg["num_styles"],
        style_dim=flow_cfg.get("style_dim", 32),
        cond_dim=flow_cfg.get("cond_dim", 256),
        cond_layers=flow_cfg.get("cond_layers", 4),
        cond_heads=flow_cfg.get("cond_heads", 4),
        p_style_drop=flow_cfg.get("p_style_drop", 0.5),
    ).to(args.device)
    print(f"Loading flow model: {args.flow_ckpt}")
    state = torch.load(args.flow_ckpt, map_location=args.device)
    flow_state = state.get("ema") if args.use_ema and "ema" in state else state["model"]
    if args.use_ema and "ema" not in state:
        print("Warning: --use-ema requested but checkpoint has no EMA state; using model.")
    flow.load_state_dict(flow_state)
    flow.eval()

    print("Loading VAE")
    vae_ckpt = args.vae_ckpt or cfg.get("vae_ckpt", "checkpoints/vae/len200/motion_vae_best.pt")
    print(f"Loading VAE model: {vae_ckpt}")
    vae_state = torch.load(vae_ckpt, map_location=args.device)
    vae_config = vae_state.get("config", {})
    vae_latent_len = args.vae_latent_len
    if vae_latent_len is None:
        vae_latent_len = vae_config.get("latent_len")
    vae_latent_token_mode = (
        args.vae_latent_token_mode or vae_config.get("latent_token_mode", "pool")
    )
    vae = MotionVAE(
        d_in=cfg["d_in"],
        d_z=d_z,
        num_styles=cfg["num_styles"],
        max_len=vae_max_len,
        latent_len=vae_latent_len,
        latent_token_mode=vae_latent_token_mode,
    ).to(args.device)
    vae.load_state_dict(vae_state["model"])
    vae.eval()

    t2m_extractor = None
    t2m_stats = None
    if not args.no_dist:
        t2m_ckpt = args.t2m_motion_encoder_ckpt or cfg.get("t2m_motion_encoder_ckpt")
        t2m_mean_path = args.t2m_mean_path or cfg.get("t2m_eval_mean_path")
        t2m_std_path = args.t2m_std_path or cfg.get("t2m_eval_std_path")
        if not t2m_ckpt:
            raise ValueError(
                "Distribution metrics requested but no T2M checkpoint provided. "
                "Use --t2m-motion-encoder-ckpt or set t2m_motion_encoder_ckpt in config."
            )
        if not t2m_mean_path or not t2m_std_path:
            raise ValueError(
                "Distribution metrics requested but no T2M mean/std provided. "
                "Use --t2m-mean-path/--t2m-std-path or set t2m_eval_mean_path/t2m_eval_std_path in config."
            )
        if not os.path.exists(t2m_mean_path) or not os.path.exists(t2m_std_path):
            raise FileNotFoundError(
                f"T2M mean/std not found: mean={t2m_mean_path}, std={t2m_std_path}"
            )
        print(f"Loading T2M motion evaluator: {t2m_ckpt}")
        t2m_extractor = T2MMotionFeatureExtractor(input_size=cfg["d_in"]).to(args.device)
        t2m_extractor.load_pretrained(t2m_ckpt)
        t2m_extractor.eval()
        t2m_mean = np.load(t2m_mean_path).astype(np.float32)
        t2m_std = np.load(t2m_std_path).astype(np.float32)
        expected_t2m_dim = cfg["d_in"]
        if t2m_mean.shape[-1] != expected_t2m_dim or t2m_std.shape[-1] != expected_t2m_dim:
            raise ValueError(
                f"T2M mean/std dims must be {expected_t2m_dim} (MLD convention), "
                f"got {t2m_mean.shape}, {t2m_std.shape}"
            )
        t2m_stats = (t2m_mean, t2m_std)

    print("Building datasets")
    requested_datasets = {name.strip().upper() for name in args.datasets.split(",") if name.strip()}
    allowed_datasets = {"AIST", "MVH"}
    unknown_datasets = requested_datasets - allowed_datasets
    if unknown_datasets:
        raise ValueError(f"Unknown --datasets entries: {sorted(unknown_datasets)}")
    aist_paths = (
        _aist_paths_for_splits(cfg, args.aist_splits.split(","))
        if "AIST" in requested_datasets
        else []
    )
    mvh_dirs = _read_lines(cfg["mvh_split_val"]) if "MVH" in requested_datasets else []
    genre_to_id = build_genre_to_id(cfg.get("aist_genres", []))
    loaders = []
    if "AIST" in requested_datasets:
        aist_ds = AISTDataset(
            cfg["aist_motions_dir"],
            genre_to_id,
            seq_len,
            mean=mean,
            std=std,
            files=aist_paths,
            cache_root=cfg["cache_root"],
            target_fps=cfg.get("target_fps", 30),
            src_fps=cfg.get("aist_fps", 60),
            camera_ids=cfg.get("aist_cameras", ["01", "02", "08", "09"]),
            expand_cameras=True,
            include_cond=True,
            openpose_dir=cfg.get("aist_openpose_dir", "data/AIST++/Annotations/openpose"),
            cond_cache_root=cfg.get("cond_cache_root", "data/cached_cond"),
            cond_frames_min=args.cond_frames or flow_cfg.get("cond_frames_min", 7),
            cond_frames_max=args.cond_frames or flow_cfg.get("cond_frames_max", 7),
            cond_drop_prob=args.cond_drop_prob,
            crop_mode=args.aist_crop_mode,
        )
        loaders.append((
            "AIST",
            torch.utils.data.DataLoader(
                aist_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
            ),
        ))
    if "MVH" in requested_datasets:
        mvh_ds = MVHumanNetDataset(
            cfg["mvhumannet_root"],
            seq_len,
            mean=mean,
            std=std,
            sequence_dirs=mvh_dirs,
            cache_root=cfg["cache_root"],
            target_fps=cfg.get("target_fps", 30),
            src_fps=cfg.get("mvh_fps", 5),
            camera_ids=cfg.get("mvh_cameras", ["22327091", "22327113", "22327084"]),
            expand_cameras=True,
            include_cond=True,
            openpose_root=cfg.get("mvh_openpose_root", "data/MVHumanNet"),
            cond_cache_root=cfg.get("cond_cache_root", "data/cached_cond"),
            cond_frames_min=args.cond_frames or flow_cfg.get("cond_frames_min", 7),
            cond_frames_max=args.cond_frames or flow_cfg.get("cond_frames_max", 7),
            cond_drop_prob=args.cond_drop_prob,
        )
        loaders.append((
            "MVH",
            torch.utils.data.DataLoader(
                mvh_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
            ),
        ))

    eval_cfg = EvalConfig(
        seq_len=seq_len,
        d_z=d_z,
        latent_len=args.vae_latent_len or seq_len,
        fps=cfg.get("target_fps", 30),
        slack_seconds=args.slack_seconds,
        cam_mode=args.cam_mode,
    )

    openpose_cfg = {
        "aist_dir": cfg.get("aist_openpose_dir", "data/AIST++/Annotations/openpose"),
        "mvh_root": cfg.get("mvh_openpose_root", "data/MVHumanNet"),
        "mv_root": cfg["mvhumannet_root"],
        "mvh_cameras": cfg.get("mvh_cameras", ["22327091", "22327113", "22327084"]),
        "cond_frames_min": args.cond_frames or flow_cfg.get("cond_frames_min", 2),
        "cond_frames_max": args.cond_frames or flow_cfg.get("cond_frames_max", 10),
        "cond_drop_prob": args.cond_drop_prob,
        "aist_fps": cfg.get("aist_fps", 60),
        "mvh_fps": cfg.get("mvh_fps", 5),
        "target_fps": cfg.get("target_fps", 30),
        "cond_cache_root": cfg.get("cond_cache_root", "data/cached_cond"),
        "mean": k2d_mean,
        "std": k2d_std,
    }

    steps_list = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    model_dir = os.path.basename(os.path.dirname(args.flow_ckpt.rstrip(os.sep)))
    out_dir = os.path.join("output", "eval", model_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_json = args.save_json or os.path.join(out_dir, "flow_eval.json")
    save_csv = args.save_csv or os.path.join(out_dir, "flow_eval.csv")
    save_plot = args.save_plot or os.path.join(out_dir, "flow_eval_steps.png")
    summary_rows = []
    per_sample = []
    for steps in steps_list:
        print(f"Evaluating steps={steps}")
        for dataset_name, loader in loaders:
            summary, samples = evaluate_dataset(
                dataset_name,
                loader,
                flow,
                vae,
                eval_cfg,
                mapping,
                computed,
                openpose_cfg,
                (mean, std),
                (latent_mean, latent_std),
                steps,
                args.solver,
                args.num_samples,
                args.seed,
                device=args.device,
                compute_dist=not args.no_dist,
                save_per_sample=args.save_per_sample,
                metric_extractor=t2m_extractor,
                t2m_stats=t2m_stats,
                diversity_times=args.diversity_times,
                multimodality_repeats=args.multimodality_repeats,
                multimodality_times=args.multimodality_times,
                guidance_scale=args.guidance_scale,
            )
            row = {"dataset": dataset_name, "steps": steps}
            row.update(summary)
            summary_rows.append(row)
            if samples:
                for rec in samples:
                    rec["steps"] = steps
                    per_sample.append(rec)

    print(f"Writing outputs: {save_csv}, {save_json}, {save_plot}")
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    fieldnames = ["dataset", "steps"] + [
        k for k in summary_rows[0].keys() if k not in ("dataset", "steps")
    ]
    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    out = {"summary": summary_rows, "per_sample": per_sample}
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        for dataset_name in ("AIST", "MVH"):
            ys = [
                r.get("e2d_strict", 0.0)
                for r in summary_rows
                if r["dataset"] == dataset_name
            ]
            xs = [
                r["steps"]
                for r in summary_rows
                if r["dataset"] == dataset_name
            ]
            plt.plot(xs, ys, marker="o", label=dataset_name)
        plt.xlabel("Steps")
        plt.ylabel("E2D strict")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_plot)
    except Exception:
        pass


if __name__ == "__main__":
    main()
