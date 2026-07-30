import argparse
import json
import os
import random
import shlex
import sys
from datetime import datetime

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.cond_api import build_cond_inputs, build_dummy_cond
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.checkpoint import (
    flow_state_uses_latent_slot_adapter,
    flow_state_uses_relative_time_bias,
    flow_state_uses_true_null_condition,
    infer_latent_slot_adapter_config,
    infer_relative_time_hidden_dim,
    load_flow_state_dict,
)
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.backend import decode_motion_latent, load_vae_backend
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.motion.process_motion import (
    align_smpl22_with_contact_and_center,
    ik263_to_smpl22,
)
from flowmimic.src.data.openpose import (
    VIS_CONF_THRESHOLD,
    load_aist_openpose,
    load_mvh_openpose,
)
from flowmimic.src.model.vae.stats import load_mean_std
from flowmimic.src.data.dataloader import yup_to_blender
from flowmimic.src.model.vae.datasets.aist_filename_parser import get_genre_code
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id


def _join_cmd(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)


def _checkpoint_metadata(state):
    metadata = state.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _metadata_section(metadata, key):
    value = metadata.get(key, {})
    return value if isinstance(value, dict) else {}


def _select_preview_indices(indices, max_items=24):
    indices = [int(i) for i in indices]
    if max_items <= 0 or len(indices) <= max_items:
        return indices
    pick = np.linspace(0, len(indices) - 1, max_items)
    pick = np.unique(np.round(pick).astype(int))
    return [indices[int(i)] for i in pick]


def main():
    config = load_config()
    sample_cfg = config.get("sample", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--latent-stats-path", default=None)
    parser.add_argument(
        "--vae-type",
        choices=("auto", "motion_vae", "motion_vqvae"),
        default="auto",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument(
        "--cond-frames",
        type=int,
        default=None,
        help="Number of uniformly spaced condition frames; defaults to checkpoint metadata.",
    )
    parser.add_argument("--steps", type=int, default=sample_cfg.get("steps", 8))
    parser.add_argument("--solver", type=str, default=sample_cfg.get("solver", "heun"))
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=sample_cfg.get("guidance_scale", 1.0),
    )
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--k2d-npy", type=str, default=None)
    parser.add_argument("--sample-path", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--src-fps", type=float, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--dataset", type=str, choices=["auto", "aist", "mvh"], default="auto")
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=sample_cfg.get("output_name", "result_smpl22.npy"))
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--out-dir", type=str, default=sample_cfg.get("output_dir", "output/flow"))
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    ckpt_metadata = _checkpoint_metadata(state)
    seq_len = args.seq_len or ckpt_metadata.get("seq_len") or config["seq_len"]
    openpose_stats_path = ckpt_metadata.get("openpose_stats_path") or config.get(
        "openpose_stats_path", "data/openpose_stats.npz"
    )
    target_fps = (
        args.target_fps
        or sample_cfg.get("target_fps")
        or config.get("target_fps", 30)
    )
    source_fps_override = args.src_fps or sample_cfg.get("src_fps")
    vae_ckpt_path = args.vae_checkpoint or ckpt_metadata.get("vae_ckpt") or config.get(
        "vae_ckpt", "checkpoints/motion_vae_best.pt"
    )
    ckpt_parent = os.path.basename(os.path.dirname(os.path.normpath(args.checkpoint)))
    model_name = ckpt_parent if ckpt_parent else "model"
    ts_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = ts_base
    run_out_dir = os.path.join(args.out_dir, model_name, run_tag)
    suffix = 1
    while os.path.exists(run_out_dir):
        run_tag = f"{ts_base}_{suffix:02d}"
        run_out_dir = os.path.join(args.out_dir, model_name, run_tag)
        suffix += 1
    stats_path = ckpt_metadata.get("stats_path") or config.get(
        "stats_path", "data/mean_std_263_train.npz"
    )
    latent_stats_path = (
        args.latent_stats_path
        or ckpt_metadata.get("latent_stats_path")
        or config.get("latent_stats_path", "data/latent_stats.npz")
    )
    conditioning_metadata = _metadata_section(ckpt_metadata, "conditioning")
    flow_config_metadata = _metadata_section(ckpt_metadata, "flow_config")
    if args.cond_frames is not None and args.cond_frames < 1:
        parser.error("--cond-frames must be at least 1")
    cond_frames_min = args.cond_frames or (
        conditioning_metadata.get("eval_cond_frames")
        or conditioning_metadata.get("cond_frames_min")
        or flow_config_metadata.get("cond_frames_min")
        or config.get("flow", {}).get("cond_frames_min", 2)
    )
    cond_frames_max = args.cond_frames or (
        conditioning_metadata.get("eval_cond_frames")
        or conditioning_metadata.get("cond_frames_max")
        or flow_config_metadata.get("cond_frames_max")
        or config.get("flow", {}).get("cond_frames_max", 10)
    )
    cond_frames_min = max(1, min(int(cond_frames_min), int(seq_len)))
    cond_frames_max = max(cond_frames_min, min(int(cond_frames_max), int(seq_len)))
    cond_cache_root = config.get("cond_cache_root", "data/cached_cond")
    aist_cameras = config.get("aist_cameras", ["01", "02", "08", "09"])
    mvh_cameras = config.get("mvh_cameras", ["22327091", "22327113", "22327084"])
    aist_openpose_dir = config.get(
        "aist_openpose_dir", "data/AIST++/Annotations/openpose"
    )
    mvh_openpose_root = config.get("mvh_openpose_root", "data/MVHumanNet")
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    aist_split_val = config.get("aist_split_val")
    mvh_split_val = config.get("mvh_split_val")
    genre_to_id = build_genre_to_id(config.get("aist_genres", []))

    seed_used = args.seed
    if seed_used is None:
        seed_used = random.SystemRandom().randint(0, 2**31 - 1)
    random.seed(seed_used)
    np.random.seed(seed_used)
    torch.manual_seed(seed_used)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_used)

    vae_backend = load_vae_backend(
        vae_ckpt_path,
        config,
        device,
        seq_len=seq_len,
        vae_type=args.vae_type,
        latent_len=sample_cfg.get("vae_latent_len"),
        latent_token_mode=sample_cfg.get("vae_latent_token_mode"),
    )
    vae = vae_backend.model
    d_z = vae_backend.d_z
    latent_len = vae_backend.latent_len

    flow_cfg = config.get("flow", {})
    flow_state = state["ema"] if args.use_ema and "ema" in state else state["model"]
    relative_time_bias = flow_state_uses_relative_time_bias(flow_state)
    latent_slot_adapter = flow_state_uses_latent_slot_adapter(flow_state)
    true_null_condition = flow_state_uses_true_null_condition(flow_state)
    slot_adapter_config = infer_latent_slot_adapter_config(
        flow_state,
        default_latent_len=latent_len,
        default_ffn_dim=flow_cfg.get("latent_slot_adapter_ffn_dim", 1024),
    )
    flow_architecture = state.get("metadata", {}).get("flow_architecture", {})
    flow = ConditionalRectFlow(
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
        relative_time_bias=relative_time_bias,
        relative_time_hidden_dim=infer_relative_time_hidden_dim(flow_state),
        latent_len=slot_adapter_config["latent_len"],
        latent_slot_adapter=latent_slot_adapter,
        latent_slot_adapter_heads=int(
            flow_architecture.get(
                "latent_slot_adapter_heads",
                flow_cfg.get("latent_slot_adapter_heads", 8),
            )
        ),
        latent_slot_adapter_ffn_dim=slot_adapter_config["ffn_dim"],
        true_null_condition=true_null_condition,
    )
    load_flow_state_dict(flow, flow_state)
    flow.to(device)
    flow.eval()

    k2d_mean = None
    k2d_std = None
    if os.path.exists(openpose_stats_path):
        stats = np.load(openpose_stats_path)
        k2d_mean = stats["mean"]
        k2d_std = stats["std"]
    latent_mean = None
    latent_std = None
    if os.path.exists(latent_stats_path):
        stats = np.load(latent_stats_path)
        latent_mean = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
        latent_std = torch.tensor(stats["std"], device=device, dtype=torch.float32)

    meta = {}
    style_id_user_set = args.style_id is not None
    style_id_value = args.style_id if args.style_id is not None else 0
    cond_style_id_value = 0
    cond_genre_value = ""
    domain_id_value = args.domain_id
    if args.start is not None and args.start < 0:
        raise ValueError("--start must be >= 0")
    k2d = None
    vis = None
    if args.k2d_npy:
        k2d = np.load(args.k2d_npy)
        vis = None
        if k2d.ndim == 3 and k2d.shape[-1] == 3:
            vis = k2d[..., 2] >= VIS_CONF_THRESHOLD
            k2d = k2d[..., :2]
    elif args.sample_path or args.dataset in ("aist", "mvh", "auto"):
        dataset = args.dataset
        if dataset == "auto":
            if args.sample_path and args.sample_path.endswith(".pkl"):
                dataset = "aist"
            elif args.sample_path:
                dataset = "mvh"
            else:
                dataset = "aist" if random.random() < 0.5 else "mvh"
        if dataset == "aist":
            source_fps = source_fps_override or config.get("aist_fps", 60)
            if args.sample_path:
                pkl_path = args.sample_path
            else:
                if not aist_split_val:
                    raise ValueError("aist_split_val is required for random sampling")
                with open(aist_split_val, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f if line.strip()]
                name = random.choice(names)
                pkl_path = os.path.join(aist_dir, f"{name}.pkl")
            cam = args.camera or random.choice(aist_cameras)
            k2d, vis = load_aist_openpose(
                pkl_path,
                aist_openpose_dir,
                src_fps=source_fps,
                target_fps=target_fps,
                cache_root=cond_cache_root,
                camera=cam,
            )
            genre = get_genre_code(pkl_path)
            cond_genre_value = genre
            cond_style_id_value = genre_to_id.get(genre, 0)
            if not style_id_user_set:
                style_id_value = cond_style_id_value
            domain_id_value = 1
            meta = {
                "dataset": "aist",
                "path": pkl_path,
                "camera": cam,
                "genre": genre,
                "source_fps": float(source_fps),
                "target_fps": float(target_fps),
            }
        else:
            source_fps = source_fps_override or config.get("mvh_fps", 5)
            if args.sample_path:
                seq_dir = args.sample_path
            else:
                if not mvh_split_val:
                    raise ValueError("mvh_split_val is required for random sampling")
                with open(mvh_split_val, "r", encoding="utf-8") as f:
                    seqs = [line.strip() for line in f if line.strip()]
                seq_dir = random.choice(seqs)
            cam = args.camera or random.choice(mvh_cameras)
            k2d, vis = load_mvh_openpose(
                seq_dir,
                mv_root,
                mvh_openpose_root,
                mvh_cameras,
                src_fps=source_fps,
                target_fps=target_fps,
                cache_root=cond_cache_root,
                camera=cam,
            )
            if not style_id_user_set:
                style_id_value = 0
            cond_style_id_value = 0
            domain_id_value = 0
            meta = {
                "dataset": "mvh",
                "path": seq_dir,
                "camera": cam,
                "source_fps": float(source_fps),
                "target_fps": float(target_fps),
            }

    if k2d is None:
        cond = build_dummy_cond(1, device=device)
        tau_cond = cond["tau_cond"].squeeze(0).cpu().numpy()
        sample_idx = list(range(tau_cond.shape[0]))
    else:
        orig_len = k2d.shape[0]
        start = 0
        if orig_len >= seq_len:
            max_start = orig_len - seq_len
            if args.start is None:
                start = random.randint(0, max_start)
            else:
                start = min(args.start, max_start)
            k2d = k2d[start : start + seq_len]
            vis = vis[start : start + seq_len] if vis is not None else None
        else:
            pad_len = seq_len - orig_len
            k2d = np.concatenate(
                [k2d, np.zeros((pad_len, 25, 2), dtype=np.float32)], axis=0
            )
            if vis is not None:
                vis = np.concatenate(
                    [vis, np.zeros((pad_len, 25), dtype=np.float32)], axis=0
                )
        meta["orig_len"] = orig_len
        meta["start"] = start
        t_len = k2d.shape[0]
        k_frames = cond_frames_min
        if t_len <= k_frames:
            sample_idx = np.arange(t_len)
        else:
            sample_idx = np.linspace(0, t_len - 1, k_frames)
            sample_idx = np.unique(np.round(sample_idx).astype(int))
        tau_cond = sample_idx.astype(np.float32) / max(t_len - 1, 1)
        k2d_sparse = k2d[sample_idx]
        vis_sparse = vis[sample_idx] if vis is not None else None
        cond = build_cond_inputs(k2d_sparse, tau_cond, device, vis_mask=vis_sparse)

    tau_out = torch.linspace(0.0, 1.0, steps=latent_len, device=device)
    x0 = torch.randn(1, latent_len, d_z, device=device)

    g, mem, _vis = flow.cond_encoder(
        cond["k2d"],
        cond["tau_cond"],
        vis_mask=cond.get("vis_mask"),
        mean=k2d_mean,
        std=k2d_std,
    )
    style_id = torch.tensor([style_id_value], device=device)
    domain_id = torch.tensor([domain_id_value], device=device)
    style = flow.style_emb(style_id, domain_id, apply_dropout=False)
    g = flow.cond_mlp(torch.cat([g, style], dim=-1))

    cond_batch = {
        "tau_out": tau_out,
        "tau_cond": cond["tau_cond"],
        "mem": mem,
        "g": g,
    }
    if args.guidance_scale != 1.0:
        if true_null_condition:
            (
                g_uncond,
                mem_uncond,
                mem_mask_uncond,
                tau_cond_uncond,
            ) = flow.encode_null_condition(style_id, domain_id)
        else:
            k2d_uncond = torch.zeros_like(cond["k2d"])
            vis_uncond = (
                torch.zeros_like(cond["vis_mask"]) if "vis_mask" in cond else None
            )
            g2d_uncond, mem_uncond, _ = flow.cond_encoder(
                k2d_uncond,
                cond["tau_cond"],
                vis_mask=vis_uncond,
                mean=k2d_mean,
                std=k2d_std,
            )
            style_uncond = flow.style_emb(
                torch.zeros_like(style_id), domain_id, apply_dropout=False
            )
            g_uncond = flow.cond_mlp(
                torch.cat([g2d_uncond, style_uncond], dim=-1)
            )
            mem_mask_uncond = None
            tau_cond_uncond = cond["tau_cond"]
        cond_batch.update(
            {
                "mem_uncond": mem_uncond,
                "g_uncond": g_uncond,
                "mem_mask_uncond": mem_mask_uncond,
                "tau_cond_uncond": tau_cond_uncond,
                "guidance_scale": args.guidance_scale,
            }
        )
    z_hat = solve_flow(flow.flow, x0, cond_batch, num_steps=args.steps, method=args.solver)
    if latent_mean is not None and latent_std is not None:
        z_hat = z_hat * (latent_std + 1e-6) + latent_mean

    with torch.no_grad():
        x_hat = decode_motion_latent(
            vae, z_hat, domain_id, style_id, out_len=seq_len
        )
    ik263 = x_hat.squeeze(0).cpu().numpy()
    mean, std = load_mean_std(stats_path)
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    ik263[:, :cont_end] = ik263[:, :cont_end] * std + mean
    joints = ik263_to_smpl22(ik263)
    joints = align_smpl22_with_contact_and_center(ik263, joints)
    joints = yup_to_blender(joints)

    os.makedirs(run_out_dir, exist_ok=True)
    out_npy = os.path.join(run_out_dir, args.out)
    np.save(out_npy, joints)

    meta_path = os.path.join(run_out_dir, "result_meta.json")
    if hasattr(sample_idx, "tolist"):
        sample_idx_out = sample_idx.tolist()
    else:
        sample_idx_out = list(sample_idx)
    sample_idx_out = [int(i) for i in sample_idx_out]
    preview_idx_out = _select_preview_indices(sample_idx_out, max_items=24)
    meta_out = {
        "dataset": meta.get("dataset", "unknown"),
        "path": meta.get("path", ""),
        "camera": meta.get("camera", ""),
        "style_id": style_id_value,
        "cond_style_id": cond_style_id_value,
        "cond_genre": cond_genre_value,
        "domain_id": domain_id_value,
        "flow_checkpoint": args.checkpoint,
        "flow_checkpoint_dir": os.path.dirname(args.checkpoint),
        "flow_model_name": model_name,
        "vae_checkpoint": vae_ckpt_path,
        "vae_checkpoint_dir": os.path.dirname(vae_ckpt_path),
        "run_timestamp": run_tag,
        "output_dir": run_out_dir,
        "orig_len": meta.get("orig_len", ""),
        "start": meta.get("start", ""),
        "source_fps": meta.get("source_fps"),
        "target_fps": meta.get("target_fps", float(target_fps)),
        "seed": seed_used,
        "seq_len": seq_len,
        "latent_len": latent_len,
        "vae_type": vae_backend.vae_type,
        "guidance_scale": args.guidance_scale,
        "solver_steps": args.steps,
        "solver": args.solver,
        "use_ema": args.use_ema,
        "sparse_indices": sample_idx_out,
        "condition_indices": sample_idx_out,
        "condition_frame_count": len(sample_idx_out),
        "condition_preview_indices": preview_idx_out,
        "condition_preview_frame_count": len(preview_idx_out),
        "condition_frames_min": cond_frames_min,
        "condition_frames_max": cond_frames_max,
        "condition_frames_requested": args.cond_frames,
        "tau_cond": tau_cond.tolist(),
    }

    replicate_cmd = [
        "python",
        "flowmimic/scripts/sample_flow.py",
        "--checkpoint",
        args.checkpoint,
        "--vae-checkpoint",
        vae_ckpt_path,
        "--vae-type",
        args.vae_type,
        "--seq-len",
        str(seq_len),
        "--steps",
        str(args.steps),
        "--solver",
        args.solver,
        "--guidance-scale",
        str(args.guidance_scale),
        "--cond-frames",
        str(len(sample_idx_out)),
        "--out",
        args.out,
        "--out-dir",
        args.out_dir,
        "--device",
        args.device,
    ]
    if args.use_ema:
        replicate_cmd.append("--use-ema")
    if args.latent_stats_path is not None:
        replicate_cmd.extend(["--latent-stats-path", args.latent_stats_path])
    if args.k2d_npy:
        replicate_cmd.extend(["--k2d-npy", args.k2d_npy])
    elif meta_out["dataset"] in ("aist", "mvh"):
        replicate_cmd.extend(["--dataset", meta_out["dataset"]])
        if meta_out["path"]:
            replicate_cmd.extend(["--sample-path", meta_out["path"]])
    if style_id_value != 0 or style_id_user_set:
        replicate_cmd.extend(["--style-id", str(style_id_value)])
    if domain_id_value != 0 or args.domain_id != 0:
        replicate_cmd.extend(["--domain-id", str(domain_id_value)])
    if args.camera is not None:
        replicate_cmd.extend(["--camera", args.camera])
    if meta_out.get("source_fps") is not None:
        replicate_cmd.extend(["--src-fps", str(meta_out["source_fps"])])
    if meta_out.get("target_fps") is not None:
        replicate_cmd.extend(["--target-fps", str(meta_out["target_fps"])])
    replicate_cmd.extend(["--seed", str(seed_used)])
    if meta_out.get("start", None) is not None:
        replicate_cmd.extend(["--start", str(meta_out["start"])])
    replicate_command = _join_cmd(replicate_cmd)
    meta_out["replicate_command"] = replicate_command

    last_link = os.path.join(args.out_dir, "last")
    if os.path.lexists(last_link):
        if os.path.islink(last_link) or os.path.isfile(last_link):
            os.unlink(last_link)
        else:
            raise RuntimeError(
                f"Cannot update last symlink because '{last_link}' exists as a directory. "
                "Please move/remove it first."
            )
    rel_target = os.path.relpath(run_out_dir, os.path.dirname(last_link))
    os.symlink(rel_target, last_link)
    meta_out["last_symlink"] = last_link

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Updated latest symlink: {last_link} -> {run_out_dir}")
    print(f"seed: {seed_used}")
    print(f"start: {meta_out.get('start', None)}")
    print(f"replicate_command: {replicate_command}")
    if meta.get("dataset") in ("aist", "mvh"):
        print(
            "Run 'python flowmimic/tools/extract_cond_media.py' to process the latest sample "
            f"(or pass --meta {meta_path})."
        )


if __name__ == "__main__":
    main()
