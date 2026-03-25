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
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.motion.process_motion import ik263_to_smpl22
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--solver", type=str, default="heun")
    parser.add_argument("--style-id", type=int, default=0)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--k2d-npy", type=str, default=None)
    parser.add_argument("--tau-cond-npy", type=str, default=None)
    parser.add_argument("--sample-path", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--dataset", type=str, choices=["auto", "aist", "mvh"], default="auto")
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="result_smpl22.npy")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--src-fps", type=int, default=None)
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="output/flow")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    config = load_config()
    seq_len = config["seq_len"]
    d_z = config["d_z"]
    openpose_stats_path = config.get("openpose_stats_path", "data/openpose_stats.npz")
    target_fps = args.target_fps or config.get("target_fps", None)
    vae_ckpt_path = args.vae_checkpoint or config.get(
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
    stats_path = config.get("stats_path", "data/mean_std_263_train.npz")
    latent_stats_path = config.get("latent_stats_path", "data/latent_stats.npz")
    cond_frames_min = config.get("flow", {}).get("cond_frames_min", 2)
    cond_frames_max = config.get("flow", {}).get("cond_frames_max", 10)
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

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    flow_cfg = config.get("flow", {})
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
    )
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    if args.use_ema and "ema" in state:
        flow.load_state_dict(state["ema"])
    else:
        flow.load_state_dict(state["model"])
    flow.to(device)
    flow.eval()

    vae = MotionVAE(d_in=config["d_in"], d_z=d_z, num_styles=config["num_styles"], max_len=seq_len)
    vae_state = torch.load(vae_ckpt_path, map_location=device)
    vae.load_state_dict(vae_state["model"])
    vae.to(device)
    vae.eval()

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
    style_id_value = args.style_id
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
                src_fps=args.src_fps,
                target_fps=target_fps,
                cache_root=cond_cache_root,
                camera=cam,
            )
            genre = get_genre_code(pkl_path)
            style_id_value = genre_to_id.get(genre, 0)
            domain_id_value = 1
            meta = {"dataset": "aist", "path": pkl_path, "camera": cam, "genre": genre}
        else:
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
                src_fps=args.src_fps,
                target_fps=target_fps,
                cache_root=cond_cache_root,
                camera=cam,
            )
            style_id_value = 0
            domain_id_value = 0
            meta = {"dataset": "mvh", "path": seq_dir, "camera": cam}

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

    tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
    x0 = torch.randn(1, seq_len, d_z, device=device)

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

    cond_batch = {"tau_out": tau_out, "mem": mem, "g": g}
    z_hat = solve_flow(flow.flow, x0, cond_batch, num_steps=args.steps, method=args.solver)
    if latent_mean is not None and latent_std is not None:
        z_hat = z_hat * (latent_std + 1e-6) + latent_mean

    with torch.no_grad():
        x_hat = vae.decode(z_hat, vae.cond(domain_id, style_id))
    ik263 = x_hat.squeeze(0).cpu().numpy()
    mean, std = load_mean_std(stats_path)
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    ik263[:, :cont_end] = ik263[:, :cont_end] * std + mean
    joints = ik263_to_smpl22(ik263)
    joints = yup_to_blender(joints)
    joints = joints - joints[0:1, 0:1, :]

    os.makedirs(run_out_dir, exist_ok=True)
    out_npy = os.path.join(run_out_dir, args.out)
    np.save(out_npy, joints)

    meta_path = os.path.join(run_out_dir, "result_meta.json")
    if hasattr(sample_idx, "tolist"):
        sample_idx_out = sample_idx.tolist()
    else:
        sample_idx_out = list(sample_idx)
    meta_out = {
        "dataset": meta.get("dataset", "unknown"),
        "path": meta.get("path", ""),
        "camera": meta.get("camera", ""),
        "style_id": style_id_value,
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
        "seq_len": seq_len,
        "sparse_indices": sample_idx_out,
        "tau_cond": tau_cond.tolist(),
    }

    replicate_cmd = [
        "python",
        "flowmimic/scripts/sample_flow.py",
        "--checkpoint",
        args.checkpoint,
        "--vae-checkpoint",
        vae_ckpt_path,
        "--steps",
        str(args.steps),
        "--solver",
        args.solver,
        "--out",
        args.out,
        "--out-dir",
        args.out_dir,
        "--device",
        args.device,
    ]
    if args.use_ema:
        replicate_cmd.append("--use-ema")
    if args.src_fps is not None:
        replicate_cmd.extend(["--src-fps", str(args.src_fps)])
    if args.target_fps is not None:
        replicate_cmd.extend(["--target-fps", str(args.target_fps)])
    if args.k2d_npy:
        replicate_cmd.extend(["--k2d-npy", args.k2d_npy])
        if args.tau_cond_npy:
            replicate_cmd.extend(["--tau-cond-npy", args.tau_cond_npy])
        if args.style_id != 0:
            replicate_cmd.extend(["--style-id", str(args.style_id)])
        if args.domain_id != 0:
            replicate_cmd.extend(["--domain-id", str(args.domain_id)])
    elif meta_out["dataset"] in ("aist", "mvh"):
        replicate_cmd.extend(["--dataset", meta_out["dataset"]])
        if meta_out["path"]:
            replicate_cmd.extend(["--sample-path", meta_out["path"]])
    if args.camera is not None:
        replicate_cmd.extend(["--camera", args.camera])
    if args.seed is not None:
        replicate_cmd.extend(["--seed", str(args.seed)])
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
    print(f"start: {meta_out.get('start', None)}")
    print(f"replicate_command: {replicate_command}")
    if meta.get("dataset") in ("aist", "mvh"):
        print(
            "Run 'python flowmimic/tools/extract_cond_media.py' to process the latest sample "
            f"(or pass --meta {meta_path})."
        )


if __name__ == "__main__":
    main()
