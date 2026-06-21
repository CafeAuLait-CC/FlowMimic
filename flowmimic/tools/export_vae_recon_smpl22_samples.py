#!/usr/bin/env python3
"""Export AIST VAE reconstruction examples as SMPL22 xyz npy clips."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CONT_END = 259


def _load_names(root: Path, split: str) -> list[str]:
    names = []
    for part in split.split(","):
        part = part.strip()
        if not part:
            continue
        path = root / f"{part}.txt"
        names.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    seen = set()
    unique = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _pick_names(names: list[str], count: int, seed: int) -> list[str]:
    if count >= len(names):
        return names
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(names), size=count, replace=False))
    return [names[int(i)] for i in idx]


def _load_names_from_manifest(path: str) -> list[str]:
    names = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["sample"]
            if name not in names:
                names.append(name)
    return names


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _recover_smpl22(raw_features: torch.Tensor) -> np.ndarray:
    from mld.data.humanml.scripts.motion_process import recover_from_ric

    joints = recover_from_ric(raw_features, 22)
    joints = joints.detach().cpu().numpy().astype(np.float32)
    if joints.shape[-2:] != (22, 3):
        raise ValueError(f"Unexpected SMPL22 shape: {joints.shape}")
    return joints


def _convert_output_space(joints: np.ndarray, output_space: str) -> np.ndarray:
    if output_space == "yup":
        return joints
    if output_space == "blender":
        from flowmimic.src.data.dataloader import yup_to_blender

        return yup_to_blender(joints).astype(np.float32)
    raise ValueError(f"Unsupported output space: {output_space}")


def _load_flowmimic_model(ckpt_path: str, device: torch.device):
    from flowmimic.src.model.vae.motion_vae import MotionVAE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    config = ckpt.get("config", {})
    latent_len = config.get("latent_len")
    latent_token_mode = config.get("latent_token_mode", "pool")
    max_len = state["enc_pos"].shape[1]
    num_styles = state["cond.style_emb.weight"].shape[0]
    model = MotionVAE(
        max_len=max_len,
        num_styles=num_styles,
        latent_len=latent_len,
        latent_token_mode=latent_token_mode,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _load_vqvae_model(ckpt_path: str, device: torch.device):
    from flowmimic.src.model.vae.motion_vqvae import MotionVQVAE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    config = ckpt.get("config", {})
    latent_token_mode = config.get(
        "latent_token_mode", "query" if "latent_queries" in state else "pool"
    )
    if "latent_queries" in state:
        latent_len = state["latent_queries"].shape[1]
    elif "latent_pos" in state:
        latent_len = state["latent_pos"].shape[1]
    else:
        latent_len = config.get("latent_len")
    model = MotionVQVAE(
        d_in=state["enc_in.weight"].shape[1],
        d_z=state["to_latent.weight"].shape[0],
        d_model=state["enc_in.weight"].shape[0],
        max_len=state["enc_pos"].shape[1],
        num_styles=state["cond.style_emb.weight"].shape[0],
        latent_len=latent_len,
        latent_token_mode=latent_token_mode,
        codebook_size=state["quantizer.embed"].shape[0],
        commitment_weight=config.get("commitment_weight", 0.25),
        codebook_decay=config.get("codebook_decay", 0.99),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _load_stats_npz(path: str, device: torch.device):
    stats = np.load(path)
    mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(device)
    std = torch.from_numpy(stats["std"].astype(np.float32)).to(device)
    return mean, std


def _flowmimic_reconstruct(
    model,
    raw: torch.Tensor,
    stats_path: str,
    names: list[str],
    genre_to_id: dict[str, int],
    device: torch.device,
):
    from flowmimic.src.model.vae.datasets.aist_filename_parser import get_genre_code

    mean, std = _load_stats_npz(stats_path, device)
    norm = raw.clone()
    norm[..., :CONT_END] = (norm[..., :CONT_END] - mean) / std
    domain_id = torch.ones(raw.shape[0], dtype=torch.long, device=device)
    style_id = torch.zeros(raw.shape[0], dtype=torch.long, device=device)
    for i, name in enumerate(names):
        style_id[i] = int(genre_to_id.get(get_genre_code(name), 0))
    mask = torch.ones(raw.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        cond = model.cond(domain_id, style_id)
        _enc_h, mu, _logvar = model.encode(norm, cond, mask=mask)
        pred_norm = model.decode(mu, cond, mask=mask, out_len=raw.shape[1])
    pred = pred_norm.clone()
    pred[..., :CONT_END] = pred[..., :CONT_END] * std + mean
    pred[..., CONT_END:] = torch.sigmoid(pred[..., CONT_END:])
    return pred


def _vqvae_reconstruct(
    model,
    raw: torch.Tensor,
    stats_path: str,
    names: list[str],
    genre_to_id: dict[str, int],
    device: torch.device,
):
    from flowmimic.src.model.vae.datasets.aist_filename_parser import get_genre_code

    mean, std = _load_stats_npz(stats_path, device)
    norm = raw.clone()
    norm[..., :CONT_END] = (norm[..., :CONT_END] - mean) / std
    domain_id = torch.ones(raw.shape[0], dtype=torch.long, device=device)
    style_id = torch.zeros(raw.shape[0], dtype=torch.long, device=device)
    for i, name in enumerate(names):
        style_id[i] = int(genre_to_id.get(get_genre_code(name), 0))
    mask = torch.ones(raw.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        outputs = model(
            norm,
            domain_id,
            style_id,
            mask=mask,
            update_codebook=False,
        )
    pred = outputs["x_hat"].clone()
    pred[..., :CONT_END] = pred[..., :CONT_END] * std + mean
    pred[..., CONT_END:] = torch.sigmoid(pred[..., CONT_END:])
    return pred


def _load_mld_model(ckpt_path: str, device: torch.device):
    workspace = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(workspace / "motion-latent-diffusion"))
    from mld.models.architectures.mld_vae import MldVae

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
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("vae."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("vae.")
    }
    model.load_state_dict(state)
    model.eval()
    return model


def _mld_reconstruct(model, raw: torch.Tensor, data_root: Path, device: torch.device):
    mean = torch.from_numpy(np.load(data_root / "Mean.npy").astype(np.float32)).to(device)
    std = torch.from_numpy(np.load(data_root / "Std.npy").astype(np.float32)).to(device)
    norm = (raw - mean) / std
    lengths = [raw.shape[1]] * raw.shape[0]
    with torch.no_grad():
        _z, dist = model.encode(norm, lengths)
        pred_norm = model.decode(dist.loc, lengths)
    return pred_norm * std + mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="prepared/aist_mld_humanml3d")
    parser.add_argument("--split", default="val,test")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional previous export manifest; sample names are reused in manifest order.",
    )
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/vae_recon_smpl22_samples")
    parser.add_argument(
        "--old-flowmimic-ckpt",
        default="checkpoints/vae/len200_smooth_decoder/motion_vae_best.pt",
    )
    parser.add_argument(
        "--old-flowmimic-stats",
        default="data/mean_std_263_train.npz",
    )
    parser.add_argument(
        "--compact-flowmimic-ckpt",
        default="checkpoints/vae/aist_mvh_len196_latent1_query_gpu1_260522-233636/motion_vae_best.pt",
    )
    parser.add_argument(
        "--mld-ckpt",
        default="runs/mld/mld/aist_ik263_vae_196/checkpoints/epoch=5999.ckpt",
    )
    parser.add_argument(
        "--vqvae-latest-ckpt",
        default="checkpoints/vqvae/aist_mvh_len196_latent16_code1024_ddp_260609-134633/motion_vqvae_latest.pt",
    )
    parser.add_argument(
        "--vqvae-best-ckpt",
        default="checkpoints/vqvae/aist_mvh_len196_latent16_code1024_ddp_260609-134633/motion_vqvae_best.pt",
    )
    parser.add_argument(
        "--vqvae-stats",
        default=None,
        help="Optional stats path override for VQ-VAE checkpoints.",
    )
    parser.add_argument("--genre-map", default="flowmimic/src/config/genre_to_id.json")
    parser.add_argument(
        "--models",
        default="input,old,compact,mld",
        help="Comma-separated subset: input,old,compact,mld,vqvae_latest,vqvae_best.",
    )
    parser.add_argument(
        "--output-space",
        choices=("blender", "yup"),
        default="blender",
        help="Coordinate space for saved SMPL22 npy clips. blender is Z-up; yup preserves the model/eval space.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        names = _load_names_from_manifest(args.manifest)
    else:
        names = _pick_names(_load_names(data_root, args.split), args.num_samples, args.seed)
    raw_np = np.stack(
        [np.load(data_root / "new_joint_vecs" / f"{name}.npy").astype(np.float32) for name in names],
        axis=0,
    )
    if raw_np.shape[1:] != (196, 263):
        raise ValueError(f"Expected [N,196,263] features, got {raw_np.shape}")
    raw = torch.from_numpy(raw_np).to(device)

    with open(args.genre_map, "r", encoding="utf-8") as f:
        genre_to_id = json.load(f)

    requested_models = {item.strip().lower() for item in args.models.split(",") if item.strip()}
    unknown_models = requested_models - {
        "input",
        "old",
        "compact",
        "mld",
        "vqvae_latest",
        "vqvae_best",
    }
    if unknown_models:
        raise ValueError(f"Unknown --models entries: {sorted(unknown_models)}")

    recon = {}
    old_stats = args.old_flowmimic_stats
    compact_stats = None
    if "input" in requested_models:
        recon["input"] = raw
    old_ckpt = None
    if "old" in requested_models:
        old_flow, old_ckpt = _load_flowmimic_model(args.old_flowmimic_ckpt, device)
        recon["flowmimic_old_len200"] = _flowmimic_reconstruct(
            old_flow, raw, old_stats, names, genre_to_id, device
        )
    if "compact" in requested_models:
        compact_flow, compact_ckpt = _load_flowmimic_model(args.compact_flowmimic_ckpt, device)
        compact_stats = compact_ckpt.get("stats_path")
        if compact_stats is None and old_ckpt is not None:
            compact_stats = old_ckpt.get("stats_path")
        compact_stats = compact_stats or old_stats
        recon["flowmimic_compact_query"] = _flowmimic_reconstruct(
            compact_flow, raw, compact_stats, names, genre_to_id, device
        )
    if "mld" in requested_models:
        mld = _load_mld_model(args.mld_ckpt, device)
        recon["mld"] = _mld_reconstruct(mld, raw, data_root, device)
    if "vqvae_latest" in requested_models:
        vqvae_latest, vqvae_latest_ckpt = _load_vqvae_model(args.vqvae_latest_ckpt, device)
        vqvae_latest_stats = args.vqvae_stats or vqvae_latest_ckpt.get("stats_path") or old_stats
        recon["vqvae_latest"] = _vqvae_reconstruct(
            vqvae_latest, raw, vqvae_latest_stats, names, genre_to_id, device
        )
    if "vqvae_best" in requested_models:
        vqvae_best, vqvae_best_ckpt = _load_vqvae_model(args.vqvae_best_ckpt, device)
        vqvae_best_stats = args.vqvae_stats or vqvae_best_ckpt.get("stats_path") or old_stats
        recon["vqvae_best_val"] = _vqvae_reconstruct(
            vqvae_best, raw, vqvae_best_stats, names, genre_to_id, device
        )
    joints = {key: _recover_smpl22(value) for key, value in recon.items()}

    manifest_rows = []
    for i, name in enumerate(names):
        safe = _safe_name(name)
        for key, value in joints.items():
            out_path = output_dir / f"{i:03d}_{safe}_{key}.npy"
            clip = _convert_output_space(value[i], args.output_space)
            if clip.shape != (196, 22, 3):
                raise ValueError(f"{out_path} has bad shape {clip.shape}")
            if not np.isfinite(clip).all():
                raise ValueError(f"{out_path} contains non-finite values")
            np.save(out_path, clip)
            manifest_rows.append(
                {
                    "index": i,
                    "sample": name,
                    "kind": key,
                    "path": str(out_path),
                    "shape": "196,22,3",
                    "coordinate_space": args.output_space,
                }
            )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "sample",
                "kind",
                "path",
                "shape",
                "coordinate_space",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    config_path = output_dir / "export_config.json"
    config_path.write_text(
        json.dumps(
            {
                "samples": names,
                "old_flowmimic_ckpt": args.old_flowmimic_ckpt,
                "old_flowmimic_stats": old_stats,
                "compact_flowmimic_ckpt": args.compact_flowmimic_ckpt,
                "compact_flowmimic_stats": compact_stats,
                "mld_ckpt": args.mld_ckpt,
                "vqvae_latest_ckpt": args.vqvae_latest_ckpt,
                "vqvae_best_ckpt": args.vqvae_best_ckpt,
                "vqvae_stats": args.vqvae_stats,
                "split": args.split,
                "manifest": args.manifest,
                "models": sorted(requested_models),
                "output_space": args.output_space,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported {len(names)} samples to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
