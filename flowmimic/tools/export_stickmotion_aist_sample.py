#!/usr/bin/env python3
"""Generate one aligned AIST++ sample with StickMotion and export its sketches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from flowmimic.tools.export_stickmotion_aist_samples import (
    _collate_fn,
    _convert_output_space,
    _patch_stickmotion_runtime,
    _recover_smpl22,
    _tensor_length,
    _to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--text", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=196)
    parser.add_argument("--sketch-frames", type=int, nargs=3, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-output", default=None)
    parser.add_argument("--tracks-output", required=True)
    parser.add_argument("--locus-output", default=None)
    parser.add_argument("--meta", required=True)
    parser.add_argument(
        "--config", default="configs/aist/stickmotion_remodiffuse_aist_eval.py"
    )
    parser.add_argument(
        "--ckpt", default="runs/stickmotion/human_ml3d/aist_remodiffuse/last.ckpt"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[2]
    stick_root = workspace / "stickmotion"
    config_path = (workspace / args.config).resolve()
    ckpt_path = (workspace / args.ckpt).resolve()
    output_path = (workspace / args.output).resolve()
    ref_path = (workspace / args.reference_output).resolve() if args.reference_output else None
    tracks_path = (workspace / args.tracks_output).resolve()
    locus_path = (workspace / args.locus_output).resolve() if args.locus_output else None
    meta_path = (workspace / args.meta).resolve()

    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(stick_root))
    old_cwd = Path.cwd()
    os.chdir(stick_root)
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-codex")
    try:
        from lightning.pytorch import seed_everything
        from mmcv import Config

        from mogen.apis.lg_train import LgModel
        from mogen.datasets import build_dataset
        from mogen.utils.plot_utils import recover_from_ric

        _patch_stickmotion_runtime()
        seed_everything(args.seed, workers=True)
        device = torch.device(args.device)

        cfg = Config.fromfile(str(config_path))
        cfg.data.test.ann_file = f"{args.split}.txt"
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
        ann_path = Path(dataset.ann_file)
        sample_ids = [
            line.strip()
            for line in ann_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        try:
            sample_index = sample_ids.index(args.sample_id)
        except ValueError as exc:
            raise ValueError(f"{args.sample_id} is not in {ann_path}") from exc

        if args.start < 0:
            raise ValueError("--start must be >= 0")
        if args.length != int(dataset.crop_size):
            raise ValueError(
                f"StickMotion is configured for {dataset.crop_size} frames, got {args.length}"
            )
        full_motion = np.asarray(dataset.data_infos[sample_index]["motion"])
        clip_end = args.start + args.length
        if clip_end > len(full_motion):
            raise ValueError(
                f"Requested clip [{args.start}:{clip_end}] exceeds {len(full_motion)} frames"
            )
        clip_motion = full_motion[args.start:clip_end].copy()
        sketch_frames = args.sketch_frames or [
            int(p * args.length) for p in (0.125, 0.5, 0.875)
        ]
        if len(set(sketch_frames)) != len(sketch_frames):
            raise ValueError("--sketch-frames must contain three distinct indices")
        if any(index < 0 or index >= args.length for index in sketch_frames):
            raise ValueError(f"--sketch-frames must be within [0, {args.length - 1}]")

        # Force the camera-matched caption selected by the comparison driver.
        dataset.data_infos[sample_index]["motion"] = clip_motion
        dataset.data_infos[sample_index]["text"] = [args.text]
        dataset.data_infos[sample_index]["token"] = [args.token]
        sample = dataset.prepare_data(sample_index)

        # The upstream test dataset fixes sketches at 12.5%, 50%, and 87.5%.
        # Replace those dense conditioning tensors when custom local frames are requested.
        if args.sketch_frames is not None:
            joints = recover_from_ric(
                torch.as_tensor(clip_motion, dtype=torch.float32), dataset.joint_num
            ).cpu().numpy()
            dense_tracks = torch.zeros_like(sample["stickman_tracks"])
            stick_mask = torch.zeros_like(sample["stick_mask"])
            for index in sketch_frames:
                track, _ = dataset.stickman(
                    joints[index], return_array=True, point_len=64
                )
                dense_tracks[index] = torch.as_tensor(
                    track, dtype=dense_tracks.dtype
                )
                stick_mask[index] = 1
            sample["stickman_tracks"] = dense_tracks
            sample["stick_mask"] = stick_mask

        batch = _to_device(_collate_fn([sample]), device)

        model = LgModel(cfg, dataset)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        model.eval()
        model.model.others_cuda()

        with torch.no_grad():
            result = model.model(return_loss=False, **batch)[0]

        length = _tensor_length(result["motion_length"])
        mean = torch.as_tensor(dataset.mean, dtype=torch.float32)
        std = torch.as_tensor(dataset.std, dtype=torch.float32)
        pred_raw = result["pred_motion"][:length].float().cpu() * (std + 1e-9) + mean
        ref_raw = result["motion"][:length].float().cpu() * (std + 1e-9) + mean
        pred_joints = _convert_output_space(_recover_smpl22(pred_raw), "blender")
        ref_joints = _convert_output_space(_recover_smpl22(ref_raw), "blender")

        stick_mask = result["stick_mask"].detach().cpu().numpy().reshape(-1)
        active_indices = np.flatnonzero(stick_mask > 0.5).astype(np.int64)
        padded_tracks = result["stickman_tracks"].detach().cpu().numpy()
        tracks = padded_tracks[active_indices].astype(np.float32)
        locus = batch["locus"][0, :length].detach().cpu().numpy().astype(np.float32)

        for path in (output_path, tracks_path, meta_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, pred_joints)
        np.save(tracks_path, tracks)
        if ref_path is not None:
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(ref_path, ref_joints)
        if locus_path is not None:
            locus_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(locus_path, locus)

        metadata = {
            "method": "stickmotion",
            "sample_id": args.sample_id,
            "split": args.split,
            "sample_index": sample_index,
            "text": args.text,
            "token": args.token,
            "clip_start": args.start,
            "length": length,
            "seed": args.seed,
            "checkpoint": str(ckpt_path),
            "config": str(config_path),
            "output": str(output_path),
            "reference_output": str(ref_path) if ref_path else None,
            "stickman_tracks": str(tracks_path),
            "stickman_frame_indices": active_indices.tolist(),
            "stickman_source_frame_indices": (
                active_indices + args.start
            ).tolist(),
            "locus": str(locus_path) if locus_path else None,
            "space": "blender_zup",
        }
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
