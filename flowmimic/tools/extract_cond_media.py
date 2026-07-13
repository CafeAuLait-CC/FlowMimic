"""Extract a cropped AIST++ video clip and condition preview frames from flow sample metadata.

Example:
  python flowmimic/tools/extract_cond_media.py
"""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys

import numpy as np
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.data.dataloader import (
    blender_to_yup,
    load_aistpp_smpl22_30fps,
    load_mvhumannet_sequence_smpl22_30fps,
    yup_to_blender,
)
from flowmimic.src.motion.process_motion import align_smpl22_floor_and_center


def _parse_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_list(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []


def _as_int_list(value):
    if not isinstance(value, list):
        value = _parse_list(str(value))
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _select_preview_indices(indices, max_items):
    indices = [int(i) for i in indices]
    if max_items <= 0 or len(indices) <= max_items:
        return indices
    pick = np.linspace(0, len(indices) - 1, max_items)
    pick = np.unique(np.round(pick).astype(int))
    return [indices[int(i)] for i in pick]


def _write_preview_meta(path, condition_indices, frame_indices, all_frames):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "condition_frame_count": len(condition_indices),
                "preview_frame_count": len(frame_indices),
                "preview_indices": [int(i) for i in frame_indices],
                "all_frames": bool(all_frames),
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="output/flow/last/result_meta.json")
    parser.add_argument("--out-dir", default="output/flow/last/cond_media")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--all-frames", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.meta):
        raise FileNotFoundError(
            f"Meta file not found: {args.meta}. Run sample_flow.py first or pass --meta explicitly."
        )

    config = load_config()
    target_fps = args.fps or config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = 25
    video_root = "data/AIST++/Videos"
    mvh_root = os.path.join(os.path.expanduser("~"), "hdd", "MVHumanNet_Data")

    meta = _parse_meta(args.meta)
    dataset = meta.get("dataset", "aist")
    motion_path = meta.get("path", "")
    if not motion_path:
        raise ValueError("Missing motion path in meta file")
    camera = meta.get("camera", "")
    if not camera:
        raise ValueError("Missing camera in meta file")

    seq_len = int(meta.get("seq_len", 0) or 0)
    orig_len = int(meta.get("orig_len", 0) or 0)
    start = int(meta.get("start", 0) or 0)
    if seq_len <= 0:
        raise ValueError("Invalid seq_len in meta file")
    if orig_len <= 0:
        orig_len = seq_len

    condition_indices = _as_int_list(
        meta.get("condition_indices", meta.get("sparse_indices", []))
    )
    preview_indices = _as_int_list(meta.get("condition_preview_indices", []))
    if args.all_frames:
        frame_indices = condition_indices
    elif preview_indices:
        frame_indices = preview_indices
    else:
        frame_indices = _select_preview_indices(condition_indices, args.max_frames)

    tag = "result"
    out_dir = args.out_dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, f"{tag}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    smpl_out = os.path.join(out_dir, "cond_clip_smpl22.npy")
    preview_meta_path = os.path.join(out_dir, "condition_preview_meta.json")
    _write_preview_meta(
        preview_meta_path,
        condition_indices=condition_indices,
        frame_indices=frame_indices,
        all_frames=args.all_frames,
    )

    if dataset == "aist":
        base = os.path.splitext(os.path.basename(motion_path))[0]
        base_cam = base.replace("_cAll_", f"_c{camera}_")
        video_path = os.path.join(video_root, f"{base_cam}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        clip_frames = min(seq_len, max(orig_len - start, 0))
        clip_duration = clip_frames / float(target_fps)
        clip_start = start / float(target_fps)

        clip_path = os.path.join(out_dir, f"{tag}_clip.mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{clip_start:.6f}",
            "-t",
            f"{clip_duration:.6f}",
            "-i",
            video_path,
            "-vcodec",
            "libx264",
            "-acodec",
            "aac",
            "-movflags",
            "faststart",
            clip_path,
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        for idx in frame_indices:
            ts = float(idx) / float(target_fps)
            out_path = os.path.join(frame_dir, f"frame_{int(idx):06d}.png")
            frame_cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.6f}",
                "-i",
                clip_path,
                "-vframes",
                "1",
                out_path,
            ]
            subprocess.run(frame_cmd, check=True)

        joints = load_aistpp_smpl22_30fps(
            motion_path, target_fps=target_fps, src_fps=aist_fps
        )
        joints = blender_to_yup(joints)
        if joints.shape[0] >= seq_len:
            joints = joints[start : start + seq_len]
        else:
            pad_len = seq_len - joints.shape[0]
            joints = np.concatenate(
                [joints, np.zeros((pad_len, 22, 3), dtype=joints.dtype)], axis=0
            )
        joints = align_smpl22_floor_and_center(joints)
        joints = yup_to_blender(joints)
        np.save(smpl_out, joints)

        print(f"Saved clip: {clip_path}")
        print(
            f"Saved preview frames: {frame_dir} "
            f"({len(frame_indices)}/{len(condition_indices)} condition frames)"
        )
        print(f"Saved smpl22: {smpl_out}")
        print(f"Saved preview meta: {preview_meta_path}")
        return

    if dataset not in ("mvh", "mvh_kinematic"):
        raise ValueError(f"Unsupported dataset in meta: {dataset}")

    if dataset == "mvh_kinematic" and motion_path.endswith(".pkl"):
        raise ValueError("mvh_kinematic meta expects a MVHumanNet sequence directory")

    parts = motion_path.split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"Unexpected MVH path: {motion_path}")
    part = parts[-3]
    motion = parts[-2]
    image_root = os.path.join(mvh_root, part, motion, "images_lr", camera)

    def _find_frame_path(frame_idx):
        candidates = [
            f"{frame_idx}_img.jpg",
            f"{frame_idx:04d}_img.jpg",
            f"{frame_idx:06d}_img.jpg",
            f"{frame_idx:08d}_img.jpg",
        ]
        for name in candidates:
            path = os.path.join(image_root, name)
            if os.path.exists(path):
                return path
        return None

    frame_map = []
    for idx in frame_indices:
        abs_idx = start + int(idx)
        time_sec = float(abs_idx) / float(target_fps)
        frame_src = int(round(time_sec * mvh_fps))
        frame_snap = int(round(frame_src / 5.0) * 5)
        frame_snap = max(frame_snap, 0)
        src_path = _find_frame_path(frame_snap)
        if not src_path:
            continue
        out_path = os.path.join(frame_dir, f"frame_{int(idx):06d}.jpg")
        shutil.copy2(src_path, out_path)
        frame_map.append((idx, frame_snap, out_path))

    joints = load_mvhumannet_sequence_smpl22_30fps(
        motion_path, target_fps=target_fps, src_fps=5
    )
    joints = blender_to_yup(joints)
    if joints.shape[0] >= seq_len:
        joints = joints[start : start + seq_len]
    else:
        pad_len = seq_len - joints.shape[0]
        joints = np.concatenate(
            [joints, np.zeros((pad_len, 22, 3), dtype=joints.dtype)], axis=0
        )
    joints = align_smpl22_floor_and_center(joints)
    joints = yup_to_blender(joints)
    np.save(smpl_out, joints)

    map_path = os.path.join(out_dir, f"{tag}_frame_map.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        for idx, src_frame, out_path in frame_map:
            f.write(f"{idx}\t{src_frame}\t{out_path}\n")

    print(
        f"Saved preview frames: {frame_dir} "
        f"({len(frame_indices)}/{len(condition_indices)} condition frames)"
    )
    print(f"Saved smpl22: {smpl_out}")
    print(f"Saved preview meta: {preview_meta_path}")


if __name__ == "__main__":
    main()
