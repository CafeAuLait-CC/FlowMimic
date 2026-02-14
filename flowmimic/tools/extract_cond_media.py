"""Extract a cropped AIST++ video clip and sparse frames from flow sample metadata.

Example:
  python flowmimic/tools/extract_cond_media.py --meta output/flow/gLO_sBM_cAll_d13_mLO3_ch02_c02_meta.txt
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


def _parse_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_list(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out-dir", default="output/flow/cond_media")
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

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

    sparse_indices = meta.get("sparse_indices", [])
    if not isinstance(sparse_indices, list):
        sparse_indices = _parse_list(str(sparse_indices))

    tag = "result"
    out_dir = args.out_dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, f"{tag}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    smpl_out = os.path.join(out_dir, "cond_clip_smpl22.npy")

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

        for idx in sparse_indices:
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
        joints = yup_to_blender(joints)
        np.save(smpl_out, joints)

        print(f"Saved clip: {clip_path}")
        print(f"Saved frames: {frame_dir}")
        print(f"Saved smpl22: {smpl_out}")
        return

    if dataset != "mvh":
        raise ValueError(f"Unsupported dataset in meta: {dataset}")

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
    for idx in sparse_indices:
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
    joints = yup_to_blender(joints)
    np.save(smpl_out, joints)

    map_path = os.path.join(out_dir, f"{tag}_frame_map.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        for idx, src_frame, out_path in frame_map:
            f.write(f"{idx}\t{src_frame}\t{out_path}\n")

    print(f"Saved frames: {frame_dir}")
    print(f"Saved smpl22: {smpl_out}")


if __name__ == "__main__":
    main()
