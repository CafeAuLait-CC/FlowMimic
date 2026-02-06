#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _frame_index_from_name(p: Path) -> Optional[int]:
    """
    OpenPose json files are commonly like: <prefix>_000000000000_keypoints.json
    Extract the numeric frame index. If none, return None (will fallback to lexicographic sort).
    """
    m = re.search(r"_(\d+)_keypoints\.json$", p.name)
    if m:
        return int(m.group(1))
    return None


def _sorted_json_files(video_dir: Path) -> List[Path]:
    files = sorted(video_dir.glob("*.json"))
    if not files:
        return []
    # Prefer numeric frame index sorting if pattern matches
    idxs = [(_frame_index_from_name(f), f) for f in files]
    if all(i is not None for i, _ in idxs):
        return [f for _, f in sorted(idxs, key=lambda x: x[0])]
    return sorted(files, key=lambda x: x.name)


def _extract_pose_25x3(
    data: dict, person_index: int = 0, fill: float = 0.0
) -> np.ndarray:
    """
    Return pose keypoints in shape (25, 3) for a given person.
    If no person or missing field, return fill array.
    """
    people = data.get("people", [])
    out = np.full((25, 3), fill, dtype=np.float32)
    if not people or person_index >= len(people):
        return out

    pose = people[person_index].get("pose_keypoints_2d", [])
    if not isinstance(pose, list) or len(pose) < 25 * 3:
        return out

    arr = np.asarray(pose[: 25 * 3], dtype=np.float32).reshape(25, 3)
    return arr


def main():
    ap = argparse.ArgumentParser(
        description="Convert OpenPose per-frame JSONs to a single npy array [T,25,3]."
    )
    ap.add_argument(
        "--video_dir",
        required=True,
        type=str,
        help="Directory containing OpenPose per-frame JSON files for ONE video.",
    )
    ap.add_argument(
        "--out",
        default=None,
        type=str,
        help="Output .npy path. If omitted, saves next to video_dir as <video_dir_name>.npy",
    )
    ap.add_argument(
        "--person_index",
        default=0,
        type=int,
        help="Which person to take from `people` list (default 0).",
    )
    ap.add_argument(
        "--fill",
        default=0.0,
        type=float,
        help="Fill value for frames with missing detections (default 0.0).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, raise error when any frame has no people / missing pose_keypoints_2d.",
    )

    args = ap.parse_args()
    video_dir = Path(args.video_dir).expanduser().resolve()
    if not video_dir.is_dir():
        raise SystemExit(f"[ERROR] --video_dir is not a directory: {video_dir}")

    json_files = _sorted_json_files(video_dir)
    if not json_files:
        raise SystemExit(f"[ERROR] No .json files found in: {video_dir}")

    frames: List[np.ndarray] = []
    missing = 0

    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise SystemExit(f"[ERROR] Failed to read JSON: {jf}\n{e}")

        people = data.get("people", [])
        if args.strict and (
            not people
            or "pose_keypoints_2d"
            not in people[min(args.person_index, len(people) - 1)]
        ):
            raise SystemExit(f"[ERROR] Missing detection in frame: {jf}")

        arr = _extract_pose_25x3(data, person_index=args.person_index, fill=args.fill)
        if people == []:
            missing += 1
        frames.append(arr)

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (video_dir.parent / f"{video_dir.name}.npy")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stacked = np.stack(frames, axis=0)  # [T,25,3]
    np.save(out_path, stacked)

    print(f"[OK] video_dir: {video_dir}")
    print(f"[OK] frames: {stacked.shape[0]}, joints: {stacked.shape[1]}")
    print(f"[OK] missing_frames: {missing}")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
