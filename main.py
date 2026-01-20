import glob
import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import numpy as np

from tools.aistpp2body25 import pose_from_aist


def load_body25_mapping(def_path):
    mapping = {}
    names = {}
    computed = []

    with open(def_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for joint in cfg.get("smpl_joints", []):
        smpl_idx = joint.get("smpl_idx")
        body_idx = joint.get("body25_idx")
        smpl_name = joint.get("name")
        if body_idx is None:
            continue
        mapping[body_idx] = smpl_idx
        names[body_idx] = smpl_name

    for rule in cfg.get("computed_body25", []):
        body_idx = rule.get("body25_idx")
        smpl_indices = rule.get("smpl_indices", [])
        op = rule.get("op")
        name = rule.get("name", f"body25_{body_idx}")
        if body_idx is None or not smpl_indices or op != "midpoint":
            continue
        computed.append((body_idx, smpl_indices, name))
        names[body_idx] = name

    if len(mapping) != 25:
        expected = 25 - len(computed)
        if len(mapping) != expected:
            raise ValueError(
                f"Expected {expected} BODY-25 joints from direct mapping, got {len(mapping)}"
            )

    return mapping, names, computed


def build_body25(joints3d, mapping, computed):
    num_frames = joints3d.shape[0]
    body25 = np.zeros((num_frames, 25, 3), dtype=joints3d.dtype)
    for body_idx, smpl_idx in mapping.items():
        body25[:, body_idx] = joints3d[:, smpl_idx]

    for body_idx, smpl_indices, _name in computed:
        body25[:, body_idx] = joints3d[:, smpl_indices].mean(axis=1)

    return body25


def _stats_for_file(args):
    pkl_path, mapping, computed = args
    data = np.load(pkl_path, allow_pickle=True)
    if hasattr(data, "item"):
        data = data.item()

    joints3d = pose_from_aist(data)
    body25 = build_body25(joints3d, mapping, computed)

    pelvis_idx = 8
    left_heel_idx = 21
    right_heel_idx = 24
    nose_idx = 0

    pelvis = body25[:, pelvis_idx]
    left_heel = body25[:, left_heel_idx]
    right_heel = body25[:, right_heel_idx]
    nose = body25[:, nose_idx]

    sum_pelvis = pelvis.sum(axis=0)
    sum_left_heel = left_heel.sum(axis=0)
    sum_right_heel = right_heel.sum(axis=0)

    height_left = np.linalg.norm(nose - left_heel, axis=1)
    height_right = np.linalg.norm(nose - right_heel, axis=1)
    height = 0.5 * (height_left + height_right)
    sum_height = height.sum()

    mins = body25.min(axis=1)
    maxs = body25.max(axis=1)
    sum_width = (maxs[:, 0] - mins[:, 0]).sum()
    sum_y = (maxs[:, 1] - mins[:, 1]).sum()
    sum_depth = (maxs[:, 2] - mins[:, 2]).sum()

    total_frames = body25.shape[0]
    return (
        sum_pelvis,
        sum_left_heel,
        sum_right_heel,
        sum_height,
        sum_width,
        sum_y,
        sum_depth,
        total_frames,
    )


def get_aist_stats(motions_dir, def_path, out_path):
    motion_files = sorted(glob.glob(os.path.join(motions_dir, "*.pkl")))
    if not motion_files:
        raise FileNotFoundError(f"No motion files found in {motions_dir}")

    mapping, _names, computed = load_body25_mapping(def_path)

    sum_pelvis = np.zeros(3, dtype=np.float64)
    sum_left_heel = np.zeros(3, dtype=np.float64)
    sum_right_heel = np.zeros(3, dtype=np.float64)
    sum_height = 0.0
    sum_width = 0.0
    sum_y = 0.0
    sum_depth = 0.0
    total_frames = 0

    jobs = 20  # min(cpu_count(), len(motion_files))
    with Pool(processes=jobs) as pool:
        args = [(p, mapping, computed) for p in motion_files]
        for result in tqdm(
            pool.imap_unordered(_stats_for_file, args),
            total=len(motion_files),
            desc="Processing files",
        ):
            (
                file_sum_pelvis,
                file_sum_left_heel,
                file_sum_right_heel,
                file_sum_height,
                file_sum_width,
                file_sum_y,
                file_sum_depth,
                file_total_frames,
            ) = result

            sum_pelvis += file_sum_pelvis
            sum_left_heel += file_sum_left_heel
            sum_right_heel += file_sum_right_heel
            sum_height += file_sum_height
            sum_width += file_sum_width
            sum_y += file_sum_y
            sum_depth += file_sum_depth
            total_frames += file_total_frames

    if total_frames == 0:
        raise ValueError("No frames found across the dataset")

    avg_pelvis = sum_pelvis / total_frames
    avg_left_heel = sum_left_heel / total_frames
    avg_right_heel = sum_right_heel / total_frames
    avg_height = sum_height / total_frames
    avg_width = sum_width / total_frames
    avg_y = sum_y / total_frames
    avg_depth = sum_depth / total_frames

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Frames: {total_frames}\n")
        f.write(f"Avg pelvis: {avg_pelvis}\n")
        f.write(f"Avg left_heel: {avg_left_heel}\n")
        f.write(f"Avg right_heel: {avg_right_heel}\n")
        f.write(f"Avg height (nose-to-heel): {avg_height:.6f}\n")
        f.write(f"Avg width (X_max - X_min): {avg_width:.6f}\n")
        f.write(f"Avg height span (Y_max - Y_min): {avg_y:.6f}\n")
        f.write(f"Avg depth (Z_max - Z_min): {avg_depth:.6f}\n")


def main():
    motions_dir = os.path.join("data", "AIST++", "Annotations", "motions")
    def_path = os.path.join("config", "def_aist2body25.json")
    out_path = "stats_aist.txt"
    get_aist_stats(motions_dir, def_path, out_path)


if __name__ == "__main__":
    main()
