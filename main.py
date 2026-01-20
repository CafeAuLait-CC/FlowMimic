import glob
import os
import json

import numpy as np

from tools.aistpp2body25 import pose_from_aist


def load_body25_mapping(def_path):
    mapping = {}
    names = {}

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

    if len(mapping) != 25:
        raise ValueError(f"Expected 25 BODY-25 joints, got {len(mapping)}")

    return mapping, names


def main():
    motions_dir = os.path.join("data", "AIST++", "Annotations", "motions")
    motion_files = sorted(glob.glob(os.path.join(motions_dir, "*.pkl")))
    if not motion_files:
        raise FileNotFoundError(f"No motion files found in {motions_dir}")

    first_file = motion_files[0]
    data = np.load(first_file, allow_pickle=True)
    if hasattr(data, "item"):
        data = data.item()

    joints3d = pose_from_aist(data)
    first_frame = joints3d[0]

    def_path = os.path.join("config", "def_aist2body25.json")
    mapping, names = load_body25_mapping(def_path)

    body25 = np.zeros((25, 3), dtype=first_frame.dtype)
    for body_idx in range(25):
        smpl_idx = mapping[body_idx]
        body25[body_idx] = first_frame[smpl_idx]

    for body_idx in range(25):
        name = names[body_idx]
        x, y, z = body25[body_idx].tolist()
        print(f"{body_idx:02d} {name}: {x:.6f} {y:.6f} {z:.6f}")


if __name__ == "__main__":
    main()
