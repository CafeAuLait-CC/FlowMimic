import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.data.dataloader import load_aistpp_smpl22
from flowmimic.src.motion.ik.common.quaternion import qbetween_np, qrot_np
from flowmimic.src.motion.process_motion import (
    _FACE_JOINT_INDX,
    ik263_to_smpl22,
    smpl_to_ik263,
)


def _normalize_smpl22(positions):
    positions = positions.astype(np.float64, copy=True)

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = _FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init, positions)

    return positions


def _load_first_aist_smpl22():
    with open("flowmimic/src/config/config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    motions_dir = cfg["aist_motions_dir"]
    files = sorted(
        f for f in os.listdir(motions_dir) if f.endswith(".pkl")
    )
    if not files:
        raise FileNotFoundError(f"No motion files found in {motions_dir}")

    pkl_path = os.path.join(motions_dir, files[0])
    joints = load_aistpp_smpl22(pkl_path)
    return pkl_path, joints


def main():
    pkl_path, joints = _load_first_aist_smpl22()

    ik_data = smpl_to_ik263(joints)
    recovered = ik263_to_smpl22(ik_data)
    expected = _normalize_smpl22(joints)

    diff = recovered - expected
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    print(f"Sample file: {pkl_path}")
    print(f"Input shape: {joints.shape}")
    print(f"IK263 shape: {ik_data.shape}")
    print(f"Recovered shape: {recovered.shape}")
    print(f"Max abs diff: {max_abs:.6f}")
    print(f"Mean abs diff: {mean_abs:.6f}")
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
