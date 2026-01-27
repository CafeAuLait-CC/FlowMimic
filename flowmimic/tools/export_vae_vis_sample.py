import argparse
import os
import random
import sys

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.data.dataloader import (
    load_aistpp_smpl22,
    load_mvhumannet_sequence_smpl22,
)
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std
from flowmimic.src.motion.ik.common.quaternion import qbetween_np, qinv_np, qrot_np
from flowmimic.src.motion.process_motion import (
    _FACE_JOINT_INDX,
    ik263_to_smpl22,
    smpl_to_ik263,
)


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _pad_or_crop_features(sequence, target_len):
    length = sequence.shape[0]
    if length == target_len:
        mask = np.ones(target_len, dtype=bool)
        return sequence, mask

    if length > target_len:
        clip = sequence[:target_len]
        mask = np.ones(target_len, dtype=bool)
        return clip, mask

    pad_len = target_len - length
    pad = np.zeros((pad_len,) + sequence.shape[1:], dtype=sequence.dtype)
    clip = np.concatenate([sequence, pad], axis=0)
    mask = np.zeros(target_len, dtype=bool)
    mask[:length] = True
    return clip, mask


def _load_sample_joints(dataset, index, config):
    if dataset == "aist":
        aist_dir = config["aist_motions_dir"]
        split_path = config["aist_split_val"]
        names = _read_lines(split_path)
        if not names:
            raise FileNotFoundError(f"No entries in {split_path}")
        if index is None:
            index = random.randrange(len(names))
        name = names[index % len(names)]
        path = os.path.join(aist_dir, f"{name}.pkl")
        joints = load_aistpp_smpl22(path)
        return joints, {"path": path, "domain_id": 1, "style_id": 0}

    if dataset == "mvh":
        split_path = config["mvh_split_val"]
        seq_dirs = _read_lines(split_path)
        if not seq_dirs:
            raise FileNotFoundError(f"No entries in {split_path}")
        if index is None:
            index = random.randrange(len(seq_dirs))
        seq_dir = seq_dirs[index % len(seq_dirs)]
        joints = load_mvhumannet_sequence_smpl22(seq_dir)
        return joints, {"path": seq_dir, "domain_id": 0, "style_id": 0}

    raise ValueError("dataset must be 'aist' or 'mvh'")


def _canonical_transform(joints):
    joints = joints.astype(np.float64, copy=True)

    floor_height = joints.min(axis=0).min(axis=0)[1]
    joints[:, :, 1] -= floor_height

    root_pos_init = joints[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])

    r_hip, l_hip, sdr_r, sdr_l = _FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = (
        forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]
    )

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    return floor_height, root_pose_init_xz, root_quat_init


def _invert_canonical(joints, floor_height, root_pose_init_xz, root_quat_init):
    inv_quat = qinv_np(root_quat_init)
    inv_quat = np.ones(joints.shape[:-1] + (4,)) * inv_quat
    joints = qrot_np(inv_quat, joints)
    joints = joints + root_pose_init_xz
    joints[:, :, 1] += floor_height
    return joints


def _sample_name(meta, mv_root):
    path = meta["path"]
    if path.endswith(".pkl"):
        return os.path.splitext(os.path.basename(path))[0]
    rel = os.path.relpath(path, mv_root)
    return rel.replace(os.sep, "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", choices=["aist", "mvh"], default="aist")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="./output/")
    parser.add_argument("--space", choices=["canonical", "raw"], default="canonical")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    config = load_config()
    seq_len = args.seq_len or config["seq_len"]
    stats_path = config["stats_path"]
    d_in = config["d_in"]
    d_z = config["d_z"]
    num_styles = config["num_styles"]

    joints, meta = _load_sample_joints(args.dataset, args.index, config)
    if joints.shape[0] > seq_len:
        joints_clip = joints[:seq_len]
        ik_data = smpl_to_ik263(joints_clip)
        ik_clip = ik_data
        mask = np.ones(seq_len, dtype=bool)
    else:
        joints_clip = joints
        ik_data = smpl_to_ik263(joints)
        ik_clip, mask = _pad_or_crop_features(ik_data, seq_len)

    before_joints = ik263_to_smpl22(ik_clip)

    mean, std = load_mean_std(stats_path)
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    ik_norm = ik_clip.copy()
    ik_norm[:, :cont_end] = (ik_norm[:, :cont_end] - mean) / std

    motion = torch.from_numpy(ik_norm).float().unsqueeze(0).to(args.device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(args.device)
    domain_id = torch.tensor([meta["domain_id"]], dtype=torch.long, device=args.device)
    style_id = torch.tensor([meta["style_id"]], dtype=torch.long, device=args.device)

    model = MotionVAE(d_in=d_in, d_z=d_z, num_styles=num_styles, max_len=seq_len)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        outputs = model(motion, domain_id, style_id, mask=mask_t)
        x_hat = outputs["x_hat"].squeeze(0).cpu().numpy()

    x_hat[:, :cont_end] = x_hat[:, :cont_end] * std + mean
    after_joints = ik263_to_smpl22(x_hat)

    os.makedirs(args.out_dir, exist_ok=True)
    name = _sample_name(meta, config["mvhumannet_root"])
    before_path = os.path.join(args.out_dir, f"before_{name}.npy")
    after_path = os.path.join(args.out_dir, f"after_{name}.npy")

    if args.space == "raw":
        floor_height, root_pose_init_xz, root_quat_init = _canonical_transform(
            joints_clip
        )
        before_joints = _invert_canonical(
            before_joints, floor_height, root_pose_init_xz, root_quat_init
        )
        after_joints = _invert_canonical(
            after_joints, floor_height, root_pose_init_xz, root_quat_init
        )
    np.save(before_path, before_joints)
    np.save(after_path, after_joints)

    print(f"Saved before file to {before_path}")
    print(f"Saved after file to {after_path}")
    print(f"Sample source: {meta['path']}")


if __name__ == "__main__":
    main()
