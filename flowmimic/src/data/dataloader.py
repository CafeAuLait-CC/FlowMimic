import json
import os

import numpy as np

from flowmimic.src.data.smpl2joints import joints_from_smpl_param

_MVH_BASE_PELVIS = {}


def _transform_aistpp(joints3d):
    x = joints3d[..., 0]
    y = joints3d[..., 1]
    z = joints3d[..., 2]
    return np.stack([x, -z, y], axis=-1)


def _transform_mvhumannet(joints3d):
    x = joints3d[..., 0]
    y = joints3d[..., 1]
    z = joints3d[..., 2]
    return np.stack([-y, -x, -z], axis=-1)


def _load_mvhumannet_joints_noalign(pkl_path):
    data = np.load(pkl_path, allow_pickle=True)
    if hasattr(data, "item"):
        data = data.item()

    joints = data["joints"]
    if joints.ndim == 2:
        joints = joints[None, ...]
    return joints


def _get_mvhumannet_base_pelvis(smpl_param_dir):
    if smpl_param_dir in _MVH_BASE_PELVIS:
        return _MVH_BASE_PELVIS[smpl_param_dir]

    frame_files = sorted(
        f for f in os.listdir(smpl_param_dir) if f.endswith(".pkl")
    )
    if not frame_files:
        raise FileNotFoundError(f"No frame files found in {smpl_param_dir}")

    first_path = os.path.join(smpl_param_dir, frame_files[0])
    joints = _load_mvhumannet_joints_noalign(first_path)
    joints = _transform_mvhumannet(joints)
    base_pelvis = joints[0, 0].copy()
    _MVH_BASE_PELVIS[smpl_param_dir] = base_pelvis
    return base_pelvis


def load_aistpp_raw_joints(pkl_path):
    data = np.load(pkl_path, allow_pickle=True)

    joints3d = joints_from_smpl_param(data)
    joints3d = _transform_aistpp(joints3d)

    scale = float(np.array(data["smpl_scaling"]).reshape(-1)[0])
    if scale > 0:
        pelvis = joints3d[:, 0]
        delta = pelvis - pelvis[0]
        scaled_pelvis = pelvis[0] + (delta / scale)
        joints3d = joints3d - pelvis[:, None, :] + scaled_pelvis[:, None, :]

    base_pelvis = joints3d[0, 0].copy()
    return joints3d - base_pelvis


def load_aistpp_smpl22(pkl_path):
    joints3d = load_aistpp_raw_joints(pkl_path)
    return joints3d[:, :22]


def load_mvhumannet_raw_joints(pkl_path):
    joints = _load_mvhumannet_joints_noalign(pkl_path)
    joints = _transform_mvhumannet(joints)
    smpl_param_dir = os.path.dirname(pkl_path)
    base_pelvis = _get_mvhumannet_base_pelvis(smpl_param_dir)
    return joints - base_pelvis


def load_mvhumannet_smpl22(pkl_path):
    joints3d = load_mvhumannet_raw_joints(pkl_path)
    return joints3d[:, :22]


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


def build_body25_from_joints(joints3d, def_path):
    mapping, _names, computed = load_body25_mapping(def_path)
    return build_body25(joints3d, mapping, computed)


def load_body25_data(pkl_path, def_path):
    joints3d = load_aistpp_smpl22(pkl_path)
    return build_body25_from_joints(joints3d, def_path)


def load_aistpp_body25(pkl_path, def_path):
    joints3d = load_aistpp_raw_joints(pkl_path)
    return build_body25_from_joints(joints3d, def_path)


def load_mvhumannet_body25(pkl_path, def_path):
    joints3d = load_mvhumannet_raw_joints(pkl_path)
    return build_body25_from_joints(joints3d, def_path)


def load_mvhumannet_sequence_smpl22(smpl_param_dir):
    frame_files = sorted(
        f for f in os.listdir(smpl_param_dir) if f.endswith(".pkl")
    )
    if not frame_files:
        raise FileNotFoundError(f"No frame files found in {smpl_param_dir}")

    frames = []
    for name in frame_files:
        pkl_path = os.path.join(smpl_param_dir, name)
        frame = load_mvhumannet_smpl22(pkl_path)[0]
        frames.append(frame)

    return np.stack(frames, axis=0)
