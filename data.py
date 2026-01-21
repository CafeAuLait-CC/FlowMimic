import json

import numpy as np

from utils.smpl2joints import joints_from_smpl_param


def load_aistpp_raw_joints(pkl_path):
    data = np.load(pkl_path, allow_pickle=True)

    joints3d = joints_from_smpl_param(data)
    return joints3d


def load_aistpp_smpl22(pkl_path):
    joints3d = load_aistpp_raw_joints(pkl_path)
    return joints3d[:, :22]


def load_mvhumannet_raw_joints(pkl_path):
    data = np.load(pkl_path, allow_pickle=True)

    joints = data["joints"]
    if joints.ndim == 3 and joints.shape[0] == 1:
        joints = joints[0]
    return joints


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
