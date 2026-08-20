"""Confidence-weighted held-out 2D reprojection metrics for SMPL22 motion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_smpl22_body25_mapping(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    direct = []
    for joint in config.get("smpl_joints", []):
        smpl_index = joint.get("smpl_idx")
        body_index = joint.get("body25_idx")
        if smpl_index is not None and body_index is not None and smpl_index < 22:
            direct.append((int(body_index), int(smpl_index)))
    computed = []
    for rule in config.get("computed_body25", []):
        body_index = rule.get("body25_idx")
        smpl_indices = [int(index) for index in rule.get("smpl_indices", [])]
        if (
            body_index is not None
            and smpl_indices
            and all(index < 22 for index in smpl_indices)
        ):
            computed.append((int(body_index), smpl_indices))
    return direct, computed


def smpl22_to_body25(joints: np.ndarray, direct, computed) -> np.ndarray:
    joints = np.asarray(joints)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"Expected [T,22,3] joints, got {joints.shape}")
    body = np.full((joints.shape[0], 25, 3), np.nan, dtype=joints.dtype)
    for body_index, smpl_index in direct:
        body[:, body_index] = joints[:, smpl_index]
    for body_index, smpl_indices in computed:
        body[:, body_index] = joints[:, smpl_indices].mean(axis=1)
    return body


def _fit_shared_weak_perspective(xy, keypoints, confidence, frame_mask):
    valid = (
        frame_mask[:, None]
        & np.isfinite(xy).all(axis=-1)
        & np.isfinite(keypoints).all(axis=-1)
        & np.isfinite(confidence)
        & (confidence > 0)
    )
    if valid.sum() < 2:
        return None
    source = xy[valid].astype(np.float64)
    target = keypoints[valid].astype(np.float64)
    weights = np.sqrt(np.clip(confidence[valid], 0.0, 1.0)).astype(np.float64)
    matrix = np.zeros((source.shape[0] * 2, 3), dtype=np.float64)
    values = np.zeros((source.shape[0] * 2,), dtype=np.float64)
    matrix[0::2, 0] = source[:, 0]
    matrix[0::2, 1] = 1.0
    matrix[1::2, 0] = source[:, 1]
    matrix[1::2, 2] = 1.0
    values[0::2] = target[:, 0]
    values[1::2] = target[:, 1]
    matrix *= np.repeat(weights, 2)[:, None]
    values *= np.repeat(weights, 2)
    try:
        camera, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return camera


def _frame_rpd(projected, keypoints, confidence, frame_mask):
    valid = (
        np.isfinite(projected).all(axis=-1)
        & np.isfinite(keypoints).all(axis=-1)
        & np.isfinite(confidence)
        & (confidence > 0)
    )
    weights = np.where(valid, np.clip(confidence, 0.0, 1.0), 0.0)
    differences = np.where(valid[..., None], projected - keypoints, 0.0)
    distances = np.linalg.norm(differences, axis=-1)
    weight_sum = weights.sum(axis=-1)
    selected = np.asarray(frame_mask, dtype=bool) & (weight_sum > 0)
    if not selected.any():
        return None
    frame_scores = np.sum(weights * distances, axis=-1) / np.maximum(
        weight_sum, 1e-8
    )
    return float(np.mean(frame_scores[selected]))


def summarize_reprojection_distance(
    generated_joints: np.ndarray,
    target_keypoints: np.ndarray,
    confidence: np.ndarray,
    observed_indices: np.ndarray,
    direct_mapping,
    computed_mapping,
) -> dict[str, float]:
    """Compute observed, held-out, and all-frame RPD with one shared camera.

    The weak-perspective scale and translation are fitted only on observed
    frames, then frozen while every metric subset is scored.
    """
    keypoints = np.asarray(target_keypoints, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (25, 2):
        raise ValueError(f"Expected [T,25,2] keypoints, got {keypoints.shape}")
    if confidence.shape != keypoints.shape[:2]:
        raise ValueError("Keypoint confidence shape does not match coordinates")
    body25 = smpl22_to_body25(
        np.asarray(generated_joints), direct_mapping, computed_mapping
    )
    if body25.shape[0] != keypoints.shape[0]:
        raise ValueError("Generated joints and target keypoints have different lengths")

    observed = np.zeros((body25.shape[0],), dtype=bool)
    indices = np.asarray(observed_indices, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < body25.shape[0])]
    observed[np.unique(indices)] = True
    held = ~observed
    camera = _fit_shared_weak_perspective(
        body25[..., :2], keypoints, confidence, observed
    )
    if camera is None:
        raise ValueError("Cannot fit weak-perspective camera from observed frames")
    projected = camera[0] * body25[..., :2] + camera[1:3]

    output = {
        "rpd_obs": _frame_rpd(projected, keypoints, confidence, observed),
        "rpd_all": _frame_rpd(
            projected,
            keypoints,
            confidence,
            np.ones_like(observed),
        ),
        "rpd_obs_frames": int(observed.sum()),
        "rpd_held_frames": int(held.sum()),
    }
    if held.any():
        output["rpd_held"] = _frame_rpd(projected, keypoints, confidence, held)
    return {key: value for key, value in output.items() if value is not None}
