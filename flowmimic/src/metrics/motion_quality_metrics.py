"""Small physical-motion metrics shared by evaluation entrypoints."""

from __future__ import annotations

import numpy as np


def aggregate_replications(rows: list[dict]) -> dict:
    """Aggregate numeric replication metrics using FlowMimic's convention."""
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"replication", "seed"}
            and isinstance(value, (int, float))
        }
    )
    output = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[key] = float(values.mean())
        if values.size > 1:
            std = float(values.std(ddof=1))
            output[f"{key}_std"] = std
            output[f"{key}_conf"] = float(1.96 * std / np.sqrt(values.size))
    return output


def motion_smoothness(joints: np.ndarray, fps: float) -> dict[str, float]:
    """Return acceleration and jerk summaries for one SMPL22 sequence."""
    velocity = np.diff(joints, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    acceleration_norm = np.linalg.norm(acceleration, axis=-1).reshape(-1)
    jerk_norm = np.linalg.norm(jerk, axis=-1).reshape(-1)
    return {
        "accel_median": float(np.median(acceleration_norm)),
        "accel_p90": float(np.percentile(acceleration_norm, 90)),
        "jerk_median": float(np.median(jerk_norm)),
        "jerk_p90": float(np.percentile(jerk_norm, 90)),
    }


def foot_skate(
    joints: np.ndarray,
    contact_values: np.ndarray,
    fps: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Measure foot speed on frames selected by generated contact channels."""
    contact_prob = 1.0 / (1.0 + np.exp(-np.clip(contact_values, -30.0, 30.0)))
    left_contact = np.maximum(contact_prob[..., 0], contact_prob[..., 1]) > threshold
    right_contact = np.maximum(contact_prob[..., 2], contact_prob[..., 3]) > threshold
    velocity = np.linalg.norm(np.diff(joints, axis=0), axis=-1) * fps
    left_values = velocity[:, 7][left_contact[: velocity.shape[0]]]
    right_values = velocity[:, 8][right_contact[: velocity.shape[0]]]
    return {
        "skate_left": float(left_values.mean()) if left_values.size else 0.0,
        "skate_right": float(right_values.mean()) if right_values.size else 0.0,
    }


def summarize_physical_motion(
    generated_joints: np.ndarray,
    reference_joints: np.ndarray,
    generated_contacts: np.ndarray,
    fps: float,
) -> dict[str, float]:
    """Average per-sequence physical metrics using eval_flow.py semantics."""
    generated_rows = []
    reference_rows = []
    skate_rows = []
    for generated, reference, contacts in zip(
        generated_joints, reference_joints, generated_contacts
    ):
        generated_rows.append(motion_smoothness(generated, fps))
        reference_rows.append(motion_smoothness(reference, fps))
        skate_rows.append(foot_skate(generated, contacts, fps))

    output = {}
    for key in generated_rows[0]:
        output[key] = float(np.mean([row[key] for row in generated_rows]))
        output[f"{key}_ref"] = float(np.mean([row[key] for row in reference_rows]))
    output["skate_left"] = float(np.mean([row["skate_left"] for row in skate_rows]))
    output["skate_right"] = float(np.mean([row["skate_right"] for row in skate_rows]))
    output["skate_mean"] = 0.5 * (output["skate_left"] + output["skate_right"])
    return output
