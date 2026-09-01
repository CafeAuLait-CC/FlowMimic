#!/usr/bin/env python3
"""Compare MediaPipe Pose Landmarker with cached OpenPose BODY25 output."""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np


BODY25_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "mid_hip",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)

BODY25_EDGES = (
    (1, 0),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 22),
    (22, 23),
    (11, 24),
    (8, 12),
    (12, 13),
    (13, 14),
    (14, 19),
    (19, 20),
    (14, 21),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
)

# BODY25 index -> MediaPipe PoseLandmark index.
DIRECT_MAP = {
    0: 0,
    2: 12,
    3: 14,
    4: 16,
    5: 11,
    6: 13,
    7: 15,
    9: 24,
    10: 26,
    11: 28,
    12: 23,
    13: 25,
    14: 27,
    15: 5,
    16: 2,
    17: 8,
    18: 7,
    19: 31,
    21: 29,
    22: 32,
    24: 30,
}
DIRECT_BODY25 = tuple(sorted(DIRECT_MAP))
SYNTHESIZED_BODY25 = (1, 8, 20, 23)

# BODY25 small-toe offset from big toe in a heel-to-big-toe local frame.
# These pooled medians were measured over 665k+ valid foot frames from the
# official AIST++ pose_train split. Literal p90 offsets overextended the toes
# and increased held-out error, so they are not used as coordinate estimates.
SMALL_TOE_FORWARD_OFFSET = -0.052586
SMALL_TOE_OUTWARD_OFFSET = 0.339154


def _score(landmark):
    values = [
        float(value)
        for value in (landmark.visibility, landmark.presence)
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.clip(min(values) if values else 0.0, 0.0, 1.0))


def _synthesize_small_toe(source, big_index, heel_index, hip_index, other_hip_index):
    big_toe = source[big_index]
    heel = source[heel_index]
    forward = big_toe[:2] - heel[:2]
    foot_length = float(np.linalg.norm(forward))
    if not math.isfinite(foot_length) or foot_length <= 1e-6:
        return big_toe.copy()

    forward /= foot_length
    outward = np.array([-forward[1], forward[0]], dtype=np.float32)
    body_outward = source[hip_index, :2] - source[other_hip_index, :2]
    if float(np.dot(outward, body_outward)) < 0.0:
        outward *= -1.0

    small_toe = big_toe.copy()
    small_toe[:2] += foot_length * (
        SMALL_TOE_FORWARD_OFFSET * forward
        + SMALL_TOE_OUTWARD_OFFSET * outward
    )
    small_toe[2] = min(
        source[index, 2]
        for index in (big_index, heel_index, hip_index, other_hip_index)
    )
    return small_toe


def mediapipe_to_body25(landmarks, width, height):
    """Map 33 normalized MediaPipe landmarks to pixel-space BODY25."""
    source = np.zeros((33, 3), dtype=np.float32)
    for index, landmark in enumerate(landmarks):
        source[index, 0] = float(landmark.x) * width
        source[index, 1] = float(landmark.y) * height
        source[index, 2] = _score(landmark)

    output = np.zeros((25, 3), dtype=np.float32)
    for body25_index, mediapipe_index in DIRECT_MAP.items():
        output[body25_index] = source[mediapipe_index]

    for output_index, left_index, right_index in ((1, 11, 12), (8, 23, 24)):
        output[output_index, :2] = 0.5 * (
            source[left_index, :2] + source[right_index, :2]
        )
        output[output_index, 2] = min(source[left_index, 2], source[right_index, 2])

    # MediaPipe foot-index landmarks map directly to BODY25 big toes. Estimate
    # only the missing small toes from the big-toe/heel direction and scale.
    output[20] = _synthesize_small_toe(source, 31, 29, 23, 24)
    output[23] = _synthesize_small_toe(source, 32, 30, 24, 23)
    return output


def extract_mediapipe(video_path, model_path, max_frames=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    frames = []
    missing = 0
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        previous_timestamp = -1
        while max_frames is None or frame_index < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = max(
                previous_timestamp + 1,
                int(round(frame_index * 1000.0 / fps)),
            )
            result = landmarker.detect_for_video(image, timestamp_ms)
            if result.pose_landmarks:
                frames.append(
                    mediapipe_to_body25(result.pose_landmarks[0], width, height)
                )
            else:
                frames.append(np.zeros((25, 3), dtype=np.float32))
                missing += 1
            previous_timestamp = timestamp_ms
            frame_index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No video frames decoded from {video_path}")
    return np.stack(frames), {
        "fps": fps,
        "width": width,
        "height": height,
        "expected_frames": expected_frames,
        "decoded_frames": len(frames),
        "missing_pose_frames": missing,
    }


def flowmimic_normalize(body25, confidence_threshold):
    """Reproduce load_openpose_npy plus CondEncoder2D geometric normalization."""
    coords = np.asarray(body25[..., :2], dtype=np.float32).copy()
    confidence = np.asarray(body25[..., 2], dtype=np.float32).copy()
    coords[~np.isfinite(coords)] = 0.0
    confidence[~np.isfinite(confidence)] = 0.0
    confidence = np.clip(confidence, 0.0, 1.0)
    coords[..., 1] *= -1.0
    visible = confidence >= float(confidence_threshold)

    if visible[0, 8]:
        center = coords[0, 8].copy()
    elif visible[0].any():
        center = coords[0, visible[0]].mean(axis=0)
    else:
        center = np.zeros(2, dtype=np.float32)
    centered = coords - center[None, None, :]

    normalized = np.zeros_like(centered)
    frame_scale = np.ones(len(centered), dtype=np.float32)
    for frame_index in range(len(centered)):
        mask = visible[frame_index]
        if not mask.any():
            continue
        values = centered[frame_index, mask]
        extent = values.max(axis=0) - values.min(axis=0)
        frame_scale[frame_index] = max(float(extent.max()), 1e-6)
        normalized[frame_index] = centered[frame_index] / frame_scale[frame_index]
    return normalized, visible, confidence, frame_scale


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _dynamics_summary(openpose, mediapipe, common):
    output = {}
    direct = np.asarray(DIRECT_BODY25, dtype=np.int64)
    op_values = openpose[:, direct]
    mp_values = mediapipe[:, direct]
    valid = common[:, direct]
    for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        op_delta = np.diff(op_values, n=order, axis=0)
        mp_delta = np.diff(mp_values, n=order, axis=0)
        delta_valid = valid.copy()
        for _ in range(order):
            delta_valid = delta_valid[1:] & delta_valid[:-1]
        op_magnitude = np.linalg.norm(op_delta, axis=-1)[delta_valid]
        mp_magnitude = np.linalg.norm(mp_delta, axis=-1)[delta_valid]
        if len(op_magnitude) > 1 and np.std(op_magnitude) > 0.0 and np.std(mp_magnitude) > 0.0:
            correlation = float(np.corrcoef(op_magnitude, mp_magnitude)[0, 1])
        else:
            correlation = None
        op_summary = _summary(op_magnitude)
        mp_summary = _summary(mp_magnitude)
        output[name] = {
            "openpose": op_summary,
            "mediapipe": mp_summary,
            "mean_ratio_mediapipe_over_openpose": (
                mp_summary["mean"] / op_summary["mean"]
                if op_summary["mean"] not in (None, 0.0)
                else None
            ),
            "magnitude_correlation": correlation,
        }
    return output


def compare(openpose, mediapipe, confidence_threshold):
    frame_count = min(len(openpose), len(mediapipe))
    openpose = openpose[:frame_count]
    mediapipe = mediapipe[:frame_count]
    op_norm, op_vis, op_conf, op_scale = flowmimic_normalize(
        openpose, confidence_threshold
    )
    mp_norm, mp_vis, mp_conf, mp_scale = flowmimic_normalize(
        mediapipe, confidence_threshold
    )
    common = op_vis & mp_vis
    distance = np.linalg.norm(op_norm - mp_norm, axis=-1)

    per_joint = []
    for joint_index, name in enumerate(BODY25_NAMES):
        mask = common[:, joint_index]
        row = {
            "index": joint_index,
            "name": name,
            "mapping": (
                "direct" if joint_index in DIRECT_BODY25 else "synthesized"
            ),
            "openpose_coverage": float(op_vis[:, joint_index].mean()),
            "mediapipe_coverage": float(mp_vis[:, joint_index].mean()),
            **_summary(distance[mask, joint_index]),
        }
        per_joint.append(row)

    direct_mask = common[:, DIRECT_BODY25]
    synth_mask = common[:, SYNTHESIZED_BODY25]
    pelvis_error = np.linalg.norm(op_norm[:, 8] - mp_norm[:, 8], axis=-1)
    metrics = {
        "frames_compared": frame_count,
        "confidence_threshold": confidence_threshold,
        "distance_unit": "per-frame visible BODY25 bounding-box extent",
        "all_common_joints": _summary(distance[common]),
        "direct_common_joints": _summary(
            distance[:, DIRECT_BODY25][direct_mask]
        ),
        "synthesized_joints": _summary(
            distance[:, SYNTHESIZED_BODY25][synth_mask]
        ),
        "pelvis_trajectory": _summary(pelvis_error[common[:, 8]]),
        "openpose_visible_fraction": float(op_vis.mean()),
        "mediapipe_visible_fraction": float(mp_vis.mean()),
        "common_visible_fraction": float(common.mean()),
        "pixel_bbox_scale_ratio_mediapipe_over_openpose": _summary(
            mp_scale / np.maximum(op_scale, 1e-6)
        ),
        "confidence": {
            "openpose": _summary(op_conf),
            "mediapipe": _summary(mp_conf),
        },
        "temporal_dynamics": _dynamics_summary(op_norm, mp_norm, common),
    }
    return metrics, per_joint, (op_norm, op_vis, mp_norm, mp_vis)


def _draw_skeleton(ax, points, visible, color, label, linewidth=1.2):
    for start, end in BODY25_EDGES:
        if visible[start] and visible[end]:
            ax.plot(
                [points[start, 0], points[end, 0]],
                [points[start, 1], points[end, 1]],
                color=color,
                linewidth=linewidth,
                alpha=0.85,
            )
    mask = visible.astype(bool)
    ax.scatter(
        points[mask, 0],
        points[mask, 1],
        s=9,
        color=color,
        edgecolors="none",
        label=label,
        alpha=0.9,
    )


def save_overlay(video_path, openpose, mediapipe, normalized, output_path):
    frame_count = min(len(openpose), len(mediapipe))
    indices = np.linspace(0, frame_count - 1, min(6, frame_count), dtype=int)
    wanted = set(int(index) for index in indices)
    capture = cv2.VideoCapture(str(video_path))
    images = {}
    frame_index = 0
    while wanted:
        ok, bgr = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            images[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            wanted.remove(frame_index)
        frame_index += 1
    capture.release()

    op_norm, op_vis, mp_norm, mp_vis = normalized
    op_raw_vis = np.clip(openpose[..., 2], 0.0, 1.0) >= 0.4
    mp_raw_vis = np.clip(mediapipe[..., 2], 0.0, 1.0) >= 0.4
    figure, axes = plt.subplots(2, len(indices), figsize=(3.4 * len(indices), 7.0))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)
    for column, index in enumerate(indices):
        raw_ax = axes[0, column]
        raw_ax.imshow(images[int(index)])
        _draw_skeleton(
            raw_ax, openpose[index, :, :2], op_raw_vis[index], "#00a6a6", "OpenPose"
        )
        _draw_skeleton(
            raw_ax, mediapipe[index, :, :2], mp_raw_vis[index], "#d63384", "MediaPipe"
        )
        raw_ax.set_title(f"Frame {index}: pixels")
        raw_ax.axis("off")

        norm_ax = axes[1, column]
        _draw_skeleton(norm_ax, op_norm[index], op_vis[index], "#00a6a6", "OpenPose")
        _draw_skeleton(norm_ax, mp_norm[index], mp_vis[index], "#d63384", "MediaPipe")
        norm_ax.set_title("FlowMimic space")
        norm_ax.set_aspect("equal")
        norm_ax.grid(alpha=0.2)
        norm_ax.invert_yaxis()
        if column == 0:
            norm_ax.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--openpose", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    for path in (args.video, args.openpose, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mediapipe_body25, video_info = extract_mediapipe(
        args.video, args.model, max_frames=args.max_frames
    )
    openpose_body25 = np.load(args.openpose).astype(np.float32)
    metrics, per_joint, normalized = compare(
        openpose_body25, mediapipe_body25, args.confidence_threshold
    )
    metrics["video"] = str(args.video)
    metrics["openpose"] = str(args.openpose)
    metrics["mediapipe_model"] = str(args.model)
    metrics["video_info"] = video_info
    metrics["mapping_notes"] = {
        "neck": "midpoint of MediaPipe shoulders",
        "mid_hip": "midpoint of MediaPipe hips",
        "toes": (
            "MediaPipe foot-index mapped to BODY25 big toe; small toe synthesized "
            "from fixed AIST++ OpenPose geometry in the big-toe/heel local frame"
        ),
    }

    np.save(args.output_dir / "mediapipe_body25.npy", mediapipe_body25)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_joint.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_joint[0].keys())
        writer.writeheader()
        writer.writerows(per_joint)
    save_overlay(
        args.video,
        openpose_body25[: len(mediapipe_body25)],
        mediapipe_body25,
        normalized,
        args.output_dir / "overlay.png",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
