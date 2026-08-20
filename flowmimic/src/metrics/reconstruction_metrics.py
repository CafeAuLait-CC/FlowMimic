"""Paired physical metrics for motion autoencoder reconstructions."""

from __future__ import annotations

import numpy as np


DISTAL_JOINTS = (10, 11, 20, 21)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-8))


def physical_reconstruction_metrics(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    fps: float,
    fast_quantile: float,
) -> dict[str, float | int]:
    diff = reconstructed - reference
    joint_l2 = np.linalg.norm(diff, axis=-1)
    distal_l2 = joint_l2[..., list(DISTAL_JOINTS)]
    reconstructed_root_relative = reconstructed - reconstructed[:, :, :1]
    reference_root_relative = reference - reference[:, :, :1]
    root_relative_l2 = np.linalg.norm(
        reconstructed_root_relative - reference_root_relative, axis=-1
    )
    root_relative_distal_l2 = root_relative_l2[..., list(DISTAL_JOINTS)]

    reconstructed_velocity = np.diff(reconstructed, axis=1) * fps
    reference_velocity = np.diff(reference, axis=1) * fps
    velocity_error = np.linalg.norm(
        reconstructed_velocity - reference_velocity, axis=-1
    )
    root_velocity_error = velocity_error[..., 0]
    reconstructed_distal_velocity = reconstructed_velocity[
        ..., list(DISTAL_JOINTS), :
    ]
    reference_distal_velocity = reference_velocity[..., list(DISTAL_JOINTS), :]
    distal_velocity_error = np.linalg.norm(
        reconstructed_distal_velocity - reference_distal_velocity, axis=-1
    )
    reconstructed_distal_speed = np.linalg.norm(
        reconstructed_distal_velocity, axis=-1
    )
    reference_distal_speed = np.linalg.norm(reference_distal_velocity, axis=-1)
    fast_threshold = float(np.quantile(reference_distal_speed, fast_quantile))
    fast_events = reference_distal_speed >= fast_threshold

    per_clip_fast_score = np.quantile(reference_distal_speed, 0.9, axis=(1, 2))
    fast_clip_threshold = float(np.quantile(per_clip_fast_score, fast_quantile))
    fast_clips = per_clip_fast_score >= fast_clip_threshold

    reconstructed_root_xz = reconstructed[:, :, 0][:, :, [0, 2]]
    reference_root_xz = reference[:, :, 0][:, :, [0, 2]]
    reconstructed_root_drift = reconstructed_root_xz[:, -1] - reconstructed_root_xz[:, 0]
    reference_root_drift = reference_root_xz[:, -1] - reference_root_xz[:, 0]
    root_drift_l2 = np.linalg.norm(
        reconstructed_root_drift - reference_root_drift, axis=-1
    )
    reconstructed_root_path = np.linalg.norm(
        np.diff(reconstructed_root_xz, axis=1), axis=-1
    ).sum(axis=1)
    reference_root_path = np.linalg.norm(
        np.diff(reference_root_xz, axis=1), axis=-1
    ).sum(axis=1)

    reconstructed_accel = np.diff(reconstructed_velocity, axis=1) * fps
    reference_accel = np.diff(reference_velocity, axis=1) * fps
    reconstructed_jerk = np.diff(reconstructed_accel, axis=1) * fps
    reference_jerk = np.diff(reference_accel, axis=1) * fps
    reconstructed_accel_norm = np.linalg.norm(reconstructed_accel, axis=-1)
    reference_accel_norm = np.linalg.norm(reference_accel, axis=-1)
    reconstructed_jerk_norm = np.linalg.norm(reconstructed_jerk, axis=-1)
    reference_jerk_norm = np.linalg.norm(reference_jerk, axis=-1)
    accel_error = np.linalg.norm(reconstructed_accel - reference_accel, axis=-1)
    jerk_error = np.linalg.norm(reconstructed_jerk - reference_jerk, axis=-1)
    accel_p90 = float(np.percentile(reconstructed_accel_norm, 90))
    accel_p90_ref = float(np.percentile(reference_accel_norm, 90))
    jerk_p90 = float(np.percentile(reconstructed_jerk_norm, 90))
    jerk_p90_ref = float(np.percentile(reference_jerk_norm, 90))

    reconstructed_spectrum = np.abs(
        np.fft.rfft(reconstructed_distal_velocity, axis=1)
    )
    reference_spectrum = np.abs(np.fft.rfft(reference_distal_velocity, axis=1))
    spectrum_error = np.abs(reconstructed_spectrum - reference_spectrum)
    frequencies = np.fft.rfftfreq(reconstructed_distal_velocity.shape[1], d=1.0 / fps)
    high_frequency_mask = frequencies >= 3.0
    reconstructed_high_frequency_energy = np.square(
        reconstructed_spectrum[:, high_frequency_mask]
    ).sum()
    reference_high_frequency_energy = np.square(
        reference_spectrum[:, high_frequency_mask]
    ).sum()

    return {
        "joint_l2": float(joint_l2.mean()),
        "distal_l2": float(distal_l2.mean()),
        "root_relative_joint_l2": float(root_relative_l2.mean()),
        "root_relative_distal_l2": float(root_relative_distal_l2.mean()),
        "joint_velocity_l2": float(velocity_error.mean()),
        "root_velocity_l2": float(root_velocity_error.mean()),
        "distal_velocity_l2": float(distal_velocity_error.mean()),
        "root_endpoint_drift_l2": float(root_drift_l2.mean()),
        "root_path_ratio": _safe_ratio(
            reconstructed_root_path.sum(), reference_root_path.sum()
        ),
        "distal_speed_ratio": _safe_ratio(
            reconstructed_distal_speed.sum(), reference_distal_speed.sum()
        ),
        "fast_event_quantile": float(fast_quantile),
        "fast_event_speed_threshold": fast_threshold,
        "fast_event_count": int(fast_events.sum()),
        "fast_event_distal_speed_ratio": _safe_ratio(
            reconstructed_distal_speed[fast_events].sum(),
            reference_distal_speed[fast_events].sum(),
        ),
        "fast_event_distal_velocity_l2": float(
            distal_velocity_error[fast_events].mean()
        ),
        "fast_clip_count": int(fast_clips.sum()),
        "fast_clip_distal_speed_ratio": _safe_ratio(
            reconstructed_distal_speed[fast_clips].sum(),
            reference_distal_speed[fast_clips].sum(),
        ),
        "fast_clip_distal_velocity_l2": float(
            distal_velocity_error[fast_clips].mean()
        ),
        "accel_l2": float(accel_error.mean()),
        "jerk_l2": float(jerk_error.mean()),
        "accel_p90": accel_p90,
        "accel_p90_ref": accel_p90_ref,
        "accel_p90_ratio": _safe_ratio(accel_p90, accel_p90_ref),
        "jerk_p90": jerk_p90,
        "jerk_p90_ref": jerk_p90_ref,
        "jerk_p90_ratio": _safe_ratio(jerk_p90, jerk_p90_ref),
        "distal_velocity_spectrum_l1_relative": _safe_ratio(
            spectrum_error.sum(), reference_spectrum.sum()
        ),
        "distal_velocity_high_frequency_hz": 3.0,
        "distal_velocity_high_frequency_spectrum_l1_relative": _safe_ratio(
            spectrum_error[:, high_frequency_mask].sum(),
            reference_spectrum[:, high_frequency_mask].sum(),
        ),
        "distal_velocity_high_frequency_energy_ratio": _safe_ratio(
            reconstructed_high_frequency_energy,
            reference_high_frequency_energy,
        ),
    }
