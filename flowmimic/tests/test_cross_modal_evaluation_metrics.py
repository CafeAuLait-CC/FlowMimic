import numpy as np

from flowmimic.src.metrics.distribution_metrics import (
    summarize_motion_feature_metrics,
    summarize_text_motion_metrics,
)
from flowmimic.src.metrics.reprojection_metrics import (
    summarize_reprojection_distance,
)


def test_pmfd_and_text_mmdist_are_distinct_metrics():
    generated = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    reference = np.asarray([[0.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    text = np.asarray([[3.0, 0.0], [0.0, 4.0], [4.0, 4.0]])

    motion_metrics = summarize_motion_feature_metrics(
        generated, reference, diversity_times=None
    )
    text_metrics = summarize_text_motion_metrics(
        text,
        generated,
        r_precision_batch_size=3,
        reference_motion_features=reference,
    )

    assert motion_metrics["pmfd"] == motion_metrics["mmdist"]
    assert text_metrics["mmdist"] != motion_metrics["pmfd"]
    assert 0.0 <= text_metrics["r_precision_top_1"] <= 1.0
    assert text_metrics["r_precision_top_3"] == 1.0


def test_held_rpd_reuses_camera_fitted_on_observed_frames():
    frames = 4
    joints = np.zeros((frames, 22, 3), dtype=np.float32)
    joints[:, 0, 0] = np.arange(frames)
    joints[:, 1, 0] = np.arange(frames) + 1.0
    joints[:, 1, 1] = 1.0

    target = np.zeros((frames, 25, 2), dtype=np.float32)
    target[:, 0] = 2.0 * joints[:, 0, :2] + np.asarray([10.0, -4.0])
    target[:, 1] = 2.0 * joints[:, 1, :2] + np.asarray([10.0, -4.0])
    confidence = np.zeros((frames, 25), dtype=np.float32)
    confidence[:, :2] = 1.0

    # Observed frames remain exact; held frames contain a generated-motion error.
    generated = joints.copy()
    generated[1:3, :2, 0] += 3.0
    metrics = summarize_reprojection_distance(
        generated,
        target,
        confidence,
        np.asarray([0, 3]),
        direct_mapping=[(0, 0), (1, 1)],
        computed_mapping=[],
    )

    assert metrics["rpd_obs"] < 1e-5
    assert metrics["rpd_held"] > 1.0
    assert metrics["rpd_obs_frames"] == 2
    assert metrics["rpd_held_frames"] == 2
