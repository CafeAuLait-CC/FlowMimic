"""Distribution metrics for motion feature embeddings.

Source reference:
- motion-latent-diffusion (preferred source): https://github.com/ChenFengYe/motion-latent-diffusion
  Reused/adapted from: mld/data/humanml/utils/metrics.py
"""

import numpy as np
from scipy import linalg


def euclidean_distance_matrix(matrix1, matrix2):
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(np.clip(d1 + d2 + d3, a_min=0.0, a_max=None))
    return dists


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]
    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    return dist


def calculate_top_k(sorted_indices, top_k):
    """Return cumulative top-k matches for a paired retrieval matrix."""
    size = sorted_indices.shape[0]
    paired = np.arange(size)[:, None]
    correct = np.zeros((size,), dtype=bool)
    columns = []
    for index in range(min(int(top_k), sorted_indices.shape[1])):
        correct = correct | (sorted_indices[:, index : index + 1] == paired).reshape(-1)
        columns.append(correct[:, None])
    while len(columns) < int(top_k):
        columns.append(correct[:, None])
    return np.concatenate(columns, axis=1)


def summarize_text_motion_metrics(
    text_features,
    motion_features,
    *,
    r_precision_batch_size=32,
    top_k=3,
    reference_motion_features=None,
):
    """Compute standard T2M matching distance and batch-wise R-precision.

    R-precision follows the HumanML3D/T2M protocol: retrieval candidates are
    restricted to deterministic contiguous batches, normally of size 32.
    """
    text_features = np.asarray(text_features)
    motion_features = np.asarray(motion_features)
    if text_features.ndim != 2 or motion_features.ndim != 2:
        raise ValueError("Text and motion features must be [N,D]")
    if text_features.shape != motion_features.shape:
        raise ValueError(
            "Text and motion features must have matching shape, got "
            f"{text_features.shape} and {motion_features.shape}"
        )
    if r_precision_batch_size < 2:
        raise ValueError("r_precision_batch_size must be at least 2")

    def _retrieval(candidate_features):
        matched_distance = calculate_matching_score(
            text_features, candidate_features
        )
        correct = np.zeros((int(top_k),), dtype=np.float64)
        count = 0
        for start in range(0, len(text_features), int(r_precision_batch_size)):
            end = min(start + int(r_precision_batch_size), len(text_features))
            text_batch = text_features[start:end]
            motion_batch = candidate_features[start:end]
            distances = euclidean_distance_matrix(text_batch, motion_batch)
            ranked = np.argsort(distances, axis=1)
            batch_top_k = calculate_top_k(ranked, top_k)
            correct[: batch_top_k.shape[1]] += batch_top_k.sum(axis=0)
            count += end - start
        return float(matched_distance.mean()), correct / max(count, 1)

    mmdist, r_precision = _retrieval(motion_features)
    output = {
        "mmdist": mmdist,
        "matching_score": mmdist,
        "r_precision_n": int(len(text_features)),
        "r_precision_batch_size": int(r_precision_batch_size),
    }
    for index, value in enumerate(r_precision, start=1):
        output[f"r_precision_top_{index}"] = float(value)

    if reference_motion_features is not None:
        reference_motion_features = np.asarray(reference_motion_features)
        if reference_motion_features.shape != text_features.shape:
            raise ValueError("Reference motion features do not match text features")
        reference_mmdist, reference_r_precision = _retrieval(
            reference_motion_features
        )
        output["mmdist_ref"] = reference_mmdist
        for index, value in enumerate(reference_r_precision, start=1):
            output[f"r_precision_ref_top_{index}"] = float(value)
    return output


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return float(dist.mean())


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_cond = activation.shape[1]
    first_indices = np.random.choice(num_per_cond, multimodality_times, replace=False)
    second_indices = np.random.choice(num_per_cond, multimodality_times, replace=False)
    dist = linalg.norm(
        activation[:, first_indices] - activation[:, second_indices], axis=2
    )
    return float(dist.mean())


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


def summarize_motion_feature_metrics(
    feats_gen,
    feats_ref,
    diversity_times=300,
    multimodality_feats=None,
    multimodality_times=20,
):
    """Compute common T2M-style distribution metrics from motion embeddings.

    Args:
        feats_gen: [N, D] generated-motion embeddings.
        feats_ref: [N, D] reference-motion embeddings, aligned with feats_gen.
        diversity_times: number of random pairs used by diversity metric.
        multimodality_feats: optional [N_cond, N_repeat, D] embeddings.
        multimodality_times: number of random pairs used for multimodality.
    """
    if feats_gen.ndim != 2 or feats_ref.ndim != 2:
        raise ValueError("feats_gen and feats_ref must be 2D arrays [N, D]")
    if feats_gen.shape != feats_ref.shape:
        raise ValueError("feats_gen and feats_ref must have matching shape")
    if feats_gen.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute distribution metrics")

    mu_gen, cov_gen = calculate_activation_statistics(feats_gen)
    mu_ref, cov_ref = calculate_activation_statistics(feats_ref)
    pmfd = float(calculate_matching_score(feats_gen, feats_ref).mean())

    metrics = {
        "fid": float(calculate_frechet_distance(mu_ref, cov_ref, mu_gen, cov_gen)),
        "pmfd": pmfd,
        # Backward-compatible aliases. Dedicated text-aware evaluators replace
        # these two fields with standard text-motion matching metrics.
        "mmdist": pmfd,
        "matching_score": pmfd,
        "fid_n": int(feats_gen.shape[0]),
    }

    if diversity_times is not None:
        max_pairs = max(1, min(int(diversity_times), feats_gen.shape[0] - 1))
        if feats_gen.shape[0] > max_pairs:
            metrics["diversity"] = float(calculate_diversity(feats_gen, max_pairs))
            metrics["diversity_ref"] = float(calculate_diversity(feats_ref, max_pairs))

    if multimodality_feats is not None and multimodality_times is not None:
        if multimodality_feats.ndim != 3:
            raise ValueError("multimodality_feats must be 3D [N_cond, N_repeat, D]")
        if multimodality_feats.shape[1] > int(multimodality_times):
            metrics["multimodality"] = float(
                calculate_multimodality(
                    multimodality_feats, int(multimodality_times)
                )
            )

    return metrics
