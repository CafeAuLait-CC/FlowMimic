import os

import numpy as np

from models.vae.losses import LAYOUT_SLICES


def compute_mean_std(dataset, out_path, eps=1e-6):
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    sum_vec = np.zeros(cont_end, dtype=np.float64)
    sum_sq = np.zeros(cont_end, dtype=np.float64)
    count = 0.0

    for sample in dataset:
        motion = sample["motion"].numpy()
        mask = sample["mask"].numpy()
        valid = motion[mask][:, :cont_end]
        if valid.size == 0:
            continue
        sum_vec += valid.sum(axis=0)
        sum_sq += (valid ** 2).sum(axis=0)
        count += valid.shape[0]

    if count == 0:
        raise ValueError("No valid frames found for statistics")

    mean = sum_vec / count
    var = sum_sq / count - mean ** 2
    std = np.sqrt(np.maximum(var, eps))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, mean=mean, std=std)
    return mean, std


def load_mean_std(path):
    data = np.load(path)
    return data["mean"], data["std"]
