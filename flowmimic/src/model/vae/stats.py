import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from flowmimic.src.data.dataloader import (
    load_aistpp_smpl22_30fps,
    load_mvhumannet_sequence_smpl22_30fps,
)
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.motion.process_motion import smpl_to_ik263
import torch


def compute_mean_std(dataset, out_path, eps=1e-6, desc="Computing mean/std"):
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    sum_vec = np.zeros(cont_end, dtype=np.float64)
    sum_sq = np.zeros(cont_end, dtype=np.float64)
    count = 0.0

    for sample in tqdm(dataset, desc=desc):
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


def _stats_for_sequence(args):
    kind, path, target_fps, aist_fps, mvh_fps = args
    if kind == "aist":
        joints = load_aistpp_smpl22_30fps(
            path, target_fps=target_fps, src_fps=aist_fps
        )
    elif kind == "mvh":
        joints = load_mvhumannet_sequence_smpl22_30fps(
            path, target_fps=target_fps, src_fps=mvh_fps
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")

    motion = smpl_to_ik263(joints)
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    cont = motion[:, :cont_end]
    if not np.isfinite(cont).all():
        return None
    sum_vec = cont.sum(axis=0)
    sum_sq = (cont ** 2).sum(axis=0)
    count = cont.shape[0]
    return sum_vec, sum_sq, count


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def compute_mean_std_from_splits(
    aist_paths,
    mvh_dirs,
    out_path,
    workers=10,
    eps=1e-6,
    target_fps=30,
    aist_fps=60,
    mvh_fps=5,
):
    cont_end = LAYOUT_SLICES["feet_contact"][0]
    sum_vec = np.zeros(cont_end, dtype=np.float64)
    sum_sq = np.zeros(cont_end, dtype=np.float64)
    count = 0.0

    tasks = [("aist", p, target_fps, aist_fps, mvh_fps) for p in aist_paths] + [
        ("mvh", p, target_fps, aist_fps, mvh_fps) for p in mvh_dirs
    ]
    if not tasks:
        raise ValueError("No samples provided for stats computation")

    skipped = 0
    with Pool(processes=workers, initializer=_init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(_stats_for_sequence, tasks),
            total=len(tasks),
            desc="Computing mean/std",
        ):
            if result is None:
                skipped += 1
                continue
            s_vec, s_sq, n = result
            sum_vec += s_vec
            sum_sq += s_sq
            count += n

    if skipped:
        print(f"Skipped {skipped} sequences with non-finite values")

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
