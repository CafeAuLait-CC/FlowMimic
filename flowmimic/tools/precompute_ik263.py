import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.data.dataloader import (
    load_aistpp_smpl22,
    load_mvhumannet_sequence_smpl22,
)
from flowmimic.src.motion.process_motion import smpl_to_ik263


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def aist_split_paths(aist_dir, split_path):
    names = read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def _cache_aist(args):
    pkl_path, cache_root, overwrite = args
    name = os.path.splitext(os.path.basename(pkl_path))[0]
    out_path = os.path.join(cache_root, "aist", f"{name}.npy")
    if os.path.exists(out_path) and not overwrite:
        return
    joints = load_aistpp_smpl22(pkl_path)
    motion = smpl_to_ik263(joints)
    if not np.isfinite(motion).all():
        return ("aist", pkl_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, motion)


def _cache_mvh(args):
    seq_dir, mv_root, cache_root, overwrite = args
    rel = os.path.relpath(seq_dir, mv_root)
    out_path = os.path.join(cache_root, "mvh", f"{rel}.npy")
    if os.path.exists(out_path) and not overwrite:
        return
    joints = load_mvhumannet_sequence_smpl22(seq_dir)
    motion = smpl_to_ik263(joints)
    if not np.isfinite(motion).all():
        return ("mvh", seq_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, motion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    cache_root = config["cache_root"]
    aist_split_train = config["aist_split_train"]
    aist_split_val = config["aist_split_val"]
    mvh_split_train = config["mvh_split_train"]
    mvh_split_val = config["mvh_split_val"]

    aist_paths = aist_split_paths(aist_dir, aist_split_train) + aist_split_paths(
        aist_dir, aist_split_val
    )
    mvh_dirs = read_lines(mvh_split_train) + read_lines(mvh_split_val)

    tasks_aist = [(p, cache_root, args.overwrite) for p in aist_paths]
    tasks_mvh = [(p, mv_root, cache_root, args.overwrite) for p in mvh_dirs]

    bad_aist = []
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(_cache_aist, tasks_aist),
            total=len(tasks_aist),
            desc="Cache AIST",
        ):
            if result:
                bad_aist.append(result[1])

    bad_mvh = []
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(_cache_mvh, tasks_mvh),
            total=len(tasks_mvh),
            desc="Cache MVH",
        ):
            if result:
                bad_mvh.append(result[1])

    if bad_aist:
        with open(os.path.join(cache_root, "bad_aist.txt"), "w", encoding="utf-8") as f:
            for item in bad_aist:
                f.write(f"{item}\n")
    if bad_mvh:
        with open(os.path.join(cache_root, "bad_mvh.txt"), "w", encoding="utf-8") as f:
            for item in bad_mvh:
                f.write(f"{item}\n")


if __name__ == "__main__":
    main()
