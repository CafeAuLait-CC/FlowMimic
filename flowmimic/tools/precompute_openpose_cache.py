"""Precompute cached OpenPose 2D keypoints for AIST++ and MVHumanNet splits.

Example:
  python flowmimic/tools/precompute_openpose_cache.py --workers 10 --overwrite
"""

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
from flowmimic.src.data.openpose import load_aist_openpose, load_openpose_npy


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _aist_split_paths(aist_dir, split_path):
    names = _read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def _cache_aist(args):
    pkl_path, openpose_dir, cache_root, overwrite, target_fps, aist_fps = args
    name = os.path.splitext(os.path.basename(pkl_path))[0]
    missing = False
    for cam in ["01", "02", "08", "09"]:
        out_path = os.path.join(cache_root, "aist", f"{name}_c{cam}.npz")
        if os.path.exists(out_path) and not overwrite:
            continue
        name_cam = name.replace("_cAll_", f"_c{cam}_")
        in_path = os.path.join(openpose_dir, f"{name_cam}.npy")
        if not os.path.exists(in_path):
            missing = True
            continue
        coords, vis = load_openpose_npy(in_path, src_fps=aist_fps, target_fps=target_fps)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, coords=coords.astype(np.float32), vis=vis.astype(np.float32))
    if missing:
        return ("aist", pkl_path)


def _cache_mvh(args):
    seq_dir, mv_root, openpose_root, cameras, cache_root, overwrite, target_fps, mvh_fps = args
    rel = os.path.relpath(seq_dir, mv_root)
    missing = False
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return ("mvh", seq_dir)
    part, motion = parts[0], parts[1]
    for cam in cameras:
        out_path = os.path.join(cache_root, "mvh", rel, "openpose", f"{cam}.npz")
        if os.path.exists(out_path) and not overwrite:
            continue
        in_path = os.path.join(openpose_root, part, motion, f"{cam}_2d_body25.npy")
        if not os.path.exists(in_path):
            missing = True
            continue
        coords, vis = load_openpose_npy(in_path, src_fps=mvh_fps, target_fps=target_fps)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, coords=coords.astype(np.float32), vis=vis.astype(np.float32))
    if missing:
        return ("mvh", seq_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only", type=str, choices=["aist", "mvh", "all"], default="all"
    )
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    openpose_aist_dir = config.get("aist_openpose_dir", "data/AIST++/Annotations/openpose")
    openpose_mvh_root = config.get("mvh_openpose_root", "data/MVHumanNet")
    mvh_cameras = config.get("mvh_cameras", ["22327091", "22327113", "22327084"])
    cache_root = config.get("cond_cache_root", "data/cached_cond")
    aist_split_train = config["aist_split_train"]
    aist_split_val = config["aist_split_val"]
    mvh_split_train = config["mvh_split_train"]
    mvh_split_val = config["mvh_split_val"]
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)

    aist_paths = _aist_split_paths(aist_dir, aist_split_train) + _aist_split_paths(
        aist_dir, aist_split_val
    )
    mvh_dirs = _read_lines(mvh_split_train) + _read_lines(mvh_split_val)

    tasks_aist = [
        (p, openpose_aist_dir, cache_root, args.overwrite, target_fps, aist_fps)
        for p in aist_paths
    ]
    tasks_mvh = [
        (
            p,
            mv_root,
            openpose_mvh_root,
            mvh_cameras,
            cache_root,
            args.overwrite,
            target_fps,
            mvh_fps,
        )
        for p in mvh_dirs
    ]

    bad_aist = []
    if args.only in ("aist", "all"):
        with Pool(processes=args.workers, initializer=_init_worker) as pool:
            for result in tqdm(
                pool.imap_unordered(_cache_aist, tasks_aist),
                total=len(tasks_aist),
                desc="Cache OpenPose AIST",
            ):
                if result:
                    bad_aist.append(result[1])

    bad_mvh = []
    if args.only in ("mvh", "all"):
        with Pool(processes=args.workers, initializer=_init_worker) as pool:
            for result in tqdm(
                pool.imap_unordered(_cache_mvh, tasks_mvh),
                total=len(tasks_mvh),
                desc="Cache OpenPose MVH",
            ):
                if result:
                    bad_mvh.append(result[1])

    if bad_aist:
        os.makedirs(cache_root, exist_ok=True)
        with open(os.path.join(cache_root, "bad_openpose_aist.txt"), "w", encoding="utf-8") as f:
            for item in bad_aist:
                f.write(f"{item}\n")
    if bad_mvh:
        os.makedirs(cache_root, exist_ok=True)
        with open(os.path.join(cache_root, "bad_openpose_mvh.txt"), "w", encoding="utf-8") as f:
            for item in bad_mvh:
                f.write(f"{item}\n")


if __name__ == "__main__":
    main()
