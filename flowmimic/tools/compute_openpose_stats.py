"""Compute per-joint mean/std for BODY25 2D keypoints.

Example:
  python flowmimic/tools/compute_openpose_stats.py
"""

import argparse
import os
import sys

from tqdm import tqdm
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.data.openpose import compute_openpose_stats


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _aist_split_paths(aist_dir, split_path):
    names = _read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    out_path = args.out or config.get("openpose_stats_path", "data/openpose_stats.npz")
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    aist_split_train = config["aist_split_train"]
    mvh_split_train = config["mvh_split_train"]
    aist_openpose_dir = config.get("aist_openpose_dir", "data/AIST++/Annotations/openpose")
    mvh_openpose_root = config.get("mvh_openpose_root", "data/MVHumanNet")
    mvh_cameras = config.get("mvh_cameras", ["22327091", "22327113", "22327084"])
    aist_cameras = config.get("aist_cameras", ["01", "02", "08", "09"])
    cond_cache_root = config.get("cond_cache_root", "data/cached_cond")
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)

    aist_paths = _aist_split_paths(aist_dir, aist_split_train)
    mvh_dirs = _read_lines(mvh_split_train)

    compute_openpose_stats(
        aist_paths,
        mvh_dirs,
        aist_openpose_dir,
        mvh_openpose_root,
        mv_root,
        mvh_cameras,
        target_fps,
        aist_fps,
        mvh_fps,
        out_path,
        progress=tqdm,
        cache_root=cond_cache_root,
        aist_cameras=aist_cameras,
        mvh_cameras=mvh_cameras,
    )
    tqdm.write(f"Saved OpenPose stats to {out_path}")


if __name__ == "__main__":
    main()
