import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.vae.stats import compute_mean_std_from_splits
from utils.config import load_config


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def aist_split_paths(aist_dir, split_path):
    names = read_lines(split_path)
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    aist_split_train = config["aist_split_train"]
    mvh_split_train = config["mvh_split_train"]
    out_path = args.out or config["stats_path"]

    if not os.path.exists(aist_split_train):
        raise FileNotFoundError(f"AIST split file not found: {aist_split_train}")
    if not os.path.exists(mvh_split_train):
        raise FileNotFoundError(f"MVHumanNet split file not found: {mvh_split_train}")

    aist_train_paths = aist_split_paths(aist_dir, aist_split_train)
    mvh_train_dirs = read_lines(mvh_split_train)

    compute_mean_std_from_splits(
        aist_train_paths,
        mvh_train_dirs,
        out_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
