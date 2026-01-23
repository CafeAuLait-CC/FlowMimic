import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.config import load_config


def iter_npy(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".npy"):
                yield os.path.join(dirpath, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    cache_root = args.cache_root or config["cache_root"]
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"Cache root not found: {cache_root}")

    bad = 0
    total = 0
    for path in tqdm(list(iter_npy(cache_root)), desc="Validating cache"):
        total += 1
        data = np.load(path)
        if not np.isfinite(data).all():
            bad += 1
            print(f"Non-finite values: {path}")

    print(f"Checked {total} files, bad={bad}")


if __name__ == "__main__":
    main()
