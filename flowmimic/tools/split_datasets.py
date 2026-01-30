"""Create MVHumanNet train/val splits.

Example:
  python flowmimic/tools/split_datasets.py \
    --mv-root data/MVHumanNet \
    --out-train data/MVHumanNet/mvh_train.txt \
    --out-val data/MVHumanNet/mvh_val.txt
"""

import argparse
import glob
import os
import random


def write_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mv-root", required=True)
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-val", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mv_dirs = sorted(
        glob.glob(os.path.join(args.mv_root, "MVHumanNet_24_Part_0*", "*", "smpl_param"))
    )
    if not mv_dirs:
        raise FileNotFoundError(f"No MVHumanNet smpl_param dirs found in {args.mv_root}")

    rng = random.Random(args.seed)
    rng.shuffle(mv_dirs)

    val_count = int(len(mv_dirs) * args.val_ratio)
    val = mv_dirs[:val_count]
    train = mv_dirs[val_count:]

    write_list(args.out_train, train)
    write_list(args.out_val, val)


if __name__ == "__main__":
    main()
