import glob
import os

from data import load_aistpp_smpl22
from utils.config import load_config


def main():
    paths = load_config()

    aist_dir = paths["aist_motions_dir"]

    aist_files = sorted(glob.glob(os.path.join(aist_dir, "*.pkl")))
    if not aist_files:
        raise FileNotFoundError(f"No AIST++ motion files found in {aist_dir}")

    joints3d = load_aistpp_smpl22(aist_files[0])
    first_frame = joints3d[0]
    last_frame = joints3d[-1]
    pelvis = joints3d[:, 0]

    write_frame_list(first_frame, "aist_first_frame.txt")
    write_frame_list(last_frame, "aist_last_frame.txt")
    write_trajectory_list(pelvis, "pelvis_trajectory.txt")


def write_frame_list(points, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for x, y, z in points:
            f.write(f"  [{x}, {y}, {z}],\n")
        f.write("]\n")


def write_trajectory_list(points, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for x, y, z in points:
            f.write(f"  [{x}, {y}, {z}],\n")
        f.write("]\n")


if __name__ == "__main__":
    main()
