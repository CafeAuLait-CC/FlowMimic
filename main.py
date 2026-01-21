import glob
import os

from common.dataloader import load_aistpp_smpl22, load_mvhumannet_sequence_smpl22
from utils.config import load_config


def main():
    paths = load_config()

    aist_dir = paths["aist_motions_dir"]
    mv_root = paths["mvhumannet_root"]

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

    mv_dirs = sorted(
        glob.glob(
            os.path.join(mv_root, "MVHumanNet_24_Part_0*", "*", "smpl_param")
        )
    )
    if not mv_dirs:
        raise FileNotFoundError(f"No MVHumanNet smpl_param dirs found in {mv_root}")

    mv_joints3d = load_mvhumannet_sequence_smpl22(mv_dirs[0])
    mv_first_frame = mv_joints3d[0]
    mv_last_frame = mv_joints3d[-1]
    mv_pelvis = mv_joints3d[:, 0]

    write_frame_list(mv_first_frame, "mv_first_frame.txt")
    write_frame_list(mv_last_frame, "mv_last_frame.txt")
    write_trajectory_list(mv_pelvis, "mv_pelvis_trajectory.txt")


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
