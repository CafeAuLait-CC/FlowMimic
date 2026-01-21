import glob
import os

from data import load_aistpp_smpl22
from process_motion import smpl_to_ik263


def main():
    motions_dir = os.path.join("data", "AIST++", "Annotations", "motions")
    motion_files = sorted(glob.glob(os.path.join(motions_dir, "*.pkl")))
    if not motion_files:
        raise FileNotFoundError(f"No motion files found in {motions_dir}")

    first_file = motion_files[0]
    joints3d = load_aistpp_smpl22(first_file)
    print(f"SMPL joints ({joints3d.shape}):")
    for [x, y, z] in joints3d[0]:
        print(f"[{x},{y},{z}]")

    ik263 = smpl_to_ik263(joints3d)
    ik163 = ik263[:, :163]
    print(f"IK features ({ik163.shape}):")
    print(ik163)


if __name__ == "__main__":
    main()
