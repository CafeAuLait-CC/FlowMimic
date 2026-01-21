import glob
import os

from data import load_mvhumannet_raw_joints, load_mvhumannet_smpl22
from utils.config import load_config


def main():
    paths = load_config()

    mv_root = paths["mvhumannet_root"]
    mv_files = sorted(
        glob.glob(
            os.path.join(mv_root, "MVHumanNet_24_Part_0*", "*", "smpl_param", "*.pkl")
        )
    )
    if not mv_files:
        raise FileNotFoundError(f"No MVHumanNet smpl_param files found in {mv_root}")

    first_file = mv_files[0]
    joints3d = load_mvhumannet_raw_joints(first_file)
    smpl22 = load_mvhumannet_smpl22(first_file)

    print(f"Raw joints shape: {joints3d.shape}")
    print(f"SMPL-22 shape: {smpl22.shape}")
    print(smpl22)


if __name__ == "__main__":
    main()
