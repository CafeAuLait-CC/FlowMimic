import os

import torch
from smplx import SMPL


def joints_from_smpl_param(aist_data):
    smpl_dir = os.path.join("data", "smpl_models")

    smpl_poses = aist_data["smpl_poses"]
    smpl_trans = aist_data["smpl_trans"]
    smpl_scaling = aist_data["smpl_scaling"]

    N = smpl_poses.shape[0]
    smpl = SMPL(model_path=smpl_dir, gender="MALE", batch_size=N)

    poses = torch.from_numpy(smpl_poses).float().reshape(N, 24, 3)
    trans = torch.from_numpy(smpl_trans).float()
    scale = torch.from_numpy(smpl_scaling.reshape(1, 1)).float()

    with torch.no_grad():
        out = smpl(
            global_orient=poses[:, 0:1],  # (N,1,3)
            body_pose=poses[:, 1:],  # (N,23,3)
            transl=trans,  # (N,3)
            scaling=scale,  # (1,1)
        )

    joints3d = out.joints.detach().cpu().numpy()  # (N, J, 3)
    return joints3d
