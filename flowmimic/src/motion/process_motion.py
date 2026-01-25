import numpy as np
import torch

from flowmimic.src.motion.ik.common.quaternion import (
    qbetween_np,
    qinv_np,
    qmul_np,
    qrot_np,
    quaternion_to_cont6d_np,
)
from flowmimic.src.motion.ik.common.skeleton import Skeleton
from flowmimic.src.motion.ik.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets


# Indices follow the Text2Motion 22-joint skeleton definition.
_L_IDX1 = 5
_L_IDX2 = 8
_FID_R = [8, 11]
_FID_L = [7, 10]
_FACE_JOINT_INDX = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l


def _foot_detect(positions, thres):
    velfactor = np.array([thres, thres])

    feet_l_x = (positions[1:, _FID_L, 0] - positions[:-1, _FID_L, 0]) ** 2
    feet_l_y = (positions[1:, _FID_L, 1] - positions[:-1, _FID_L, 1]) ** 2
    feet_l_z = (positions[1:, _FID_L, 2] - positions[:-1, _FID_L, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float64)

    feet_r_x = (positions[1:, _FID_R, 0] - positions[:-1, _FID_R, 0]) ** 2
    feet_r_y = (positions[1:, _FID_R, 1] - positions[:-1, _FID_R, 1]) ** 2
    feet_r_z = (positions[1:, _FID_R, 2] - positions[:-1, _FID_R, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float64)

    return feet_l, feet_r


def _get_cont6d_params(positions, n_raw_offsets, kinematic_chain, face_joint_indx):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()

    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    return cont_6d_params, r_velocity, velocity, r_rot


def _get_rifke(positions, r_rot):
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
    return positions


def smpl_to_ik263(positions, feet_thre=0.002):
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError("Expected positions with shape (N, 22, 3)")

    positions = positions.astype(np.float64, copy=True)

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = _FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init, positions)

    global_positions = positions.copy()

    feet_l, feet_r = _foot_detect(positions, feet_thre)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    cont_6d_params, r_velocity, velocity, r_rot = _get_cont6d_params(
        positions, n_raw_offsets, t2m_kinematic_chain, _FACE_JOINT_INDX
    )
    positions = _get_rifke(positions, r_rot)

    root_y = positions[:, 0, 1:2]

    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    if data.shape[0] == 0:
        return np.zeros((0, 263), dtype=np.float64)

    if data.shape[1] != 263:
        raise ValueError(f"Expected 263 features, got {data.shape[1]}")

    last_row = data[-1:]
    data = np.concatenate([data, last_row], axis=0)
    return data


def _recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = np.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = np.cumsum(r_rot_ang, axis=-1)

    r_rot_quat = np.zeros(rot_vel.shape + (4,), dtype=data.dtype)
    r_rot_quat[..., 0] = np.cos(r_rot_ang)
    r_rot_quat[..., 2] = np.sin(r_rot_ang)

    r_pos = np.zeros(rot_vel.shape + (3,), dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot_np(qinv_np(r_rot_quat), r_pos)
    r_pos = np.cumsum(r_pos, axis=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def ik263_to_smpl22(features):
    if features.ndim not in (2, 3):
        raise ValueError("Expected features with shape (N, 263) or (B, N, 263)")
    if features.shape[-1] != 263:
        raise ValueError(f"Expected 263 features, got {features.shape[-1]}")

    single = features.ndim == 2
    data = features[None, ...] if single else features

    r_rot_quat, r_pos = _recover_root_rot_pos(data)

    joints_num = 22
    start = 4
    end = (joints_num - 1) * 3 + start
    positions = data[..., start:end]
    positions = positions.reshape(positions.shape[:-1] + (joints_num - 1, 3))

    r_rot_rep = np.repeat(r_rot_quat[..., None, :], positions.shape[-2], axis=-2)
    positions = qrot_np(qinv_np(r_rot_rep), positions)

    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = np.concatenate([r_pos[..., None, :], positions], axis=-2)

    if single:
        return positions[0]
    return positions
