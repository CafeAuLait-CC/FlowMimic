import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from flowmimic.src.data.dataloader import (
    blender_to_yup,
    load_mvhumannet_sequence_smpl22_30fps,
)
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.motion.process_motion import smpl_to_ik263


def _pad_or_crop(sequence, target_len):
    length = sequence.shape[0]
    if length == target_len:
        mask = np.ones(target_len, dtype=bool)
        return sequence, mask, 0, length

    if length > target_len:
        start = random.randint(0, length - target_len)
        clip = sequence[start : start + target_len]
        mask = np.ones(target_len, dtype=bool)
        return clip, mask, start, length

    pad_len = target_len - length
    pad = np.zeros((pad_len,) + sequence.shape[1:], dtype=sequence.dtype)
    clip = np.concatenate([sequence, pad], axis=0)
    mask = np.zeros(target_len, dtype=bool)
    mask[:length] = True
    return clip, mask, 0, length


class MVHumanNetDataset(Dataset):
    def __init__(
        self,
        mv_root,
        seq_len,
        mean=None,
        std=None,
        normalize=True,
        sequence_dirs=None,
        cache_root=None,
        target_fps=30,
        src_fps=5,
        camera_ids=None,
        expand_cameras=False,
        include_cond=False,
        openpose_root=None,
        cond_cache_root=None,
        cond_frames_min=None,
        cond_frames_max=None,
        cond_drop_prob=0.0,
    ):
        if sequence_dirs is None:
            self.sequence_dirs = sorted(
                glob.glob(
                    os.path.join(mv_root, "MVHumanNet_24_Part_0*", "*", "smpl_param")
                )
            )
        else:
            self.sequence_dirs = list(sequence_dirs)
        if not self.sequence_dirs:
            raise FileNotFoundError(f"No MVHumanNet smpl_param dirs found in {mv_root}")
        self.mv_root = mv_root
        self.seq_len = seq_len
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.cache_root = cache_root
        self.target_fps = target_fps
        self.src_fps = src_fps
        self.camera_ids = list(camera_ids) if camera_ids else []
        self.expand_cameras = expand_cameras
        self.include_cond = include_cond
        self.openpose_root = openpose_root
        self.cond_cache_root = cond_cache_root
        self.cond_frames_min = cond_frames_min
        self.cond_frames_max = cond_frames_max
        self.cond_drop_prob = cond_drop_prob
        self._clip_counts = None
        self._index_map = None
        self._build_index_map()

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, idx):
        entry = self._index_map[idx]
        if isinstance(entry, tuple):
            seq_idx, camera = entry
        else:
            seq_idx, camera = entry, None
        seq_dir = self.sequence_dirs[seq_idx]
        motion = None
        if self.cache_root:
            rel = os.path.relpath(seq_dir, self.mv_root)
            cache_path = os.path.join(self.cache_root, "mvh", f"{rel}.npy")
            if os.path.exists(cache_path):
                motion = np.load(cache_path)

        if motion is None:
            joints = load_mvhumannet_sequence_smpl22_30fps(
                seq_dir, target_fps=self.target_fps, src_fps=self.src_fps
            )
            joints = blender_to_yup(joints)
            motion = smpl_to_ik263(joints)
            if self.cache_root:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, motion)
        motion, mask, start, orig_len = _pad_or_crop(motion, self.seq_len)
        if not np.isfinite(motion).all():
            return self.__getitem__((idx + 1) % len(self))
        if motion.shape[-1] != 263:
            raise ValueError(f"Expected 263 features, got {motion.shape[-1]} in {seq_dir}")

        cont_end = LAYOUT_SLICES["feet_contact"][0]
        contact = motion[:, cont_end:]
        if not np.isin(contact, [0.0, 1.0]).all():
            raise ValueError(f"Contact channels are not binary in {seq_dir}")

        if self.normalize:
            if self.mean is None or self.std is None:
                raise ValueError("mean/std required for normalization")
            motion[:, :cont_end] = (motion[:, :cont_end] - self.mean) / self.std
            if not np.isfinite(motion).all():
                return self.__getitem__((idx + 1) % len(self))

        meta = {"path": seq_dir, "start": start, "orig_len": orig_len}
        if camera is not None:
            meta["camera"] = camera
        sample = {
            "motion": torch.from_numpy(motion).float(),
            "domain_id": torch.tensor(0, dtype=torch.long),
            "style_id": torch.tensor(0, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "meta": meta,
        }
        if self.include_cond:
            if not self.openpose_root:
                raise ValueError("openpose_root is required when include_cond=True")
            from flowmimic.src.data.openpose import load_mvh_openpose

            k2d, vis = load_mvh_openpose(
                seq_dir,
                self.mv_root,
                self.openpose_root,
                self.camera_ids,
                src_fps=self.src_fps,
                target_fps=self.target_fps,
                cache_root=self.cond_cache_root,
                camera=camera,
            )
            if k2d is None:
                k2d = np.zeros((self.seq_len, 25, 2), dtype=np.float32)
                vis = np.zeros((self.seq_len, 25), dtype=np.float32)
            if orig_len >= self.seq_len:
                k2d = k2d[start : start + self.seq_len]
                vis = vis[start : start + self.seq_len]
            else:
                pad_len = self.seq_len - orig_len
                k2d = np.concatenate(
                    [k2d, np.zeros((pad_len, 25, 2), dtype=np.float32)], axis=0
                )
                vis = np.concatenate(
                    [vis, np.zeros((pad_len, 25), dtype=np.float32)], axis=0
                )
            t_len = k2d.shape[0]
            k_frames = self.cond_frames_min or 1
            if t_len <= k_frames:
                idxs = np.arange(t_len)
            else:
                idxs = np.linspace(0, t_len - 1, k_frames)
                idxs = np.unique(np.round(idxs).astype(int))
            k2d_sparse = k2d[idxs]
            vis_sparse = vis[idxs]
            valid_len = k2d_sparse.shape[0]
            if self.cond_drop_prob > 0:
                drop = np.random.rand(*vis_sparse.shape) < self.cond_drop_prob
                vis_sparse = vis_sparse * (~drop)
                k2d_sparse = k2d_sparse * vis_sparse[..., None]
            pad = k_frames - valid_len
            if pad > 0:
                k2d_sparse = np.concatenate(
                    [k2d_sparse, np.zeros((pad, 25, 2), dtype=np.float32)], axis=0
                )
                vis_sparse = np.concatenate(
                    [vis_sparse, np.zeros((pad, 25), dtype=np.float32)], axis=0
                )
                mask_cond = np.concatenate(
                    [np.ones(valid_len, dtype=bool), np.zeros(pad, dtype=bool)]
                )
                tau_cond = np.concatenate(
                    [
                        idxs.astype(np.float32) / max(t_len - 1, 1),
                        np.zeros(pad, dtype=np.float32),
                    ]
                )
            else:
                mask_cond = np.ones((k2d_sparse.shape[0],), dtype=bool)
                tau_cond = idxs.astype(np.float32) / max(t_len - 1, 1)
            sample["k2d"] = torch.from_numpy(k2d_sparse).float()
            sample["vis"] = torch.from_numpy(vis_sparse).float()
            sample["tau_cond"] = torch.from_numpy(tau_cond).float()
            sample["mask_cond"] = torch.from_numpy(mask_cond)
        return sample

    def _build_index_map(self):
        clip_counts = []
        index_map = []
        cams = self.camera_ids if self.expand_cameras and self.camera_ids else None
        for i, seq_dir in enumerate(self.sequence_dirs):
            length = self._sequence_length(seq_dir)
            clips = max(1, length // self.seq_len)
            clip_counts.append(clips)
            if cams is None:
                index_map.extend([i] * clips)
            else:
                for _ in range(clips):
                    for cam in cams:
                        index_map.append((i, cam))
        self._clip_counts = clip_counts
        self._index_map = index_map

    def _sequence_length(self, seq_dir):
        if self.cache_root:
            rel = os.path.relpath(seq_dir, self.mv_root)
            cache_path = os.path.join(self.cache_root, "mvh", f"{rel}.npy")
            if os.path.exists(cache_path):
                motion = np.load(cache_path, mmap_mode="r")
                return motion.shape[0]
        joints = load_mvhumannet_sequence_smpl22_30fps(
            seq_dir, target_fps=self.target_fps, src_fps=self.src_fps
        )
        return joints.shape[0]
