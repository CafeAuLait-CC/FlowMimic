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
