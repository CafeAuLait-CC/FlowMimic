import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from common.dataloader import load_mvhumannet_sequence_smpl22
from process_motion import smpl_to_ik263
from models.vae.losses import LAYOUT_SLICES


def _pad_or_crop(sequence, target_len):
    length = sequence.shape[0]
    if length == target_len:
        mask = np.ones(target_len, dtype=bool)
        return sequence, mask

    if length > target_len:
        start = random.randint(0, length - target_len)
        clip = sequence[start : start + target_len]
        mask = np.ones(target_len, dtype=bool)
        return clip, mask

    pad_len = target_len - length
    pad = np.zeros((pad_len,) + sequence.shape[1:], dtype=sequence.dtype)
    clip = np.concatenate([sequence, pad], axis=0)
    mask = np.zeros(target_len, dtype=bool)
    mask[:length] = True
    return clip, mask


class MVHumanNetDataset(Dataset):
    def __init__(self, mv_root, seq_len, mean=None, std=None, normalize=True):
        self.sequence_dirs = sorted(
            glob.glob(
                os.path.join(mv_root, "MVHumanNet_24_Part_0*", "*", "smpl_param")
            )
        )
        if not self.sequence_dirs:
            raise FileNotFoundError(f"No MVHumanNet smpl_param dirs found in {mv_root}")
        self.seq_len = seq_len
        self.mean = mean
        self.std = std
        self.normalize = normalize

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        seq_dir = self.sequence_dirs[idx]
        joints = load_mvhumannet_sequence_smpl22(seq_dir)
        motion = smpl_to_ik263(joints)
        motion, mask = _pad_or_crop(motion, self.seq_len)
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

        sample = {
            "motion": torch.from_numpy(motion).float(),
            "domain_id": torch.tensor(0, dtype=torch.long),
            "style_id": torch.tensor(0, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "meta": {"path": seq_dir},
        }
        return sample
