import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from common.dataloader import load_mvhumannet_sequence_smpl22


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
    def __init__(self, mv_root, seq_len):
        self.sequence_dirs = sorted(
            glob.glob(
                os.path.join(mv_root, "MVHumanNet_24_Part_0*", "*", "smpl_param")
            )
        )
        if not self.sequence_dirs:
            raise FileNotFoundError(f"No MVHumanNet smpl_param dirs found in {mv_root}")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        seq_dir = self.sequence_dirs[idx]
        joints = load_mvhumannet_sequence_smpl22(seq_dir)
        motion, mask = _pad_or_crop(joints, self.seq_len)
        motion = motion.reshape(motion.shape[0], -1)

        sample = {
            "motion": torch.from_numpy(motion).float(),
            "domain_id": torch.tensor(0, dtype=torch.long),
            "style_id": torch.tensor(0, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "meta": {"path": seq_dir},
        }
        return sample
