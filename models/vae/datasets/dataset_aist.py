import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from models.vae.datasets.aist_filename_parser import get_genre_code
from common.dataloader import load_aistpp_smpl22


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


class AISTDataset(Dataset):
    def __init__(self, aist_dir, genre_to_id, seq_len):
        self.files = sorted(glob.glob(os.path.join(aist_dir, "*.pkl")))
        if not self.files:
            raise FileNotFoundError(f"No AIST++ files found in {aist_dir}")
        self.genre_to_id = genre_to_id
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        joints = load_aistpp_smpl22(path)
        motion, mask = _pad_or_crop(joints, self.seq_len)
        motion = motion.reshape(motion.shape[0], -1)

        genre = get_genre_code(path)
        style_id = self.genre_to_id.get(genre, 0)
        sample = {
            "motion": torch.from_numpy(motion).float(),
            "domain_id": torch.tensor(1, dtype=torch.long),
            "style_id": torch.tensor(style_id, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "meta": {"path": path, "genre": genre},
        }
        return sample
