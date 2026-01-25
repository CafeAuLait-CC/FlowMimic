import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from flowmimic.src.data.dataloader import load_aistpp_smpl22
from flowmimic.src.model.vae.datasets.aist_filename_parser import get_genre_code
from flowmimic.src.model.vae.losses import LAYOUT_SLICES
from flowmimic.src.motion.process_motion import smpl_to_ik263


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
    def __init__(
        self,
        aist_dir,
        genre_to_id,
        seq_len,
        mean=None,
        std=None,
        normalize=True,
        files=None,
        cache_root=None,
    ):
        if files is None:
            self.files = sorted(glob.glob(os.path.join(aist_dir, "*.pkl")))
        else:
            self.files = list(files)
        if not self.files:
            raise FileNotFoundError(f"No AIST++ files found in {aist_dir}")
        self.genre_to_id = genre_to_id
        self.seq_len = seq_len
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.cache_root = cache_root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        motion = None
        if self.cache_root:
            name = os.path.splitext(os.path.basename(path))[0]
            cache_path = os.path.join(self.cache_root, "aist", f"{name}.npy")
            if os.path.exists(cache_path):
                motion = np.load(cache_path)

        if motion is None:
            joints = load_aistpp_smpl22(path)
            motion = smpl_to_ik263(joints)
            if self.cache_root:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, motion)
        motion, mask = _pad_or_crop(motion, self.seq_len)
        if not np.isfinite(motion).all():
            return self.__getitem__((idx + 1) % len(self.files))
        if motion.shape[-1] != 263:
            raise ValueError(f"Expected 263 features, got {motion.shape[-1]} in {path}")

        cont_end = LAYOUT_SLICES["feet_contact"][0]
        contact = motion[:, cont_end:]
        if not np.isin(contact, [0.0, 1.0]).all():
            raise ValueError(f"Contact channels are not binary in {path}")

        if self.normalize:
            if self.mean is None or self.std is None:
                raise ValueError("mean/std required for normalization")
            motion[:, :cont_end] = (motion[:, :cont_end] - self.mean) / self.std
            if not np.isfinite(motion).all():
                return self.__getitem__((idx + 1) % len(self.files))

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
