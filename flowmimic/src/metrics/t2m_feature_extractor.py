"""T2M motion feature extractor used by common text-to-motion metrics.

Source reference:
- motion-latent-diffusion (preferred source): https://github.com/ChenFengYe/motion-latent-diffusion
  Reused/adapted from: mld/models/architectures/t2m_motionenc.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    def forward(self, inputs, lengths):
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        packed = pack_padded_sequence(
            input_embs, lengths.data.tolist(), batch_first=True, enforce_sorted=True
        )
        _, gru_last = self.gru(packed, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class T2MMotionFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_size=263,
        movement_hidden_size=512,
        movement_latent_size=512,
        motion_hidden_size=1024,
        motion_latent_size=512,
    ):
        super().__init__()
        self.movement_encoder = MovementConvEncoder(
            input_size=input_size - 4,
            hidden_size=movement_hidden_size,
            output_size=movement_latent_size,
        )
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=movement_latent_size,
            hidden_size=motion_hidden_size,
            output_size=motion_latent_size,
        )

    def load_pretrained(self, ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if "movement_encoder" in state and "motion_encoder" in state:
            move_state = state["movement_encoder"]
            motion_state = state["motion_encoder"]
        elif "state_dict" in state:
            sd = state["state_dict"]
            move_state = {
                k.replace("movement_encoder.", "", 1): v
                for k, v in sd.items()
                if k.startswith("movement_encoder.")
            }
            motion_state = {
                k.replace("motion_encoder.", "", 1): v
                for k, v in sd.items()
                if k.startswith("motion_encoder.")
            }
            if not move_state or not motion_state:
                raise KeyError(
                    "Checkpoint state_dict does not contain movement_encoder/motion_encoder keys"
                )
        else:
            raise KeyError(
                "Unsupported T2M checkpoint format: expected movement_encoder/motion_encoder"
            )
        self.movement_encoder.load_state_dict(move_state)
        self.motion_encoder.load_state_dict(motion_state)

    @torch.no_grad()
    def encode(self, motion, motion_length):
        # motion: [B, T, 263], motion_length: [B]
        motion = motion.float()
        motion_length = motion_length.long()
        sort_idx = np.argsort(motion_length.cpu().numpy())[::-1].copy()
        rank_idx = np.empty_like(sort_idx)
        rank_idx[sort_idx] = np.arange(len(sort_idx))
        sort_idx_t = torch.from_numpy(sort_idx).to(motion.device, dtype=torch.long)
        rank_idx_t = torch.from_numpy(rank_idx).to(motion.device, dtype=torch.long)

        motion_sorted = motion[sort_idx_t]
        length_sorted = motion_length[sort_idx_t]
        movement = self.movement_encoder(motion_sorted[..., :-4]).detach()
        movement_lens = torch.clamp(length_sorted // 4, min=1)
        emb_sorted = self.motion_encoder(movement, movement_lens)
        return emb_sorted[rank_idx_t]
