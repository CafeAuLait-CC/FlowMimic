import math

import torch
from torch import nn


class TimeEmbed(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t, batch_size=None, seq_len=None):
        t = t.float()
        if t.dim() == 0:
            t = t.view(1)
        if t.dim() == 1 and batch_size is not None and seq_len is not None:
            t = t.view(1, -1).expand(batch_size, seq_len)
        elif t.dim() == 1 and batch_size is not None:
            t = t.view(-1).expand(batch_size)

        emb = sinusoidal_embedding(t, self.dim)
        return self.mlp(emb)


def sinusoidal_embedding(t, dim):
    if t.dim() == 1:
        t = t[:, None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half, 1)
    )
    args = t * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
