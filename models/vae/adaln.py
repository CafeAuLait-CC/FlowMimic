import torch
from torch import nn


class AdaLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        h = self.norm(x)
        gamma_beta = self.cond_proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return h * (1 + gamma) + beta
