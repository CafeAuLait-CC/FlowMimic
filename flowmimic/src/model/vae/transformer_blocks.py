import torch
from torch import nn

from flowmimic.src.model.vae.adaln import AdaLayerNorm


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln1 = AdaLayerNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = AdaLayerNorm(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond, key_padding_mask=None):
        h = self.ln1(x, cond)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        h = self.ln2(x, cond)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x
