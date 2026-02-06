import torch
from torch import nn


class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_gamma_beta = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        h = self.norm(x)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class FlowBlock(nn.Module):
    def __init__(self, d_model, cond_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.adaln = AdaLN(d_model, cond_dim)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.Sigmoid(),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, cond, mem, mem_mask=None):
        h = self.adaln(x, cond)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=None, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        cross_out, _ = self.cross_attn(h, mem, mem, key_padding_mask=mem_mask, need_weights=False)
        gate = self.gate(cond).unsqueeze(1)
        x = x + self.dropout(cross_out * gate)

        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))
        return x
