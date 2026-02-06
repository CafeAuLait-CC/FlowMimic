import torch
from torch import nn


class StyleEmbedding(nn.Module):
    def __init__(self, num_styles, dim=32, p_drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(num_styles, dim)
        self.p_drop = p_drop

    def forward(self, style_id, domain_id, apply_dropout=True):
        if apply_dropout and self.p_drop > 0:
            drop_mask = (domain_id == 1) & (torch.rand_like(style_id.float()) < self.p_drop)
            style_id = style_id.clone()
            style_id[drop_mask] = 0
        return self.emb(style_id)
