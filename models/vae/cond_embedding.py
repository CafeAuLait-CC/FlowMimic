import torch
from torch import nn


class CondEmbedding(nn.Module):
    def __init__(self, num_domains, num_styles, dom_dim=16, style_dim=32, cond_dim=256):
        super().__init__()
        self.domain_emb = nn.Embedding(num_domains, dom_dim)
        self.style_emb = nn.Embedding(num_styles, style_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(dom_dim + style_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, domain_id, style_id):
        dom = self.domain_emb(domain_id)
        sty = self.style_emb(style_id)
        cond = torch.cat([dom, sty], dim=-1)
        return self.cond_mlp(cond)
