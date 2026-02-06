import torch
from torch import nn

from flowmimic.src.model.flow.flow_blocks import FlowBlock
from flowmimic.src.model.flow.time_embed import TimeEmbed


class FlowNet(nn.Module):
    def __init__(
        self,
        d_z=256,
        d_model=512,
        n_layers=8,
        n_heads=8,
        ffn_dim=2048,
        cond_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_z, d_model)
        self.out_proj = nn.Linear(d_model, d_z)
        self.tau_embed = TimeEmbed(d_model)
        self.t_embed = TimeEmbed(d_model)
        self.blocks = nn.ModuleList(
            [
                FlowBlock(d_model, cond_dim, n_heads, ffn_dim, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x_t, t_flow, tau_out, mem, g, mem_mask=None):
        # x_t: [B,T,Dz], t_flow: [B] or [B,T], tau_out: [T] or [B,T]
        b, t, _ = x_t.shape
        h = self.in_proj(x_t)
        if tau_out.dim() == 1:
            tau_out = tau_out.unsqueeze(0).expand(b, -1)
        h = h + self.tau_embed(tau_out)
        if t_flow.dim() == 1:
            t_flow = t_flow.unsqueeze(1).expand(b, t)
        h = h + self.t_embed(t_flow)

        for block in self.blocks:
            h = block(h, g, mem, mem_mask=mem_mask)

        return self.out_proj(h)
