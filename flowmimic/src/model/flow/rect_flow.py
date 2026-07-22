import torch
from torch import nn

from flowmimic.src.model.flow.cond_encoder_2d import CondEncoder2D
from flowmimic.src.model.flow.flow_net import FlowNet
from flowmimic.src.model.flow.style_embed import StyleEmbedding


class ConditionalRectFlow(nn.Module):
    def __init__(
        self,
        d_z=256,
        d_model=512,
        n_layers=8,
        n_heads=8,
        ffn_dim=2048,
        dropout=0.1,
        num_styles=1,
        style_dim=32,
        cond_dim=512,
        num_joints_2d=25,
        cond_layers=4,
        cond_heads=4,
        p_style_drop=0.5,
        relative_time_bias=False,
        relative_time_hidden_dim=32,
        latent_len=16,
        latent_slot_adapter=False,
        latent_slot_adapter_heads=8,
        latent_slot_adapter_ffn_dim=1024,
    ):
        super().__init__()
        self.cond_encoder = CondEncoder2D(
            num_joints=num_joints_2d,
            d_model=d_model,
            n_layers=cond_layers,
            n_heads=cond_heads,
            dropout=dropout,
        )
        self.style_emb = StyleEmbedding(num_styles, dim=style_dim, p_drop=p_style_drop)
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model + style_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.flow = FlowNet(
            d_z=d_z,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            cond_dim=d_model,
            dropout=dropout,
            relative_time_bias=relative_time_bias,
            relative_time_hidden_dim=relative_time_hidden_dim,
            latent_len=latent_len,
            latent_slot_adapter=latent_slot_adapter,
            latent_slot_adapter_heads=latent_slot_adapter_heads,
            latent_slot_adapter_ffn_dim=latent_slot_adapter_ffn_dim,
        )

    def encode_cond(self, k2d, tau_cond, style_id, domain_id, apply_style_dropout=True, vis_mask=None):
        g_2d, mem, vis_mask = self.cond_encoder(k2d, tau_cond, vis_mask=vis_mask)
        style = self.style_emb(style_id, domain_id, apply_dropout=apply_style_dropout)
        g = self.cond_mlp(torch.cat([g_2d, style], dim=-1))
        return g, mem, vis_mask

    def forward(
        self,
        x_t,
        t_flow,
        tau_out,
        k2d=None,
        tau_cond=None,
        style_id=None,
        domain_id=None,
        apply_style_dropout=True,
        vis_mask=None,
        mem=None,
        g=None,
        mem_mask=None,
    ):
        if mem is None or g is None:
            if any(value is None for value in (k2d, tau_cond, style_id, domain_id)):
                raise ValueError("Raw conditions or pre-encoded mem/g are required")
            g, mem, _vis = self.encode_cond(
                k2d,
                tau_cond,
                style_id,
                domain_id,
                apply_style_dropout=apply_style_dropout,
                vis_mask=vis_mask,
            )
        return self.flow(
            x_t,
            t_flow,
            tau_out,
            mem,
            g,
            mem_mask=mem_mask,
            tau_cond=tau_cond,
        )
