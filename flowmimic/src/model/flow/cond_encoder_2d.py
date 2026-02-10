import torch
from torch import nn

from flowmimic.src.model.flow.time_embed import TimeEmbed


def normalize_keypoints(k2d, vis_mask=None, pelvis_idx=8, eps=1e-6, center_mode="none"):
    # k2d: [B, T_c, J, 2]
    if vis_mask is None:
        vis_mask = torch.ones(k2d.shape[:-1], device=k2d.device, dtype=k2d.dtype)

    vis_mask = vis_mask.float()
    k = k2d.clone()
    if center_mode == "first_pelvis":
        pelvis = k[:, :1, pelvis_idx : pelvis_idx + 1, :]
        pelvis_vis = vis_mask[:, :1, pelvis_idx : pelvis_idx + 1].unsqueeze(-1)
        center = torch.where(
            pelvis_vis > 0.5,
            pelvis,
            (k[:, :1] * vis_mask[:, :1].unsqueeze(-1)).sum(dim=2, keepdim=True),
        )
        denom = vis_mask[:, :1].sum(dim=2, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        center = center / denom
        k = k - center
    elif center_mode == "per_frame":
        pelvis = k[:, :, pelvis_idx : pelvis_idx + 1, :]
        pelvis_vis = vis_mask[:, :, pelvis_idx : pelvis_idx + 1].unsqueeze(-1)
        center = torch.where(
            pelvis_vis > 0.5,
            pelvis,
            (k * vis_mask.unsqueeze(-1)).sum(dim=2, keepdim=True),
        )
        denom = vis_mask.sum(dim=2, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        center = center / denom
        k = k - center

    min_xy = torch.where(vis_mask.unsqueeze(-1) > 0.5, k, torch.full_like(k, 1e6)).amin(dim=2)
    max_xy = torch.where(vis_mask.unsqueeze(-1) > 0.5, k, torch.full_like(k, -1e6)).amax(dim=2)
    size = (max_xy - min_xy).amax(dim=-1, keepdim=True).clamp_min(eps)
    k = k / size.unsqueeze(-1)
    return k, vis_mask


class CondEncoder2D(nn.Module):
    def __init__(self, num_joints=25, d_model=256, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        token_dim = num_joints * 2 + num_joints
        self.input_proj = nn.Linear(token_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.time_embed = TimeEmbed(d_model)

    def forward(self, k2d, tau_cond, vis_mask=None, mask_cond=None, mean=None, std=None):
        # k2d: [B, T_c, J, 2], tau_cond: [B, T_c] or [T_c]
        k2d_norm, vis_mask = normalize_keypoints(
            k2d, vis_mask=vis_mask, center_mode="none"
        )
        if mean is not None and std is not None:
            mean_t = torch.as_tensor(mean, device=k2d_norm.device, dtype=k2d_norm.dtype)
            std_t = torch.as_tensor(std, device=k2d_norm.device, dtype=k2d_norm.dtype)
            k2d_norm = (k2d_norm - mean_t) / (std_t + 1e-6)
            k2d_norm = k2d_norm * vis_mask.unsqueeze(-1)
        b, t, j, _ = k2d_norm.shape
        flat = k2d_norm.reshape(b, t, j * 2)
        token = torch.cat([flat, vis_mask.reshape(b, t, j)], dim=-1)
        h = self.input_proj(token)

        if tau_cond.dim() == 1:
            tau_cond = tau_cond.unsqueeze(0).expand(b, -1)
        h = h + self.time_embed(tau_cond)

        key_padding_mask = None
        if mask_cond is not None:
            key_padding_mask = ~mask_cond
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        if mask_cond is None:
            g = h.mean(dim=1)
        else:
            mask_f = mask_cond.float()
            g = (h * mask_f.unsqueeze(-1)).sum(dim=1)
            g = g / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return g, h, vis_mask
