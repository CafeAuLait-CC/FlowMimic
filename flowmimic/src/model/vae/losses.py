import torch
from torch import nn

LAYOUT_SLICES = {
    "root_yaw_vel": (0, 1),
    "root_xz_vel": (1, 3),
    "root_y": (3, 4),
    "ric": (4, 67),
    "rot_6d": (67, 193),
    "local_vel": (193, 259),
    "feet_contact": (259, 263),
}

SMOOTH_SLICES = [
    (0, 4),
    (67, 193),
    (193, 259),
]


def _masked_mean(loss, mask):
    if mask is None:
        return loss.mean()
    denom = mask.sum().clamp_min(1.0) * loss.shape[-1]
    return (loss * mask.unsqueeze(-1).float()).sum() / denom


def masked_smooth_l1(pred, target, mask=None):
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return _masked_mean(loss, mask)


def masked_bce_with_logits(logits, target, mask=None):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _masked_mean(loss, mask)


def masked_kl(mu, logvar, mask=None):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return _masked_mean(kl, mask)


def grouped_recon_loss(x_hat, x, mask=None, w_contact=5.0):
    cont_start, cont_end = LAYOUT_SLICES["root_yaw_vel"][0], LAYOUT_SLICES["feet_contact"][0]
    cont_loss = masked_smooth_l1(x_hat[..., cont_start:cont_end], x[..., cont_start:cont_end], mask)
    contact_slice = LAYOUT_SLICES["feet_contact"]
    contact_loss = masked_bce_with_logits(
        x_hat[..., contact_slice[0] : contact_slice[1]],
        x[..., contact_slice[0] : contact_slice[1]],
        mask,
    )
    return cont_loss + w_contact * contact_loss, cont_loss, contact_loss


def smoothness_loss(x_hat, x, mask=None, slices=None):
    if slices is None:
        slices = SMOOTH_SLICES

    def _gather(tensor):
        parts = [tensor[..., start:end] for start, end in slices]
        return torch.cat(parts, dim=-1)

    x_hat_s = _gather(x_hat)
    x_s = _gather(x)

    dx_hat = x_hat_s[:, 1:] - x_hat_s[:, :-1]
    dx = x_s[:, 1:] - x_s[:, :-1]
    vel_mask = mask[:, 1:] if mask is not None else None
    vel = masked_smooth_l1(dx_hat, dx, vel_mask)

    ddx_hat = dx_hat[:, 1:] - dx_hat[:, :-1]
    ddx = dx[:, 1:] - dx[:, :-1]
    acc_mask = mask[:, 2:] if mask is not None else None
    acc = masked_smooth_l1(ddx_hat, ddx, acc_mask)

    return vel, acc


def style_ce_loss(logits, style_id, domain_id):
    if logits is None:
        return None
    mask = (domain_id == 1) & (style_id != 0)
    if mask.sum() == 0:
        return logits.sum() * 0
    return nn.functional.cross_entropy(logits[mask], style_id[mask])
