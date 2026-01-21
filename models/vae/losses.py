import torch
from torch import nn


def masked_smooth_l1(pred, target, mask=None, reduction="mean"):
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    if mask is not None:
        loss = loss * mask.unsqueeze(-1).float()
    if reduction == "mean":
        if mask is not None:
            denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
        else:
            denom = loss.numel()
        return loss.sum() / denom
    if reduction == "sum":
        return loss.sum()
    return loss


def masked_kl(mu, logvar, mask=None):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if mask is not None:
        kl = kl * mask.unsqueeze(-1).float()
        denom = mask.sum().clamp_min(1.0) * mu.shape[-1]
    else:
        denom = kl.numel()
    return kl.sum() / denom


def masked_velocity_loss(x_hat, x, mask=None):
    dx_hat = x_hat[:, 1:] - x_hat[:, :-1]
    dx = x[:, 1:] - x[:, :-1]
    vel_mask = mask[:, 1:] if mask is not None else None
    return masked_smooth_l1(dx_hat, dx, vel_mask)


def masked_acceleration_loss(x_hat, x, mask=None):
    dx_hat = x_hat[:, 1:] - x_hat[:, :-1]
    dx = x[:, 1:] - x[:, :-1]
    ddx_hat = dx_hat[:, 1:] - dx_hat[:, :-1]
    ddx = dx[:, 1:] - dx[:, :-1]
    acc_mask = mask[:, 2:] if mask is not None else None
    return masked_smooth_l1(ddx_hat, ddx, acc_mask)


def style_ce_loss(logits, style_id, domain_id):
    if logits is None:
        return None
    mask = domain_id == 1
    if mask.sum() == 0:
        return logits.sum() * 0
    return nn.functional.cross_entropy(logits[mask], style_id[mask])
