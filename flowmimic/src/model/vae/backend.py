from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.motion_vqvae import MotionVQVAE


VAE_TYPE_AUTO = "auto"
VAE_TYPE_MOTION = "motion_vae"
VAE_TYPE_VQ = "motion_vqvae"


@dataclass
class LoadedVAE:
    model: torch.nn.Module
    ckpt: dict[str, Any]
    vae_type: str
    latent_len: int
    d_z: int
    max_len: int


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def infer_vae_type(state: dict[str, torch.Tensor]) -> str:
    if "quantizer.embed" in state or any(key.startswith("quantizer.") for key in state):
        return VAE_TYPE_VQ
    return VAE_TYPE_MOTION


def _infer_vq_latent_len(
    state: dict[str, torch.Tensor], config: dict[str, Any], latent_len: int | None
) -> int:
    if latent_len is not None:
        return int(latent_len)
    if "latent_queries" in state:
        return int(state["latent_queries"].shape[1])
    if "latent_pos" in state:
        return int(state["latent_pos"].shape[1])
    value = config.get("latent_len")
    if value is None:
        raise ValueError("VQ-VAE checkpoint does not expose latent_len")
    return int(value)


def load_vae_backend(
    ckpt_path: str,
    config: dict[str, Any],
    device: torch.device | str,
    *,
    seq_len: int | None = None,
    vae_type: str = VAE_TYPE_AUTO,
    latent_len: int | None = None,
    latent_token_mode: str | None = None,
) -> LoadedVAE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = _strip_module_prefix(ckpt["model"])
    ckpt_config = ckpt.get("config", {})
    inferred_type = infer_vae_type(state)
    if vae_type == VAE_TYPE_AUTO:
        vae_type = inferred_type
    if vae_type != inferred_type:
        raise ValueError(
            f"Requested vae_type={vae_type}, but checkpoint looks like {inferred_type}: {ckpt_path}"
        )

    max_len = int(state["enc_pos"].shape[1])
    if seq_len is not None and int(seq_len) > max_len:
        raise ValueError(
            f"Requested seq_len={seq_len}, but VAE checkpoint max_len is {max_len}: {ckpt_path}"
        )

    if vae_type == VAE_TYPE_VQ:
        mode = latent_token_mode or ckpt_config.get(
            "latent_token_mode", "query" if "latent_queries" in state else "pool"
        )
        n_latent = _infer_vq_latent_len(state, ckpt_config, latent_len)
        d_model = int(state["enc_in.weight"].shape[0])
        d_z = int(state["to_latent.weight"].shape[0])
        model = MotionVQVAE(
            d_in=int(state["enc_in.weight"].shape[1]),
            d_z=d_z,
            d_model=d_model,
            max_len=max_len,
            num_styles=int(state["cond.style_emb.weight"].shape[0]),
            latent_len=n_latent,
            latent_token_mode=mode,
            codebook_size=int(state["quantizer.embed"].shape[0]),
            commitment_weight=float(ckpt_config.get("commitment_weight", 0.25)),
            codebook_decay=float(ckpt_config.get("codebook_decay", 0.99)),
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        return LoadedVAE(model, ckpt, vae_type, n_latent, d_z, max_len)

    mode = latent_token_mode or ckpt_config.get("latent_token_mode", "pool")
    n_latent = latent_len if latent_len is not None else ckpt_config.get("latent_len")
    d_z = int(state["to_mu.weight"].shape[0])
    model = MotionVAE(
        d_in=int(state["enc_in.weight"].shape[1]),
        d_z=d_z,
        num_styles=int(state["cond.style_emb.weight"].shape[0]),
        max_len=max_len,
        latent_len=n_latent,
        latent_token_mode=mode,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    effective_latent_len = int(n_latent) if n_latent is not None else int(seq_len or max_len)
    return LoadedVAE(model, ckpt, vae_type, effective_latent_len, d_z, max_len)


def encode_motion_latent(
    vae: torch.nn.Module,
    motion: torch.Tensor,
    domain_id: torch.Tensor,
    style_id: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    cond = vae.cond(domain_id, style_id)
    if isinstance(vae, MotionVQVAE):
        _enc_h, _z_e, z_q, _code_ids, _vq_loss, _perplexity, _codes_used = vae.encode(
            motion, cond, mask=mask, update_codebook=False
        )
        return z_q
    _enc_h, mu, _logvar = vae.encode(motion, cond, mask=mask)
    return mu


def decode_motion_latent(
    vae: torch.nn.Module,
    z: torch.Tensor,
    domain_id: torch.Tensor,
    style_id: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    out_len: int | None = None,
) -> torch.Tensor:
    return vae.decode(z, vae.cond(domain_id, style_id), mask=mask, out_len=out_len)
