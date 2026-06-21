import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from flowmimic.src.model.vae.cond_embedding import CondEmbedding
from flowmimic.src.model.vae.transformer_blocks import AdaLNTransformerBlock


class EMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes=1024,
        code_dim=256,
        commitment_weight=0.25,
        decay=0.99,
        eps=1e-5,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.commitment_weight = float(commitment_weight)
        self.decay = float(decay)
        self.eps = float(eps)

        embed = torch.randn(self.num_codes, self.code_dim) * 0.02
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.num_codes))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z_e, update_codebook=True):
        if z_e.shape[-1] != self.code_dim:
            raise ValueError(f"Expected code dim {self.code_dim}, got {z_e.shape[-1]}")

        flat = z_e.reshape(-1, self.code_dim)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.square().sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.num_codes).type(flat.dtype)

        if self.training and update_codebook:
            cluster_size = encodings.sum(dim=0)
            embed_sum = encodings.t() @ flat.detach()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(cluster_size)
                dist.all_reduce(embed_sum)
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(
                    cluster_size, alpha=1.0 - self.decay
                )
                self.embed_avg.mul_(self.decay).add_(
                    embed_sum, alpha=1.0 - self.decay
                )
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps)
                    / (n + self.num_codes * self.eps)
                    * n.clamp_min(self.eps)
                )
                self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))

        z_q = self.embed[indices].view_as(z_e)
        loss = self.commitment_weight * F.mse_loss(z_e, z_q.detach())
        z_q_st = z_e + (z_q - z_e).detach()

        avg_probs = encodings.float().mean(dim=0)
        perplexity = torch.exp(
            -(avg_probs * torch.log(avg_probs + self.eps)).sum()
        )
        codes_used = (avg_probs > 0).float().sum()
        indices = indices.view(z_e.shape[:-1])
        return z_q_st, indices, loss, perplexity, codes_used


class MotionVQVAE(nn.Module):
    def __init__(
        self,
        d_in=263,
        d_z=256,
        d_model=512,
        max_len=256,
        n_layers=8,
        n_heads=8,
        ffn_dim=2048,
        dropout=0.1,
        num_domains=2,
        num_styles=1,
        dom_dim=16,
        style_dim=32,
        cond_dim=256,
        use_style_head=True,
        latent_len=16,
        latent_token_mode="query",
        codebook_size=1024,
        commitment_weight=0.25,
        codebook_decay=0.99,
    ):
        super().__init__()
        if latent_len is not None and latent_token_mode not in {"pool", "query"}:
            raise ValueError(f"Unsupported latent_token_mode: {latent_token_mode}")
        self.d_in = d_in
        self.d_z = d_z
        self.d_model = d_model
        self.num_styles = num_styles
        self.max_len = max_len
        self.latent_len = latent_len
        self.latent_token_mode = latent_token_mode

        self.cond = CondEmbedding(num_domains, num_styles, dom_dim, style_dim, cond_dim)
        self.enc_in = nn.Linear(d_in, d_model)
        self.dec_in = nn.Linear(d_z, d_model)
        self.enc_pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dec_pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        if latent_len is not None:
            if latent_token_mode == "pool":
                self.latent_from_pooled = nn.Linear(d_model, latent_len * d_model)
            else:
                self.latent_queries = nn.Parameter(torch.empty(1, latent_len, d_model))
                nn.init.normal_(self.latent_queries, std=0.02)
            self.latent_pos = nn.Parameter(torch.empty(1, latent_len, d_model))
            self.dec_latent_pos = nn.Parameter(torch.empty(1, latent_len, d_model))
            nn.init.normal_(self.latent_pos, std=0.02)
            nn.init.normal_(self.dec_latent_pos, std=0.02)

        self.encoder = nn.ModuleList(
            [
                AdaLNTransformerBlock(d_model, cond_dim, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                AdaLNTransformerBlock(d_model, cond_dim, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.to_latent = nn.Linear(d_model, d_z)
        self.quantizer = EMAVectorQuantizer(
            num_codes=codebook_size,
            code_dim=d_z,
            commitment_weight=commitment_weight,
            decay=codebook_decay,
        )
        self.to_out = nn.Linear(d_model, d_in)

        if use_style_head:
            self.style_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, num_styles),
            )
        else:
            self.style_head = None

    def _positional(self, x, pos_param):
        if x.shape[1] > pos_param.shape[1]:
            raise ValueError("Sequence length exceeds max_len for positional embeddings")
        pos = pos_param[:, : x.shape[1]].expand(x.shape[0], -1, -1)
        return x + pos

    def _masked_pool(self, h, mask=None):
        if mask is None:
            return h.mean(dim=1)
        mask_f = mask.float()
        pooled = (h * mask_f.unsqueeze(-1)).sum(dim=1)
        return pooled / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

    def encode_continuous(self, x, cond, mask=None):
        h = self.enc_in(x)
        h = self._positional(h, self.enc_pos)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask
        if self.latent_len is not None and self.latent_token_mode == "query":
            latent_h = self.latent_queries + self.latent_pos
            latent_h = latent_h.expand(h.shape[0], -1, -1)
            h = torch.cat([latent_h, h], dim=1)
            if key_padding_mask is not None:
                latent_mask = torch.zeros(
                    h.shape[0],
                    self.latent_len,
                    dtype=torch.bool,
                    device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat([latent_mask, key_padding_mask], dim=1)
        for block in self.encoder:
            h = block(h, cond, key_padding_mask=key_padding_mask)

        if self.latent_len is None:
            latent_h = h
            enc_h = h
        elif self.latent_token_mode == "query":
            latent_h = h[:, : self.latent_len]
            enc_h = h[:, self.latent_len :]
        else:
            pooled = self._masked_pool(h, mask)
            latent_h = self.latent_from_pooled(pooled).view(
                h.shape[0], self.latent_len, self.d_model
            )
            latent_h = latent_h + self.latent_pos.expand(h.shape[0], -1, -1)
            enc_h = h
        z_e = self.to_latent(latent_h)
        return enc_h, z_e

    def encode(self, x, cond, mask=None, update_codebook=True):
        enc_h, z_e = self.encode_continuous(x, cond, mask=mask)
        z_q, code_ids, vq_loss, perplexity, codes_used = self.quantizer(
            z_e, update_codebook=update_codebook
        )
        return enc_h, z_e, z_q, code_ids, vq_loss, perplexity, codes_used

    def decode(self, z, cond, mask=None, out_len=None):
        if self.latent_len is None:
            h = self.dec_in(z)
            h = self._positional(h, self.dec_pos)
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = ~mask
            for block in self.decoder:
                h = block(h, cond, key_padding_mask=key_padding_mask)
            return self.to_out(h)

        if out_len is None:
            out_len = mask.shape[1] if mask is not None else self.max_len
        if out_len > self.max_len:
            raise ValueError("Output length exceeds max_len for positional embeddings")
        latent_h = self.dec_in(z)
        if z.shape[1] <= self.dec_latent_pos.shape[1]:
            latent_h = latent_h + self.dec_latent_pos[:, : z.shape[1]].expand(
                z.shape[0], -1, -1
            )
        frame_h = self.dec_pos[:, :out_len].expand(z.shape[0], -1, -1)
        h = torch.cat([latent_h, frame_h], dim=1)
        key_padding_mask = None
        if mask is not None:
            latent_mask = torch.zeros(
                z.shape[0], z.shape[1], dtype=torch.bool, device=mask.device
            )
            key_padding_mask = torch.cat([latent_mask, ~mask], dim=1)
        for block in self.decoder:
            h = block(h, cond, key_padding_mask=key_padding_mask)
        frame_h = h[:, z.shape[1] :]
        return self.to_out(frame_h)

    def forward(self, x, domain_id, style_id, mask=None, update_codebook=True):
        cond = self.cond(domain_id, style_id)
        enc_h, z_e, z_q, code_ids, vq_loss, perplexity, codes_used = self.encode(
            x, cond, mask=mask, update_codebook=update_codebook
        )
        x_hat = self.decode(z_q, cond, mask=mask, out_len=x.shape[1])
        outputs = {
            "x_hat": x_hat,
            "z_e": z_e,
            "z_q": z_q,
            "code_ids": code_ids,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "codes_used": codes_used,
        }
        if self.style_head is not None:
            if mask is None:
                pooled = enc_h.mean(dim=1)
            else:
                pooled = self._masked_pool(enc_h, mask)
            outputs["style_logits"] = self.style_head(pooled)
        return outputs
