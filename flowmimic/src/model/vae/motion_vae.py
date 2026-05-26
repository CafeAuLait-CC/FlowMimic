import torch
from torch import nn

from flowmimic.src.model.vae.cond_embedding import CondEmbedding
from flowmimic.src.model.vae.transformer_blocks import AdaLNTransformerBlock


class MotionVAE(nn.Module):
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
        latent_len=None,
        latent_token_mode="pool",
    ):
        super().__init__()
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
            if latent_token_mode not in {"pool", "query"}:
                raise ValueError(f"Unsupported latent_token_mode: {latent_token_mode}")
            if latent_token_mode == "pool":
                self.latent_from_pooled = nn.Linear(d_model, latent_len * d_model)
            else:
                self.latent_queries = nn.Parameter(torch.zeros(1, latent_len, d_model))
            self.latent_pos = nn.Parameter(torch.zeros(1, latent_len, d_model))
            self.dec_latent_pos = nn.Parameter(torch.zeros(1, latent_len, d_model))

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

        self.to_mu = nn.Linear(d_model, d_z)
        self.to_logvar = nn.Linear(d_model, d_z)
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

    def encode(self, x, cond, mask=None):
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
        mu = self.to_mu(latent_h)
        logvar = self.to_logvar(latent_h)
        return enc_h, mu, logvar

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

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x, domain_id, style_id, mask=None):
        cond = self.cond(domain_id, style_id)
        enc_h, mu, logvar = self.encode(x, cond, mask=mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cond, mask=mask, out_len=x.shape[1])

        outputs = {
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

        if self.style_head is not None:
            if mask is None:
                pooled = enc_h.mean(dim=1)
            else:
                pooled = self._masked_pool(enc_h, mask)
            outputs["style_logits"] = self.style_head(pooled)

        return outputs
