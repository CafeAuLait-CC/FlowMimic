import torch
from torch import nn

from models.vae.cond_embedding import CondEmbedding
from models.vae.transformer_blocks import AdaLNTransformerBlock


class MotionVAE(nn.Module):
    def __init__(
        self,
        d_in,
        d_z=128,
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
    ):
        super().__init__()
        self.d_in = d_in
        self.d_z = d_z
        self.d_model = d_model
        self.num_styles = num_styles

        self.cond = CondEmbedding(num_domains, num_styles, dom_dim, style_dim, cond_dim)

        self.enc_in = nn.Linear(d_in, d_model)
        self.dec_in = nn.Linear(d_z, d_model)
        self.enc_pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dec_pos = nn.Parameter(torch.zeros(1, max_len, d_model))

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

    def encode(self, x, cond, mask=None):
        h = self.enc_in(x)
        h = self._positional(h, self.enc_pos)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask
        for block in self.encoder:
            h = block(h, cond, key_padding_mask=key_padding_mask)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return h, mu, logvar

    def decode(self, z, cond, mask=None):
        h = self.dec_in(z)
        h = self._positional(h, self.dec_pos)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask
        for block in self.decoder:
            h = block(h, cond, key_padding_mask=key_padding_mask)
        return self.to_out(h)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x, domain_id, style_id, mask=None):
        cond = self.cond(domain_id, style_id)
        enc_h, mu, logvar = self.encode(x, cond, mask=mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cond, mask=mask)

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
                mask_f = mask.float()
                pooled = (enc_h * mask_f.unsqueeze(-1)).sum(dim=1)
                pooled = pooled / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            outputs["style_logits"] = self.style_head(pooled)

        return outputs
