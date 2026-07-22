import torch
from torch import nn


class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_gamma_beta = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        h = self.norm(x)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class RelativeTimeBias(nn.Module):
    """Learn a per-head cross-attention bias from relative timestamps."""

    def __init__(self, n_heads, hidden_dim=32):
        super().__init__()
        self.n_heads = int(n_heads)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_heads),
        )
        # Phase 1B starts exactly at the source model's attention function.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tau_out, tau_cond, key_padding_mask=None, dtype=None):
        delta = tau_out.unsqueeze(-1) - tau_cond.unsqueeze(-2)
        features = torch.stack((delta, delta.abs()), dim=-1)
        bias = self.net(features).permute(0, 3, 1, 2)
        if dtype is not None:
            bias = bias.to(dtype=dtype)
        if key_padding_mask is not None:
            mask_value = torch.finfo(bias.dtype).min
            bias = bias.masked_fill(
                key_padding_mask[:, None, None, :],
                mask_value,
            )
        batch, heads, out_len, cond_len = bias.shape
        return bias.reshape(batch * heads, out_len, cond_len)


class LatentSlotConditionAdapter(nn.Module):
    """Resample timestamped condition memory into learned VQ latent slots."""

    def __init__(
        self,
        latent_len,
        d_model,
        n_heads=8,
        ffn_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.latent_len = int(latent_len)
        self.slot_queries = nn.Parameter(
            torch.empty(1, self.latent_len, d_model)
        )
        nn.init.normal_(self.slot_queries, std=0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        # Migration from a legacy flow starts at exactly the source function.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @torch.no_grad()
    def initialize_slot_queries(self, queries):
        queries = torch.as_tensor(
            queries,
            device=self.slot_queries.device,
            dtype=self.slot_queries.dtype,
        )
        if queries.ndim == 2:
            queries = queries.unsqueeze(0)
        if queries.shape != self.slot_queries.shape:
            raise ValueError(
                "VQ slot query shape mismatch: "
                f"expected {tuple(self.slot_queries.shape)}, "
                f"got {tuple(queries.shape)}"
            )
        self.slot_queries.copy_(queries)

    def forward(self, memory, memory_mask=None):
        slots = self.slot_queries.expand(memory.shape[0], -1, -1)
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attn(
            self.query_norm(slots),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_mask,
            need_weights=False,
        )
        slots = slots + self.dropout(attended)
        slots = slots + self.dropout(self.ffn(self.ffn_norm(slots)))
        return self.out_proj(slots)


class FlowBlock(nn.Module):
    def __init__(
        self,
        d_model,
        cond_dim,
        n_heads,
        ffn_dim,
        dropout=0.1,
        relative_time_bias=False,
        relative_time_hidden_dim=32,
    ):
        super().__init__()
        self.adaln = AdaLN(d_model, cond_dim)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.Sigmoid(),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.relative_time_bias = (
            RelativeTimeBias(n_heads, hidden_dim=relative_time_hidden_dim)
            if relative_time_bias
            else None
        )

    def forward(
        self,
        x,
        cond,
        mem,
        mem_mask=None,
        tau_out=None,
        tau_cond=None,
    ):
        h = self.adaln(x, cond)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=None, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        attn_mask = None
        key_padding_mask = mem_mask
        if self.relative_time_bias is not None:
            if tau_out is None or tau_cond is None:
                raise ValueError(
                    "Relative-time attention requires tau_out and tau_cond"
                )
            attn_mask = self.relative_time_bias(
                tau_out,
                tau_cond,
                key_padding_mask=mem_mask,
                dtype=h.dtype,
            )
            # The padding mask is already represented by additive -inf bias.
            key_padding_mask = None
        cross_out, _ = self.cross_attn(
            h,
            mem,
            mem,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        gate = self.gate(cond).unsqueeze(1)
        x = x + self.dropout(cross_out * gate)

        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))
        return x
