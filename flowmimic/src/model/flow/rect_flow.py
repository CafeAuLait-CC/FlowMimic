import torch
from torch import nn

from flowmimic.src.model.flow.cond_encoder_2d import CondEncoder2D
from flowmimic.src.model.flow.flow_net import FlowNet
from flowmimic.src.model.flow.style_embed import StyleEmbedding


POSE_CONDITIONING_MODES = (
    "full",
    "memory_only",
    "global_only",
    "style_only",
)


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
        true_null_condition=False,
        pose_conditioning="full",
    ):
        super().__init__()
        if pose_conditioning not in POSE_CONDITIONING_MODES:
            raise ValueError(
                f"Unknown pose conditioning mode {pose_conditioning!r}; "
                f"expected one of {POSE_CONDITIONING_MODES}"
            )
        self.pose_conditioning = pose_conditioning
        self.use_pose_memory = pose_conditioning in ("full", "memory_only")
        self.use_pose_global = pose_conditioning in ("full", "global_only")
        self.true_null_condition = bool(true_null_condition)
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
            use_condition_memory=self.use_pose_memory,
        )
        if self.true_null_condition:
            self.null_memory = nn.Parameter(torch.zeros(1, 1, d_model))
            self.null_global = nn.Parameter(torch.zeros(1, d_model))

    def encode_null_condition(self, style_id, domain_id):
        """Build a true pose-null condition while retaining style and domain."""
        if not self.true_null_condition:
            raise RuntimeError("True-null conditioning is not enabled")
        batch_size = int(style_id.shape[0])
        memory = self.null_memory.expand(batch_size, -1, -1)
        pose_global = self.null_global.expand(batch_size, -1)
        style = self.style_emb(
            style_id,
            domain_id,
            apply_dropout=False,
        )
        global_context = self.combine_pose_style(pose_global, style)
        memory_mask = torch.zeros(
            (batch_size, 1),
            dtype=torch.bool,
            device=style_id.device,
        )
        tau_cond = torch.zeros(
            (batch_size, 1),
            dtype=memory.dtype,
            device=memory.device,
        )
        return global_context, memory, memory_mask, tau_cond

    def apply_true_null_condition(
        self,
        global_context,
        memory,
        tau_cond,
        mask_cond,
        style_id,
        domain_id,
        null_mask,
    ):
        """Replace selected batch rows with one valid learned null token."""
        if not self.true_null_condition:
            raise RuntimeError("True-null conditioning is not enabled")
        if null_mask.ndim != 1 or null_mask.shape[0] != memory.shape[0]:
            raise ValueError("null_mask must have shape [B]")

        # Keep both learned null parameters in the graph for DDP control runs
        # where a batch may contain no null samples.
        null_link = 0.0 * (self.null_memory.sum() + self.null_global.sum())
        global_context = global_context + null_link
        if not bool(null_mask.any()):
            return global_context, memory, tau_cond, mask_cond

        null_global, null_memory, _, _ = self.encode_null_condition(
            style_id[null_mask],
            domain_id[null_mask],
        )
        global_out = global_context.clone()
        memory_out = memory.clone()
        tau_out = tau_cond.clone()
        mask_out = mask_cond.clone()

        global_out[null_mask] = null_global
        memory_out[null_mask] = 0.0
        memory_out[null_mask, 0] = null_memory[:, 0]
        tau_out[null_mask] = 0.0
        mask_out[null_mask] = False
        mask_out[null_mask, 0] = True
        return global_out, memory_out, tau_out, mask_out

    def combine_pose_style(self, pose_global, style):
        """Build global context while preserving matched ablation parameters."""
        if not self.use_pose_global:
            pose_global = pose_global * 0.0
        return self.cond_mlp(torch.cat([pose_global, style], dim=-1))

    def encode_cond(
        self,
        k2d,
        tau_cond,
        style_id,
        domain_id,
        apply_style_dropout=True,
        vis_mask=None,
        mask_cond=None,
        mean=None,
        std=None,
    ):
        g_2d, mem, vis_mask = self.cond_encoder(
            k2d,
            tau_cond,
            vis_mask=vis_mask,
            mask_cond=mask_cond,
            mean=mean,
            std=std,
        )
        style = self.style_emb(style_id, domain_id, apply_dropout=apply_style_dropout)
        g = self.combine_pose_style(g_2d, style)
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
        null_mask=None,
        return_cfg_diagnostics=False,
        return_cfg_pair=False,
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
        pred = self.flow(
            x_t,
            t_flow,
            tau_out,
            mem,
            g,
            mem_mask=mem_mask,
            tau_cond=tau_cond,
        )
        if return_cfg_pair:
            if not self.true_null_condition:
                raise RuntimeError("CFG pairs require true-null conditioning")
            if style_id is None or domain_id is None:
                raise ValueError(
                    "style_id and domain_id are required for a CFG pair"
                )
            null_g, null_mem, null_mem_mask, null_tau = self.encode_null_condition(
                style_id,
                domain_id,
            )
            null_pred = self.flow(
                x_t,
                t_flow,
                tau_out,
                null_mem,
                null_g,
                mem_mask=null_mem_mask,
                tau_cond=null_tau,
            )
            if return_cfg_diagnostics:
                guidance_delta_l2 = torch.linalg.vector_norm(
                    pred.detach().flatten(1) - null_pred.detach().flatten(1),
                    dim=1,
                )
                return (pred, null_pred), {
                    "guidance_delta_l2": guidance_delta_l2
                }
            return pred, null_pred
        if not self.true_null_condition:
            if return_cfg_diagnostics:
                return pred, {"guidance_delta_l2": pred.new_empty((0,))}
            return pred
        if null_mask is None:
            null_mask = torch.zeros(
                x_t.shape[0],
                dtype=torch.bool,
                device=x_t.device,
            )
        if null_mask.ndim != 1 or null_mask.shape[0] != x_t.shape[0]:
            raise ValueError("null_mask must have shape [B]")

        # Keep the normal condition memory in place. Only the small dropped
        # subset needs a second forward through the learned null condition.
        # This avoids cloning the large [B, K, D] condition memory.
        if bool(null_mask.any()):
            if style_id is None or domain_id is None:
                raise ValueError(
                    "style_id and domain_id are required for true-null rows"
                )
            null_indices = torch.nonzero(null_mask, as_tuple=False).flatten()
            null_g, null_mem, null_mem_mask, null_tau = self.encode_null_condition(
                style_id.index_select(0, null_indices),
                domain_id.index_select(0, null_indices),
            )
            null_pred = self.flow(
                x_t.index_select(0, null_indices),
                t_flow.index_select(0, null_indices),
                tau_out,
                null_mem,
                null_g,
                mem_mask=null_mem_mask,
                tau_cond=null_tau,
            )
            with torch.no_grad():
                guidance_delta_l2 = torch.linalg.vector_norm(
                    pred.index_select(0, null_indices).detach().flatten(1)
                    - null_pred.detach().flatten(1),
                    dim=1,
                )
            replacement = torch.zeros_like(pred).index_copy(
                0,
                null_indices,
                null_pred,
            )
            output = torch.where(
                null_mask[:, None, None],
                replacement,
                pred,
            )
            if return_cfg_diagnostics:
                return output, {"guidance_delta_l2": guidance_delta_l2}
            return output

        # DDP expects the learned null parameters to participate even in a
        # batch where the Bernoulli draw contains no null examples.
        null_link = 0.0 * (self.null_memory.sum() + self.null_global.sum())
        output = pred + null_link
        if return_cfg_diagnostics:
            return output, {"guidance_delta_l2": pred.new_empty((0,))}
        return output
