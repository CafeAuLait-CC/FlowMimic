import torch

from flowmimic.src.model.flow.solver import solve_flow


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state, allow_missing=False):
        current_keys = set(self.shadow)
        state_keys = set(state)
        missing = sorted(current_keys - state_keys)
        unexpected = sorted(state_keys - current_keys)
        if (missing or unexpected) and not allow_missing:
            raise RuntimeError(
                f"EMA state is incompatible: missing={missing}, "
                f"unexpected={unexpected}"
            )
        merged = {}
        for key, value in self.shadow.items():
            source = state.get(key, value)
            merged[key] = source.detach().clone()
        self.shadow = merged


class Teacher:
    def __init__(self, model, solver_cfg):
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.solver_cfg = dict(solver_cfg)

    @torch.no_grad()
    def build_condition_batch(
        self,
        *,
        k2d,
        tau_cond,
        vis_mask,
        mask_cond,
        style_id,
        domain_id,
        tau_out,
        mean=None,
        std=None,
        null_mask=None,
    ):
        """Encode raw conditions with the frozen teacher's own modules."""
        self.model.eval()
        g2d, memory, _ = self.model.cond_encoder(
            k2d,
            tau_cond,
            vis_mask=vis_mask,
            mask_cond=mask_cond,
            mean=mean,
            std=std,
        )
        style = self.model.style_emb(
            style_id,
            domain_id,
            apply_dropout=False,
        )
        global_context = self.model.cond_mlp(torch.cat([g2d, style], dim=-1))

        if null_mask is None:
            null_mask = torch.zeros(
                k2d.shape[0],
                dtype=torch.bool,
                device=k2d.device,
            )
        if bool(null_mask.any()):
            if not self.model.true_null_condition:
                raise RuntimeError(
                    "Teacher received true-null rows but has no true-null condition"
                )
            (
                global_context,
                memory,
                tau_cond,
                mask_cond,
            ) = self.model.apply_true_null_condition(
                global_context,
                memory,
                tau_cond,
                mask_cond,
                style_id,
                domain_id,
                null_mask,
            )

        return {
            "tau_out": tau_out,
            "tau_cond": tau_cond,
            "mem": memory,
            "g": global_context,
            "mem_mask": ~mask_cond,
        }

    @torch.no_grad()
    def generate_x1_hat(self, x0, cond_batch):
        self.model.eval()
        return solve_flow(self.model, x0, cond_batch, **self.solver_cfg)
