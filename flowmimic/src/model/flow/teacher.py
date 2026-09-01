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
        global_context, memory, _ = self.model.encode_cond(
            k2d,
            tau_cond,
            style_id,
            domain_id,
            apply_style_dropout=False,
            vis_mask=vis_mask,
            mask_cond=mask_cond,
            mean=mean,
            std=std,
        )

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
    def build_guided_condition_batch(
        self,
        *,
        k2d,
        tau_cond,
        vis_mask,
        mask_cond,
        style_id,
        domain_id,
        tau_out,
        guidance_scale,
        mean=None,
        std=None,
    ):
        """Encode matching conditional/null memories for guided transport."""
        cond_batch = self.build_condition_batch(
            k2d=k2d,
            tau_cond=tau_cond,
            vis_mask=vis_mask,
            mask_cond=mask_cond,
            style_id=style_id,
            domain_id=domain_id,
            tau_out=tau_out,
            mean=mean,
            std=std,
        )
        null_g, null_mem, null_mem_mask, null_tau = self.model.encode_null_condition(
            style_id,
            domain_id,
        )
        cond_batch.update(
            {
                "mem_uncond": null_mem,
                "g_uncond": null_g,
                "mem_mask_uncond": null_mem_mask,
                "tau_cond_uncond": null_tau,
                "guidance_scale": guidance_scale,
            }
        )
        return cond_batch

    @torch.no_grad()
    def generate_x1_hat(self, x0, cond_batch):
        self.model.eval()
        return solve_flow(self.model, x0, cond_batch, **self.solver_cfg)

    @torch.no_grad()
    def predict_cfg_pair(self, x_t, t_flow, cond_batch):
        """Evaluate frozen conditional/null fields at one shared guided point."""
        required = (
            "tau_out",
            "tau_cond",
            "mem",
            "g",
            "mem_uncond",
            "g_uncond",
        )
        missing = [key for key in required if key not in cond_batch]
        if missing:
            raise ValueError(f"Guided condition batch is missing fields: {missing}")
        self.model.eval()
        v_cond = self.model.flow(
            x_t,
            t_flow,
            cond_batch["tau_out"],
            cond_batch["mem"],
            cond_batch["g"],
            mem_mask=cond_batch.get("mem_mask"),
            tau_cond=cond_batch["tau_cond"],
        )
        v_uncond = self.model.flow(
            x_t,
            t_flow,
            cond_batch["tau_out"],
            cond_batch["mem_uncond"],
            cond_batch["g_uncond"],
            mem_mask=cond_batch.get("mem_mask_uncond"),
            tau_cond=cond_batch.get("tau_cond_uncond", cond_batch["tau_cond"]),
        )
        return v_cond, v_uncond
