import torch
from torch.utils.checkpoint import checkpoint


def _flow_eval(flow_fn, x, t, cond_batch):
    v_cond = flow_fn(
        x,
        t,
        cond_batch["tau_out"],
        cond_batch["mem"],
        cond_batch["g"],
        cond_batch.get("mem_mask"),
    )
    guidance_scale = cond_batch.get("guidance_scale")
    if (
        guidance_scale is None
        or float(guidance_scale) == 1.0
        or "mem_uncond" not in cond_batch
        or "g_uncond" not in cond_batch
    ):
        return v_cond
    v_uncond = flow_fn(
        x,
        t,
        cond_batch["tau_out"],
        cond_batch["mem_uncond"],
        cond_batch["g_uncond"],
        cond_batch.get("mem_mask_uncond"),
    )
    return v_uncond + float(guidance_scale) * (v_cond - v_uncond)


def solve_flow(
    model,
    x0,
    cond_batch,
    num_steps=8,
    method="euler",
    use_activation_checkpoint=False,
):
    dt = 1.0 / max(num_steps, 1)
    x = x0
    t = torch.zeros(x0.shape[0], dtype=x0.dtype, device=x0.device)
    flow_fn = model.flow if hasattr(model, "flow") else model

    def _eval(x_in, t_in):
        mem_mask = cond_batch.get("mem_mask")
        if (
            not use_activation_checkpoint
            or not torch.is_grad_enabled()
            or mem_mask is None
        ):
            return _flow_eval(flow_fn, x_in, t_in, cond_batch)
        return checkpoint(
            _flow_eval_ckpt,
            x_in,
            t_in,
            cond_batch["tau_out"],
            cond_batch["mem"],
            cond_batch["g"],
            mem_mask,
            use_reentrant=False,
        )

    def _flow_eval_ckpt(x_in, t_in, tau_out, mem, g, mem_mask):
        return flow_fn(x_in, t_in, tau_out, mem, g, mem_mask)

    for _ in range(num_steps):
        if method == "heun":
            v1 = _eval(x, t)
            x1 = x + v1 * dt
            t1 = t + dt
            v2 = _eval(x1, t1)
            x = x + 0.5 * (v1 + v2) * dt
        else:
            v = _eval(x, t)
            x = x + v * dt
        t = t + dt
    return x
