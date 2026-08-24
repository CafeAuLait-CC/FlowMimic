import torch
from torch.utils.checkpoint import checkpoint


def combine_cfg_velocities(v_cond, v_uncond, guidance_scale):
    """Combine conditional and null velocities for scalar or per-row CFG."""
    scale = torch.as_tensor(
        guidance_scale,
        dtype=v_cond.dtype,
        device=v_cond.device,
    )
    if scale.ndim > 1 or (scale.ndim == 1 and scale.shape[0] != v_cond.shape[0]):
        raise ValueError("guidance_scale must be scalar or have shape [B]")
    while scale.ndim < v_cond.ndim:
        scale = scale.unsqueeze(-1)
    return v_uncond + scale * (v_cond - v_uncond)


def cfg_branch_preservation_loss(
    student_cond,
    student_uncond,
    teacher_cond,
    teacher_uncond,
):
    """Return equally weighted conditional/null local velocity distillation."""
    shapes = {
        tuple(value.shape)
        for value in (student_cond, student_uncond, teacher_cond, teacher_uncond)
    }
    if len(shapes) != 1:
        raise ValueError("CFG branch-preservation tensors must have matching shapes")
    reduce_dims = tuple(range(1, student_cond.ndim))
    cond = torch.mean((student_cond - teacher_cond.detach()) ** 2, dim=reduce_dims)
    uncond = torch.mean(
        (student_uncond - teacher_uncond.detach()) ** 2,
        dim=reduce_dims,
    )
    return 0.5 * (cond.mean() + uncond.mean()), cond.mean(), uncond.mean()


def _flow_eval(flow_fn, x, t, cond_batch):
    v_cond = flow_fn(
        x,
        t,
        cond_batch["tau_out"],
        cond_batch["mem"],
        cond_batch["g"],
        cond_batch.get("mem_mask"),
        cond_batch.get("tau_cond"),
    )
    guidance_scale = cond_batch.get("guidance_scale")
    if guidance_scale is None or "mem_uncond" not in cond_batch or "g_uncond" not in cond_batch:
        return v_cond
    scale = torch.as_tensor(guidance_scale)
    if scale.numel() == 1 and float(scale) == 1.0:
        return v_cond
    v_uncond = flow_fn(
        x,
        t,
        cond_batch["tau_out"],
        cond_batch["mem_uncond"],
        cond_batch["g_uncond"],
        cond_batch.get("mem_mask_uncond"),
        cond_batch.get("tau_cond_uncond", cond_batch.get("tau_cond")),
    )
    return combine_cfg_velocities(v_cond, v_uncond, guidance_scale)


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
            cond_batch.get("tau_cond"),
            use_reentrant=False,
        )

    def _flow_eval_ckpt(x_in, t_in, tau_out, mem, g, mem_mask, tau_cond):
        return flow_fn(x_in, t_in, tau_out, mem, g, mem_mask, tau_cond)

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


def solve_flow_to_time(
    model,
    x0,
    cond_batch,
    end_time,
    num_steps=1,
    method="heun",
):
    """Integrate from zero to a scalar or per-row flow time."""
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    end_time = torch.as_tensor(end_time, dtype=x0.dtype, device=x0.device)
    if end_time.ndim == 0:
        end_time = end_time.expand(x0.shape[0])
    if end_time.ndim != 1 or end_time.shape[0] != x0.shape[0]:
        raise ValueError("end_time must be scalar or have shape [B]")
    if bool(((end_time < 0.0) | (end_time > 1.0)).any()):
        raise ValueError("end_time must lie in [0, 1]")

    flow_fn = model.flow if hasattr(model, "flow") else model
    dt = end_time / int(num_steps)
    dt_view = dt.view(-1, *([1] * (x0.ndim - 1)))
    x = x0
    t = torch.zeros_like(end_time)
    for _ in range(int(num_steps)):
        if method == "heun":
            v0 = _flow_eval(flow_fn, x, t, cond_batch)
            proposal = x + dt_view * v0
            v1 = _flow_eval(flow_fn, proposal, t + dt, cond_batch)
            x = x + 0.5 * dt_view * (v0 + v1)
        elif method == "euler":
            x = x + dt_view * _flow_eval(flow_fn, x, t, cond_batch)
        else:
            raise ValueError(f"Unknown solver method: {method}")
        t = t + dt
    return x
