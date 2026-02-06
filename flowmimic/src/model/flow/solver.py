import torch


def solve_flow(model, x0, cond_batch, num_steps=8, method="euler"):
    dt = 1.0 / max(num_steps, 1)
    x = x0
    t = torch.zeros(x0.shape[0], device=x0.device)
    for _ in range(num_steps):
        if method == "heun":
            v1 = model(x, t, cond_batch["tau_out"], cond_batch["mem"], cond_batch["g"], cond_batch.get("mem_mask"))
            x1 = x + v1 * dt
            t1 = t + dt
            v2 = model(x1, t1, cond_batch["tau_out"], cond_batch["mem"], cond_batch["g"], cond_batch.get("mem_mask"))
            x = x + 0.5 * (v1 + v2) * dt
        else:
            v = model(x, t, cond_batch["tau_out"], cond_batch["mem"], cond_batch["g"], cond_batch.get("mem_mask"))
            x = x + v * dt
        t = t + dt
    return x
