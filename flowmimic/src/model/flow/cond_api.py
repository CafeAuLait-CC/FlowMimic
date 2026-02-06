import torch


def build_cond_inputs(k2d, tau_cond, device):
    if k2d is None or tau_cond is None:
        raise ValueError("k2d and tau_cond must be provided")
    if not torch.is_tensor(k2d):
        k2d = torch.tensor(k2d, dtype=torch.float32, device=device)
    if not torch.is_tensor(tau_cond):
        tau_cond = torch.tensor(tau_cond, dtype=torch.float32, device=device)
    if k2d.dim() == 3:
        k2d = k2d.unsqueeze(0)
    if tau_cond.dim() == 1:
        tau_cond = tau_cond.unsqueeze(0)
    return {"k2d": k2d, "tau_cond": tau_cond}


def build_dummy_cond(batch_size, t_cond=6, num_joints=25, device="cpu"):
    k2d = torch.zeros(batch_size, t_cond, num_joints, 2, device=device)
    tau_cond = torch.linspace(0.0, 1.0, steps=t_cond, device=device)
    tau_cond = tau_cond.unsqueeze(0).expand(batch_size, -1)
    return {"k2d": k2d, "tau_cond": tau_cond}
