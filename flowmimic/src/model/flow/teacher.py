import copy

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

    def load_state_dict(self, state):
        self.shadow = {k: v.detach().clone() for k, v in state.items()}


class Teacher:
    def __init__(self, model, solver_cfg):
        self.model = copy.deepcopy(model)
        self.solver_cfg = solver_cfg

    @torch.no_grad()
    def generate_x1_hat(self, x0, cond_batch):
        return solve_flow(self.model, x0, cond_batch, **self.solver_cfg)
