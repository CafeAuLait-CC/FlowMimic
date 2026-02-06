import argparse
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.cond_api import build_cond_inputs, build_dummy_cond
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.motion.process_motion import ik263_to_smpl22


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", default="checkpoints/motion_vae_best.pt")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--solver", type=str, default="heun")
    parser.add_argument("--style-id", type=int, default=0)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--k2d-npy", type=str, default=None)
    parser.add_argument("--tau-cond-npy", type=str, default=None)
    parser.add_argument("--out", type=str, default="sample_flow.npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config()
    seq_len = config["seq_len"]
    d_z = config["d_z"]

    flow_cfg = config.get("flow", {})
    flow = ConditionalRectFlow(
        d_z=d_z,
        d_model=flow_cfg.get("d_model", 512),
        n_layers=flow_cfg.get("n_layers", 8),
        n_heads=flow_cfg.get("n_heads", 8),
        ffn_dim=flow_cfg.get("ffn_dim", 2048),
        dropout=flow_cfg.get("dropout", 0.1),
        num_styles=config["num_styles"],
        style_dim=flow_cfg.get("style_dim", 32),
        cond_dim=flow_cfg.get("cond_dim", 256),
        cond_layers=flow_cfg.get("cond_layers", 4),
        cond_heads=flow_cfg.get("cond_heads", 4),
        p_style_drop=flow_cfg.get("p_style_drop", 0.5),
    )
    state = torch.load(args.checkpoint, map_location=args.device)
    flow.load_state_dict(state["model"])
    flow.to(args.device)
    flow.eval()

    vae = MotionVAE(d_in=config["d_in"], d_z=d_z, num_styles=config["num_styles"], max_len=seq_len)
    vae_state = torch.load(args.vae_checkpoint, map_location=args.device)
    vae.load_state_dict(vae_state["model"])
    vae.to(args.device)
    vae.eval()

    if args.k2d_npy and args.tau_cond_npy:
        k2d = np.load(args.k2d_npy)
        tau_cond = np.load(args.tau_cond_npy)
        cond = build_cond_inputs(k2d, tau_cond, args.device)
    else:
        cond = build_dummy_cond(1, device=args.device)

    tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=args.device)
    x0 = torch.randn(1, seq_len, d_z, device=args.device)

    g, mem, _vis = flow.cond_encoder(cond["k2d"], cond["tau_cond"])
    style_id = torch.tensor([args.style_id], device=args.device)
    domain_id = torch.tensor([args.domain_id], device=args.device)
    style = flow.style_emb(style_id, domain_id, apply_dropout=False)
    g = flow.cond_mlp(torch.cat([g, style], dim=-1))

    cond_batch = {"tau_out": tau_out, "mem": mem, "g": g}
    z_hat = solve_flow(flow.flow, x0, cond_batch, num_steps=args.steps, method=args.solver)

    with torch.no_grad():
        x_hat = vae.decode(z_hat, vae.cond(domain_id, style_id))
    joints = ik263_to_smpl22(x_hat.squeeze(0).cpu().numpy())
    np.save(args.out, joints)


if __name__ == "__main__":
    main()
