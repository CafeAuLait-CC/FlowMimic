import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.cond_api import build_dummy_cond
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.teacher import EMA, Teacher
from flowmimic.src.model.flow.solver import solve_flow
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std
from flowmimic.src.model.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from flowmimic.src.model.vae.datasets.label_map_builder import build_genre_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reflow-round", type=int, default=0)
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--teacher-solver", type=str, default=None)
    parser.add_argument("--use-ema-teacher", action="store_true")
    parser.add_argument("--teacher-mode", type=str, choices=["strict", "mixed"], default=None)
    parser.add_argument("--p-teacher", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_flow")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    seq_len = config["seq_len"]
    stats_path = config["stats_path"]
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)

    mean, std = load_mean_std(stats_path)

    genre_to_id = build_genre_to_id(config.get("aist_genres", []))

    flow_cfg = config.get("flow", {})
    lr = args.lr or flow_cfg.get("lr", 2e-4)
    weight_decay = flow_cfg.get("weight_decay", 1e-2)
    teacher_steps = args.teacher_steps or flow_cfg.get("teacher_steps", 16)
    teacher_solver = args.teacher_solver or flow_cfg.get("teacher_solver", "heun")
    teacher_mode = args.teacher_mode or flow_cfg.get("teacher_mode", "strict")
    p_teacher = args.p_teacher if args.p_teacher is not None else flow_cfg.get("p_teacher", 1.0)
    ema_decay = flow_cfg.get("ema_decay", 0.999)

    dataset_a = AISTDataset(
        aist_dir,
        genre_to_id=genre_to_id,
        seq_len=seq_len,
        mean=mean,
        std=std,
        cache_root=config["cache_root"],
        target_fps=target_fps,
        src_fps=aist_fps,
    )
    dataset_b = MVHumanNetDataset(
        mv_root,
        seq_len=seq_len,
        mean=mean,
        std=std,
        cache_root=config["cache_root"],
        target_fps=target_fps,
        src_fps=mvh_fps,
    )

    loader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=True, drop_last=True)
    batch_iter = balanced_batch_iter(loader_a, loader_b, 1, 1)

    vae = MotionVAE(d_in=config["d_in"], d_z=config["d_z"], num_styles=config["num_styles"], max_len=seq_len)
    vae_ckpt = torch.load(config.get("vae_ckpt", "checkpoints/motion_vae_best.pt"), map_location=args.device)
    vae.load_state_dict(vae_ckpt["model"])
    vae.to(args.device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    flow = ConditionalRectFlow(
        d_z=config["d_z"],
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
    flow.to(args.device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)

    teacher = None
    if args.reflow_round >= 1:
        if not args.teacher_ckpt:
            raise ValueError("teacher_ckpt is required for reflow_round >= 1")
        teacher_flow = ConditionalRectFlow(d_z=config["d_z"], num_styles=config["num_styles"])
        state = torch.load(args.teacher_ckpt, map_location=args.device)
        if "ema" in state:
            teacher_flow.load_state_dict(state["ema"])
        else:
            teacher_flow.load_state_dict(state["model"])
        teacher_flow.to(args.device)
        teacher = Teacher(
            teacher_flow,
            solver_cfg={"num_steps": teacher_steps, "method": teacher_solver},
        )

    ema = EMA(flow, decay=ema_decay) if args.use_ema_teacher else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tau_out = torch.linspace(0.0, 1.0, steps=seq_len, device=args.device)

    for epoch in range(args.epochs):
        flow.train()
        total_loss = 0.0
        total_count = 0
        for _ in tqdm(range(min(len(loader_a), len(loader_b))), desc=f"Flow Epoch {epoch+1}", leave=False):
            batch = next(batch_iter)
            motion, domain_id, style_id, mask = _merge_batches(batch)
            motion = motion.to(args.device)
            domain_id = domain_id.to(args.device)
            style_id = style_id.to(args.device)

            with torch.no_grad():
                enc_h, mu, _logvar = vae.encode(motion, vae.cond(domain_id, style_id), mask=mask.to(args.device))
                z_data = mu

            x0 = torch.randn_like(z_data)
            t = torch.rand(z_data.shape[0], device=args.device)
            x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * z_data

            cond = build_dummy_cond(z_data.shape[0], device=args.device)
            g2d, mem, _vis = flow.cond_encoder(cond["k2d"], cond["tau_cond"])
            style = flow.style_emb(style_id, domain_id, apply_dropout=True)
            g = flow.cond_mlp(torch.cat([g2d, style], dim=-1))
            v_pred = flow.flow(x_t, t, tau_out, mem, g)
            target = z_data - x0

            if teacher is not None and args.reflow_round >= 1:
                use_teacher = True
                if teacher_mode == "mixed":
                    use_teacher = torch.rand(1).item() < p_teacher
                if use_teacher:
                    with torch.no_grad():
                        style_t = flow.style_emb(style_id, domain_id, apply_dropout=False)
                        g_t = flow.cond_mlp(torch.cat([g2d, style_t], dim=-1))
                        cond_batch = {"tau_out": tau_out, "mem": mem, "g": g_t}
                        z_data = teacher.generate_x1_hat(x0, cond_batch)
                        target = z_data - x0

            loss = torch.mean((v_pred - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(flow)
            total_loss += loss.item()
            total_count += 1

        if args.debug:
            print(f"Epoch {epoch+1} avg_flow_loss={total_loss / max(total_count, 1):.6f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"flow_round{args.reflow_round}_epoch{epoch+1}.pt")
        state = {"model": flow.state_dict()}
        if ema is not None:
            state["ema"] = ema.state_dict()
        torch.save(state, ckpt_path)


def _merge_batches(batches):
    motions = []
    domain_ids = []
    style_ids = []
    masks = []
    for batch in batches:
        motions.append(batch["motion"])
        domain_ids.append(batch["domain_id"])
        style_ids.append(batch["style_id"])
        masks.append(batch["mask"])
    motion = torch.cat(motions, dim=0)
    domain_id = torch.cat(domain_ids, dim=0)
    style_id = torch.cat(style_ids, dim=0)
    mask = torch.cat(masks, dim=0)
    return motion, domain_id, style_id, mask


if __name__ == "__main__":
    main()
