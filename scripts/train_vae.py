import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from models.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from models.vae.datasets.dataset_aist import AISTDataset
from models.vae.datasets.dataset_mvh import MVHumanNetDataset
from models.vae.datasets.label_map_builder import build_genre_to_id, save_genre_to_id
from models.vae.losses import (
    masked_acceleration_loss,
    masked_kl,
    masked_smooth_l1,
    masked_velocity_loss,
    style_ce_loss,
)
from models.vae.motion_vae import MotionVAE
from utils.config import load_config


def merge_batches(batches):
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


def kl_weight(step, warmup_steps, max_weight):
    if warmup_steps <= 0:
        return max_weight
    return min(max_weight, max_weight * (step / warmup_steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ratio-aist", type=int, default=1)
    parser.add_argument("--ratio-mvh", type=int, default=1)
    parser.add_argument("--kl-warmup", type=int, default=20000)
    parser.add_argument("--kl-weight", type=float, default=2e-3)
    parser.add_argument("--w-vel", type=float, default=0.1)
    parser.add_argument("--w-acc", type=float, default=0.05)
    parser.add_argument("--w-style", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--genre-map", type=str, default="genre_to_id.json")
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]

    if os.path.exists(args.genre_map):
        with open(args.genre_map, "r", encoding="utf-8") as f:
            genre_to_id = json.load(f)
    else:
        genre_to_id = build_genre_to_id(aist_dir)
        save_genre_to_id(genre_to_id, args.genre_map)

    num_styles = max(genre_to_id.values()) + 1

    dataset_a = AISTDataset(aist_dir, genre_to_id, args.seq_len)
    dataset_b = MVHumanNetDataset(mv_root, args.seq_len)

    loader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=True, drop_last=True)

    d_in = 22 * 3
    model = MotionVAE(d_in=d_in, num_styles=num_styles)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        batch_iter = balanced_batch_iter(loader_a, loader_b, args.ratio_aist, args.ratio_mvh)

        for _ in range(min(len(loader_a), len(loader_b))):
            batches = next(batch_iter)
            motion, domain_id, style_id, mask = merge_batches(batches)
            motion = motion.to(args.device)
            domain_id = domain_id.to(args.device)
            style_id = style_id.to(args.device)
            mask = mask.to(args.device)

            outputs = model(motion, domain_id, style_id, mask=mask)
            x_hat = outputs["x_hat"]

            recon = masked_smooth_l1(x_hat, motion, mask)
            kl = masked_kl(outputs["mu"], outputs["logvar"], mask)
            vel = masked_velocity_loss(x_hat, motion, mask)
            acc = masked_acceleration_loss(x_hat, motion, mask)
            style_loss = style_ce_loss(outputs.get("style_logits"), style_id, domain_id)

            kld_weight = kl_weight(step, args.kl_warmup, args.kl_weight)
            loss = recon + kld_weight * kl + args.w_vel * vel + args.w_acc * acc
            if style_loss is not None:
                loss = loss + args.w_style * style_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

        ckpt_path = os.path.join(args.checkpoint_dir, f"motion_vae_epoch{epoch + 1}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "genre_to_id": genre_to_id,
                "config": vars(args),
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
