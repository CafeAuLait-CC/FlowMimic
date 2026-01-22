import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from models.vae.datasets.dataset_aist import AISTDataset
from models.vae.datasets.dataset_mvh import MVHumanNetDataset
from models.vae.datasets.label_map_builder import build_genre_to_id, save_genre_to_id
from models.vae.losses import grouped_recon_loss, masked_kl, smoothness_loss, style_ce_loss
from models.vae.motion_vae import MotionVAE
from models.vae.stats import compute_mean_std_from_splits, load_mean_std
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


def apply_style_dropout(style_id, domain_id, p):
    if p <= 0:
        return style_id
    drop_mask = (domain_id == 1) & (torch.rand_like(style_id.float()) < p)
    style_id = style_id.clone()
    style_id[drop_mask] = 0
    return style_id


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def aist_split_paths(aist_dir, split_path):
    names = read_lines(split_path)
    return [os.path.join(aist_dir, f\"{name}.pkl\") for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ratio-aist", type=int, default=1)
    parser.add_argument("--ratio-mvh", type=int, default=1)
    parser.add_argument("--kl-warmup", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--w-vel", type=float, default=None)
    parser.add_argument("--w-acc", type=float, default=None)
    parser.add_argument("--w-style", type=float, default=None)
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--genre-map", type=str, default="genre_to_id.json")
    # stats paths are taken from config (separate per dataset)
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    aist_genres = config["aist_genres"]
    num_styles = config["num_styles"]
    d_in = config["d_in"]
    d_z = config["d_z"]
    seq_len = args.seq_len or config["seq_len"]
    kl_warmup = args.kl_warmup or config["kl_warmup_steps"]
    kl_weight_target = args.kl_weight or config["kl_target_weight"]
    w_vel = args.w_vel or config["w_vel"]
    w_acc = args.w_acc or config["w_acc"]
    w_style = args.w_style or config["w_style"]
    w_contact = args.w_contact or config["w_contact"]
    style_dropout_p = config["style_dropout_p"]
    stats_path = config["stats_path"]
    aist_split_train = config["aist_split_train"]
    mvh_split_train = config["mvh_split_train"]

    if not os.path.exists(aist_split_train):
        raise FileNotFoundError(f"AIST split file not found: {aist_split_train}")
    if not os.path.exists(mvh_split_train):
        raise FileNotFoundError(f"MVHumanNet split file not found: {mvh_split_train}")

    if os.path.exists(args.genre_map):
        with open(args.genre_map, "r", encoding="utf-8") as f:
            genre_to_id = json.load(f)
    else:
        genre_to_id = build_genre_to_id(aist_genres)
        save_genre_to_id(genre_to_id, args.genre_map)

    aist_train_paths = aist_split_paths(aist_dir, aist_split_train)
    mvh_train_dirs = read_lines(mvh_split_train)

    if not os.path.exists(stats_path):
        compute_mean_std_from_splits(
            aist_train_paths,
            mvh_train_dirs,
            stats_path,
            workers=10,
        )

    mean, std = load_mean_std(stats_path)

    dataset_a = AISTDataset(
        aist_dir,
        genre_to_id,
        seq_len,
        mean=mean,
        std=std,
        files=aist_train_paths,
    )

    loader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = MotionVAE(d_in=d_in, d_z=d_z, num_styles=num_styles, max_len=seq_len)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        rng = random.Random(epoch)
        rng.shuffle(mvh_train_dirs)
        mvh_subset = mvh_train_dirs[: min(len(mvh_train_dirs), len(aist_train_paths))]

        dataset_b = MVHumanNetDataset(
            mv_root,
            seq_len,
            mean=mean,
            std=std,
            sequence_dirs=mvh_subset,
        )
        loader_b = DataLoader(
            dataset_b, batch_size=args.batch_size, shuffle=True, drop_last=True
        )

        batch_iter = balanced_batch_iter(loader_a, loader_b, args.ratio_aist, args.ratio_mvh)

        for _ in range(min(len(loader_a), len(loader_b))):
            batches = next(batch_iter)
            motion, domain_id, style_id, mask = merge_batches(batches)
            motion = motion.to(args.device)
            domain_id = domain_id.to(args.device)
            style_id = style_id.to(args.device)
            mask = mask.to(args.device)

            style_id_in = apply_style_dropout(style_id, domain_id, style_dropout_p)
            outputs = model(motion, domain_id, style_id_in, mask=mask)
            x_hat = outputs["x_hat"]

            recon, cont_loss, contact_loss = grouped_recon_loss(
                x_hat, motion, mask, w_contact=w_contact
            )
            kl = masked_kl(outputs["mu"], outputs["logvar"], mask)
            vel, acc = smoothness_loss(x_hat, motion, mask)
            style_loss = style_ce_loss(outputs.get("style_logits"), style_id_in, domain_id)

            kld_weight = kl_weight(step, kl_warmup, kl_weight_target)
            loss = recon + kld_weight * kl + w_vel * vel + w_acc * acc
            if style_loss is not None:
                loss = loss + w_style * style_loss

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
