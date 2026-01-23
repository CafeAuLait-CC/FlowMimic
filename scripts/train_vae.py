import argparse
import json
import os
import random
import sys
import time

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
torch.multiprocessing.set_sharing_strategy("file_system")

from models.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from models.vae.datasets.dataset_aist import AISTDataset
from models.vae.datasets.dataset_mvh import MVHumanNetDataset
from models.vae.datasets.label_map_builder import build_genre_to_id, save_genre_to_id
from models.vae.losses import (
    grouped_recon_loss,
    masked_kl,
    smoothness_loss,
    style_ce_loss,
)
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
    return [os.path.join(aist_dir, f"{name}.pkl") for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ratio-aist", type=int, default=1)
    parser.add_argument("--ratio-mvh", type=int, default=1)
    parser.add_argument("--kl-warmup", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--w-vel", type=float, default=None)
    parser.add_argument("--w-acc", type=float, default=None)
    parser.add_argument("--w-style", type=float, default=None)
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--genre-map", type=str, default="config/genre_to_id.json")
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument("--debug-every", type=int, default=50)
    # stats paths are taken from config (separate per dataset)
    args = parser.parse_args()

    config = load_config()
    print("Config loaded")
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    aist_genres = config["aist_genres"]
    num_styles = config["num_styles"]
    d_in = config["d_in"]
    d_z = config["d_z"]
    seq_len = args.seq_len or config["seq_len"]
    batch_size = args.batch_size or config["train_batch_size"]
    num_workers = config["num_workers"]
    pin_memory = config["pin_memory"]
    prefetch_factor = config["prefetch_factor"]
    persistent_workers = config["persistent_workers"]
    val_every_epochs = config["val_every_epochs"]
    eval_batch_size = config["eval_batch_size"]
    kl_warmup = args.kl_warmup or config["kl_warmup_steps"]
    kl_weight_target = args.kl_weight or config["kl_target_weight"]
    w_vel = args.w_vel or config["w_vel"]
    w_acc = args.w_acc or config["w_acc"]
    w_style = args.w_style or config["w_style"]
    w_contact = args.w_contact or config["w_contact"]
    style_dropout_p = config["style_dropout_p"]
    stats_path = config["stats_path"]
    cache_root = config["cache_root"]
    aist_split_train = config["aist_split_train"]
    mvh_split_train = config["mvh_split_train"]
    grad_clip_norm = config["grad_clip_norm"]

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

    print("Loading dataset splits")
    aist_train_paths = aist_split_paths(aist_dir, aist_split_train)
    mvh_train_dirs = read_lines(mvh_split_train)

    if not os.path.exists(stats_path):
        print("Computing mean/std (training splits)")
        compute_mean_std_from_splits(
            aist_train_paths,
            mvh_train_dirs,
            stats_path,
            workers=10,
        )

    print("Loading mean/std")
    mean, std = load_mean_std(stats_path)

    print("Building AIST++ datasets")
    dataset_a = AISTDataset(
        aist_dir,
        genre_to_id,
        seq_len,
        mean=mean,
        std=std,
        files=aist_train_paths,
        cache_root=cache_root,
    )

    loader_a = DataLoader(
        dataset_a,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    print("Starting training loop")

    model = MotionVAE(d_in=d_in, d_z=d_z, num_styles=num_styles, max_len=seq_len)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    best_val = None
    best_epoch = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
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
            cache_root=cache_root,
        )
        loader_b = DataLoader(
            dataset_b,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

        batch_iter = balanced_batch_iter(
            loader_a, loader_b, args.ratio_aist, args.ratio_mvh
        )

        num_steps = min(len(loader_a), len(loader_b))
        if num_steps == 0:
            raise ValueError(
                "No training steps available; check batch size and split sizes."
            )

        recon_sum = 0.0
        recon_count = 0
        for step_idx in tqdm(range(num_steps), desc="Training", leave=False):
            t0 = time.perf_counter()
            batches = next(batch_iter)
            t1 = time.perf_counter()
            motion, domain_id, style_id, mask = merge_batches(batches)
            motion = motion.to(args.device)
            domain_id = domain_id.to(args.device)
            style_id = style_id.to(args.device)
            mask = mask.to(args.device)
            if not torch.isfinite(motion).all():
                print("Warning: non-finite motion batch; skipping")
                continue
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            style_id_in = apply_style_dropout(style_id, domain_id, style_dropout_p)
            outputs = model(motion, domain_id, style_id_in, mask=mask)
            x_hat = outputs["x_hat"]
            if not torch.isfinite(x_hat).all():
                print("Warning: non-finite model output; skipping")
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    latest_path = os.path.join(args.checkpoint_dir, "motion_vae_latest.pt")
                    if os.path.exists(latest_path):
                        print("Reloading latest checkpoint after NaNs")
                        state = torch.load(latest_path, map_location=args.device)
                        model.load_state_dict(state["model"])
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise ValueError("Model parameters contain NaNs and no checkpoint to recover")
                continue
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            recon, cont_loss, contact_loss = grouped_recon_loss(
                x_hat, motion, mask, w_contact=w_contact
            )
            kl = masked_kl(outputs["mu"], outputs["logvar"], mask)
            vel, acc = smoothness_loss(x_hat, motion, mask)
            style_loss = style_ce_loss(
                outputs.get("style_logits"), style_id_in, domain_id
            )

            kld_weight = kl_weight(step, kl_warmup, kl_weight_target)
            loss = recon + kld_weight * kl + w_vel * vel + w_acc * acc
            if style_loss is not None:
                loss = loss + w_style * style_loss
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t4 = time.perf_counter()
            if not torch.isfinite(loss):
                print("Warning: non-finite loss; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t5 = time.perf_counter()

            step += 1
            recon_sum += recon.item()
            recon_count += 1

            if args.debug_timing and (step_idx % args.debug_every == 0):
                print(
                    "timing (s) load={:.4f} to_gpu={:.4f} fwd={:.4f} loss={:.4f} bwd_step={:.4f}".format(
                        t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4
                    )
                )

        avg_recon = recon_sum / max(recon_count, 1)
        print(f"Epoch {epoch + 1} avg_recon={avg_recon:.6f}")

        save_ckpt = (epoch + 1) % val_every_epochs == 0
        if save_ckpt:
            print("Running validation")
            model.eval()
            aist_val_paths = aist_split_paths(
                aist_dir, config["aist_split_val"]
            )
            mvh_val_dirs = read_lines(config["mvh_split_val"])
            val_a = AISTDataset(
                aist_dir,
                genre_to_id,
                seq_len,
                mean=mean,
                std=std,
                files=aist_val_paths,
                cache_root=cache_root,
            )
            val_b = MVHumanNetDataset(
                mv_root,
                seq_len,
                mean=mean,
                std=std,
                sequence_dirs=mvh_val_dirs,
                cache_root=cache_root,
            )
            val_loader_a = DataLoader(
                val_a,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
            val_loader_b = DataLoader(
                val_b,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
            val_recon_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for loader in (val_loader_a, val_loader_b):
                    for batch in loader:
                        motion = batch["motion"].to(args.device)
                        domain_id = batch["domain_id"].to(args.device)
                        style_id = batch["style_id"].to(args.device)
                        mask = batch["mask"].to(args.device)
                        outputs = model(motion, domain_id, style_id, mask=mask)
                        v_recon, _, _ = grouped_recon_loss(
                            outputs["x_hat"], motion, mask, w_contact=w_contact
                        )
                        val_recon_sum += v_recon.item()
                        val_count += 1
            val_recon = val_recon_sum / max(val_count, 1)
            print(f"Validation recon={val_recon:.6f}")
            model.train()

        if save_ckpt:
            latest_path = os.path.join(args.checkpoint_dir, "motion_vae_latest.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "genre_to_id": genre_to_id,
                    "config": vars(args),
                    "epoch": epoch + 1,
                },
                latest_path,
            )
            print(f"Saved checkpoint: {latest_path}")

            if val_count > 0 and (best_val is None or val_recon < best_val):
                best_val = val_recon
                best_epoch = epoch + 1
                best_path = os.path.join(args.checkpoint_dir, "motion_vae_best.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "genre_to_id": genre_to_id,
                        "config": vars(args),
                        "epoch": best_epoch,
                        "best_val": best_val,
                    },
                    best_path,
                )
                print(f"Saved best checkpoint: {best_path} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
