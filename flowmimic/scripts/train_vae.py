import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
torch.multiprocessing.set_sharing_strategy("file_system")

from flowmimic.src.config.config import load_config
from flowmimic.src.model.vae.datasets.balanced_batch_sampler import balanced_batch_iter
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.datasets.label_map_builder import (
    build_genre_to_id,
    save_genre_to_id,
)
from flowmimic.src.model.vae.losses import (
    continuous_smoothness_loss,
    grouped_recon_loss,
    masked_kl,
    style_ce_loss,
)
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import compute_mean_std_from_splits, load_mean_std


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


def _ramp_weight(step, total_steps, target, warmup_frac):
    if total_steps <= 0:
        return target
    progress = step / float(total_steps)
    if progress <= warmup_frac:
        return 0.0
    ramp = (progress - warmup_frac) / max(1e-8, 1.0 - warmup_frac)
    return target * min(1.0, ramp)


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


def main_legacy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-len", type=int, default=None)
    parser.add_argument(
        "--latent-token-mode",
        choices=("pool", "query"),
        default="pool",
        help="How compact latent tokens are formed when --latent-len is set.",
    )
    parser.add_argument(
        "--val-datasets",
        type=str,
        default=None,
        help="Comma-separated validation datasets used for checkpoint selection; defaults to --datasets.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ratio-aist", type=int, default=1)
    parser.add_argument("--ratio-mvh", type=int, default=1)
    parser.add_argument("--kl-warmup", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--w-vel", type=float, default=None)
    parser.add_argument("--w-acc", type=float, default=None)
    parser.add_argument("--smooth-warmup-frac", type=float, default=0.2)
    parser.add_argument("--w-style", type=float, default=None)
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--genre-map", type=str, default="flowmimic/src/config/genre_to_id.json"
    )
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument("--debug-every", type=int, default=50)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--finetune-decoder", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="FlowMimic")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=("allow", "must", "never"),
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
    )
    # stats paths are taken from config (separate per dataset)
    args = parser.parse_args()

    config = load_config()
    print("Config loaded")
    wandb_run = None
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError:
        wandb = None
        print("wandb not installed; continuing without logging.")
    if wandb is not None:
        tags = [t for t in (args.wandb_tags or "").split(",") if t]
        run_name = args.wandb_name
        if run_name is None:
            stamp = datetime.now().strftime("%y%m%d-%H%M%S")
            run_name = f"vae-{stamp}"
        run_group = args.wandb_group or "VAE"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=run_group,
            id=args.wandb_id,
            resume=args.wandb_resume if args.wandb_id else None,
            tags=tags or None,
            mode=args.wandb_mode,
            config={
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "kl_warmup": args.kl_warmup,
                "kl_weight": args.kl_weight,
                "w_vel": args.w_vel,
                "w_acc": args.w_acc,
                "smooth_warmup_frac": args.smooth_warmup_frac,
                "w_style": args.w_style,
                "w_contact": args.w_contact,
                "resume_ckpt": args.resume_ckpt,
                "finetune_decoder": args.finetune_decoder,
            },
        )
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
    w_root = config.get("w_root", 1.0)
    w_root_late_start = config.get("w_root_late_start", 1.0)
    w_root_late_factor = config.get("w_root_late_factor", 1.0)
    style_dropout_p = config["style_dropout_p"]
    stats_path = config["stats_path"]
    smooth_warmup_frac = args.smooth_warmup_frac
    target_fps = config.get("target_fps", 30)
    aist_fps = config.get("aist_fps", 60)
    mvh_fps = config.get("mvh_fps", 5)
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
            target_fps=target_fps,
            aist_fps=aist_fps,
            mvh_fps=mvh_fps,
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
        target_fps=target_fps,
        src_fps=aist_fps,
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
    if args.finetune_decoder and not args.resume_ckpt:
        raise ValueError("--finetune-decoder requires --resume-ckpt")
    if args.resume_ckpt:
        state = torch.load(args.resume_ckpt, map_location=args.device)
        model.load_state_dict(state["model"])
    model.to(args.device)

    if args.finetune_decoder:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = True
        for p in model.dec_in.parameters():
            p.requires_grad = True
        model.dec_pos.requires_grad = True
        for p in model.to_out.parameters():
            p.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    total_steps_est = 1
    best_val = None
    best_epoch = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_frac = (epoch + 1) / max(args.epochs, 1)
        w_root_epoch = w_root * (w_root_late_factor if epoch_frac >= w_root_late_start else 1.0)
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
            target_fps=target_fps,
            src_fps=mvh_fps,
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
        if epoch == 0:
            total_steps_est = max(1, args.epochs * num_steps)
        if num_steps == 0:
            raise ValueError(
                "No training steps available; check batch size and split sizes."
            )

        recon_sum = 0.0
        cont_sum = 0.0
        contact_sum = 0.0
        root_sum = 0.0
        kl_sum = 0.0
        vel_sum = 0.0
        acc_sum = 0.0
        style_sum = 0.0
        total_sum = 0.0
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
                if args.debug_timing:
                    print("Warning: non-finite motion batch; skipping")
                continue
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            style_id_in = apply_style_dropout(style_id, domain_id, style_dropout_p)
            outputs = model(motion, domain_id, style_id_in, mask=mask)
            x_hat = outputs["x_hat"]
            if not torch.isfinite(x_hat).all():
                if args.debug_timing:
                    print("Warning: non-finite model output; skipping")
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    latest_path = os.path.join(args.checkpoint_dir, "motion_vae_latest.pt")
                    if os.path.exists(latest_path):
                        if args.debug_timing:
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

            recon, cont_loss, contact_loss, root_loss = grouped_recon_loss(
                x_hat, motion, mask, w_contact=w_contact, w_root=w_root_epoch
            )
            kl = masked_kl(outputs["mu"], outputs["logvar"], mask)
            vel, acc = continuous_smoothness_loss(x_hat, motion, mask)
            style_loss = style_ce_loss(
                outputs.get("style_logits"), style_id_in, domain_id
            )

            kld_weight = kl_weight(step, kl_warmup, kl_weight_target)
            vel_w = _ramp_weight(step, total_steps_est, w_vel, smooth_warmup_frac)
            acc_w = _ramp_weight(step, total_steps_est, w_acc, smooth_warmup_frac)
            loss = recon + kld_weight * kl + vel_w * vel + acc_w * acc
            if style_loss is not None:
                loss = loss + w_style * style_loss
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t4 = time.perf_counter()
            if not torch.isfinite(loss):
                if args.debug_timing:
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
            cont_sum += cont_loss.item()
            contact_sum += contact_loss.item()
            root_sum += root_loss.item()
            kl_sum += kl.item()
            vel_sum += vel.item()
            acc_sum += acc.item()
            style_sum += style_loss.item() if style_loss is not None else 0.0
            total_sum += loss.item()
            recon_count += 1

            if args.debug_timing and (step_idx % args.debug_every == 0):
                print(
                    "timing (s) load={:.4f} to_gpu={:.4f} fwd={:.4f} loss={:.4f} bwd_step={:.4f}".format(
                        t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4
                    )
                )

        denom = max(recon_count, 1)
        avg_recon = recon_sum / denom
        avg_cont = cont_sum / denom
        avg_contact = contact_sum / denom
        avg_root = root_sum / denom
        avg_kl = kl_sum / denom
        avg_vel = vel_sum / denom
        avg_acc = acc_sum / denom
        avg_style = style_sum / denom
        avg_total = total_sum / denom
        print(
            "Epoch {} loss_total={:.6f} recon={:.6f} cont={:.6f} contact={:.6f} "
            "root={:.6f} kl={:.6f} vel={:.6f} acc={:.6f} style={:.6f}".format(
                epoch + 1,
                avg_total,
                avg_recon,
                avg_cont,
                avg_contact,
                avg_root,
                avg_kl,
                avg_vel,
                avg_acc,
                avg_style,
            )
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "loss/total": avg_total,
                    "loss/recon": avg_recon,
                    "loss/cont": avg_cont,
                    "loss/contact": avg_contact,
                    "loss/root": avg_root,
                    "loss/kl": avg_kl,
                    "loss/vel": avg_vel,
                    "loss/acc": avg_acc,
                    "loss/style": avg_style,
                },
                step=epoch + 1,
            )

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
                target_fps=target_fps,
                src_fps=aist_fps,
            )
            val_b = MVHumanNetDataset(
                mv_root,
                seq_len,
                mean=mean,
                std=std,
                sequence_dirs=mvh_val_dirs,
                cache_root=cache_root,
                target_fps=target_fps,
                src_fps=mvh_fps,
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
                        v_recon, _, _, _ = grouped_recon_loss(
                            outputs["x_hat"], motion, mask, w_contact=w_contact, w_root=w_root
                        )
                        val_recon_sum += v_recon.item()
                        val_count += 1
            val_recon = val_recon_sum / max(val_count, 1)
            print(f"Validation recon={val_recon:.6f}")
            if wandb_run is not None:
                wandb_run.log(
                    {"val/recon": val_recon},
                    step=epoch + 1,
                )
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


def _parse_dataset_names(value):
    names = {name.strip().upper() for name in value.split(",") if name.strip()}
    valid = {"AIST", "MVH"}
    unknown = names - valid
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")
    if not names:
        raise ValueError("At least one dataset must be selected")
    return names


def _single_loader_iter(loader):
    while True:
        for batch in loader:
            yield [batch]


def _ddp_any(ddp, value, device):
    if not ddp:
        return bool(value)
    flag = torch.tensor([1 if value else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _seed_worker(base_seed):
    def _inner(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="AIST")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-len", type=int, default=None)
    parser.add_argument(
        "--latent-token-mode",
        choices=("pool", "query"),
        default="pool",
        help="How compact latent tokens are formed when --latent-len is set.",
    )
    parser.add_argument(
        "--val-datasets",
        type=str,
        default=None,
        help="Comma-separated validation datasets used for checkpoint selection; defaults to --datasets.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--ratio-aist", type=int, default=1)
    parser.add_argument("--ratio-mvh", type=int, default=1)
    parser.add_argument(
        "--aist-crop-mode", choices=("first", "random", "uniform"), default="first"
    )
    parser.add_argument(
        "--aist-val-crop-mode",
        choices=("first", "random", "uniform"),
        default="first",
        help="AIST crop mode used for validation/checkpoint selection.",
    )
    parser.add_argument("--aist-clip-repeat", type=int, default=1)
    parser.add_argument(
        "--aist-val-clip-repeat",
        type=int,
        default=1,
        help="Deterministic validation crop repeat count when using uniform crops.",
    )
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--val-every-epochs", type=int, default=None)
    parser.add_argument("--kl-warmup", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--w-vel", type=float, default=None)
    parser.add_argument("--w-acc", type=float, default=None)
    parser.add_argument("--smooth-warmup-frac", type=float, default=0.2)
    parser.add_argument("--w-style", type=float, default=None)
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--genre-map", type=str, default="flowmimic/src/config/genre_to_id.json"
    )
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument("--debug-every", type=int, default=50)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument(
        "--reset-best-val",
        action="store_true",
        help="Ignore best_val/best_epoch from a resumed checkpoint.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many validation checks without improvement; 0 disables.",
    )
    parser.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=0,
        help="Do not early-stop before this absolute epoch number.",
    )
    parser.add_argument("--finetune-decoder", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="FlowMimic")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=("allow", "must", "never"),
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
    )
    args = parser.parse_args()

    ddp = args.ddp
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("--ddp requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device(args.device)
        rank = 0
        world_size = 1
    is_main = rank == 0
    args.device = str(device)

    try:
        config = load_config()
        datasets = _parse_dataset_names(args.datasets)
        val_datasets = (
            _parse_dataset_names(args.val_datasets)
            if args.val_datasets is not None
            else set(datasets)
        )
        use_aist = "AIST" in datasets
        use_mvh = "MVH" in datasets
        val_use_aist = "AIST" in val_datasets
        val_use_mvh = "MVH" in val_datasets

        seed = config.get("seed", 42) + rank
        random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        wandb_run = None
        if is_main:
            print("Config loaded")
            try:
                import wandb  # type: ignore
            except ModuleNotFoundError:
                wandb = None
                print("wandb not installed; continuing without logging.")
            if wandb is not None:
                tags = [t for t in (args.wandb_tags or "").split(",") if t]
                run_name = args.wandb_name
                if run_name is None:
                    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
                    run_name = f"vae-{stamp}"
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    group=args.wandb_group or "VAE",
                    id=args.wandb_id,
                    resume=args.wandb_resume if args.wandb_id else None,
                    tags=tags or None,
                    mode=args.wandb_mode,
                    config={
                        **vars(args),
                        "world_size": world_size,
                        "selected_datasets": sorted(datasets),
                        "selected_val_datasets": sorted(val_datasets),
                    },
                )

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
        val_every_epochs = args.val_every_epochs or config["val_every_epochs"]
        eval_batch_size = config["eval_batch_size"]
        kl_warmup = args.kl_warmup or config["kl_warmup_steps"]
        kl_weight_target = args.kl_weight or config["kl_target_weight"]
        w_vel = args.w_vel or config["w_vel"]
        w_acc = args.w_acc or config["w_acc"]
        w_style = args.w_style or config["w_style"]
        w_contact = args.w_contact or config["w_contact"]
        w_root = config.get("w_root", 1.0)
        w_root_late_start = config.get("w_root_late_start", 1.0)
        w_root_late_factor = config.get("w_root_late_factor", 1.0)
        style_dropout_p = config["style_dropout_p"]
        stats_path = args.stats_path or config["stats_path"]
        smooth_warmup_frac = args.smooth_warmup_frac
        target_fps = config.get("target_fps", 30)
        aist_fps = config.get("aist_fps", 60)
        mvh_fps = config.get("mvh_fps", 5)
        cache_root = config["cache_root"]
        aist_split_train = config["aist_split_train"]
        mvh_split_train = config["mvh_split_train"]
        grad_clip_norm = config["grad_clip_norm"]

        if use_aist and not os.path.exists(aist_split_train):
            raise FileNotFoundError(f"AIST split file not found: {aist_split_train}")
        if use_mvh and not os.path.exists(mvh_split_train):
            raise FileNotFoundError(f"MVHumanNet split file not found: {mvh_split_train}")

        if is_main and not os.path.exists(args.genre_map):
            genre_to_id_init = build_genre_to_id(aist_genres)
            save_genre_to_id(genre_to_id_init, args.genre_map)
        if ddp:
            dist.barrier()
        with open(args.genre_map, "r", encoding="utf-8") as f:
            genre_to_id = json.load(f)

        if is_main:
            print(
                f"Training VAE datasets={sorted(datasets)} seq_len={seq_len} "
                f"latent_len={args.latent_len or seq_len} "
                f"latent_token_mode={args.latent_token_mode} "
                f"val_datasets={sorted(val_datasets)} "
                f"per_gpu_batch={batch_size} global_batch={batch_size * world_size}"
            )
        aist_train_paths = aist_split_paths(aist_dir, aist_split_train) if use_aist else []
        mvh_train_dirs = read_lines(mvh_split_train) if use_mvh else []

        if not os.path.exists(stats_path):
            if is_main:
                print(f"Computing mean/std: {stats_path}")
                compute_mean_std_from_splits(
                    aist_train_paths,
                    mvh_train_dirs,
                    stats_path,
                    workers=10,
                    target_fps=target_fps,
                    aist_fps=aist_fps,
                    mvh_fps=mvh_fps,
                )
            if ddp:
                dist.barrier()
        mean, std = load_mean_std(stats_path)

        loader_a = None
        sampler_a = None
        if use_aist:
            if is_main:
                print(f"Building AIST++ train dataset ({len(aist_train_paths)} files)")
            dataset_a = AISTDataset(
                aist_dir,
                genre_to_id,
                seq_len,
                mean=mean,
                std=std,
                files=aist_train_paths,
                cache_root=cache_root,
                target_fps=target_fps,
                src_fps=aist_fps,
                crop_mode=args.aist_crop_mode,
                clip_repeat=args.aist_clip_repeat,
            )
            sampler_a = (
                DistributedSampler(dataset_a, shuffle=True, drop_last=True)
                if ddp
                else None
            )
            loader_a = DataLoader(
                dataset_a,
                batch_size=batch_size,
                shuffle=sampler_a is None,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                sampler=sampler_a,
                worker_init_fn=_seed_worker(seed),
            )

        loader_b = None
        sampler_b = None
        if use_mvh:
            if is_main:
                print(f"Building MVHumanNet train dataset ({len(mvh_train_dirs)} dirs)")
            dataset_b = MVHumanNetDataset(
                mv_root,
                seq_len,
                mean=mean,
                std=std,
                sequence_dirs=mvh_train_dirs,
                cache_root=cache_root,
                target_fps=target_fps,
                src_fps=mvh_fps,
            )
            sampler_b = (
                DistributedSampler(dataset_b, shuffle=True, drop_last=True)
                if ddp
                else None
            )
            loader_b = DataLoader(
                dataset_b,
                batch_size=batch_size,
                shuffle=sampler_b is None,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                sampler=sampler_b,
                worker_init_fn=_seed_worker(seed + 10000),
            )

        if is_main:
            print("Starting training loop")
        model = MotionVAE(
            d_in=d_in,
            d_z=d_z,
            num_styles=num_styles,
            max_len=seq_len,
            latent_len=args.latent_len,
            latent_token_mode=args.latent_token_mode,
        )
        if args.finetune_decoder and not args.resume_ckpt:
            raise ValueError("--finetune-decoder requires --resume-ckpt")
        resume_state = None
        if args.resume_ckpt:
            resume_state = torch.load(args.resume_ckpt, map_location=device)
            model.load_state_dict(resume_state["model"])
        model.to(device)

        if args.finetune_decoder:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.decoder.parameters():
                p.requires_grad = True
            for p in model.dec_in.parameters():
                p.requires_grad = True
            model.dec_pos.requires_grad = True
            for p in model.to_out.parameters():
                p.requires_grad = True

        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        model_for_state = model.module if ddp else model
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=1e-2,
        )
        if resume_state is not None and "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])

        if is_main:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if ddp:
            dist.barrier()

        loader_lens = [len(x) for x in (loader_a, loader_b) if x is not None]
        num_steps_base = max(loader_lens) if len(loader_lens) > 1 else loader_lens[0]
        total_steps_est = max(1, args.epochs * num_steps_base)
        start_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
        step = start_epoch * num_steps_base
        best_val = resume_state.get("best_val") if resume_state else None
        best_epoch = resume_state.get("best_epoch") if resume_state else None
        if args.reset_best_val:
            best_val = None
            best_epoch = None
        if best_val is not None and best_epoch is None:
            best_epoch = start_epoch or None
        stale_val_checks = 0

        for epoch in range(start_epoch, args.epochs):
            if is_main:
                print(f"Epoch {epoch + 1}/{args.epochs}")
            model.train()
            if sampler_a is not None:
                sampler_a.set_epoch(epoch)
            if sampler_b is not None:
                sampler_b.set_epoch(epoch)
            if loader_a is not None and loader_b is not None:
                batch_iter = balanced_batch_iter(
                    loader_a, loader_b, args.ratio_aist, args.ratio_mvh
                )
                num_steps = max(len(loader_a), len(loader_b))
            elif loader_a is not None:
                batch_iter = _single_loader_iter(loader_a)
                num_steps = len(loader_a)
            elif loader_b is not None:
                batch_iter = _single_loader_iter(loader_b)
                num_steps = len(loader_b)
            else:
                raise ValueError("No train loaders were created")
            if num_steps == 0:
                raise ValueError("No training steps available; lower batch size.")

            epoch_frac = (epoch + 1) / max(args.epochs, 1)
            w_root_epoch = w_root * (
                w_root_late_factor if epoch_frac >= w_root_late_start else 1.0
            )
            recon_sum = cont_sum = contact_sum = root_sum = 0.0
            kl_sum = vel_sum = acc_sum = style_sum = total_sum = 0.0
            recon_count = 0
            iter_range = range(num_steps)
            if is_main:
                iter_range = tqdm(iter_range, desc="Training", leave=False)

            for step_idx in iter_range:
                t0 = time.perf_counter()
                batches = next(batch_iter)
                t1 = time.perf_counter()
                motion, domain_id, style_id, mask = merge_batches(batches)
                motion = motion.to(device)
                domain_id = domain_id.to(device)
                style_id = style_id.to(device)
                mask = mask.to(device)
                if _ddp_any(ddp, not torch.isfinite(motion).all().item(), device):
                    if args.debug_timing and is_main:
                        print("Warning: non-finite motion batch; skipping")
                    continue
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                style_id_in = apply_style_dropout(style_id, domain_id, style_dropout_p)
                outputs = model(motion, domain_id, style_id_in, mask=mask)
                x_hat = outputs["x_hat"]
                if _ddp_any(ddp, not torch.isfinite(x_hat).all().item(), device):
                    if args.debug_timing and is_main:
                        print("Warning: non-finite model output; skipping")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()

                recon, cont_loss, contact_loss, root_loss = grouped_recon_loss(
                    x_hat, motion, mask, w_contact=w_contact, w_root=w_root_epoch
                )
                kl = masked_kl(outputs["mu"], outputs["logvar"], mask)
                vel, acc = continuous_smoothness_loss(x_hat, motion, mask)
                style_loss = style_ce_loss(
                    outputs.get("style_logits"), style_id_in, domain_id
                )

                kld_weight = kl_weight(step, kl_warmup, kl_weight_target)
                vel_w = _ramp_weight(step, total_steps_est, w_vel, smooth_warmup_frac)
                acc_w = _ramp_weight(step, total_steps_est, w_acc, smooth_warmup_frac)
                loss = recon + kld_weight * kl + vel_w * vel + acc_w * acc
                if style_loss is not None:
                    loss = loss + w_style * style_loss
                if _ddp_any(ddp, not torch.isfinite(loss).item(), device):
                    if args.debug_timing and is_main:
                        print("Warning: non-finite loss; skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t4 = time.perf_counter()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t5 = time.perf_counter()

                step += 1
                recon_sum += recon.item()
                cont_sum += cont_loss.item()
                contact_sum += contact_loss.item()
                root_sum += root_loss.item()
                kl_sum += kl.item()
                vel_sum += vel.item()
                acc_sum += acc.item()
                style_sum += style_loss.item() if style_loss is not None else 0.0
                total_sum += loss.item()
                recon_count += 1

                if args.debug_timing and is_main and (step_idx % args.debug_every == 0):
                    print(
                        "timing (s) load={:.4f} to_gpu={:.4f} fwd={:.4f} "
                        "loss={:.4f} bwd_step={:.4f}".format(
                            t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4
                        )
                    )

            if ddp:
                sums = torch.tensor(
                    [
                        recon_sum,
                        cont_sum,
                        contact_sum,
                        root_sum,
                        kl_sum,
                        vel_sum,
                        acc_sum,
                        style_sum,
                        total_sum,
                        recon_count,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                (
                    recon_sum,
                    cont_sum,
                    contact_sum,
                    root_sum,
                    kl_sum,
                    vel_sum,
                    acc_sum,
                    style_sum,
                    total_sum,
                    recon_count,
                ) = sums.tolist()

            if is_main:
                denom = max(float(recon_count), 1.0)
                avg_recon = recon_sum / denom
                avg_cont = cont_sum / denom
                avg_contact = contact_sum / denom
                avg_root = root_sum / denom
                avg_kl = kl_sum / denom
                avg_vel = vel_sum / denom
                avg_acc = acc_sum / denom
                avg_style = style_sum / denom
                avg_total = total_sum / denom
                print(
                    "Epoch {} loss_total={:.6f} recon={:.6f} cont={:.6f} contact={:.6f} "
                    "root={:.6f} kl={:.6f} vel={:.6f} acc={:.6f} style={:.6f}".format(
                        epoch + 1,
                        avg_total,
                        avg_recon,
                        avg_cont,
                        avg_contact,
                        avg_root,
                        avg_kl,
                        avg_vel,
                        avg_acc,
                        avg_style,
                    )
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "loss/total": avg_total,
                            "loss/recon": avg_recon,
                            "loss/cont": avg_cont,
                            "loss/contact": avg_contact,
                            "loss/root": avg_root,
                            "loss/kl": avg_kl,
                            "loss/vel": avg_vel,
                            "loss/acc": avg_acc,
                            "loss/style": avg_style,
                        },
                        step=epoch + 1,
                    )

            save_ckpt = val_every_epochs and (epoch + 1) % val_every_epochs == 0
            stop_training = False
            if save_ckpt and is_main:
                print("Running validation")
                model_for_state.eval()
                val_loaders = []
                if val_use_aist:
                    aist_val_paths = aist_split_paths(aist_dir, config["aist_split_val"])
                    val_a = AISTDataset(
                        aist_dir,
                        genre_to_id,
                        seq_len,
                        mean=mean,
                        std=std,
                        files=aist_val_paths,
                        cache_root=cache_root,
                        target_fps=target_fps,
                        src_fps=aist_fps,
                        crop_mode=args.aist_val_crop_mode,
                        clip_repeat=args.aist_val_clip_repeat,
                    )
                    val_loaders.append(
                        DataLoader(
                            val_a,
                            batch_size=eval_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor
                            if num_workers > 0
                            else None,
                        )
                    )
                if val_use_mvh:
                    mvh_val_dirs = read_lines(config["mvh_split_val"])
                    val_b = MVHumanNetDataset(
                        mv_root,
                        seq_len,
                        mean=mean,
                        std=std,
                        sequence_dirs=mvh_val_dirs,
                        cache_root=cache_root,
                        target_fps=target_fps,
                        src_fps=mvh_fps,
                    )
                    val_loaders.append(
                        DataLoader(
                            val_b,
                            batch_size=eval_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor
                            if num_workers > 0
                            else None,
                        )
                    )
                val_recon_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for loader in val_loaders:
                        for batch in loader:
                            motion = batch["motion"].to(device)
                            domain_id = batch["domain_id"].to(device)
                            style_id = batch["style_id"].to(device)
                            mask = batch["mask"].to(device)
                            outputs = model_for_state(
                                motion, domain_id, style_id, mask=mask
                            )
                            v_recon, _, _, _ = grouped_recon_loss(
                                outputs["x_hat"],
                                motion,
                                mask,
                                w_contact=w_contact,
                                w_root=w_root,
                            )
                            val_recon_sum += v_recon.item()
                            val_count += 1
                val_recon = val_recon_sum / max(val_count, 1)
                print(f"Validation recon={val_recon:.6f}")
                if wandb_run is not None:
                    wandb_run.log({"val/recon": val_recon}, step=epoch + 1)
                latest_path = os.path.join(
                    args.checkpoint_dir, "motion_vae_latest.pt"
                )
                ckpt_state = {
                    "model": model_for_state.state_dict(),
                    "genre_to_id": genre_to_id,
                    "config": vars(args),
                    "epoch": epoch + 1,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "optimizer": optimizer.state_dict(),
                    "stats_path": stats_path,
                    "selected_datasets": sorted(datasets),
                    "selected_val_datasets": sorted(val_datasets),
                }
                torch.save(ckpt_state, latest_path)
                print(f"Saved checkpoint: {latest_path}")
                improved = val_count > 0 and (best_val is None or val_recon < best_val)
                if improved:
                    best_val = val_recon
                    best_epoch = epoch + 1
                    stale_val_checks = 0
                    best_path = os.path.join(
                        args.checkpoint_dir, "motion_vae_best.pt"
                    )
                    ckpt_state = dict(ckpt_state)
                    ckpt_state["best_val"] = best_val
                    ckpt_state["epoch"] = best_epoch
                    torch.save(ckpt_state, best_path)
                    print(
                        f"Saved best checkpoint: {best_path} (epoch {best_epoch})"
                    )
                elif val_count > 0:
                    stale_val_checks += 1
                    if args.early_stop_patience > 0:
                        print(
                            "Validation did not improve "
                            f"({stale_val_checks}/{args.early_stop_patience}); "
                            f"best={best_val:.6f} at epoch {best_epoch}"
                        )
                model_for_state.train()
                if (
                    args.early_stop_patience > 0
                    and stale_val_checks >= args.early_stop_patience
                    and epoch + 1 >= args.early_stop_min_epochs
                ):
                    print(
                        "Early stopping after "
                        f"{stale_val_checks} stale validation checks."
                    )
                    stop_training = True
            if ddp and save_ckpt:
                dist.barrier()
            if _ddp_any(ddp, stop_training, device):
                break

        if is_main and wandb_run is not None:
            wandb_run.finish()
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
