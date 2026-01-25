import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
torch.multiprocessing.set_sharing_strategy("file_system")

from flowmimic.src.config.config import load_config
from flowmimic.src.model.vae.datasets.dataset_aist import AISTDataset
from flowmimic.src.model.vae.datasets.dataset_mvh import MVHumanNetDataset
from flowmimic.src.model.vae.losses import (
    grouped_recon_loss,
    masked_kl,
    style_ce_loss,
    LAYOUT_SLICES,
)
from flowmimic.src.model.vae.motion_vae import MotionVAE
from flowmimic.src.model.vae.stats import load_mean_std


def run_eval(loader, model, device, mean, std, w_contact):
    model.eval()
    total_recon = 0.0
    total_kl = 0.0
    total_style = 0.0
    total_batches = 0
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            domain_id = batch["domain_id"].to(device)
            style_id = batch["style_id"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(motion, domain_id, style_id, mask=mask)
            recon, _cont, _contact = grouped_recon_loss(
                outputs["x_hat"], motion, mask, w_contact=w_contact
            )
            kl = masked_kl(outputs["mu"], outputs["logvar"], mask)
            style_loss = style_ce_loss(outputs.get("style_logits"), style_id, domain_id)

            cont_end = LAYOUT_SLICES["feet_contact"][0]
            x_hat = outputs["x_hat"]
            x_hat_denorm = x_hat.clone()
            x_hat_denorm[..., :cont_end] = x_hat_denorm[..., :cont_end] * std_t + mean_t
            contact_logits = x_hat_denorm[..., cont_end:]
            _ = (torch.sigmoid(contact_logits) > 0.5).float()

            total_recon += recon.item()
            total_kl += kl.item()
            total_style += style_loss.item() if style_loss is not None else 0.0
            total_batches += 1

    if total_batches == 0:
        return {"recon": 0.0, "kl": 0.0, "style": 0.0}

    return {
        "recon": total_recon / total_batches,
        "kl": total_kl / total_batches,
        "style": total_style / total_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--genre-map", type=str, default="flowmimic/src/config/genre_to_id.json"
    )
    # stats paths are taken from config (separate per dataset)
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    seq_len = args.seq_len or config["seq_len"]
    batch_size = args.batch_size or config["eval_batch_size"]
    num_workers = config["num_workers"]
    pin_memory = config["pin_memory"]
    persistent_workers = config["persistent_workers"]
    prefetch_factor = config["prefetch_factor"]
    stats_path = config["stats_path"]
    cache_root = config["cache_root"]
    aist_split_val = config["aist_split_val"]
    mvh_split_val = config["mvh_split_val"]

    if not os.path.exists(aist_split_val):
        raise FileNotFoundError(f"AIST split file not found: {aist_split_val}")
    if not os.path.exists(mvh_split_val):
        raise FileNotFoundError(f"MVHumanNet split file not found: {mvh_split_val}")
    w_contact = config["w_contact"]

    if args.genre_map and os.path.exists(args.genre_map):
        with open(args.genre_map, "r", encoding="utf-8") as f:
            genre_to_id = json.load(f)
    else:
        genre_to_id = {"unknown": 0}

    num_styles = config["num_styles"]
    d_in = config["d_in"]
    d_z = config["d_z"]
    mean, std = load_mean_std(stats_path)

    def read_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def aist_split_paths(split_path):
        names = read_lines(split_path)
        return [os.path.join(aist_dir, f"{name}.pkl") for name in names]

    aist_val_paths = aist_split_paths(aist_split_val)
    mvh_val_dirs = read_lines(mvh_split_val)

    dataset_a = AISTDataset(
        aist_dir,
        genre_to_id,
        seq_len,
        mean=mean,
        std=std,
        files=aist_val_paths,
        cache_root=cache_root,
    )
    dataset_b = MVHumanNetDataset(
        mv_root,
        seq_len,
        mean=mean,
        std=std,
        sequence_dirs=mvh_val_dirs,
        cache_root=cache_root,
    )

    loader_a = DataLoader(
        dataset_a,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    loader_b = DataLoader(
        dataset_b,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    model = MotionVAE(d_in=d_in, d_z=d_z, num_styles=num_styles, max_len=seq_len)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.to(args.device)

    aist_metrics = run_eval(loader_a, model, args.device, mean, std, w_contact)
    mvh_metrics = run_eval(loader_b, model, args.device, mean, std, w_contact)

    print("AIST++ metrics", aist_metrics)
    print("MVHumanNet metrics", mvh_metrics)


if __name__ == "__main__":
    main()
