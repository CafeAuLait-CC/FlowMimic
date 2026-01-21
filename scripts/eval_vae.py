import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from models.vae.datasets.dataset_aist import AISTDataset
from models.vae.datasets.dataset_mvh import MVHumanNetDataset
from models.vae.losses import masked_kl, masked_smooth_l1, style_ce_loss
from models.vae.motion_vae import MotionVAE
from utils.config import load_config


def run_eval(loader, model, device):
    model.eval()
    total_recon = 0.0
    total_kl = 0.0
    total_style = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            domain_id = batch["domain_id"].to(device)
            style_id = batch["style_id"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(motion, domain_id, style_id, mask=mask)
            recon = masked_smooth_l1(outputs["x_hat"], motion, mask)
            kl = masked_kl(outputs["mu"], outputs["logvar"], mask)
            style_loss = style_ce_loss(outputs.get("style_logits"), style_id, domain_id)

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
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--genre-map", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]

    if args.genre_map and os.path.exists(args.genre_map):
        with open(args.genre_map, "r", encoding="utf-8") as f:
            genre_to_id = json.load(f)
    else:
        genre_to_id = {"unknown": 0}

    num_styles = max(genre_to_id.values()) + 1

    dataset_a = AISTDataset(aist_dir, genre_to_id, args.seq_len)
    dataset_b = MVHumanNetDataset(mv_root, args.seq_len)

    loader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=False)
    loader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=False)

    d_in = 22 * 3
    model = MotionVAE(d_in=d_in, num_styles=num_styles)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.to(args.device)

    aist_metrics = run_eval(loader_a, model, args.device)
    mvh_metrics = run_eval(loader_b, model, args.device)

    print("AIST++ metrics", aist_metrics)
    print("MVHumanNet metrics", mvh_metrics)


if __name__ == "__main__":
    main()
