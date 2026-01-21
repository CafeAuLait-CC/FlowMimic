import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from models.vae.datasets.dataset_aist import AISTDataset
from models.vae.datasets.dataset_mvh import MVHumanNetDataset
from models.vae.losses import grouped_recon_loss, masked_kl, style_ce_loss, LAYOUT_SLICES
from models.vae.motion_vae import MotionVAE
from models.vae.stats import load_mean_std
from utils.config import load_config


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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--genre-map", type=str, default=None)
    # stats paths are taken from config (separate per dataset)
    args = parser.parse_args()

    config = load_config()
    aist_dir = config["aist_motions_dir"]
    mv_root = config["mvhumannet_root"]
    seq_len = args.seq_len or config["seq_len"]
    stats_path_aist = config["stats_path_aist"]
    stats_path_mvh = config["stats_path_mvh"]
    w_contact = config["w_contact"]

    if args.genre_map and os.path.exists(args.genre_map):
        with open(args.genre_map, "r", encoding="utf-8") as f:
            genre_to_id = json.load(f)
    else:
        genre_to_id = {"unknown": 0}

    num_styles = config["num_styles"]
    d_in = config["d_in"]
    d_z = config["d_z"]
    mean_a, std_a = load_mean_std(stats_path_aist)
    mean_b, std_b = load_mean_std(stats_path_mvh)

    dataset_a = AISTDataset(aist_dir, genre_to_id, seq_len, mean=mean_a, std=std_a)
    dataset_b = MVHumanNetDataset(mv_root, seq_len, mean=mean_b, std=std_b)

    loader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=False)
    loader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=False)

    model = MotionVAE(d_in=d_in, d_z=d_z, num_styles=num_styles, max_len=seq_len)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.to(args.device)

    aist_metrics = run_eval(loader_a, model, args.device, mean_a, std_a, w_contact)
    mvh_metrics = run_eval(loader_b, model, args.device, mean_b, std_b, w_contact)

    print("AIST++ metrics", aist_metrics)
    print("MVHumanNet metrics", mvh_metrics)


if __name__ == "__main__":
    main()
