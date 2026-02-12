import argparse
import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flowmimic.src.config.config import load_config
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = load_config()
    flow_cfg = config.get("flow", {})
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
    state = torch.load(args.checkpoint, map_location=args.device)
    if args.use_ema and "ema" in state:
        flow.load_state_dict(state["ema"])
        label = "ema"
    else:
        flow.load_state_dict(state["model"])
        label = "model"

    bad = []
    for name, param in flow.state_dict().items():
        if not torch.isfinite(param).all():
            bad.append(name)

    if bad:
        print(f"[FAIL] non-finite params in {label}: {len(bad)}")
        for name in bad[:20]:
            print(f"  {name}")
    else:
        print(f"[OK] all params finite in {label}")


if __name__ == "__main__":
    main()
