#!/usr/bin/env python3
"""Run a canonical three-replication AIST evaluation for the MLD checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/aist/mld_eval_aist.yaml")
    parser.add_argument("--cfg-assets", default="configs/aist/mld_assets_aist.yaml")
    parser.add_argument(
        "--ckpt",
        default="runs/mld/mld/aist_ik263_mld_196/checkpoints/epoch=2999.ckpt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--diversity-times", type=int, default=300)
    parser.add_argument(
        "--output", default="training_logs/baseline_eval_corrected_20260715/mld.json"
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[2]
    mld_root = workspace / "motion-latent-diffusion"
    cfg_path = (workspace / args.cfg).resolve()
    assets_path = (workspace / args.cfg_assets).resolve()
    ckpt_path = (workspace / args.ckpt).resolve()
    output_path = (workspace / args.output).resolve()

    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(mld_root))
    os.chdir(mld_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import pytorch_lightning as pl

    from flowmimic.src.metrics.distribution_metrics import (
        summarize_motion_feature_metrics,
    )
    from flowmimic.src.metrics.motion_quality_metrics import (
        aggregate_replications,
        summarize_physical_motion,
    )
    from mld.config import get_module_config
    from mld.data.get_data import get_datasets
    from mld.data.humanml.scripts.motion_process import recover_from_ric
    from mld.models.get_model import get_model

    cfg_base = OmegaConf.load(mld_root / "configs" / "base.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, OmegaConf.load(assets_path))
    cfg.TEST.CHECKPOINTS = str(ckpt_path)
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.TEST.NUM_WORKERS = args.workers
    cfg.TEST.REPLICATION_TIMES = args.replications
    cfg.ACCELERATOR = "gpu" if args.device.startswith("cuda") else "cpu"
    cfg.DEVICE = [0]
    cfg.DEBUG = False
    cfg.TIME = "canonical_aist_rep3"

    _set_seed(args.seed)
    datamodule = get_datasets(cfg, phase="test")[0]
    # Prepared AIST test motions are exactly the first 196 frames. Prevent the
    # HumanML loader from randomly shortening one third of them to 192 frames.
    datamodule.test_dataset.unit_length = 196

    model = get_model(cfg, datamodule)
    evaluator_states = {
        name: copy.deepcopy(getattr(model, name).state_dict())
        for name in ("t2m_textencoder", "t2m_moveencoder", "t2m_motionencoder")
    }
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # MLD checkpoints contain the evaluator used during training. Restore the
    # configured AIST evaluator after loading so HumanML3D weights cannot leak in.
    for name, state in evaluator_states.items():
        getattr(model, name).load_state_dict(state, strict=True)

    captured: list[dict[str, torch.Tensor]] = []
    original_t2m_eval = model.t2m_eval

    def capture_t2m_eval(batch):
        result = original_t2m_eval(batch)
        if not model.trainer.datamodule.is_mm:
            captured.append(
                {
                    key: result[key].detach().cpu()
                    for key in ("m_rst", "m_ref", "lat_rm", "lat_m")
                }
            )
        return result

    model.t2m_eval = capture_t2m_eval
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    eval_mean = np.asarray(datamodule.hparams.mean_eval, dtype=np.float32)
    eval_std = np.asarray(datamodule.hparams.std_eval, dtype=np.float32)
    evaluator_root = Path(str(cfg.model.t2m_path)) / "t2m"
    evaluator_checkpoint = evaluator_root / "text_mot_match" / "model" / "finest.tar"
    evaluator_mean_path = evaluator_root / "Comp_v6_KLD01" / "meta" / "mean.npy"
    evaluator_std_path = evaluator_root / "Comp_v6_KLD01" / "meta" / "std.npy"
    rows = []
    for replication in range(args.replications):
        seed = args.seed + replication
        _set_seed(seed)
        pl.seed_everything(seed, workers=True)
        captured.clear()
        trainer.test(model, datamodule=datamodule, verbose=False)

        generated_eval = torch.cat([item["m_rst"] for item in captured]).numpy()
        reference_eval = torch.cat([item["m_ref"] for item in captured]).numpy()
        generated_features = torch.cat([item["lat_rm"] for item in captured]).numpy()
        reference_features = torch.cat([item["lat_m"] for item in captured]).numpy()
        if generated_eval.shape != (470, 196, 263):
            raise ValueError(f"Unexpected MLD evaluation shape: {generated_eval.shape}")

        np.random.seed(seed + 100_000)
        row = summarize_motion_feature_metrics(
            generated_features,
            reference_features,
            diversity_times=args.diversity_times,
        )
        generated_raw = generated_eval * eval_std + eval_mean
        reference_raw = reference_eval * eval_std + eval_mean
        generated_joints = recover_from_ric(
            torch.from_numpy(generated_raw), joints_num=22
        ).numpy()
        reference_joints = recover_from_ric(
            torch.from_numpy(reference_raw), joints_num=22
        ).numpy()
        row.update(
            summarize_physical_motion(
                generated_joints,
                reference_joints,
                generated_raw[..., -4:],
                args.fps,
            )
        )
        row.update({"replication": replication, "seed": seed})
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "method": "MLD",
        "summary": aggregate_replications(rows),
        "replications": rows,
        "protocol": {
            "split": "AIST++ official test",
            "samples_per_replication": 470,
            "crop": "first 196 frames",
            "replication_seeds": [args.seed + i for i in range(args.replications)],
            "checkpoint": str(ckpt_path),
            "evaluator_checkpoint": str(evaluator_checkpoint),
            "evaluator_mean": str(evaluator_mean_path),
            "evaluator_std": str(evaluator_std_path),
            "normalization": "MLD -> physical IK263 -> AIST T2M",
            "condition": "one randomly selected caption per motion",
            "fps": args.fps,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
