"""Evaluate the workspace StickMotion AIST checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_devices(gpu: str) -> list[int]:
    return [int(part) for part in gpu.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aist/stickmotion_remodiffuse_aist_eval.py")
    parser.add_argument("--ckpt", default="runs/stickmotion/human_ml3d/aist_remodiffuse/last.ckpt")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--diversity-times", type=int, default=300)
    parser.add_argument(
        "--guidance-repeat",
        type=int,
        default=None,
        help="Override inference-time locus guidance iterations per diffusion step.",
    )
    parser.add_argument(
        "--ablate-locus",
        action="store_true",
        help="Replace the full ground-truth root locus condition with zeros.",
    )
    parser.add_argument(
        "--ablate-stick",
        action="store_true",
        help="Replace ground-truth stickman sketches and their mask with zeros.",
    )
    parser.add_argument(
        "--condition-branch",
        choices=("schedule", "both", "text", "stick", "none"),
        default="schedule",
        help="Select one learned condition branch instead of the inference CFG schedule.",
    )
    parser.add_argument(
        "--eval-mean",
        default="runs/aist_t2m_evaluator_deps/t2m/Comp_v6_KLD01/meta/mean.npy",
    )
    parser.add_argument(
        "--eval-std",
        default="runs/aist_t2m_evaluator_deps/t2m/Comp_v6_KLD01/meta/std.npy",
    )
    parser.add_argument(
        "--output",
        default="training_logs/baseline_eval_corrected_20260715/stickmotion.json",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[2]
    stick_root = (workspace / "stickmotion").resolve()
    config_path = (workspace / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    ckpt_path = (workspace / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    eval_mean_path = (workspace / args.eval_mean).resolve()
    eval_std_path = (workspace / args.eval_std).resolve()
    output_path = (workspace / args.output).resolve()

    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(stick_root))
    os.chdir(stick_root)
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-codex")

    import torch
    from lightning import Trainer
    from lightning.pytorch import seed_everything
    from mmcv import Config
    from mmcv.parallel import DataContainer
    from torch.utils.data import DataLoader

    from flowmimic.src.metrics.distribution_metrics import summarize_motion_feature_metrics
    from flowmimic.src.metrics.motion_quality_metrics import (
        aggregate_replications,
        summarize_physical_motion,
    )
    from mogen.apis.lg_train import LgModel
    from mogen.core.evaluation.get_model import get_motion_model
    from mogen.datasets import build_dataset
    from mogen.models.attentions.semantics_modulated import SemanticsModulatedAttention
    from mogen.models.utils import gaussian_diffusion
    from mogen.utils.plot_utils import recover_from_ric

    def collate_fn(batch):
        keys = batch[0].keys()
        final_batch = {}
        for key in keys:
            if isinstance(batch[0][key], DataContainer):
                data = [item[key]._data for item in batch]
                final_batch[key] = data
                if key == "motion_metas" and data and isinstance(data[0], dict):
                    for meta_key in ("text", "token"):
                        if meta_key in data[0]:
                            final_batch[meta_key] = [item[meta_key] for item in data]
            elif isinstance(batch[0][key], torch.Tensor):
                final_batch[key] = torch.stack([item[key] for item in batch], 0)
            else:
                raise NotImplementedError(f"Unsupported batch value for {key}: {type(batch[0][key])}")
        return final_batch

    def sample_timesteps(self, batch_size, device):
        w = self.weights()
        if not torch.is_tensor(w):
            w = torch.as_tensor(w, dtype=torch.float32, device=device)
        else:
            w = w.to(device=device, dtype=torch.float32)
        p = w / w.sum()
        indices = torch.multinomial(p, batch_size, replacement=True).long()
        weights = (1 / (len(p) * p[indices])).float()
        return indices, weights

    def extract_into_tensor(arr, timesteps, broadcast_shape):
        if not torch.is_tensor(arr):
            arr = torch.from_numpy(arr).to(device=timesteps.device)
        else:
            arr = arr.to(device=timesteps.device)
        res = arr[timesteps.long()].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    gaussian_diffusion.ScheduleSampler.sample = sample_timesteps
    gaussian_diffusion._extract_into_tensor = extract_into_tensor

    original_semantics_forward = SemanticsModulatedAttention.forward

    def semantics_forward_compat(
        self,
        x,
        text_emb,
        stick_emb,
        other_emb,
        src_mask,
        cond_type,
        stick_mask,
        locus_emb,
        mid_query=None,
    ):
        return original_semantics_forward(
            self,
            x,
            text_emb,
            stick_emb,
            other_emb,
            src_mask,
            cond_type,
            stick_mask,
            locus_emb,
            mid_query,
        )

    SemanticsModulatedAttention.forward = semantics_forward_compat

    replication_rows = []
    current_seed = args.seed

    def validation_epoch_end_aist(self) -> None:
        nonlocal current_seed
        self.outputs = [item for batch_output in self.outputs for item in batch_output]
        results = self.outputs[: len(self.dataset)]
        self.outputs = []

        generated_norm = torch.stack([item["pred_motion"] for item in results]).float()
        reference_norm = torch.stack([item["motion"] for item in results]).float()
        lengths = torch.as_tensor(
            [int(item["motion_length"].item()) for item in results], dtype=torch.long
        )
        if generated_norm.shape != (470, 196, 263) or not torch.all(lengths == 196):
            raise ValueError(
                f"Unexpected StickMotion evaluation data: shape={generated_norm.shape}, "
                f"lengths={torch.unique(lengths).tolist()}"
            )

        generated_raw = generated_norm.numpy() * generator_std + generator_mean
        reference_raw = reference_norm.numpy() * generator_std + generator_mean
        generated_eval = (generated_raw - evaluator_mean) / (evaluator_std + 1e-6)
        reference_eval = (reference_raw - evaluator_mean) / (evaluator_std + 1e-6)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path.with_suffix(".generated.npy"), generated_norm.numpy())
        motion_encoder.to(self.device)

        def encode(motion):
            chunks = []
            for start in range(0, len(motion), 64):
                end = min(start + 64, len(motion))
                batch = torch.from_numpy(motion[start:end]).to(self.device)
                batch_lengths = lengths[start:end].to(self.device)
                batch_mask = torch.ones(
                    (end - start, motion.shape[1]), device=self.device
                )
                chunks.append(
                    motion_encoder(batch, batch_lengths, batch_mask).detach().cpu()
                )
            return torch.cat(chunks).numpy()

        generated_features = encode(generated_eval)
        reference_features = encode(reference_eval)
        np.random.seed(current_seed + 100_000)
        row = summarize_motion_feature_metrics(
            generated_features,
            reference_features,
            diversity_times=args.diversity_times,
        )
        generated_joints = recover_from_ric(
            torch.from_numpy(generated_raw), joints_num=22, ifnorm=False
        ).numpy()
        reference_joints = recover_from_ric(
            torch.from_numpy(reference_raw), joints_num=22, ifnorm=False
        ).numpy()
        row.update(
            summarize_physical_motion(
                generated_joints,
                reference_joints,
                generated_raw[..., -4:],
                args.fps,
            )
        )
        row.update(
            {
                "replication": len(replication_rows),
                "seed": current_seed,
            }
        )
        replication_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    LgModel.on_validation_epoch_end = validation_epoch_end_aist

    seed_everything(args.seed, workers=True)
    cfg = Config.fromfile(str(config_path))
    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = args.workers
    if args.guidance_repeat is not None:
        cfg.model.guidance.repeat = args.guidance_repeat
    cfg.data.test.test_mode = True
    cfg.data.test.eval_cfg.replication_times = 1
    cfg.data.test.eval_cfg.shuffle_indexes = False
    cfg.data.test.eval_cfg.metrics = []
    dataset = build_dataset(cfg.data.test)
    dataset.prepare_evaluation()
    if args.ablate_locus or args.ablate_stick:
        prepare_data = dataset.prepare_data

        def prepare_data_with_ablation(idx):
            sample = prepare_data(idx)
            if args.ablate_locus:
                sample["locus"] = torch.zeros_like(sample["locus"])
            if args.ablate_stick:
                sample["stickman_tracks"] = torch.zeros_like(sample["stickman_tracks"])
                sample["stick_mask"] = torch.zeros_like(sample["stick_mask"])
            return sample

        dataset.prepare_data = prepare_data_with_ablation
    generator_mean = np.asarray(dataset.mean, dtype=np.float32)
    generator_std = np.asarray(dataset.std, dtype=np.float32)
    evaluator_mean = np.load(eval_mean_path).astype(np.float32)
    evaluator_std = np.load(eval_std_path).astype(np.float32)

    evaluator_ckpt = Path(
        Config.fromfile(str(config_path)).data.test.eval_cfg.motion_encoder_path
    )
    motion_encoder = get_motion_model("aist60", str(evaluator_ckpt))
    motion_encoder.eval()

    pid_seed = os.getpid()
    model = LgModel(cfg, dataset, unit=hashlib.md5(str(pid_seed).encode()).hexdigest()[:8])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if args.condition_branch != "schedule":
        branch_coefficients = {
            "both": (1.0, 0.0, 0.0, 0.0),
            "text": (0.0, 1.0, 0.0, 0.0),
            "stick": (0.0, 0.0, 1.0, 0.0),
            "none": (0.0, 0.0, 0.0, 1.0),
        }
        both, text, stick, none = branch_coefficients[args.condition_branch]
        model.model.model.scale_func = lambda timestep: {
            "both_coef": both,
            "text_coef": text,
            "retr_coef": stick,
            "none_coef": none,
        }
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=False,
        num_workers=cfg.data.workers_per_gpu,
        collate_fn=collate_fn,
    )
    devices = parse_devices(args.gpu)
    if len(devices) != 1:
        raise ValueError("Canonical StickMotion evaluation currently expects one GPU")
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        logger=False,
        precision="16-mixed",
        inference_mode=False,
    )
    for replication in range(args.replications):
        current_seed = args.seed + replication
        seed_everything(current_seed, workers=True)
        trainer.validate(model, loader)

    output = {
        "method": "StickMotion",
        "summary": aggregate_replications(replication_rows),
        "replications": replication_rows,
        "protocol": {
            "split": "AIST++ official test",
            "samples_per_replication": 470,
            "crop": "first 196 frames",
            "replication_seeds": [args.seed + i for i in range(args.replications)],
            "checkpoint": str(ckpt_path),
            "evaluator_checkpoint": str(evaluator_ckpt),
            "evaluator_mean": str(eval_mean_path),
            "evaluator_std": str(eval_std_path),
            "normalization": "StickMotion -> physical IK263 -> AIST T2M",
            "condition": "text, fixed GT sketches at frames 24/98/171, full GT root locus",
            "guidance_repeat": int(cfg.model.guidance.repeat),
            "ablate_locus": args.ablate_locus,
            "ablate_stick": args.ablate_stick,
            "condition_branch": args.condition_branch,
            "fps": args.fps,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
