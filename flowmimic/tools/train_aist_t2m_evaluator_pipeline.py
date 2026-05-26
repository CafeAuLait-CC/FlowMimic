#!/usr/bin/env python
"""Train T2M evaluator assets on the prepared AIST dataset.

This runs the two upstream Text2Motion stages needed by the evaluator:
1. movement autoencoder (`train_decomp_v3.py`)
2. text/motion matching encoders (`train_tex_mot_match.py`)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp()} {message}\n")
        handle.flush()


def run_stage(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    stdout_path: Path,
    log_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    log_line(log_path, f"Starting {name}: {' '.join(cmd)}")
    with stdout_path.open("w", encoding="utf-8") as stdout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        log_line(log_path, f"{name} PID {proc.pid}; stdout/stderr at {stdout_path}")
        return_code = proc.wait()
    log_line(log_path, f"{name} exited with return code {return_code}")
    return return_code


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def export_mld_layout(args: argparse.Namespace, log_path: Path) -> None:
    ckpt_root = Path(args.checkpoint_dir)
    export_root = Path(args.export_t2m_root)
    source_finest = ckpt_root / "aist" / args.match_name / "model" / "finest.tar"
    source_latest = ckpt_root / "aist" / args.match_name / "model" / "latest.tar"
    source_model = source_finest if source_finest.exists() else source_latest
    source_mean = ckpt_root / "aist" / args.decomp_name / "meta" / "mean.npy"
    source_std = ckpt_root / "aist" / args.decomp_name / "meta" / "std.npy"

    if not source_model.exists():
        raise FileNotFoundError(f"No trained text/motion matcher checkpoint found: {source_model}")
    if not source_mean.exists() or not source_std.exists():
        raise FileNotFoundError(f"Missing AIST evaluator stats: {source_mean}, {source_std}")

    replace_symlink(export_root / "t2m" / "text_mot_match" / "model" / "finest.tar", source_model)
    replace_symlink(export_root / "t2m" / "Comp_v6_KLD01" / "meta" / "mean.npy", source_mean)
    replace_symlink(export_root / "t2m" / "Comp_v6_KLD01" / "meta" / "std.npy", source_std)
    log_line(
        log_path,
        "Exported AIST evaluator in MLD-compatible t2m layout at "
        f"{export_root}; model={source_model}, mean={source_mean}, std={source_std}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", default="mld")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--text-to-motion-dir", default="/mnt/data3_hdd/alex/FlowMimic/text-to-motion")
    parser.add_argument("--data-root", default="/mnt/data3_hdd/alex/FlowMimic/prepared/aist_mld_humanml3d")
    parser.add_argument("--checkpoint-dir", default="/mnt/data3_hdd/alex/FlowMimic/runs/t2m_evaluator_checkpoints")
    parser.add_argument("--export-t2m-root", default="/mnt/data3_hdd/alex/FlowMimic/runs/aist_t2m_evaluator_deps")
    parser.add_argument("--log-path", default="/mnt/data3_hdd/alex/FlowMimic/training_logs/aist_t2m_evaluator.log")
    parser.add_argument("--decomp-stdout", default="/mnt/data3_hdd/alex/FlowMimic/training_logs/aist_t2m_decomp_train.out")
    parser.add_argument("--match-stdout", default="/mnt/data3_hdd/alex/FlowMimic/training_logs/aist_t2m_match_train.out")
    parser.add_argument("--decomp-name", default="Decomp_AIST_SM001_H512")
    parser.add_argument("--match-name", default="text_mot_match")
    parser.add_argument("--decomp-batch-size", type=int, default=8192)
    parser.add_argument("--match-batch-size", type=int, default=1024)
    parser.add_argument("--match-val-batch-size", type=int, default=128)
    parser.add_argument("--match-train-split", default="train.txt")
    parser.add_argument("--match-val-split", default="val_test.txt")
    parser.add_argument("--match-train-dataset-repeat", type=int, default=20)
    parser.add_argument("--match-max-text-len", type=int, default=60)
    parser.add_argument("--match-lr", type=float, default=1e-4)
    parser.add_argument("--match-reset-optimizer-lr", action="store_true")
    parser.add_argument("--match-validate-every-e", type=int, default=10)
    parser.add_argument("--match-save-latest-every-e", type=int, default=10)
    parser.add_argument("--decomp-epochs", type=int, default=270)
    parser.add_argument("--match-epochs", type=int, default=1200)
    parser.add_argument("--match-continue", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    t2m_dir = Path(args.text_to_motion_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    decomp_latest = checkpoint_dir / "aist" / args.decomp_name / "model" / "latest.tar"
    decomp_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "train_decomp_v3.py",
        "--name",
        args.decomp_name,
        "--gpu_id",
        str(args.gpu_id),
        "--window_size",
        "24",
        "--dataset_name",
        "aist",
        "--data_root",
        args.data_root,
        "--checkpoints_dir",
        str(checkpoint_dir),
        "--batch_size",
        str(args.decomp_batch_size),
        "--max_epoch",
        str(args.decomp_epochs),
        "--eval_every_e",
        "999999",
        "--save_every_e",
        "10",
        "--log_every",
        "20",
    ]
    if decomp_latest.exists():
        log_line(log_path, f"Skipping decomp; existing checkpoint found at {decomp_latest}")
    else:
        code = run_stage(
            name="AIST T2M movement autoencoder",
            cmd=decomp_cmd,
            cwd=t2m_dir,
            stdout_path=Path(args.decomp_stdout),
            log_path=log_path,
        )
        if code != 0:
            return code

    match_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "train_tex_mot_match.py",
        "--name",
        args.match_name,
        "--gpu_id",
        str(args.gpu_id),
        "--batch_size",
        str(args.match_batch_size),
        "--val_batch_size",
        str(args.match_val_batch_size),
        "--max_text_len",
        str(args.match_max_text_len),
        "--dataset_name",
        "aist",
        "--data_root",
        args.data_root,
        "--train_split",
        args.match_train_split,
        "--val_split",
        args.match_val_split,
        "--train_dataset_repeat",
        str(args.match_train_dataset_repeat),
        "--checkpoints_dir",
        str(checkpoint_dir),
        "--decomp_name",
        args.decomp_name,
        "--max_epoch",
        str(args.match_epochs),
        "--lr",
        str(args.match_lr),
        "--save_every_e",
        "5",
        "--eval_every_e",
        str(args.match_validate_every_e),
        "--validate_every_e",
        str(args.match_validate_every_e),
        "--save_latest_every_e",
        str(args.match_save_latest_every_e),
        "--save_latest",
        "2000",
        "--log_every",
        "20",
    ]
    if args.match_continue:
        match_cmd.append("--is_continue")
    if args.match_reset_optimizer_lr:
        match_cmd.append("--reset_optimizer_lr")
    code = run_stage(
        name="AIST T2M text/motion matcher",
        cmd=match_cmd,
        cwd=t2m_dir,
        stdout_path=Path(args.match_stdout),
        log_path=log_path,
    )
    if code != 0:
        return code

    export_mld_layout(args, log_path)
    log_line(log_path, "AIST T2M evaluator pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
