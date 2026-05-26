"""Prepare AIST++ IK263 data for MLD and StickMotion training."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


NAME_RE = re.compile(
    r"^(?P<genre>g[^_]+)_(?P<setup>s[^_]+)_(?P<camera>c[^_]+)_"
    r"(?P<dance>d[^_]+)_(?P<music>m[^_]+)_(?P<ch>ch\d+)$"
)


def parse_name(name: str) -> dict[str, str]:
    match = NAME_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected AIST++ name: {name}")
    return match.groupdict()


def text_key(name: str) -> tuple[str, str, str, str]:
    parts = parse_name(name)
    return parts["genre"], parts["setup"], parts["dance"], parts["ch"]


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_mld_text_index(text_dir: Path) -> dict[tuple[str, str, str, str], list[str]]:
    index: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for path in sorted(text_dir.glob("*.txt")):
        key = text_key(path.stem)
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        index[key].extend(lines)
    return index


def build_stick_text_index(
    text_dir: Path, token_dir: Path
) -> dict[tuple[str, str, str, str], tuple[list[str], list[str]]]:
    text_index: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    token_index: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for path in sorted(text_dir.glob("*.txt")):
        key = text_key(path.stem)
        token_path = token_dir / path.name
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tokens = [line.strip() for line in token_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(texts) != len(tokens):
            raise ValueError(f"Text/token line mismatch: {path} vs {token_path}")
        text_index[key].extend(texts)
        token_index[key].extend(tokens)
    return {key: (text_index[key], token_index[key]) for key in text_index}


def compute_or_load_stats(
    train_ids: list[str],
    motion_source_dir: Path,
    stats_cache: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if stats_cache.exists() and not force:
        data = np.load(stats_cache)
        return data["mean"], data["std"]

    total_frames = 0
    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    for name in train_ids:
        arr = np.load(motion_source_dir / f"{name}.npy").astype(np.float64, copy=False)
        if arr.ndim != 2 or arr.shape[1] != 263:
            raise ValueError(f"Expected (T, 263) motion, got {arr.shape}: {name}")
        if sum_x is None:
            sum_x = np.zeros(arr.shape[1], dtype=np.float64)
            sum_x2 = np.zeros(arr.shape[1], dtype=np.float64)
        sum_x += arr.sum(axis=0)
        sum_x2 += np.square(arr).sum(axis=0)
        total_frames += arr.shape[0]

    if sum_x is None or sum_x2 is None or total_frames == 0:
        raise ValueError("No training frames found for stats")

    mean = sum_x / total_frames
    var = np.maximum(sum_x2 / total_frames - np.square(mean), 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    stats_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_cache,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        frames=np.array(total_frames, dtype=np.int64),
        samples=np.array(len(train_ids), dtype=np.int64),
    )
    return mean.astype(np.float32), std.astype(np.float32)


def first_crop(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.shape[0] <= length:
        return arr
    return arr[:length]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def prepare(
    split_dir: Path,
    motion_source_dir: Path,
    mld_text_source_dir: Path,
    stick_text_source_dir: Path,
    stick_token_source_dir: Path,
    mld_out: Path,
    stick_out: Path,
    max_len: int,
    copy_stick_motions: bool,
    force_stats: bool,
) -> dict[str, int | str]:
    splits = {
        "train": read_split(split_dir / "pose_train.txt"),
        "val": read_split(split_dir / "pose_val.txt"),
        "test": read_split(split_dir / "pose_test.txt"),
    }
    all_ids = sorted({name for ids in splits.values() for name in ids})

    stats_cache = mld_out.parent / "aist_mean_std_263_train.npz"
    mean, std = compute_or_load_stats(splits["train"], motion_source_dir, stats_cache, force_stats)

    mld_text_index = build_mld_text_index(mld_text_source_dir)
    stick_text_index = build_stick_text_index(stick_text_source_dir, stick_token_source_dir)

    missing_motion: list[str] = []
    missing_text: list[str] = []
    for name in all_ids:
        if not (motion_source_dir / f"{name}.npy").exists():
            missing_motion.append(name)
        key = text_key(name)
        if key not in mld_text_index or key not in stick_text_index:
            missing_text.append(name)
    if missing_motion or missing_text:
        raise FileNotFoundError(
            json.dumps(
                {
                    "missing_motion_count": len(missing_motion),
                    "missing_text_count": len(missing_text),
                    "missing_motion_examples": missing_motion[:10],
                    "missing_text_examples": missing_text[:10],
                },
                indent=2,
            )
        )

    mld_motion_dir = mld_out / "new_joint_vecs"
    mld_text_dir = mld_out / "texts"
    stick_dataset = stick_out / "datasets" / "human_ml3d"
    stick_motion_dir = stick_dataset / "motions"
    stick_text_dir = stick_dataset / "texts"
    stick_token_dir = stick_dataset / "tokens"
    for path in (mld_motion_dir, mld_text_dir, stick_motion_dir, stick_text_dir, stick_token_dir):
        path.mkdir(parents=True, exist_ok=True)

    for dataset_root in (mld_out, stick_dataset):
        dataset_root.mkdir(parents=True, exist_ok=True)
        np.save(dataset_root / "Mean.npy", mean)
        np.save(dataset_root / "Std.npy", std)
        np.save(dataset_root / "mean.npy", mean)
        np.save(dataset_root / "std.npy", std)

    for split_name, ids in splits.items():
        write_lines(mld_out / f"{split_name}.txt", ids)
        write_lines(stick_dataset / f"{split_name}.txt", ids)
    write_lines(mld_out / "all.txt", all_ids)
    write_lines(stick_dataset / "all.txt", all_ids)

    for name in all_ids:
        src = motion_source_dir / f"{name}.npy"
        arr = np.load(src).astype(np.float32, copy=False)
        cropped = first_crop(arr, max_len)
        np.save(mld_motion_dir / f"{name}.npy", cropped)
        link_or_copy(src, stick_motion_dir / f"{name}.npy", copy=copy_stick_motions)

        key = text_key(name)
        write_lines(mld_text_dir / f"{name}.txt", mld_text_index[key])
        texts, tokens = stick_text_index[key]
        write_lines(stick_text_dir / f"{name}.txt", texts)
        write_lines(stick_token_dir / f"{name}.txt", tokens)

    summary = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "all_unique": len(all_ids),
        "stats_cache": str(stats_cache),
        "stats_frames": int(np.load(stats_cache)["frames"]),
        "mld_root": str(mld_out),
        "stickmotion_data_prefix": str(stick_out),
    }
    (mld_out.parent / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/AIST++/Annotations/splits")
    parser.add_argument("--motion-source-dir", default="data/cached_ik/aist")
    parser.add_argument("--mld-text-source-dir", default="data/AIST++/TextTokens/mld_texts")
    parser.add_argument("--stick-text-source-dir", default="data/AIST++/TextTokens/stickmotion/texts")
    parser.add_argument("--stick-token-source-dir", default="data/AIST++/TextTokens/stickmotion/tokens")
    parser.add_argument("--mld-out", default="prepared/aist_mld_humanml3d")
    parser.add_argument("--stick-out", default="prepared/aist_stickmotion")
    parser.add_argument("--max-len", type=int, default=196)
    parser.add_argument("--copy-stick-motions", action="store_true")
    parser.add_argument("--force-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare(
        split_dir=Path(args.split_dir),
        motion_source_dir=Path(args.motion_source_dir),
        mld_text_source_dir=Path(args.mld_text_source_dir),
        stick_text_source_dir=Path(args.stick_text_source_dir),
        stick_token_source_dir=Path(args.stick_token_source_dir),
        mld_out=Path(args.mld_out),
        stick_out=Path(args.stick_out),
        max_len=args.max_len,
        copy_stick_motions=args.copy_stick_motions,
        force_stats=args.force_stats,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
