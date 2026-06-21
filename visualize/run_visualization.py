#!/usr/bin/env python
"""Run SMPL fitting and Blender rendering for a FlowMimic result.

The fitting/rendering approach is adapted from MLD:
https://github.com/ChenFengYe/motion-latent-diffusion
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve()


def _split_ranges(total_frames: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, total_frames))
    base = total_frames // workers
    remainder = total_frames % workers
    ranges = []
    start = 0
    for worker_idx in range(workers):
        length = base + (1 if worker_idx < remainder else 0)
        end = start + length
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def _discover_gpu_indices() -> list[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _resolve_render_devices(render_device: str) -> list[str]:
    if render_device == "all":
        devices = _discover_gpu_indices()
        return devices if devices else ["all"]
    devices = [item.strip() for item in render_device.split(",") if item.strip()]
    return devices if devices else ["0"]


def _video_frame_dir(video_path: Path) -> Path:
    frame_dir = video_path.with_suffix("")
    return frame_dir.with_name(frame_dir.name + "_frames")


def _encode_video(frame_dir: Path, video_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; cannot encode parallel render frames.")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            str(video_path),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/flow/last/result_smpl22.npy")
    parser.add_argument("--out-dir", default="output/flow/last/visualize")
    parser.add_argument("--mld-root", default="motion-latent-diffusion")
    parser.add_argument("--input-space", choices=("blender", "yup"), default="blender")
    parser.add_argument("--fit-python", default=sys.executable)
    parser.add_argument("--blender", default="/snap/bin/blender")
    parser.add_argument("--device", default="auto", help="Fitting device.")
    parser.add_argument(
        "--fit-devices",
        default=None,
        help=(
            "Comma-separated fitting devices, e.g. cuda:0,cuda:1. "
            "Defaults to --device; --device cuda expands to all visible GPUs."
        ),
    )
    parser.add_argument(
        "--render-device",
        default="0",
        help="Blender Cycles GPU index, or 'all' to use every visible GPU.",
    )
    parser.add_argument(
        "--gender",
        choices=("female", "male", "neutral", "custom"),
        default="female",
        help="SMPL model file to load during fitting.",
    )
    parser.add_argument("--num-smplify-iters", type=int, default=100)
    parser.add_argument(
        "--optimizer",
        choices=("lbfgs", "adam"),
        default="lbfgs",
        help="SMPLify optimizer. Adam is much faster; LBFGS usually fits tighter.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--fit-workers", type=int, default=None)
    parser.add_argument("--worker-threads", type=int, default=None)
    parser.add_argument("--save-frame-files", action="store_true")
    parser.add_argument("--show-inner-progress", action="store_true")
    parser.add_argument("--worker-log", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--res", choices=("low", "med", "high"), default="med")
    parser.add_argument(
        "--render-workers",
        type=int,
        default=1,
        help="Parallel Blender render workers. 0 means one worker per render device.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = _resolve(args.input)
    out_dir = _resolve(args.out_dir)
    mld_root = _resolve(args.mld_root)
    fit_script = _repo_root() / "visualize" / "fit_smpl.py"
    render_script = _repo_root() / "visualize" / "render_mesh_blender.py"

    fit_cmd = [
        args.fit_python,
        str(fit_script),
        "--input",
        str(input_path),
        "--out-dir",
        str(out_dir),
        "--mld-root",
        str(mld_root),
        "--input-space",
        args.input_space,
        "--device",
        args.device,
        "--gender",
        args.gender,
        "--num-smplify-iters",
        str(args.num_smplify_iters),
        "--optimizer",
        args.optimizer,
    ]
    if args.max_frames is not None:
        fit_cmd += ["--max-frames", str(args.max_frames)]
    if args.num_threads is not None:
        fit_cmd += ["--num-threads", str(args.num_threads)]
    if args.fit_devices is not None:
        fit_cmd += ["--fit-devices", args.fit_devices]
    if args.fit_workers is not None:
        fit_cmd += ["--fit-workers", str(args.fit_workers)]
    if args.worker_threads is not None:
        fit_cmd += ["--worker-threads", str(args.worker_threads)]
    if args.save_frame_files:
        fit_cmd.append("--save-frame-files")
    if args.show_inner_progress:
        fit_cmd.append("--show-inner-progress")
    if args.worker_log:
        fit_cmd.append("--worker-log")
    if args.overwrite:
        fit_cmd.append("--overwrite")

    print("Running fit:")
    print(" ".join(fit_cmd))
    subprocess.run(fit_cmd, check=True)

    manifest_path = out_dir / input_path.stem / "fit_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    mesh_path = Path(manifest["mesh_path"])
    faces_path = Path(manifest["faces_path"])
    video_path = out_dir / input_path.stem / f"{input_path.stem}_smpl.mp4"

    render_devices = _resolve_render_devices(args.render_device)
    total_frames = int(np.load(mesh_path, mmap_mode="r").shape[0])
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)
    render_workers = args.render_workers
    if render_workers == 0:
        render_workers = len(render_devices)
    render_workers = max(1, min(render_workers, total_frames))

    if render_workers == 1:
        render_cmd = [
            args.blender,
            "--background",
            "--python",
            str(render_script),
            "--",
            "--mesh",
            str(mesh_path),
            "--faces",
            str(faces_path),
            "--out",
            str(video_path),
            "--fps",
            str(args.fps),
            "--res",
            args.res,
            "--device",
            args.render_device,
        ]
        if args.max_frames is not None:
            render_cmd += ["--max-frames", str(args.max_frames)]
        if args.keep_frames:
            render_cmd.append("--keep-frames")

        print("Running render:")
        print(" ".join(render_cmd))
        subprocess.run(render_cmd, check=True)
    else:
        frame_dir = _video_frame_dir(video_path)
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)

        ranges = _split_ranges(total_frames, render_workers)
        render_cmds = []
        for worker_idx, (frame_start, frame_end) in enumerate(ranges):
            device = render_devices[worker_idx % len(render_devices)]
            render_cmds.append(
                [
                    args.blender,
                    "--background",
                    "--python",
                    str(render_script),
                    "--",
                    "--mesh",
                    str(mesh_path),
                    "--faces",
                    str(faces_path),
                    "--mode",
                    "frames",
                    "--out",
                    str(frame_dir),
                    "--fps",
                    str(args.fps),
                    "--res",
                    args.res,
                    "--device",
                    device,
                    "--frame-start",
                    str(frame_start),
                    "--frame-end",
                    str(frame_end),
                    "--no-clear-frames",
                ]
            )

        print(
            "Running parallel render: "
            f"{render_workers} workers on devices {', '.join(render_devices)}"
        )
        for cmd in render_cmds:
            print(" ".join(cmd))
        processes = [subprocess.Popen(cmd) for cmd in render_cmds]
        failed = []
        for process in processes:
            code = process.wait()
            if code != 0:
                failed.append(code)
        if failed:
            raise subprocess.CalledProcessError(failed[0], "parallel blender render")

        _encode_video(frame_dir, video_path, args.fps)
        if not args.keep_frames and video_path.exists():
            shutil.rmtree(frame_dir)
    print(f"Done: {video_path}")


if __name__ == "__main__":
    main()
