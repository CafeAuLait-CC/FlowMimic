#!/usr/bin/env python3
"""Generate an aligned FlowMimic/MLD/StickMotion AIST++ comparison bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from flowmimic.src.data.dataloader import yup_to_blender
from flowmimic.src.motion.process_motion import (
    align_smpl22_with_contact_and_center,
    ik263_to_smpl22,
)


NAME_RE = re.compile(
    r"^(?P<genre>g[^_]+)_(?P<setup>s[^_]+)_(?P<camera>c[^_]+)_"
    r"(?P<dance>d[^_]+)_(?P<music>m[^_]+)_(?P<ch>ch\d+)$"
)


def _resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _read_nonempty(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text_stem(sample_id: str, camera: str) -> str:
    match = NAME_RE.match(sample_id)
    if not match:
        raise ValueError(f"Unexpected AIST++ sample id: {sample_id}")
    parts = match.groupdict()
    camera_token = camera if camera.startswith("c") else f"c{camera}"
    return (
        f"{parts['genre']}_{parts['setup']}_{camera_token}_{parts['dance']}_"
        f"mAll_{parts['ch']}"
    )


def _load_caption(
    workspace: Path, sample_id: str, camera: str, caption_index: int
) -> dict:
    stem = _text_stem(sample_id, camera)
    mld_path = workspace / "data" / "AIST++" / "TextTokens" / "mld_texts" / f"{stem}.txt"
    stick_text_path = (
        workspace / "data" / "AIST++" / "TextTokens" / "stickmotion" / "texts" / f"{stem}.txt"
    )
    stick_token_path = (
        workspace / "data" / "AIST++" / "TextTokens" / "stickmotion" / "tokens" / f"{stem}.txt"
    )
    for path in (mld_path, stick_text_path, stick_token_path):
        if not path.exists():
            raise FileNotFoundError(path)

    mld_rows = _read_nonempty(mld_path)
    stick_texts = _read_nonempty(stick_text_path)
    stick_tokens = _read_nonempty(stick_token_path)
    if not (len(mld_rows) == len(stick_texts) == len(stick_tokens)):
        raise ValueError(f"Caption/token count mismatch for {stem}")
    if not 0 <= caption_index < len(stick_texts):
        raise IndexError(
            f"caption index {caption_index} outside [0, {len(stick_texts) - 1}] for {stem}"
        )

    mld_parts = mld_rows[caption_index].split("#")
    mld_text = mld_parts[0].strip()
    mld_token = mld_parts[1].strip() if len(mld_parts) > 1 else ""
    text = stick_texts[caption_index]
    token = stick_tokens[caption_index]
    if mld_text != text:
        raise ValueError(f"MLD and StickMotion caption mismatch for {stem}")
    return {
        "text": text,
        "token": token,
        "mld_token": mld_token,
        "camera": camera.removeprefix("c"),
        "index": caption_index,
        "stem": stem,
        "mld_source": str(mld_path.relative_to(workspace)),
        "stickmotion_text_source": str(stick_text_path.relative_to(workspace)),
        "stickmotion_token_source": str(stick_token_path.relative_to(workspace)),
    }


def _tokenize_caption(python: str, workspace: Path, text: str) -> str:
    command = [
        python,
        "flowmimic/tools/process_aist_text_tokens.py",
        "--text",
        text,
    ]
    result = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"HumanML3D caption tokenization failed: {detail}")
    token = result.stdout.strip()
    if not token:
        raise RuntimeError("HumanML3D caption tokenization returned no tokens")
    return token


def _run(name: str, command: list[str], cwd: Path, env: dict, log_path: Path) -> None:
    shown = shlex.join(command)
    print(f"[{name}] {shown}", flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(f"$ {shown}\n\n{result.stdout}", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with code {result.returncode}; see {log_path}")


def _gpu_env(gpu: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _device_runtime(device: str | None, gpu: int) -> tuple[str, dict]:
    if device is None:
        return "cuda", _gpu_env(gpu)
    normalized = device.strip().lower()
    if not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", normalized):
        raise ValueError(
            f"Unsupported device {device!r}; use cpu, cuda, or cuda:<index>"
        )
    env = os.environ.copy()
    if normalized == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return normalized, env


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _resolve_executable(value: str, label: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute() or path.parent != Path("."):
        resolved = path.absolute()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise FileNotFoundError(f"{label} is not executable: {resolved}")
        return str(resolved)
    resolved = shutil.which(expanded)
    if resolved is None:
        raise FileNotFoundError(f"{label} was not found in PATH: {expanded}")
    return resolved


def _conda_env_python(env_name: str, override: str | None, env_var: str) -> str:
    configured = override or os.environ.get(env_var)
    if configured:
        return _resolve_executable(configured, f"{env_name} Python")

    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
    envs_dir = prefix.parent if prefix.parent.name == "envs" else prefix / "envs"
    candidate = envs_dir / env_name / "bin" / "python"
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(
            f"Python for Conda environment {env_name!r} was not found at {candidate}. "
            f"Set {env_var} or pass the matching --*-python option."
        )
    return str(candidate.absolute())


def _blender_executable(override: str | None) -> str:
    configured = override or os.environ.get("FLOWMIMIC_BLENDER")
    if configured:
        return _resolve_executable(configured, "Blender")
    discovered = shutil.which("blender")
    if discovered is None:
        raise FileNotFoundError(
            "Blender was not found in PATH. Install Blender or set FLOWMIMIC_BLENDER."
        )
    return discovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--camera", default="01")
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument(
        "--caption-text",
        default=None,
        help="Override the selected caption and regenerate its StickMotion tokens.",
    )
    parser.add_argument("--seq-len", type=int, default=196)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--condition-frames", type=int, default=28)
    parser.add_argument(
        "--stickmotion-sketch-frames", type=int, nargs=3, default=None
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--flow-steps", type=int, default=50)
    parser.add_argument("--flow-solver", choices=("heun", "euler"), default="heun")
    parser.add_argument("--flow-use-ema", action="store_true")
    parser.add_argument(
        "--flow-ckpt",
        default=(
            "checkpoints/flow/vqflow_stage2_ep120_varK_sharedcam_ddp2_b640_260714/"
            "flow_round0_epoch140.pt"
        ),
    )
    parser.add_argument(
        "--mld-ckpt",
        default="runs/mld/mld/aist_ik263_mld_196/checkpoints/epoch=2999.ckpt",
    )
    parser.add_argument(
        "--stickmotion-ckpt",
        default="runs/stickmotion/human_ml3d/aist_remodiffuse/last.ckpt",
    )
    parser.add_argument("--existing-flow-motion", default=None)
    parser.add_argument("--existing-flow-meta", default=None)
    parser.add_argument("--flow-gpu", type=int, default=0)
    parser.add_argument("--mld-gpu", type=int, default=0)
    parser.add_argument("--stickmotion-gpu", type=int, default=0)
    parser.add_argument("--flow-device", default=None)
    parser.add_argument("--mld-device", default=None)
    parser.add_argument("--stickmotion-device", default=None)
    parser.add_argument("--flow-python", default=None)
    parser.add_argument("--mld-python", default=None)
    parser.add_argument("--stickmotion-python", default=None)
    parser.add_argument("--output-root", default="output/aist_method_comparisons")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--launch-blender", action="store_true")
    parser.add_argument(
        "--save-blend",
        nargs="?",
        const="comparison.blend",
        default=None,
        metavar="FILENAME",
    )
    parser.add_argument("--blender", default=None)
    parser.add_argument(
        "--visualization-mode",
        choices=("skeleton", "rigged"),
        default="skeleton",
        help="Geometry used in the Blender comparison scene.",
    )
    parser.add_argument(
        "--rigged-model",
        default=None,
        help="SMPL22 GLB used when --visualization-mode=rigged.",
    )
    args = parser.parse_args()
    try:
        args.flow_python = _conda_env_python(
            "flowmimic-310", args.flow_python, "FLOWMIMIC_PYTHON"
        )
        args.mld_python = _conda_env_python(
            "mld", args.mld_python, "MLD_PYTHON"
        )
        args.stickmotion_python = _conda_env_python(
            "stickmotion", args.stickmotion_python, "STICKMOTION_PYTHON"
        )
        args.blender = _blender_executable(args.blender)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    workspace = WORKSPACE
    flow_device, flow_env = _device_runtime(args.flow_device, args.flow_gpu)
    mld_device, mld_env = _device_runtime(args.mld_device, args.mld_gpu)
    stickmotion_device, stickmotion_env = _device_runtime(
        args.stickmotion_device, args.stickmotion_gpu
    )
    default_rigged_model = workspace / "web_view" / "assets" / "smpl22_rigged_calibrated.glb"
    if not default_rigged_model.is_file():
        default_rigged_model = workspace / "web_view" / "assets" / "smpl22_rigged.glb"
    rigged_model = Path(args.rigged_model) if args.rigged_model else default_rigged_model
    if not rigged_model.is_absolute():
        rigged_model = (workspace / rigged_model).resolve()
    if args.visualization_mode == "rigged" and not rigged_model.is_file():
        raise FileNotFoundError(f"Rigged SMPL22 model not found: {rigged_model}")
    if args.condition_frames < 1 or args.condition_frames > args.seq_len:
        raise ValueError("condition_frames must be within [1, seq_len]")
    if args.start < 0:
        raise ValueError("start must be >= 0")
    stickmotion_sketch_frames = args.stickmotion_sketch_frames or [
        int(p * args.seq_len) for p in (0.125, 0.5, 0.875)
    ]
    if len(set(stickmotion_sketch_frames)) != len(stickmotion_sketch_frames):
        raise ValueError("stickmotion sketch frames must be distinct")
    if any(index < 0 or index >= args.seq_len for index in stickmotion_sketch_frames):
        raise ValueError(
            f"stickmotion sketch frames must be within [0, {args.seq_len - 1}]"
        )

    split_path = workspace / "data" / "AIST++" / "Annotations" / "splits" / f"pose_{args.split}.txt"
    sample_ids = _read_nonempty(split_path)
    if args.sample_id is None:
        if not 0 <= args.sample_index < len(sample_ids):
            raise IndexError(f"sample index outside [0, {len(sample_ids) - 1}]")
        sample_id = sample_ids[args.sample_index]
    else:
        sample_id = args.sample_id
        if sample_id not in sample_ids:
            raise ValueError(f"{sample_id} is not in {split_path}")

    caption = _load_caption(workspace, sample_id, args.camera, args.caption_index)
    selected_text = caption["text"]
    if args.caption_text is not None:
        confirmed_text = args.caption_text.replace("#", " ").strip()
        if not confirmed_text:
            raise ValueError("--caption-text must contain visible text")
        if len(confirmed_text) > 1000:
            raise ValueError("--caption-text must not exceed 1000 characters")
        if confirmed_text != selected_text:
            caption["token"] = _tokenize_caption(
                args.mld_python, workspace, confirmed_text
            )
            caption["mld_token"] = caption["token"]
            caption["token_source"] = "generated_from_confirmed_text"
        caption["original_text"] = selected_text
        caption["text"] = confirmed_text
        caption["edited"] = confirmed_text != selected_text
    if args.run_dir:
        run_dir = _resolve(workspace, args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _resolve(workspace, args.output_root) / f"{stamp}_{sample_id}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Comparison run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    motion_path = workspace / "data" / "cached_ik" / "aist" / f"{sample_id}.npy"
    aist_pkl = workspace / "data" / "AIST++" / "Annotations" / "motions" / f"{sample_id}.pkl"
    motion = np.load(motion_path).astype(np.float32)
    clip_end = args.start + args.seq_len
    if motion.shape[0] < clip_end or motion.shape[1:] != (263,):
        raise ValueError(
            f"Expected at least ({clip_end}, 263), got {motion.shape}: {motion_path}"
        )
    motion = motion[args.start:clip_end]
    reference = ik263_to_smpl22(motion)
    reference = align_smpl22_with_contact_and_center(motion, reference)
    reference_path = run_dir / "reference.npy"
    np.save(reference_path, yup_to_blender(reference).astype(np.float32))

    initial_manifest = {
        "version": 1,
        "status": "running",
        "sample_id": sample_id,
        "split": args.split,
        "sample_index": sample_ids.index(sample_id),
        "camera": args.camera.removeprefix("c"),
        "clip_start": args.start,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "caption": caption,
        "source_motion": str(motion_path.relative_to(workspace)),
    }
    manifest_path = run_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(initial_manifest, indent=2) + "\n", encoding="utf-8")

    flow_path = run_dir / "flowmimic.npy"
    flow_meta_path = run_dir / "flowmimic_meta.json"
    if bool(args.existing_flow_motion) != bool(args.existing_flow_meta):
        raise ValueError(
            "--existing-flow-motion and --existing-flow-meta must be provided together"
        )
    if args.existing_flow_motion:
        existing_motion = _resolve(workspace, args.existing_flow_motion)
        existing_meta = _resolve(workspace, args.existing_flow_meta)
        if not existing_motion.is_file() or not existing_meta.is_file():
            raise FileNotFoundError(
                f"Existing FlowMimic result is incomplete: {existing_motion}, {existing_meta}"
            )
        flow_meta = json.loads(existing_meta.read_text(encoding="utf-8"))
        expected = {
            "dataset": "aist",
            "camera": args.camera.removeprefix("c"),
            "start": args.start,
            "seq_len": args.seq_len,
        }
        for key, value in expected.items():
            actual = flow_meta.get(key)
            if str(actual) != str(value):
                raise ValueError(
                    f"Existing FlowMimic metadata mismatch for {key}: {actual!r} != {value!r}"
                )
        if Path(str(flow_meta.get("path", ""))).stem != sample_id:
            raise ValueError("Existing FlowMimic metadata does not match the requested sample")
        condition_count = int(
            flow_meta.get("condition_frame_count")
            or len(flow_meta.get("condition_indices", []))
        )
        if condition_count != args.condition_frames:
            raise ValueError(
                f"Existing FlowMimic condition count {condition_count} != {args.condition_frames}"
            )
        existing_array = np.load(existing_motion, mmap_mode="r")
        if existing_array.shape != (args.seq_len, 22, 3):
            raise ValueError(
                f"Expected existing FlowMimic motion {(args.seq_len, 22, 3)}, "
                f"got {existing_array.shape}"
            )
        shutil.copy2(existing_motion, flow_path)
        shutil.copy2(existing_meta, flow_meta_path)
        (run_dir / "flowmimic.log").write_text(
            f"Reused FlowMimic motion: {existing_motion}\n"
            f"Reused FlowMimic metadata: {existing_meta}\n",
            encoding="utf-8",
        )
        print(f"[flowmimic] reused {existing_motion}", flush=True)
    else:
        flow_runs = run_dir / "flow_runs"
        flow_command = [
            args.flow_python,
            "flowmimic/scripts/sample_flow.py",
            "--checkpoint",
            args.flow_ckpt,
            "--dataset",
            "aist",
            "--sample-path",
            str(aist_pkl),
            "--camera",
            args.camera,
            "--start",
            str(args.start),
            "--seq-len",
            str(args.seq_len),
            "--cond-frames",
            str(args.condition_frames),
            "--steps",
            str(args.flow_steps),
            "--solver",
            args.flow_solver,
            "--seed",
            str(args.seed),
            "--out",
            "flowmimic.npy",
            "--out-dir",
            str(flow_runs),
            "--device",
            flow_device,
        ]
        if args.flow_use_ema:
            flow_command.append("--use-ema")
        _run(
            "flowmimic",
            flow_command,
            workspace,
            flow_env,
            run_dir / "flowmimic.log",
        )
        flow_source_dir = (flow_runs / "last").resolve()
        shutil.copy2(flow_source_dir / "flowmimic.npy", flow_path)
        shutil.copy2(flow_source_dir / "result_meta.json", flow_meta_path)
    flow_meta = json.loads(flow_meta_path.read_text(encoding="utf-8"))

    mld_path = run_dir / "mld.npy"
    mld_meta_path = run_dir / "mld_meta.json"
    mld_command = [
        args.mld_python,
        "baselines/mld/tools/export_aist_sample.py",
        "--text",
        caption["text"],
        "--length",
        str(args.seq_len),
        "--sample-id",
        sample_id,
        "--seed",
        str(args.seed),
        "--device",
        mld_device,
        "--ckpt",
        args.mld_ckpt,
        "--output",
        str(mld_path),
        "--meta",
        str(mld_meta_path),
    ]
    _run("mld", mld_command, workspace, mld_env, run_dir / "mld.log")

    stick_path = run_dir / "stickmotion.npy"
    stick_ref_path = run_dir / "stickmotion_reference.npy"
    tracks_path = run_dir / "stickman_tracks.npy"
    locus_path = run_dir / "stickmotion_locus.npy"
    stick_meta_path = run_dir / "stickmotion_meta.json"
    stick_command = [
        args.stickmotion_python,
        "baselines/stickmotion/tools/export_aist_sample.py",
        "--sample-id",
        sample_id,
        "--split",
        args.split,
        "--text",
        caption["text"],
        "--token",
        caption["token"],
        "--start",
        str(args.start),
        "--length",
        str(args.seq_len),
        "--sketch-frames",
        *[str(index) for index in stickmotion_sketch_frames],
        "--seed",
        str(args.seed),
        "--device",
        stickmotion_device,
        "--ckpt",
        args.stickmotion_ckpt,
        "--output",
        str(stick_path),
        "--reference-output",
        str(stick_ref_path),
        "--tracks-output",
        str(tracks_path),
        "--locus-output",
        str(locus_path),
        "--meta",
        str(stick_meta_path),
    ]
    _run(
        "stickmotion",
        stick_command,
        workspace,
        stickmotion_env,
        run_dir / "stickmotion.log",
    )
    stick_meta = json.loads(stick_meta_path.read_text(encoding="utf-8"))

    manifest = {
        **initial_manifest,
        "visualization_mode": args.visualization_mode,
        "status": "complete",
        "condition": {
            "method": "flowmimic_openpose",
            "requested_frames": args.condition_frames,
            "frame_indices": flow_meta["condition_indices"],
            "source_frame_indices": [
                args.start + index for index in flow_meta["condition_indices"]
            ],
            "clip_start": args.start,
            "camera": args.camera.removeprefix("c"),
        },
        "motions": [
            {"key": "reference", "label": "Reference", "path": _relative(reference_path, run_dir)},
            {"key": "flowmimic", "label": "FlowMimic", "path": _relative(flow_path, run_dir)},
            {"key": "mld", "label": "MLD", "path": _relative(mld_path, run_dir)},
            {"key": "stickmotion", "label": "StickMotion", "path": _relative(stick_path, run_dir)},
        ],
        "methods": {
            "flowmimic": {
                "checkpoint": args.flow_ckpt,
                "steps": args.flow_steps,
                "solver": args.flow_solver,
                "use_ema": args.flow_use_ema,
                "metadata": _relative(flow_meta_path, run_dir),
            },
            "mld": {
                "checkpoint": args.mld_ckpt,
                "text": caption["text"],
                "metadata": _relative(mld_meta_path, run_dir),
            },
            "stickmotion": {
                "checkpoint": args.stickmotion_ckpt,
                "text": caption["text"],
                "token": caption["token"],
                "metadata": _relative(stick_meta_path, run_dir),
                "stickman_tracks": _relative(tracks_path, run_dir),
                "stickman_frame_indices": stick_meta["stickman_frame_indices"],
                "stickman_source_frame_indices": stick_meta[
                    "stickman_source_frame_indices"
                ],
                "locus": _relative(locus_path, run_dir),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    blender_command = [
        args.blender,
        "--python",
        str(workspace / "flowmimic" / "tools" / "vis_smpl22_blender.py"),
        "--",
        "--manifest",
        str(manifest_path),
        "--visualization-mode",
        args.visualization_mode,
        "--rigged-model",
        str(rigged_model),
    ]
    print(f"Comparison bundle: {run_dir}")
    print(f"Blender: {shlex.join(blender_command)}")
    if args.save_blend:
        save_command = [
            args.blender,
            "--background",
            "--python",
            str(workspace / "flowmimic" / "tools" / "vis_smpl22_blender.py"),
            "--",
            "--manifest",
            str(manifest_path),
            "--save-blend",
            args.save_blend,
            "--visualization-mode",
            args.visualization_mode,
            "--rigged-model",
            str(rigged_model),
        ]
        _run("blender", save_command, workspace, os.environ.copy(), run_dir / "blender.log")
    if args.launch_blender:
        subprocess.run(blender_command, cwd=workspace, check=True)


if __name__ == "__main__":
    main()
