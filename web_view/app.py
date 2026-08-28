from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from flowmimic.src.data.dataloader import blender_to_yup
from flowmimic.src.config.config import load_config


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
ASSET_DIR = WEB_DIR / "assets"
RIGGED_MODEL_SOURCE_PATH = ASSET_DIR / "smpl22_rigged.glb"
RIGGED_MODEL_CALIBRATED_PATH = ASSET_DIR / "smpl22_rigged_calibrated.glb"
RIGGED_MODEL_PATH = Path(
    os.environ.get(
        "FLOWMIMIC_RIGGED_MODEL",
        str(
            RIGGED_MODEL_CALIBRATED_PATH
            if RIGGED_MODEL_CALIBRATED_PATH.is_file()
            else RIGGED_MODEL_SOURCE_PATH
        ),
    )
).resolve()
OUTPUT_ROOT = (ROOT_DIR / "output" / "flow").resolve()
SAMPLE_SCRIPT = ROOT_DIR / "flowmimic" / "scripts" / "sample_flow.py"
EXTRACT_SCRIPT = ROOT_DIR / "flowmimic" / "tools" / "extract_cond_media.py"
COMPARISON_SCRIPT = (
    ROOT_DIR / "flowmimic" / "tools" / "sample_aist_method_comparison.py"
)
GENRE_TO_ID_PATH = ROOT_DIR / "flowmimic" / "src" / "config" / "genre_to_id.json"
COND_PREVIEW_MAX_FRAMES = int(os.environ.get("FLOWMIMIC_COND_PREVIEW_MAX_FRAMES", "24"))
COMPARISON_JOB_ROOT = OUTPUT_ROOT / "comparison_jobs"
GENERATION_JOB_ROOT = OUTPUT_ROOT / "generation_jobs"
WEB_RESULT_RETENTION = max(
    1, int(os.environ.get("FLOWMIMIC_WEB_RESULT_RETENTION", "10"))
)
COMPARISON_MLD_GPU = int(os.environ.get("FLOWMIMIC_COMPARISON_MLD_GPU", "0"))
COMPARISON_STICKMOTION_GPU = int(
    os.environ.get("FLOWMIMIC_COMPARISON_STICKMOTION_GPU", "0")
)
COMPARISON_MLD_CHECKPOINT = os.environ.get(
    "FLOWMIMIC_COMPARISON_MLD_CHECKPOINT",
    "runs/mld/mld/aist_ik263_mld_196_aistmvh_vae/checkpoints/epoch=2499.ckpt",
)
COMPARISON_MLD_CONFIG = os.environ.get(
    "FLOWMIMIC_COMPARISON_MLD_CONFIG",
    "baselines/mld/configs/mld_diffusion_aist_mvhvae.yaml",
)
COMPARISON_MLD_ASSETS_CONFIG = os.environ.get(
    "FLOWMIMIC_COMPARISON_MLD_ASSETS_CONFIG",
    "baselines/mld/configs/mld_assets_aist_aistmvh_stats.yaml",
)
COMPARISON_STICKMOTION_CHECKPOINT = os.environ.get(
    "FLOWMIMIC_COMPARISON_STICKMOTION_CHECKPOINT",
    (
        "runs/stickmotion/human_ml3d/aist_remodiffuse_no_locus_260730/"
        "epoch=591-step=25644.ckpt"
    ),
)
COMPARISON_STICKMOTION_CONFIG = os.environ.get(
    "FLOWMIMIC_COMPARISON_STICKMOTION_CONFIG",
    (
        "baselines/stickmotion/configs/"
        "stickmotion_remodiffuse_aist_no_locus_eval.py"
    ),
)
COMPARISON_BLENDER = shutil.which(
    os.environ.get("FLOWMIMIC_BLENDER", "blender")
)
DEPLOYED_CHECKPOINT_PRESETS = [
    {
        "id": "round0",
        "label": "Deployed Round 0",
        "model_name": "deployed",
        "model_filename": "round0.pt",
        "steps": 8,
        "guidance_scale": 5.0,
    },
    {
        "id": "reflow1",
        "label": "Deployed Reflow Round 1",
        "model_name": "deployed",
        "model_filename": "reflow1.pt",
        "steps": 1,
        "guidance_scale": 5.0,
    },
]
DEPLOYED_VAE_ALIASES = [
    {
        "label": "Deployed Motion VQ-VAE",
        "path": "checkpoints/vqvae/deployed/motion_vqvae.pt",
    }
]
DEPLOYED_CHECKPOINT_ALIAS_TARGETS = {
    ROOT_DIR / "checkpoints" / "flow" / "deployed" / "round0.pt": (
        ROOT_DIR
        / "checkpoints"
        / "flow"
        / "vqflow_aist_zq16_unified_sparse_cfg5_260728"
        / "flow_round0_update68220.pt"
    ),
    ROOT_DIR / "checkpoints" / "flow" / "deployed" / "reflow1.pt": (
        ROOT_DIR
        / "checkpoints"
        / "flow"
        / "vqflow_aist_zq16_reflow1_cfg5_endpoint_rollout_260822"
        / "flow_round1_update33334.pt"
    ),
    ROOT_DIR / "checkpoints" / "vqvae" / "deployed" / "motion_vqvae.pt": (
        ROOT_DIR
        / "checkpoints"
        / "vqvae"
        / "aist_mvh_len196_latent16_code1024_visible_retrain_to200_ddp2_retry_260717"
        / "motion_vqvae_epoch200.pt"
    ),
}


def _ensure_deployed_checkpoint_aliases() -> None:
    logger = logging.getLogger(__name__)
    for alias, target in DEPLOYED_CHECKPOINT_ALIAS_TARGETS.items():
        if alias.is_symlink():
            if alias.resolve(strict=False) != target.resolve(strict=False):
                logger.warning("Checkpoint alias has an unexpected target: %s", alias)
            continue
        if alias.exists():
            logger.warning("Checkpoint alias path is not a symlink: %s", alias)
            continue
        if not target.is_file():
            logger.warning("Cannot create checkpoint alias; target is missing: %s", target)
            continue
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(Path(os.path.relpath(target, alias.parent)))


_ensure_deployed_checkpoint_aliases()
BASE_PATH = os.environ.get("FLOWMIMIC_BASE_PATH", "/flowmimic").strip()
if BASE_PATH in ("", "/"):
    BASE_PATH = ""
elif not BASE_PATH.startswith("/"):
    BASE_PATH = "/" + BASE_PATH
BASE_PATH = BASE_PATH.rstrip("/")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
COMPARISON_JOB_ROOT.mkdir(parents=True, exist_ok=True)
GENERATION_JOB_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FlowMimic Web View", version="0.1.0")


@app.middleware("http")
async def _disable_web_asset_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == _prefix("/") or path.startswith(_prefix("/static/")):
        response.headers["Cache-Control"] = "no-store"
    return response


def _configure_uvicorn_timestamp_logging() -> None:
    try:
        from uvicorn.logging import AccessFormatter, DefaultFormatter
    except ImportError:
        return

    datefmt = os.environ.get("FLOWMIMIC_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S %Z")
    default_fmt = os.environ.get(
        "FLOWMIMIC_UVICORN_LOG_FMT",
        "%(asctime)s %(levelprefix)s %(message)s",
    )
    access_fmt = os.environ.get(
        "FLOWMIMIC_UVICORN_ACCESS_LOG_FMT",
        '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
    )

    for logger_name in ("uvicorn", "uvicorn.error"):
        for handler in logging.getLogger(logger_name).handlers:
            handler.setFormatter(DefaultFormatter(default_fmt, datefmt=datefmt))
    for handler in logging.getLogger("uvicorn.access").handlers:
        handler.setFormatter(AccessFormatter(access_fmt, datefmt=datefmt))


_configure_uvicorn_timestamp_logging()


@app.on_event("startup")
async def _startup_configure_logging() -> None:
    _configure_uvicorn_timestamp_logging()

GENRE_FULL_BY_ABBR = {
    "BR": "Break",
    "PO": "Pop",
    "LO": "Lock",
    "MH": "Middle Hip-hop",
    "LH": "LA style Hip-hop",
    "HO": "House",
    "WA": "Waack",
    "KR": "Krump",
    "JS": "Street Jazz",
    "JB": "Ballet Jazz",
}


def _prefix(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    if not BASE_PATH:
        return path
    return f"{BASE_PATH}{path}"


app.mount(_prefix("/static"), StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount(_prefix("/assets"), StaticFiles(directory=str(ASSET_DIR)), name="assets")
app.mount(_prefix("/files"), StaticFiles(directory=str(OUTPUT_ROOT)), name="files")


class GenerateRequest(BaseModel):
    checkpoint: str | None = None
    model_name: str | None = None
    model_filename: str | None = None
    vae_checkpoint: str | None = None
    condition_frames: int | None = None
    condition_pattern: Literal["even", "random", "boundary_gap"] = "even"
    steps: int = 8
    solver: str = "heun"
    guidance_scale: float = 1.0
    style_id: int | None = None
    domain_id: int = 0
    k2d_npy: str | None = None
    tau_cond_npy: str | None = None
    sample_path: str | None = None
    dataset: Literal["auto", "aist", "mvh"] = "aist"
    camera: str | None = None
    seed: int | None = None
    start: int | None = None
    out: str = "result_smpl22.npy"
    use_ema: bool = True
    src_fps: int | None = None
    target_fps: int | None = None
    out_dir: str = "output/flow"
    device: str | None = None


class ComparisonBlendRequest(BaseModel):
    result_id: str
    motion_filename: str = "result_smpl22.npy"
    stickmotion_sketch_frames: list[int]
    caption_index: int
    caption_text: str
    visualization_mode: Literal["skeleton", "rigged"] = "skeleton"
    device: str | None = None


class ComparisonCaptionRequest(BaseModel):
    result_id: str
    exclude_index: int | None = None


def _resolve_path(text: str | Path) -> Path:
    p = Path(text)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p.resolve()


def _file_url(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise HTTPException(
            status_code=500, detail=f"File outside output root: {resolved}"
        ) from exc
    return _prefix("/files/" + quote(rel.as_posix()))


def _resolve_output_child(relative_path: str, field_name: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}")
    resolved = (OUTPUT_ROOT / path).resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}") from exc
    return resolved


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Comparison job not found")
    return COMPARISON_JOB_ROOT / job_id


def _generation_job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Generation job not found")
    return GENERATION_JOB_ROOT / job_id


def _write_job(job_dir: Path, payload: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "job.json"
    temp_path = job_dir / "job.json.tmp"
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def _read_job(job_id: str) -> tuple[Path, dict]:
    job_dir = _job_path(job_id)
    status_path = job_dir / "job.json"
    if not status_path.is_file():
        raise HTTPException(status_code=404, detail="Comparison job not found")
    try:
        return job_dir, json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Comparison job status is unreadable") from exc


def _read_generation_job(job_id: str) -> tuple[Path, dict]:
    job_dir = _generation_job_path(job_id)
    status_path = job_dir / "job.json"
    if not status_path.is_file():
        raise HTTPException(status_code=404, detail="Generation job not found")
    try:
        return job_dir, json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Generation job status is unreadable") from exc


def _validate_device(value: str | None, field_name: str) -> str | None:
    if value is None or not value.strip():
        return None
    device = value.strip().lower()
    if not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", device):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be cpu, cuda, or cuda:<index>",
        )
    return device


def _replicate_arg(meta: dict, name: str, default=None):
    tokens = shlex.split(str(meta.get("replicate_command", "")))
    flag = f"--{name}"
    if flag not in tokens:
        return default
    index = tokens.index(flag)
    if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
        return True
    return tokens[index + 1]


def _aist_split(sample_id: str) -> str | None:
    split_root = ROOT_DIR / "data" / "AIST++" / "Annotations" / "splits"
    for split in ("test", "val"):
        path = split_root / f"pose_{split}.txt"
        if path.is_file() and sample_id in {
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        }:
            return split
    return None


def _aist_caption_options(meta: dict) -> tuple[list[dict], str]:
    if meta.get("dataset") != "aist":
        raise ValueError("Comparison captions require an AIST++ web generation")
    sample_id = Path(str(meta.get("path", ""))).stem
    match = re.fullmatch(
        r"(?P<genre>g[^_]+)_(?P<setup>s[^_]+)_c[^_]+_"
        r"(?P<dance>d[^_]+)_m[^_]+_(?P<ch>ch\d+)",
        sample_id,
    )
    if not match:
        raise ValueError(f"Unexpected AIST++ sample id: {sample_id}")
    camera = str(meta.get("camera") or "").removeprefix("c")
    if not camera:
        raise ValueError("A camera is required for comparison captions")
    parts = match.groupdict()
    stem = (
        f"{parts['genre']}_{parts['setup']}_c{camera}_{parts['dance']}_"
        f"mAll_{parts['ch']}"
    )
    token_root = ROOT_DIR / "data" / "AIST++" / "TextTokens"
    paths = {
        "mld": token_root / "mld_texts" / f"{stem}.txt",
        "text": token_root / "stickmotion" / "texts" / f"{stem}.txt",
        "token": token_root / "stickmotion" / "tokens" / f"{stem}.txt",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = {
        key: [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for key, path in paths.items()
    }
    if not rows["text"] or not (
        len(rows["mld"]) == len(rows["text"]) == len(rows["token"])
    ):
        raise ValueError(f"Caption/token count mismatch for {stem}")
    captions = []
    for index, text in enumerate(rows["text"]):
        mld_text = rows["mld"][index].split("#", 1)[0].strip()
        if text != mld_text:
            raise ValueError(f"MLD and StickMotion caption mismatch for {stem}")
        captions.append({"index": index, "text": text})
    return captions, str(paths["text"].relative_to(ROOT_DIR))


def _comparison_command(
    result_dir: Path,
    motion_path: Path,
    meta_path: Path,
    bundle_dir: Path,
    sketch_frames: list[int],
    caption_index: int,
    caption_text: str,
    visualization_mode: Literal["skeleton", "rigged"],
    device: str | None,
) -> tuple[list[str], dict]:
    if COMPARISON_BLENDER is None:
        raise ValueError(
            "Blender was not found in PATH. Install Blender or set FLOWMIMIC_BLENDER."
        )
    if visualization_mode == "rigged" and not RIGGED_MODEL_PATH.is_file():
        raise ValueError(f"Rigged SMPL22 model not found: {RIGGED_MODEL_PATH}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("dataset") != "aist":
        raise ValueError("Comparison blends require an AIST++ web generation")
    sample_id = Path(str(meta.get("path", ""))).stem
    split = _aist_split(sample_id)
    if split is None:
        raise ValueError(f"{sample_id} is not in the AIST++ test or validation split")
    seq_len = int(meta.get("seq_len") or 0)
    if seq_len != 196:
        raise ValueError(f"StickMotion comparison requires seq_len=196, got {seq_len}")
    start = int(meta.get("start") or 0)
    camera = str(meta.get("camera") or "")
    if not camera:
        raise ValueError("A camera is required for the comparison blend")
    condition_frames = int(
        meta.get("condition_frame_count")
        or len(meta.get("condition_indices") or meta.get("sparse_indices") or [])
    )
    if condition_frames < 1:
        raise ValueError("FlowMimic metadata contains no condition frames")
    flow_steps = int(
        meta.get("solver_steps") or _replicate_arg(meta, "steps", 8)
    )
    flow_solver = str(meta.get("solver") or _replicate_arg(meta, "solver", "heun"))
    flow_checkpoint = str(meta.get("flow_checkpoint") or "")
    if not flow_checkpoint:
        raise ValueError("FlowMimic metadata contains no checkpoint")
    seed = int(meta.get("seed") or 123)
    use_ema = bool(meta.get("use_ema") or _replicate_arg(meta, "use-ema", False))

    command = [
        sys.executable,
        str(COMPARISON_SCRIPT),
        "--split",
        split,
        "--sample-id",
        sample_id,
        "--camera",
        camera,
        "--start",
        str(start),
        "--seq-len",
        str(seq_len),
        "--condition-frames",
        str(condition_frames),
        "--stickmotion-sketch-frames",
        *[str(index) for index in sketch_frames],
        "--caption-index",
        str(caption_index),
        "--caption-text",
        caption_text,
        "--seed",
        str(seed),
        "--flow-steps",
        str(flow_steps),
        "--flow-solver",
        flow_solver,
        "--flow-ckpt",
        flow_checkpoint,
        "--mld-ckpt",
        COMPARISON_MLD_CHECKPOINT,
        "--mld-config",
        COMPARISON_MLD_CONFIG,
        "--mld-assets-config",
        COMPARISON_MLD_ASSETS_CONFIG,
        "--stickmotion-ckpt",
        COMPARISON_STICKMOTION_CHECKPOINT,
        "--stickmotion-config",
        COMPARISON_STICKMOTION_CONFIG,
        "--existing-flow-motion",
        str(motion_path),
        "--existing-flow-meta",
        str(meta_path),
        "--run-dir",
        str(bundle_dir),
        "--save-blend",
        "comparison.blend",
        "--blender",
        COMPARISON_BLENDER,
        "--visualization-mode",
        visualization_mode,
    ]
    if device is None:
        command.extend(
            [
                "--mld-gpu",
                str(COMPARISON_MLD_GPU),
                "--stickmotion-gpu",
                str(COMPARISON_STICKMOTION_GPU),
            ]
        )
    else:
        command.extend(
            [
                "--mld-device",
                device,
                "--stickmotion-device",
                device,
            ]
        )
    if visualization_mode == "rigged":
        command.extend(["--rigged-model", str(RIGGED_MODEL_PATH)])
    if use_ema:
        command.append("--flow-use-ema")
    return command, {
        "sample_id": sample_id,
        "split": split,
        "camera": camera,
        "clip_start": start,
        "seq_len": seq_len,
        "condition_frames": condition_frames,
        "caption_index": caption_index,
        "caption_text": caption_text,
        "stickmotion_sketch_frames": sketch_frames,
        "stickmotion_source_frames": [start + index for index in sketch_frames],
        "visualization_mode": visualization_mode,
        "device": device or "configured CUDA device",
        "mld_checkpoint": COMPARISON_MLD_CHECKPOINT,
        "stickmotion_checkpoint": COMPARISON_STICKMOTION_CHECKPOINT,
        "source_result": str(result_dir.relative_to(OUTPUT_ROOT)),
    }


def _run_comparison_job(
    job_id: str,
    result_dir: Path,
    motion_path: Path,
    meta_path: Path,
    sketch_frames: list[int],
    caption_index: int,
    caption_text: str,
    visualization_mode: Literal["skeleton", "rigged"],
    device: str | None,
) -> None:
    job_dir = COMPARISON_JOB_ROOT / job_id
    bundle_dir = job_dir / "bundle"
    log_path = job_dir / "build.log"
    status = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    try:
        command, details = _comparison_command(
            result_dir,
            motion_path,
            meta_path,
            bundle_dir,
            sketch_frames,
            caption_index,
            caption_text,
            visualization_mode,
            device,
        )
        status.update(
            {
                "status": "running",
                "stage": "Preparing comparison",
                "details": details,
                "command": shlex.join(command),
            }
        )
        _write_job(job_dir, status)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {shlex.join(command)}\n\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            status["pid"] = process.pid
            _write_job(job_dir, status)
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                stage = None
                if line.startswith("[mld]"):
                    stage = "Generating MLD motion"
                elif line.startswith("[stickmotion]"):
                    stage = "Generating StickMotion motion and sketches"
                elif line.startswith("[blender]"):
                    stage = "Building Blender scene"
                if stage and stage != status.get("stage"):
                    status["stage"] = stage
                    _write_job(job_dir, status)
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"Comparison builder exited with code {returncode}")
        blend_path = bundle_dir / "comparison.blend"
        manifest_path = bundle_dir / "comparison_manifest.json"
        if not blend_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Comparison builder did not produce the expected artifacts")
        status.update(
            {
                "status": "complete",
                "stage": "Complete",
                "blend_file": str(blend_path.relative_to(OUTPUT_ROOT)),
                "manifest_file": str(manifest_path.relative_to(OUTPUT_ROOT)),
                "log_file": str(log_path.relative_to(OUTPUT_ROOT)),
            }
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Comparison job %s failed", job_id)
        status.update(
            {
                "status": "failed",
                "stage": "Failed",
                "error": str(exc),
                "log_file": str(log_path.relative_to(OUTPUT_ROOT))
                if log_path.exists()
                else None,
            }
        )
    finally:
        status.pop("pid", None)
        _write_job(job_dir, status)


def _comparison_response(job_id: str, status: dict) -> dict:
    response = dict(status)
    response["job_id"] = job_id
    response["status_url"] = _prefix(f"/api/comparison-jobs/{job_id}")
    if status.get("status") == "complete":
        response["download_url"] = _prefix(
            f"/api/comparison-jobs/{job_id}/download"
        )
        response["results_url"] = _prefix(
            f"/api/comparison-jobs/{job_id}/results"
        )
        manifest_file = status.get("manifest_file")
        if manifest_file:
            response["manifest_url"] = _file_url(OUTPUT_ROOT / manifest_file)
    log_file = status.get("log_file")
    if log_file:
        response["log_url"] = _file_url(OUTPUT_ROOT / log_file)
    return response


def _comparison_identity(meta: dict) -> tuple[str, int] | None:
    sample_id = Path(str(meta.get("path", ""))).stem
    if not sample_id:
        return None
    try:
        start = int(meta.get("start") or 0)
    except (TypeError, ValueError):
        return None
    return sample_id, start


def _latest_matching_comparison(meta: dict, result_id: str) -> dict | None:
    identity = _comparison_identity(meta)
    if identity is None:
        return None
    sample_id, start = identity
    candidates = sorted(
        COMPARISON_JOB_ROOT.glob("*/job.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for status_path in candidates:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        details = status.get("details", {})
        try:
            clip_start = int(details.get("clip_start") or 0)
        except (TypeError, ValueError):
            continue
        if (
            status.get("status") != "complete"
            or details.get("sample_id") != sample_id
            or clip_start != start
        ):
            continue
        job_id = status_path.parent.name
        bundle_dir = status_path.parent / "bundle"
        if not (bundle_dir / "comparison_manifest.json").is_file():
            continue
        response = _comparison_response(job_id, status)
        response["matches_current_flow"] = details.get("source_result") == result_id
        return response
    return None


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _load_motion(path: Path) -> list | None:
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise HTTPException(
            status_code=500, detail=f"Unexpected motion shape in {path}: {arr.shape}"
        )
    arr = blender_to_yup(arr.astype(np.float32))
    return arr.tolist()


def _result_response(
    run_dir: Path,
    generated_motion_name: str,
    *,
    sample_run: dict | None = None,
    extract_run: dict | None = None,
    restored: bool = False,
    preview_only: bool = False,
) -> dict:
    meta_path = run_dir / "result_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Result metadata not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Result metadata is unreadable") from exc

    extract_out_dir = run_dir / "cond_media"
    preview_meta_path = extract_out_dir / "condition_preview_meta.json"
    preview_meta = {}
    if preview_meta_path.exists():
        try:
            preview_meta = json.loads(preview_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500, detail="Condition preview metadata is unreadable"
            ) from exc

    condition_indices = meta.get("condition_indices") or meta.get("sparse_indices") or []
    condition_preview_indices = meta.get("condition_preview_indices") or condition_indices
    gen_motion_path = run_dir / generated_motion_name
    cond_motion_path = extract_out_dir / "cond_clip_smpl22.npy"
    video_path = extract_out_dir / "result_clip.mp4"
    frames_dir = extract_out_dir / "result_frames"

    frame_urls = []
    if frames_dir.exists():
        frame_files = sorted(
            p
            for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        frame_urls = [_file_url(path) for path in frame_files]

    response = {
        "ok": True,
        "restored": restored,
        "result_dir": str(run_dir),
        "result_id": str(run_dir.relative_to(OUTPUT_ROOT)),
        "generated_motion_name": generated_motion_name,
        "meta": meta,
        "generated_motion": None
        if preview_only
        else _load_motion(gen_motion_path),
        "condition_motion": _load_motion(cond_motion_path),
        "generated_motion_url": _file_url(gen_motion_path)
        if gen_motion_path.exists()
        else None,
        "condition_motion_url": _file_url(cond_motion_path)
        if cond_motion_path.exists()
        else None,
        "video_url": _file_url(video_path) if video_path.exists() else None,
        "frame_urls": frame_urls,
        "condition_frame_info": {
            "total": int(meta.get("condition_frame_count") or len(condition_indices)),
            "shown": len(frame_urls),
            "limit": COND_PREVIEW_MAX_FRAMES,
            "indices": condition_indices,
            "preview_indices": preview_meta.get(
                "preview_indices", condition_preview_indices
            ),
        },
        "meta_url": _file_url(meta_path),
    }
    if not preview_only:
        comparison = _latest_matching_comparison(meta, response["result_id"])
        if comparison is not None:
            response["comparison"] = comparison
    if sample_run is not None:
        response["sample_run"] = sample_run
    if extract_run is not None:
        response["extract_run"] = extract_run
    return response


def _latest_result() -> tuple[Path, str]:
    last_link = OUTPUT_ROOT / "last"
    if not last_link.exists():
        raise HTTPException(status_code=404, detail="No generated result is available")
    try:
        run_dir = last_link.resolve(strict=True)
        run_dir.relative_to(OUTPUT_ROOT)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Latest result link is invalid") from exc

    meta_path = run_dir / "result_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Latest result metadata not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail="Latest result metadata is unreadable"
        ) from exc

    generated_motion_name = str(
        _replicate_arg(meta, "out", "result_smpl22.npy") or "result_smpl22.npy"
    )
    try:
        generated_motion_name = _validate_rel_component(
            generated_motion_name, "generated motion name"
        )
    except HTTPException:
        generated_motion_name = "result_smpl22.npy"
    generated_motion_path = run_dir / generated_motion_name
    if not generated_motion_path.is_file():
        candidates = sorted(run_dir.glob("*.npy"))
        if len(candidates) != 1:
            raise HTTPException(
                status_code=404, detail="Latest generated motion could not be identified"
            )
        generated_motion_name = candidates[0].name
    return run_dir, generated_motion_name


def _generation_response(job_id: str, status: dict) -> dict:
    response = dict(status)
    response["job_id"] = job_id
    response["status_url"] = _prefix(f"/api/generation-jobs/{job_id}")
    response["result_url"] = _prefix(f"/api/generation-jobs/{job_id}/result")
    for key in ("sample_log_file", "extract_log_file"):
        if status.get(key):
            response[key.replace("_file", "_url")] = _file_url(
                OUTPUT_ROOT / status[key]
            )
    return response


def _write_result_meta(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def _prune_web_results(retain: int = WEB_RESULT_RETENTION) -> None:
    candidates = []
    for meta_path in OUTPUT_ROOT.glob("*/*/result_meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not meta.get("web_generation_job_id"):
            continue
        candidates.append((meta_path.stat().st_mtime, meta_path.parent))
    candidates.sort(reverse=True)
    for _, run_dir in candidates[retain:]:
        try:
            run_dir.relative_to(OUTPUT_ROOT)
        except ValueError:
            continue
        shutil.rmtree(run_dir, ignore_errors=True)


def _run_generation_job(
    job_id: str,
    sample_command: list[str],
    run_dir: Path,
    generated_motion_name: str,
) -> None:
    job_dir = GENERATION_JOB_ROOT / job_id
    status_path = job_dir / "job.json"
    sample_log_path = job_dir / "sample.log"
    extract_log_path = job_dir / "extract.log"
    meta_path = run_dir / "result_meta.json"
    extract_out_dir = run_dir / "cond_media"
    preview_ready_path = extract_out_dir / "condition_preview_ready.json"
    video_path = extract_out_dir / "result_clip.mp4"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    sample_process = None
    extract_process = None
    sample_log = None
    extract_log = None
    preview_ready = False
    video_ready = False
    motion_ready = False
    extract_done = False
    extract_error = None
    try:
        status.update(
            {
                "status": "running",
                "stage": "Selecting condition frames",
                "sample_log_file": str(sample_log_path.relative_to(OUTPUT_ROOT)),
                "extract_log_file": str(extract_log_path.relative_to(OUTPUT_ROOT)),
            }
        )
        _write_job(job_dir, status)
        sample_log = sample_log_path.open("w", encoding="utf-8")
        sample_log.write(f"$ {shlex.join(sample_command)}\n\n")
        sample_log.flush()
        sample_process = subprocess.Popen(
            sample_command,
            cwd=str(ROOT_DIR),
            stdout=sample_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        status["pid"] = sample_process.pid
        _write_job(job_dir, status)

        while True:
            if extract_process is None and meta_path.is_file():
                try:
                    preview_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    preview_meta = {}
                if preview_meta.get("condition_indices"):
                    extract_command = [
                        sys.executable,
                        str(EXTRACT_SCRIPT),
                        "--meta",
                        str(meta_path),
                        "--out-dir",
                        str(extract_out_dir),
                        "--max-frames",
                        str(COND_PREVIEW_MAX_FRAMES),
                    ]
                    extract_log = extract_log_path.open("w", encoding="utf-8")
                    extract_log.write(f"$ {shlex.join(extract_command)}\n\n")
                    extract_log.flush()
                    extract_process = subprocess.Popen(
                        extract_command,
                        cwd=str(ROOT_DIR),
                        stdout=extract_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    status.update(
                        {
                            "stage": "Generating motion and condition preview",
                            "result_id": str(run_dir.relative_to(OUTPUT_ROOT)),
                            "preview_started": True,
                        }
                    )
                    _write_job(job_dir, status)

            if (
                extract_process is not None
                and not preview_ready
                and preview_ready_path.is_file()
            ):
                preview_ready = True
                status["preview_ready"] = True
                status["stage"] = "Condition frames ready; generating motion and video"
                _write_job(job_dir, status)

            if extract_process is not None and not extract_done:
                extract_returncode = extract_process.poll()
                if extract_returncode is not None:
                    extract_done = True
                    if extract_log is not None:
                        extract_log.close()
                        extract_log = None
                    if extract_returncode == 0:
                        preview_ready = preview_ready or (
                            preview_ready_path.is_file()
                            and (extract_out_dir / "condition_preview_meta.json").is_file()
                            and (extract_out_dir / "cond_clip_smpl22.npy").is_file()
                        )
                        video_ready = video_path.is_file()
                        status["preview_ready"] = preview_ready
                        status["video_ready"] = video_ready
                        status["stage"] = (
                            "Condition media ready; generating motion"
                            if not motion_ready
                            else "Complete"
                        )
                    else:
                        extract_error = (
                            f"Condition media extraction exited with code "
                            f"{extract_returncode}"
                        )
                        status["preview_error"] = extract_error
                    _write_job(job_dir, status)

            sample_returncode = sample_process.poll()
            if sample_returncode is not None and not motion_ready:
                if sample_log is not None:
                    sample_log.close()
                    sample_log = None
                if sample_returncode != 0:
                    if extract_process is not None and extract_process.poll() is None:
                        extract_process.terminate()
                        extract_process.wait(timeout=10)
                    raise RuntimeError(
                        f"FlowMimic sampler exited with code {sample_returncode}"
                    )
                if not meta_path.is_file() or not (
                    run_dir / generated_motion_name
                ).is_file():
                    raise RuntimeError(
                        "FlowMimic sampler did not produce the expected result files"
                    )
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["web_generation_job_id"] = job_id
                _write_result_meta(meta_path, meta)
                motion_ready = True
                status["motion_ready"] = True
                status["stage"] = (
                    "Complete"
                    if extract_done
                    else "Motion ready; preparing condition media"
                )
                _write_job(job_dir, status)

            if motion_ready and extract_done:
                break
            time.sleep(0.15)

        status.update(
            {
                "status": "complete",
                "stage": "Complete",
                "preview_ready": preview_ready,
                "video_ready": video_ready,
                "motion_ready": True,
            }
        )
        if extract_error:
            status["preview_error"] = extract_error
        _prune_web_results()
    except Exception as exc:
        logging.getLogger(__name__).exception("Generation job %s failed", job_id)
        for process in (sample_process, extract_process):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        status.update(
            {
                "status": "failed",
                "stage": "Failed",
                "error": str(exc),
            }
        )
    finally:
        if sample_log is not None:
            sample_log.close()
        if extract_log is not None:
            extract_log.close()
        status.pop("pid", None)
        _write_job(job_dir, status)


def _list_checkpoints(max_items: int = 300) -> list[str]:
    ckpts = sorted((ROOT_DIR / "checkpoints" / "flow").glob("**/*.pt"))
    out = []
    for p in ckpts:
        if not p.is_file():
            continue
        out.append(str(p.relative_to(ROOT_DIR)))
        if len(out) >= max_items:
            break
    return out


def _list_model_names(max_items: int = 300) -> list[str]:
    root = ROOT_DIR / "checkpoints" / "flow"
    names = []
    if not root.exists():
        return names
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        names.append(p.name)
        if len(names) >= max_items:
            break
    return names


def _validate_rel_component(text: str, field_name: str) -> str:
    p = Path(text)
    if (
        p.is_absolute()
        or len(p.parts) != 1
        or any(part in ("", ".", "..") for part in p.parts)
    ):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {text}")
    return text


def _build_style_options() -> list[dict]:
    if not GENRE_TO_ID_PATH.exists():
        return [
            {"id": None, "label": "Default"},
            {"id": 0, "label": "0: Unknown"},
        ]
    with open(GENRE_TO_ID_PATH, "r", encoding="utf-8") as f:
        genre_to_id = json.load(f)
    id_to_abbr = {}
    for abbr, gid in genre_to_id.items():
        if isinstance(gid, int):
            id_to_abbr[gid] = abbr

    options = [
        {"id": None, "label": "Default"},
        {"id": 0, "label": "0: Unknown"},
    ]
    for gid in range(1, 11):
        abbr = id_to_abbr.get(gid, "")
        full_name = GENRE_FULL_BY_ABBR.get(abbr, abbr or f"Style {gid}")
        options.append({"id": gid, "label": f"{gid}: {full_name} - {abbr}"})
    return options


@app.get(_prefix("/"), include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if BASE_PATH:

    @app.get(BASE_PATH, include_in_schema=False)
    def index_redirect() -> RedirectResponse:
        return RedirectResponse(url=_prefix("/"))


@app.get(_prefix("/api/defaults"))
def defaults() -> dict:
    config_path = ROOT_DIR / "flowmimic" / "src" / "config" / "config.json"
    cfg = {}
    if config_path.exists():
        cfg = load_config(str(config_path))
    return {
        "checkpoints": _list_checkpoints(),
        "model_names": _list_model_names(),
        "checkpoint_presets": DEPLOYED_CHECKPOINT_PRESETS,
        "vae_checkpoint_aliases": DEPLOYED_VAE_ALIASES,
        "default_checkpoint": "checkpoints/flow/deployed/round0.pt",
        "default_checkpoint_preset": "round0",
        "default_model_name": "deployed",
        "default_model_filename": "round0.pt",
        "default_vae_checkpoint": "",
        "configured_vae_checkpoint": cfg.get("vae_ckpt", ""),
        "default_condition_frames": None,
        "default_condition_pattern": "even",
        "default_steps": 8,
        "default_solver": "heun",
        "default_guidance_scale": 5.0,
        "default_dataset": "aist",
        "default_device": "",
        "default_style_id": None,
        "style_options": _build_style_options(),
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT_DIR)),
        "last_meta_exists": (OUTPUT_ROOT / "last" / "result_meta.json").exists(),
        "rigged_model_available": RIGGED_MODEL_PATH.is_file(),
        "rigged_model_url": _prefix(f"/assets/{quote(RIGGED_MODEL_PATH.name)}"),
    }


@app.post(_prefix("/api/comparison-caption"))
def random_comparison_caption(req: ComparisonCaptionRequest) -> dict:
    result_dir = _resolve_output_child(req.result_id, "result_id")
    meta_path = result_dir / "result_meta.json"
    if not result_dir.is_dir() or not meta_path.is_file():
        raise HTTPException(status_code=404, detail="FlowMimic result not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        captions, source = _aist_caption_options(meta)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    candidates = captions
    if len(captions) > 1 and req.exclude_index is not None:
        candidates = [item for item in captions if item["index"] != req.exclude_index]
    selected = secrets.choice(candidates)
    return {
        **selected,
        "count": len(captions),
        "source": source,
    }


@app.post(_prefix("/api/comparison-jobs"))
def create_comparison_job(
    req: ComparisonBlendRequest, background_tasks: BackgroundTasks
) -> dict:
    device = _validate_device(req.device, "device")
    result_dir = _resolve_output_child(req.result_id, "result_id")
    if not result_dir.is_dir():
        raise HTTPException(status_code=404, detail="FlowMimic result not found")
    motion_filename = _validate_rel_component(
        req.motion_filename, "motion_filename"
    )
    if Path(motion_filename).suffix.lower() != ".npy":
        raise HTTPException(status_code=400, detail="Motion filename must end in .npy")
    motion_path = result_dir / motion_filename
    meta_path = result_dir / "result_meta.json"
    if not motion_path.is_file() or not meta_path.is_file():
        raise HTTPException(
            status_code=404, detail="FlowMimic motion or metadata is missing"
        )

    frames = [int(index) for index in req.stickmotion_sketch_frames]
    if len(frames) != 3 or len(set(frames)) != 3:
        raise HTTPException(
            status_code=400,
            detail="Select three distinct StickMotion sketch frames",
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid FlowMimic metadata") from exc
    seq_len = int(meta.get("seq_len") or 0)
    if any(index < 0 or index >= seq_len for index in frames):
        raise HTTPException(
            status_code=400,
            detail=f"StickMotion sketch frames must be within [0, {seq_len - 1}]",
        )
    try:
        captions, _ = _aist_caption_options(meta)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not 0 <= req.caption_index < len(captions):
        raise HTTPException(
            status_code=400,
            detail=f"Caption index must be within [0, {len(captions) - 1}]",
        )
    caption_text = req.caption_text.replace("#", " ").strip()
    if not caption_text:
        raise HTTPException(status_code=400, detail="Text description is required")
    if len(caption_text) > 1000:
        raise HTTPException(
            status_code=400, detail="Text description must not exceed 1000 characters"
        )

    job_id = uuid.uuid4().hex
    job_dir = COMPARISON_JOB_ROOT / job_id
    try:
        _, details = _comparison_command(
            result_dir,
            motion_path,
            meta_path,
            job_dir / "bundle",
            frames,
            req.caption_index,
            caption_text,
            req.visualization_mode,
            device,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    status = {
        "job_id": job_id,
        "status": "queued",
        "stage": "Queued",
        "details": details,
    }
    _write_job(job_dir, status)
    background_tasks.add_task(
        _run_comparison_job,
        job_id,
        result_dir,
        motion_path,
        meta_path,
        frames,
        req.caption_index,
        caption_text,
        req.visualization_mode,
        device,
    )
    return _comparison_response(job_id, status)


@app.get(_prefix("/api/comparison-jobs/{job_id}"))
def comparison_job_status(job_id: str) -> dict:
    _, status = _read_job(job_id)
    return _comparison_response(job_id, status)


@app.get(_prefix("/api/comparison-jobs/{job_id}/download"))
def download_comparison_blend(job_id: str) -> FileResponse:
    _, status = _read_job(job_id)
    if status.get("status") != "complete" or not status.get("blend_file"):
        raise HTTPException(status_code=409, detail="Comparison blend is not ready")
    blend_path = _resolve_output_child(status["blend_file"], "blend_file")
    if not blend_path.is_file():
        raise HTTPException(status_code=404, detail="Comparison blend not found")
    sample_id = status.get("details", {}).get("sample_id", "flowmimic")
    return FileResponse(
        blend_path,
        media_type="application/octet-stream",
        filename=f"{sample_id}_comparison.blend",
    )


@app.get(_prefix("/api/comparison-jobs/{job_id}/results"))
def comparison_job_results(job_id: str) -> dict:
    job_dir, status = _read_job(job_id)
    if status.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Comparison results are not ready")
    bundle_dir = job_dir / "bundle"
    manifest_path = bundle_dir / "comparison_manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Comparison manifest not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "reference_motion": bundle_dir / "reference.npy",
        "mld_motion": bundle_dir / "mld.npy",
        "stickmotion_motion": bundle_dir / "stickmotion.npy",
        "stickman_tracks": bundle_dir / "stickman_tracks.npy",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison artifacts are missing: {', '.join(missing)}",
        )
    tracks = np.load(required["stickman_tracks"]).astype(np.float32)
    if tracks.ndim != 4 or tracks.shape[1:] != (6, 64, 2):
        raise HTTPException(
            status_code=500, detail=f"Unexpected StickMotion tracks: {tracks.shape}"
        )
    stick = manifest.get("methods", {}).get("stickmotion", {})
    mld = manifest.get("methods", {}).get("mld", {})
    return {
        "job_id": job_id,
        "sample_id": manifest.get("sample_id", ""),
        "clip_start": manifest.get("clip_start", 0),
        "reference_motion": _load_motion(required["reference_motion"]),
        "mld_motion": _load_motion(required["mld_motion"]),
        "stickmotion_motion": _load_motion(required["stickmotion_motion"]),
        "mld_text": mld.get("text", manifest.get("caption", {}).get("text", "")),
        "stickmotion_text": stick.get(
            "text", manifest.get("caption", {}).get("text", "")
        ),
        "stickman_tracks": tracks.tolist(),
        "stickman_frame_indices": stick.get("stickman_frame_indices", []),
        "stickman_source_frame_indices": stick.get(
            "stickman_source_frame_indices", []
        ),
        "manifest_url": _file_url(manifest_path),
    }


@app.get(_prefix("/api/results/latest"))
def latest_generated_result() -> dict:
    run_dir, generated_motion_name = _latest_result()
    return _result_response(
        run_dir,
        generated_motion_name,
        restored=True,
    )


@app.get(_prefix("/api/generation-jobs/{job_id}"))
def generation_job_status(job_id: str) -> dict:
    _, status = _read_generation_job(job_id)
    return _generation_response(job_id, status)


@app.get(_prefix("/api/generation-jobs/{job_id}/result"))
def generation_job_result(job_id: str, preview_only: bool = False) -> dict:
    _, status = _read_generation_job(job_id)
    if not status.get("preview_ready") and not status.get("motion_ready"):
        raise HTTPException(status_code=409, detail="Generation result is not ready")
    result_id = status.get("result_id")
    if not result_id:
        raise HTTPException(status_code=409, detail="Generation result is unresolved")
    run_dir = _resolve_output_child(result_id, "result_id")
    response = _result_response(
        run_dir,
        str(status.get("generated_motion_name") or "result_smpl22.npy"),
        preview_only=preview_only,
    )
    # FFmpeg creates the output path before the MP4 is finalized. Do not expose
    # that path to the browser until the extractor has exited successfully.
    if preview_only and not status.get("video_ready"):
        response["video_url"] = None
    return response


@app.post(_prefix("/api/generate"))
def generate(req: GenerateRequest, background_tasks: BackgroundTasks) -> dict:
    out_root = _resolve_path(req.out_dir)
    if out_root != OUTPUT_ROOT:
        raise HTTPException(
            status_code=400,
            detail=f"Only out_dir='output/flow' is supported by this web module, got: {req.out_dir}",
        )

    checkpoint = (req.checkpoint or "").strip()
    if not checkpoint:
        model_name = (req.model_name or "").strip()
        model_filename = (req.model_filename or "").strip()
        if not model_name or not model_filename:
            raise HTTPException(
                status_code=400,
                detail="Provide checkpoint or both model_name and model_filename.",
            )
        model_name = _validate_rel_component(model_name, "model_name")
        model_filename = _validate_rel_component(model_filename, "model_filename")
        checkpoint = f"checkpoints/flow/{model_name}/{model_filename}"

    if not math.isfinite(req.guidance_scale) or not 0.0 <= req.guidance_scale <= 5.0:
        raise HTTPException(
            status_code=400,
            detail="guidance_scale must be a finite value between 0.0 and 5.0",
        )
    device = _validate_device(req.device, "device")
    generated_motion_name = _validate_rel_component(req.out, "output name")
    if Path(generated_motion_name).suffix.lower() != ".npy":
        raise HTTPException(status_code=400, detail="Output name must end in .npy")
    output_model_name = _validate_rel_component(
        Path(checkpoint).parent.name or "model", "checkpoint model name"
    )
    run_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    result_id = f"{output_model_name}/{run_tag}"
    run_dir = OUTPUT_ROOT / output_model_name / run_tag

    sample_cmd = [
        sys.executable,
        str(SAMPLE_SCRIPT),
        "--checkpoint",
        checkpoint,
        "--steps",
        str(req.steps),
        "--solver",
        req.solver,
        "--guidance-scale",
        str(req.guidance_scale),
        "--cond-pattern",
        req.condition_pattern,
        "--dataset",
        req.dataset,
        "--out",
        generated_motion_name,
        "--out-dir",
        req.out_dir,
        "--run-tag",
        run_tag,
    ]
    if req.condition_frames is not None:
        if req.condition_frames < 1:
            raise HTTPException(
                status_code=400, detail="condition_frames must be at least 1"
            )
        sample_cmd.extend(["--cond-frames", str(req.condition_frames)])
    if req.vae_checkpoint:
        sample_cmd.extend(["--vae-checkpoint", req.vae_checkpoint])
    if req.k2d_npy:
        sample_cmd.extend(["--k2d-npy", req.k2d_npy])
    if req.tau_cond_npy:
        sample_cmd.extend(["--tau-cond-npy", req.tau_cond_npy])
    if req.sample_path:
        sample_cmd.extend(["--sample-path", req.sample_path])
    if req.camera:
        sample_cmd.extend(["--camera", req.camera])
    if req.seed is not None:
        sample_cmd.extend(["--seed", str(req.seed)])
    if req.start is not None:
        sample_cmd.extend(["--start", str(req.start)])
    if req.style_id is not None:
        sample_cmd.extend(["--style-id", str(req.style_id)])
    if req.domain_id != 0:
        sample_cmd.extend(["--domain-id", str(req.domain_id)])
    if req.use_ema:
        sample_cmd.append("--use-ema")
    if req.src_fps is not None:
        sample_cmd.extend(["--src-fps", str(req.src_fps)])
    if req.target_fps is not None:
        sample_cmd.extend(["--target-fps", str(req.target_fps)])
    if device:
        sample_cmd.extend(["--device", device])

    job_id = uuid.uuid4().hex
    job_dir = GENERATION_JOB_ROOT / job_id
    status = {
        "job_id": job_id,
        "status": "queued",
        "stage": "Queued",
        "result_id": result_id,
        "generated_motion_name": generated_motion_name,
        "preview_ready": False,
        "video_ready": False,
        "motion_ready": False,
    }
    _write_job(job_dir, status)
    background_tasks.add_task(
        _run_generation_job,
        job_id,
        sample_cmd,
        run_dir,
        generated_motion_name,
    )
    return _generation_response(job_id, status)
