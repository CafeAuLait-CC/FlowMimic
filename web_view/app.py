from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import uuid
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
OUTPUT_ROOT = (ROOT_DIR / "output" / "flow").resolve()
SAMPLE_SCRIPT = ROOT_DIR / "flowmimic" / "scripts" / "sample_flow.py"
EXTRACT_SCRIPT = ROOT_DIR / "flowmimic" / "tools" / "extract_cond_media.py"
COMPARISON_SCRIPT = (
    ROOT_DIR / "flowmimic" / "tools" / "sample_aist_method_comparison.py"
)
GENRE_TO_ID_PATH = ROOT_DIR / "flowmimic" / "src" / "config" / "genre_to_id.json"
COND_PREVIEW_MAX_FRAMES = int(os.environ.get("FLOWMIMIC_COND_PREVIEW_MAX_FRAMES", "24"))
COMPARISON_JOB_ROOT = OUTPUT_ROOT / "comparison_jobs"
COMPARISON_MLD_GPU = int(os.environ.get("FLOWMIMIC_COMPARISON_MLD_GPU", "0"))
COMPARISON_STICKMOTION_GPU = int(
    os.environ.get("FLOWMIMIC_COMPARISON_STICKMOTION_GPU", "0")
)
COMPARISON_BLENDER = shutil.which(
    os.environ.get("FLOWMIMIC_BLENDER", "blender")
)
BASE_PATH = os.environ.get("FLOWMIMIC_BASE_PATH", "/flowmimic").strip()
if BASE_PATH in ("", "/"):
    BASE_PATH = ""
elif not BASE_PATH.startswith("/"):
    BASE_PATH = "/" + BASE_PATH
BASE_PATH = BASE_PATH.rstrip("/")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
COMPARISON_JOB_ROOT.mkdir(parents=True, exist_ok=True)

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
app.mount(_prefix("/files"), StaticFiles(directory=str(OUTPUT_ROOT)), name="files")


class GenerateRequest(BaseModel):
    checkpoint: str | None = None
    model_name: str | None = None
    model_filename: str | None = None
    vae_checkpoint: str | None = None
    condition_frames: int | None = None
    steps: int = 8
    solver: str = "heun"
    style_id: int | None = None
    domain_id: int = 0
    k2d_npy: str | None = None
    tau_cond_npy: str | None = None
    sample_path: str | None = None
    dataset: Literal["auto", "aist", "mvh"] = "auto"
    camera: str | None = None
    seed: int | None = None
    start: int | None = None
    out: str = "result_smpl22.npy"
    use_ema: bool = False
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
) -> tuple[list[str], dict]:
    if COMPARISON_BLENDER is None:
        raise ValueError(
            "Blender was not found in PATH. Install Blender or set FLOWMIMIC_BLENDER."
        )
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
        "--existing-flow-motion",
        str(motion_path),
        "--existing-flow-meta",
        str(meta_path),
        "--mld-gpu",
        str(COMPARISON_MLD_GPU),
        "--stickmotion-gpu",
        str(COMPARISON_STICKMOTION_GPU),
        "--run-dir",
        str(bundle_dir),
        "--save-blend",
        "comparison.blend",
        "--blender",
        COMPARISON_BLENDER,
    ]
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
        "generated_motion": _load_motion(gen_motion_path),
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
        "default_checkpoint": "",
        "default_model_name": "",
        "default_model_filename": "flow_round0_last.pt",
        "default_vae_checkpoint": "",
        "configured_vae_checkpoint": cfg.get("vae_ckpt", ""),
        "default_condition_frames": None,
        "default_steps": 8,
        "default_solver": "heun",
        "default_dataset": "auto",
        "default_device": "",
        "default_style_id": None,
        "style_options": _build_style_options(),
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT_DIR)),
        "last_meta_exists": (OUTPUT_ROOT / "last" / "result_meta.json").exists(),
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


@app.post(_prefix("/api/generate"))
def generate(req: GenerateRequest) -> dict:
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

    sample_cmd = [
        sys.executable,
        str(SAMPLE_SCRIPT),
        "--checkpoint",
        checkpoint,
        "--steps",
        str(req.steps),
        "--solver",
        req.solver,
        "--dataset",
        req.dataset,
        "--out",
        req.out,
        "--out-dir",
        req.out_dir,
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
    if req.device:
        sample_cmd.extend(["--device", req.device])

    sample_run = _run(sample_cmd)
    if sample_run["returncode"] != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "stage": "sample_flow",
                "run": sample_run,
            },
        )

    last_link = OUTPUT_ROOT / "last"
    if not last_link.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "stage": "sample_flow",
                "error": "output/flow/last not found",
                "run": sample_run,
            },
        )
    run_dir = last_link.resolve()
    meta_path = run_dir / "result_meta.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "stage": "sample_flow",
                "error": "result_meta.json not found",
                "run_dir": str(run_dir),
            },
        )

    extract_out_dir = run_dir / "cond_media"
    extract_cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--meta",
        str(meta_path),
        "--out-dir",
        str(extract_out_dir),
        "--max-frames",
        str(COND_PREVIEW_MAX_FRAMES),
    ]
    if req.target_fps is not None:
        extract_cmd.extend(["--fps", str(req.target_fps)])
    extract_run = _run(extract_cmd)
    if extract_run["returncode"] != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "stage": "extract_cond_media",
                "run": extract_run,
                "sample_run": sample_run,
            },
        )

    return _result_response(
        run_dir,
        req.out,
        sample_run=sample_run,
        extract_run=extract_run,
    )
