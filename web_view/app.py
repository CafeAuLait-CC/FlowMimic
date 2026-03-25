from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from flowmimic.src.data.dataloader import blender_to_yup


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
OUTPUT_ROOT = (ROOT_DIR / "output" / "flow").resolve()
SAMPLE_SCRIPT = ROOT_DIR / "flowmimic" / "scripts" / "sample_flow.py"
EXTRACT_SCRIPT = ROOT_DIR / "flowmimic" / "tools" / "extract_cond_media.py"
GENRE_TO_ID_PATH = ROOT_DIR / "flowmimic" / "src" / "config" / "genre_to_id.json"
BASE_PATH = os.environ.get("FLOWMIMIC_BASE_PATH", "/flowmimic").strip()
if BASE_PATH in ("", "/"):
    BASE_PATH = ""
elif not BASE_PATH.startswith("/"):
    BASE_PATH = "/" + BASE_PATH
BASE_PATH = BASE_PATH.rstrip("/")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FlowMimic Web View", version="0.1.0")

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
        raise HTTPException(status_code=500, detail=f"File outside output root: {resolved}") from exc
    return _prefix("/files/" + quote(rel.as_posix()))


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
        raise HTTPException(status_code=500, detail=f"Unexpected motion shape in {path}: {arr.shape}")
    arr = blender_to_yup(arr.astype(np.float32))
    return arr.tolist()


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
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    return {
        "checkpoints": _list_checkpoints(),
        "model_names": _list_model_names(),
        "default_checkpoint": "",
        "default_model_name": "",
        "default_model_filename": "flow_round0_last.pt",
        "default_vae_checkpoint": cfg.get("vae_ckpt", ""),
        "default_steps": 8,
        "default_solver": "heun",
        "default_dataset": "auto",
        "default_device": "",
        "default_style_id": None,
        "style_options": _build_style_options(),
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT_DIR)),
        "last_meta_exists": (OUTPUT_ROOT / "last" / "result_meta.json").exists(),
    }


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
            detail={"stage": "sample_flow", "error": "output/flow/last not found", "run": sample_run},
        )
    run_dir = last_link.resolve()
    meta_path = run_dir / "result_meta.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=500,
            detail={"stage": "sample_flow", "error": "result_meta.json not found", "run_dir": str(run_dir)},
        )

    extract_out_dir = run_dir / "cond_media"
    extract_cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--meta",
        str(meta_path),
        "--out-dir",
        str(extract_out_dir),
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

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    gen_motion_path = run_dir / req.out
    cond_motion_path = extract_out_dir / "cond_clip_smpl22.npy"
    video_path = extract_out_dir / "result_clip.mp4"
    frames_dir = extract_out_dir / "result_frames"

    frame_urls = []
    if frames_dir.exists():
        frame_files = sorted(
            [
                p
                for p in frames_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        frame_urls = [_file_url(p) for p in frame_files]

    return {
        "ok": True,
        "sample_run": sample_run,
        "extract_run": extract_run,
        "result_dir": str(run_dir),
        "meta": meta,
        "generated_motion": _load_motion(gen_motion_path),
        "condition_motion": _load_motion(cond_motion_path),
        "generated_motion_url": _file_url(gen_motion_path) if gen_motion_path.exists() else None,
        "condition_motion_url": _file_url(cond_motion_path) if cond_motion_path.exists() else None,
        "video_url": _file_url(video_path) if video_path.exists() else None,
        "frame_urls": frame_urls,
        "meta_url": _file_url(meta_path),
    }
