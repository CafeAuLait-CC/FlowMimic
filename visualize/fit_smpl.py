#!/usr/bin/env python
"""Fit SMPL meshes to FlowMimic SMPL22 joint sequences.

This file adapts the fitting flow from MLD:
https://github.com/ChenFengYe/motion-latent-diffusion

Main differences from MLD's ``fit.py``:
- device selection is explicit instead of hard-coded;
- MLD dependency paths are resolved from ``--mld-root``;
- FlowMimic's saved Blender-space joints can be converted back to Y-up;
- mesh outputs and render inputs use one predictable output directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import multiprocessing as mp
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path

import h5py
import joblib
import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_mld_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = _repo_root() / root
    return root.resolve()


def _prepare_mld_imports(mld_root: Path, *, suppress_inner_progress: bool = True):
    joints2rots = mld_root / "mld" / "transforms" / "joints2rots"
    if not joints2rots.exists():
        raise FileNotFoundError(f"MLD joints2rots folder not found: {joints2rots}")

    for path in (mld_root, joints2rots):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    import config as mld_config  # type: ignore
    import smplify as smplify_module  # type: ignore

    if suppress_inner_progress:
        smplify_module.tqdm = lambda iterable, *args, **kwargs: iterable
    SMPLify3D = smplify_module.SMPLify3D

    model_root = mld_root / "deps" / "smpl_models"
    mld_config.SMPL_MODEL_DIR = str(model_root)
    mld_config.GMM_MODEL_DIR = str(model_root)
    mld_config.SMPL_MEAN_FILE = str(model_root / "neutral_smpl_mean_params.h5")
    mld_config.Part_Seg_DIR = str(model_root / "smplx_parts_segm.pkl")
    return mld_config, SMPLify3D


def blender_to_yup(joints: np.ndarray) -> np.ndarray:
    x = joints[..., 0]
    y = joints[..., 1]
    z = joints[..., 2]
    return np.stack([x, z, -y], axis=-1)


def _discover_inputs(input_path: str | None, input_dir: str | None) -> list[Path]:
    if not input_path and not input_dir:
        input_path = "output/flow/last/result_smpl22.npy"

    paths: list[Path] = []
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = _repo_root() / path
        paths.append(path.resolve())

    if input_dir:
        root = Path(input_dir)
        if not root.is_absolute():
            root = _repo_root() / root
        paths.extend(sorted(p.resolve() for p in root.glob("*.npy")))

    return paths


def _load_joints(path: Path, input_space: str, max_frames: int | None) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 3 or data.shape[1:] != (22, 3):
        raise ValueError(f"Expected {path} to have shape (T, 22, 3), got {data.shape}")
    if input_space == "blender":
        data = blender_to_yup(data)
    elif input_space != "yup":
        raise ValueError(f"Unknown input space: {input_space}")
    if max_frames is not None:
        data = data[:max_frames]
    return np.asarray(data, dtype=np.float32)


def _load_mean_params(mean_path: str, batch_size: int, device: torch.device):
    with h5py.File(mean_path, "r") as f:
        init_mean_pose = (
            torch.from_numpy(f["pose"][:])
            .unsqueeze(0)
            .float()
            .repeat(batch_size, 1)
            .to(device)
        )
        init_mean_shape = (
            torch.from_numpy(f["shape"][:])
            .unsqueeze(0)
            .float()
            .repeat(batch_size, 1)
            .to(device)
        )
    return init_mean_pose, init_mean_shape


def _validate_smpl_model_file(model_root: Path, gender: str) -> Path:
    model_path = model_root / "smpl" / f"SMPL_{gender.upper()}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"SMPL model file not found: {model_path}")

    required = {
        "J",
        "J_regressor",
        "bs_style",
        "bs_type",
        "f",
        "kintree_table",
        "posedirs",
        "shapedirs",
        "v_template",
        "weights",
    }
    with model_path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            data = pickle.load(f, encoding="latin1")
    if not isinstance(data, dict):
        raise ValueError(f"{model_path} must contain a dict, got {type(data).__name__}.")

    missing = sorted(required.difference(data))
    if missing:
        keys = ", ".join(sorted(data.keys()))
        raise ValueError(
            f"{model_path} is not a SMPL model-definition pickle. "
            f"Missing required keys: {', '.join(missing)}. Present keys: {keys}"
        )

    v_template = np.asarray(data["v_template"])
    faces = np.asarray(data["f"])
    if v_template.shape != (6890, 3) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"{model_path} has incompatible topology: "
            f"v_template={v_template.shape}, f={faces.shape}."
        )
    return model_path


def _merge_plys_to_npy(ply_dir: Path, out_path: Path) -> None:
    paths = sorted(
        p for p in ply_dir.glob("*.ply") if not p.name.endswith("_gt.ply")
    )
    if not paths:
        raise FileNotFoundError(f"No PLY files found in {ply_dir}")

    frames = []
    for path in paths:
        mesh = trimesh.load_mesh(path, process=False)
        frames.append(np.asarray(mesh.vertices, dtype=np.float32))

    mesh_arr = np.stack(frames, axis=0)
    np.save(out_path, mesh_arr)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA requested with {device_arg}, but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def _visible_cuda_devices() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def _expand_device_token(token: str) -> list[str]:
    token = token.strip().lower()
    if not token:
        return []
    if token == "auto":
        return ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
    if token in {"all", "cuda"}:
        cuda_devices = _visible_cuda_devices()
        if cuda_devices:
            return cuda_devices
        print(f"CUDA requested with {token}, but CUDA is unavailable; using CPU.")
        return ["cpu"]
    if token.isdigit():
        token = f"cuda:{token}"
    if token.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA requested with {token}, but CUDA is unavailable; using CPU.")
        return ["cpu"]
    return [token]


def _resolve_fit_devices(device_arg: str, fit_devices_arg: str | None) -> list[str]:
    raw = fit_devices_arg if fit_devices_arg else device_arg
    devices: list[str] = []
    for token in raw.split(","):
        devices.extend(_expand_device_token(token))

    deduped: list[str] = []
    for device in devices:
        if device not in deduped:
            deduped.append(device)
    if not deduped:
        return ["cpu"]
    return deduped


def _fit_frame_chunk(payload: dict) -> list[tuple[int, float, np.ndarray]]:
    args = argparse.Namespace(**payload["args"])
    worker_threads = payload["worker_threads"]
    if worker_threads is not None:
        torch.set_num_threads(worker_threads)
        try:
            torch.set_num_interop_threads(max(1, min(worker_threads, 16)))
        except RuntimeError:
            pass

    mld_root = Path(payload["mld_root"])
    mld_config, SMPLify3D = _prepare_mld_imports(
        mld_root, suppress_inner_progress=not args.show_inner_progress
    )
    device = _select_device(payload["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    frame_dir = Path(payload["frame_dir"])
    input_name = payload["input_name"]
    frame_ids = payload["frame_ids"]
    joints_chunk = payload["joints"]
    progress_queue = payload.get("progress_queue")
    progress_path = payload.get("progress_path")

    batch_size = 1
    model_root = mld_root / "deps" / "smpl_models"
    _validate_smpl_model_file(model_root, args.gender)
    create_context = (
        contextlib.nullcontext()
        if args.worker_log
        else contextlib.redirect_stdout(io.StringIO())
    )
    with create_context:
        smpl_model = smplx.create(
            str(model_root),
            model_type="smpl",
            gender=args.gender,
            ext="pkl",
            batch_size=batch_size,
        ).to(device)

    init_mean_pose, init_mean_shape = _load_mean_params(
        mld_config.SMPL_MEAN_FILE, batch_size, device
    )
    cam_trans_zero = torch.zeros(batch_size, 3, device=device)
    smplify = SMPLify3D(
        smplxmodel=smpl_model,
        batch_size=batch_size,
        joints_category="AMASS",
        num_iters=args.num_smplify_iters,
        use_lbfgs=args.optimizer == "lbfgs",
        device=device,
    )

    pred_pose = torch.zeros(batch_size, 72, device=device)
    pred_betas = torch.zeros(batch_size, 10, device=device)
    pred_cam_t = torch.zeros(batch_size, 3, device=device)
    keypoints_3d = torch.zeros(batch_size, 22, 3, device=device)
    confidence_input = torch.ones(22, device=device)
    if args.fix_foot:
        confidence_input[[7, 8, 10, 11]] = 1.5

    prev_pose = None
    prev_betas = None
    prev_cam_t = None
    results: list[tuple[int, float, np.ndarray]] = []
    for local_idx, (frame_idx, joints3d) in enumerate(zip(frame_ids, joints_chunk)):
        if args.worker_log:
            print(
                f"Fitting {input_name} frame {int(frame_idx) + 1} "
                f"({local_idx + 1}/{len(frame_ids)})"
            )
        keypoints_3d[0].copy_(torch.from_numpy(joints3d).to(device=device))
        if prev_pose is None:
            pred_pose[0].copy_(init_mean_pose[0])
            pred_betas[0].copy_(init_mean_shape[0])
            pred_cam_t[0].copy_(cam_trans_zero[0])
        else:
            pred_pose.copy_(prev_pose)
            pred_betas.copy_(prev_betas)
            pred_cam_t.copy_(prev_cam_t)

        (
            _new_opt_vertices,
            _new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_cam_t,
            new_opt_joint_loss,
        ) = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input,
            seq_ind=local_idx,
        )

        output = smpl_model(
            betas=new_opt_betas,
            global_orient=new_opt_pose[:, :3],
            body_pose=new_opt_pose[:, 3:],
            transl=new_opt_cam_t,
            return_verts=True,
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze().astype(np.float32)
        loss = float(new_opt_joint_loss.detach().cpu().item())
        results.append((int(frame_idx), loss, vertices))
        if progress_queue is not None:
            progress_queue.put((int(frame_idx), loss))
        if progress_path is not None:
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"frame": int(frame_idx), "loss": loss}) + "\n")

        if args.save_frame_files:
            ply_path = frame_dir / f"motion_{int(frame_idx):04d}.ply"
            pkl_path = frame_dir / f"motion_{int(frame_idx):04d}.pkl"
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=smpl_model.faces,
                process=False,
            )
            mesh.export(ply_path)
            joblib.dump(
                {
                    "beta": new_opt_betas.detach().cpu().numpy(),
                    "pose": new_opt_pose.detach().cpu().numpy(),
                    "cam": new_opt_cam_t.detach().cpu().numpy(),
                },
                pkl_path,
                compress=3,
            )
            del mesh

        prev_pose = new_opt_pose.detach().clone()
        prev_betas = new_opt_betas.detach().clone()
        prev_cam_t = new_opt_cam_t.detach().clone()

        del output, new_opt_pose, new_opt_betas, new_opt_cam_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def _split_frame_chunks(total_frames: int, num_workers: int) -> list[tuple[list[int], slice]]:
    num_workers = max(1, min(num_workers, total_frames))
    chunks: list[tuple[list[int], slice]] = []
    base = total_frames // num_workers
    remainder = total_frames % num_workers
    start = 0
    for worker_idx in range(num_workers):
        length = base + (1 if worker_idx < remainder else 0)
        end = start + length
        if start < end:
            chunks.append((list(range(start, end)), slice(start, end)))
        start = end
    return chunks


class _LocalProgressQueue:
    def __init__(self, pbar: tqdm) -> None:
        self.pbar = pbar

    def put(self, item) -> None:
        self.pbar.update(1)


def _poll_progress_files(
    progress_paths: list[Path],
    seen_counts: dict[Path, int],
    pbar: tqdm,
) -> None:
    for path in progress_paths:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        seen = seen_counts.get(path, 0)
        if len(lines) <= seen:
            continue
        for line in lines[seen:]:
            if line.strip():
                pbar.update(1)
        seen_counts[path] = len(lines)


def _fit_one_sequence(
    *,
    input_path: Path,
    out_root: Path,
    joints: np.ndarray,
    args: argparse.Namespace,
    mld_config,
    SMPLify3D,
    device: torch.device,
) -> Path:
    sequence_dir = out_root / input_path.stem
    frame_dir = sequence_dir / "fit_frames"
    mesh_path = sequence_dir / f"{input_path.stem}_mesh.npy"
    faces_path = sequence_dir / f"{input_path.stem}_faces.npy"
    manifest_path = sequence_dir / "fit_manifest.json"
    fit_input_path = sequence_dir / f"{input_path.stem}_fit_input.npy"
    fit_devices = _resolve_fit_devices(args.device, args.fit_devices)

    batch_size = 1
    model_root = _resolve_mld_root(args.mld_root) / "deps" / "smpl_models"
    smpl_model_path = _validate_smpl_model_file(model_root, args.gender)
    print(f"SMPL model file: {smpl_model_path}")
    smpl_model = smplx.create(
        str(model_root),
        model_type="smpl",
        gender=args.gender,
        ext="pkl",
        batch_size=batch_size,
    ).to(torch.device("cpu"))
    smpl_faces = np.asarray(smpl_model.faces, dtype=np.int32)

    if mesh_path.exists() and not args.overwrite:
        sequence_dir.mkdir(parents=True, exist_ok=True)
        if not faces_path.exists():
            np.save(faces_path, smpl_faces)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {}
        manifest.update(
            {
                "source": "https://github.com/ChenFengYe/motion-latent-diffusion",
                "input_path": str(input_path),
                "input_space": args.input_space,
                "mesh_path": str(mesh_path),
                "faces_path": str(faces_path),
                "gender": args.gender,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Mesh already exists, skipping fit: {mesh_path}")
        print(f"Saved SMPL faces sidecar: {faces_path}")
        return mesh_path

    if args.overwrite and sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.save(fit_input_path, joints)
    np.save(faces_path, smpl_faces)

    start_time = time.time()
    if args.fit_workers <= 0:
        fit_workers = max(1, len(fit_devices))
    else:
        fit_workers = args.fit_workers
    fit_workers = max(1, min(fit_workers, len(joints)))
    worker_threads = args.worker_threads
    if worker_threads is None and args.num_threads is not None and fit_workers > 1:
        worker_threads = max(1, args.num_threads // fit_workers)
    cuda_fit_devices = [dev for dev in fit_devices if dev.startswith("cuda")]
    if fit_workers > len(fit_devices) and cuda_fit_devices:
        print(
            "Warning: more fit workers than CUDA devices; some workers will "
            "share a GPU. This may be slower or run out of memory."
        )
    print(
        f"Fitting with {fit_workers} worker(s), optimizer={args.optimizer}, "
        f"iters={args.num_smplify_iters}, worker_threads={worker_threads}"
    )
    print(f"Fit devices: {', '.join(fit_devices)}")

    payloads = []
    progress_dir = sequence_dir / "fit_progress"
    if progress_dir.exists():
        shutil.rmtree(progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    for worker_id, (frame_ids, frame_slice) in enumerate(
        _split_frame_chunks(len(joints), fit_workers)
    ):
        payloads.append(
            {
                "args": vars(args),
                "mld_root": str(_resolve_mld_root(args.mld_root)),
                "input_name": input_path.name,
                "frame_dir": str(frame_dir),
                "joints": joints[frame_slice],
                "frame_ids": frame_ids,
                "device": fit_devices[worker_id % len(fit_devices)],
                "worker_threads": worker_threads,
                "worker_id": worker_id,
                "progress_path": str(progress_dir / f"worker_{worker_id:03d}.jsonl"),
            }
        )

    fit_results: list[tuple[int, float, np.ndarray]] = []
    progress_desc = f"Fitting {input_path.name}"
    if fit_workers == 1:
        with tqdm(total=len(joints), desc=progress_desc, unit="frame") as pbar:
            payloads[0]["progress_queue"] = _LocalProgressQueue(pbar)
            fit_results.extend(_fit_frame_chunk(payloads[0]))
    else:
        ctx = mp.get_context("spawn")
        progress_paths = [Path(payload["progress_path"]) for payload in payloads]
        seen_counts = {path: 0 for path in progress_paths}
        with tqdm(total=len(joints), desc=progress_desc, unit="frame") as pbar:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=fit_workers, mp_context=ctx
            ) as executor:
                futures = [
                    executor.submit(_fit_frame_chunk, payload)
                    for payload in payloads
                ]
                pending = set(futures)
                while pending:
                    _poll_progress_files(progress_paths, seen_counts, pbar)
                    done = {future for future in pending if future.done()}
                    for future in done:
                        pending.remove(future)
                        fit_results.extend(future.result())
                    if pending:
                        time.sleep(0.1)
                _poll_progress_files(progress_paths, seen_counts, pbar)

    fit_results.sort(key=lambda item: item[0])
    frame_losses = [float(loss) for _idx, loss, _vertices in fit_results]
    mesh_arr = np.stack([vertices for _idx, _loss, vertices in fit_results], axis=0)
    np.save(mesh_path, mesh_arr)
    elapsed = time.time() - start_time
    manifest = {
        "source": "https://github.com/ChenFengYe/motion-latent-diffusion",
        "input_path": str(input_path),
        "input_space": args.input_space,
        "fit_input_path": str(fit_input_path),
        "mesh_path": str(mesh_path),
        "faces_path": str(faces_path),
        "frame_dir": str(frame_dir),
        "frames": int(joints.shape[0]),
        "gender": args.gender,
        "device": args.device,
        "fit_devices": fit_devices,
        "optimizer": args.optimizer,
        "num_smplify_iters": int(args.num_smplify_iters),
        "fit_workers": int(fit_workers),
        "worker_threads": worker_threads,
        "save_frame_files": bool(args.save_frame_files),
        "fix_foot": bool(args.fix_foot),
        "frame_losses": frame_losses,
        "elapsed_sec": elapsed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved SMPL mesh sequence: {mesh_path}")
    print(f"Saved fit manifest: {manifest_path}")
    return mesh_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None, help="Input (T,22,3) npy.")
    parser.add_argument("--dir", type=str, default=None, help="Directory of input npy files.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/flow/last/visualize",
        help="Output folder for fit products.",
    )
    parser.add_argument(
        "--mld-root",
        type=str,
        default="motion-latent-diffusion",
        help="Path to the MLD repository.",
    )
    parser.add_argument(
        "--input-space",
        choices=("blender", "yup"),
        default="blender",
        help="Coordinate space of input joints. FlowMimic sample outputs use blender.",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--fit-devices",
        type=str,
        default=None,
        help=(
            "Comma-separated fitting devices, e.g. cuda:0,cuda:1. "
            "Defaults to --device; --device cuda expands to all visible GPUs."
        ),
    )
    parser.add_argument(
        "--gender",
        choices=("female", "male", "neutral", "custom"),
        default="female",
        help="SMPL model file to load: SMPL_FEMALE/MALE/NEUTRAL/CUSTOM.pkl.",
    )
    parser.add_argument("--num-smplify-iters", type=int, default=100)
    parser.add_argument(
        "--optimizer",
        choices=("lbfgs", "adam"),
        default="lbfgs",
        help="SMPLify optimizer. Adam is much faster; LBFGS usually fits tighter.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for testing.")
    parser.add_argument("--num-threads", type=int, default=None, help="Torch CPU threads.")
    parser.add_argument(
        "--fit-workers",
        type=int,
        default=0,
        help=(
            "Parallel fitting workers. 0 means auto: one worker per fit device. "
            "Frames are split into contiguous chunks."
        ),
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=None,
        help="Torch CPU threads per fit worker.",
    )
    parser.add_argument(
        "--save-frame-files",
        action="store_true",
        help="Also write per-frame PLY/PKL files. Slower; mesh npy is always saved.",
    )
    parser.add_argument(
        "--show-inner-progress",
        action="store_true",
        help="Show MLD's per-frame optimizer bars. Noisy with parallel workers.",
    )
    parser.add_argument(
        "--worker-log",
        action="store_true",
        help="Print per-worker messages. Disabled by default for clean tqdm output.",
    )
    parser.add_argument("--fix-foot", action="store_true", help="Upweight foot joints.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.num_threads, 16)))
        except RuntimeError:
            pass

    mld_root = _resolve_mld_root(args.mld_root)
    mld_config, SMPLify3D = _prepare_mld_imports(
        mld_root, suppress_inner_progress=not args.show_inner_progress
    )
    device = _select_device(args.device)
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = _repo_root() / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"MLD root: {mld_root}")
    print(f"SMPL model dir: {mld_config.SMPL_MODEL_DIR}")
    print(f"Device: {device}")
    if device.type == "cpu":
        print(f"Torch CPU threads: {torch.get_num_threads()}")

    input_paths = _discover_inputs(args.input, args.dir)
    if not input_paths:
        raise FileNotFoundError("No input npy files found.")

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        joints = _load_joints(input_path, args.input_space, args.max_frames)
        _fit_one_sequence(
            input_path=input_path,
            out_root=out_root,
            joints=joints,
            args=args,
            mld_config=mld_config,
            SMPLify3D=SMPLify3D,
            device=device,
        )


if __name__ == "__main__":
    main()
