"""Render fitted SMPL mesh sequences with Blender.

This renderer is a small FlowMimic-focused replacement for the MLD/TEMOS
Blender render path:
https://github.com/ChenFengYe/motion-latent-diffusion

It intentionally avoids MLD's older Blender scene helpers, because Blender 5
removed or changed some render/tile/preference APIs used there.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError as exc:  # pragma: no cover - only valid inside Blender.
    raise ImportError("Run this script with Blender: blender --background --python ...") from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Mesh npy with shape (T,V,3), Y-up.")
    parser.add_argument(
        "--faces",
        default=None,
        help="SMPL faces npy file. Defaults to a sidecar next to *_mesh.npy.",
    )
    parser.add_argument("--out", default=None, help="Output mp4 path or frame directory.")
    parser.add_argument("--mode", choices=("video", "frames", "frame"), default="video")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--res", choices=("low", "med", "high"), default="med")
    parser.add_argument(
        "--device",
        default="0",
        help="Blender Cycles GPU index, or 'all' to use every visible GPU.",
    )
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--no-clear-frames", action="store_true")
    return parser.parse_args(argv)


def yup_to_blender(vertices: np.ndarray) -> np.ndarray:
    x = vertices[..., 0]
    y = vertices[..., 1]
    z = vertices[..., 2]
    return np.stack([x, -z, y], axis=-1)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _repo_root() / resolved
    return resolved.resolve()


def _infer_faces_path(mesh_path: Path) -> Path:
    name = mesh_path.name
    if name.endswith("_mesh.npy"):
        return mesh_path.with_name(name[: -len("_mesh.npy")] + "_faces.npy")
    return mesh_path.with_name(mesh_path.stem + "_faces.npy")


def _load_faces(path: str | None, mesh_path: Path) -> np.ndarray:
    if path is None:
        faces_path = _infer_faces_path(mesh_path)
        if not faces_path.exists():
            fallback = _resolve_path("motion-latent-diffusion/deps/smpl_models/smpl.faces")
            print(
                "Faces sidecar not found; falling back to MLD smpl.faces. "
                f"For FlowMimic fitted meshes, create {faces_path} with fit_smpl.py first."
            )
            faces_path = fallback
    else:
        faces_path = _resolve_path(path)
    faces = np.load(faces_path)
    print(f"Using SMPL faces: {faces_path}")
    return np.asarray(faces, dtype=np.int32)


def _setup_scene(res: str, device: str) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    resolution = {"low": 720, "med": 1080, "high": 1440}[res]
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64 if res == "low" else 128
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world.color = (1.0, 1.0, 1.0)

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        selected = []
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
            except Exception:
                continue

            gpu_devices = [d for d in prefs.devices if d.type == backend]
            if not gpu_devices:
                continue

            for cycle_device in prefs.devices:
                cycle_device.use = False
            if device == "all":
                selected = gpu_devices
            else:
                try:
                    selected = [gpu_devices[int(device)]]
                except (ValueError, IndexError):
                    selected = [gpu_devices[0]]
            for cycle_device in selected:
                cycle_device.use = True
            scene.cycles.device = "GPU"
            print(
                "Using Blender GPU devices: "
                + ", ".join(f"{d.name} ({d.type})" for d in selected)
            )
            print(f"Cycles compute backend: {prefs.compute_device_type}")
            print(f"Cycles scene device: {scene.cycles.device}")
            print(
                "Cycles devices: "
                + "; ".join(
                    f"{d.name} ({d.type}) use={bool(d.use)}" for d in prefs.devices
                )
            )
            break
        if not selected:
            raise RuntimeError("no Blender GPU device found")
    except Exception as exc:
        print(f"Could not enable Blender GPU rendering, using CPU: {exc}")
        scene.cycles.device = "CPU"

    light_data = bpy.data.lights.new("Key", type="AREA")
    light_data.energy = 700
    light_data.size = 5
    light = bpy.data.objects.new("Key", light_data)
    light.location = (0.0, -3.0, 5.0)
    bpy.context.collection.objects.link(light)

    fill_data = bpy.data.lights.new("Fill", type="POINT")
    fill_data.energy = 60
    fill = bpy.data.objects.new("Fill", fill_data)
    fill.location = (3.0, 4.0, 4.0)
    bpy.context.collection.objects.link(fill)


def _make_material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.62
    return material


def _add_floor(bounds_min: np.ndarray, bounds_max: np.ndarray):
    size = float(max(bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], 2.5) * 1.8)
    center = ((bounds_min[0] + bounds_max[0]) * 0.5, (bounds_min[1] + bounds_max[1]) * 0.5, 0.0)
    bpy.ops.mesh.primitive_plane_add(size=size, location=center)
    floor = bpy.context.object
    floor.name = "Ground"
    floor.data.materials.append(_make_material("GroundMat", (0.82, 0.84, 0.82, 1.0)))
    return floor


def _look_at(obj, target: np.ndarray) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_camera(bounds_min: np.ndarray, bounds_max: np.ndarray) -> None:
    center = (bounds_min + bounds_max) * 0.5
    extent = np.maximum(bounds_max - bounds_min, 1e-3)
    radius = float(max(extent[0], extent[1], extent[2], 1.0))
    distance = radius * 2.5
    height = max(float(extent[2]) * 0.55, 1.0)
    angle = math.radians(35.0)
    cam_loc = (
        float(center[0] + math.sin(angle) * distance),
        float(center[1] - math.cos(angle) * distance),
        float(center[2] + height),
    )
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 45
    camera = bpy.data.objects.new("Camera", camera_data)
    camera.location = cam_loc
    bpy.context.collection.objects.link(camera)
    _look_at(camera, center)
    bpy.context.scene.camera = camera


def _mesh_object(vertices: np.ndarray, faces: np.ndarray, material):
    mesh = bpy.data.meshes.new("SMPLMesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("SMPLMesh", mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def _update_mesh_vertices(obj, vertices: np.ndarray) -> None:
    flat_vertices = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1)
    obj.data.vertices.foreach_set("co", flat_vertices)
    obj.data.update()


def _output_paths(args: argparse.Namespace, mesh_path: Path) -> tuple[Path, Path | None]:
    if args.out:
        out = _resolve_path(args.out)
    else:
        suffix = ".mp4" if args.mode == "video" else "_frames"
        out = mesh_path.with_name(mesh_path.stem + suffix)

    if args.mode == "video":
        frame_dir = out.with_suffix("")
        frame_dir = frame_dir.with_name(frame_dir.name + "_frames")
        video_path = out
    elif args.mode == "frame":
        frame_dir = out.parent
        video_path = None
    else:
        frame_dir = out
        video_path = None
    return frame_dir, video_path


def _encode_video(frame_dir: Path, video_path: Path, fps: int, start_number: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"ffmpeg not found; frames kept at {frame_dir}")
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    print(f"Blender PID: {os.getpid()}")
    mesh_path = _resolve_path(args.mesh)
    mesh_yup = np.load(mesh_path)
    if mesh_yup.ndim != 3 or mesh_yup.shape[-1] != 3:
        raise ValueError(f"Expected mesh npy shape (T,V,3), got {mesh_yup.shape}")
    mesh_yup = mesh_yup[:: max(1, args.stride)]
    if args.max_frames is not None:
        mesh_yup = mesh_yup[: args.max_frames]

    faces = _load_faces(args.faces, mesh_path)
    mesh = yup_to_blender(mesh_yup).astype(np.float32)
    mesh[..., 2] -= float(mesh[..., 2].min())
    bounds_min = mesh.reshape(-1, 3).min(axis=0)
    bounds_max = mesh.reshape(-1, 3).max(axis=0)

    _setup_scene(args.res, args.device)
    _add_floor(bounds_min, bounds_max)
    _add_camera(bounds_min, bounds_max)
    material = _make_material("BodyMat", (0.34, 0.53, 0.86, 1.0))

    frame_dir, video_path = _output_paths(args, mesh_path)
    if frame_dir.exists() and not args.no_clear_frames:
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "frame":
        frame_ids = [len(mesh) // 2]
    else:
        frame_ids = list(range(len(mesh)))
    if args.frame_start is not None or args.frame_end is not None:
        start = 0 if args.frame_start is None else max(0, args.frame_start)
        end = len(mesh) if args.frame_end is None else min(len(mesh), args.frame_end)
        frame_ids = [frame_idx for frame_idx in frame_ids if start <= frame_idx < end]
    if not frame_ids:
        raise ValueError("No frames to render.")

    obj = _mesh_object(mesh[frame_ids[0]], faces, material)

    for out_idx, frame_idx in enumerate(frame_ids):
        _update_mesh_vertices(obj, mesh[frame_idx])
        bpy.context.scene.render.filepath = str(frame_dir / f"frame_{frame_idx:04d}.png")
        bpy.ops.render.render(write_still=True)

    if video_path is not None:
        _encode_video(frame_dir, video_path, args.fps, min(frame_ids))
        if not args.keep_frames and video_path.exists():
            shutil.rmtree(frame_dir)
        print(f"Saved video: {video_path}")
    else:
        print(f"Saved frames: {frame_dir}")


if __name__ == "__main__":
    main()
