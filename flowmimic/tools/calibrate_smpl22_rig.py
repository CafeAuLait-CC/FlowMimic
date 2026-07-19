"""Create a minimally symmetrized SMPL22 GLB for FlowMimic visualization.

Run with Blender:
  blender --background --python flowmimic/tools/calibrate_smpl22_rig.py -- \
    --input web_view/assets/smpl22_rigged.glb \
    --output web_view/assets/smpl22_rigged_calibrated.glb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


CENTER_CHAIN = ("pelvis", "spine1", "spine2", "spine3", "neck", "head")
MIRROR_PAIRS = (
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_foot", "right_foot"),
    ("left_collar", "right_collar"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--knee-center-blend",
        type=float,
        default=0.70,
        help="Fraction of the bone-to-mesh knee-center offset to correct.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if not 0.0 <= args.knee_center_blend <= 1.0:
        parser.error("--knee-center-blend must be within [0, 1]")
    return args


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _knee_mesh_center_x(mesh, knee_head) -> float:
    vertices = np.asarray(
        [tuple(mesh.matrix_world @ vertex.co) for vertex in mesh.data.vertices],
        dtype=np.float64,
    )
    same_side = vertices[:, 0] > 0 if knee_head.x > 0 else vertices[:, 0] < 0
    section = vertices[
        same_side
        & (np.abs(vertices[:, 2] - knee_head.z) < 0.025)
        & (np.abs(vertices[:, 1] - knee_head.y) < 0.20)
    ]
    if len(section) < 8:
        raise ValueError(f"Too few mesh vertices near knee pivot: {len(section)}")
    return float((section[:, 0].min() + section[:, 0].max()) * 0.5)


def _move_edit_bone(edit_bone, target_head) -> None:
    delta = target_head - edit_bone.head
    edit_bone.head += delta
    edit_bone.tail += delta


def _calibrate_armature(armature, mesh, knee_center_blend: float) -> dict:
    original_heads = {
        bone.name: bone.head_local.copy()
        for bone in armature.data.bones
    }
    required = set(CENTER_CHAIN)
    required.update(name for pair in MIRROR_PAIRS for name in pair)
    missing = sorted(required.difference(original_heads))
    if missing:
        raise ValueError(f"Missing SMPL22 bones: {', '.join(missing)}")

    left_knee_mesh_x = _knee_mesh_center_x(mesh, original_heads["left_knee"])
    right_knee_mesh_x = _knee_mesh_center_x(mesh, original_heads["right_knee"])
    knee_mesh_abs_x = (abs(left_knee_mesh_x) + abs(right_knee_mesh_x)) * 0.5

    targets = {}
    for name in CENTER_CHAIN:
        target = original_heads[name].copy()
        target.x = 0.0
        targets[name] = target

    for left_name, right_name in MIRROR_PAIRS:
        left = original_heads[left_name]
        right = original_heads[right_name]
        half_width = (abs(left.x) + abs(right.x)) * 0.5
        if left_name == "left_knee":
            half_width += (knee_mesh_abs_x - half_width) * knee_center_blend
        shared_y = (left.y + right.y) * 0.5
        shared_z = (left.z + right.z) * 0.5
        left_target = left.copy()
        right_target = right.copy()
        left_target[:] = (half_width, shared_y, shared_z)
        right_target[:] = (-half_width, shared_y, shared_z)
        targets[left_name] = left_target
        targets[right_name] = right_target

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for name, target in targets.items():
        _move_edit_bone(armature.data.edit_bones[name], target)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()

    armature["flowmimic_rig_calibrated"] = True
    armature["flowmimic_knee_center_blend"] = float(knee_center_blend)
    return {
        "knee_bone_abs_x_before": (
            abs(original_heads["left_knee"].x)
            + abs(original_heads["right_knee"].x)
        )
        * 0.5,
        "knee_mesh_abs_x": knee_mesh_abs_x,
        "knee_bone_abs_x_after": abs(targets["left_knee"].x),
    }


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("Input and output GLB paths must differ")

    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(input_path))
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError(f"Expected one armature, found {len(armatures)}")
    armature = armatures[0]
    meshes = [obj for obj in armature.children_recursive if obj.type == "MESH"]
    if len(meshes) != 1:
        raise ValueError(f"Expected one skinned mesh, found {len(meshes)}")
    mesh = meshes[0]

    measurements = _calibrate_armature(
        armature, mesh, float(args.knee_center_blend)
    )
    for pose_bone in armature.pose.bones:
        pose_bone.custom_shape = None
    keep = {armature, *armature.children_recursive}
    for obj in list(bpy.context.scene.objects):
        if obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in keep:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_leaf_bone=False,
        export_extras=True,
        export_rest_position_armature=True,
    )
    print(f"Saved calibrated SMPL22 rig: {output_path}")
    for key, value in measurements.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
