"""Build an interactive Blender scene from an AIST method-comparison manifest.

Run with:
  blender --python flowmimic/tools/vis_smpl22_blender.py -- \
    --manifest output/aist_method_comparisons/<run>/comparison_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


CHAINS = [
    [0, 3, 6, 9, 12, 15],
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [12, 13, 16, 18, 20],
    [12, 14, 17, 19, 21],
]
PALETTE = {
    "reference": ((0.08, 0.36, 0.68, 1.0), (0.16, 0.55, 0.91, 1.0)),
    "flowmimic": ((0.04, 0.48, 0.36, 1.0), (0.18, 0.72, 0.52, 1.0)),
    "mld": ((0.66, 0.22, 0.12, 1.0), (0.92, 0.38, 0.20, 1.0)),
    "stickmotion": ((0.48, 0.20, 0.62, 1.0), (0.71, 0.42, 0.82, 1.0)),
}
ROOT_COLLECTION = "FlowMimicComparison"
SMPL22_BONES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
PRIMARY_CHILD = {
    0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11,
    9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21,
}
# The pelvis uses its hip triangle; pointing it at spine1 shears the waist.
ORIENTATION_EDGE = {
    **{
        joint: (joint, child)
        for joint, child in PRIMARY_CHILD.items()
        if joint != 0
    },
    15: (12, 15),
    20: (18, 20),
    21: (19, 21),
}


def _script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=os.environ.get("FLOWMIMIC_COMPARISON_MANIFEST"),
        required=os.environ.get("FLOWMIMIC_COMPARISON_MANIFEST") is None,
    )
    parser.add_argument("--save-blend", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rig-spacing", type=float, default=3.6)
    parser.add_argument("--sphere-radius", type=float, default=0.025)
    parser.add_argument("--bone-radius", type=float, default=0.012)
    parser.add_argument(
        "--visualization-mode", choices=("skeleton", "rigged"), default="skeleton"
    )
    parser.add_argument(
        "--rigged-model", default=None
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _remove_collection(name: str) -> None:
    collection = bpy.data.collections.get(name)
    if collection is None:
        return
    for child in list(collection.children):
        _remove_collection(child.name)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def _clear_scene_objects() -> None:
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _new_collection(name: str, parent=None):
    collection = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(collection)
    return collection


def _move_to_collection(obj, collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def _material(name: str, color, roughness=0.55):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
    return material


def _assign_material(obj, material) -> None:
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def _load_motion(path: Path, scale: float) -> np.ndarray:
    motion = np.load(path).astype(np.float32) * float(scale)
    if motion.ndim != 3 or motion.shape[1:] != (22, 3):
        raise ValueError(f"{path}: expected [T,22,3], got {motion.shape}")
    return motion


def _create_sphere(name, location, radius, collection, material):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    _move_to_collection(obj, collection)
    _assign_material(obj, material)
    return obj


def _create_bone(name, start, end, radius, collection, material):
    vector = end - start
    length = max(vector.length, 1e-6)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=1.0, location=(start + end) * 0.5)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(vector.normalized())
    obj.scale.z = length
    _move_to_collection(obj, collection)
    _assign_material(obj, material)
    return obj


def _build_rig(key, label, motion, offset, parent, sphere_radius, bone_radius):
    collection = _new_collection(f"Rig_{key}", parent)
    bone_color, joint_color = PALETTE.get(key, PALETTE["reference"])
    bone_mat = _material(f"MAT_{key}_bones", bone_color)
    joint_mat = _material(f"MAT_{key}_joints", joint_color, roughness=0.4)
    points0 = [Vector(point) + offset for point in motion[0]]
    joints = [
        _create_sphere(f"{key}_joint_{idx:02d}", point, sphere_radius, collection, joint_mat)
        for idx, point in enumerate(points0)
    ]
    pairs = [
        (a, b)
        for chain in CHAINS
        for a, b in zip(chain[:-1], chain[1:])
    ]
    bones = [
        _create_bone(
            f"{key}_bone_{a:02d}_{b:02d}",
            points0[a],
            points0[b],
            bone_radius,
            collection,
            bone_mat,
        )
        for a, b in pairs
    ]

    z_axis = Vector((0, 0, 1))
    for frame, pose in enumerate(motion, start=1):
        points = [Vector(point) + offset for point in pose]
        for joint, point in zip(joints, points):
            joint.location = point
            joint.keyframe_insert("location", frame=frame)
        for bone, (a, b) in zip(bones, pairs):
            vector = points[b] - points[a]
            length = max(vector.length, 1e-6)
            bone.location = (points[a] + points[b]) * 0.5
            bone.rotation_quaternion = z_axis.rotation_difference(vector.normalized())
            bone.scale.z = length
            bone.keyframe_insert("location", frame=frame)
            bone.keyframe_insert("rotation_quaternion", frame=frame)
            bone.keyframe_insert("scale", frame=frame)

    _create_text(
        f"Label_{key}",
        label,
        (offset.x, -0.55, 2.35),
        parent,
        size=0.34,
        align="CENTER",
    )


def _pelvis_basis(points):
    x_axis = (points[1] - points[2]).normalized()
    hip_center = (points[1] + points[2]) * 0.5
    up_hint = points[0] - hip_center
    if x_axis.length_squared < 1e-10 or up_hint.length_squared < 1e-10:
        return None
    z_axis = x_axis.cross(up_hint).normalized()
    if z_axis.length_squared < 1e-10:
        return None
    y_axis = z_axis.cross(x_axis).normalized()
    return Matrix((x_axis, y_axis, z_axis)).transposed().to_quaternion()


def _build_rigged_body(key, label, motion, offset, parent, model_path):
    collection = _new_collection(f"Rig_{key}", parent)
    existing = {obj.as_pointer() for obj in bpy.data.objects}
    bpy.ops.import_scene.gltf(filepath=str(model_path))
    imported = [obj for obj in bpy.data.objects if obj.as_pointer() not in existing]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError(f"{model_path}: expected one armature, found {len(armatures)}")
    armature = armatures[0]
    keep = {armature, *armature.children_recursive}
    for obj in imported:
        if obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in keep:
        _move_to_collection(obj, collection)

    armature.name = f"{key}_SMPL22_Armature"
    armature.data.name = f"{key}_SMPL22_ArmatureData"
    armature.show_in_front = True
    body_material = _material(f"MAT_{key}_body", PALETTE.get(key, PALETTE["reference"])[1], roughness=0.48)
    for obj in keep:
        if obj.type == "MESH":
            obj.name = f"{key}_SMPL22_Mesh"
            obj.data.materials.clear()
            obj.data.materials.append(body_material)

    missing = [name for name in SMPL22_BONES if name not in armature.data.bones]
    if missing:
        raise ValueError(f"{model_path}: missing SMPL22 bones: {', '.join(missing)}")
    rest_points = [armature.data.bones[name].head_local.copy() for name in SMPL22_BONES]
    rest_basis = _pelvis_basis(rest_points)
    if rest_basis is None:
        raise ValueError(f"{model_path}: degenerate rest-pose body basis")
    rest_quaternions = [
        armature.data.bones[name].matrix_local.to_quaternion()
        for name in SMPL22_BONES
    ]
    rest_directions = []
    for index in range(len(SMPL22_BONES)):
        edge = ORIENTATION_EDGE.get(index)
        rest_directions.append(
            None
            if edge is None
            else (rest_points[edge[1]] - rest_points[edge[0]]).normalized()
        )
    pose_bones = [armature.pose.bones[name] for name in SMPL22_BONES]
    for pose_bone in pose_bones:
        pose_bone.rotation_mode = "QUATERNION"

    for frame_index, pose in enumerate(motion, start=1):
        bpy.context.scene.frame_set(frame_index)
        points = [Vector(point) + offset for point in pose]
        target_basis = _pelvis_basis(points)
        if target_basis is None:
            continue
        basis_delta = target_basis @ rest_basis.inverted()
        for index, pose_bone in enumerate(pose_bones):
            world_quaternion = basis_delta @ rest_quaternions[index]
            edge = ORIENTATION_EDGE.get(index)
            if edge is not None:
                target_direction = points[edge[1]] - points[edge[0]]
                if target_direction.length_squared > 1e-10:
                    source_direction = rest_directions[index].copy()
                    source_direction.rotate(basis_delta)
                    align = source_direction.normalized().rotation_difference(
                        target_direction.normalized()
                    )
                    world_quaternion = align @ world_quaternion
            pose_bone.matrix = (
                Matrix.Translation(points[index])
                @ world_quaternion.to_matrix().to_4x4()
            )
            bpy.context.view_layer.update()
        for pose_bone in pose_bones:
            pose_bone.keyframe_insert("location", frame=frame_index)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame_index)
            pose_bone.keyframe_insert("scale", frame=frame_index)

    _create_text(
        f"Label_{key}",
        label,
        (offset.x, -0.55, 2.35),
        parent,
        size=0.34,
        align="CENTER",
    )


def _create_text(name, body, location, collection, size=0.24, align="LEFT", width=0.0):
    curve = bpy.data.curves.new(name=f"{name}_curve", type="FONT")
    curve.body = body
    curve.align_x = align
    curve.align_y = "CENTER"
    curve.size = size
    curve.extrude = 0.004
    if width > 0:
        curve.text_boxes[0].width = width
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (math.radians(90), 0.0, 0.0)
    _assign_material(obj, _material("MAT_text", (0.82, 0.84, 0.82, 1.0), roughness=0.7))
    return obj


def _curve_polyline(name, points, collection, material, bevel=0.012):
    curve = bpy.data.curves.new(name=f"{name}_curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = bevel
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for dst, point in zip(spline.points, points):
        dst.co = (*point, 1.0)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    _assign_material(obj, material)
    return obj


def _build_stickman_panels(manifest, manifest_path, parent, rig_span):
    stick = manifest.get("methods", {}).get("stickmotion", {})
    tracks_value = stick.get("stickman_tracks")
    if not tracks_value:
        return
    tracks = np.load(_resolve(manifest_path, tracks_value)).astype(np.float32)
    indices = stick.get("stickman_frame_indices", [])
    source_indices = stick.get("stickman_source_frame_indices", [])
    if tracks.ndim != 4 or tracks.shape[1:] != (6, 64, 2):
        raise ValueError(f"Expected stickman tracks [K,6,64,2], got {tracks.shape}")

    collection = _new_collection("StickMotionSketches", parent)
    stroke_mat = _material("MAT_stick_strokes", (0.02, 0.02, 0.02, 1.0), roughness=0.8)
    panel_mat = _material("MAT_stick_panels", (0.92, 0.93, 0.90, 1.0), roughness=0.9)
    panel_spacing = min(2.5, rig_span / max(len(tracks), 1))
    centers = (np.arange(len(tracks)) - (len(tracks) - 1) * 0.5) * panel_spacing
    for sketch_idx, (track_set, center_x) in enumerate(zip(tracks, centers)):
        bpy.ops.mesh.primitive_plane_add(
            size=1.8,
            location=(float(center_x), 1.15, 3.65),
            rotation=(math.radians(90), 0.0, 0.0),
        )
        panel = bpy.context.object
        panel.name = f"StickPanel_{sketch_idx:02d}"
        _move_to_collection(panel, collection)
        _assign_material(panel, panel_mat)
        for stroke_idx, stroke in enumerate(track_set):
            centered = stroke - track_set.reshape(-1, 2).mean(axis=0, keepdims=True)
            points = [
                (float(center_x + p[0] * 0.72), 1.10, float(3.65 + p[1] * 0.72))
                for p in centered
            ]
            _curve_polyline(
                f"Stick_{sketch_idx:02d}_{stroke_idx:02d}",
                points,
                collection,
                stroke_mat,
            )
        frame_index = indices[sketch_idx] if sketch_idx < len(indices) else "?"
        source_index = (
            source_indices[sketch_idx] if sketch_idx < len(source_indices) else None
        )
        frame_label = f"Stick frame {frame_index}"
        if source_index is not None:
            frame_label += f" (source {source_index})"
        _create_text(
            f"StickLabel_{sketch_idx:02d}",
            frame_label,
            (float(center_x), 1.08, 2.66),
            collection,
            size=0.20,
            align="CENTER",
        )


def _point_camera(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def _set_linear_interpolation() -> None:
    for action in bpy.data.actions:
        fcurves = getattr(action, "fcurves", None)
        if fcurves is not None:
            action_fcurves = fcurves
        else:
            action_fcurves = (
                fcurve
                for layer in getattr(action, "layers", ())
                for strip in getattr(layer, "strips", ())
                for channelbag in getattr(strip, "channelbags", ())
                for fcurve in getattr(channelbag, "fcurves", ())
            )
        for fcurve in action_fcurves:
            for point in fcurve.keyframe_points:
                point.interpolation = "LINEAR"


def _build_scene(manifest_path: Path, args: argparse.Namespace) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motions_spec = manifest.get("motions", [])
    if len(motions_spec) != 4:
        raise ValueError(f"Expected four motions in manifest, got {len(motions_spec)}")
    motions = [
        (item, _load_motion(_resolve(manifest_path, item["path"]), args.scale))
        for item in motions_spec
    ]
    frame_count = min(len(motion) for _, motion in motions)
    motions = [(item, motion[:frame_count]) for item, motion in motions]
    rigged_model = Path(
        args.rigged_model or "web_view/assets/smpl22_rigged_calibrated.glb"
    )
    if not rigged_model.is_absolute():
        rigged_model = (Path.cwd() / rigged_model).resolve()
    if args.rigged_model is None and not rigged_model.is_file():
        rigged_model = (Path.cwd() / "web_view/assets/smpl22_rigged.glb").resolve()
    if args.visualization_mode == "rigged" and not rigged_model.is_file():
        raise FileNotFoundError(f"Rigged SMPL22 model not found: {rigged_model}")

    _remove_collection(ROOT_COLLECTION)
    _clear_scene_objects()
    root = _new_collection(ROOT_COLLECTION)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.render.fps = 30
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    count = len(motions)
    offsets_x = (np.arange(count) - (count - 1) * 0.5) * args.rig_spacing
    for (item, motion), x in zip(motions, offsets_x):
        offset = Vector((float(x), 0.0, 0.0))
        if args.visualization_mode == "rigged":
            _build_rigged_body(
                item["key"], item["label"], motion, offset, root, rigged_model
            )
        else:
            _build_rig(
                item["key"],
                item["label"],
                motion,
                offset,
                root,
                args.sphere_radius,
                args.bone_radius,
            )

    span = args.rig_spacing * max(count - 1, 1) + 3.0
    caption = manifest.get("caption", {}).get("text", "")
    header = (
        f"{manifest.get('sample_id', '')} | {manifest.get('split', '')} | "
        f"start={manifest.get('clip_start', 0)} | "
        f"FlowMimic K={manifest.get('condition', {}).get('requested_frames', '?')}\n"
        f"MLD / StickMotion text: {textwrap.fill(caption, width=105)}"
    )
    _create_text("ComparisonCaption", header, (0.0, 1.05, 5.15), root, size=0.21, align="CENTER")
    _build_stickman_panels(manifest, manifest_path, root, span)

    floor_mat = _material("MAT_floor", (0.20, 0.22, 0.21, 1.0), roughness=0.95)
    bpy.ops.mesh.primitive_plane_add(size=max(18.0, span + 3.0), location=(0.0, 0.0, -0.02))
    floor = bpy.context.object
    floor.name = "ComparisonFloor"
    _move_to_collection(floor, root)
    _assign_material(floor, floor_mat)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 8.0))
    key_light = bpy.context.object
    key_light.name = "ComparisonKeyLight"
    key_light.data.energy = 1800
    key_light.data.shape = "RECTANGLE"
    key_light.data.size = 10.0
    _move_to_collection(key_light, root)
    _point_camera(key_light, (0.0, 0.0, 1.5))
    key_light.hide_set(True)
    key_light.hide_render = False

    bpy.ops.object.light_add(type="AREA", location=(0.0, 5.0, 5.0))
    fill_light = bpy.context.object
    fill_light.name = "ComparisonFillLight"
    fill_light.data.energy = 900
    fill_light.data.size = 8.0
    _move_to_collection(fill_light, root)
    _point_camera(fill_light, (0.0, 0.0, 2.0))
    fill_light.hide_set(True)
    fill_light.hide_render = False

    horizontal_extent = max(
        abs(float(motion[:, :, 0].min()) + float(x))
        for (_, motion), x in zip(motions, offsets_x)
    )
    horizontal_extent = max(
        horizontal_extent,
        max(
            abs(float(motion[:, :, 0].max()) + float(x))
            for (_, motion), x in zip(motions, offsets_x)
        ),
    )
    camera_distance = max(18.0, (horizontal_extent + 0.8) * 2.8)
    bpy.ops.object.camera_add(location=(0.0, -camera_distance, 5.2))
    camera = bpy.context.object
    camera.name = "ComparisonCamera"
    camera.data.lens = 48
    _point_camera(camera, (0.0, 0.4, 2.45))
    _move_to_collection(camera, root)
    scene.camera = camera

    for marker in list(scene.timeline_markers):
        scene.timeline_markers.remove(marker)
    flow_indices = manifest.get("condition", {}).get("frame_indices", [])
    for index in flow_indices:
        scene.timeline_markers.new(f"FlowCond_{int(index):03d}", frame=int(index) + 1)
    stick_indices = manifest.get("methods", {}).get("stickmotion", {}).get(
        "stickman_frame_indices", []
    )
    for index in stick_indices:
        scene.timeline_markers.new(f"StickCond_{int(index):03d}", frame=int(index) + 1)

    _set_linear_interpolation()

    scene["comparison_manifest"] = str(manifest_path)
    scene["sample_id"] = manifest.get("sample_id", "")
    scene["text_description"] = caption
    scene["visualization_mode"] = args.visualization_mode
    scene.frame_set(1)
    if args.save_blend:
        save_path = Path(args.save_blend)
        if not save_path.is_absolute():
            save_path = (manifest_path.parent / save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
        print(f"Saved comparison scene: {save_path}")
    print(
        f"Loaded {manifest.get('sample_id', '')}: {frame_count} frames, "
        f"4 motions, {len(stick_indices)} StickMotion sketches"
    )


def main() -> None:
    args = _script_args()
    _build_scene(Path(args.manifest).resolve(), args)


if __name__ == "__main__":
    main()
