import bpy
import numpy as np
from mathutils import Vector

NPY_PATH = "/home/aris/Downloads/array.npy"
SCALE = 1.0
SPHERE_RADIUS = 0.015
BONE_RADIUS = 0.006
COLLECTION_NAME = "KeypointsRig"

SMPL24_CHAINS = [
    [0, 3, 6, 9, 12, 15],
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [12, 13, 16, 18, 20],
    [12, 14, 17, 19, 21],
]


# ------------------
# Utils
# ------------------
def get_or_create_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def clear_collection(col):
    for o in list(col.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def create_sphere(name, loc, r, col):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=r, location=loc)
    obj = bpy.context.object
    obj.name = name
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def create_cyl(name, p0, p1, r, col):
    v = p1 - p0
    L = v.length
    mid = (p0 + p1) / 2
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=L, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(v.normalized())
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


# ------------------
# Load data
# ------------------
data = np.load(NPY_PATH) * SCALE  # (T,J,3)
T, J, _ = data.shape

col = get_or_create_collection(COLLECTION_NAME)
clear_collection(col)

# ------------------
# Create objects at frame 0
# ------------------
kps0 = [Vector(p) for p in data[0]]

spheres = [create_sphere(f"kp_{i}", p, SPHERE_RADIUS, col) for i, p in enumerate(kps0)]
bones = []

for chain in SMPL24_CHAINS:
    for a, b in zip(chain[:-1], chain[1:]):
        if a < J and b < J:
            bones.append(
                create_cyl(f"bone_{a}_{b}", kps0[a], kps0[b], BONE_RADIUS, col)
            )

# ------------------
# Animate
# ------------------
scene = bpy.context.scene
scene.frame_start = 0
scene.frame_end = T - 1

for t in range(T):
    scene.frame_set(t)
    kps = [Vector(p) for p in data[t]]

    # update spheres
    for i, obj in enumerate(spheres):
        obj.location = kps[i]
        obj.keyframe_insert("location")

    # update bones
    bone_i = 0
    for chain in SMPL24_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            if a < J and b < J:
                obj = bones[bone_i]
                v = kps[b] - kps[a]
                mid = (kps[a] + kps[b]) / 2
                obj.location = mid
                obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(
                    v.normalized()
                )
                obj.keyframe_insert("location")
                obj.keyframe_insert("rotation_quaternion")
                bone_i += 1

print("Animation done")
