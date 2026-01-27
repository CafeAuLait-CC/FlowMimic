import bpy
import numpy as np
from mathutils import Vector

# =========================
# Paths
# =========================
ROOT_PATH = "/home/aris/Downloads/"
BEFORE_PATH = ROOT_PATH + "before_gWA_sFM_cAll_d26_mWA3_ch11.npy"
AFTER_PATH  = ROOT_PATH + "after_gWA_sFM_cAll_d26_mWA3_ch11.npy"

# =========================
# Visual params
# =========================
SCALE = 1.0
SPHERE_RADIUS = 0.015
BONE_RADIUS   = 0.006

# Add some offsets to avoid occlusion. (0, 0, 0) to remove offset
OFFSET_BEFORE = Vector((0.0, 0.0, 0.0))
OFFSET_AFTER  = Vector((0.0, 0.0, 0.0))  # E.g. 0.25m offsets in x-axis

# Blender Collection Name
COL_BEFORE = "Rig_before"
COL_AFTER  = "Rig_after"

# =========================
# Skeleton chains
# =========================
CHAINS = [
    [0, 3, 6, 9, 12, 15],     # spine
    [0, 1, 4, 7, 10],         # left leg
    [0, 2, 5, 8, 11],         # right leg
    [12, 13, 16, 18, 20],     # left arm
    [12, 14, 17, 19, 21],     # right arm
]

# =========================
# Utils
# =========================
def get_or_create_collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def clear_collection(col):
    for o in list(col.objects):
        bpy.data.objects.remove(o, do_unlink=True)

def make_material(name: str, rgba):
    """
    Use Base Color from Principled BSDF to distinguish colors
    rgba: (r,g,b,a)
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        # Higher Emission for brighter color (optional)
        # bsdf.inputs["Emission"].default_value = rgba[:3] + (1.0,)
        # bsdf.inputs["Emission Strength"].default_value = 0.2
    return mat

def link_to_collection(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)

def create_sphere(name, loc, r, col, mat):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=r, location=loc)
    obj = bpy.context.object
    obj.name = name
    link_to_collection(obj, col)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj

def create_cylinder(name, p0, p1, r, col, mat):
    v = p1 - p0
    L = v.length
    if L < 1e-8:
        return None

    mid = (p0 + p1) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=L, location=mid)
    obj = bpy.context.object
    obj.name = name

    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(v.normalized())

    link_to_collection(obj, col)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj

def load_npy(path: str):
    arr = np.load(path).astype(np.float32) * float(SCALE)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path}: expected shape (T,J,3), got {arr.shape}")
    return arr

def build_rig_objects(kps0, chains, col, sphere_mat, bone_mat):
    """
    Only create objects once (spheres + bones), 
    return spheres and bones + bone_pairs (for later frame refreshing)
    """
    J = len(kps0)
    spheres = [create_sphere(f"kp_{i:03d}", kps0[i], SPHERE_RADIUS, col, sphere_mat) for i in range(J)]

    bones = []
    bone_pairs = []  # [(a,b), ...] corresponding to bones
    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            if 0 <= a < J and 0 <= b < J:
                obj = create_cylinder(f"bone_{a}_{b}", kps0[a], kps0[b], BONE_RADIUS, col, bone_mat)
                if obj is not None:
                    bones.append(obj)
                    bone_pairs.append((a, b))
    return spheres, bones, bone_pairs

def animate_rig(scene, data, offset, spheres, bones, bone_pairs):
    """
    data: (T,J,3)
    offset: Vector
    """
    T, J, _ = data.shape
    z_axis = Vector((0, 0, 1))

    for t in range(T):
        scene.frame_set(t)
        kps = [Vector(p) + offset for p in data[t]]

        # spheres: location
        for i in range(J):
            spheres[i].location = kps[i]
            spheres[i].keyframe_insert("location")

        # bones: location + rotation_quaternion
        for obj, (a, b) in zip(bones, bone_pairs):
            v = kps[b] - kps[a]
            L = v.length
            if L < 1e-8:
                continue
            obj.location = (kps[a] + kps[b]) * 0.5
            obj.rotation_quaternion = z_axis.rotation_difference(v.normalized())
            obj.keyframe_insert("location")
            obj.keyframe_insert("rotation_quaternion")


# =========================
# Main
# =========================
before = load_npy(BEFORE_PATH)
after  = load_npy(AFTER_PATH)

if before.shape != after.shape:
    raise ValueError(f"Shape mismatch: before {before.shape} vs after {after.shape}")

T, J, _ = before.shape

scene = bpy.context.scene
scene.frame_start = 0
scene.frame_end = T - 1

# collections
col_b = get_or_create_collection(COL_BEFORE)
col_a = get_or_create_collection(COL_AFTER)
clear_collection(col_b)
clear_collection(col_a)

# materials
# before: blue
mat_b_sphere = make_material("MAT_before_sphere", (0.20, 0.55, 1.00, 1.0))
mat_b_bone   = make_material("MAT_before_bone",   (0.10, 0.35, 0.90, 1.0))
# after: red/orange
mat_a_sphere = make_material("MAT_after_sphere",  (1.00, 0.35, 0.20, 1.0))
mat_a_bone   = make_material("MAT_after_bone",    (0.90, 0.20, 0.10, 1.0))

# build at frame 0
kps0_before = [Vector(p) + OFFSET_BEFORE for p in before[0]]
kps0_after  = [Vector(p) + OFFSET_AFTER  for p in after[0]]

sph_b, bones_b, pairs_b = build_rig_objects(kps0_before, CHAINS, col_b, mat_b_sphere, mat_b_bone)
sph_a, bones_a, pairs_a = build_rig_objects(kps0_after,  CHAINS, col_a, mat_a_sphere, mat_a_bone)

# animate
animate_rig(scene, before, OFFSET_BEFORE, sph_b, bones_b, pairs_b)
animate_rig(scene, after,  OFFSET_AFTER,  sph_a, bones_a, pairs_a)

print(f"Done. Two motions animated: T={T}, J={J}. Collections: {COL_BEFORE}, {COL_AFTER}")