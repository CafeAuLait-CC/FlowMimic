import os

import numpy as np
import torch

from flowmimic.src.model.flow.cond_encoder_2d import normalize_keypoints


def _resample_to_fps(values, src_fps, dst_fps, mode="pchip"):
    if src_fps == dst_fps:
        return values
    if values.shape[0] < 2:
        return values
    if not np.isfinite(values).all():
        return values
    if src_fps == 60 and dst_fps == 30:
        return values[::2]
    if src_fps < dst_fps:
        try:
            from scipy.interpolate import PchipInterpolator
        except ImportError:
            raise ImportError(
                "scipy is required for PCHIP upsampling; please install scipy"
            )
        t_src = np.arange(values.shape[0], dtype=np.float64) / float(src_fps)
        t_dst = np.arange(
            int(np.round(t_src[-1] * dst_fps)) + 1, dtype=np.float64
        ) / float(dst_fps)
        if mode == "nearest":
            idx = np.clip(np.round(t_dst * src_fps).astype(int), 0, values.shape[0] - 1)
            return values[idx]
        if mode == "geom":
            idx0 = np.floor(t_dst * src_fps).astype(int)
            idx0 = np.clip(idx0, 0, values.shape[0] - 1)
            idx1 = np.clip(idx0 + 1, 0, values.shape[0] - 1)
            alpha = (t_dst * src_fps) - idx0
            alpha = np.clip(alpha, 0.0, 1.0)
            c0 = values[idx0]
            c1 = values[idx1]
            eps = 1e-6
            log_c0 = np.log(np.clip(c0, eps, None))
            log_c1 = np.log(np.clip(c1, eps, None))
            log_c = (1.0 - alpha)[:, None] * log_c0 + alpha[:, None] * log_c1
            out = np.exp(log_c)
            zero_mask = (c0 <= 0.0) | (c1 <= 0.0)
            out[zero_mask] = 0.0
            return np.clip(out, 0.0, 1.0)
        flat = values.reshape(values.shape[0], -1)
        interp = PchipInterpolator(t_src, flat, axis=0)
        out = interp(t_dst)
        return out.reshape(len(t_dst), *values.shape[1:])
    step = int(round(src_fps / float(dst_fps)))
    return values[::step]


def load_openpose_npy(npy_path, src_fps=None, target_fps=None):
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[1:] != (25, 3):
        raise ValueError(f"Expected [T,25,3] array, got {arr.shape} in {npy_path}")
    coords = arr[..., :2].astype(np.float32)
    conf = arr[..., 2].astype(np.float32)
    coords[~np.isfinite(coords)] = 0.0
    conf[~np.isfinite(conf)] = 0.0
    coords[..., 1] = -coords[..., 1]
    if src_fps is not None and target_fps is not None:
        coords = _resample_to_fps(
            coords, src_fps=src_fps, dst_fps=target_fps, mode="pchip"
        )
        conf = _resample_to_fps(conf, src_fps=src_fps, dst_fps=target_fps, mode="geom")
        conf = np.clip(conf, 0.0, 1.0)
    vis_mask = conf > 0.0

    pelvis = coords[0, 8]
    if not vis_mask[0, 8]:
        visible = vis_mask[0]
        if visible.any():
            pelvis = coords[0, visible].mean(axis=0)
        else:
            pelvis = np.zeros(2, dtype=np.float32)
    coords = coords - pelvis[None, None, :]
    return coords, vis_mask.astype(np.float32)


def _save_openpose_cache(cache_path, coords, vis_mask):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        coords=coords.astype(np.float32),
        vis=vis_mask.astype(np.float32),
    )


def _load_openpose_cache(cache_path):
    data = np.load(cache_path)
    return data["coords"], data["vis"]


def _sanitize_openpose(coords, vis):
    if coords is None or vis is None:
        return coords, vis
    coords = coords.astype(np.float32, copy=False)
    vis = vis.astype(np.float32, copy=False)
    coords[~np.isfinite(coords)] = 0.0
    vis[~np.isfinite(vis)] = 0.0
    vis = np.clip(vis, 0.0, 1.0)
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    return coords, vis


def cache_openpose_npy(npy_path, cache_path, src_fps=None, target_fps=None):
    coords, vis = load_openpose_npy(npy_path, src_fps=src_fps, target_fps=target_fps)
    _save_openpose_cache(cache_path, coords, vis)
    return coords, vis


def _aist_cache_path(cache_root, pkl_path, camera=None):
    name = os.path.splitext(os.path.basename(pkl_path))[0]
    if camera is None:
        return os.path.join(cache_root, "aist", f"{name}.npz")
    return os.path.join(cache_root, "aist", f"{name}_c{camera}.npz")


def _mvh_cache_path(cache_root, mv_root, seq_dir, cam):
    rel = os.path.relpath(seq_dir, mv_root)
    return os.path.join(cache_root, "mvh", rel, "openpose", f"{cam}.npz")


def load_aist_openpose(
    pkl_path,
    openpose_dir,
    src_fps=None,
    target_fps=None,
    cache_root=None,
    write_cache=False,
    camera=None,
):
    cache_path = None
    if cache_root:
        cache_path = _aist_cache_path(cache_root, pkl_path, camera=camera)
        if os.path.exists(cache_path):
            coords, vis = _load_openpose_cache(cache_path)
            return _sanitize_openpose(coords, vis)
    name = os.path.splitext(os.path.basename(pkl_path))[0]
    cam_list = [camera] if camera is not None else ["01", "02", "08", "09"]
    for cam in cam_list:
        name_cam = name.replace("_cAll_", f"_c{cam}_")
        path = os.path.join(openpose_dir, f"{name_cam}.npy")
        if not os.path.exists(path):
            continue
        coords, vis = load_openpose_npy(path, src_fps=src_fps, target_fps=target_fps)
        if cache_root and write_cache:
            _save_openpose_cache(_aist_cache_path(cache_root, pkl_path, camera=cam), coords, vis)
        return _sanitize_openpose(coords, vis)
    return None, None


def load_mvh_openpose(
    seq_dir,
    mv_root,
    openpose_root,
    cameras,
    src_fps=None,
    target_fps=None,
    cache_root=None,
    write_cache=False,
    camera=None,
):
    rel = os.path.relpath(seq_dir, mv_root)
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return None, None
    part, motion = parts[0], parts[1]
    cam_list = [camera] if camera is not None else cameras
    for cam in cam_list:
        if cache_root:
            cache_path = _mvh_cache_path(cache_root, mv_root, seq_dir, cam)
            if os.path.exists(cache_path):
                coords, vis = _load_openpose_cache(cache_path)
                return _sanitize_openpose(coords, vis)
        path = os.path.join(openpose_root, part, motion, f"{cam}_2d_body25.npy")
        if os.path.exists(path):
            coords, vis = load_openpose_npy(path, src_fps=src_fps, target_fps=target_fps)
            if cache_root and write_cache:
                _save_openpose_cache(cache_path, coords, vis)
            return _sanitize_openpose(coords, vis)
    return None, None


def compute_openpose_stats(
    aist_paths,
    mvh_dirs,
    aist_openpose_dir,
    mvh_openpose_root,
    mv_root,
    cameras,
    target_fps,
    aist_fps,
    mvh_fps,
    out_path,
    eps=1e-6,
    progress=None,
    cache_root=None,
    aist_cameras=None,
    mvh_cameras=None,
):
    sum_xy = np.zeros((25, 2), dtype=np.float64)
    sum_sq = np.zeros((25, 2), dtype=np.float64)
    count = np.zeros((25, 2), dtype=np.float64)

    def _accumulate(k2d, vis):
        if k2d is None:
            return
        k_t = torch.from_numpy(k2d).unsqueeze(0)
        vis_t = torch.from_numpy(vis).unsqueeze(0)
        k_norm, vis_out = normalize_keypoints(k_t, vis_mask=vis_t, center_mode="none")
        k_norm = k_norm.squeeze(0).numpy()
        vis_out = vis_out.squeeze(0).numpy()
        mask = vis_out > 0.5
        for j in range(25):
            if not mask[:, j].any():
                continue
            vals = k_norm[mask[:, j], j]
            sum_xy[j] += vals.sum(axis=0)
            sum_sq[j] += (vals**2).sum(axis=0)
            count[j] += vals.shape[0]

    aist_iter = aist_paths
    mvh_iter = mvh_dirs
    if progress is not None:
        aist_iter = progress(aist_paths, desc="OpenPose stats (AIST)", leave=False)
        mvh_iter = progress(mvh_dirs, desc="OpenPose stats (MVH)", leave=True)

    cam_list = aist_cameras or ["01", "02", "08", "09"]
    mvh_cam_list = mvh_cameras or cameras
    for path in aist_iter:
        for cam in cam_list:
            k2d, vis = load_aist_openpose(
                path,
                aist_openpose_dir,
                src_fps=aist_fps,
                target_fps=target_fps,
                cache_root=cache_root,
                camera=cam,
            )
            _accumulate(k2d, vis)

    for seq_dir in mvh_iter:
        for cam in mvh_cam_list:
            k2d, vis = load_mvh_openpose(
                seq_dir,
                mv_root,
                mvh_openpose_root,
                cameras,
                src_fps=mvh_fps,
                target_fps=target_fps,
                cache_root=cache_root,
                camera=cam,
            )
            _accumulate(k2d, vis)

    mean = np.zeros((25, 2), dtype=np.float32)
    std = np.ones((25, 2), dtype=np.float32)
    for j in range(25):
        for d in range(2):
            if count[j, d] > 0:
                mean[j, d] = sum_xy[j, d] / count[j, d]
                var = sum_sq[j, d] / count[j, d] - mean[j, d] ** 2
                std[j, d] = np.sqrt(max(var, eps))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, mean=mean, std=std)
    return mean, std
