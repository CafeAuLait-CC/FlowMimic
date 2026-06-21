#!/usr/bin/env python
"""Create a SMPL_CUSTOM.pkl model definition from a saved SMPL output pickle.

Use this when the "custom" file contains fitted output fields such as
``betas``/``vertices``/``joints`` rather than a real SMPL model definition.
The converter copies an existing SMPL model file and bakes the saved betas into
``v_template`` so beta=0 becomes the custom body shape.
"""

from __future__ import annotations

import argparse
import copy
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve()


def _load_pickle(path: Path):
    with path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return pickle.load(f, encoding="latin1")


def _save_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _extract_betas(path: Path) -> np.ndarray:
    data = _load_pickle(path)
    if not isinstance(data, dict) or "betas" not in data:
        raise ValueError(f"{path} must contain a dict with a 'betas' field.")
    betas = _to_numpy(data["betas"]).astype(np.float64, copy=False).reshape(-1)
    if betas.size == 0:
        raise ValueError(f"{path} has an empty 'betas' field.")
    return betas


def _bake_shape(base_data: dict, betas: np.ndarray) -> dict:
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
    missing = sorted(required.difference(base_data))
    if missing:
        raise ValueError(f"Base SMPL model is missing keys: {', '.join(missing)}")

    custom = copy.deepcopy(base_data)
    v_template = np.asarray(base_data["v_template"], dtype=np.float64)
    shapedirs = np.asarray(base_data["shapedirs"], dtype=np.float64)
    num_betas = min(betas.size, shapedirs.shape[-1])
    if num_betas == 0:
        raise ValueError("No compatible shape coefficients found.")

    offset = np.einsum("vcn,n->vc", shapedirs[:, :, :num_betas], betas[:num_betas])
    custom_v_template = v_template + offset
    custom["v_template"] = custom_v_template

    j_regressor = base_data["J_regressor"]
    custom["J"] = np.asarray(j_regressor.dot(custom_v_template), dtype=np.float64)
    if isinstance(custom.get("pose_training_info"), dict):
        custom["pose_training_info"] = dict(custom["pose_training_info"])
        custom["pose_training_info"]["gender"] = "custom"
        custom["pose_training_info"]["custom_betas_source"] = True
    return custom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape-pkl",
        default="motion-latent-diffusion/deps/smpl_models/smpl/SMPL_CUSTOM.pkl",
        help="Saved SMPL output pickle containing 'betas'.",
    )
    parser.add_argument(
        "--base",
        default="motion-latent-diffusion/deps/smpl_models/smpl/SMPL_FEMALE.pkl",
        help="Base SMPL model-definition pickle to copy.",
    )
    parser.add_argument(
        "--out",
        default="motion-latent-diffusion/deps/smpl_models/smpl/SMPL_CUSTOM.pkl",
        help="Output SMPL model-definition pickle.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--backup-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When overwriting, keep the previous output as '<out>.bak'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape_path = _resolve(args.shape_pkl)
    base_path = _resolve(args.base)
    out_path = _resolve(args.out)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")

    betas = _extract_betas(shape_path)
    base_data = _load_pickle(base_path)
    if not isinstance(base_data, dict):
        raise ValueError(f"{base_path} must contain a SMPL model-definition dict.")
    custom = _bake_shape(base_data, betas)

    if out_path.exists() and args.overwrite and args.backup_existing:
        backup_path = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, backup_path)
        print(f"Backed up existing file: {backup_path}")

    _save_pickle(out_path, custom)
    print(f"Saved custom SMPL model: {out_path}")
    print(f"Baked {min(betas.size, np.asarray(base_data['shapedirs']).shape[-1])} betas from: {shape_path}")
    print(f"Base model: {base_path}")


if __name__ == "__main__":
    main()
