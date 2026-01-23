# FlowMimic

This repo contains data loading, preprocessing, and a multi-domain conditional Motion VAE for AIST++ and MVHumanNet. The README documents the current state so future works can pick up quickly.

## Project structure

```
common/
  dataloader.py            # AIST/MVHumanNet loading, alignment, SMPL22 selection
config/
  config.json              # Main config (paths + hyperparams)
  def_aist2body25.json      # SMPL to BODY-25 mapping w/ computed pelvis/neck
  genre_to_id.json          # Genre id mapping (0=unknown)
  mean_std_263_train.npz    # Blended mean/std for 263D features (train splits)
models/
  vae/
    motion_vae.py           # Conditional VAE (AdaLN Transformer)
    cond_embedding.py       # Domain/style embeddings + cond MLP
    adaln.py                # AdaLayerNorm
    transformer_blocks.py   # Transformer block with AdaLN
    losses.py               # 263D grouped losses + smoothness
    stats.py                # Mean/std computation helpers
    datasets/
      dataset_aist.py        # AIST dataset (SMPL22 -> 263D)
      dataset_mvh.py         # MVH dataset (sequence -> 263D)
      aist_filename_parser.py
      label_map_builder.py
      balanced_batch_sampler.py
motion_ik/
  ...                       # Minimal IK deps (skeleton + quaternion + params)
process_motion.py           # SMPL22 -> 263D feature extractor
scripts/
  train_vae.py              # Training loop (balanced batches)
  eval_vae.py               # Eval on val splits
utils/
  config.py                 # load_config()
  smpl2joints.py            # SMPL -> joints for AIST
  stats.py                  # (old stats helpers for BODY-25)

tools/
  split_datasets.py         # Create MVH train/val splits
  precompute_ik263.py        # Cache 263D features (AIST + MVH)
  compute_stats.py           # Compute blended mean/std from splits
  validate_cache.py          # Check cached .npy for NaN/Inf
  decompress_mvhumannet.sh   # Tar.gz decompressor for MVHumanNet

main.py                      # Small demo scripts (frames, pelvis trajectory)
implementation explanation.txt
Task Summary.txt
Multi-Domain Conditional Motion VAE.txt
ver1.1.txt                   # V1.1 requirements for 263D features
```

## Data and preprocessing

### AIST++

- Files: `data/AIST++/Annotations/motions/*.pkl`
- Joints are produced from SMPL params (SMPLX) in `utils/smpl2joints.py`.
- Alignment in `common/dataloader.py`:
  - Axis remap: `(x, y, z) -> (x, -z, y)`
  - Translation scaling fix using `smpl_scaling` on pelvis trajectory delta only
  - Root centering: subtract pelvis of first frame

### MVHumanNet

- Per-frame files: `data/MVHumanNet/<part>/<sequence>/smpl_param/*.pkl`
- `joints` key is read directly (shape (1,45,3)).
- Alignment in `common/dataloader.py`:
  - Axis remap: `(x, y, z) -> (-y, -x, -z)`
  - Root centering: subtract pelvis of first frame in the sequence

### Feature representation (263D)

- `process_motion.smpl_to_ik263()` converts SMPL22 to the 263D vector.
- Layout (0-based, end-exclusive):
  - root_yaw_vel:   [0:1]
  - root_xz_vel:    [1:3]
  - root_y:         [3:4]
  - ric:            [4:67]   (21*3)
  - rot_6d:         [67:193] (21*6)
  - local_vel:      [193:259] (22*3)
  - feet_contact:   [259:263] (4)

### Normalization

- Mean/std computed from blended train splits (AIST + MVH) for dims [0:259].
- feet_contact [259:263] is NOT normalized.
- Stats stored at `data/mean_std_263_train.npz`.

## Splits

- AIST splits: `data/AIST++/Annotations/splits/pose_train.txt` and `pose_val.txt`.
- MVH splits are created by `tools/split_datasets.py`:
  - `data/MVHumanNet/mvh_train.txt`
  - `data/MVHumanNet/mvh_val.txt`
- Training uses AIST train + MVH train; evaluation uses val splits.

## Caching pipeline

To avoid slow on-the-fly IK, cache all 263D features:

```
python tools/precompute_ik263.py --workers 10
```

Cache layout:

- AIST: `data/cached_ik/aist/<name>.npy`
- MVH:  `data/cached_ik/mvh/<relative smpl_param dir>.npy`

Validate cache:

```
python tools/validate_cache.py
```

Bad files are logged in:

- `data/cached_ik/bad_aist.txt`
- `data/cached_ik/bad_mvh.txt`

## Stats computation

Compute blended mean/std from train splits:

```
python tools/compute_stats.py --workers 10
```

This skips sequences with NaN/Inf and logs a skip count.

## VAE architecture (summary)

- Input: `[B,T,263]`
- Encoder/Decoder: AdaLN Transformer, D_model=512, 8 layers, 8 heads
- Latent: per-frame `z` with D_z=256
- Conditioning: domain + style embeddings -> MLP -> AdaLN
- Style head: optional classifier on pooled encoder states

Losses (models/vae/losses.py):

- Recon (continuous [0:259]): SmoothL1 (normalized space)
- Contact (259:263): BCEWithLogits
- KL (masked) with warmup
- Smoothness: only on [0:4], [67:193], [193:259]
- Style CE on AIST only (style_id != 0)

## Training

```
python scripts/train_vae.py
```

Defaults from `config/config.json`:

- seq_len=120
- train_batch_size=64
- d_in=263, d_z=256
- KL warmup 40000 steps
- Save checkpoints every `val_every_epochs` (default 10)
- Save latest + best-val checkpoints

Notes:

- MVH is subsampled each epoch to roughly match AIST train size.
- Dataloading uses cached features if present.

## Evaluation

```
python scripts/eval_vae.py --checkpoint checkpoints/motion_vae_best.pt
```

Outputs recon/KL/style metrics for AIST and MVH val splits.

## Config

Main config: `config/config.json`

- Dataset paths: `aist_motions_dir`, `mvhumannet_root`
- Splits: `aist_split_train`, `aist_split_val`, `mvh_split_train`, `mvh_split_val`
- Cache: `cache_root`
- Stats: `stats_path`
- Hyperparams: d_in, d_z, seq_len, batch sizes, kl weights, etc.

## Notes / Known issues

- Some MVHumanNet sequences contain NaNs in raw joints; they are skipped in
  preprocessing and caching.
- If training reports non-finite outputs, check cached data + mean/std.
