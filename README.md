# FlowMimic

FlowMimic provides data loading, preprocessing, a multi-domain Motion VAE, and a conditional rectified flow model for motion generation on AIST++ and MVHumanNet.

## Project structure

```
flowmimic/
  src/
    config/
      config.json            # Main config (paths + hyperparams)
      def_smpl45_to_body25.json
      genre_to_id.json
    data/
      dataloader.py           # 3D loading + alignment + fps unify
      smpl2joints.py          # SMPL -> joints (AIST)
      stats.py                # legacy BODY-25 stats
      openpose.py             # OpenPose 2D loading + preprocessing
    model/
      vae/
        motion_vae.py
        cond_embedding.py
        adaln.py
        transformer_blocks.py
        losses.py
        stats.py
        datasets/
          dataset_aist.py
          dataset_mvh.py
          aist_filename_parser.py
          label_map_builder.py
          balanced_batch_sampler.py
      flow/
        cond_encoder_2d.py
        style_embed.py
        flow_blocks.py
        flow_net.py
        rect_flow.py
        time_embed.py
        solver.py
        teacher.py
        cond_api.py
    motion/
      process_motion.py       # SMPL22 -> 263D feature extractor
      ik/                     # Minimal IK deps
  scripts/
    train_vae.py
    eval_vae.py
    train_flow.py
    sample_flow.py
  tools/
    split_datasets.py
    compute_stats.py
    precompute_ik263.py
    validate_cache.py
    run_openpose_on_aist.sh
    json2npy_openpose.py
    combine_mvh_openpose.sh
    compute_openpose_stats.py
    decompress_mvhumannet.sh
    test_ik_roundtrip.py

main.py
```

## Data and preprocessing

### 3D (AIST++)

- Files: `data/AIST++/Annotations/motions/*.pkl`
- SMPL -> joints in `flowmimic/src/data/smpl2joints.py`
- Alignment in `flowmimic/src/data/dataloader.py`:
  - Axis remap: `(x, y, z) -> (x, -z, y)`
  - Translation scaling fix on pelvis delta
  - Root centering by first-frame pelvis

### 3D (MVHumanNet)

- Per-frame files: `data/MVHumanNet/<part>/<seq>/smpl_param/*.pkl`
- Uses `joints` directly
- Alignment in `flowmimic/src/data/dataloader.py`:
  - Axis remap: `(x, y, z) -> (-y, -x, -z)`
  - Root centering by first-frame pelvis

### 2D OpenPose (BODY-25)

- Stored as `.npy`: `[T, 25, 3] = (x, y, conf)`
- Preprocessing in `flowmimic/src/data/openpose.py`:
  - y-down -> y-up: `(x, y) -> (x, -y)`
  - Root center by pelvis at t=0 (index 8)
  - FPS unify to `target_fps` (AIST 60→30 via stride, MVH 5→30 via PCHIP)
  - Conf upsample via geometric-mean interpolation
  - vis_mask = (conf > 0)
- Additional normalization in `flowmimic/src/model/flow/cond_encoder_2d.py`:
  - bbox scale normalization
  - per-joint mean/std (training stats in `data/openpose_stats.npz`)

### 263D feature representation

- Blender Z-up -> SMPL Y-up before IK: `(x, y, z) -> (x, z, -y)`
- `flowmimic/src/motion/process_motion.py` converts SMPL22 to 263D:
  - root_yaw_vel: [0:1]
  - root_xz_vel: [1:3]
  - root_y: [3:4]
  - ric: [4:67]
  - rot_6d: [67:193]
  - local_vel: [193:259]
  - feet_contact: [259:263]

### 263D normalization

- Mean/std over training set dims [0:259]
- feet_contact [259:263] not normalized
- Stored in `data/mean_std_263_train.npz`

## Splits

- AIST: `data/AIST++/Annotations/splits/pose_train.txt`, `pose_val.txt`
- MVH: `data/MVHumanNet/mvh_train.txt`, `mvh_val.txt`

## Tools (recommended order)

1) MVH splits:
```
python flowmimic/tools/split_datasets.py \
  --mv-root data/MVHumanNet \
  --out-train data/MVHumanNet/mvh_train.txt \
  --out-val data/MVHumanNet/mvh_val.txt
```

2) 263D mean/std:
```
python flowmimic/tools/compute_stats.py --workers 10
```

3) IK cache:
```
python flowmimic/tools/precompute_ik263.py --workers 10 --overwrite
```

4) Cache validation:
```
python flowmimic/tools/validate_cache.py
```

5) OpenPose utilities:
```
bash flowmimic/tools/run_openpose_on_aist.sh
python flowmimic/tools/json2npy_openpose.py --video_dir <json_dir> --out <output.npy>
bash flowmimic/tools/combine_mvh_openpose.sh
python flowmimic/tools/compute_openpose_stats.py
```

## VAE training / evaluation

Training:
```
python flowmimic/scripts/train_vae.py
```

Evaluation:
```
python flowmimic/scripts/eval_vae.py --checkpoint checkpoints/motion_vae_best.pt
```

## Flow training / sampling

Training (round 0):
```
python flowmimic/scripts/train_flow.py --reflow-round 0
```

Reflow rounds:
```
python flowmimic/scripts/train_flow.py --reflow-round 1 --teacher-ckpt checkpoints_flow/flow_round0_last.pt
python flowmimic/scripts/train_flow.py --reflow-round 2 --teacher-ckpt checkpoints_flow/flow_round1_last.pt
```

Sampling:
```
python flowmimic/scripts/sample_flow.py --checkpoint <flow_ckpt>
```

Random sample from val split (auto dataset):
```
python flowmimic/scripts/sample_flow.py --checkpoint checkpoints/flow/flow_round0_last.pt
```

Specific sample with camera:
```
python flowmimic/scripts/sample_flow.py --checkpoint checkpoints/flow/flow_round0_last.pt --dataset aist --sample-path data/AIST++/Annotations/motions/xxx.pkl --camera 01
```

## Config highlights

Main config: `flowmimic/src/config/config.json`

- FPS: `target_fps=30` (`aist_fps=60`, `mvh_fps=5`)
- 263D stats: `stats_path`
- OpenPose stats: `openpose_stats_path`
- Flow hyperparams: `flow.*`

## Notes

- MVH has NaNs in raw joints; those sequences are skipped in caching/stats.
- Flow training currently uses sparse 2D conditions sampled on the fly.
