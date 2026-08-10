# FlowMimic

FlowMimic provides data loading, preprocessing, VAE/VQ-VAE motion autoencoders, and conditional rectified-flow models for motion generation on AIST++ and MVHumanNet.

The current active AIST++ generation path is:

1. Convert SMPL22 motion to HumanML3D-style 263D IK features.
2. Train or load a motion autoencoder:
   - continuous `MotionVAE`, or
   - discrete-latent `MotionVQVAE`.
3. Compute latent mean/std for the selected autoencoder.
4. Train a rectified flow model in normalized latent space.
5. Decode generated latents back to 263D motion and recover SMPL22 joints for evaluation or visualization.

Detailed experiment notes are in `docs/`.

## Project Structure

```text
baselines/
  common/
    evaluation.py             # Shared canonical baseline metrics
  mld/
    configs/                  # MLD AIST++ experiment configs
    scripts/                  # MLD train/eval entrypoints
    tools/                    # MLD preparation and export tools
  motionhiflow/
    configs/                  # MotionHiFlow AIST++ architecture/training config
    scripts/                  # DDP-resumable train and canonical eval entrypoints
    tools/                    # Fresh AIST++ preparation and train-only statistics
  stickmotion/
    configs/                  # StickMotion AIST++ experiment configs
    scripts/                  # StickMotion train/eval entrypoints
    tools/                    # StickMotion training and export tools
flowmimic/
  src/
    config/
      config.json              # Main path and hyperparameter config
      def_smpl45_to_body25.json
      genre_to_id.json
    data/
      dataloader.py             # 3D loading, alignment, fps handling
      openpose.py               # BODY-25 2D loading and preprocessing
      smpl2joints.py            # AIST SMPL -> joints
    metrics/
      distribution_metrics.py
      t2m_feature_extractor.py
    model/
      vae/
        motion_vae.py           # Continuous MotionVAE
        motion_vqvae.py         # MotionVQVAE
        backend.py              # Shared loader/encode/decode wrapper
        datasets/
          dataset_aist.py
          dataset_mvh.py
          balanced_batch_sampler.py
      flow/
        cond_encoder_2d.py
        flow_net.py
        rect_flow.py
        solver.py
        teacher.py
    motion/
      process_motion.py         # SMPL22 <-> 263D feature utilities
  scripts/
    train_vae.py
    eval_vae.py
    train_vqvae.py
    train_flow.py
    eval_flow.py
    sample_flow.py
  tools/
    compute_stats.py
    compute_vae_latent_stats.py
    precompute_ik263.py
    precompute_openpose_cache.py
    prepare_aist_t2m_datasets.py
    process_aist_text_tokens.py
    train_aist_t2m_evaluator_pipeline.py
    export_vae_recon_smpl22_samples.py
    eval_solver_endpoint_gap.py
```

## Data And Preprocessing

### AIST++ 3D Motion

- Raw files: `data/AIST++/Annotations/motions/*.pkl`
- Official splits:
  - `data/AIST++/Annotations/splits/pose_train.txt`
  - `data/AIST++/Annotations/splits/pose_val.txt`
  - `data/AIST++/Annotations/splits/pose_test.txt`
- Alignment in `flowmimic/src/data/dataloader.py`:
  - axis remap: `(x, y, z) -> (x, -z, y)`
  - translation scaling fix on pelvis delta
  - root centering by first-frame pelvis

AIST filenames may differ between motion and text sources: motion files include a concrete music id such as `_mBR0_`, while generated text files may use `_mAll_`. For text-to-motion baselines, the preparation tools treat the music id as irrelevant and match by the remaining AIST fields.

### MVHumanNet 3D Motion

- Per-frame files: `data/MVHumanNet/<part>/<seq>/smpl_param/*.pkl`
- Uses provided joints directly.
- Alignment in `flowmimic/src/data/dataloader.py`:
  - axis remap: `(x, y, z) -> (-y, -x, -z)`
  - root centering by first-frame pelvis

### 2D OpenPose Conditions

- Stored as `.npy`: `[T, 25, 3] = (x, y, conf)`
- Preprocessing in `flowmimic/src/data/openpose.py`:
  - y-down to y-up: `(x, y) -> (x, -y)`
  - root center by pelvis at t=0, BODY-25 index 8
  - FPS unify to `target_fps`
  - confidence upsample via geometric-mean interpolation
  - visibility mask: `conf >= 0.4`
  - cache stores both `vis` and `conf`
- `flowmimic/src/model/flow/cond_encoder_2d.py` additionally applies bbox scale normalization and per-joint mean/std.

### 263D Motion Features

`flowmimic/src/motion/process_motion.py` converts SMPL22 joints to HumanML3D-style 263D features:

```text
[0:1]     root yaw velocity
[1:3]     root x/z velocity
[3:4]     root height
[4:67]    local root-relative joint positions
[67:193]  local 6D rotations
[193:259] local joint velocities
[259:263] foot contacts
```

Mean/std normalization applies to dims `[0:259]`. Foot-contact dims `[259:263]` are not normalized.

## Common Data Tools

```bash
# MVHumanNet split files
python flowmimic/tools/split_datasets.py \
  --mv-root data/MVHumanNet \
  --out-train data/MVHumanNet/mvh_train.txt \
  --out-val data/MVHumanNet/mvh_val.txt

# 263D train statistics
python flowmimic/tools/compute_stats.py --workers 10

# IK263 cache
python flowmimic/tools/precompute_ik263.py --workers 10 --overwrite

# Cache validation
python flowmimic/tools/validate_cache.py

# OpenPose cache and stats
python flowmimic/tools/precompute_openpose_cache.py --workers 10
python flowmimic/tools/compute_openpose_stats.py
```

For AIST++ text-to-motion baselines:

```bash
python flowmimic/tools/process_aist_text_tokens.py

python flowmimic/tools/prepare_aist_t2m_datasets.py \
  --mld-out prepared/aist_mld_humanml3d \
  --stick-out prepared/aist_stickmotion

python baselines/motionhiflow/tools/prepare_aist.py \
  --output prepared/motionhiflow_aist_20260808
```

The prepared MLD/StickMotion baseline motions are first-cropped to the configured max length, currently 196 frames, so their test protocol can be matched by FlowMimic with `--aist-splits test --aist-cameras 01 --aist-crop-mode first`.

## Motion Autoencoders

FlowMimic currently has two motion autoencoder backends.

### Continuous MotionVAE

This is the original continuous latent VAE used by earlier FlowMimic runs.

```bash
python flowmimic/scripts/train_vae.py

python flowmimic/scripts/eval_vae.py \
  --checkpoint checkpoints/vae/len200_smooth_decoder/motion_vae_best.pt
```

The legacy high-dimensional checkpoint `checkpoints/vae/len200_smooth_decoder/motion_vae_best.pt` uses a sequence-like latent and can accept shorter clips such as 196 frames when `seq_len <= max_len`.

### MotionVQVAE

`MotionVQVAE` is a parallel VQ-VAE path. The current AIST++/MVHumanNet experiment uses:

```text
input motion:       [B, 196, 263]
quantized latent:   [B, 16, 256]
code ids:           [B, 16]
reconstruction:     [B, 196, 263]
```

Typical training command:

```bash
torchrun --standalone --nproc_per_node=2 \
  flowmimic/scripts/train_vqvae.py \
  --ddp \
  --datasets AIST,MVH \
  --val-datasets AIST \
  --seq-len 196 \
  --batch-size 48 \
  --epochs 800 \
  --lr 0.0002 \
  --latent-len 16 \
  --latent-token-mode query \
  --codebook-size 1024 \
  --commitment-weight 0.25 \
  --codebook-decay 0.99 \
  --aist-crop-mode random \
  --aist-clip-repeat 32 \
  --aist-val-crop-mode uniform \
  --aist-val-clip-repeat 4 \
  --stats-path data/mean_std_263_train.npz
```

`train_vqvae.py` saves:

```text
motion_vqvae_latest.pt
motion_vqvae_best.pt
```

Checkpoint selection defaults to `val_quality/aist/score`, not plain `val/recon`. Plain 263D validation reconstruction can fail to match visual quality, especially for root drift and fast distal motion.

### Latent Stats

Flow training expects the selected VAE latent target to be normalized. Compute latent stats after choosing the VAE checkpoint:

```bash
python flowmimic/tools/compute_vae_latent_stats.py \
  --checkpoint checkpoints/vqvae/aist_mvh_len196_latent16_code1024_visible_retrain_to200_ddp2_retry_260717/motion_vqvae_epoch200.pt \
  --vae-type motion_vqvae \
  --seq-len 196 \
  --split train \
  --aist-crop-mode first \
  --out-path data/vqvae_latent_stats_aist_train_latent16_epoch200_retry.npz
```

The selected flow checkpoint, VQ-VAE checkpoint, and latent-statistics file must remain a matched set. When selecting a different autoencoder checkpoint, compute a new statistics file with a name that identifies that checkpoint. For continuous MotionVAE checkpoints, use `--vae-type motion_vae` or leave `--vae-type auto`.

## Flow Training

`flowmimic/scripts/train_flow.py` uses `flowmimic/src/model/vae/backend.py`, so the same script can train against either `MotionVAE` or `MotionVQVAE`.

The VQ-flow setup predicts velocity in normalized VQ-VAE latent space:

```text
noise x0 -> rectified flow -> normalized z_q -> latent denorm -> VQ-VAE decoder -> 263D motion -> SMPL22 joints
```

The selected AIST-only Round 0 setup is defined by the `unified_round0_phase1d_cfg5` update-based curriculum. Launch it through the maintained wrapper:

```bash
# One GPU. The launcher resumes flow_round0_last_good.pt when available.
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash flowmimic/scripts/run_aist_unified_phase1d_cfg.sh

# Two GPUs with the same global batch and update schedule.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 GLOBAL_BATCH_SIZE=896 \
  bash flowmimic/scripts/run_aist_unified_phase1d_cfg.sh
```

The curriculum is expressed in optimizer updates so changing GPU count does not change its schedule. It starts with dense `K=196` conditioning, then ramps to `K={196,98,49,28,14,7}` with final probabilities `{0.30,0.15,0.20,0.15,0.12,0.08}`. Condition timestamp patterns ramp from fully even to `{even: 0.50, random: 0.30, boundary_gap: 0.20}`, with additional quotas for difficult sparse cases. True-null CFG dropout ramps from 0% to 5% over the first 1,730 reference updates. The selected setup keeps 5% per-joint dropout, disables blank-frame masking, uses shared-camera condition matching, disables flow-side smoothness regularization, and uses EMA decay 0.99.

Train-time evaluation uses the official AIST test split, camera `01`, first crop, the fixed boundary-gap `K=7` manifest, 470 motions x 3 replications, 50 Heun steps, and guidance scale 2.5. See `flowmimic/scripts/run_aist_unified_phase1d_cfg.sh` and the `unified_round0_phase1d_cfg5` entry in `flowmimic/src/config/config.json` for the complete reproducible configuration.

Less commonly changed defaults live in `flowmimic/src/config/config.json`, grouped under `flow.architecture`, `flow.optimization`, `flow.conditioning`, `flow.regularization`, `flow.eval`, `flow.checkpointing`, and `flow.wandb`. `train_flow.py` still exposes the high-impact experiment knobs on the CLI, then fills internal defaults from config.

Important training controls:

- `--vae-type auto|motion_vae|motion_vqvae` selects or verifies the autoencoder backend.
- `--latent-stats-path` must match the VAE checkpoint and latent shape.
- `--cond-frames-min/--cond-frames-max` controls condition density.
- `--cond-drop-prob` drops individual visible 2D joints.
- `--cond-frame-drop-prob` masks whole condition frames as valid blank tokens. It is disabled in the selected unified pipeline; sparse frames are instead omitted from attention through the condition-count/pattern curriculum.
- `--cfg-drop-prob` replaces the full condition with the learned true-null condition for classifier-free guidance training when the selected curriculum enables true-null conditioning.
- `--cfg-start-epoch` and `--cond-frame-drop-start-epoch` can delay those augmentations.
- `--solver-cond-start-epoch` and `--solver-smooth-start-epoch` schedule solver-side condition and smoothness losses.
- `--solver-reg-subbatch-size` controls how much of each batch runs through the expensive differentiable solver/decode regularizers.
- `--lr-decay-epoch` applies a one-time 0.5 LR decay. Resume checkpoints persist whether this decay has already happened, so resuming no longer halves LR repeatedly.
- `flow.regularization` in config controls the solver method, ramp lengths, solver-step schedule, smoothness domain, decode chunk size, and checkpointing internals.
- `flow.eval` and `evaluator` in config control train-time eval defaults and AIST-trained T2M evaluator asset paths.
- New flow checkpoints include `metadata` plus top-level aliases for `vae_ckpt`, `vae_type`, `latent_stats_path`, `stats_path`, and `openpose_stats_path`.

DDP is supported with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  flowmimic/scripts/train_flow.py \
  --ddp \
  --batch-size 256 \
  <other args>
```

## Flow Sampling

```bash
python flowmimic/scripts/sample_flow.py \
  --checkpoint checkpoints/flow/vqflow_aist_zq16_unified_sparse_cfg5_260728/flow_round0_update68220.pt \
  --seq-len 196 \
  --steps 50 \
  --solver heun \
  --guidance-scale 2.5 \
  --use-ema
```

Current flow checkpoints record the matching VQ-VAE and latent-statistics paths in their metadata. `sample_flow.py` resolves those assets automatically; pass `--vae-checkpoint` and `--latent-stats-path` explicitly only for legacy checkpoints without complete metadata or when intentionally overriding the matched assets.

Specific AIST sample and camera:

```bash
python flowmimic/scripts/sample_flow.py \
  --checkpoint checkpoints/flow/vqflow_aist_zq16_unified_sparse_cfg5_260728/flow_round0_update68220.pt \
  --dataset aist \
  --sample-path data/AIST++/Annotations/motions/<motion>.pkl \
  --camera 01 \
  --seq-len 196 \
  --steps 50 \
  --guidance-scale 2.5 \
  --use-ema
```

Each invocation creates a timestamped run directory and updates `output/flow/last` to point to the newest result:

```text
output/flow/<checkpoint_name>/<timestamp>/result_smpl22.npy
output/flow/<checkpoint_name>/<timestamp>/result_meta.json
output/flow/last -> output/flow/<checkpoint_name>/<timestamp>
```

`result_smpl22.npy` is an SMPL22 joint sequence. Use `flowmimic/tools/extract_cond_media.py --meta output/flow/last/result_meta.json` to extract the matching conditioning video/frames.

## Flow Evaluation

`eval_flow.py` supports geometric metrics, smoothness metrics, condition error metrics, and optional AIST-trained T2M distribution metrics.

Matched AIST text-baseline-style evaluation:

```bash
python flowmimic/scripts/eval_flow.py \
  --flow-ckpt checkpoints/flow/<run>/flow_round0_last.pt \
  --vae-ckpt checkpoints/vqvae/<run>/motion_vqvae_latest.pt \
  --vae-type motion_vqvae \
  --datasets AIST \
  --aist-splits test \
  --aist-cameras 01 \
  --aist-crop-mode first \
  --num-samples 0 \
  --replications 3 \
  --steps 16,50 \
  --solver heun \
  --cond-frames 196 \
  --guidance-scale 1.0 \
  --use-ema
```

Notes:

- `--num-samples 0` means use the full selected split/camera set.
- For the AIST test split with camera `01`, this gives one video per unique test motion, matching the 470-motion baseline setup.
- FlowMimic evaluation repeats stochastic generation with `--replications`; summaries include mean/std/confidence fields in JSON/CSV.
- During flow eval, generated latents are denormalized with the latent stats path recorded in the flow checkpoint metadata, or the config fallback. For older checkpoints without metadata, pass `--latent-stats-path` if the config default does not match that run.

Default output files:

```text
output/eval/<flow_ckpt_parent>/flow_eval.csv
output/eval/<flow_ckpt_parent>/flow_eval.json
output/eval/<flow_ckpt_parent>/flow_eval_steps.png
```

If T2M evaluator paths are configured in `config.json`, `eval_flow.py` computes:

```text
fid
mmdist
matching_score
diversity
multimodality, when requested
```

Use `--no-dist` to skip T2M distribution metrics.

Train-time eval in `train_flow.py` uses the same core evaluator. W&B logging is intentionally curated to the main metrics only, while the full JSON/CSV eval files still keep auxiliary values such as std/conf/ref counts.

## Baseline Utilities

MLD, StickMotion, and MotionHiFlow AIST training pipelines:

```bash
bash baselines/mld/scripts/run_aist_mvh_pipeline.sh
bash baselines/stickmotion/scripts/run_aist_no_locus.sh

conda activate motionhiflow
CUDA_VISIBLE_DEVICES=0,1 bash baselines/motionhiflow/scripts/launch.sh vae 2
```

Canonical baseline evaluation uses the same official AIST test split, first-196 crop, frozen AIST T2M motion encoder, evaluator normalization, and physical-motion metrics as `eval_flow.py`:

```bash
python baselines/mld/scripts/eval.py \
  --mld-ckpt /path/to/mld.ckpt \
  --replications 3 \
  --save-json output/eval/mld.json

python baselines/stickmotion/scripts/eval.py \
  --stickmotion-ckpt /path/to/stickmotion.ckpt \
  --replications 3 \
  --save-json output/eval/stickmotion.json

conda run -n motionhiflow python baselines/motionhiflow/scripts/eval.py \
  --checkpoint /path/to/motionhiflow_flow.pt \
  --replications 10 \
  --save-json output/eval/motionhiflow.json
```

MotionHiFlow-specific architecture, fresh-statistics, DDP switching, W&B, and evaluation details are documented in `baselines/motionhiflow/README.md`.

StickMotion sample export:

```bash
python baselines/stickmotion/tools/export_aist_samples.py \
  --ckpt /path/to/no_locus_stickmotion.ckpt \
  --max-samples 4 \
  --output-space blender
```

Generate an aligned visual comparison for one AIST test or validation motion:

```bash
python flowmimic/tools/sample_aist_method_comparison.py \
  --split test \
  --sample-index 0 \
  --camera 01 \
  --start 0 \
  --condition-frames 28 \
  --stickmotion-sketch-frames 24 98 171 \
  --flow-steps 50 \
  --flow-ckpt checkpoints/flow/vqflow_aist_zq16_unified_sparse_cfg5_260728/flow_round0_update68220.pt \
  --flow-use-ema \
  --mld-ckpt runs/mld/mld/aist_ik263_mld_196_aistmvh_vae/checkpoints/epoch=2499.ckpt \
  --stickmotion-ckpt runs/stickmotion/human_ml3d/aist_remodiffuse_no_locus_260730/epoch=591-step=25644.ckpt \
  --visualization-mode rigged \
  --rigged-model web_view/assets/smpl22_rigged_calibrated.glb \
  --flow-gpu 0 \
  --mld-gpu 0 \
  --stickmotion-gpu 1
```

The comparison uses a 196-frame clip beginning at `--start`. FlowMimic receives the requested number of uniformly spaced camera-specific pose conditions. MLD and StickMotion receive the same camera-matched caption, and StickMotion also receives its three generated stickman sketches. The selected MLD checkpoint uses its retrained AIST+MVHumanNet VAE and matched generator statistics. The selected StickMotion baseline disables its trajectory/locus branch. The output bundle contains four Blender Z-up `[196, 22, 3]` arrays, the StickMotion sketch tracks, per-method metadata/logs, and `comparison_manifest.json`.

Load all four motions, the text, and the StickMotion sketches into one Blender scene:

```bash
blender --python flowmimic/tools/vis_smpl22_blender.py -- \
  --manifest output/aist_method_comparisons/<run>/comparison_manifest.json \
  --visualization-mode rigged \
  --rigged-model web_view/assets/smpl22_rigged_calibrated.glb
```

Pass `--save-blend comparison.blend` after the manifest to save the scene beside the
manifest. The timeline includes markers for FlowMimic condition frames and StickMotion
sketch frames. Use `--sample-id` instead of `--sample-index` to select a specific motion,
and `--caption-index` when a text file contains multiple descriptions. Pass
`--caption-text "..."` to edit the selected description; the tool regenerates the
HumanML3D token sequence for StickMotion while sending the same confirmed text to MLD.

Rebuild the calibrated visualization rig after replacing the source GLB:

```bash
blender --background --python flowmimic/tools/calibrate_smpl22_rig.py -- \
  --input web_view/assets/smpl22_rigged.glb \
  --output web_view/assets/smpl22_rigged_calibrated.glb \
  --knee-center-blend 0.70
```

The calibration centers the torso chain, mirrors and levels paired joints, and moves the knee pivots conservatively toward the measured mesh cross-section centers without changing the mesh rest shape.

## Diagnostic Tools

```bash
# Compare one-step rectified-flow endpoint estimates with multi-step solver endpoints.
python flowmimic/tools/eval_solver_endpoint_gap.py \
  --checkpoint checkpoints/flow/vqflow_aist_zq16_unified_sparse_cfg5_260728/flow_round0_update68220.pt \
  --vae-ckpt checkpoints/vqvae/aist_mvh_len196_latent16_code1024_visible_retrain_to200_ddp2_retry_260717/motion_vqvae_epoch200.pt \
  --latent-stats-path data/vqvae_latent_stats_aist_train_latent16_epoch200_retry.npz

# Export paired input/reconstruction SMPL22 clips for visual inspection.
python flowmimic/tools/export_vae_recon_smpl22_samples.py
```

## Config Highlights

Main config: `flowmimic/src/config/config.json`

Important keys:

- `data.motion.seq_len`: default sequence length, currently 200 in config; many AIST VQ-flow experiments override this to 196.
- `paths.stats_path`, `paths.latent_stats_path`, `paths.openpose_stats_path`: default 263D, latent, and BODY-25 stats paths. New flow checkpoints also record these paths in metadata.
- `vae.ckpt`: default continuous MotionVAE checkpoint.
- `evaluator.*`: AIST-trained T2M evaluator assets.
- `flow.*`: flow architecture, optimization, conditioning, regularization, eval, checkpointing, and W&B defaults.
- `sample.*`: default sample solver/output settings and rare VAE backend overrides.

## Current Caveats

- The VQ-VAE path is visually much better than the compact continuous VAE experiments, but it can still under-reconstruct very fast distal-limb motion and high-amplitude side-to-side motion. See `docs/[VQ-VAE] Fast Motion Weakness and Flow Integration.md`.
- Plain 263D `val/recon` is not always aligned with visual reconstruction quality. Prefer fixed-joint quality metrics and visual exports when selecting VAE checkpoints.
- AIST text baselines use first-cropped 196-frame motions. FlowMimic training can use random crops to enlarge the effective dataset, but matched baseline eval should use the test split, camera `01`, first crop, and 3 replications.
- Solver-side condition and smoothness losses are expensive because they decode generated latents and recover SMPL22 joints. Use subbatching/chunking controls when VRAM or wall-clock time becomes limiting.
