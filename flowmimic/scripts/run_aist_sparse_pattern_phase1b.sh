#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_BASE="${CONDA_BASE:-$("$CONDA_EXE" info --base)}"
PYTHON_BIN="${FLOWMIMIC_PYTHON:-$CONDA_BASE/envs/flowmimic-310/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "FlowMimic Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-vqflow_aist_zq16_sparse_pattern_phase1b_relative_time_260721}"
WANDB_ID="${WANDB_ID:-sp1brt21}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
MAX_UPDATES="${MAX_UPDATES:-}"
CHECKPOINT_DIR="checkpoints/flow/${RUN_NAME}"
LOG_DIR="training_logs/${RUN_NAME}"
LOG_PATH="${LOG_DIR}/train.out"
EVAL_DIR="${LOG_DIR}/async_eval"
LAST_GOOD="${CHECKPOINT_DIR}/flow_round0_last_good.pt"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-checkpoints/flow/vqflow_aist_zq16_sparse_pattern_phase1a_target_260721/flow_round0_update76276.pt}"
EVAL_MANIFEST="${EVAL_MANIFEST:-training_logs/phase1a_sparse_condition_20260721/manifests/val_boundary_gap_k7.json}"

mkdir -p "$CHECKPOINT_DIR" "$EVAL_DIR"
echo "$$" > "${LOG_DIR}/launcher.pid"

RESET_OPT_ARGS=()
if [[ -s "$LAST_GOOD" ]]; then
  RESUME_PATH="$LAST_GOOD"
else
  RESUME_PATH="$SOURCE_CHECKPOINT"
  RESET_OPT_ARGS=(--reset-optimizer)
fi
if [[ ! -s "$RESUME_PATH" ]]; then
  echo "Resume checkpoint does not exist: $RESUME_PATH" >&2
  exit 1
fi
if [[ ! -s "$EVAL_MANIFEST" ]]; then
  echo "Evaluation condition manifest does not exist: $EVAL_MANIFEST" >&2
  exit 1
fi

MAX_UPDATE_ARGS=()
if [[ -n "$MAX_UPDATES" ]]; then
  MAX_UPDATE_ARGS=(--max-updates "$MAX_UPDATES")
fi

exec > >(tee -a "$LOG_PATH") 2>&1
printf '[%s] Starting %s\n' "$(date -Is)" "$RUN_NAME"
printf '[%s] GPU=%s curriculum=%s max_updates=%s resume=%s reset_optimizer=%s\n' \
  "$(date -Is)" "$GPU_ID" "sparse_pattern_phase1b_relative_time" \
  "${MAX_UPDATES:-config}" "$RESUME_PATH" "${#RESET_OPT_ARGS[@]}"

env \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u flowmimic/scripts/train_flow.py \
  --epochs 1000 \
  "${MAX_UPDATE_ARGS[@]}" \
  "${RESET_OPT_ARGS[@]}" \
  --curriculum sparse_pattern_phase1b_relative_time \
  --batch-size 640 \
  --datasets AIST \
  --seq-len 196 \
  --latent-stats-path data/vqvae_latent_stats_aist_train_latent16_epoch200_retry.npz \
  --vae-ckpt checkpoints/vqvae/aist_mvh_len196_latent16_code1024_visible_retrain_to200_ddp2_retry_260717/motion_vqvae_epoch200.pt \
  --vae-type motion_vqvae \
  --aist-crop-mode random \
  --aist-clip-repeat 64 \
  --cond-drop-prob 0.05 \
  --cond-frame-drop-prob 0.0 \
  --cond-frame-drop-start-epoch 999999 \
  --cfg-drop-prob 0.0 \
  --cfg-start-epoch 999999 \
  --ema \
  --ema-decay 0.99 \
  --eval-guidance-scale 1.0 \
  --eval-steps 50 \
  --eval-aist-splits val \
  --eval-aist-cameras 01 \
  --eval-aist-crop-mode first \
  --eval-replications 3 \
  --eval-cond-frames 7 \
  --eval-cond-pattern boundary_gap \
  --eval-cond-pattern-seed 20260721 \
  --eval-condition-manifest "$EVAL_MANIFEST" \
  --async-cpu-eval \
  --async-eval-log-dir "$EVAL_DIR" \
  --num-workers 8 \
  --solver-reg-subbatch-size 32 \
  --lambda-cond 0.001 \
  --solver-smooth-start-epoch 999999 \
  --lambda-acc 0.0 \
  --lambda-jerk 0.0 \
  --smooth-every 999999 \
  --cond-match-camera-mode shared \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --resume "$RESUME_PATH" \
  --wandb-project FlowMimic \
  --wandb-group VQFlow-Sparse-Pattern-Phase1B \
  --wandb-name "$RUN_NAME" \
  --wandb-id "$WANDB_ID" \
  --wandb-resume allow \
  --wandb-mode "$WANDB_MODE" \
  --wandb-tags "aist,vqvae,zq16,phase1b,relative-time-bias,boundary-k7-quota,shared-camera,joint-drop5,no-cfg,no-frame-mask,no-flow-smooth,ema099,val-boundary-gap-eval,k7,eval3,steps50"

status=$?
printf '[%s] %s exited with status %s\n' "$(date -Is)" "$RUN_NAME" "$status"
exit "$status"
