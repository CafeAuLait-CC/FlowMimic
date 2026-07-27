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

RUN_NAME="${RUN_NAME:-vqflow_aist_zq16_unified_phase1d_260722}"
WANDB_ID="${WANDB_ID:-sp1d0722}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29623}"
BATCH_SIZE="${BATCH_SIZE:-896}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SOLVER_REG_SUBBATCH_SIZE="${SOLVER_REG_SUBBATCH_SIZE:-32}"
MAX_UPDATES="${MAX_UPDATES:-62280}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
CHECKPOINT_DIR="checkpoints/flow/${RUN_NAME}"
LOG_DIR="training_logs/${RUN_NAME}"
LOG_PATH="${LOG_DIR}/train.out"
EVAL_DIR="${LOG_DIR}/async_eval"
LAST_GOOD="${CHECKPOINT_DIR}/flow_round0_last_good.pt"
EVAL_MANIFEST="${EVAL_MANIFEST:-training_logs/phase0_sparse_condition_zero_shot_20260720/manifests/boundary_gap_k7.json}"

mkdir -p "$CHECKPOINT_DIR" "$EVAL_DIR"
echo "$$" > "${LOG_DIR}/launcher.pid"

RESUME_ARGS=()
if [[ -s "$LAST_GOOD" ]]; then
  RESUME_ARGS=(--resume "$LAST_GOOD")
fi
if [[ ! -s "$EVAL_MANIFEST" ]]; then
  echo "Evaluation condition manifest does not exist: $EVAL_MANIFEST" >&2
  exit 1
fi

exec > >(tee -a "$LOG_PATH") 2>&1
printf '[%s] Starting %s\n' "$(date -Is)" "$RUN_NAME"
printf '[%s] GPUs=%s nproc=%s per_gpu_batch=%s global_batch=%s max_updates=%s resume=%s\n' \
  "$(date -Is)" "$GPU_IDS" "$NPROC_PER_NODE" "$BATCH_SIZE" \
  "$((BATCH_SIZE * NPROC_PER_NODE))" "$MAX_UPDATES" \
  "${RESUME_ARGS[*]:-none}"

LAUNCH=("$PYTHON_BIN" -u)
DDP_ARGS=()
if (( NPROC_PER_NODE > 1 )); then
  LAUNCH+=(
    -m torch.distributed.run
    --nproc_per_node="$NPROC_PER_NODE"
    --master_port="$MASTER_PORT"
  )
  DDP_ARGS=(--ddp)
fi
LAUNCH+=(flowmimic/scripts/train_flow.py)

env \
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${LAUNCH[@]}" \
  "${DDP_ARGS[@]}" \
  --epochs 1000 \
  --max-updates "$MAX_UPDATES" \
  --curriculum unified_round0_phase1d \
  --batch-size "$BATCH_SIZE" \
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
  --eval-aist-splits test \
  --eval-aist-cameras 01 \
  --eval-aist-crop-mode first \
  --eval-replications 3 \
  --eval-cond-frames 7 \
  --eval-cond-pattern boundary_gap \
  --eval-cond-pattern-seed 20260720 \
  --eval-condition-manifest "$EVAL_MANIFEST" \
  --async-cpu-eval \
  --async-eval-log-dir "$EVAL_DIR" \
  --num-workers "$NUM_WORKERS" \
  --solver-reg-subbatch-size "$SOLVER_REG_SUBBATCH_SIZE" \
  --lambda-cond 0.001 \
  --solver-smooth-start-epoch 999999 \
  --lambda-acc 0.0 \
  --lambda-jerk 0.0 \
  --smooth-every 999999 \
  --cond-match-camera-mode shared \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  "${RESUME_ARGS[@]}" \
  --wandb-project FlowMimic \
  --wandb-group VQFlow-Unified-Phase1D \
  --wandb-name "$RUN_NAME" \
  --wandb-id "$WANDB_ID" \
  --wandb-resume "$WANDB_RESUME" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-tags aist,vqvae,zq16,phase1d,integrated-round0,pattern-curriculum,sparse-joint-quotas,legacy-attention,shared-camera,joint-drop5,no-cfg,no-frame-mask,no-flow-smooth,ema099,test-selection,boundary-gap-k7,eval470x3,steps50

status=$?
printf '[%s] %s exited with status %s\n' "$(date -Is)" "$RUN_NAME" "$status"
exit "$status"
