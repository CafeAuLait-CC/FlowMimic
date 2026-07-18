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

RUN_NAME="${RUN_NAME:-vqflow_aist_zq16_unified_round0_260718}"
WANDB_ID="${WANDB_ID:-u0r16j18}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_UPDATES="${MAX_UPDATES:-48440}"
MASTER_PORT="${MASTER_PORT:-29618}"
CHECKPOINT_DIR="checkpoints/flow/${RUN_NAME}"
LOG_DIR="training_logs/${RUN_NAME}"
LOG_PATH="${LOG_DIR}/train.out"
EVAL_DIR="${LOG_DIR}/async_eval"
LAST_GOOD="${CHECKPOINT_DIR}/flow_round0_last_good.pt"

mkdir -p "$CHECKPOINT_DIR" "$EVAL_DIR"
echo "$$" > "${LOG_DIR}/launcher.pid"

RESUME_ARGS=()
if [[ -s "$LAST_GOOD" ]]; then
  RESUME_ARGS=(--resume "$LAST_GOOD")
fi

exec > >(tee -a "$LOG_PATH") 2>&1
printf '[%s] Starting %s\n' "$(date -Is)" "$RUN_NAME"
printf '[%s] GPUs=%s nproc=%s max_updates=%s resume=%s\n' \
  "$(date -Is)" "$GPU_IDS" "$NPROC_PER_NODE" "$MAX_UPDATES" \
  "${RESUME_ARGS[*]:-none}"

env \
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  flowmimic/scripts/train_flow.py \
  --ddp \
  --epochs 1000 \
  --max-updates "$MAX_UPDATES" \
  --curriculum unified_round0 \
  --batch-size 512 \
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
  --eval-steps 16,50 \
  --eval-aist-splits test \
  --eval-aist-cameras 01 \
  --eval-aist-crop-mode first \
  --eval-replications 3 \
  --eval-cond-frames 196 \
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
  "${RESUME_ARGS[@]}" \
  --wandb-project FlowMimic \
  --wandb-group VQFlow-Unified-Round0 \
  --wandb-name "$RUN_NAME" \
  --wandb-id "$WANDB_ID" \
  --wandb-resume allow \
  --wandb-tags aist,vqvae,zq16,unified-round0,update-curriculum,shared-camera,joint-drop5,no-cfg,no-frame-mask,no-flow-smooth,ema099,eval470x3,steps16_50

status=$?
printf '[%s] %s exited with status %s\n' "$(date -Is)" "$RUN_NAME" "$status"
exit "$status"
