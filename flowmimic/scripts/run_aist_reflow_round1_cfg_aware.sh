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

RUN_NAME="${RUN_NAME:-vqflow_aist_zq16_reflow1_cfgaware_260821}"
WANDB_ID="${WANDB_ID:-r1cfg2p5}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29821}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-896}"
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if (( GLOBAL_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by NPROC_PER_NODE" >&2
    exit 1
  fi
  BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NPROC_PER_NODE))
fi
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_UPDATES="${MAX_UPDATES:-8650}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/flow/vqflow_aist_zq16_unified_sparse_cfg5_260728/flow_round0_update68220.pt}"
INIT_CKPT="${INIT_CKPT:-checkpoints/flow/vqflow_aist_zq16_reflow1_strict_260820/flow_round1_update17300.pt}"
VAE_CKPT="${VAE_CKPT:-checkpoints/vqvae/aist_mvh_len196_latent16_code1024_visible_retrain_to200_ddp2_retry_260717/motion_vqvae_epoch200.pt}"
LATENT_STATS="${LATENT_STATS:-data/vqvae_latent_stats_aist_train_latent16_epoch200_retry.npz}"
EVAL_MANIFEST="${EVAL_MANIFEST:-training_logs/phase0_sparse_condition_zero_shot_20260720/manifests/boundary_gap_k7.json}"

CHECKPOINT_DIR="checkpoints/flow/${RUN_NAME}"
LOG_DIR="training_logs/${RUN_NAME}"
LOG_PATH="${LOG_DIR}/train.out"
EVAL_DIR="${LOG_DIR}/async_eval"
LAST_GOOD="${CHECKPOINT_DIR}/flow_round1_last_good.pt"

mkdir -p "$CHECKPOINT_DIR" "$EVAL_DIR"
echo "$$" > "${LOG_DIR}/launcher.pid"

for required_path in "$TEACHER_CKPT" "$INIT_CKPT" "$VAE_CKPT" "$LATENT_STATS" "$EVAL_MANIFEST"; do
  if [[ ! -s "$required_path" ]]; then
    echo "Required CFG-aware reflow input does not exist or is empty: $required_path" >&2
    exit 1
  fi
done

START_ARGS=()
if [[ -s "$LAST_GOOD" ]]; then
  START_ARGS=(--resume "$LAST_GOOD")
else
  START_ARGS=(--init-from "$INIT_CKPT" --init-use-ema)
fi

exec > >(tee -a "$LOG_PATH") 2>&1
printf '[%s] Starting %s\n' "$(date -Is)" "$RUN_NAME"
printf '[%s] GPUs=%s nproc=%s per_gpu_batch=%s global_batch=%s max_updates=%s start=%s\n' \
  "$(date -Is)" "$GPU_IDS" "$NPROC_PER_NODE" "$BATCH_SIZE" \
  "$((BATCH_SIZE * NPROC_PER_NODE))" "$MAX_UPDATES" "${START_ARGS[*]}"

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
  FLOWMIMIC_WANDB_DISABLE_STATS=1 \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${LAUNCH[@]}" \
  "${DDP_ARGS[@]}" \
  --epochs 1000 \
  --max-updates "$MAX_UPDATES" \
  --curriculum reflow_round1_cfg_aware \
  --reflow-round 1 \
  --teacher-ckpt "$TEACHER_CKPT" \
  --use-ema-teacher \
  "${START_ARGS[@]}" \
  --batch-size "$BATCH_SIZE" \
  --datasets AIST \
  --seq-len 196 \
  --latent-stats-path "$LATENT_STATS" \
  --vae-ckpt "$VAE_CKPT" \
  --vae-type motion_vqvae \
  --aist-crop-mode random \
  --aist-clip-repeat 64 \
  --cond-drop-prob 0.0 \
  --cond-frame-drop-prob 0.0 \
  --cond-frame-drop-start-epoch 999999 \
  --cfg-start-epoch 0 \
  --ema \
  --ema-decay 0.99 \
  --lambda-cond 0.0 \
  --solver-cond-start-epoch 999999 \
  --solver-smooth-start-epoch 999999 \
  --lambda-acc 0.0 \
  --lambda-jerk 0.0 \
  --smooth-every 999999 \
  --eval-guidance-scale 2.5 \
  --eval-steps 1,2,4,8,16 \
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
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --wandb-project FlowMimic \
  --wandb-group VQFlow-Reflow \
  --wandb-name "$RUN_NAME" \
  --wandb-id "$WANDB_ID" \
  --wandb-resume "$WANDB_RESUME" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-tags aist,vqvae,zq16,reflow1,cfg-aware,guidance2p5,scale-anchors,round0-ema-teacher,round1-ema-init,velocity-only,true-null,sparse-patterns,joint-drop5,no-frame-mask,no-flow-regularization,ema099,test-boundary-gap-k7

status=$?
printf '[%s] %s exited with status %s\n' "$(date -Is)" "$RUN_NAME" "$status"
exit "$status"
