#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="training_logs/vqflow_aist_zq16_round0_mg_ablation_260828"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/queue.out") 2>&1
echo "$$" > "$LOG_DIR/queue.pid"
printf 'pending\n' > "$LOG_DIR/style_only_dual_gpu.state"

printf '[%s] Launching M-only on GPU0\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29682 \
  bash flowmimic/scripts/run_aist_round0_mg_ablation.sh memory_only &
GPU0_PID=$!

printf '[%s] Launching g-only on GPU1\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29683 \
  bash flowmimic/scripts/run_aist_round0_mg_ablation.sh global_only &
GPU1_PID=$!

printf '[%s] Queue children: gpu0=%s gpu1=%s\n' "$(date -Is)" "$GPU0_PID" "$GPU1_PID"
wait "$GPU0_PID"
wait "$GPU1_PID"
printf '[%s] M-only and g-only completed; launching no-pose on both GPUs\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=29684 \
  bash flowmimic/scripts/run_aist_round0_mg_ablation.sh style_only
printf '[%s] M/g ablation queue completed\n' "$(date -Is)"
