#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET_UPDATES=68220
CHECKPOINT="checkpoints/flow/vqflow_aist_zq16_round0_no_pose_260828/flow_round0_last_good.pt"
LOG_DIR="training_logs/vqflow_aist_zq16_round0_mg_ablation_260828"
LOG_PATH="$LOG_DIR/no_pose_handoff.out"
CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_BASE="${CONDA_BASE:-$("$CONDA_EXE" info --base)}"
PYTHON_BIN="${FLOWMIMIC_PYTHON:-$CONDA_BASE/envs/flowmimic-310/bin/python}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1
echo "$$" > "$LOG_DIR/no_pose_handoff.pid"

checkpoint_updates() {
  if [[ ! -s "$CHECKPOINT" ]]; then
    printf '0\n'
    return
  fi
  "$PYTHON_BIN" -c \
    "import torch; s=torch.load('$CHECKPOINT', map_location='cpu', weights_only=False); print(int(s.get('optimizer_updates', 0)))"
}

while true; do
  updates="$(checkpoint_updates)"
  if (( updates >= TARGET_UPDATES )); then
    printf '[%s] No-pose control is complete at update %s\n' "$(date -Is)" "$updates"
    exit 0
  fi

  if pgrep -f 'flowmimic/scripts/train_flow.py .*--pose-conditioning (memory_only|global_only)' >/dev/null; then
    sleep 60
    continue
  fi

  if pgrep -f 'flowmimic/scripts/train_flow.py .*--pose-conditioning style_only' >/dev/null; then
    sleep 60
    continue
  fi

  # Give the original queue a chance to perform its handoff. The launcher lock
  # prevents a race if both paths reach the start command together.
  sleep 120
  if pgrep -f 'flowmimic/scripts/train_flow.py .*--pose-conditioning style_only' >/dev/null; then
    continue
  fi

  printf '[%s] Starting/resuming no-pose DDP on GPUs 0,1 at update %s\n' \
    "$(date -Is)" "$updates"
  CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=29684 \
    bash flowmimic/scripts/run_aist_round0_mg_ablation.sh style_only || true
  sleep 30
done
