#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_NAME="${RUN_NAME:-vqflow_aist_zq16_reflow1_cfgbranch_w002_260821}"
export WANDB_ID="${WANDB_ID:-r1cfgbp02}"
export MASTER_PORT="${MASTER_PORT:-29823}"
export CURRICULUM="reflow_round1_cfg_branch_preserve_w002"
export BRANCH_TAG="w002"

exec bash "$ROOT_DIR/flowmimic/scripts/run_aist_reflow_round1_cfg_branch_preserve.sh"
