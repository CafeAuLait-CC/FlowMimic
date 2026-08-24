#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_NAME="${RUN_NAME:-vqflow_aist_zq16_reflow1_cfgdeploy_w000_260822}"
export WANDB_ID="${WANDB_ID:-r1cfgdo0}"
export CURRICULUM="reflow_round1_cfg_deploy_only"
export BRANCH_TAG="w000"
export WANDB_TAGS="${WANDB_TAGS:-aist,vqvae,zq16,reflow1,cfg-aware,guidance2p5,deployment-only,no-branch-preservation,round0-ema-teacher,cfg-aware-ema-resume,velocity-only,true-null,sparse-patterns,joint-drop5,no-frame-mask,no-flow-regularization,ema099,test-boundary-gap-k7}"

exec bash "$ROOT_DIR/flowmimic/scripts/run_aist_reflow_round1_cfg_branch_preserve.sh"
