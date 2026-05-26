#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
CONDA_EXE="${CONDA_EXE:-/mnt/data5_hdd/alex/miniconda3/bin/conda}"
BATCH_SIZE="${BATCH_SIZE:-96}"
STAMP="$(date +%y%m%d-%H%M%S)"
RUN_DIR="checkpoints/vae/aist_mvh_len196_latent1_query_gpu1_${STAMP}"
LOG_DIR="training_logs"
LOG_FILE="${LOG_DIR}/aist_mvh_vae_latent1_query_gpu1_${STAMP}.out"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

{
  printf '[%s] Starting AIST+MVH compact query-token VAE\n' "$(date -Is)"
  printf '[%s] GPU_ID=%s\n' "$(date -Is)" "${GPU_ID}"
  printf '[%s] RUN_DIR=%s\n' "$(date -Is)" "${RUN_DIR}"
  printf '[%s] LOG_FILE=%s\n' "$(date -Is)" "${LOG_FILE}"
} | tee -a "${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${CONDA_EXE}" run --no-capture-output -n flowmimic-310 \
  python flowmimic/scripts/train_vae.py \
    --datasets AIST,MVH \
    --val-datasets AIST \
    --seq-len 196 \
    --latent-len 1 \
    --latent-token-mode query \
    --aist-crop-mode random \
    --aist-clip-repeat 32 \
    --aist-val-crop-mode uniform \
    --aist-val-clip-repeat 4 \
    --ratio-aist 1 \
    --ratio-mvh 1 \
    --stats-path data/mean_std_263_train.npz \
    --checkpoint-dir "${RUN_DIR}" \
    --epochs 500 \
    --batch-size "${BATCH_SIZE}" \
    --lr 2e-4 \
    --val-every-epochs 10 \
    --early-stop-patience 8 \
    --early-stop-min-epochs 80 \
    --w-contact 1.0 \
    --wandb-project FlowMimic \
    --wandb-entity cvi-aris \
    --wandb-group FlowMimic-VAE-AIST-MVH \
    --wandb-name "aist-mvh-vae-latent1-query-gpu${GPU_ID}-${STAMP}" \
    --wandb-tags "aist,mvh,vae,latent1,query-token,union-stats,aist-val" \
  2>&1 | tee -a "${LOG_FILE}"

printf '[%s] Finished AIST+MVH compact query-token VAE\n' "$(date -Is)" | tee -a "${LOG_FILE}"
