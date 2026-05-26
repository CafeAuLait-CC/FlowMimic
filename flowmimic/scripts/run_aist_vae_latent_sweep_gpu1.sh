#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-flowmimic-310}"
GPU_ID="${GPU_ID:-1}"
SEQ_LEN="${SEQ_LEN:-196}"
DIMS="${DIMS:-1 2 4 8}"
EPOCHS="${EPOCHS:-160}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RETRY_BATCH_SIZE="${RETRY_BATCH_SIZE:-224}"
CLIP_REPEAT="${CLIP_REPEAT:-16}"
VAL_CLIP_REPEAT="${VAL_CLIP_REPEAT:-4}"
VAL_EVERY="${VAL_EVERY:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-70}"
STATS_PATH="${STATS_PATH:-prepared/flowmimic_aist_mean_std_259_train.npz}"
BASE_DIR="${BASE_DIR:-checkpoints/vae/aist_len196_latent_sweep_gpu1}"
LOG_DIR="${LOG_DIR:-training_logs/aist_vae_latent_sweep_gpu1}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_DIR}/summary.csv}"

mkdir -p "${BASE_DIR}" "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_DIR}/sweep.log"
}

best_val() {
  local ckpt="$1"
  conda run --no-capture-output -n "${CONDA_ENV}" python -c \
    "import torch; s=torch.load('${ckpt}', map_location='cpu'); print(s.get('best_val', 'nan'))"
}

train_one() {
  local latent_len="$1"
  local batch_size="$2"
  local run_dir="${BASE_DIR}/latent${latent_len}_b${batch_size}"
  local run_log="${LOG_DIR}/latent${latent_len}_b${batch_size}.log"
  mkdir -p "${run_dir}"
  log "Starting latent_len=${latent_len}, batch=${batch_size}, run_dir=${run_dir}"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python flowmimic/scripts/train_vae.py \
      --datasets AIST \
      --seq-len "${SEQ_LEN}" \
      --latent-len "${latent_len}" \
      --batch-size "${batch_size}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --stats-path "${STATS_PATH}" \
      --aist-crop-mode random \
      --aist-clip-repeat "${CLIP_REPEAT}" \
      --aist-val-crop-mode uniform \
      --aist-val-clip-repeat "${VAL_CLIP_REPEAT}" \
      --w-contact 1.0 \
      --val-every-epochs "${VAL_EVERY}" \
      --early-stop-patience "${EARLY_STOP_PATIENCE}" \
      --early-stop-min-epochs "${EARLY_STOP_MIN_EPOCHS}" \
      --checkpoint-dir "${run_dir}" \
      --wandb-project FlowMimic \
      --wandb-group flowmimic-aist-vae-latent-sweep \
      --wandb-name "aist-vae-latent${latent_len}-gpu${GPU_ID}-$(date +%y%m%d-%H%M%S)" \
      --wandb-tags "aist,vae,latent${latent_len},gpu${GPU_ID},uniform-val,randomclips" \
      2>&1 | tee -a "${run_log}"
  local ckpt="${run_dir}/motion_vae_best.pt"
  local val="nan"
  if [[ -f "${ckpt}" ]]; then
    val="$(best_val "${ckpt}")"
  else
    log "No best checkpoint produced for latent_len=${latent_len}, batch=${batch_size}"
    return 1
  fi
  printf '%s,%s,%s,%s,%s\n' "$(date -Is)" "${latent_len}" "${batch_size}" "${val}" "${run_dir}" >> "${SUMMARY_CSV}"
  log "Finished latent_len=${latent_len}, batch=${batch_size}, best_val=${val}"
}

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  printf 'timestamp,latent_len,batch_size,best_val,run_dir\n' > "${SUMMARY_CSV}"
fi

log "GPU${GPU_ID} latent sweep started: dims=${DIMS}; seq_len=${SEQ_LEN}; epochs=${EPOCHS}; lr=${LR}; train_clip_repeat=${CLIP_REPEAT}; val_clip_repeat=${VAL_CLIP_REPEAT}"

for latent_len in ${DIMS}; do
  if ! train_one "${latent_len}" "${BATCH_SIZE}"; then
    log "latent_len=${latent_len}, batch=${BATCH_SIZE} failed; retrying with batch=${RETRY_BATCH_SIZE}"
    train_one "${latent_len}" "${RETRY_BATCH_SIZE}"
  fi
done

log "Sweep finished. Summary: ${SUMMARY_CSV}"
