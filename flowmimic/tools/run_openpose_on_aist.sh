#!/usr/bin/env bash
set -euo pipefail

# ===== configurable =====
OPENPOSE_ROOT="$HOME/hdd/openpose"
VIDEO_DIR="$HOME/hdd/AIST++/Videos"
OUT_DIR="$HOME/hdd/AIST++/Annotations/openpose"
PY_SCRIPT="$HOME/hdd/FlowMimic/flowmimic/tools/json2npy_openpose.py"
# ========================

mkdir -p "$OUT_DIR"
cd "$OPENPOSE_ROOT"

shopt -s nullglob
videos=("$VIDEO_DIR"/*.mp4)
shopt -u nullglob

if [ ${#videos[@]} -eq 0 ]; then
  echo "[ERROR] No videos found."
  exit 1
fi

for v in "${videos[@]}"; do
  base=$(basename "$v")

  # ===== 新增过滤条件 =====
  if [[ ! "$base" =~ _c(01|02|08|09)_ ]]; then
    echo "[SKIP] channel filter: $base"
    continue
  fi
  # ======================

  name="${base%.*}"
  json_dir="$OUT_DIR/$name"
  npy_path="$OUT_DIR/$name.npy"

  if [ -f "$npy_path" ]; then
    echo "[SKIP] npy exists: $name"
    continue
  fi

  mkdir -p "$json_dir"

  echo "[RUN] OpenPose: $base"

  ./build/examples/openpose/openpose.bin \
    --video "$v" \
    --write_json "$json_dir" \
    --display 0 --render_pose 0 \
    --number_people_max 1 \
    --scale_number 1

  echo "[RUN] JSON → NPY: $name"

  python3 "$PY_SCRIPT" \
    --video_dir "$json_dir" \
    --out "$npy_path"

  echo "[DONE] $name"
done

cd -
