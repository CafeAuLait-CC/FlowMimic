#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash flowmimic/tools/combine_mvh_openpose.sh

MVH_ROOT="$HOME/hdd/MVHumanNet_Data"
OUT_ROOT="./data/MVHumanNet"
PY_SCRIPT="./flowmimic/tools/json2npy_openpose.py"
CAMERAS=("22327091" "22327113" "22327084")
CURRENT_TMP=""
OK='\033[1;38;2;115;218;202m[OK]\033[0m'
WARN='\033[1;38;2;215;215;95m[WARN]\033[0m'
SKIP='\033[1;38;2;125;207;255m[SKIP]\033[0m'

cleanup_on_interrupt() {
  if [ -n "$CURRENT_TMP" ] && [ -f "$CURRENT_TMP" ]; then
    rm -f "$CURRENT_TMP"
  fi
  exit 130
}

trap cleanup_on_interrupt INT

for part in MVHumanNet_24_Part_01 MVHumanNet_24_Part_02 MVHumanNet_24_Part_03 MVHumanNet_24_Part_04; do
  part_dir="$MVH_ROOT/$part"
  [ -d "$part_dir" ] || continue

  for motion_dir in "$part_dir"/*; do
    [ -d "$motion_dir" ] || continue
    motion_name="$(basename "$motion_dir")"

    for cam in "${CAMERAS[@]}"; do
      json_dir="$motion_dir/openpose/$cam"
      [ -d "$json_dir" ] || continue

      out_dir="$OUT_ROOT/$part/$motion_name"
      out_path="$out_dir/${cam}_2d_body25.npy"
      if [ -f "$out_path" ]; then
        echo -e "${SKIP} exists: $out_path"
        continue
      fi

      mkdir -p "$out_dir"
      tmp_path="${out_path}.partial.npy"
      CURRENT_TMP="$tmp_path"
      python3 "$PY_SCRIPT" --video_dir "$json_dir" --out "$tmp_path"
      mv "$tmp_path" "$out_path"
      CURRENT_TMP=""
      echo -e "${OK} $out_path"
    done
  done
done
