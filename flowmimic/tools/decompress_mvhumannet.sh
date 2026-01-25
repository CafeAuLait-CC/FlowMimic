#!/usr/bin/env bash
set -euo pipefail

root="./data/MVHumanNet"
jobs="${JOBS:-}"
use_pigz="0"

if [[ ! -d "$root" ]]; then
  echo "MVHumanNet root not found: $root" >&2
  exit 1
fi

if [[ -z "$jobs" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
  else
    jobs="4"
  fi
fi

if command -v pigz >/dev/null 2>&1; then
  use_pigz="1"
fi
export use_pigz

cleanup_on_interrupt() {
  echo "[cleanup] removing incomplete .partial folders"
  find "$root" -type d -name '*.partial' -prune -exec rm -rf {} + >/dev/null 2>&1 || true
  exit 130
}

trap cleanup_on_interrupt INT

shopt -s nullglob

extract_one() {
  local archive="$1"
  local out_dir="$2"
  local base_name dest_dir tmp_root first_entry
  local tar_compress=()

  if [[ "${use_pigz}" == "1" ]]; then
    tar_compress=(--use-compress-program=pigz)
  fi

  base_name="$(basename "$archive" .tar.gz)"
  dest_dir="$out_dir/$base_name"

  if [[ -d "$dest_dir" ]]; then
    echo "[skip] $archive -> $dest_dir exists"
    return 0
  fi

  tmp_root="${dest_dir}.partial"
  rm -rf "$tmp_root"
  mkdir -p "$tmp_root"

  first_entry="$(tar -tzf "$archive" | head -n 1 || true)"
  if [[ -n "$first_entry" && "${first_entry%%/*}" == "$base_name" ]]; then
    echo "[extract] $archive (contains $base_name/) -> $out_dir"
    tar -xzf "$archive" "${tar_compress[@]}" -C "$tmp_root"
    if [[ -d "$tmp_root/$base_name" ]]; then
      mv "$tmp_root/$base_name" "$dest_dir"
    else
      mv "$tmp_root" "$dest_dir"
    fi
    rm -rf "$tmp_root"
  else
    echo "[extract] $archive -> $dest_dir"
    tar -xzf "$archive" "${tar_compress[@]}" -C "$tmp_root"
    mv "$tmp_root" "$dest_dir"
  fi
}
export -f extract_one

for part_dir in "$root"/*_Part_*/; do
  [[ -d "$part_dir" ]] || continue

  part_name="$(basename "$part_dir")"
  out_dir="$root/${part_name%/}_decompressed"
  mkdir -p "$out_dir"

  find "$part_dir" -maxdepth 1 -type f -name '*.tar.gz' -print0 \
    | xargs -0 -P "$jobs" -I {} bash -c 'extract_one "$0" "$1"' {} "$out_dir"
done

shopt -u nullglob
