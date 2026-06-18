#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(upload|download|check)$ ]]; then
  echo "Usage: $0 <upload|download|check>"
  exit 2
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is not installed. Install it with: sudo apt install rclone"
  exit 1
fi

action="$1"
remote="${BUCKETRU_REMOTE:-bucketru}"
bucket="${BUCKETRU_BUCKET:-chukcha}"
prefix="${BUCKETRU_PREFIX:-project-assets}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

common_args=(
  --s3-no-check-bucket
  --human-readable
  --progress
  --transfers 4
  --checkers 8
  --exclude '*.zip'
  --exclude '.DS_Store'
)

assets=(audio data external models reports)

case "$action" in
  upload)
    for asset in "${assets[@]}"; do
      if [[ -e "$asset" ]]; then
        rclone copy "$asset/" "${remote}:${bucket}/${prefix}/${asset}/" "${common_args[@]}"
      fi
    done
    ;;
  download)
    for asset in "${assets[@]}"; do
      mkdir -p "$asset"
      rclone copy "${remote}:${bucket}/${prefix}/${asset}/" "$asset/" "${common_args[@]}"
    done
    ;;
  check)
    rclone check "audio/" "${remote}:${bucket}/${prefix}/audio/" \
      --s3-no-check-bucket \
      --one-way \
      --exclude '*.zip' \
      --exclude '.DS_Store'
    ;;
esac
