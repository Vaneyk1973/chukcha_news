#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <destination-project-directory>"
  echo "Example: $0 user@host:/home/user/chukcha_news/"
  exit 2
fi

destination="$1"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

assets=(audio data external models reports)
existing=()
for asset in "${assets[@]}"; do
  if [[ -e "$asset" ]]; then
    existing+=("$asset")
  fi
done

rsync \
  --archive \
  --human-readable \
  --partial \
  --info=progress2 \
  --exclude='*.zip' \
  --exclude='.DS_Store' \
  "${existing[@]}" \
  "$destination"
