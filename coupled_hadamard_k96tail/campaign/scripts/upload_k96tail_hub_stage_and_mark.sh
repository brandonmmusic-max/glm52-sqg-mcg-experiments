#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'K96 Hub stage upload: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 2 ]] || die "usage: $0 STAGE_DIR COMPLETE_MARKER"

stage=$1
complete_marker=$2
repo=brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail

[[ -d "$stage" && ! -L "$stage" ]] || die "unsafe stage directory: $stage"
[[ ! -L "$complete_marker" ]] || die "unsafe completion marker: $complete_marker"
mkdir -p "$(dirname "$complete_marker")"

export HF_HOME=${HF_HOME:-/home/brandonmusic/.cache/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HF_HOME/token}
export HF_XET_HIGH_PERFORMANCE=1

hf upload "$repo" "$stage" --repo-type model
touch "$complete_marker"
printf 'K96 Hub stage upload complete: stage=%s marker=%s\n' \
  "$stage" "$complete_marker"
