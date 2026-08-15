#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'vast wave uploader: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 5 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ && \
   $3 =~ ^[0-9]+$ && $4 =~ ^[0-9]+$ ]] || \
  die "usage: $0 START END GPU_BASE GPU_SPAN HF_REPO"

start=$1
end=$2
gpu_base=$3
gpu_span=$4
repo=$5
project=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
stage=/workspace/k96-hub-staging/wave-$(printf '%03d-%03d' "$start" "$end")
log=/workspace/k96-logs/campaign-$(printf '%03d-%03d' "$start" "$end").log
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HF_HOME/token}
mkdir -p "$(dirname "$log")" "$stage"

START_WAVE=$start STOP_WAVE=$start PARTIAL_ONLY=1 CLEANUP_VALIDATED_WAVES=0 \
GPU_BASE=$gpu_base GPU_SPAN=$gpu_span CAMPAIGN_LOG=$log \
  "$project/scripts/run_full_coupled_3p0625_campaign.sh"

"$project/scripts/stage_sealed_layers_for_hub.sh" "$start" "$end" "$stage"
cp "$project/hub/k96tail-staging/README.md" "$stage/README.md"
HF_XET_HIGH_PERFORMANCE=1 hf upload-large-folder "$repo" "$stage" \
  --num-workers 4 --no-bars

printf 'wave encoded, sealed, and persisted to Hub: %s..%s repo=%s\n' \
  "$start" "$end" "$repo"
