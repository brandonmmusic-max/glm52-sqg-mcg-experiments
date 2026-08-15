#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'K96 merge/finalize: %s\n' "$*" >&2; exit 2; }
log() { printf 'K96 merge/finalize: %s %s\n' "$(date --iso-8601=seconds)" "$*"; }

project=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
acceptance=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812
repo=brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail
layer_root=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers
merge_stage=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-hub-merge
local_stage=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-hub-staging/local-047-050
local_upload_unit=glm52-k96tail-hub-local-047-050.service
recipe=/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1
scores=/media/brandonmusic/nvme1n1p3/glm52-coupled-tail-rate-scores-no-shortcut-v5
allocations=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-allocations-v1
work=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-work-v1
evidence=$project/evidence/full-coupled-k96tail-no-shortcut
reproduction=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-reproduction
stop_file=$acceptance/STOP_FULL_COUPLED_K96TAIL_NO_SHORTCUT
local_unit=glm52-full-coupled-k96tail-no-shortcut-goal019ffa7c.service
state_root=$acceptance/RESULTS/k96tail-distributed-merge-state
hf_python=/home/brandonmusic/.hf-cli/venv/bin/python

export HF_HOME=${HF_HOME:-/home/brandonmusic/.cache/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HF_HOME/token}
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$state_root" "$merge_stage" "$reproduction/remote-campaign-logs"
exec 9>"$state_root/finalizer.lock"
flock -n 9 || die "another merge finalizer already holds the lock"

oracle_passes() {
  local layer=$1 padded path
  padded=$(printf '%03d' "$layer")
  path=$layer_root/runtime-oracle-layer-${padded}.json
  [[ -f "$path" && ! -L "$path" ]] || return 1
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-selected-layer-b12x-oracle-v2" and
     .complete == true and .pass == true and .finite == true and
     .nonzero == true and .layer == $layer and
     .bit_census == {"k3": 672, "k4": 96, "total": 768} and
     .bits_per_weight == 3.125 and
     .production_endpoint == "route_packed_direct_e4m3_w4a8"' \
    "$path" >/dev/null 2>&1
}

log "waiting for the local 47..50 wave to seal"
while :; do
  local_ready=1
  for layer in $(seq 47 50); do
    if ! oracle_passes "$layer"; then local_ready=0; break; fi
  done
  ((local_ready == 1)) && break
  sleep 60
done

if [[ ! -f "$state_root/local-047-050-upload.complete" ]]; then
  log "staging sealed local layers 47..50"
  sudo -n env LAYER_ROOT="$layer_root" \
    "$project/scripts/stage_sealed_layers_for_hub.sh" 47 50 "$local_stage"
  sudo -n cp "$project/hub/k96tail-staging/README.md" "$local_stage/README.md"
  sudo -n chmod a+r "$local_stage"/*
  sudo -n chown brandonmusic:brandonmusic "$local_stage" "$local_stage/README.md"
  if ! systemctl --user is-active --quiet "$local_upload_unit"; then
    systemctl --user reset-failed "$local_upload_unit" >/dev/null 2>&1 || true
    systemd-run --user --collect --unit="$local_upload_unit" \
      --description="Upload sealed local K96-tail layers 47 through 50" \
      --property=KillMode=control-group --property=TimeoutStopSec=30 \
      --property=Restart=on-failure --property=RestartSec=60 \
      /usr/bin/bash "$project/scripts/upload_k96tail_hub_stage_and_mark.sh" \
      "$local_stage" "$state_root/local-047-050-upload.complete"
  fi
  log "local layers 47..50 are uploading asynchronously via $local_upload_unit"
fi

log "waiting for all remote layer seals and reproduction bundles on the public Hub"
until "$hf_python" "$project/scripts/check_k96tail_hub_ready.py"; do sleep 60; done

includes=()
for layer in $(seq 51 77); do
  padded=$(printf '%03d' "$layer")
  includes+=(
    --include "r7-experts-layer-${padded}.safetensors"
    --include "r7-experts-layer-${padded}.json"
    --include "r7-experts-layer-${padded}.quality.json"
    --include "runtime-oracle-layer-${padded}.json"
  )
done
includes+=(--include 'reproduction/*')
log "downloading the complete remote layer/evidence set"
hf download "$repo" --local-dir "$merge_stage" --max-workers 8 "${includes[@]}"

for layer in $(seq 51 77); do
  padded=$(printf '%03d' "$layer")
  shard=$merge_stage/r7-experts-layer-${padded}.safetensors
  manifest=$merge_stage/r7-experts-layer-${padded}.json
  quality=$merge_stage/r7-experts-layer-${padded}.quality.json
  oracle=$merge_stage/runtime-oracle-layer-${padded}.json
  for path in "$shard" "$manifest" "$quality" "$oracle"; do
    [[ -f "$path" && ! -L "$path" ]] || die "downloaded layer $layer is incomplete: $path"
  done
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-selected-layer-runtime-v3" and
     .complete == true and .layer == $layer and
     .bit_census == {"k3": 672, "k4": 96, "total": 768} and
     .bits_per_weight == 3.125 and
     .final_profile_binding.no_b300_owner_speed_rescue == true and
     (.shard_sha256 | type == "string")' "$manifest" >/dev/null || \
    die "downloaded layer manifest differs: $layer"
  [[ $(sha256sum "$shard" | cut -d' ' -f1) == $(jq -r .shard_sha256 "$manifest") ]] || \
    die "downloaded shard hash differs: $layer"
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-layer-quality-tails-v2" and
     .complete == true and .layer == $layer and
     .allocation_binding.histogram == {"3": 672, "4": 96} and
     .allocation_binding.bpw == 3.125' "$quality" >/dev/null || \
    die "downloaded quality seal differs: $layer"
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-selected-layer-b12x-oracle-v2" and
     .complete == true and .pass == true and .finite == true and
     .nonzero == true and .layer == $layer and
     .bit_census == {"k3": 672, "k4": 96, "total": 768} and
     .bits_per_weight == 3.125 and
     .production_endpoint == "route_packed_direct_e4m3_w4a8"' "$oracle" >/dev/null || \
    die "downloaded runtime oracle differs: $layer"
  for path in "$shard" "$manifest" "$quality" "$oracle"; do
    target=$layer_root/$(basename "$path")
    if [[ -e "$target" ]]; then
      [[ -f "$target" && ! -L "$target" ]] || die "unsafe existing layer target: $target"
      [[ $(sha256sum "$target" | cut -d' ' -f1) == $(sha256sum "$path" | cut -d' ' -f1) ]] || \
        die "existing layer target differs: $target"
    else
      ln "$path" "$target" 2>/dev/null || cp --reflink=auto "$path" "$target"
    fi
  done
done

safe_extract() {
  local archive=$1 destination=$2
  tar -tzf "$archive" | awk '
    /^\// || /(^|\/)\.\.($|\/)/ { bad=1 }
    END { exit bad }
  ' || die "unsafe path in evidence archive: $archive"
  mkdir -p "$destination"
  tar -xzf "$archive" -C "$destination"
}

waves=(051-054 055-058 059-062 063-066 067-070 071-074 075-077)

# Remote evidence is authoritative for layers 51-77.  An older local
# run-ahead controller may have left incomplete recipe/profile directories;
# preserve them for audit, but never overlay a sealed remote archive on top of
# that partial state.
recipe_quarantine=$state_root/pre-remote-recipe-$(date +%Y%m%dT%H%M%S)
for layer in $(seq 51 77); do
  padded=$(printf '%03d' "$layer")
  for path in \
    "$recipe/layer_${padded}" \
    "$recipe/final_profiles/layer_${padded}"; do
    if [[ -e "$path" ]]; then
      [[ -d "$path" && ! -L "$path" ]] || \
        die "unsafe pre-existing remote-owned recipe path: $path"
      relative=${path#"$recipe"/}
      mkdir -p "$recipe_quarantine/$(dirname "$relative")"
      mv -- "$path" "$recipe_quarantine/$relative"
    fi
  done
done

for wave in "${waves[@]}"; do
  remote=$merge_stage/reproduction/wave-$wave
  (cd "$remote" && sha256sum -c SHA256SUMS)
  safe_extract "$remote/recipe-and-profiles.tgz" "$recipe"
  safe_extract "$remote/tail-scores.tgz" "$scores"
  safe_extract "$remote/allocations.tgz" "$allocations"
  safe_extract "$remote/parity-proofs.tgz" "$evidence"
  if [[ -f "$remote/candidate-metadata.tgz" ]]; then
    safe_extract "$remote/candidate-metadata.tgz" "$work"
  fi
  cp "$remote/campaign.log" "$reproduction/remote-campaign-logs/wave-$wave.log"
done

for layer in $(seq 51 77); do oracle_passes "$layer" || die "merged oracle failed: $layer"; done
touch "$state_root/remote-051-077-merge.complete"

[[ -f "$stop_file" ]] || die "expected local boundary stop marker is absent"
rm -- "$stop_file"
log "all layers/evidence merged; launching full validation, assembly, and KLD"
CAMPAIGN_LOG="$acceptance/RESULTS/full_coupled_k96tail_distributed_finalize.log" \
START_WAVE=3 STOP_WAVE=75 CLEANUP_VALIDATED_WAVES=1 PARTIAL_ONLY=0 \
  "$project/scripts/run_full_coupled_3p0625_campaign.sh"
touch "$state_root/final-kld.complete"
log "distributed K96-tail merge, assembly, and KLD complete"
