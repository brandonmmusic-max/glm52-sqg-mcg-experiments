#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'stage sealed layers: %s\n' "$*" >&2; exit 2; }

[[ $# -eq 3 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ ]] || \
  die "usage: $0 START_LAYER END_LAYER STAGING_ROOT"
start=$1
end=$2
stage=$3
((start >= 3 && end <= 77 && start <= end)) || die "invalid layer range"

layer_root=${LAYER_ROOT:-/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers}
[[ -d "$layer_root" && ! -L "$layer_root" ]] || die "unsafe layer root: $layer_root"
mkdir -p "$stage"
[[ -d "$stage" && ! -L "$stage" ]] || die "unsafe staging root: $stage"

for layer in $(seq "$start" "$end"); do
  padded=$(printf '%03d' "$layer")
  shard=$layer_root/r7-experts-layer-${padded}.safetensors
  manifest=$layer_root/r7-experts-layer-${padded}.json
  quality=$layer_root/r7-experts-layer-${padded}.quality.json
  oracle=$layer_root/runtime-oracle-layer-${padded}.json
  for path in "$shard" "$manifest" "$quality" "$oracle"; do
    [[ -f "$path" && ! -L "$path" ]] || die "layer $layer is not fully sealed: $path"
  done
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-selected-layer-b12x-oracle-v2" and
     .complete == true and .pass == true and .finite == true and
     .nonzero == true and .layer == $layer and
     (if $layer == 3 then
        .bit_census == {"k3": 720, "k4": 48, "total": 768} and
        .bits_per_weight == 3.0625
      else
        .bit_census == {"k3": 672, "k4": 96, "total": 768} and
        .bits_per_weight == 3.125
      end) and
     .production_endpoint == "route_packed_direct_e4m3_w4a8"' \
    "$oracle" >/dev/null || die "layer $layer runtime oracle seal differs"
  for path in "$shard" "$manifest" "$quality" "$oracle"; do
    target=$stage/$(basename "$path")
    if [[ -e "$target" ]]; then
      [[ -f "$target" && ! -L "$target" ]] || die "unsafe staging target: $target"
      [[ $(sha256sum "$path" | cut -d' ' -f1) == $(sha256sum "$target" | cut -d' ' -f1) ]] || \
        die "staged artifact differs: $target"
    else
      ln "$path" "$target"
    fi
  done
done

printf 'staged sealed layer range %s..%s at %s\n' "$start" "$end" "$stage"
