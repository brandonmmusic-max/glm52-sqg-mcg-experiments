#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'Vast K96 oracle recovery: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 3 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ && $3 =~ ^[0-9]+$ ]] || \
  die "usage: $0 START END GPU_BASE"

start=$1
end=$2
gpu_base=$3
((start >= 3 && end <= 77 && start <= end && end - start <= 7)) || \
  die "invalid recovery range: $start..$end"
((gpu_base >= 0 && gpu_base <= 7)) || die "GPU_BASE must lie in 0..7"

project=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
layer_root=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers
export LAYER_ROOT=$layer_root

for layer in $(seq "$start" "$end"); do
  padded=$(printf '%03d' "$layer")
  result=$layer_root/runtime-oracle-layer-${padded}.json
  if [[ -f "$result" && ! -L "$result" ]]; then
    printf 'runtime oracle already present: layer=%s path=%s\n' "$layer" "$result"
    continue
  fi
  printf 'starting sequential runtime oracle: layer=%s gpu=%s\n' "$layer" "$gpu_base"
  GPU=$gpu_base timeout --signal=TERM --kill-after=30s 10m \
    "$project/scripts/run_validate_coupled_runtime_layer_vast_native.sh" "$layer"
  [[ -f "$result" && ! -L "$result" ]] || \
    die "runtime oracle did not produce a sealed result for layer $layer"
done

printf 'sequential runtime oracle recovery complete: layers=%s..%s\n' "$start" "$end"
