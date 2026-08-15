#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# -ge 2 ]] || die "usage: $0 WAIT_FOR_LAYER LAYER [LAYER ...]"

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
INPUT_BASE=/media/brandonmusic/nvme1n1p3/glm52-coupled-wave-inputs
RECIPE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1
STOP_FILE=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/STOP_FULL_COUPLED_K96TAIL_NO_SHORTCUT
wait_layer=$1
shift

recipe_passes() {
  local layer=$1 padded path
  padded=$(printf '%03d' "$layer")
  path=$RECIPE_ROOT/layer_${padded}/NO_SHORTCUT_COUPLED_RECIPE.json
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-updated-qsrt-coupled-no-shortcut-layer-recipe-v1" and
     .complete == true and .layer == $layer and
     .no_b300_owner_speed_rescue == true and
     .no_fleet_beta_shortcut == true and
     .source_is_frozen_sqg_checkpoint == true and
     .official_bf16_weight_shards_read == false' \
    "$path" >/dev/null 2>&1
}

wave_inputs_ready() {
  local start=$1 end=$2 wave input_root layer padded
  wave=$(printf 'wave-%03d-%03d' "$start" "$end")
  input_root=$INPUT_BASE/$wave
  [[ -d "$input_root" && ! -L "$input_root" ]] || return 1
  for path in \
    "$input_root/derived/wave_preflights/$wave/preflight.json" \
    "$input_root/derived/wave_preflights/$wave/bit-contract.json" \
    "$input_root/derived/wave_preflights/$wave/source-seal.json" \
    "$input_root/derived/wave_preflights/$wave/wave-bf16-shard-manifest.json" \
    "$input_root/capture_view/capture_manifest.json"; do
    [[ -f "$path" && ! -L "$path" ]] || return 1
  done
  for layer in $(seq "$start" "$end"); do
    padded=$(printf '%03d' "$layer")
    [[ -f "$input_root/capture_view/layer_${padded}/hidden.bf16.bin" ]] || return 1
  done
}

[[ "$wait_layer" =~ ^[0-9]+$ ]] || die "WAIT_FOR_LAYER must be numeric"
while ! recipe_passes "$wait_layer"; do
  [[ ! -e "$STOP_FILE" ]] || die "campaign stop file exists: $STOP_FILE"
  sleep 15
done

for layer in "$@"; do
  [[ "$layer" =~ ^[0-9]+$ ]] || die "layer must be numeric: $layer"
  ((layer >= 4 && layer <= 78)) || die "layer lies outside 4..78: $layer"
  if recipe_passes "$layer"; then
    printf 'recipe-ahead layer %s already sealed\n' "$layer"
    continue
  fi
  start=$((3 + ((layer - 3) / 4) * 4))
  end=$((start + 3))
  wave=$(printf 'wave-%03d-%03d' "$start" "$end")
  input_root=$INPUT_BASE/$wave
  while ! wave_inputs_ready "$start" "$end"; do
    [[ ! -e "$STOP_FILE" ]] || die "campaign stop file exists: $STOP_FILE"
    sleep 15
  done
  [[ ! -e "$STOP_FILE" ]] || die "campaign stop file exists: $STOP_FILE"
  printf 'recipe-ahead starting layer %s from %s\n' "$layer" "$wave"
  RUN_LAYERS=$layer WAVE_INPUT_ROOT=$input_root RECIPE_ROOT=$RECIPE_ROOT DETACH=0 \
    "$PROJECT_ROOT/scripts/run_coupled_recipe_wave.sh" "$start" "$end"
  recipe_passes "$layer" || die "recipe-ahead layer did not seal: $layer"
done

printf 'recipe-ahead lane complete: wait_layer=%s layers=%s\n' \
  "$wait_layer" "$*"
