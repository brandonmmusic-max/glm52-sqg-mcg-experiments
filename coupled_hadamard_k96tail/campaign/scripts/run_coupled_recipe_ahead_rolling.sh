#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
INPUT_BASE=/media/brandonmusic/nvme1n1p3/glm52-coupled-wave-inputs
RECIPE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1
STOP_FILE=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/STOP_FULL_COUPLED_K96TAIL_NO_SHORTCUT

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

# The four lane helpers consume the already-staged waves through layer 18.
# Wait for their last wave before extending the rolling window.
while :; do
  complete=1
  for layer in 15 16 17 18; do recipe_passes "$layer" || complete=0; done
  [[ "$complete" == 1 ]] && break
  [[ ! -e "$STOP_FILE" ]] || die "campaign stop file exists: $STOP_FILE"
  sleep 30
done

# Keep exactly the existing four-wave input window: after the authoritative
# campaign validates and removes the wave sixteen layers behind, stage one new
# wave, run its exact recipe, and wait for the next cleanup boundary.  This
# overlaps preparation without allowing saved capture data to fill the NVMe.
for start in $(seq 19 4 75); do
  end=$((start + 3))
  predecessor=$((start - 16))
  wave=$(printf 'wave-%03d-%03d' "$start" "$end")
  predecessor_wave=$(printf 'wave-%03d-%03d' "$predecessor" "$((predecessor + 3))")
  while [[ -e "$INPUT_BASE/$predecessor_wave" ]]; do
    [[ ! -e "$STOP_FILE" ]] || die "campaign stop file exists: $STOP_FILE"
    sleep 30
  done
  [[ ! -e "$STOP_FILE" ]] || die "campaign stop file exists: $STOP_FILE"
  if [[ ! -f "$INPUT_BASE/$wave/capture_view/capture_manifest.json" ]]; then
    WAVE_INPUT_ROOT=$INPUT_BASE/$wave \
      "$PROJECT_ROOT/scripts/download_coupled_wave_inputs.sh" "$start" "$end"
  fi
  layers=$(seq -s, "$start" "$end")
  RUN_LAYERS=$layers WAVE_INPUT_ROOT=$INPUT_BASE/$wave \
    RECIPE_ROOT=$RECIPE_ROOT DETACH=0 \
    "$PROJECT_ROOT/scripts/run_coupled_recipe_wave.sh" "$start" "$end"
  for layer in $(seq "$start" "$end"); do
    recipe_passes "$layer" || die "rolling recipe did not seal layer $layer"
  done
done

printf 'rolling recipe-ahead campaign sealed through layer 78\n'
