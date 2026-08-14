#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 2 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ ]] || \
  die "usage: $0 START_LAYER END_LAYER"
start_layer=$1
end_layer=$2
((end_layer >= start_layer && start_layer >= 3 && end_layer <= 78)) || \
  die "layer range must lie within 3..78"

wave=$(printf 'wave-%03d-%03d' "$start_layer" "$end_layer")
selected_layers=$(seq -s, "$start_layer" "$end_layer")
run_layers_csv=${RUN_LAYERS:-$selected_layers}
IFS=, read -r -a run_layers <<< "$run_layers_csv"
(( ${#run_layers[@]} >= 1 )) || die "RUN_LAYERS must select at least one layer"
seen=,
for layer in "${run_layers[@]}"; do
  [[ "$layer" =~ ^[0-9]+$ ]] || die "RUN_LAYERS contains a nonnumeric layer"
  ((layer >= start_layer && layer <= end_layer)) || \
    die "RUN_LAYERS layer $layer lies outside $start_layer..$end_layer"
  [[ "$seen" != *",$layer,"* ]] || die "RUN_LAYERS repeats layer $layer"
  seen+="$layer,"
done
PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
CANDIDATE_ROOT=${CANDIDATE_ROOT:-/media/brandonmusic/nvme1n1p3/glm52-coupled-3p0625-no-shortcut-work/$wave}
LAYER_ROOT=${LAYER_ROOT:-/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-3.0625bpw-no-shortcut-layers}
RUN_RUNTIME_ORACLES=${RUN_RUNTIME_ORACLES:-1}
[[ "$RUN_RUNTIME_ORACLES" == 0 || "$RUN_RUNTIME_ORACLES" == 1 ]] || \
  die "RUN_RUNTIME_ORACLES must be 0 or 1"
mkdir -p "$LAYER_ROOT"
cd "$PROJECT_ROOT"

# Refuse the whole wave before starting any writers if one layer is ambiguous.
for layer in "${run_layers[@]}"; do
  padded=$(printf '%03d' "$layer")
  shard="$LAYER_ROOT/r7-experts-layer-${padded}.safetensors"
  manifest="$LAYER_ROOT/r7-experts-layer-${padded}.json"
  quality="$LAYER_ROOT/r7-experts-layer-${padded}.quality.json"
  if [[ -e "$shard" || -e "$manifest" || -e "$quality" ]]; then
    die "refusing ambiguous existing layer output: $padded"
  fi
done

pids=()
labels=()
for layer in "${run_layers[@]}"; do
  padded=$(printf '%03d' "$layer")
  shard="$LAYER_ROOT/r7-experts-layer-${padded}.safetensors"
  quality="$LAYER_ROOT/r7-experts-layer-${padded}.quality.json"
  (
    python3 scripts/materialize_selected_coupled_layer.py \
      --candidate-root "$CANDIDATE_ROOT" --layer "$layer" --output "$shard" \
      >"$LAYER_ROOT/materialize-layer-${padded}.log"
    python3 scripts/archive_coupled_layer_quality.py \
      --candidate-root "$CANDIDATE_ROOT" --layer "$layer" --output "$quality" \
      >"$LAYER_ROOT/quality-layer-${padded}.log"
    if [[ "$RUN_RUNTIME_ORACLES" == 1 ]]; then
      gpu=$(((layer - start_layer) % 4))
      LAYER_ROOT=$LAYER_ROOT GPU=$gpu \
        "$PROJECT_ROOT/scripts/run_validate_coupled_runtime_layer.sh" "$layer" \
        >"$LAYER_ROOT/runtime-oracle-layer-${padded}.log" 2>&1
    fi
  ) &
  pids+=("$!")
  labels+=("$layer")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'materialization/oracle worker failed: layer=%s\n' "${labels[$index]}" >&2
    failed=1
  fi
done
((failed == 0)) || exit 1

printf 'coupled wave materialization complete: layers=%s..%s root=%s\n' \
  "$start_layer" "$end_layer" "$LAYER_ROOT"
