#!/usr/bin/env bash
set -euo pipefail

# Exploratory all-SQG layerwise refinement of the completed uniform-alpha
# panel.  Selection may mix alpha by layer but may never retain MCG.  The
# frozen mapping is then evaluated on the document holdout before any combined
# artifact is constructed.

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test
WORKSPACE=/home/brandonmusic/KLC_SANDBOXES
CAPTURE_ROOT=/home/brandonmusic/KLC_CAPTURE_RUNS/contig-late-capture-r1/contig-late-r1
BF16_ROOT="$PROJECT_ROOT/bf16_contiguous_late"
PERMUTATION_ROOT="$WORKSPACE/fresh-sqg-contig-late-a025-r1"
SELECTION_SCORES="$PROJECT_ROOT/results/contiguous_late_h13_blend_selection_r1.json"
LAYERWISE_SELECTION="$PROJECT_ROOT/results/contiguous_late_h13_layerwise_selection_r1.json"
HOLDOUT_SCORES="$PROJECT_ROOT/results/contiguous_late_h13_blend_holdout_full_r1.json"
LAYERWISE_HOLDOUT="$PROJECT_ROOT/results/contiguous_late_h13_layerwise_holdout_r1.json"
RECEIPT="$PROJECT_ROOT/results/contiguous_late_h13_layerwise_followup_receipt_r1.json"
OUTPUT_ROOT="$WORKSPACE/fresh-sqg-contig-late-layerwise-r1"

declare -A ROOTS=(
  [sqg_a000]="$WORKSPACE/fresh-sqg-contig-late-alpha000-r1"
  [sqg_a025]="$WORKSPACE/fresh-sqg-contig-late-final-a025-r1"
  [sqg_a050]="$WORKSPACE/fresh-sqg-contig-late-alpha050-r1"
  [sqg_a075]="$WORKSPACE/fresh-sqg-contig-late-alpha075-r1"
  [sqg_a100]="$WORKSPACE/fresh-sqg-contig-late-alpha100-r1"
)

[[ -f "$SELECTION_SCORES" && ! -L "$SELECTION_SCORES" ]] || \
  die "uniform panel selection result is absent"
for label in sqg_a000 sqg_a025 sqg_a050 sqg_a075 sqg_a100; do
  root="${ROOTS[$label]}"
  [[ -d "$root" && ! -L "$root" && -f "$root/run_seal.json" ]] || \
    die "sealed alpha root is absent: $label=$root"
done
for output in "$LAYERWISE_SELECTION" "$HOLDOUT_SCORES" \
  "$LAYERWISE_HOLDOUT" "$RECEIPT"; do
  [[ ! -e "$output" && ! -L "$output" ]] || die "output exists: $output"
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || \
  die "combined output root exists: $OUTPUT_ROOT"

python3 "$PROJECT_ROOT/scripts/select_layerwise_h13_blend.py" \
  --input "$SELECTION_SCORES" --baseline-label mcg \
  --require-all-nonbaseline --output "$LAYERWISE_SELECTION"

SQG_SCORE_LAYERS=74,75,76,77 \
SQG_SCORE_MCG_BASELINE_LABEL=mcg \
FRESH_SQG_BF16_ROOT="$BF16_ROOT" \
FRESH_SQG_CAPTURE_ROOT="$CAPTURE_ROOT" \
FRESH_SQG_PERMUTATION_ROOT="$PERMUTATION_ROOT" \
  bash "$PROJECT_ROOT/scripts/run_signed_top8_blend_score.sh" \
    holdout "$HOLDOUT_SCORES" mcg \
    "sqg_a000=${ROOTS[sqg_a000]}" \
    "sqg_a025=${ROOTS[sqg_a025]}" \
    "sqg_a050=${ROOTS[sqg_a050]}" \
    "sqg_a075=${ROOTS[sqg_a075]}" \
    "sqg_a100=${ROOTS[sqg_a100]}"

python3 "$PROJECT_ROOT/scripts/evaluate_layerwise_h13_mapping.py" \
  --scores "$HOLDOUT_SCORES" --selection "$LAYERWISE_SELECTION" \
  --baseline-label mcg --output "$LAYERWISE_HOLDOUT"

selection_passed="$(jq -r '.status == "tail_and_mean_constraints_passed"' \
  "$LAYERWISE_SELECTION")"
holdout_passed="$(jq -r '.passes_all_hard_constraints' "$LAYERWISE_HOLDOUT")"
built=false
if [[ "$selection_passed" == true && "$holdout_passed" == true ]]; then
  python3 "$PROJECT_ROOT/scripts/build_layerwise_h13_artifact.py" \
    --selection "$LAYERWISE_SELECTION" \
    --candidate "sqg_a000=${ROOTS[sqg_a000]}" \
    --candidate "sqg_a025=${ROOTS[sqg_a025]}" \
    --candidate "sqg_a050=${ROOTS[sqg_a050]}" \
    --candidate "sqg_a075=${ROOTS[sqg_a075]}" \
    --candidate "sqg_a100=${ROOTS[sqg_a100]}" \
    --output-root "$OUTPUT_ROOT"
  built=true
fi

jq -n \
  --arg selection "$LAYERWISE_SELECTION" \
  --arg selection_sha256 "$(sha256sum "$LAYERWISE_SELECTION" | awk '{print $1}')" \
  --arg holdout_scores "$HOLDOUT_SCORES" \
  --arg holdout_scores_sha256 "$(sha256sum "$HOLDOUT_SCORES" | awk '{print $1}')" \
  --arg holdout "$LAYERWISE_HOLDOUT" \
  --arg holdout_sha256 "$(sha256sum "$LAYERWISE_HOLDOUT" | awk '{print $1}')" \
  --argjson selection_passed "$selection_passed" \
  --argjson holdout_passed "$holdout_passed" \
  --argjson artifact_built "$built" \
  --arg artifact_root "$OUTPUT_ROOT" \
  '{
    schema:"glm52-contiguous-late-all-sqg-layerwise-followup-v1",
    selection:{path:$selection,sha256:$selection_sha256,passed:$selection_passed},
    holdout_scores:{path:$holdout_scores,sha256:$holdout_scores_sha256},
    frozen_mapping_holdout:{path:$holdout,sha256:$holdout_sha256,passed:$holdout_passed},
    all_selected_layers_sqg_required:true,
    artifact_built:$artifact_built,
    artifact_root:(if $artifact_built then $artifact_root else null end)
  }' > "$RECEIPT"

printf 'late all-SQG layerwise follow-up complete: selection=%s holdout=%s built=%s\n' \
  "$selection_passed" "$holdout_passed" "$built"
