#!/usr/bin/env bash
set -euo pipefail

# Full late-block alpha panel.  Each arm re-encodes all 3,072 tensors and
# rebuilds expert-private candidate-conditioned H2; no MCG bytes or stale down
# Hessians are retained.  Selection uses MCG as the fixed functional baseline.

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test
WORKSPACE=/home/brandonmusic/KLC_SANDBOXES
PREPARATION_ROOT="$WORKSPACE/fresh-sqg-contig-late-a025-r1"
CAPTURE_ROOT=/home/brandonmusic/KLC_CAPTURE_RUNS/contig-late-capture-r1/contig-late-r1
BF16_ROOT="$PROJECT_ROOT/bf16_contiguous_late"
ALPHA025_ROOT="$WORKSPACE/fresh-sqg-contig-late-final-a025-r1"
SELECTION_JSON="$PROJECT_ROOT/results/contiguous_late_h13_blend_selection_r1.json"
HOLDOUT_JSON="$PROJECT_ROOT/results/contiguous_late_h13_blend_holdout_r1.json"
BIT_SHA256=b70773e4d11fb0495d74f4a6977d1e10a9417907d31efdc3ea5de0b243c32a09

declare -A ROOTS=(
  [sqg_a000]="$WORKSPACE/fresh-sqg-contig-late-alpha000-r1"
  [sqg_a025]="$ALPHA025_ROOT"
  [sqg_a050]="$WORKSPACE/fresh-sqg-contig-late-alpha050-r1"
  [sqg_a075]="$WORKSPACE/fresh-sqg-contig-late-alpha075-r1"
  [sqg_a100]="$WORKSPACE/fresh-sqg-contig-late-alpha100-r1"
)
declare -A VALUES=(
  [sqg_a000]=0
  [sqg_a050]=0.5
  [sqg_a075]=0.75
  [sqg_a100]=1.0
)

for path in "$PROJECT_ROOT" "$PREPARATION_ROOT" "$CAPTURE_ROOT" \
  "$BF16_ROOT" "$ALPHA025_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]] || die "required directory is absent: $path"
done
for label in sqg_a000 sqg_a050 sqg_a075 sqg_a100; do
  root="${ROOTS[$label]}"
  if [[ -f "$root/run_seal.json" && ! -L "$root" ]]; then
    printf 'reusing sealed alpha arm: label=%s root=%s\n' "$label" "$root"
    continue
  fi
  if [[ ! -e "$root" ]]; then
    mkdir "$root"
  fi
  [[ -d "$root" && ! -L "$root" ]] || die "alpha root is not a safe directory: $root"
  printf 'encoding or resuming alpha arm: label=%s root=%s\n' "$label" "$root"
  FRESH_SQG_BF16_ROOT="$BF16_ROOT" \
  FRESH_SQG_CAPTURE_ROOT="$CAPTURE_ROOT" \
  FRESH_SQG_BASELINE_ROOT="$PREPARATION_ROOT" \
  FRESH_SQG_SELECTED_LAYERS=74,75,76,77 \
  FRESH_SQG_PLAN_CONTRACT=/work/evidence/contiguous_document_plan_r1.json \
  FRESH_SQG_BF16_MANIFEST=/work/evidence/contiguous_late_bf16_manifest_r1.json \
  FRESH_SQG_BIT_CONTRACT_SHA256="$BIT_SHA256" \
    bash "$PROJECT_ROOT/scripts/run_h13_blend_four_layers.sh" \
      "$root" "${VALUES[$label]}"
done

score_common=(
  SQG_SCORE_LAYERS=74,75,76,77
  SQG_SCORE_MCG_BASELINE_LABEL=mcg
  FRESH_SQG_BF16_ROOT="$BF16_ROOT"
  FRESH_SQG_CAPTURE_ROOT="$CAPTURE_ROOT"
  FRESH_SQG_PERMUTATION_ROOT="$PREPARATION_ROOT"
)

if [[ ! -e "$SELECTION_JSON" ]]; then
  env "${score_common[@]}" \
    bash "$PROJECT_ROOT/scripts/run_signed_top8_blend_score.sh" \
      selection "$SELECTION_JSON" mcg \
      "sqg_a000=${ROOTS[sqg_a000]}" \
      "sqg_a025=${ROOTS[sqg_a025]}" \
      "sqg_a050=${ROOTS[sqg_a050]}" \
      "sqg_a075=${ROOTS[sqg_a075]}" \
      "sqg_a100=${ROOTS[sqg_a100]}"
else
  [[ -f "$SELECTION_JSON" && ! -L "$SELECTION_JSON" ]] || \
    die "selection result is not a safe regular file: $SELECTION_JSON"
  printf 'reusing completed selection score: %s\n' "$SELECTION_JSON"
fi

winner="$(jq -r '.aggregate.selection_policy.winner' "$SELECTION_JSON")"
if [[ "$winner" == mcg ]]; then
  winner="$(jq -r '.aggregate.selection_policy.diagnostic_fallback_nonbaseline' \
    "$SELECTION_JSON")"
fi
[[ -n "${ROOTS[$winner]:-}" ]] || die "selection winner/fallback is unknown: $winner"

if [[ ! -e "$HOLDOUT_JSON" ]]; then
  env "${score_common[@]}" \
    bash "$PROJECT_ROOT/scripts/run_signed_top8_blend_score.sh" \
      holdout "$HOLDOUT_JSON" mcg "$winner=${ROOTS[$winner]}"
else
  [[ -f "$HOLDOUT_JSON" && ! -L "$HOLDOUT_JSON" ]] || \
    die "holdout result is not a safe regular file: $HOLDOUT_JSON"
  printf 'reusing completed holdout score: %s\n' "$HOLDOUT_JSON"
fi

jq -n \
  --arg selection "$SELECTION_JSON" \
  --arg selection_sha256 "$(sha256sum "$SELECTION_JSON" | awk '{print $1}')" \
  --arg holdout "$HOLDOUT_JSON" \
  --arg holdout_sha256 "$(sha256sum "$HOLDOUT_JSON" | awk '{print $1}')" \
  --arg selected_or_diagnostic "$winner" \
  '{
    schema:"glm52-contiguous-late-h13-blend-panel-receipt-v1",
    selection:{path:$selection,sha256:$selection_sha256},
    holdout:{path:$holdout,sha256:$holdout_sha256},
    selected_or_diagnostic:$selected_or_diagnostic,
    note:"If MCG retained the hard-gate win, holdout evaluates the named best nonbaseline diagnostic; it does not override MCG."
  }' > "$PROJECT_ROOT/results/contiguous_late_h13_blend_panel_receipt_r1.json.tmp"
mv "$PROJECT_ROOT/results/contiguous_late_h13_blend_panel_receipt_r1.json.tmp" \
  "$PROJECT_ROOT/results/contiguous_late_h13_blend_panel_receipt_r1.json"

printf 'late H13 blend panel complete: selection=%s holdout=%s\n' \
  "$SELECTION_JSON" "$HOLDOUT_JSON"
