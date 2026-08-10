#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

[[ $# -eq 4 ]] || \
  die "usage: $0 /absolute/candidate /absolute/artifacts /absolute/eval-root STAMP"
[[ "$1" = /* && "$2" = /* && "$3" = /* ]] || die "paths must be absolute"
[[ "$4" =~ ^[A-Za-z0-9._-]+$ ]] || die "stamp is unsafe"

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
CANDIDATE="$(realpath -e -- "$1")"
ARTIFACTS_ROOT="$(realpath -e -- "$2")"
EVAL_ROOT="$3"
STAMP="$4"
RUNS="${RUNS:-5}"
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"

for required in .manifest_verified MANIFEST.json FRESH_SQG_RUN_SEAL.json; do
  [[ -f "$CANDIDATE/$required" ]] || die "candidate is missing $required"
done
[[ -f "$ARTIFACTS_ROOT/run_seal.json" ]] || die "artifact run seal is absent"
mkdir -p "$EVAL_ROOT/candidate"

# This seed contains compiled executable code only.  The runner independently
# checks its receipt and accepted record before making an isolated per-boot
# copy.  It contains no model weights, logits, KV state, RNG state, or process
# state and avoids recompiling the unchanged normal SQG runtime overlay.
CACHE_OUT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-r1/candidate/fresh-sqg4-r1-candidate-kld-fp8-dcp4"
CACHE_SEED="$CACHE_OUT/run1-candidate-runtime-cache"
CACHE_RECEIPT="$CACHE_OUT/run1-runtime-cache-receipt.json"
CACHE_RECORD="$CACHE_OUT/run1-record.accepted.json"
for input in "$CACHE_SEED" "$CACHE_RECEIPT" "$CACHE_RECORD"; do
  [[ -e "$input" && ! -L "$input" ]] || die "compiled-cache evidence is absent: $input"
done

RUNS="$RUNS" \
SNAPSHOT_SEED_FROM_RUN1=0 \
CANDIDATE_BOOT_CACHE_SEED="$CACHE_SEED" \
CANDIDATE_BOOT_CACHE_SEED_RECEIPT="$CACHE_RECEIPT" \
CANDIDATE_BOOT_CACHE_SEED_RECORD="$CACHE_RECORD" \
DIRECTIONAL_TEST_FAST=1 \
STAMP="$STAMP" \
RESULTS_ROOT="$EVAL_ROOT/candidate" \
FRESH_ARTIFACTS_ROOT="$ARTIFACTS_ROOT" \
RUNTIME_OVERLAY="$PROJECT_ROOT/evaluation/runtime_overlay" \
EXTRA_DOCKER_ARGS_FILE="$PROJECT_ROOT/evaluation/r33_exact_runtime.args" \
  "$PROJECT_ROOT/evaluation/run_fresh_kld.sh" "$CANDIDATE"

summary="$EVAL_ROOT/candidate/$STAMP-candidate-kld-fp8-dcp4/summary.json"
[[ -f "$summary" ]] || die "candidate KLD summary is absent"
jq -e --argjson runs "$RUNS" '
  .schema == "glm52-fresh-sqg-candidate-kld-result-v2" and
  .baseline_was_rerun == false and
  .candidate_result.runs == $runs and
  (.candidate_result.paired_per_position_outputs | length) == $runs
' "$summary" >/dev/null || die "candidate KLD summary differs"

printf 'H13 blend KLD complete: %s\n' "$summary"
