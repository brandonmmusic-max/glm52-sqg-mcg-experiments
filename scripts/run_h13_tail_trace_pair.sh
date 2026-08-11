#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test
TRACE_OVERLAY=/home/brandonmusic/KLC_SANDBOXES/sqg-tail-trace-overlay-r1
ALPHA0_MODEL=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-FRESH-SQG4-ABSRMS-r2
ALPHA025_MODEL=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-SQG-H13A025-r1
ALPHA0_ARTIFACTS=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2
ALPHA025_ARTIFACTS=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-h13-alpha025-r1
OUTPUT_ROOT="${1:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-tail-trace-h13a025-r1}"
CACHE_ROOT=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-r1/candidate/fresh-sqg4-r1-candidate-kld-fp8-dcp4
CACHE_SEED="$CACHE_ROOT/run1-candidate-runtime-cache"
CACHE_RECEIPT="$CACHE_ROOT/run1-runtime-cache-receipt.json"
CACHE_RECORD="$CACHE_ROOT/run1-record.accepted.json"

[[ "$OUTPUT_ROOT" = /* ]] || die "output root must be absolute"
[[ ! -e "$OUTPUT_ROOT" ]] || die "output root already exists: $OUTPUT_ROOT"
for path in \
  "$TRACE_OVERLAY" "$ALPHA0_MODEL" "$ALPHA025_MODEL" \
  "$ALPHA0_ARTIFACTS" "$ALPHA025_ARTIFACTS" "$CACHE_SEED"; do
  [[ -d "$path" && ! -L "$path" ]] || die "required directory is absent or unsafe: $path"
done
for path in "$CACHE_RECEIPT" "$CACHE_RECORD"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required file is absent or unsafe: $path"
done
mkdir -p "$OUTPUT_ROOT/alpha0" "$OUTPUT_ROOT/alpha025"

run_arm() {
  local label="$1" model="$2" artifacts="$3"
  RUNS=1 \
  SNAPSHOT_SEED_FROM_RUN1=0 \
  CANDIDATE_BOOT_CACHE_SEED="$CACHE_SEED" \
  CANDIDATE_BOOT_CACHE_SEED_RECEIPT="$CACHE_RECEIPT" \
  CANDIDATE_BOOT_CACHE_SEED_RECORD="$CACHE_RECORD" \
  DIRECTIONAL_TEST_FAST=1 \
  SQG_TAIL_TRACE=1 \
  SQG_TAIL_TRACE_LAYERS=6,28,52,77 \
  STAMP="h13-tail-trace-$label-r1" \
  RESULTS_ROOT="$OUTPUT_ROOT/$label" \
  FRESH_ARTIFACTS_ROOT="$artifacts" \
  RUNTIME_OVERLAY="$TRACE_OVERLAY" \
  EXTRA_DOCKER_ARGS_FILE="$PROJECT_ROOT/evaluation/r33_exact_runtime.args" \
    "$PROJECT_ROOT/evaluation/run_fresh_kld.sh" "$model"
}

run_arm alpha0 "$ALPHA0_MODEL" "$ALPHA0_ARTIFACTS"
run_arm alpha025 "$ALPHA025_MODEL" "$ALPHA025_ARTIFACTS"

ALPHA0_RUN="$OUTPUT_ROOT/alpha0/h13-tail-trace-alpha0-r1-candidate-kld-fp8-dcp4"
ALPHA025_RUN="$OUTPUT_ROOT/alpha025/h13-tail-trace-alpha025-r1-candidate-kld-fp8-dcp4"
OUTPUT_JSON="$PROJECT_ROOT/results/h13_blend_tail_trace_alpha025_r1.json"
[[ ! -e "$OUTPUT_JSON" && ! -L "$OUTPUT_JSON" ]] || \
  die "analysis output already exists: $OUTPUT_JSON"

docker run --rm --network none --user 1000:1000 \
  --mount type=bind,src=/home/brandonmusic/KLC_SANDBOXES,dst=/home/brandonmusic/KLC_SANDBOXES \
  -e PYTHONPATH="$PROJECT_ROOT" \
  --entrypoint /opt/venv/bin/python \
  -w "$PROJECT_ROOT" \
  sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82 \
  scripts/analyze_tail_trace_pair.py \
    --baseline-trace "$ALPHA0_RUN/run1-container-output/tail-trace" \
    --candidate-trace "$ALPHA025_RUN/run1-container-output/tail-trace" \
    --baseline-kld "$ALPHA0_RUN/run1-container-output/run1-position-kld.safetensors" \
    --candidate-kld "$ALPHA025_RUN/run1-container-output/run1-position-kld.safetensors" \
    --output "$OUTPUT_JSON"

printf 'Paired H13 tail trace complete: %s\n' "$OUTPUT_JSON"
