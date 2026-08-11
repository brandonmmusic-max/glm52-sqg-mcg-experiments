#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test
TRACE_OVERLAY=/home/brandonmusic/KLC_SANDBOXES/sqg-tail-trace-overlay-r1
CANDIDATE=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-SQG-CONTIG-L74-77-A025-r1
ARTIFACTS=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-contig-late-final-a025-r1
OUTPUT_ROOT="${1:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-tail-trace-contig-late-a025-r1}"
CACHE_ROOT=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-r1/candidate/fresh-sqg4-r1-candidate-kld-fp8-dcp4
CACHE_SEED="$CACHE_ROOT/run1-candidate-runtime-cache"
CACHE_RECEIPT="$CACHE_ROOT/run1-runtime-cache-receipt.json"
CACHE_RECORD="$CACHE_ROOT/run1-record.accepted.json"

[[ "$OUTPUT_ROOT" = /* ]] || die "output root must be absolute"
[[ ! -L "$OUTPUT_ROOT" ]] || die "output root may not be a symlink"
for path in "$TRACE_OVERLAY" "$CANDIDATE" "$ARTIFACTS" "$CACHE_SEED"; do
  [[ -d "$path" && ! -L "$path" ]] || die "required directory is absent: $path"
done
for path in "$CACHE_RECEIPT" "$CACHE_RECORD"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required file is absent: $path"
done
if [[ ! -e "$OUTPUT_ROOT" ]]; then
  mkdir -m 0700 "$OUTPUT_ROOT"
elif [[ ! -d "$OUTPUT_ROOT" ]]; then
  die "output root exists but is not a directory: $OUTPUT_ROOT"
fi

# The r33 arm is a one-boot trace diagnostic, not a rerun of the five-boot
# native-dispatch MCG control.  It retains the protected historical r33
# runtime; layers 74--77 are already outside the fused allowlist and therefore
# use the native/nonfused MCG path without changing the checkpoint.
if [[ ! -e "$OUTPUT_ROOT/r33" ]]; then
  bash "$PROJECT_ROOT/evaluation/run_r33_trace_once.sh" "$OUTPUT_ROOT/r33"
elif [[ ! -f "$OUTPUT_ROOT/r33/record.json" || \
        ! -f "$OUTPUT_ROOT/r33/r33-position-kld.safetensors" || \
        ! -d "$OUTPUT_ROOT/r33/tail-trace" ]]; then
  die "partial r33 arm cannot be resumed safely"
fi

# A completed diagnostic from an older launcher may have stopped after capture
# because it assumed exactly one traced invocation.  Seal that evidence in
# place after requiring every layer/rank pair; never rerun the completed arm.
if [[ ! -f "$OUTPUT_ROOT/r33/evidence.sha256" ]]; then
  for layer in 074 075 076 077; do
    for rank in 000 001 002 003; do
      compgen -G "$OUTPUT_ROOT/r33/tail-trace/layer-$layer-rank-$rank-call-*.safetensors" \
        >/dev/null || die "saved r33 trace lacks layer $layer DCP rank $rank"
    done
  done
  jq -e '
    .total_positions == 2047 and .mean_kld >= 0 and
    .per_position.positions == 2047 and
    .token_ids_u32le_sha256 == "ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56"
  ' "$OUTPUT_ROOT/r33/record.json" >/dev/null || die "saved r33 record differs"
  position_sha="$(sha256sum "$OUTPUT_ROOT/r33/r33-position-kld.safetensors" | awk '{print $1}')"
  python3 "$PROJECT_ROOT/evaluation/validate_per_position_kld.py" \
    "$OUTPUT_ROOT/r33/r33-position-kld.safetensors" \
    --expected-sha256 "$position_sha" \
    --expected-mean-kld "$(jq -r .mean_kld "$OUTPUT_ROOT/r33/record.json")" \
    > "$OUTPUT_ROOT/r33/per-position-validation.json"
  sha256sum "$OUTPUT_ROOT/r33/record.json" \
    "$OUTPUT_ROOT/r33/r33-position-kld.safetensors" \
    "$OUTPUT_ROOT/r33/per-position-validation.json" \
    "$OUTPUT_ROOT"/r33/tail-trace/*.safetensors \
    > "$OUTPUT_ROOT/r33/evidence.sha256"
fi

if [[ -e "$OUTPUT_ROOT/sqg" ]]; then
  die "SQG trace arm already exists and requires explicit inspection"
fi

SQG_EVAL_LAYERS=74,75,76,77 \
FRESH_BF16_LAYERS_ROOT="$PROJECT_ROOT/bf16_contiguous_late" \
FRESH_CAPTURE_DIR=/home/brandonmusic/KLC_CAPTURE_RUNS/contig-late-capture-r1/contig-late-r1 \
FRESH_SQG_BF16_MANIFEST="$PROJECT_ROOT/evidence/contiguous_late_bf16_manifest_r1.json" \
FRESH_SQG_PLAN_CONTRACT="$PROJECT_ROOT/evidence/contiguous_document_plan_r1.json" \
BIT_CONTRACT="$PROJECT_ROOT/contracts/contiguous_late_bit_allocations_r1.json" \
RUNS=1 \
SNAPSHOT_SEED_FROM_RUN1=0 \
CANDIDATE_BOOT_CACHE_SEED="$CACHE_SEED" \
CANDIDATE_BOOT_CACHE_SEED_RECEIPT="$CACHE_RECEIPT" \
CANDIDATE_BOOT_CACHE_SEED_RECORD="$CACHE_RECORD" \
DIRECTIONAL_TEST_FAST=1 \
SQG_TAIL_TRACE=1 \
SQG_TAIL_TRACE_LAYERS=74,75,76,77 \
STAMP=contig-late-a025-tail-trace-r1 \
RESULTS_ROOT="$OUTPUT_ROOT/sqg" \
FRESH_ARTIFACTS_ROOT="$ARTIFACTS" \
RUNTIME_OVERLAY="$TRACE_OVERLAY" \
EXTRA_DOCKER_ARGS_FILE="$PROJECT_ROOT/evaluation/r33_exact_runtime.args" \
  "$PROJECT_ROOT/evaluation/run_fresh_kld.sh" "$CANDIDATE"

SQG_RUN="$OUTPUT_ROOT/sqg/contig-late-a025-tail-trace-r1-candidate-kld-fp8-dcp4"
OUTPUT_JSON="$PROJECT_ROOT/results/contiguous_late_tail_trace_a025_r1.json"
[[ -f "$OUTPUT_ROOT/r33/r33-position-kld.safetensors" ]] || \
  die "r33 per-position trace result is absent"
[[ -f "$SQG_RUN/run1-container-output/run1-position-kld.safetensors" ]] || \
  die "SQG per-position trace result is absent"
sudo -n chown -R "$(id -un):$(id -gn)" \
  "$SQG_RUN/run1-container-output/tail-trace" || \
  die "could not make SQG trace evidence host-readable"
for path in "$OUTPUT_JSON" "${OUTPUT_JSON%.json}.npz" "${OUTPUT_JSON%.json}.md"; do
  [[ ! -e "$path" && ! -L "$path" ]] || die "analysis output already exists: $path"
done

docker run --rm --network none --user 1000:1000 \
  --mount type=bind,src=/home/brandonmusic/KLC_SANDBOXES,dst=/home/brandonmusic/KLC_SANDBOXES \
  -e PYTHONPATH="$PROJECT_ROOT" \
  --entrypoint /opt/venv/bin/python \
  -w "$PROJECT_ROOT" \
  sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82 \
  scripts/analyze_tail_trace_pair.py \
    --baseline-trace "$OUTPUT_ROOT/r33/tail-trace" \
    --candidate-trace "$SQG_RUN/run1-container-output/tail-trace" \
    --baseline-kld "$OUTPUT_ROOT/r33/r33-position-kld.safetensors" \
    --candidate-kld "$SQG_RUN/run1-container-output/run1-position-kld.safetensors" \
    --output "$OUTPUT_JSON"

sha256sum "$OUTPUT_JSON" "${OUTPUT_JSON%.json}.npz" "${OUTPUT_JSON%.json}.md" \
  > "$PROJECT_ROOT/results/contiguous_late_tail_trace_a025_r1.sha256"
printf 'late paired trace complete: %s\n' "$OUTPUT_JSON"
