#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -le 1 ]] || die "usage: $0 [/absolute/path/to/candidate-model]"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALUATION_DIR="$PROJECT_DIR/evaluation"
CANDIDATE="${1:-/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-FRESH-SQG4-r1}"
CANDIDATE="$(realpath -e -- "$CANDIDATE")"
ARTIFACTS_ROOT="${FRESH_ARTIFACTS_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode.LFlh2R}"
ARTIFACTS_ROOT="$(realpath -e -- "$ARTIFACTS_ROOT")"
EVAL_ROOT="${FRESH_EVAL_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-r1}"
STAMP="${KLD_STAMP:-fresh-sqg4-r1}"
[[ "$STAMP" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe KLD_STAMP"

CANDIDATE_ROOT="$EVAL_ROOT/candidate"
NATIVE_ROOT="$EVAL_ROOT/native"
PAIRED_ROOT="$EVAL_ROOT/paired"
CANDIDATE_OUT="$CANDIDATE_ROOT/$STAMP-candidate-kld-fp8-dcp4"
NATIVE_OUT="$NATIVE_ROOT/$STAMP-native-mcg-control-kld-fp8-dcp4"
PAIR_JSON="$PAIRED_ROOT/$STAMP-sqg-vs-native-mcg.json"
PAIR_TENSOR="$PAIRED_ROOT/$STAMP-sqg-vs-native-mcg.safetensors"

[[ -d "$CANDIDATE_OUT" && ! -L "$CANDIDATE_OUT" ]] || \
  die "candidate run-1 output is absent"
if [[ ! -f "$CANDIDATE_OUT/run1-record.accepted.json" ]]; then
  "$EVALUATION_DIR/salvage_completed_candidate_run1.sh" "$CANDIDATE_OUT"
fi
jq -e '.mean_kld == 1.8742683687382462 and
  .runtime_cache.started_empty == true and
  .runtime_cache.byte_hashing_skipped == true and
  .runtime_dispatch.sqg_dispatch_proved == true and
  .per_position.independently_validated == true' \
  "$CANDIDATE_OUT/run1-record.accepted.json" >/dev/null || \
  die "preserved candidate run 1 differs"

[[ "$(docker inspect --format '{{.State.Running}}' glm-r33-fixed \
  2>/dev/null || true)" != true ]] || die "production container is running"
[[ ! -e "$CANDIDATE_OUT/summary.json" ]] || \
  die "candidate output already has a summary"
for run in 2 3 4 5; do
  if find "$CANDIDATE_OUT" -maxdepth 1 -name "run${run}*" -print -quit | \
    grep -q .; then
    die "candidate run $run has incomplete artifacts; quarantine them before continuation"
  fi
done
[[ ! -e "$NATIVE_OUT" && ! -L "$NATIVE_OUT" ]] || \
  die "native output already exists"
[[ ! -e "$PAIR_JSON" && ! -L "$PAIR_JSON" &&
   ! -e "$PAIR_TENSOR" && ! -L "$PAIR_TENSOR" ]] || \
  die "paired output already exists"

printf 'Candidate KLD continuation: preserve run 1; snapshot-seeded boots 2-5\n'
RUNS=5 \
RESUME_FROM_RUN=2 \
SNAPSHOT_SEED_FROM_RUN1=1 \
DIRECTIONAL_TEST_FAST=1 \
STAMP="$STAMP" \
RESULTS_ROOT="$CANDIDATE_ROOT" \
FRESH_ARTIFACTS_ROOT="$ARTIFACTS_ROOT" \
RUNTIME_OVERLAY="$EVALUATION_DIR/runtime_overlay" \
EXTRA_DOCKER_ARGS_FILE="$EVALUATION_DIR/r33_exact_runtime.args" \
  "$EVALUATION_DIR/run_fresh_kld.sh" "$CANDIDATE"

jq -e '.schema == "glm52-fresh-sqg-candidate-kld-result-v2" and
  .candidate_result.runs == 5' "$CANDIDATE_OUT/summary.json" >/dev/null || \
  die "candidate continuation did not publish five accepted boots"

printf 'Native-MCG control: candidate-cache-seeded boot 1; native-run1-seeded boots 2-5\n'
RUNS=5 \
SNAPSHOT_SEED_RUNTIME_CACHE=1 \
NATIVE_BOOT1_CACHE_SEED="$CANDIDATE_OUT/run1-candidate-runtime-cache" \
DIRECTIONAL_TEST_FAST=1 \
STAMP="$STAMP" \
NATIVE_MCG_RESULTS_ROOT="$NATIVE_ROOT" \
  "$EVALUATION_DIR/run_native_mcg_control.sh"

jq -e '.schema == "glm52-native-mcg-control-kld-result-v1" and
  .control_result.runs == 5' "$NATIVE_OUT/summary.json" >/dev/null || \
  die "native-MCG runner did not publish five accepted boots"

python3 "$EVALUATION_DIR/analyze_native_mcg_pair.py" \
  "$CANDIDATE_OUT/summary.json" "$NATIVE_OUT/summary.json" \
  --expected-runs 5 --json-output "$PAIR_JSON" --tensor-output "$PAIR_TENSOR"

jq -n --arg candidate_summary "$CANDIDATE_OUT/summary.json" \
  --arg native_summary "$NATIVE_OUT/summary.json" \
  --arg paired_json "$PAIR_JSON" --arg paired_tensor "$PAIR_TENSOR" '
  {
    candidate_summary:$candidate_summary,native_mcg_summary:$native_summary,
    paired_analysis:$paired_json,paired_tensor:$paired_tensor
  }
'
