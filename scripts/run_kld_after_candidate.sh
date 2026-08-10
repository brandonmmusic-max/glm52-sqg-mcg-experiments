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
[[ -d "$CANDIDATE" ]] || die "candidate directory is absent: $CANDIDATE"
CANDIDATE="$(realpath -e -- "$CANDIDATE")"

[[ -d "$CANDIDATE" ]] || die "candidate is not a directory: $CANDIDATE"
for required in \
  .manifest_verified MANIFEST.json FRESH_SQG_RUN_SEAL.json \
  config.json quantization_config.json model.safetensors.index.json; do
  [[ -f "$CANDIDATE/$required" ]] || \
    die "candidate is missing required file: $required"
done

ARTIFACTS_ROOT="${FRESH_ARTIFACTS_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode.LFlh2R}"
ARTIFACTS_ROOT="$(realpath -e -- "$ARTIFACTS_ROOT")"
[[ -f "$ARTIFACTS_ROOT/run_seal.json" ]] || \
  die "fresh artifact run seal is absent: $ARTIFACTS_ROOT/run_seal.json"

EVAL_ROOT="${FRESH_EVAL_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-r1}"
CANDIDATE_ROOT="$EVAL_ROOT/candidate"
NATIVE_ROOT="$EVAL_ROOT/native"
PAIRED_ROOT="$EVAL_ROOT/paired"
mkdir -p -- "$CANDIDATE_ROOT" "$NATIVE_ROOT" "$PAIRED_ROOT"
CANDIDATE_ROOT="$(realpath -e -- "$CANDIDATE_ROOT")"
NATIVE_ROOT="$(realpath -e -- "$NATIVE_ROOT")"
PAIRED_ROOT="$(realpath -e -- "$PAIRED_ROOT")"
[[ "$CANDIDATE_ROOT" != "$NATIVE_ROOT" && \
   "$CANDIDATE_ROOT" != "$PAIRED_ROOT" && \
   "$NATIVE_ROOT" != "$PAIRED_ROOT" ]] || \
  die "candidate, native-control, and paired result roots must be distinct"

STAMP="${KLD_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-fresh-sqg4-r1}"
[[ "$STAMP" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "KLD_STAMP may contain only letters, digits, dot, underscore, and dash"

CANDIDATE_OUT="$CANDIDATE_ROOT/$STAMP-candidate-kld-fp8-dcp4"
NATIVE_OUT="$NATIVE_ROOT/$STAMP-native-mcg-control-kld-fp8-dcp4"
PAIR_JSON="$PAIRED_ROOT/$STAMP-sqg-vs-native-mcg.json"
PAIR_TENSOR="$PAIRED_ROOT/$STAMP-sqg-vs-native-mcg.safetensors"
for output in "$CANDIDATE_OUT" "$NATIVE_OUT" "$PAIR_JSON" "$PAIR_TENSOR"; do
  [[ ! -e "$output" && ! -L "$output" ]] || \
    die "refusing to overwrite an existing result: $output"
done

REFERENCE_LOGITS=/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current/reference_hf/reference-logits/logits_0.safetensors
[[ -f "$REFERENCE_LOGITS" && ! -L "$REFERENCE_LOGITS" ]] || \
  die "saved BF16 reference logits are absent: $REFERENCE_LOGITS"

printf 'Candidate KLD: five fresh boots\n'
RUNS=5 \
DIRECTIONAL_TEST_FAST=1 \
STAMP="$STAMP" \
RESULTS_ROOT="$CANDIDATE_ROOT" \
FRESH_ARTIFACTS_ROOT="$ARTIFACTS_ROOT" \
RUNTIME_OVERLAY="$EVALUATION_DIR/runtime_overlay" \
EXTRA_DOCKER_ARGS_FILE="$EVALUATION_DIR/r33_exact_runtime.args" \
  "$EVALUATION_DIR/run_fresh_kld.sh" "$CANDIDATE"

[[ -f "$CANDIDATE_OUT/summary.json" ]] || \
  die "candidate runner did not publish its summary"
jq -e '
  .schema == "glm52-fresh-sqg-candidate-kld-result-v2" and
  .baseline_was_rerun == false and .candidate_result.runs == 5
' "$CANDIDATE_OUT/summary.json" >/dev/null || \
  die "candidate summary does not contain exactly five fresh boots"

printf 'Native-MCG control KLD: five fresh boots\n'
RUNS=5 \
DIRECTIONAL_TEST_FAST=1 \
STAMP="$STAMP" \
NATIVE_MCG_RESULTS_ROOT="$NATIVE_ROOT" \
  "$EVALUATION_DIR/run_native_mcg_control.sh"

[[ -f "$NATIVE_OUT/summary.json" ]] || \
  die "native-MCG runner did not publish its summary"
jq -e '
  .schema == "glm52-native-mcg-control-kld-result-v1" and
  .control_result.runs == 5
' "$NATIVE_OUT/summary.json" >/dev/null || \
  die "native-MCG summary does not contain exactly five fresh boots"

printf 'Paired per-position SQG-minus-native-MCG analysis\n'
python3 "$EVALUATION_DIR/analyze_native_mcg_pair.py" \
  "$CANDIDATE_OUT/summary.json" \
  "$NATIVE_OUT/summary.json" \
  --expected-runs 5 \
  --json-output "$PAIR_JSON" \
  --tensor-output "$PAIR_TENSOR"

jq -n \
  --arg candidate_summary "$CANDIDATE_OUT/summary.json" \
  --arg native_summary "$NATIVE_OUT/summary.json" \
  --arg paired_json "$PAIR_JSON" \
  --arg paired_tensor "$PAIR_TENSOR" '
  {
    candidate_summary: $candidate_summary,
    native_mcg_summary: $native_summary,
    paired_analysis: $paired_json,
    paired_tensor: $paired_tensor
  }
'
