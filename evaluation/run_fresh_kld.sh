#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "usage: $0 /absolute/path/to/candidate-model"

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$EXPERIMENT_DIR/.." && pwd)"
RUNNER="$EXPERIMENT_DIR/run_fresh_kld.sh"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results}"
PRODUCTION_MODEL=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED
PRODUCTION_CONTAINER=glm-r33-fixed
CANDIDATE="$(realpath -e -- "$1")"
PURE_SQG_VALIDATOR="$EXPERIMENT_DIR/validate_pure_sqg_candidate.py"
POSITION_VALIDATOR="$EXPERIMENT_DIR/validate_per_position_kld.py"
PURE_SQG_VALIDATOR_SHA256=f5b6190200b0c30c60c27f3192f56e61b3d74554e5d48df54307c3c75af00a41
POSITION_VALIDATOR_SHA256=e6b57fc605e3ec7f9ec297f10ef400f83664a06d7b0c04d3320a7f7996193758
PURE_SQG_SOURCE_EVIDENCE="$EXPERIMENT_DIR/pure_sqg_validator_sources.sha256"
PURE_SQG_SOURCE_EVIDENCE_SHA256=84421a3a35391abc61b8cf612011d11b2118d4f5289f837e23f7dbc2d0501c40
PURE_SQG_SOURCE_FILE_COUNT=58
TEACHER_RECEIPT="$(realpath -e -- "${TEACHER_RECEIPT:-$PROJECT_DIR/evidence/teacher_model_identity.json}")"
BIT_CONTRACT="$(realpath -e -- "${BIT_CONTRACT:-$PROJECT_DIR/contracts/frozen_bit_allocations.json}")"
[[ -n "${FRESH_ARTIFACTS_ROOT:-}" ]] || \
  die "FRESH_ARTIFACTS_ROOT is required and must contain run_seal.json"
ARTIFACTS_ROOT="$(realpath -e -- "$FRESH_ARTIFACTS_ROOT")"
RUN_SEAL="$(realpath -e -- "${FRESH_RUN_SEAL:-$ARTIFACTS_ROOT/run_seal.json}")"
BF16_LAYERS_SOURCE="$(realpath -e -- \
  "${FRESH_BF16_LAYERS_ROOT:-$PROJECT_DIR/bf16_layers}")"
CAPTURE_SOURCE="$(realpath -e -- \
  "${FRESH_CAPTURE_DIR:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1}")"
SQG_EXTENSION_SOURCE="$(realpath -e -- \
  "${FRESH_SQG_EXTENSION_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33.p7n1IJ/sealed}")"
MODELS_ROOT="$(realpath -e -- "$(dirname "$PRODUCTION_MODEL")")"
case "$TEACHER_RECEIPT" in
  "$PROJECT_DIR"/*)
    TEACHER_RECEIPT_CONTAINER="/work/${TEACHER_RECEIPT#"$PROJECT_DIR"/}"
    ;;
  *) die "TEACHER_RECEIPT must be below the project root for sealed validation" ;;
esac
case "$BIT_CONTRACT" in
  "$PROJECT_DIR"/*)
    BIT_CONTRACT_CONTAINER="/work/${BIT_CONTRACT#"$PROJECT_DIR"/}"
    ;;
  *) die "BIT_CONTRACT must be below the project root for sealed validation" ;;
esac
case "$RUN_SEAL" in
  "$ARTIFACTS_ROOT"/*)
    RUN_SEAL_CONTAINER="/output/${RUN_SEAL#"$ARTIFACTS_ROOT"/}"
    ;;
  *) die "FRESH_RUN_SEAL must be below FRESH_ARTIFACTS_ROOT" ;;
esac
VALIDATOR_DYNAMIC_ENV_ARGS=()
for validator_path_env in FRESH_SQG_BF16_MANIFEST FRESH_SQG_PLAN_CONTRACT; do
  validator_path="${!validator_path_env:-}"
  [[ -n "$validator_path" ]] || continue
  validator_path="$(realpath -e -- "$validator_path")"
  case "$validator_path" in
    "$PROJECT_DIR"/*)
      validator_container_path="/work/${validator_path#"$PROJECT_DIR"/}"
      ;;
    *)
      die "$validator_path_env must be below the project root for sealed validation"
      ;;
  esac
  VALIDATOR_DYNAMIC_ENV_ARGS+=(
    -e "$validator_path_env=$validator_container_path"
  )
done
BF16_LAYERS_CONTAINER="$BF16_LAYERS_SOURCE"
PURE_SQG_VALIDATOR_CONTAINER=/work/evaluation/validate_pure_sqg_candidate.py
SQG_EXTENSION_CONTAINER=/sqg-extension/kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so
SQG_EXTENSION_SHA256=c987778677653388f7766e66150850c4dda44bc6b055929ece34eaee87f3cda4

[[ -d "$CANDIDATE" ]] || die "candidate is not a directory: $CANDIDATE"
case "$CANDIDATE" in
  "$MODELS_ROOT"/*) ;;
  *) die "candidate must be below the sealed model root: $MODELS_ROOT" ;;
esac
case "$CANDIDATE" in
  "$PRODUCTION_MODEL"|"$PRODUCTION_MODEL"/*)
    die "refusing to evaluate the protected production checkpoint in place"
    ;;
esac

for required in \
  .manifest_verified MANIFEST.json FRESH_SQG_RUN_SEAL.json \
  config.json quantization_config.json model.safetensors.index.json; do
  [[ -f "$CANDIDATE/$required" ]] || \
    die "candidate is missing required file: $required"
done

if [[ "$(docker inspect --format '{{.State.Running}}' \
  "$PRODUCTION_CONTAINER" 2>/dev/null || true)" == "true" ]]; then
  die "production container is running: $PRODUCTION_CONTAINER"
fi

# A persistent background process owns about 552 MiB per GPU on this host.
# Reject any compute process above 768 MiB (which catches encoder/oracle jobs)
# as well as any GPU above 2 GiB total, and repeat this immediately before
# every candidate launch.
GPU_PROCESS_LIMIT_MIB=768
GPU_TOTAL_LIMIT_MIB=2048
require_idle_gpus() {
  local compute_snapshot busy_compute busy_gpus
  compute_snapshot="$(nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  busy_compute="$(printf '%s\n' "$compute_snapshot" | awk -F, \
    -v limit="$GPU_PROCESS_LIMIT_MIB" '
      {
        used = $NF
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", used)
        if ((used + 0) > limit) print
      }
    ')"
  if [[ -n "$busy_compute" ]]; then
    printf 'GPU compute processes above %s MiB:\n%s\n' \
      "$GPU_PROCESS_LIMIT_MIB" "$busy_compute" >&2
    die "GPU audit/encoder work is still active"
  fi
  busy_gpus="$({
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  } | awk -v limit="$GPU_TOTAL_LIMIT_MIB" \
      '$1 > limit { n++ } END { print n + 0 }')"
  [[ "$busy_gpus" -eq 0 ]] || \
    die "$busy_gpus GPU(s) have more than $GPU_TOTAL_LIMIT_MIB MiB allocated"
}
require_idle_gpus

# The SQG loader is rebased on the exact current r33 production runtime.  A
# five-run corrected-checkpoint r33 KLD result already exists, so use it as the
# same-base-image control without rerunning any baseline. The SQG overlay and
# required non-fused dispatch make this directional, not codebook-only. Retain
# the original published r26 result as a secondary historical comparison.
RUNTIME_BASELINE_DIR=/home/brandonmusic/klc-linux/release-gg-sparkinfer-20260721/results/20260809T194958Z-kld-fp8-dcp4
RUNTIME_BASELINE_SUMMARY="$RUNTIME_BASELINE_DIR/summary.json"
RUNTIME_BASELINE_SUMMARY_SHA256=07096cd5f0a683bd00a1e169c547ef48eaaec8f6270dfce355fa7b0a06834ad4
RUNTIME_BASELINE_EVIDENCE="$EXPERIMENT_DIR/r33_baseline_evidence.sha256"
RUNTIME_BASELINE_EVIDENCE_SHA256=9c4669fea9a3affa9b3fa6a164f69614f1d1336840b8638aaec2d4695d06c34c
RUNTIME_BASELINE_IMAGE_LABEL=voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12x06db0f4-fi1ac6942-cu132-20260809-r33
RUNTIME_BASELINE_IMAGE_ID=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
LEGACY_BASELINE_IMAGE_LABEL=glm52-r26-r7:v1
LEGACY_BASELINE_IMAGE_ID=sha256:e0121c436c50ab699b221fae7e1c04a655c97dd825c705f059a373928db89d56
LEGACY_BASELINE_MEAN=0.061282244905043234
LEGACY_BASELINE_SD=0.0013762397578544292
IMAGE="${IMAGE:-$RUNTIME_BASELINE_IMAGE_ID}"
RUNTIME_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
[[ "$RUNTIME_IMAGE_ID" == "$RUNTIME_BASELINE_IMAGE_ID" ]] || \
  die "candidate image does not match the preserved r33 baseline image"
SITE="${SITE:-/opt/venv/lib/python3.12/site-packages}"
PYTHON_ENTRYPOINT="${PYTHON_ENTRYPOINT:-/opt/venv/bin/python}"
[[ "$SITE" == /opt/venv/lib/python3.12/site-packages ]] || \
  die "SITE must match the exact r33 image environment"
[[ "$PYTHON_ENTRYPOINT" == /opt/venv/bin/python ]] || \
  die "PYTHON_ENTRYPOINT must match the exact r33 image environment"

# Preserve the measured regime: FP8 KV, BF16 RoPE, TP4/DCP4, A2A, and a
# 64-token DCP interleave. These affect KLD and are not tuning knobs here.
KV_DTYPE=fp8
DCP_SIZE=4
DCP_COMM_BACKEND=a2a
DCP_KV_CACHE_INTERLEAVE_SIZE=64
KLD_MAX_MODEL_LEN=2560
KLD_UTIL="${KLD_UTIL:-0.90}"
RUNS="${RUNS:-5}"
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"
SQG_EVAL_LAYERS="${SQG_EVAL_LAYERS:-6,28,52,77}"
[[ "$SQG_EVAL_LAYERS" =~ ^[1-9][0-9]*(,[1-9][0-9]*){3}$ ]] || \
  die "SQG_EVAL_LAYERS must be exactly four comma-separated integers"
IFS=, read -r -a SQG_EVAL_LAYER_ARRAY <<<"$SQG_EVAL_LAYERS"
previous_eval_layer=-1
for eval_layer in "${SQG_EVAL_LAYER_ARRAY[@]}"; do
  (( eval_layer >= 3 && eval_layer <= 77 )) || \
    die "SQG_EVAL_LAYERS entries must be routed layers in [3,77]"
  (( eval_layer > previous_eval_layer )) || \
    die "SQG_EVAL_LAYERS must contain four unique ascending layers"
  previous_eval_layer="$eval_layer"
done
SQG_EVAL_LAYERS_JSON="[$SQG_EVAL_LAYERS]"
SQG_EVAL_OVERRIDES_JSON="$(jq -cn \
  --argjson layers "$SQG_EVAL_LAYERS_JSON" '
  $layers | map({key:tostring,value:"sqg_xor_cheb_t12"}) | from_entries
')"
SQG_TAIL_TRACE="${SQG_TAIL_TRACE:-0}"
[[ "$SQG_TAIL_TRACE" == 0 || "$SQG_TAIL_TRACE" == 1 ]] || \
  die "SQG_TAIL_TRACE must be 0 or 1"
SQG_TAIL_TRACE_LAYERS="${SQG_TAIL_TRACE_LAYERS:-$SQG_EVAL_LAYERS}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || \
  die "PREFLIGHT_ONLY must be 0 or 1"
RETRY_INCOMPLETE_RUN1="${RETRY_INCOMPLETE_RUN1:-0}"
[[ "$RETRY_INCOMPLETE_RUN1" == 0 || "$RETRY_INCOMPLETE_RUN1" == 1 ]] || \
  die "RETRY_INCOMPLETE_RUN1 must be 0 or 1"
DIRECTIONAL_TEST_FAST="${DIRECTIONAL_TEST_FAST:-0}"
[[ "$DIRECTIONAL_TEST_FAST" == 0 || "$DIRECTIONAL_TEST_FAST" == 1 ]] || \
  die "DIRECTIONAL_TEST_FAST must be 0 or 1"
DIRECTIONAL_TEST_FAST_JSON=false
[[ "$DIRECTIONAL_TEST_FAST" == 0 ]] || DIRECTIONAL_TEST_FAST_JSON=true
RESUME_FROM_RUN="${RESUME_FROM_RUN:-1}"
[[ "$RESUME_FROM_RUN" =~ ^[1-9][0-9]*$ ]] || \
  die "RESUME_FROM_RUN must be a positive integer"
[[ "$RESUME_FROM_RUN" -le "$RUNS" ]] || \
  die "RESUME_FROM_RUN must not exceed RUNS"
SNAPSHOT_SEED_FROM_RUN1="${SNAPSHOT_SEED_FROM_RUN1:-0}"
[[ "$SNAPSHOT_SEED_FROM_RUN1" == 0 || \
   "$SNAPSHOT_SEED_FROM_RUN1" == 1 ]] || \
  die "SNAPSHOT_SEED_FROM_RUN1 must be 0 or 1"
CANDIDATE_BOOT_CACHE_SEED="${CANDIDATE_BOOT_CACHE_SEED:-}"
CANDIDATE_BOOT_CACHE_SEED_RECEIPT="${CANDIDATE_BOOT_CACHE_SEED_RECEIPT:-}"
CANDIDATE_BOOT_CACHE_SEED_RECORD="${CANDIDATE_BOOT_CACHE_SEED_RECORD:-}"
CANDIDATE_BOOT_CACHE_SEED_RECEIPT_SHA256=""
CANDIDATE_BOOT_CACHE_SEED_RECORD_SHA256=""
CANDIDATE_BOOT_CACHE_SEED_EXPECTED_FILES=0
RESUME_MODE=0
[[ "$RESUME_FROM_RUN" -eq 1 ]] || RESUME_MODE=1
if [[ -n "$CANDIDATE_BOOT_CACHE_SEED" ]]; then
  [[ "$SNAPSHOT_SEED_FROM_RUN1" == 0 ]] || \
    die "external candidate cache seeding and run-1 seeding are mutually exclusive"
  [[ -n "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" &&
     -n "$CANDIDATE_BOOT_CACHE_SEED_RECORD" ]] || \
    die "external candidate cache seed requires its receipt and accepted record"
  CANDIDATE_BOOT_CACHE_SEED="$(realpath -e -- "$CANDIDATE_BOOT_CACHE_SEED")"
  CANDIDATE_BOOT_CACHE_SEED_RECEIPT="$(realpath -e -- \
    "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT")"
  CANDIDATE_BOOT_CACHE_SEED_RECORD="$(realpath -e -- \
    "$CANDIDATE_BOOT_CACHE_SEED_RECORD")"
  [[ -d "$CANDIDATE_BOOT_CACHE_SEED" &&
     ! -L "$CANDIDATE_BOOT_CACHE_SEED" ]] || \
    die "external candidate cache seed is absent or unsafe"
  for cache_evidence in \
    "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" \
    "$CANDIDATE_BOOT_CACHE_SEED_RECORD"; do
    [[ -f "$cache_evidence" && ! -L "$cache_evidence" ]] || \
      die "external candidate cache evidence is absent or unsafe"
  done
  CANDIDATE_BOOT_CACHE_SEED_RECEIPT_SHA256="$(sha256sum \
    "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" | awk '{print $1}')"
  CANDIDATE_BOOT_CACHE_SEED_RECORD_SHA256="$(sha256sum \
    "$CANDIDATE_BOOT_CACHE_SEED_RECORD" | awk '{print $1}')"
  CANDIDATE_BOOT_CACHE_SEED_EXPECTED_FILES="$(jq -er \
    '.file_count_after_boot | select(. > 0)' \
    "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT")"
  jq -e --arg cache "$CANDIDATE_BOOT_CACHE_SEED" '
    .schema == "glm52-runtime-cache-receipt-v1" and
    .path == $cache and .started_empty == true and
    .seeded_snapshot == false and .isolated_per_run == true and
    .fresh_container_python_engine_workers_model_load == true and
    .file_count_after_boot > 0 and .byte_hashing_skipped == true and
    .cache_semantics ==
      "compiled executable code only; no model, logits, KV, RNG, or process state"
  ' "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" >/dev/null || \
    die "external candidate cache receipt differs"
  jq -e \
    --arg cache "$CANDIDATE_BOOT_CACHE_SEED" \
    --arg receipt "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" \
    --arg receipt_sha256 "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT_SHA256" '
    .docker_exit_status == 0 and
    .runtime_cache.path == $cache and
    .runtime_cache.receipt == $receipt and
    .runtime_cache.receipt_sha256 == $receipt_sha256 and
    .runtime_cache.byte_hashing_skipped == true and
    .runtime_dispatch.sqg_dispatch_proved == true and
    # This accepted record authenticates the historical compiled-code cache
    # seed produced by the original four-layer arm. It is provenance for the
    # cache source, not the treatment layers of the new evaluation.
    .runtime_dispatch.selected_layers == [6,28,52,77] and
    .per_position.positions == 2047 and
    .per_position.independently_validated == true
  ' "$CANDIDATE_BOOT_CACHE_SEED_RECORD" >/dev/null || \
    die "external candidate cache accepted record differs"
elif [[ -n "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" ||
        -n "$CANDIDATE_BOOT_CACHE_SEED_RECORD" ]]; then
  die "external candidate cache evidence was provided without a cache seed"
fi
if [[ "$RESUME_MODE" == 1 && "$SNAPSHOT_SEED_FROM_RUN1" != 1 &&
   -z "$CANDIDATE_BOOT_CACHE_SEED" ]]; then
  die "candidate continuation requires a controlled runtime-cache seed"
fi

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-sqg}"
[[ "$STAMP" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "STAMP may contain only letters, digits, dot, underscore, and dash"
OUT="$RESULTS_ROOT/$STAMP-candidate-kld-fp8-dcp4"
OUT="$(realpath -m -- "$OUT")"
reject_result_overlap() {
  local label="$1" protected="$2"
  case "$OUT" in
    "$protected"|"$protected"/*)
      die "result path overlaps protected $label: $OUT"
      ;;
  esac
}
reject_result_overlap "candidate" "$CANDIDATE"
reject_result_overlap "production model" "$PRODUCTION_MODEL"
reject_result_overlap "fresh artifact tree" "$ARTIFACTS_ROOT"
reject_result_overlap "teacher evidence" "$(dirname "$TEACHER_RECEIPT")"
reject_result_overlap "bit contract" "$(dirname "$BIT_CONTRACT")"
if [[ -n "$CANDIDATE_BOOT_CACHE_SEED" ]]; then
  case "$CANDIDATE_BOOT_CACHE_SEED" in
    "$CANDIDATE"|"$CANDIDATE"/*|"$ARTIFACTS_ROOT"|"$ARTIFACTS_ROOT"/*|\
    "$OUT"|"$OUT"/*)
      die "external compiled-code cache seed overlaps a treatment or output tree"
      ;;
  esac
fi
if [[ "$RETRY_INCOMPLETE_RUN1" == 1 ]]; then
  [[ "$RESUME_FROM_RUN" == 1 ]] || \
    die "incomplete run-1 retry requires RESUME_FROM_RUN=1"
  [[ -d "$OUT" && ! -L "$OUT" ]] || \
    die "incomplete run-1 retry output is absent or unsafe: $OUT"
  [[ ! -e "$OUT/summary.json" && ! -L "$OUT/summary.json" ]] || \
    die "refusing to retry a completed candidate output"
  [[ "$(find "$OUT" -maxdepth 1 -name 'run*-record.accepted.json' | wc -l)" \
      -eq 0 ]] || die "incomplete run-1 retry found accepted evidence"
  [[ ! -s "$OUT/runs.jsonl" ]] || \
    die "incomplete run-1 retry found nonempty runs.jsonl"
elif [[ "$RESUME_MODE" == 1 ]]; then
  [[ -d "$OUT" && ! -L "$OUT" ]] || \
    die "candidate continuation output is absent or unsafe: $OUT"
  [[ ! -e "$OUT/summary.json" && ! -L "$OUT/summary.json" ]] || \
    die "refusing to resume a completed candidate output"
else
  [[ ! -e "$OUT" ]] || die "result path already exists: $OUT"
  mkdir -p "$OUT"
fi

CONTAINER_NAME="${CONTAINER_NAME:-glm52-sqg-kld-${STAMP:0:36}}"
[[ "$CONTAINER_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] || \
  die "invalid Docker container name: $CONTAINER_NAME"
[[ "$CONTAINER_NAME" != "$PRODUCTION_CONTAINER" ]] || \
  die "candidate container name must not equal the protected production name"
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  die "refusing to adopt or remove pre-existing container: $CONTAINER_NAME"
fi

ROOT=/home/brandonmusic/klc-linux/release-gg-sparkinfer-20260721
KLD=/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current
PYDEPS="$KLD/pydeps"
PYDEPS_FILE_COUNT=7165
PYDEPS_TREE_SHA256=479cf6e600312ed06be923d644c52a5517fffed7b061bf202c1c96b256d92ca2
WIKITEXT_CACHE_SOURCE=/home/brandonmusic/.cache/huggingface/datasets/Salesforce___wikitext
WIKITEXT_CACHE_EVIDENCE="$EXPERIMENT_DIR/wikitext_cache.sha256"
WIKITEXT_CACHE_EVIDENCE_SHA256=03677042f4c8c8acfe8025345eded6cee9ab8b497542a441075f6418d6f1d249
WIKITEXT_CACHE_FILE_COUNT=5
REFERENCE="$KLD/reference_hf/reference-logits"
REFERENCE_LOGITS="$REFERENCE/logits_0.safetensors"
REFERENCE_MANIFEST="$REFERENCE/manifest.json"
REFERENCE_SHA256=87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063
REFERENCE_MANIFEST_SHA256=985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606
REFERENCE_TOKEN_IDS_U32LE_SHA256=ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56
PROMPT_LOGPROB_SHA256=47f867c3ff81cc1778bae3f1a3189dd2ec90d6e469cf60f7d1e43a1c76989d6c
LOGPROB_SHA256=21d98eea20b8c92e0a65a5badffc3782dfc5a77427470ecd5d9edcedc681a041
FALLBACK="$EXPERIMENT_DIR/prefill_kld_paired.py"
FALLBACK_SHA256=5b2dd2eed2d13d96c80370c0d3a86343dbc7900d9b6657cc0c164f8e2ebd7191
PROMPT_LOGPROB="$ROOT/eval-overlay/vllm/v1/worker/gpu/sample/prompt_logprob.py"
LOGPROB="$ROOT/eval-overlay/vllm/v1/worker/gpu/sample/logprob.py"

printf '%s  %s\n' "$REFERENCE_SHA256" "$REFERENCE_LOGITS" | sha256sum -c -
printf '%s  %s\n' "$REFERENCE_MANIFEST_SHA256" "$REFERENCE_MANIFEST" | \
  sha256sum -c -
printf '%s  %s\n' "$FALLBACK_SHA256" "$FALLBACK" | sha256sum -c -
printf '%s  %s\n' \
  "$PURE_SQG_VALIDATOR_SHA256" "$PURE_SQG_VALIDATOR" | sha256sum -c -
printf '%s  %s\n' \
  "$POSITION_VALIDATOR_SHA256" "$POSITION_VALIDATOR" | sha256sum -c -
printf '%s  %s\n' \
  "$PURE_SQG_SOURCE_EVIDENCE_SHA256" "$PURE_SQG_SOURCE_EVIDENCE" | \
  sha256sum -c -
pure_sqg_expected_sources=""
if [[ "$DIRECTIONAL_TEST_FAST" == 0 ]]; then
  [[ "$(wc -l < "$PURE_SQG_SOURCE_EVIDENCE")" -eq \
    "$PURE_SQG_SOURCE_FILE_COUNT" ]] || \
    die "pure-SQG validator source manifest census differs"
  pure_sqg_expected_sources="$(awk '{print $2}' \
    "$PURE_SQG_SOURCE_EVIDENCE" | sort)"
  pure_sqg_observed_sources="$({
    cd "$PROJECT_DIR"
    find src bmmlaw_r7_encoder -type f -name '*.py' -printf '%p\n' | sort
  })"
  [[ "$pure_sqg_observed_sources" == "$pure_sqg_expected_sources" ]] || \
    die "pure-SQG validator transitive source inventory differs"
  (
    cd "$PROJECT_DIR"
    sha256sum -c "$PURE_SQG_SOURCE_EVIDENCE" >/dev/null
  )
fi
printf '%s  %s\n' \
  "$WIKITEXT_CACHE_EVIDENCE_SHA256" "$WIKITEXT_CACHE_EVIDENCE" | \
  sha256sum -c -
[[ -d "$WIKITEXT_CACHE_SOURCE" && ! -L "$WIKITEXT_CACHE_SOURCE" ]] || \
  die "sealed WikiText cache source is absent or symlinked"
[[ -z "$(find "$WIKITEXT_CACHE_SOURCE" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) \
  -print -quit)" ]] || die "sealed WikiText cache contains an unsafe file"
[[ "$(find "$WIKITEXT_CACHE_SOURCE" -type f | wc -l)" -eq \
  "$WIKITEXT_CACHE_FILE_COUNT" ]] || \
  die "sealed WikiText cache file census differs"
(
  cd "$WIKITEXT_CACHE_SOURCE"
  sha256sum -c "$WIKITEXT_CACHE_EVIDENCE" >/dev/null
)
printf '%s  %s\n' "$PROMPT_LOGPROB_SHA256" "$PROMPT_LOGPROB" | sha256sum -c -
printf '%s  %s\n' "$LOGPROB_SHA256" "$LOGPROB" | sha256sum -c -
printf '%s  %s\n' \
  "$RUNTIME_BASELINE_SUMMARY_SHA256" "$RUNTIME_BASELINE_SUMMARY" | \
  sha256sum -c -
printf '%s  %s\n' \
  "$RUNTIME_BASELINE_EVIDENCE_SHA256" "$RUNTIME_BASELINE_EVIDENCE" | \
  sha256sum -c -
sha256sum -c "$RUNTIME_BASELINE_EVIDENCE" >/dev/null

jq -e --arg image_id "$RUNTIME_BASELINE_IMAGE_ID" '
  length == 1 and .[0].Id == $image_id
' "$RUNTIME_BASELINE_DIR/image-inspect.json" >/dev/null

jq -e --arg image "$RUNTIME_BASELINE_IMAGE_LABEL" '
  .image == $image and
  .kv_cache_dtype == "fp8" and
  .decode_context_parallel_size == 4 and
  .dcp_comm_backend == "a2a" and
  .dcp_kv_cache_interleave_size == 64 and
  .reference_sha256 ==
    "87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063" and
  .runs == 5 and .mean_kld == 0.0624498626218156 and
  .sample_sd_kld == 0.0015327574926078513
' "$RUNTIME_BASELINE_SUMMARY" >/dev/null
RUNTIME_BASELINE_MEAN="$(jq -r '.mean_kld' "$RUNTIME_BASELINE_SUMMARY")"
RUNTIME_BASELINE_SD="$(jq -r '.sample_sd_kld' "$RUNTIME_BASELINE_SUMMARY")"
RUNTIME_BASELINE_RUNS="$(jq -r '.runs' "$RUNTIME_BASELINE_SUMMARY")"

jq -e '
  .context_length == 2048 and
  (.token_first16 ==
    [284,8396,425,10960,465,284,14721,8396,
     425,10960,465,374,458,6364,4531,1154]) and
  (.windows | length == 1) and
  (.windows[0].shape == [2047,154880])
' "$REFERENCE_MANIFEST" >/dev/null

[[ -f "$PURE_SQG_VALIDATOR" ]] || \
  die "pure-SQG candidate validator is missing: $PURE_SQG_VALIDATOR"
[[ -f "$POSITION_VALIDATOR" ]] || \
  die "per-position KLD validator is missing: $POSITION_VALIDATOR"
[[ -d "$ARTIFACTS_ROOT" && -f "$RUN_SEAL" ]] || \
  die "fresh artifacts root/run seal is absent"
[[ -f "$TEACHER_RECEIPT" && -f "$BIT_CONTRACT" ]] || \
  die "teacher receipt or frozen bit contract is absent"
for required_dir in \
  "$BF16_LAYERS_SOURCE" "$CAPTURE_SOURCE" "$SQG_EXTENSION_SOURCE" \
  "$MODELS_ROOT"; do
  [[ -d "$required_dir" && ! -L "$required_dir" ]] || \
    die "sealed validator mount is absent or symlinked: $required_dir"
done
[[ -f "$SQG_EXTENSION_SOURCE/$(basename "$SQG_EXTENSION_CONTAINER")" ]] || \
  die "sealed SQG extension binary is absent"
[[ -d "$PYDEPS" && ! -L "$PYDEPS" ]] || \
  die "paired-KLD dependency tree is absent or symlinked"
[[ -z "$(find "$PYDEPS" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" ]] || \
  die "paired-KLD dependency tree contains a symlink or special file"

PATTERN=FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
[[ ${#PATTERN} -eq 78 ]] || die "invalid 78-layer index pattern"
HF_OVERRIDES="$(printf \
  '{\"use_index_cache\":true,\"index_topk_pattern\":\"%s\"}' "$PATTERN")"
LLM_EXTRA="$(jq -cn \
  --argjson dcp_size "$DCP_SIZE" \
  --arg dcp_comm_backend "$DCP_COMM_BACKEND" \
  --argjson dcp_interleave "$DCP_KV_CACHE_INTERLEAVE_SIZE" '
  {
    decode_context_parallel_size: $dcp_size,
    dcp_comm_backend: $dcp_comm_backend,
    dcp_kv_cache_interleave_size: $dcp_interleave,
    moe_backend: "b12x",
    kv_cache_memory_bytes: 268435456,
    enforce_eager: true,
    async_scheduling: false,
    disable_custom_all_reduce: true
  }')"

MODEL_MOUNTS=( -v "$CANDIDATE:/model:ro" )

RUNTIME_ARGS=()
runtime_overlay=""
extra_args_file=""
if [[ -n "${RUNTIME_OVERLAY:-}" ]]; then
  runtime_overlay="$(realpath -e -- "$RUNTIME_OVERLAY")"
  [[ -d "$runtime_overlay" ]] || \
    die "RUNTIME_OVERLAY must be a directory: $runtime_overlay"
  RUNTIME_ARGS+=(
    -v "$runtime_overlay:/sqg-runtime-overlay:ro"
    -e "PYTHONPATH=/sqg-runtime-overlay"
  )
fi
if [[ -n "${EXL3_EXT_SO:-}" ]]; then
  exl3_ext_so="$(realpath -e -- "$EXL3_EXT_SO")"
  [[ -f "$exl3_ext_so" ]] || die "EXL3_EXT_SO is not a file: $exl3_ext_so"
  RUNTIME_ARGS+=(
    -v "$exl3_ext_so:$SITE/exllamav3_ext.cpython-312-x86_64-linux-gnu.so:ro"
  )
fi
if [[ -n "${EXTRA_DOCKER_ARGS_FILE:-}" ]]; then
  extra_args_file="$(realpath -e -- "$EXTRA_DOCKER_ARGS_FILE")"
  [[ -f "$extra_args_file" ]] || \
    die "EXTRA_DOCKER_ARGS_FILE is not a regular file"
  while IFS= read -r extra_arg || [[ -n "$extra_arg" ]]; do
    [[ -z "$extra_arg" || "$extra_arg" == \#* ]] && continue
    RUNTIME_ARGS+=( "$extra_arg" )
  done < "$extra_args_file"
fi

# The sealed r33 argument file records the original 6/28/52 reservation.
# Append the candidate-specific expectation after that file: only selected
# SQG layers inside the preserved 48-layer fused allowlist consume slots.
SQG_EXPECTED_RESERVED_LAYERS=()
for eval_layer in "${SQG_EVAL_LAYER_ARRAY[@]}"; do
  if (( eval_layer == 6 || eval_layer == 7 || eval_layer == 8 ||
        (eval_layer >= 10 && eval_layer <= 54) )); then
    SQG_EXPECTED_RESERVED_LAYERS+=("$eval_layer")
  fi
done
if [[ ${#SQG_EXPECTED_RESERVED_LAYERS[@]} -eq 0 ]]; then
  SQG_EXPECTED_RESERVED_LAYERS_CSV=none
else
  SQG_EXPECTED_RESERVED_LAYERS_CSV="$(
    IFS=,; printf '%s' "${SQG_EXPECTED_RESERVED_LAYERS[*]}"
  )"
fi
RUNTIME_ARGS+=(
  --env="VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS=$SQG_EXPECTED_RESERVED_LAYERS_CSV"
)

[[ -n "$runtime_overlay" ]] || die "RUNTIME_OVERLAY is required for SQG"
[[ -z "${EXL3_EXT_SO:-}" ]] || \
  die "EXL3_EXT_SO must not bypass the sealed exact-r33 SQG overlay"
OVERLAY_MANIFEST="$runtime_overlay/SHA256SUMS.runtime-overlay"
if [[ "$SQG_TAIL_TRACE" == 1 ]]; then
  OVERLAY_MANIFEST_SHA256=92a79104c06fd53046448b339a438fce199028e22c888e7a3fdff076337bbe0a
else
  OVERLAY_MANIFEST_SHA256=1dc1b8439821ceecb459b42bf13a1efc3e44e376e8a2c5423e7b0a12ad5239d9
fi
[[ -f "$OVERLAY_MANIFEST" ]] || die "runtime overlay manifest is missing"
printf '%s  %s\n' "$OVERLAY_MANIFEST_SHA256" "$OVERLAY_MANIFEST" | \
  sha256sum -c -
(
  cd "$runtime_overlay"
  sha256sum -c SHA256SUMS.runtime-overlay >/dev/null
)
overlay_observed="$(
  cd "$runtime_overlay"
  find . -type f -printf '%p\n' | sort
)"
overlay_expected="$({
  awk '{print $2}' "$OVERLAY_MANIFEST"
  printf './SHA256SUMS.runtime-overlay\n'
} | sort)"
[[ "$overlay_observed" == "$overlay_expected" ]] || \
  die "runtime overlay exact file inventory differs"
[[ -z "$(find "$runtime_overlay" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" ]] || \
  die "runtime overlay contains a symlink or special file"
[[ ! -e "$runtime_overlay/b12x" ]] || \
  die "runtime overlay must not shadow the production b12x package"

[[ -n "$extra_args_file" ]] || \
  die "EXTRA_DOCKER_ARGS_FILE is required for exact-r33 boot parity"
R33_EXTRA_ARGS_SHA256=8ffce18aa541c5af26d20512aacaadf00898cd32d4d4a957454630a9d94d918a
printf '%s  %s\n' "$R33_EXTRA_ARGS_SHA256" "$extra_args_file" | \
  sha256sum -c -
R33_MOUNT_EVIDENCE="$EXPERIMENT_DIR/r33_exact_mounts.sha256"
R33_MOUNT_EVIDENCE_SHA256=ac0728029aff006cb0a5cce9053a5aabbac49567b86461071557a68d5ed8f3d7
printf '%s  %s\n' "$R33_MOUNT_EVIDENCE_SHA256" "$R33_MOUNT_EVIDENCE" | \
  sha256sum -c -
while read -r _ mounted_source; do
  [[ -f "$mounted_source" && ! -L "$mounted_source" ]] || \
    die "exact-r33 mounted source is absent or not a real file: $mounted_source"
done < "$R33_MOUNT_EVIDENCE"
sha256sum -c "$R33_MOUNT_EVIDENCE" >/dev/null
R33_EXL3_EXT_DIR=/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3
[[ -d "$R33_EXL3_EXT_DIR" && ! -L "$R33_EXL3_EXT_DIR" &&
   -f "$R33_EXL3_EXT_DIR/exllamav3_ext.cpython-312-x86_64-linux-gnu.so" &&
   ! -L "$R33_EXL3_EXT_DIR/exllamav3_ext.cpython-312-x86_64-linux-gnu.so" ]] || \
  die "exact-r33 ExLlamaV3 extension is absent or not a real file"
[[ "$(find "$R33_EXL3_EXT_DIR" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]] || \
  die "exact-r33 ExLlamaV3 extension directory inventory differs"

if [[ "$RESUME_MODE" == 1 ]]; then
  [[ -f "$OUT/candidate-pure-sqg-preflight.json" &&
     ! -L "$OUT/candidate-pure-sqg-preflight.json" ]] || \
    die "candidate continuation is missing the original SQG preflight"
elif [[ "$DIRECTIONAL_TEST_FAST" == 1 ]]; then
  # Materialization already performed the full validation before writing
  # .manifest_verified. Bind that marker, the candidate manifest, and the run
  # seal cheaply; runtime dispatch is proved independently on every KLD boot.
  fast_candidate_manifest_sha256="$(sha256sum \
    "$CANDIDATE/MANIFEST.json" | awk '{print $1}')"
  [[ "$(tr -d '\r\n' < "$CANDIDATE/.manifest_verified")" == \
    "$fast_candidate_manifest_sha256" ]] || \
    die "candidate verification marker does not bind MANIFEST.json"
  fast_run_seal_sha256="$(sha256sum "$RUN_SEAL" | awk '{print $1}')"
  [[ "$(sha256sum "$CANDIDATE/FRESH_SQG_RUN_SEAL.json" | awk '{print $1}')" == \
    "$fast_run_seal_sha256" ]] || die "candidate run-seal copy differs"
  fast_bit_contract_sha256="$(sha256sum "$BIT_CONTRACT" | awk '{print $1}')"
  jq -e \
    --arg source "$PRODUCTION_MODEL" \
    --arg manifest_sha256 "$fast_candidate_manifest_sha256" \
    --arg run_seal_sha256 "$fast_run_seal_sha256" \
    --arg bit_contract_sha256 "$fast_bit_contract_sha256" \
    --argjson selected_layers "$SQG_EVAL_LAYERS_JSON" \
    --argjson selected_overrides "$SQG_EVAL_OVERRIDES_JSON" '
    .schema == "glm52-fresh-sqg-four-layer-candidate-v1" and
    .complete == true and
    (.manifest_id | test("^[0-9a-f]{64}$")) and
    .source.root == $source and
    .source.source_bytes_mutated == false and
    .selected_layers == $selected_layers and
    .codebooks == {
      global_unselected:"mcg",
      selected_overrides:$selected_overrides,
      tensor_overrides:{}
    } and
    .run_seal.sha256 == $run_seal_sha256 and
    .bit_contract.sha256 == $bit_contract_sha256 and
    .runtime_allowlist.file_count > 0 and
    .model_workload_launched == false and
    .production_container_touched == false
  ' "$CANDIDATE/MANIFEST.json" >/dev/null
  jq -n \
    --arg source "$PRODUCTION_MODEL" \
    --arg candidate_manifest_sha256 "$fast_candidate_manifest_sha256" \
    --arg candidate_manifest_id "$(jq -er '.manifest_id' "$CANDIDATE/MANIFEST.json")" \
    --arg run_seal_sha256 "$fast_run_seal_sha256" \
    --arg run_seal_id "$(jq -er '.run_seal_id' "$RUN_SEAL")" \
    --argjson selected_layers "$SQG_EVAL_LAYERS_JSON" \
    --slurpfile manifest "$CANDIDATE/MANIFEST.json" '
    {
      schema:"glm52-pure-sqg-four-layer-preflight-v2",
      directional_test_fast_identity_only:true,
      protected_source:$source,
      candidate_manifest_sha256:$candidate_manifest_sha256,
      candidate_manifest_id:$candidate_manifest_id,
      run_seal_sha256:$run_seal_sha256,
      run_seal_id:$run_seal_id,
      sanitized_bit_contract:$manifest[0].bit_contract,
      construction_exclusions:$manifest[0].construction_exclusions,
      selected_layers:$selected_layers,
      global_unselected_layer_codebook:"mcg",
      selected_layer_codebook:"sqg_xor_cheb_t12",
      tensor_overrides:{},
      selected_trellis_tensors:3072,
      selected_sqg_markers:3072,
      selected_mcg_markers:0,
      sqg_markers_outside_selected_layers:0,
      marker_payloads_verified:false,
      selected_payload_hashes_verified:false,
      selected_legacy_inodes_reused:0,
      source_bytes_mutated:false,
      verified_marker_present:true,
      layers:[]
    }
  ' > "$OUT/candidate-pure-sqg-preflight.json"
else
  docker run --rm --network none --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=1g \
    --entrypoint "$PYTHON_ENTRYPOINT" --workdir /work \
    -v "$PROJECT_DIR:/work:ro" \
    -v "$BF16_LAYERS_SOURCE:$BF16_LAYERS_CONTAINER:ro" \
    -v "$CAPTURE_SOURCE:/capture:ro" \
    -v "$SQG_EXTENSION_SOURCE:/sqg-extension:ro" \
    -v "$R33_EXL3_EXT_DIR:/opt/exllamav3-r7ext:ro" \
    -v "$ARTIFACTS_ROOT:/output:ro" \
    -v "$MODELS_ROOT:$MODELS_ROOT:ro" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/opt/exllamav3-r7ext:/opt/exllamav3-python:/work \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e GIT_CONFIG_COUNT=1 \
    -e GIT_CONFIG_KEY_0=safe.directory \
    -e GIT_CONFIG_VALUE_0=/work/kquant \
    -e FRESH_SQG_RUNTIME_IMAGE_ID="$RUNTIME_BASELINE_IMAGE_ID" \
    -e FRESH_SQG_SELECTED_LAYERS="$SQG_EVAL_LAYERS" \
    -e FRESH_SQG_BIT_CONTRACT_SHA256="$(sha256sum "$BIT_CONTRACT" | awk '{print $1}')" \
    "${VALIDATOR_DYNAMIC_ENV_ARGS[@]}" \
    -e KQUANT_SQG_REQUIRE_PREBUILT=1 \
    -e KQUANT_SQG_EXTENSION_PATH="$SQG_EXTENSION_CONTAINER" \
    -e KQUANT_SQG_EXTENSION_SHA256="$SQG_EXTENSION_SHA256" \
    "$IMAGE" "$PURE_SQG_VALIDATOR_CONTAINER" "$CANDIDATE" \
    --source "$PRODUCTION_MODEL" \
    --teacher-receipt "$TEACHER_RECEIPT_CONTAINER" \
    --run-seal "$RUN_SEAL_CONTAINER" \
    --artifacts-root /output \
    --bit-contract "$BIT_CONTRACT_CONTAINER" \
    > "$OUT/candidate-pure-sqg-preflight.json"
  jq -e \
    --arg source "$PRODUCTION_MODEL" \
    --argjson selected_layers "$SQG_EVAL_LAYERS_JSON" '
  .schema == "glm52-pure-sqg-four-layer-preflight-v2" and
  .protected_source == $source and
  .selected_layers == $selected_layers and
  .global_unselected_layer_codebook == "mcg" and
  .selected_layer_codebook == "sqg_xor_cheb_t12" and
  .selected_trellis_tensors == 3072 and
  .selected_sqg_markers == 3072 and
  .selected_mcg_markers == 0 and
  .sqg_markers_outside_selected_layers == 0 and
  .marker_payloads_verified == true and
  .selected_payload_hashes_verified == true and
  .selected_legacy_inodes_reused == 0 and
  .source_bytes_mutated == false and
  .verified_marker_present == true and
  .sanitized_bit_contract.only_inherited_quantization_control ==
    "per-tensor K3/K4 assignment" and
  .construction_exclusions == {
    "mcg_payload_bytes":0,
    "mcg_transform_vectors":0,
    "mcg_scale_vectors":0,
    "mcg_permutations":0,
    "mcg_encoder_seeds":0,
    "mcg_decoded_weight_reads":0,
    "stale_per_tensor_shared_side_scales":0
  } and
  .tensor_overrides == {} and
  (.layers | length) == 4 and
    all(.layers[];
    .trellis_tensors == 768 and .sqg_markers == 768 and
    .mcg_markers == 0 and
    .bit_histogram == {"3":384,"4":384,"5":0} and
    .bit_map_matches_sanitized_contract == true)
  ' "$OUT/candidate-pure-sqg-preflight.json" >/dev/null
fi

CANDIDATE_MANIFEST_SHA256="$(jq -er '.candidate_manifest_sha256' \
  "$OUT/candidate-pure-sqg-preflight.json")"
CANDIDATE_MANIFEST_ID="$(jq -er '.candidate_manifest_id' \
  "$OUT/candidate-pure-sqg-preflight.json")"
RUN_SEAL_SHA256="$(jq -er '.run_seal_sha256' \
  "$OUT/candidate-pure-sqg-preflight.json")"
RUN_SEAL_ID="$(jq -er '.run_seal_id' \
  "$OUT/candidate-pure-sqg-preflight.json")"
BIT_CONTRACT_SHA256="$(sha256sum "$BIT_CONTRACT" | awk '{print $1}')"
TEACHER_RECEIPT_SHA256="$(sha256sum "$TEACHER_RECEIPT" | awk '{print $1}')"
jq -e \
  --arg candidate_manifest_sha256 "$CANDIDATE_MANIFEST_SHA256" \
  --arg candidate_manifest_id "$CANDIDATE_MANIFEST_ID" \
  --arg run_seal_sha256 "$RUN_SEAL_SHA256" \
  --arg run_seal_id "$RUN_SEAL_ID" \
  --arg bit_contract_sha256 "$BIT_CONTRACT_SHA256" '
  ($candidate_manifest_sha256 | test("^[0-9a-f]{64}$")) and
  ($candidate_manifest_id | test("^[0-9a-f]{64}$")) and
  ($run_seal_sha256 | test("^[0-9a-f]{64}$")) and
  ($run_seal_id | test("^[0-9a-f]{64}$")) and
  (.sanitized_bit_contract.sha256 == $bit_contract_sha256)
' "$OUT/candidate-pure-sqg-preflight.json" >/dev/null
[[ "$(sha256sum "$CANDIDATE/MANIFEST.json" | awk '{print $1}')" == \
  "$CANDIDATE_MANIFEST_SHA256" ]] || die "candidate manifest/preflight binding differs"
[[ "$(sha256sum "$RUN_SEAL" | awk '{print $1}')" == "$RUN_SEAL_SHA256" ]] || \
  die "external run-seal/preflight binding differs"
[[ "$(sha256sum "$CANDIDATE/FRESH_SQG_RUN_SEAL.json" | awk '{print $1}')" == \
  "$RUN_SEAL_SHA256" ]] || die "candidate run-seal copy differs"

CANDIDATE_RUNTIME_EVIDENCE="$OUT/candidate-runtime-files.sha256"
if [[ "$RESUME_MODE" == 1 ]]; then
  [[ -f "$CANDIDATE_RUNTIME_EVIDENCE" &&
     ! -L "$CANDIDATE_RUNTIME_EVIDENCE" ]] || \
    die "candidate continuation lacks original runtime evidence"
else
  : > "$CANDIDATE_RUNTIME_EVIDENCE"
  while IFS=$'\t' read -r candidate_sha256 candidate_name; do
    [[ "$candidate_sha256" =~ ^[0-9a-f]{64}$ ]] || \
      die "candidate runtime manifest contains an invalid SHA256"
    [[ -n "$candidate_name" && "$(basename -- "$candidate_name")" == \
      "$candidate_name" ]] || die "candidate runtime manifest contains an unsafe name"
    printf '%s  %s\n' \
      "$candidate_sha256" "$CANDIDATE/$candidate_name" >> \
      "$CANDIDATE_RUNTIME_EVIDENCE"
  done < <(jq -r '
    .runtime_allowlist.files | to_entries | sort_by(.key)[] |
    [.value.sha256, .key] | @tsv
  ' "$CANDIDATE/MANIFEST.json")
fi
CANDIDATE_RUNTIME_FILE_COUNT="$(wc -l < "$CANDIDATE_RUNTIME_EVIDENCE")"
[[ "$CANDIDATE_RUNTIME_FILE_COUNT" -gt 0 ]] || \
  die "candidate runtime allowlist is empty"
if [[ "$DIRECTIONAL_TEST_FAST" == 0 ]]; then
  sha256sum -c "$CANDIDATE_RUNTIME_EVIDENCE" >/dev/null
fi

SELECTED_TREATMENT_FILES=()
SELECTED_TREATMENT_SHA256=()
for sqg_layer in "${SQG_EVAL_LAYER_ARRAY[@]}"; do
  for selected_name in \
    "r7-experts-layer-$(printf '%03d' "$sqg_layer").safetensors" \
    "r7-experts-layer-$(printf '%03d' "$sqg_layer").json"; do
    selected_file="$CANDIDATE/$selected_name"
    selected_sha256="$(jq -er --arg name "$selected_name" '
      .runtime_allowlist.files[$name].sha256 |
      select(test("^[0-9a-f]{64}$"))
    ' "$CANDIDATE/MANIFEST.json")"
    [[ -f "$selected_file" && ! -L "$selected_file" ]] || \
      die "selected treatment file is absent or symlinked: $selected_name"
    if [[ "$DIRECTIONAL_TEST_FAST" == 0 ]]; then
      [[ "$(sha256sum "$selected_file" | awk '{print $1}')" == \
        "$selected_sha256" ]] || die "selected treatment file hash differs"
    fi
    SELECTED_TREATMENT_FILES+=( "$selected_file" )
    SELECTED_TREATMENT_SHA256+=( "$selected_sha256" )
  done
done
if [[ "$RESUME_MODE" == 1 ]]; then
  [[ -f "$OUT/selected-treatment-files.sha256" &&
     ! -L "$OUT/selected-treatment-files.sha256" ]] || \
    die "candidate continuation lacks original selected-treatment evidence"
  [[ "$(wc -l < "$OUT/selected-treatment-files.sha256")" -eq 8 ]] || \
    die "candidate continuation selected-treatment census differs"
else
  : > "$OUT/selected-treatment-files.sha256"
  for index in "${!SELECTED_TREATMENT_FILES[@]}"; do
    printf '%s  %s\n' \
      "${SELECTED_TREATMENT_SHA256[$index]}" \
      "${SELECTED_TREATMENT_FILES[$index]}" >> \
      "$OUT/selected-treatment-files.sha256"
  done
fi
SELECTED_TREATMENT_EVIDENCE_SHA256="$(sha256sum \
  "$OUT/selected-treatment-files.sha256" | awk '{print $1}')"

if [[ "$RESUME_MODE" == 1 ]]; then
  for preserved_evidence in \
    image-inspect.json gpu-preflight.csv gpu-compute-preflight.csv \
    candidate-path.txt candidate-metadata.sha256 \
    sealed-construction-inputs.sha256 eval-code.sha256 \
    candidate-pure-sqg-preflight.sha256 \
    runtime-baseline-evidence-manifest.sha256 \
    exact-r33-mount-evidence-manifest.sha256 runtime-overlay-path.txt \
    runtime-overlay-files.sha256 runtime-overlay-manifest.sha256 \
    extra-docker-args-path.txt extra-docker-args.sha256 \
    pydeps-files.sha256 pydeps-manifest.sha256; do
    [[ -f "$OUT/$preserved_evidence" && ! -L "$OUT/$preserved_evidence" ]] || \
      die "candidate continuation lacks original evidence: $preserved_evidence"
  done
  sha256sum "$RUNNER" "$POSITION_VALIDATOR" "$FALLBACK" > \
    "$OUT/resume-eval-code.sha256"
else
docker image inspect "$IMAGE" > "$OUT/image-inspect.json"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "$OUT/gpu-preflight.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "$OUT/gpu-compute-preflight.csv"
printf '%s\n' "$CANDIDATE" > "$OUT/candidate-path.txt"
sha256sum \
  "$CANDIDATE/.manifest_verified" \
  "$CANDIDATE/MANIFEST.json" \
  "$CANDIDATE/FRESH_SQG_RUN_SEAL.json" \
  "$CANDIDATE/config.json" \
  "$CANDIDATE/quantization_config.json" \
  "$CANDIDATE/model.safetensors.index.json" \
  > "$OUT/candidate-metadata.sha256"
sha256sum "$TEACHER_RECEIPT" "$RUN_SEAL" "$BIT_CONTRACT" > \
  "$OUT/sealed-construction-inputs.sha256"
sha256sum "$RUNNER" "$PURE_SQG_VALIDATOR" "$POSITION_VALIDATOR" \
  "$PURE_SQG_SOURCE_EVIDENCE" "$FALLBACK" "$PROMPT_LOGPROB" "$LOGPROB" > \
  "$OUT/eval-code.sha256"
sha256sum "$OUT/candidate-pure-sqg-preflight.json" > \
  "$OUT/candidate-pure-sqg-preflight.sha256"
CANDIDATE_PREFLIGHT_SHA256="$(sha256sum \
  "$OUT/candidate-pure-sqg-preflight.json" | awk '{print $1}')"
sha256sum "$RUNTIME_BASELINE_EVIDENCE" > \
  "$OUT/runtime-baseline-evidence-manifest.sha256"
sha256sum "$R33_MOUNT_EVIDENCE" > \
  "$OUT/exact-r33-mount-evidence-manifest.sha256"
if [[ -n "$runtime_overlay" ]]; then
  printf '%s\n' "$runtime_overlay" > "$OUT/runtime-overlay-path.txt"
  (
    cd "$runtime_overlay"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "$OUT/runtime-overlay-files.sha256"
  sha256sum "$OUT/runtime-overlay-files.sha256" > \
    "$OUT/runtime-overlay-manifest.sha256"
fi
if [[ -n "$extra_args_file" ]]; then
  printf '%s\n' "$extra_args_file" > "$OUT/extra-docker-args-path.txt"
  sha256sum "$extra_args_file" > "$OUT/extra-docker-args.sha256"
fi
(
  cd "$PYDEPS"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$OUT/pydeps-files.sha256"
[[ "$(wc -l < "$OUT/pydeps-files.sha256")" -eq "$PYDEPS_FILE_COUNT" ]] || \
  die "paired-KLD dependency file census differs"
[[ "$(sha256sum "$OUT/pydeps-files.sha256" | awk '{print $1}')" == \
  "$PYDEPS_TREE_SHA256" ]] || die "paired-KLD dependency tree hash differs"
sha256sum "$OUT/pydeps-files.sha256" > \
  "$OUT/pydeps-manifest.sha256"
fi
CANDIDATE_PREFLIGHT_SHA256="$(sha256sum \
  "$OUT/candidate-pure-sqg-preflight.json" | awk '{print $1}')"

verify_sealed_eval_inputs() {
  local current_overlay index mounted_source
  if [[ -n "$CANDIDATE_BOOT_CACHE_SEED" ]]; then
    [[ "$(sha256sum "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" | awk '{print $1}')" == \
      "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT_SHA256" ]] || \
      die "external candidate cache receipt changed after preflight"
    [[ "$(sha256sum "$CANDIDATE_BOOT_CACHE_SEED_RECORD" | awk '{print $1}')" == \
      "$CANDIDATE_BOOT_CACHE_SEED_RECORD_SHA256" ]] || \
      die "external candidate cache accepted record changed after preflight"
    [[ -z "$(sudo -n find "$CANDIDATE_BOOT_CACHE_SEED" \
      \( -type l -o -type b -o -type c -o -type p -o -type s \) \
      -print -quit)" ]] || \
      die "external candidate cache seed changed type after preflight"
    [[ "$(sudo -n find "$CANDIDATE_BOOT_CACHE_SEED" \
      -type f -printf '.' | wc -c)" -eq \
      "$CANDIDATE_BOOT_CACHE_SEED_EXPECTED_FILES" ]] || \
      die "external candidate cache seed census changed after preflight"
  fi
  if [[ "$DIRECTIONAL_TEST_FAST" == 1 ]]; then
    # The inputs are mounted read-only for every boot. Recheck cheap identity
    # metadata and file presence, but do not reread hundreds of GB between
    # boots. The saved BF16 logits were hashed once above.
    sha256sum -c "$OUT/candidate-metadata.sha256" >/dev/null
    printf '%s  %s\n' "$REFERENCE_MANIFEST_SHA256" "$REFERENCE_MANIFEST" | \
      sha256sum -c - >/dev/null
    for index in "${!SELECTED_TREATMENT_FILES[@]}"; do
      [[ -f "${SELECTED_TREATMENT_FILES[$index]}" &&
         ! -L "${SELECTED_TREATMENT_FILES[$index]}" ]] || \
        die "selected treatment file changed type after preflight"
    done
    [[ "$(docker inspect --format '{{.State.Running}}' \
      "$PRODUCTION_CONTAINER" 2>/dev/null || true)" != "true" ]] || \
      die "production container restarted during candidate KLD"
    return 0
  fi
  printf '%s  %s\n' "$REFERENCE_SHA256" "$REFERENCE_LOGITS" | \
    sha256sum -c - >/dev/null
  printf '%s  %s\n' "$REFERENCE_MANIFEST_SHA256" "$REFERENCE_MANIFEST" | \
    sha256sum -c - >/dev/null
  sha256sum -c "$OUT/candidate-metadata.sha256" >/dev/null
  [[ "$(wc -l < "$CANDIDATE_RUNTIME_EVIDENCE")" -eq \
    "$CANDIDATE_RUNTIME_FILE_COUNT" ]] || \
    die "candidate runtime evidence census changed after preflight"
  sha256sum -c "$CANDIDATE_RUNTIME_EVIDENCE" >/dev/null
  sha256sum -c "$OUT/sealed-construction-inputs.sha256" >/dev/null
  sha256sum -c "$OUT/eval-code.sha256" >/dev/null
  printf '%s  %s\n' \
    "$PURE_SQG_SOURCE_EVIDENCE_SHA256" "$PURE_SQG_SOURCE_EVIDENCE" | \
    sha256sum -c - >/dev/null
  [[ "$({
    cd "$PROJECT_DIR"
    find src bmmlaw_r7_encoder -type f -name '*.py' -printf '%p\n' | sort
  })" == "$pure_sqg_expected_sources" ]] || \
    die "pure-SQG validator source inventory changed after preflight"
  (
    cd "$PROJECT_DIR"
    sha256sum -c "$PURE_SQG_SOURCE_EVIDENCE" >/dev/null
  )
  printf '%s  %s\n' "$OVERLAY_MANIFEST_SHA256" "$OVERLAY_MANIFEST" | \
    sha256sum -c - >/dev/null
  (
    cd "$runtime_overlay"
    sha256sum -c SHA256SUMS.runtime-overlay >/dev/null
  )
  current_overlay="$(
    cd "$runtime_overlay"
    find . -type f -printf '%p\n' | sort
  )"
  [[ "$current_overlay" == "$overlay_expected" ]] || \
    die "runtime overlay inventory changed after preflight"
  [[ -z "$(find "$runtime_overlay" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" ]] || \
    die "runtime overlay gained a symlink or special file"
  printf '%s  %s\n' "$R33_EXTRA_ARGS_SHA256" "$extra_args_file" | \
    sha256sum -c - >/dev/null
  printf '%s  %s\n' "$R33_MOUNT_EVIDENCE_SHA256" "$R33_MOUNT_EVIDENCE" | \
    sha256sum -c - >/dev/null
  while read -r _ mounted_source; do
    [[ -f "$mounted_source" && ! -L "$mounted_source" ]] || \
      die "exact-r33 mounted source changed type after preflight"
  done < "$R33_MOUNT_EVIDENCE"
  sha256sum -c "$R33_MOUNT_EVIDENCE" >/dev/null
  [[ "$(find "$R33_EXL3_EXT_DIR" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]] || \
    die "exact-r33 extension directory changed after preflight"
  [[ "$(find "$PYDEPS" -type f | wc -l)" -eq "$PYDEPS_FILE_COUNT" ]] || \
    die "paired-KLD dependency census changed after preflight"
  [[ -z "$(find "$PYDEPS" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" ]] || \
    die "paired-KLD dependency tree changed type after preflight"
  (
    cd "$PYDEPS"
    sha256sum -c "$OUT/pydeps-files.sha256" >/dev/null
  )
  printf '%s  %s\n' \
    "$WIKITEXT_CACHE_EVIDENCE_SHA256" "$WIKITEXT_CACHE_EVIDENCE" | \
    sha256sum -c - >/dev/null
  [[ "$(find "$WIKITEXT_CACHE_SOURCE" -type f | wc -l)" -eq \
    "$WIKITEXT_CACHE_FILE_COUNT" ]] || \
    die "sealed WikiText cache census changed after preflight"
  (
    cd "$WIKITEXT_CACHE_SOURCE"
    sha256sum -c "$WIKITEXT_CACHE_EVIDENCE" >/dev/null
  )
  for index in "${!SELECTED_TREATMENT_FILES[@]}"; do
    [[ -f "${SELECTED_TREATMENT_FILES[$index]}" &&
       ! -L "${SELECTED_TREATMENT_FILES[$index]}" ]] || \
      die "selected treatment file changed type after preflight"
    [[ "$(sha256sum "${SELECTED_TREATMENT_FILES[$index]}" | awk '{print $1}')" == \
      "${SELECTED_TREATMENT_SHA256[$index]}" ]] || \
      die "selected treatment file changed after preflight"
  done
}
verify_sealed_eval_inputs

jq -n \
  --arg candidate "$CANDIDATE" \
  --arg runtime_image "$IMAGE" \
  --arg runtime_image_id "$RUNTIME_IMAGE_ID" \
  --arg baseline_image "$RUNTIME_BASELINE_IMAGE_LABEL" \
  --arg baseline_image_id "$RUNTIME_BASELINE_IMAGE_ID" \
  --arg baseline_summary "$RUNTIME_BASELINE_SUMMARY" \
  --argjson baseline_mean "$RUNTIME_BASELINE_MEAN" \
  --argjson baseline_sd "$RUNTIME_BASELINE_SD" \
  --argjson baseline_runs "$RUNTIME_BASELINE_RUNS" \
  --arg legacy_image "$LEGACY_BASELINE_IMAGE_LABEL" \
  --arg legacy_image_id "$LEGACY_BASELINE_IMAGE_ID" \
  --argjson legacy_mean "$LEGACY_BASELINE_MEAN" \
  --argjson legacy_sd "$LEGACY_BASELINE_SD" \
  --arg runtime_overlay "$runtime_overlay" \
  --arg overlay_manifest_sha256 "$OVERLAY_MANIFEST_SHA256" \
  --arg extra_args_file "$extra_args_file" \
  --arg extra_args_sha256 "$R33_EXTRA_ARGS_SHA256" \
  --arg exact_mount_evidence "$R33_MOUNT_EVIDENCE" \
  --arg exact_mount_evidence_sha256 "$R33_MOUNT_EVIDENCE_SHA256" \
  --arg pydeps "$PYDEPS" \
  --arg pydeps_tree_sha256 "$PYDEPS_TREE_SHA256" \
  --argjson pydeps_file_count "$PYDEPS_FILE_COUNT" \
  --arg baseline_evidence_manifest "$RUNTIME_BASELINE_EVIDENCE" \
  --arg baseline_evidence_sha256 "$RUNTIME_BASELINE_EVIDENCE_SHA256" \
  --arg reference_sha256 "$REFERENCE_SHA256" \
  --arg reference_token_ids_u32le_sha256 "$REFERENCE_TOKEN_IDS_U32LE_SHA256" \
  --arg candidate_manifest_sha256 "$CANDIDATE_MANIFEST_SHA256" \
  --arg candidate_manifest_id "$CANDIDATE_MANIFEST_ID" \
  --arg candidate_preflight "$OUT/candidate-pure-sqg-preflight.json" \
  --arg candidate_preflight_sha256 "$CANDIDATE_PREFLIGHT_SHA256" \
  --arg selected_treatment_evidence "$OUT/selected-treatment-files.sha256" \
  --arg selected_treatment_evidence_sha256 "$SELECTED_TREATMENT_EVIDENCE_SHA256" \
  --arg run_seal "$RUN_SEAL" \
  --arg run_seal_sha256 "$RUN_SEAL_SHA256" \
  --arg run_seal_id "$RUN_SEAL_ID" \
  --arg bit_contract "$BIT_CONTRACT" \
  --arg bit_contract_sha256 "$BIT_CONTRACT_SHA256" \
  --arg teacher_receipt "$TEACHER_RECEIPT" \
  --arg teacher_receipt_sha256 "$TEACHER_RECEIPT_SHA256" \
  --arg boot_cache_seed "$CANDIDATE_BOOT_CACHE_SEED" \
  --arg boot_cache_seed_receipt "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" \
  --arg boot_cache_seed_receipt_sha256 \
    "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT_SHA256" \
  --arg boot_cache_seed_record "$CANDIDATE_BOOT_CACHE_SEED_RECORD" \
  --arg boot_cache_seed_record_sha256 \
    "$CANDIDATE_BOOT_CACHE_SEED_RECORD_SHA256" \
  --argjson boot_cache_seed_expected_files \
    "$CANDIDATE_BOOT_CACHE_SEED_EXPECTED_FILES" \
  --argjson directional_test_fast "$DIRECTIONAL_TEST_FAST_JSON" \
  --argjson runs "$RUNS" \
  --argjson kld_util "$KLD_UTIL" \
  --argjson selected_layers "$SQG_EVAL_LAYERS_JSON" \
  '{
    schema: "glm52-fresh-sqg-candidate-kld-run-v2",
    directional_test_fast_mode: $directional_test_fast,
    baseline_was_rerun: false,
    candidate: $candidate,
    runtime_image: $runtime_image,
    runtime_image_id: $runtime_image_id,
    baselines_were_rerun: false,
    primary_same_base_image_baseline: {
      summary: $baseline_summary,
      image: $baseline_image,
      image_id: $baseline_image_id,
      runs: $baseline_runs,
      mean_kld: $baseline_mean,
      sample_sd_kld: $baseline_sd
    },
    secondary_legacy_baseline: {
      image: $legacy_image,
      image_id: $legacy_image_id,
      runs: 5,
      mean_kld: $legacy_mean,
      sample_sd_kld: $legacy_sd
    },
    runtime_base_image_matches_primary_baseline:
      ($runtime_image_id == $baseline_image_id),
    candidate_runtime_overlay_applied: true,
    runtime_overlay: {
      path: $runtime_overlay,
      manifest_sha256: $overlay_manifest_sha256
    },
    exact_r33_extension_directory_mount_applied: true,
    direct_exl3_shared_object_override_applied: false,
    extension_matches_baseline: true,
    exact_r33_extra_docker_args: {
      path: $extra_args_file,
      sha256: $extra_args_sha256
    },
    exact_r33_mount_sources: {
      manifest: $exact_mount_evidence,
      manifest_sha256: $exact_mount_evidence_sha256,
      all_bytes_validated: true,
      extension_directory_exact_single_file: true
    },
    paired_kld_dependencies: {
      path: $pydeps,
      file_count: $pydeps_file_count,
      tree_sha256: $pydeps_tree_sha256,
      all_bytes_validated: true,
      symlinks_or_special_files: 0,
      network_fetch_disabled: true
    },
    baseline_evidence: {
      manifest: $baseline_evidence_manifest,
      sha256: $baseline_evidence_sha256
    },
    runtime_cache_seed: {
      enabled: ($boot_cache_seed != ""),
      source: $boot_cache_seed,
      source_receipt: $boot_cache_seed_receipt,
      source_receipt_sha256: $boot_cache_seed_receipt_sha256,
      source_accepted_record: $boot_cache_seed_record,
      source_accepted_record_sha256: $boot_cache_seed_record_sha256,
      expected_file_count: $boot_cache_seed_expected_files,
      destination_policy:
        "independent snapshot per boot; never mount source writable",
      cache_semantics:
        "compiled executable code only; no model, logits, KV, RNG, or process state",
      cache_payload_byte_hashing_skipped: true,
      fresh_container_python_engine_workers_model_load_every_boot: true
    },
    runtime_code_matches_baseline: false,
    comparison_scope:
      "Fresh SQG calibration and encoding plus required SQG loader/non-fused dispatch; selected-layer MCG lineage is forbidden",
    candidate_construction: {
      manifest_sha256: $candidate_manifest_sha256,
      manifest_id: $candidate_manifest_id,
      preflight_report: $candidate_preflight,
      preflight_report_sha256: $candidate_preflight_sha256,
      selected_treatment_evidence: $selected_treatment_evidence,
      selected_treatment_evidence_sha256:
        $selected_treatment_evidence_sha256,
      selected_treatment_files_rehashed_before_every_launch:
        ($directional_test_fast | not),
      external_run_seal: $run_seal,
      run_seal_sha256: $run_seal_sha256,
      run_seal_id: $run_seal_id,
      sanitized_bit_contract: $bit_contract,
      sanitized_bit_contract_sha256: $bit_contract_sha256,
      teacher_receipt: $teacher_receipt,
      teacher_receipt_sha256: $teacher_receipt_sha256,
      only_inherited_selected_layer_quantization_control:
        "per-tensor K3/K4 assignment",
      legacy_mcg_payloads_transforms_scales_permutations_seeds: 0
    },
    selected_sqg_layers: $selected_layers,
    selected_layer_payloads: {
      sqg: 3072,
      mcg: 0,
      tensor_overrides: 0
    },
    reference_sha256: $reference_sha256,
    reference_token_ids_u32le_sha256: $reference_token_ids_u32le_sha256,
    regime: {
      kv_cache_dtype: "fp8",
      rope: "bfloat16",
      tensor_parallel_size: 4,
      decode_context_parallel_size: 4,
      dcp_comm_backend: "a2a",
      dcp_kv_cache_interleave_size: 64,
      context_tokens: 2048,
      scored_positions: 2047,
      gpu_memory_utilization: $kld_util
    },
    paired_per_position_kld_required: true,
    requested_candidate_runs: $runs,
    inference_limit:
      "One fixed 2047-position prompt; repeat runs estimate runtime variation, not text/model generalization"
  }' > "$OUT/run-manifest.json"
: > "$OUT/runs.jsonl"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  jq -e '.selected_layer_payloads.mcg == 0 and
    .runtime_code_matches_baseline == false and
    .paired_per_position_kld_required == true and
    .candidate_construction.legacy_mcg_payloads_transforms_scales_permutations_seeds == 0' \
    "$OUT/run-manifest.json" >/dev/null
  printf '%s\n' "$OUT"
  exit 0
fi

ACCEPTED_RECORDS=()
if [[ "$RESUME_MODE" == 1 ]]; then
  for previous_run in $(seq 1 $((RESUME_FROM_RUN - 1))); do
    previous_record="$OUT/run${previous_run}-record.accepted.json"
    [[ -f "$previous_record" && ! -L "$previous_record" ]] || \
      die "candidate run $previous_run must be accepted before continuation"
    jq -e \
      --arg external_seed "$CANDIDATE_BOOT_CACHE_SEED" \
      --argjson selected_layers "$SQG_EVAL_LAYERS_JSON" '
      .docker_exit_status == 0 and
      .total_positions == 2047 and .mean_kld >= 0 and
      .runtime_dispatch.sqg_dispatch_proved == true and
      .runtime_dispatch.selected_layers == $selected_layers and
      .per_position.positions == 2047 and
      .per_position.independently_validated == true and
      .runtime_cache.isolated_per_run == true and
      .runtime_cache.fresh_container_python_engine_workers_model_load == true and
      .runtime_cache.byte_hashing_skipped == true and
      (if $external_seed == "" then
         (.runtime_cache.started_empty == true or
          .runtime_cache.seeded_snapshot == true)
       else
         .runtime_cache.started_empty == false and
         .runtime_cache.seeded_snapshot == true and
         .runtime_cache.source == $external_seed
       end)
    ' "$previous_record" >/dev/null || \
      die "candidate run $previous_run accepted record is not continuation-safe"
    ACCEPTED_RECORDS+=( "$previous_record" )
  done
  [[ -f "$OUT/per-position-evidence.sha256" &&
     "$(wc -l < "$OUT/per-position-evidence.sha256")" -eq \
       $(((RESUME_FROM_RUN - 1) * 5)) ]] || \
    die "candidate accepted-run evidence census differs before continuation"
  sha256sum -c "$OUT/per-position-evidence.sha256" >/dev/null
  jq -n --arg output "$OUT" \
    --arg runner "$RUNNER" --arg resume_code "$OUT/resume-eval-code.sha256" \
    --arg cache_seed "$CANDIDATE_BOOT_CACHE_SEED" \
    --argjson resume_from "$RESUME_FROM_RUN" '
    {
      schema:"glm52-candidate-kld-cached-continuation-v1",
      output:$output,resume_from_run:$resume_from,final_expected_runs:5,
      preserved_completed_runs:($resume_from - 1),
      cache_seed:(if $cache_seed == "" then
        "independent snapshots of finalized candidate run-1 compiled-code cache"
        else $cache_seed end),
      byte_neutral_runtime_cache_hashing_skipped:true,
      fresh_container_python_engine_workers_model_load_every_boot:true,
      runner:$runner,resume_code:$resume_code
    }
  ' > "$OUT/resume-manifest.json"
fi
ACTIVE_CIDFILE=""
cleanup_owned_container() {
  local owned_cid current_cid
  [[ -n "$ACTIVE_CIDFILE" ]] || return 0
  if [[ ! -f "$ACTIVE_CIDFILE" || -L "$ACTIVE_CIDFILE" ]]; then
    ACTIVE_CIDFILE=""
    return 0
  fi
  owned_cid="$(tr -d '\r\n' < "$ACTIVE_CIDFILE")"
  if [[ ! "$owned_cid" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'WARNING: invalid runner-owned Docker cidfile: %s\n' \
      "$ACTIVE_CIDFILE" >&2
    ACTIVE_CIDFILE=""
    return 0
  fi
  current_cid="$(docker container inspect --format '{{.Id}}' \
    "$owned_cid" 2>/dev/null || true)"
  if [[ "$current_cid" == "$owned_cid" ]]; then
    docker rm -f "$owned_cid" >/dev/null 2>&1 || true
  fi
  ACTIVE_CIDFILE=""
}
normalize_container_artifact_permissions() {
  sudo -n chmod -R u+rwX,go+rX "$@" ||
    die "could not make container artifacts readable for host-side evidence recording"
}
cache_file_count_cheap() {
  sudo -n find "$1" -type f -printf '.' | wc -c
}
snapshot_runtime_cache() {
  local source_cache="$1" destination_cache="$2" receipt="$3"
  local source_count destination_count
  [[ -d "$source_cache" && ! -L "$source_cache" ]] || \
    die "runtime-cache seed is absent or unsafe: $source_cache"
  [[ -d "$destination_cache" && ! -L "$destination_cache" ]] || \
    die "runtime-cache destination is absent or unsafe: $destination_cache"
  [[ -z "$(sudo -n find "$source_cache" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \) \
    -print -quit)" ]] || die "runtime-cache seed contains a special file"
  source_count="$(cache_file_count_cheap "$source_cache")"
  [[ "$source_count" -gt 0 ]] || die "runtime-cache seed is empty"
  sudo -n cp -a --reflink=auto -- "$source_cache/." "$destination_cache/" || \
    die "could not snapshot the runtime-cache seed"
  normalize_container_artifact_permissions "$destination_cache"
  destination_count="$(cache_file_count_cheap "$destination_cache")"
  [[ "$destination_count" -eq "$source_count" ]] || \
    die "runtime-cache snapshot file census differs"
  jq -n \
    --arg source "$source_cache" --arg destination "$destination_cache" \
    --argjson source_files "$source_count" \
    --argjson destination_files "$destination_count" '
    {
      schema:"glm52-runtime-cache-seed-receipt-v1",
      source:$source,destination:$destination,
      copy_request:"cp -a --reflink=auto",
      source_file_count:$source_files,
      destination_file_count_before_boot:$destination_files,
      byte_hashing_skipped:true,
      cache_semantics:"compiled executable code only; no model, logits, KV, RNG, or process state",
      independently_mutable_destination:true
    }
  ' > "$receipt"
}
trap cleanup_owned_container EXIT
trap 'exit 130' INT TERM

for run in $(seq 1 "$RUNS"); do
  [[ "$run" -ge "$RESUME_FROM_RUN" ]] || continue
  verify_sealed_eval_inputs
  require_idle_gpus
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    die "candidate container name became occupied: $CONTAINER_NAME"
  fi
  run_output="$OUT/run${run}-container-output"
  run_cache="$OUT/run${run}-candidate-runtime-cache"
  run_hf_cache="$OUT/run${run}-candidate-hf-cache"
  run_cidfile="$OUT/run${run}-candidate-container.cid"
  mkdir -m 0700 "$run_output" "$run_cache" "$run_hf_cache"
  cache_seed_receipt="$OUT/run${run}-runtime-cache-seed-receipt.json"
  if [[ -n "$CANDIDATE_BOOT_CACHE_SEED" ]]; then
    snapshot_runtime_cache \
      "$CANDIDATE_BOOT_CACHE_SEED" "$run_cache" "$cache_seed_receipt"
    jq \
      --arg source_receipt "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" \
      --arg source_receipt_sha256 \
        "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT_SHA256" \
      --arg source_record "$CANDIDATE_BOOT_CACHE_SEED_RECORD" \
      --arg source_record_sha256 \
        "$CANDIDATE_BOOT_CACHE_SEED_RECORD_SHA256" \
      --argjson expected_files "$CANDIDATE_BOOT_CACHE_SEED_EXPECTED_FILES" '
      if .source_file_count != $expected_files then
        error("external candidate cache seed census differs")
      else . + {
        source_arm:"rejected_sqg_candidate_compiled_code_cache",
        source_receipt:$source_receipt,
        source_receipt_sha256:$source_receipt_sha256,
        source_accepted_record:$source_record,
        source_accepted_record_sha256:$source_record_sha256
      } end
    ' "$cache_seed_receipt" > "$cache_seed_receipt.partial"
    mv "$cache_seed_receipt.partial" "$cache_seed_receipt"
  elif [[ "$SNAPSHOT_SEED_FROM_RUN1" == 1 && "$run" -ge 2 ]]; then
    snapshot_runtime_cache \
      "$OUT/run1-candidate-runtime-cache" "$run_cache" "$cache_seed_receipt"
  fi
  cp -a --reflink=auto \
    "$WIKITEXT_CACHE_SOURCE" \
    "$run_hf_cache/Salesforce___wikitext"
  (
    cd "$run_hf_cache/Salesforce___wikitext"
    sha256sum -c "$WIKITEXT_CACHE_EVIDENCE" >/dev/null
  )
  [[ ! -e "$run_cidfile" && ! -L "$run_cidfile" ]] || \
    die "candidate run $run cidfile already exists"
  ACTIVE_CIDFILE="$run_cidfile"
  position_name="run${run}-position-kld.safetensors"
  position_container="/results/$position_name"
  position_host="$run_output/$position_name"
  position_validation="$OUT/run${run}-position-kld.validation.json"
  raw_record="$OUT/run${run}-record.raw.json"
  accepted_record="$OUT/run${run}-record.accepted.json"
  [[ ! -e "$position_host" && ! -e "$position_validation" ]] || \
    die "candidate run $run per-position destination already exists"
  set +e
  docker run --rm --name "$CONTAINER_NAME" --cidfile "$run_cidfile" \
    --gpus all --runtime nvidia --ipc host --network host --shm-size 64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${MODEL_MOUNTS[@]}" \
    -v "$run_output:/results:rw" \
    -v "$REFERENCE:/ref:ro" \
    -v "$FALLBACK:/kld/prefill_kld.py:ro" \
    -v "$PYDEPS:/deps:ro" \
    -v "$PROMPT_LOGPROB:$SITE/vllm/v1/worker/gpu/sample/prompt_logprob.py:ro" \
    -v "$LOGPROB:$SITE/vllm/v1/worker/gpu/sample/logprob.py:ro" \
    -v /home/brandonmusic/.cache/huggingface:/root/.cache/huggingface:ro \
    -v "$run_hf_cache:/hf-datasets:rw" \
    -v "$run_cache:/cache:rw" \
    -v /home/brandonmusic/klc-linux/ckv140k-test/assets:/opt/kv-scale:ro \
    -e CUDA_VISIBLE_DEVICES=3,1,2,0 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e NCCL_NVLS_ENABLE=0 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
    -e CUTE_DSL_ARCH=sm_120a \
    -e TORCH_CUDA_ARCH_LIST=12.0a \
    -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e OMP_NUM_THREADS=16 \
    -e KLD_PYDEPS=/deps \
    -e "SQG_TAIL_TRACE=$SQG_TAIL_TRACE" \
    -e "SQG_TAIL_TRACE_LAYERS=$SQG_TAIL_TRACE_LAYERS" \
    -e SQG_TAIL_TRACE_DIR=/results/tail-trace \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_DATASETS_CACHE=/hf-datasets \
    -e HF_HUB_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_FASTSAFETENSORS_QUEUE_SIZE=-1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_B12X_MOE=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_USE_B12X_DCP_A2A=1 \
    -e VLLM_DCP_A2A_MAX_TOKENS=16 \
    -e VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
    -e VLLM_DCP_GLOBAL_TOPK=1 \
    -e VLLM_DCP_QUERY_SPLIT=0 \
    -e VLLM_B12X_MLA_CKV_GATHER=1 \
    -e VLLM_USE_B12X_WO_PROJECTION=1 \
    -e VLLM_USE_B12X_MHC=1 \
    -e B12X_MLA_SM120_UNIFIED=1 \
    -e B12X_DENSE_SPLITK_TURBO=1 \
    -e B12X_MOE_FORCE_A16=1 \
    -e VLLM_NVFP4_MLA_SCALES_FILE= \
    -e VLLM_NVFP4_MLA_DYNAMIC_SCALE=0 \
    -e VLLM_EXL3_TRELLIS_MIN_M=4 \
    -e VLLM_EXL3_TRELLIS_MAX_M=32 \
    -e VLLM_EXL3_TRELLIS_BLOCK_M=8 \
    -e VLLM_EXL3_PREFILL_CHUNK=128 \
    -e VLLM_EXL3_PREFILL_TRELLIS=1 \
    -e VLLM_EXL3_PREFILL_BLOCK_M=64 \
    -e VLLM_EXL3_PREFILL_SYMMETRIC_TILE=0 \
    -e KV_FP8_ROPE=0 \
    -e CUDA_LAUNCH_BLOCKING=0 \
    -e VLLM_CACHE_DIR=/cache/jit/vllm \
    -e TRITON_CACHE_DIR=/cache/jit/triton \
    -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
    -e XDG_CACHE_HOME=/cache/jit \
    "${RUNTIME_ARGS[@]}" \
    --entrypoint "$PYTHON_ENTRYPOINT" "$IMAGE" \
      /kld/prefill_kld.py \
      --model /model \
      --reference-logits /ref \
      --context-length 2048 --stride 512 --max-windows 1 \
      --tensor-parallel-size 4 --gpu-memory-utilization "$KLD_UTIL" \
      --dtype bfloat16 --kv-cache-dtype "$KV_DTYPE" \
      --load-format safetensors \
      --max-model-len "$KLD_MAX_MODEL_LEN" \
      --max-num-batched-tokens 2048 --max-num-seqs 1 \
      --quantization exl3 --attention-backend B12X_MLA_SPARSE \
      --hf-overrides "$HF_OVERRIDES" --llm-extra-json "$LLM_EXTRA" \
      --kld-chunk-rows 32 \
      --per-position-output "$position_container" \
    2>&1 | tee "$OUT/run${run}.log"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  docker_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  cleanup_owned_container
  normalize_container_artifact_permissions "$run_cache" "$run_output"

  [[ "$tee_status" -eq 0 ]] || die "tee failed in candidate run $run"
  record="$(sed -n 's/^fallback_prefill_kld_done //p' \
    "$OUT/run${run}.log" | tail -n 1)"
  [[ -n "$record" ]] || die "candidate run $run emitted no final KLD record"
  printf '%s\n' "$record" > "$raw_record"
  jq -e \
    --arg path "$position_container" \
    --arg token_sha256 "$REFERENCE_TOKEN_IDS_U32LE_SHA256" '
    .total_positions == 2047 and .mean_kld >= 0 and .elapsed_sec > 0 and
    .token_ids_u32le_sha256 == $token_sha256 and
    .per_position.path == $path and
    (.per_position.sha256 | test("^[0-9a-f]{64}$")) and
    .per_position.positions == 2047 and
    .per_position.tensor == "kld_ref_to_model"
  ' "$raw_record" >/dev/null
  run_log="$OUT/run${run}.log"
  dispatch_proof="$OUT/run${run}-sqg-dispatch-proof.json"
  dispatch_records_json='[]'
  for sqg_layer in "${SQG_EVAL_LAYER_ARRAY[@]}"; do
    dispatch_line="EXL3 SQG layer model.layers.${sqg_layer}.mlp.experts: retaining 768 per-projection native tensors"
    dispatch_count="$(grep -Fc "$dispatch_line" "$run_log" || true)"
    [[ "$dispatch_count" -ge 1 ]] || \
      die "candidate run $run did not prove SQG dispatch for layer $sqg_layer"
    dispatch_records_json="$(jq -cn \
      --argjson records "$dispatch_records_json" \
      --argjson layer "$sqg_layer" \
      --argjson count "$dispatch_count" \
      '$records + [{layer:$layer,count:$count}]')"
  done
  run_log_sha256="$(sha256sum "$run_log" | awk '{print $1}')"
  jq -n \
    --arg log "$run_log" \
    --arg log_sha256 "$run_log_sha256" \
    --argjson selected_layers "$SQG_EVAL_LAYERS_JSON" \
    --argjson records "$dispatch_records_json" '
    {
      schema: "glm52-sqg-dispatch-proof-v1",
      log: $log,
      log_sha256: $log_sha256,
      required_layers: $selected_layers,
      records: $records,
      every_required_line_observed: true
    }
  ' > "$dispatch_proof"
  dispatch_proof_sha256="$(sha256sum "$dispatch_proof" | awk '{print $1}')"
  if [[ "$docker_status" -ne 0 ]]; then
    die "candidate run $run failed with Docker exit $docker_status"
  fi
  [[ -z "$(find "$run_cache" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \) \
    -print -quit)" ]] || \
    die "candidate run $run runtime cache contains a symlink or special file"
  cache_file_count="$(cache_file_count_cheap "$run_cache")"
  cache_receipt="$OUT/run${run}-runtime-cache-receipt.json"
  if [[ -f "$cache_seed_receipt" ]]; then
    jq --argjson post_run_file_count "$cache_file_count" \
      '. + {
        started_empty:false,
        seeded_snapshot:true,
        isolated_per_run:true,
        fresh_container_python_engine_workers_model_load:true,
        destination_file_count_after_boot:$post_run_file_count
      }' "$cache_seed_receipt" > "$cache_receipt"
  else
    jq -n --arg path "$run_cache" --argjson file_count "$cache_file_count" '
      {
        schema:"glm52-runtime-cache-receipt-v1",path:$path,
        started_empty:true,seeded_snapshot:false,isolated_per_run:true,
        fresh_container_python_engine_workers_model_load:true,
        file_count_after_boot:$file_count,byte_hashing_skipped:true,
        cache_semantics:"compiled executable code only; no model, logits, KV, RNG, or process state"
      }
    ' > "$cache_receipt"
  fi
  cache_receipt_sha256="$(sha256sum "$cache_receipt" | awk '{print $1}')"
  position_sha256="$(jq -er '.per_position.sha256' "$raw_record")"
  position_mean="$(jq -er '.mean_kld' "$raw_record")"
  [[ -f "$position_host" && ! -L "$position_host" ]] || \
    die "candidate run $run did not publish a real per-position tensor"
  python3 "$POSITION_VALIDATOR" "$position_host" \
    --expected-sha256 "$position_sha256" \
    --expected-positions 2047 \
    --expected-mean-kld "$position_mean" \
    > "$position_validation"
  jq -e --arg sha256 "$position_sha256" '
    .valid == true and .sha256 == $sha256 and .positions == 2047 and
    .tensor == "kld_ref_to_model"
  ' "$position_validation" >/dev/null
  position_validation_sha256="$(sha256sum "$position_validation" | awk '{print $1}')"
  raw_record_sha256="$(sha256sum "$raw_record" | awk '{print $1}')"
  jq \
    --arg host_path "$position_host" \
    --arg validation_report "$position_validation" \
    --arg validation_sha256 "$position_validation_sha256" \
    --arg raw_record_sha256 "$raw_record_sha256" \
    --arg cache_receipt "$cache_receipt" \
    --arg cache_receipt_sha256 "$cache_receipt_sha256" \
    --arg run_log "$run_log" \
    --arg run_log_sha256 "$run_log_sha256" \
    --arg dispatch_proof "$dispatch_proof" \
    --arg dispatch_proof_sha256 "$dispatch_proof_sha256" \
    --argjson selected_layers "$SQG_EVAL_LAYERS_JSON" \
    --slurpfile runtime_cache "$cache_receipt" '
    .raw_record_sha256 = $raw_record_sha256 |
    .docker_exit_status = 0 |
    .runtime_cache = ($runtime_cache[0] + {
      receipt:$cache_receipt,
      receipt_sha256:$cache_receipt_sha256
    }) |
    .runtime_dispatch = {
      log: $run_log,
      log_sha256: $run_log_sha256,
      proof: $dispatch_proof,
      proof_sha256: $dispatch_proof_sha256,
      selected_layers: $selected_layers,
      sqg_dispatch_proved: true
    } |
    .per_position += {
      host_path: $host_path,
      validation_report: $validation_report,
      validation_report_sha256: $validation_sha256,
      independently_validated: true
    }
  ' "$raw_record" > "$accepted_record"
  accepted_record_sha256="$(sha256sum "$accepted_record" | awk '{print $1}')"
  {
    printf '%s  %s\n' "$position_sha256" "$position_host"
    printf '%s  %s\n' \
      "$position_validation_sha256" "$position_validation"
    printf '%s  %s\n' "$run_log_sha256" "$run_log"
    printf '%s  %s\n' "$dispatch_proof_sha256" "$dispatch_proof"
    printf '%s  %s\n' "$accepted_record_sha256" "$accepted_record"
  } >> "$OUT/per-position-evidence.sha256"
  sha256sum -c "$OUT/per-position-evidence.sha256" >/dev/null
  ACCEPTED_RECORDS+=( "$accepted_record" )
done

verify_sealed_eval_inputs
[[ "$(wc -l < "$OUT/per-position-evidence.sha256")" -eq $((RUNS * 5)) ]] || \
  die "per-position evidence census differs"
sha256sum -c "$OUT/per-position-evidence.sha256" >/dev/null
POSITION_EVIDENCE_SHA256="$(sha256sum \
  "$OUT/per-position-evidence.sha256" | awk '{print $1}')"
sha256sum "$OUT/per-position-evidence.sha256" > \
  "$OUT/per-position-evidence-manifest.sha256"

: > "$OUT/runs.jsonl"
for accepted_record in "${ACCEPTED_RECORDS[@]}"; do
  jq -c . "$accepted_record" >> "$OUT/runs.jsonl"
done
RUNS_JSONL_SHA256="$(sha256sum "$OUT/runs.jsonl" | awk '{print $1}')"
printf '%s  %s\n' "$RUNS_JSONL_SHA256" "$OUT/runs.jsonl" > \
  "$OUT/runs.jsonl.sha256"

jq -s -e \
  --arg candidate "$CANDIDATE" \
  --arg runtime_image "$IMAGE" \
  --arg runtime_image_id "$RUNTIME_IMAGE_ID" \
  --arg baseline_image "$RUNTIME_BASELINE_IMAGE_LABEL" \
  --arg baseline_image_id "$RUNTIME_BASELINE_IMAGE_ID" \
  --arg baseline_summary "$RUNTIME_BASELINE_SUMMARY" \
  --arg legacy_image "$LEGACY_BASELINE_IMAGE_LABEL" \
  --arg legacy_image_id "$LEGACY_BASELINE_IMAGE_ID" \
  --arg runtime_overlay "$runtime_overlay" \
  --arg overlay_manifest_sha256 "$OVERLAY_MANIFEST_SHA256" \
  --arg extra_args_file "$extra_args_file" \
  --arg extra_args_sha256 "$R33_EXTRA_ARGS_SHA256" \
  --arg exact_mount_evidence "$R33_MOUNT_EVIDENCE" \
  --arg exact_mount_evidence_sha256 "$R33_MOUNT_EVIDENCE_SHA256" \
  --arg pydeps "$PYDEPS" \
  --arg pydeps_tree_sha256 "$PYDEPS_TREE_SHA256" \
  --argjson pydeps_file_count "$PYDEPS_FILE_COUNT" \
  --arg reference_sha256 "$REFERENCE_SHA256" \
  --arg reference_token_ids_u32le_sha256 "$REFERENCE_TOKEN_IDS_U32LE_SHA256" \
  --arg selected_treatment_evidence "$OUT/selected-treatment-files.sha256" \
  --arg selected_treatment_evidence_sha256 "$SELECTED_TREATMENT_EVIDENCE_SHA256" \
  --argjson directional_test_fast "$DIRECTIONAL_TEST_FAST_JSON" \
  --arg boot_cache_seed "$CANDIDATE_BOOT_CACHE_SEED" \
  --arg boot_cache_seed_receipt "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT" \
  --arg boot_cache_seed_receipt_sha256 \
    "$CANDIDATE_BOOT_CACHE_SEED_RECEIPT_SHA256" \
  --arg boot_cache_seed_record "$CANDIDATE_BOOT_CACHE_SEED_RECORD" \
  --arg boot_cache_seed_record_sha256 \
    "$CANDIDATE_BOOT_CACHE_SEED_RECORD_SHA256" \
  --arg per_position_evidence "$OUT/per-position-evidence.sha256" \
  --arg per_position_evidence_sha256 "$POSITION_EVIDENCE_SHA256" \
  --arg runs_jsonl "$OUT/runs.jsonl" \
  --arg runs_jsonl_sha256 "$RUNS_JSONL_SHA256" \
  --argjson baseline_mean "$RUNTIME_BASELINE_MEAN" \
  --argjson baseline_sd "$RUNTIME_BASELINE_SD" \
  --argjson baseline_runs "$RUNTIME_BASELINE_RUNS" \
  --argjson legacy_mean "$LEGACY_BASELINE_MEAN" \
  --argjson legacy_sd "$LEGACY_BASELINE_SD" \
  --argjson expected_runs "$RUNS" \
  --argjson selected_layers "$SQG_EVAL_LAYERS_JSON" '
  if length != $expected_runs then
    error("candidate run count mismatch")
  elif (all(.[].per_position;
      .positions == 2047 and .tensor == "kld_ref_to_model" and
      .independently_validated == true and
      (.sha256 | test("^[0-9a-f]{64}$")) and
      (.validation_report_sha256 | test("^[0-9a-f]{64}$"))) | not) then
    error("candidate per-position KLD evidence differs")
  elif (all(.[];
      .runtime_dispatch.sqg_dispatch_proved == true and
      .runtime_dispatch.selected_layers == $selected_layers and
      (.runtime_dispatch.log_sha256 | test("^[0-9a-f]{64}$")) and
      (.runtime_dispatch.proof_sha256 | test("^[0-9a-f]{64}$"))) | not) then
    error("candidate runtime SQG dispatch evidence differs")
  elif ($boot_cache_seed != "" and
      (all(.[].runtime_cache;
        .started_empty == false and .seeded_snapshot == true and
        .isolated_per_run == true and
        .fresh_container_python_engine_workers_model_load == true and
        .source == $boot_cache_seed and
        .source_arm == "rejected_sqg_candidate_compiled_code_cache" and
        .source_receipt == $boot_cache_seed_receipt and
        .source_receipt_sha256 == $boot_cache_seed_receipt_sha256 and
        .source_accepted_record == $boot_cache_seed_record and
        .source_accepted_record_sha256 == $boot_cache_seed_record_sha256 and
        .byte_hashing_skipped == true) | not)) then
    error("candidate external compiled-code cache evidence differs")
  else
    map(.mean_kld) as $values |
    ($values | add / length) as $mean |
    (if length > 1 then
      ([$values[] as $x | (($x - $mean) * ($x - $mean))]
       | add / (length - 1) | sqrt)
     else 0 end) as $sd |
    ($mean - $baseline_mean) as $delta |
    ($mean - $legacy_mean) as $legacy_delta |
    (($baseline_sd * $baseline_sd) / $baseline_runs) as $baseline_mean_var |
    (if length > 1 then (($sd * $sd) / length) else null end) as $candidate_mean_var |
    (if $candidate_mean_var != null then
       (($baseline_mean_var + $candidate_mean_var) | sqrt)
     else null end) as $delta_se |
    (if $delta_se != null and $delta_se > 0 then $delta / $delta_se
     else null end) as $welch_t |
    (if $candidate_mean_var != null then
       ((($baseline_mean_var + $candidate_mean_var) *
         ($baseline_mean_var + $candidate_mean_var)) /
        ((($baseline_mean_var * $baseline_mean_var) /
          ($baseline_runs - 1)) +
         (($candidate_mean_var * $candidate_mean_var) / (length - 1))))
     else null end) as $welch_df |
    {
      schema: "glm52-fresh-sqg-candidate-kld-result-v2",
      directional_test_fast_mode: $directional_test_fast,
      baseline_was_rerun: false,
      candidate: $candidate,
      selected_sqg_layers: $selected_layers,
      reference_sha256: $reference_sha256,
      reference_token_ids_u32le_sha256: $reference_token_ids_u32le_sha256,
      selected_treatment_evidence: {
        manifest: $selected_treatment_evidence,
        manifest_sha256: $selected_treatment_evidence_sha256,
        rehashed_before_every_launch: ($directional_test_fast | not)
      },
      regime: {
        kv_cache_dtype: "fp8",
        rope: "bfloat16",
        tensor_parallel_size: 4,
        decode_context_parallel_size: 4,
        dcp_comm_backend: "a2a",
        dcp_kv_cache_interleave_size: 64,
        context_tokens: 2048,
        scored_positions: 2047
      },
      runtime: {
        candidate_image: $runtime_image,
        candidate_image_id: $runtime_image_id,
        baseline_image: $baseline_image,
        baseline_image_id: $baseline_image_id,
        base_image_matches_baseline: ($runtime_image_id == $baseline_image_id),
        candidate_runtime_overlay_applied: true,
        runtime_overlay: {
          path: $runtime_overlay,
          manifest_sha256: $overlay_manifest_sha256
        },
        exact_r33_extension_directory_mount_applied: true,
        direct_exl3_shared_object_override_applied: false,
        extension_matches_baseline: true,
        exact_r33_extra_docker_args: {
          path: $extra_args_file,
          sha256: $extra_args_sha256
        },
        exact_r33_mount_sources: {
          manifest: $exact_mount_evidence,
          manifest_sha256: $exact_mount_evidence_sha256,
          all_bytes_validated: true,
          extension_directory_exact_single_file: true
        },
        paired_kld_dependencies: {
          path: $pydeps,
          file_count: $pydeps_file_count,
          tree_sha256: $pydeps_tree_sha256,
          all_bytes_validated: true,
          symlinks_or_special_files: 0,
          network_fetch_disabled: true
        },
        runtime_code_matches_baseline: false
      },
      runtime_cache_seed: {
        enabled: ($boot_cache_seed != ""),
        source: $boot_cache_seed,
        source_receipt: $boot_cache_seed_receipt,
        source_receipt_sha256: $boot_cache_seed_receipt_sha256,
        source_accepted_record: $boot_cache_seed_record,
        source_accepted_record_sha256: $boot_cache_seed_record_sha256,
        source_arm: (if $boot_cache_seed == "" then null
          else "rejected_sqg_candidate_compiled_code_cache" end),
        independent_snapshot_per_boot: true,
        cache_payload_byte_hashing_skipped: true,
        fresh_container_python_engine_workers_model_load_every_boot: true
      },
      primary_same_base_image_baseline: {
        summary: $baseline_summary,
        runs: $baseline_runs,
        mean_kld: $baseline_mean,
        sample_sd_kld: $baseline_sd
      },
      secondary_legacy_baseline: {
        runs: 5,
        mean_kld: $legacy_mean,
        sample_sd_kld: $legacy_sd,
        image: $legacy_image,
        image_id: $legacy_image_id
      },
      candidate_result: {
        runs: length,
        values: $values,
        mean_kld: $mean,
        sample_sd_kld: $sd,
        min_kld: ($values | min),
        max_kld: ($values | max),
        elapsed_seconds: map(.elapsed_sec),
        paired_per_position_outputs: map(.per_position),
        paired_per_position_evidence: {
          manifest: $per_position_evidence,
          manifest_sha256: $per_position_evidence_sha256
        },
        derived_runs_jsonl: {
          path: $runs_jsonl,
          sha256: $runs_jsonl_sha256,
          used_as_summary_input: false
        }
      },
      comparison: {
        baseline_kind: "existing-same-base-image-r33-runtime-confounded",
        scope:
          "SQG codebook plus required SQG loader/non-fused dispatch; not codebook-only",
        delta_candidate_minus_baseline: $delta,
        relative_delta: ($delta / $baseline_mean),
        mean_direction: (if $delta < 0 then "lower" elif $delta > 0 then "higher" else "equal" end),
        repeat_noise_delta_standard_error: $delta_se,
        welch_t_repeat_noise: $welch_t,
        welch_degrees_freedom: $welch_df,
        conservative_two_sided_critical_t: 2.776,
        conservative_95pct_repeat_noise_direction: (
          if length < 5 or $delta_se == null or $delta_se == 0 then "not_estimable"
          elif $delta < (-2.776 * $delta_se) then "lower"
          elif $delta > (2.776 * $delta_se) then "higher"
          else "inconclusive"
          end
        ),
        inference_limit:
          "One fixed 2047-position prompt; repeat-run Welch values describe runtime variation only, not generalization across text"
      },
      legacy_comparison: {
        baseline_kind: "published-r26",
        delta_candidate_minus_baseline: $legacy_delta,
        relative_delta: ($legacy_delta / $legacy_mean),
        mean_direction: (
          if $legacy_delta < 0 then "lower"
          elif $legacy_delta > 0 then "higher"
          else "equal" end
        )
      }
    }
  end
' "${ACCEPTED_RECORDS[@]}" > "$OUT/summary.json.partial"

sha256sum -c "$OUT/per-position-evidence.sha256" >/dev/null
printf '%s  %s\n' "$RUNS_JSONL_SHA256" "$OUT/runs.jsonl" | \
  sha256sum -c - >/dev/null
mv "$OUT/summary.json.partial" "$OUT/summary.json"

jq . "$OUT/summary.json" | tee "$OUT/summary.txt"
printf '%s\n' "$OUT"
