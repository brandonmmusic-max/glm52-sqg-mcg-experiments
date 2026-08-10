#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 0 ]] || die "usage: $0"

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$EXPERIMENT_DIR/.." && pwd)"
RUNNER="$EXPERIMENT_DIR/run_native_mcg_control.sh"
PROTECTED_MODEL=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED
PRODUCTION_CONTAINER=glm-r33-fixed
PROTECTED_MODEL="$(realpath -e -- "$PROTECTED_MODEL")"
TEACHER_RECEIPT="$(realpath -e -- "$PROJECT_DIR/evidence/teacher_model_identity.json")"
CONTROL_VALIDATOR="$EXPERIMENT_DIR/validate_native_mcg_control.py"
POSITION_VALIDATOR="$EXPERIMENT_DIR/validate_per_position_kld.py"
CONTROL_VALIDATOR_SHA256=c66730a01d125d746cad81ecf24c8e13c5c212c44e6af7766a81080f3bf350f3
POSITION_VALIDATOR_SHA256=3a3388657a0d6fa72341db9a10c2d59bada1b7be78f8792737a8aaaba1ba7017
TEACHER_RECEIPT_SHA256=eb88fcd2bf66b0cfef195a232d9b81efcb107c400381b22c913a2c738413479e
MANIFEST_VERIFIED_SHA256=aa71c9791ae8c22cbad1cd7ed940733b0144a2f8a25d0e0bc571173065782514

for required in \
  .manifest_verified config.json quantization_config.json \
  model.safetensors.index.json; do
  [[ -f "$PROTECTED_MODEL/$required" && ! -L "$PROTECTED_MODEL/$required" ]] || \
    die "protected checkpoint is missing a real file: $required"
done

if [[ "$(docker inspect --format '{{.State.Running}}' \
  "$PRODUCTION_CONTAINER" 2>/dev/null || true)" == "true" ]]; then
  die "production container is running: $PRODUCTION_CONTAINER"
fi

# This host has a persistent small background allocation. Reject encoder,
# oracle, serving, or other material GPU work immediately before every launch.
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

IMAGE_ID=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
IMAGE="${IMAGE:-$IMAGE_ID}"
RUNTIME_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
[[ "$RUNTIME_IMAGE_ID" == "$IMAGE_ID" ]] || \
  die "native-MCG image does not match the preserved r33 image"
SITE=/opt/venv/lib/python3.12/site-packages
PYTHON_ENTRYPOINT=/opt/venv/bin/python

KV_DTYPE=fp8
DCP_SIZE=4
DCP_COMM_BACKEND=a2a
DCP_KV_CACHE_INTERLEAVE_SIZE=64
KLD_MAX_MODEL_LEN=2560
KLD_UTIL=0.90
RUNS="${RUNS:-5}"
[[ "$RUNS" == 5 ]] || die "native-MCG control requires exactly five fresh boots"
RESUME_FROM_RUN="${RESUME_FROM_RUN:-1}"
[[ "$RESUME_FROM_RUN" =~ ^[1-5]$ ]] || \
  die "RESUME_FROM_RUN must be an integer from 1 through 5"
RESUME_MODE=0
[[ "$RESUME_FROM_RUN" -eq 1 ]] || RESUME_MODE=1
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || \
  die "PREFLIGHT_ONLY must be 0 or 1"
DIRECTIONAL_TEST_FAST="${DIRECTIONAL_TEST_FAST:-0}"
[[ "$DIRECTIONAL_TEST_FAST" == 0 || "$DIRECTIONAL_TEST_FAST" == 1 ]] || \
  die "DIRECTIONAL_TEST_FAST must be 0 or 1"
DIRECTIONAL_TEST_FAST_JSON=false
[[ "$DIRECTIONAL_TEST_FAST" == 0 ]] || DIRECTIONAL_TEST_FAST_JSON=true
SNAPSHOT_SEED_RUNTIME_CACHE="${SNAPSHOT_SEED_RUNTIME_CACHE:-0}"
[[ "$SNAPSHOT_SEED_RUNTIME_CACHE" == 0 || \
   "$SNAPSHOT_SEED_RUNTIME_CACHE" == 1 ]] || \
  die "SNAPSHOT_SEED_RUNTIME_CACHE must be 0 or 1"
NATIVE_BOOT1_CACHE_SEED="${NATIVE_BOOT1_CACHE_SEED:-}"
if [[ "$SNAPSHOT_SEED_RUNTIME_CACHE" == 1 ]]; then
  [[ -n "$NATIVE_BOOT1_CACHE_SEED" ]] || \
    die "NATIVE_BOOT1_CACHE_SEED is required for cached native control"
  NATIVE_BOOT1_CACHE_SEED="$(realpath -e -- "$NATIVE_BOOT1_CACHE_SEED")"
  [[ -d "$NATIVE_BOOT1_CACHE_SEED" && ! -L "$NATIVE_BOOT1_CACHE_SEED" ]] || \
    die "native boot-1 cache seed is absent or unsafe"
fi

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-native-mcg}"
[[ "$STAMP" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "STAMP may contain only letters, digits, dot, underscore, and dash"
[[ -n "${NATIVE_MCG_RESULTS_ROOT:-}" ]] || \
  die "NATIVE_MCG_RESULTS_ROOT is required and must be distinct from candidate results"
RESULTS_ROOT="$(realpath -m -- "$NATIVE_MCG_RESULTS_ROOT")"
OUT="$RESULTS_ROOT/$STAMP-native-mcg-control-kld-fp8-dcp4"
OUT="$(realpath -m -- "$OUT")"
case "$OUT" in
  "$PROTECTED_MODEL"|"$PROTECTED_MODEL"/*|"$PROJECT_DIR"|"$PROJECT_DIR"/*)
    die "control result path overlaps protected input/project: $OUT"
    ;;
esac
if [[ "$RESUME_MODE" == 1 ]]; then
  [[ -d "$OUT" && ! -L "$OUT" ]] || \
    die "native-MCG continuation output is absent or unsafe: $OUT"
  [[ ! -e "$OUT/summary.json" && ! -L "$OUT/summary.json" ]] || \
    die "refusing to resume a completed native-MCG output"
else
  [[ ! -e "$OUT" ]] || die "control result path already exists: $OUT"
  mkdir -p "$OUT"
fi

CONTAINER_NAME="${CONTAINER_NAME:-glm52-native-mcg-kld-${STAMP:0:30}}"
[[ "$CONTAINER_NAME" =~ ^glm52-native-mcg-kld-[A-Za-z0-9_.-]+$ ]] || \
  die "native-MCG container name must use the isolated control prefix"
[[ "$CONTAINER_NAME" != "$PRODUCTION_CONTAINER" ]] || \
  die "control container name must not equal the protected production name"
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
FALLBACK="$EXPERIMENT_DIR/prefill_kld_paired.py"
FALLBACK_SHA256=dd808b681cc3952c90fccf8adfa1fa7cb601be1323f06b24b4bfb1bb7ee44b31
PROMPT_LOGPROB="$ROOT/eval-overlay/vllm/v1/worker/gpu/sample/prompt_logprob.py"
PROMPT_LOGPROB_SHA256=47f867c3ff81cc1778bae3f1a3189dd2ec90d6e469cf60f7d1e43a1c76989d6c
LOGPROB="$ROOT/eval-overlay/vllm/v1/worker/gpu/sample/logprob.py"
LOGPROB_SHA256=21d98eea20b8c92e0a65a5badffc3782dfc5a77427470ecd5d9edcedc681a041

printf '%s  %s\n' "$REFERENCE_SHA256" "$REFERENCE_LOGITS" | sha256sum -c -
printf '%s  %s\n' "$REFERENCE_MANIFEST_SHA256" "$REFERENCE_MANIFEST" | \
  sha256sum -c -
printf '%s  %s\n' "$FALLBACK_SHA256" "$FALLBACK" | sha256sum -c -
printf '%s  %s\n' "$CONTROL_VALIDATOR_SHA256" "$CONTROL_VALIDATOR" | sha256sum -c -
printf '%s  %s\n' "$POSITION_VALIDATOR_SHA256" "$POSITION_VALIDATOR" | sha256sum -c -
printf '%s  %s\n' "$TEACHER_RECEIPT_SHA256" "$TEACHER_RECEIPT" | sha256sum -c -
printf '%s  %s\n' "$PROMPT_LOGPROB_SHA256" "$PROMPT_LOGPROB" | sha256sum -c -
printf '%s  %s\n' "$LOGPROB_SHA256" "$LOGPROB" | sha256sum -c -

jq -e '
  .context_length == 2048 and
  (.token_first16 ==
    [284,8396,425,10960,465,284,14721,8396,
     425,10960,465,374,458,6364,4531,1154]) and
  (.windows | length == 1) and
  (.windows[0].shape == [2047,154880])
' "$REFERENCE_MANIFEST" >/dev/null

[[ -d "$PYDEPS" && ! -L "$PYDEPS" ]] || \
  die "paired-KLD dependency tree is absent or symlinked"
[[ -z "$(find "$PYDEPS" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" ]] || \
  die "paired-KLD dependency tree contains a symlink or special file"
[[ -d "$WIKITEXT_CACHE_SOURCE" && ! -L "$WIKITEXT_CACHE_SOURCE" ]] || \
  die "sealed WikiText cache source is absent or symlinked"
printf '%s  %s\n' \
  "$WIKITEXT_CACHE_EVIDENCE_SHA256" "$WIKITEXT_CACHE_EVIDENCE" | sha256sum -c -
[[ "$(find "$WIKITEXT_CACHE_SOURCE" -type f | wc -l)" -eq \
  "$WIKITEXT_CACHE_FILE_COUNT" ]] || die "sealed WikiText cache census differs"
(
  cd "$WIKITEXT_CACHE_SOURCE"
  sha256sum -c "$WIKITEXT_CACHE_EVIDENCE" >/dev/null
)

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

RUNTIME_OVERLAY="$(realpath -e -- "${RUNTIME_OVERLAY:-$EXPERIMENT_DIR/runtime_overlay}")"
[[ "$RUNTIME_OVERLAY" == "$(realpath -e -- "$EXPERIMENT_DIR/runtime_overlay")" ]] || \
  die "native-MCG control requires the sealed project runtime overlay"
OVERLAY_MANIFEST="$RUNTIME_OVERLAY/SHA256SUMS.runtime-overlay"
OVERLAY_MANIFEST_SHA256=1dc1b8439821ceecb459b42bf13a1efc3e44e376e8a2c5423e7b0a12ad5239d9
printf '%s  %s\n' "$OVERLAY_MANIFEST_SHA256" "$OVERLAY_MANIFEST" | sha256sum -c -
(
  cd "$RUNTIME_OVERLAY"
  sha256sum -c SHA256SUMS.runtime-overlay >/dev/null
)
overlay_expected="$({
  awk '{print $2}' "$OVERLAY_MANIFEST"
  printf './SHA256SUMS.runtime-overlay\n'
} | sort)"
overlay_observed="$(
  cd "$RUNTIME_OVERLAY"
  find . -type f -printf '%p\n' | sort
)"
[[ "$overlay_observed" == "$overlay_expected" ]] || \
  die "runtime overlay exact file inventory differs"
[[ -z "$(find "$RUNTIME_OVERLAY" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" ]] || \
  die "runtime overlay contains a symlink or special file"

NATIVE_ARGS="$(realpath -e -- \
  "${EXTRA_DOCKER_ARGS_FILE:-$EXPERIMENT_DIR/r33_native_mcg_control.args}")"
[[ "$NATIVE_ARGS" == "$(realpath -e -- "$EXPERIMENT_DIR/r33_native_mcg_control.args")" ]] || \
  die "native-MCG control requires the sealed native args file"
NATIVE_ARGS_SHA256=fa572621970a0b0a3ab4fdba4af3f783eef7f9c2d7975744c8a1e09e9f67f598
printf '%s  %s\n' "$NATIVE_ARGS_SHA256" "$NATIVE_ARGS" | sha256sum -c -
RUNTIME_ARGS=(
  -v "$RUNTIME_OVERLAY:/sqg-runtime-overlay:ro"
  -e "PYTHONPATH=/sqg-runtime-overlay"
)
while IFS= read -r extra_arg || [[ -n "$extra_arg" ]]; do
  [[ -z "$extra_arg" || "$extra_arg" == \#* ]] && continue
  RUNTIME_ARGS+=( "$extra_arg" )
done < "$NATIVE_ARGS"

R33_MOUNT_EVIDENCE="$EXPERIMENT_DIR/r33_exact_mounts.sha256"
R33_MOUNT_EVIDENCE_SHA256=ac0728029aff006cb0a5cce9053a5aabbac49567b86461071557a68d5ed8f3d7
printf '%s  %s\n' "$R33_MOUNT_EVIDENCE_SHA256" "$R33_MOUNT_EVIDENCE" | sha256sum -c -
sha256sum -c "$R33_MOUNT_EVIDENCE" >/dev/null
R33_EXL3_EXT_DIR=/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3
[[ -d "$R33_EXL3_EXT_DIR" && ! -L "$R33_EXL3_EXT_DIR" &&
   -f "$R33_EXL3_EXT_DIR/exllamav3_ext.cpython-312-x86_64-linux-gnu.so" &&
   ! -L "$R33_EXL3_EXT_DIR/exllamav3_ext.cpython-312-x86_64-linux-gnu.so" ]] || \
  die "exact-r33 ExLlamaV3 extension is absent or not a real file"
[[ "$(find "$R33_EXL3_EXT_DIR" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]] || \
  die "exact-r33 ExLlamaV3 extension directory inventory differs"

if [[ "$DIRECTIONAL_TEST_FAST" == 1 ]]; then
  [[ "$(sha256sum "$PROTECTED_MODEL/.manifest_verified" | awk '{print $1}')" == \
    "$MANIFEST_VERIFIED_SHA256" ]] || \
    die "protected checkpoint verification marker differs"
  jq -e '
    .r7_routed_experts.codebook == "mcg" and
    (.r7_routed_experts.codebook_overrides // {}) == {} and
    (.r7_routed_experts.codebook_tensor_overrides // {}) == {}
  ' "$PROTECTED_MODEL/quantization_config.json" >/dev/null
  CONTROL_REQUIRED_NAMES='["config.json","quantization_config.json",
    "model.safetensors.index.json","r7-experts-layer-006.json",
    "r7-experts-layer-006.safetensors","r7-experts-layer-028.json",
    "r7-experts-layer-028.safetensors","r7-experts-layer-052.json",
    "r7-experts-layer-052.safetensors","r7-experts-layer-077.json",
    "r7-experts-layer-077.safetensors"]'
  jq -n \
    --arg model "$PROTECTED_MODEL" \
    --arg marker_sha256 "$MANIFEST_VERIFIED_SHA256" \
    --argjson required_names "$CONTROL_REQUIRED_NAMES" \
    --slurpfile teacher "$TEACHER_RECEIPT" '
    ($teacher[0].seal.files |
      reduce (.[] | select(.path as $path | $required_names | index($path))) as $item
      ({}; .[$item.path] = {
        bytes:$item.bytes,sha256:$item.sha256,role:$item.role
      })) as $verified_files |
    {
      schema:"glm52-native-mcg-control-preflight-v1",
      valid:true,
      directional_test_fast_identity_only:true,
      protected_model:$model,
      preflight_id:$marker_sha256,
      selected_layers:[6,28,52,77],
      global_codebook:"mcg",
      selected_trellis_tensors:3072,
      selected_mcg_markers:3072,
      selected_sqg_markers:0,
      codebook_overrides:0,
      tensor_overrides:0,
      protected_model_read_only_mount_required:true,
      verified_files:$verified_files,
      layers:[6,28,52,77] | map({
        layer:.,trellis_tensors:768,mcg_markers:768,sqg_markers:0,
        exclusive_mcg_marker_payloads_verified:false
      })
    }
  ' > "$OUT/native-mcg-preflight.json"
  jq -e --arg model "$PROTECTED_MODEL" '
    .schema == "glm52-native-mcg-control-preflight-v1" and
    .valid == true and .directional_test_fast_identity_only == true and
    .protected_model == $model and .selected_layers == [6,28,52,77] and
    .global_codebook == "mcg" and .selected_mcg_markers == 3072 and
    .selected_sqg_markers == 0 and (.verified_files | length) == 11
  ' "$OUT/native-mcg-preflight.json" >/dev/null
else
  python3 "$CONTROL_VALIDATOR" "$PROTECTED_MODEL" \
    --expected-model "$PROTECTED_MODEL" \
    --teacher-receipt "$TEACHER_RECEIPT" \
    --manifest-verified-sha256 "$MANIFEST_VERIFIED_SHA256" \
    > "$OUT/native-mcg-preflight.json"
  jq -e --arg model "$PROTECTED_MODEL" '
    .schema == "glm52-native-mcg-control-preflight-v1" and
    .valid == true and .protected_model == $model and
    .selected_layers == [6,28,52,77] and
    .global_codebook == "mcg" and
    .selected_trellis_tensors == 3072 and
    .selected_mcg_markers == 3072 and
    .selected_sqg_markers == 0 and
    .codebook_overrides == 0 and .tensor_overrides == 0 and
    .protected_model_read_only_mount_required == true and
    (.layers | length) == 4 and
    all(.layers[];
      .trellis_tensors == 768 and .mcg_markers == 768 and
      .sqg_markers == 0 and
      .exclusive_mcg_marker_payloads_verified == true)
  ' "$OUT/native-mcg-preflight.json" >/dev/null
fi
CONTROL_PREFLIGHT_SHA256="$(sha256sum "$OUT/native-mcg-preflight.json" | awk '{print $1}')"
CONTROL_PREFLIGHT_ID="$(jq -er '.preflight_id' "$OUT/native-mcg-preflight.json")"

CONTROL_MODEL_EVIDENCE="$OUT/control-model-selected-files.sha256"
: > "$CONTROL_MODEL_EVIDENCE"
for name in \
  config.json quantization_config.json model.safetensors.index.json \
  r7-experts-layer-006.json r7-experts-layer-006.safetensors \
  r7-experts-layer-028.json r7-experts-layer-028.safetensors \
  r7-experts-layer-052.json r7-experts-layer-052.safetensors \
  r7-experts-layer-077.json r7-experts-layer-077.safetensors; do
  file_sha256="$(jq -er --arg name "$name" \
    '.verified_files[$name].sha256 | select(test("^[0-9a-f]{64}$"))' \
    "$OUT/native-mcg-preflight.json")"
  printf '%s  %s\n' "$file_sha256" "$PROTECTED_MODEL/$name" >> \
    "$CONTROL_MODEL_EVIDENCE"
done
CONTROL_MODEL_EVIDENCE_SHA256="$(sha256sum "$CONTROL_MODEL_EVIDENCE" | awk '{print $1}')"
CONTROL_RUNTIME_EVIDENCE="$OUT/control-model-runtime-files.sha256"
jq -r --arg root "$PROTECTED_MODEL" '
  .seal.files[] | "\(.sha256)  \($root)/\(.path)"
' "$TEACHER_RECEIPT" > "$CONTROL_RUNTIME_EVIDENCE"
CONTROL_RUNTIME_FILE_COUNT="$(jq -er '.seal.total_file_count' "$TEACHER_RECEIPT")"
[[ "$(wc -l < "$CONTROL_RUNTIME_EVIDENCE")" -eq \
  "$CONTROL_RUNTIME_FILE_COUNT" ]] || die "control runtime evidence census differs"
while read -r _ runtime_file; do
  [[ -f "$runtime_file" && ! -L "$runtime_file" ]] || \
    die "sealed control runtime file is absent, non-regular, or symlinked: $runtime_file"
done < "$CONTROL_RUNTIME_EVIDENCE"
CONTROL_RUNTIME_EVIDENCE_SHA256="$(sha256sum \
  "$CONTROL_RUNTIME_EVIDENCE" | awk '{print $1}')"

docker image inspect "$IMAGE" > "$OUT/image-inspect.json"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "$OUT/gpu-preflight.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "$OUT/gpu-compute-preflight.csv"
printf '%s\n' "$PROTECTED_MODEL" > "$OUT/protected-model-path.txt"
sha256sum "$RUNNER" "$CONTROL_VALIDATOR" "$POSITION_VALIDATOR" \
  "$FALLBACK" "$PROMPT_LOGPROB" "$LOGPROB" > "$OUT/eval-code.sha256"
sha256sum "$TEACHER_RECEIPT" "$NATIVE_ARGS" "$OVERLAY_MANIFEST" \
  "$R33_MOUNT_EVIDENCE" > "$OUT/sealed-control-inputs.sha256"

(
  cd "$PYDEPS"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$OUT/pydeps-files.sha256"
[[ "$(wc -l < "$OUT/pydeps-files.sha256")" -eq "$PYDEPS_FILE_COUNT" ]] || \
  die "paired-KLD dependency file census differs"
[[ "$(sha256sum "$OUT/pydeps-files.sha256" | awk '{print $1}')" == \
  "$PYDEPS_TREE_SHA256" ]] || die "paired-KLD dependency tree hash differs"

verify_sealed_eval_inputs() {
  local current_overlay
  if [[ "$DIRECTIONAL_TEST_FAST" == 1 ]]; then
    # The protected checkpoint is mounted read-only for every boot. The
    # selected native-MCG layers were validated once above; retain only cheap
    # identity and presence checks between boots.
    printf '%s  %s\n' "$REFERENCE_MANIFEST_SHA256" "$REFERENCE_MANIFEST" | \
      sha256sum -c - >/dev/null
    sha256sum -c "$OUT/eval-code.sha256" >/dev/null
    while read -r _ runtime_file; do
      [[ -f "$runtime_file" && ! -L "$runtime_file" ]] || \
        die "protected control runtime file changed type after preflight"
    done < "$CONTROL_RUNTIME_EVIDENCE"
    [[ "$(docker inspect --format '{{.State.Running}}' \
      "$PRODUCTION_CONTAINER" 2>/dev/null || true)" != "true" ]] || \
      die "production container restarted during native-MCG control"
    return 0
  fi
  printf '%s  %s\n' "$REFERENCE_SHA256" "$REFERENCE_LOGITS" | sha256sum -c - >/dev/null
  printf '%s  %s\n' "$REFERENCE_MANIFEST_SHA256" "$REFERENCE_MANIFEST" | \
    sha256sum -c - >/dev/null
  printf '%s  %s\n' "$CONTROL_VALIDATOR_SHA256" "$CONTROL_VALIDATOR" | \
    sha256sum -c - >/dev/null
  printf '%s  %s\n' "$POSITION_VALIDATOR_SHA256" "$POSITION_VALIDATOR" | \
    sha256sum -c - >/dev/null
  printf '%s  %s\n' "$FALLBACK_SHA256" "$FALLBACK" | sha256sum -c - >/dev/null
  printf '%s  %s\n' "$TEACHER_RECEIPT_SHA256" "$TEACHER_RECEIPT" | \
    sha256sum -c - >/dev/null
  printf '%s  %s\n' "$NATIVE_ARGS_SHA256" "$NATIVE_ARGS" | sha256sum -c - >/dev/null
  printf '%s  %s\n' "$OVERLAY_MANIFEST_SHA256" "$OVERLAY_MANIFEST" | \
    sha256sum -c - >/dev/null
  (
    cd "$RUNTIME_OVERLAY"
    sha256sum -c SHA256SUMS.runtime-overlay >/dev/null
  )
  current_overlay="$(
    cd "$RUNTIME_OVERLAY"
    find . -type f -printf '%p\n' | sort
  )"
  [[ "$current_overlay" == "$overlay_expected" ]] || \
    die "runtime overlay inventory changed after preflight"
  [[ -z "$(find "$RUNTIME_OVERLAY" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" ]] || \
    die "runtime overlay changed type after preflight"
  sha256sum -c "$R33_MOUNT_EVIDENCE" >/dev/null
  [[ "$(find "$R33_EXL3_EXT_DIR" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]] || \
    die "exact-r33 extension directory changed after preflight"
  [[ "$(sha256sum "$CONTROL_MODEL_EVIDENCE" | awk '{print $1}')" == \
    "$CONTROL_MODEL_EVIDENCE_SHA256" ]] || \
    die "selected control evidence manifest changed after preflight"
  sha256sum -c "$CONTROL_MODEL_EVIDENCE" >/dev/null
  [[ "$(sha256sum "$CONTROL_RUNTIME_EVIDENCE" | awk '{print $1}')" == \
    "$CONTROL_RUNTIME_EVIDENCE_SHA256" ]] || \
    die "control runtime evidence manifest changed after preflight"
  [[ "$(wc -l < "$CONTROL_RUNTIME_EVIDENCE")" -eq \
    "$CONTROL_RUNTIME_FILE_COUNT" ]] || \
    die "control runtime evidence census changed after preflight"
  sha256sum -c "$CONTROL_RUNTIME_EVIDENCE" >/dev/null
  sha256sum -c "$OUT/eval-code.sha256" >/dev/null
  sha256sum -c "$OUT/sealed-control-inputs.sha256" >/dev/null
  (
    cd "$PYDEPS"
    sha256sum -c "$OUT/pydeps-files.sha256" >/dev/null
  )
  [[ "$(find "$PYDEPS" -type f | wc -l)" -eq "$PYDEPS_FILE_COUNT" ]] || \
    die "paired-KLD dependency census changed after preflight"
  [[ -z "$(find "$PYDEPS" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" ]] || \
    die "paired-KLD dependency tree changed type after preflight"
  (
    cd "$WIKITEXT_CACHE_SOURCE"
    sha256sum -c "$WIKITEXT_CACHE_EVIDENCE" >/dev/null
  )
  [[ "$(find "$WIKITEXT_CACHE_SOURCE" -type f | wc -l)" -eq \
    "$WIKITEXT_CACHE_FILE_COUNT" ]] || \
    die "sealed WikiText cache census changed after preflight"
  [[ "$(docker inspect --format '{{.State.Running}}' \
    "$PRODUCTION_CONTAINER" 2>/dev/null || true)" != "true" ]] || \
    die "production container restarted during native-MCG control"
}
verify_sealed_eval_inputs

jq -n \
  --arg model "$PROTECTED_MODEL" \
  --arg image "$IMAGE" \
  --arg image_id "$RUNTIME_IMAGE_ID" \
  --arg overlay "$RUNTIME_OVERLAY" \
  --arg overlay_sha256 "$OVERLAY_MANIFEST_SHA256" \
  --arg native_args "$NATIVE_ARGS" \
  --arg native_args_sha256 "$NATIVE_ARGS_SHA256" \
  --arg preflight "$OUT/native-mcg-preflight.json" \
  --arg preflight_sha256 "$CONTROL_PREFLIGHT_SHA256" \
  --arg preflight_id "$CONTROL_PREFLIGHT_ID" \
  --arg model_evidence "$CONTROL_MODEL_EVIDENCE" \
  --arg model_evidence_sha256 "$CONTROL_MODEL_EVIDENCE_SHA256" \
  --arg runtime_evidence "$CONTROL_RUNTIME_EVIDENCE" \
  --arg runtime_evidence_sha256 "$CONTROL_RUNTIME_EVIDENCE_SHA256" \
  --argjson runtime_file_count "$CONTROL_RUNTIME_FILE_COUNT" \
  --arg reference_sha256 "$REFERENCE_SHA256" \
  --arg reference_token_sha256 "$REFERENCE_TOKEN_IDS_U32LE_SHA256" \
  --argjson directional_test_fast "$DIRECTIONAL_TEST_FAST_JSON" \
  --argjson runs "$RUNS" '
  {
    schema: "glm52-native-mcg-control-kld-run-v1",
    directional_test_fast_mode: $directional_test_fast,
    arm: "native_mcg_nonfused_dispatch",
    protected_model: $model,
    protected_model_mount: "read_only",
    candidate_mounted: false,
    runtime: {
      image: $image,
      image_id: $image_id,
      runtime_overlay: {path: $overlay, manifest_sha256: $overlay_sha256},
      native_args: {path: $native_args, sha256: $native_args_sha256}
    },
    model_preflight: {
      path: $preflight,
      sha256: $preflight_sha256,
      id: $preflight_id,
      selected_file_evidence: $model_evidence,
      selected_file_evidence_sha256: $model_evidence_sha256,
      complete_runtime_evidence: $runtime_evidence,
      complete_runtime_evidence_sha256: $runtime_evidence_sha256,
      complete_runtime_file_count: $runtime_file_count,
      rehashed_before_every_launch: ($directional_test_fast | not)
    },
    selected_layers: [6,28,52,77],
    selected_codebook: "mcg",
    selected_mcg_markers: 3072,
    selected_sqg_markers: 0,
    fused_slot_accounting: {actual: 45, reserved: 3, total: 48,
      reserved_layers: [6,28,52]},
    reference_sha256: $reference_sha256,
    reference_token_ids_u32le_sha256: $reference_token_sha256,
    regime: {
      kv_cache_dtype: "fp8", rope: "bfloat16", tensor_parallel_size: 4,
      decode_context_parallel_size: 4, dcp_comm_backend: "a2a",
      dcp_kv_cache_interleave_size: 64, context_tokens: 2048,
      scored_positions: 2047, gpu_memory_utilization: 0.90
    },
    requested_control_runs: $runs,
    inference_limit:
      "One fixed 2047-position prompt; repeats estimate runtime variation, not text/model generalization"
  }
' > "$OUT/run-manifest.json"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  printf '%s\n' "$OUT"
  exit 0
fi

ACCEPTED_RECORDS=()
if [[ "$RESUME_MODE" == 1 ]]; then
  for previous_run in $(seq 1 $((RESUME_FROM_RUN - 1))); do
    previous_record="$OUT/run${previous_run}-record.accepted.json"
    [[ -f "$previous_record" && ! -L "$previous_record" ]] || \
      die "native-MCG run $previous_run must be accepted before continuation"
    jq -e '
      .docker_exit_status == 0 and
      .total_positions == 2047 and .mean_kld >= 0 and
      .runtime_dispatch.native_mcg_dispatch_proved == true and
      .runtime_dispatch.sqg_dispatch_records == 0 and
      .runtime_dispatch.selected_layers == [6,28,52,77] and
      .per_position.positions == 2047 and
      .per_position.independently_validated == true and
      .runtime_cache.isolated_per_run == true and
      .runtime_cache.fresh_container_python_engine_workers_model_load == true and
      .runtime_cache.byte_hashing_skipped == true
    ' "$previous_record" >/dev/null || \
      die "native-MCG run $previous_run accepted record is not continuation-safe"
    ACCEPTED_RECORDS+=( "$previous_record" )
  done
  [[ -f "$OUT/per-position-evidence.sha256" &&
     "$(wc -l < "$OUT/per-position-evidence.sha256")" -eq \
       $(((RESUME_FROM_RUN - 1) * 5)) ]] || \
    die "native-MCG accepted-run evidence census differs before continuation"
  sha256sum -c "$OUT/per-position-evidence.sha256" >/dev/null
  jq -n --arg output "$OUT" --arg runner "$RUNNER" \
    --argjson resume_from "$RESUME_FROM_RUN" '
    {
      schema:"glm52-native-mcg-cached-continuation-v1",
      output:$output,resume_from_run:$resume_from,final_expected_runs:5,
      preserved_completed_runs:($resume_from - 1),
      cache_seed:
        "boot 1 external compiled-code snapshot; later boots snapshot native run 1",
      byte_neutral_runtime_cache_hashing_skipped:true,
      fresh_container_python_engine_workers_model_load_every_boot:true,
      runner:$runner
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
    die "could not make container artifacts readable for host-side recording and cache reuse"
}
cache_file_count_cheap() {
  sudo -n find "$1" -type f -printf '.' | wc -c
}
snapshot_runtime_cache() {
  local source_cache="$1" destination_cache="$2" receipt="$3"
  local source_count destination_count
  [[ -d "$source_cache" && ! -L "$source_cache" ]] || \
    die "native runtime-cache seed is absent or unsafe: $source_cache"
  [[ -z "$(sudo -n find "$source_cache" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \) \
    -print -quit)" ]] || die "native runtime-cache seed contains a special file"
  source_count="$(cache_file_count_cheap "$source_cache")"
  [[ "$source_count" -gt 0 ]] || die "native runtime-cache seed is empty"
  sudo -n cp -a --reflink=auto -- "$source_cache/." "$destination_cache/" || \
    die "could not snapshot the native runtime-cache seed"
  normalize_container_artifact_permissions "$destination_cache"
  destination_count="$(cache_file_count_cheap "$destination_cache")"
  [[ "$destination_count" -eq "$source_count" ]] || \
    die "native runtime-cache snapshot file census differs"
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
    die "control container name became occupied: $CONTAINER_NAME"
  fi
  run_output="$OUT/run${run}-container-output"
  run_cache="$OUT/run${run}-native-mcg-runtime-cache"
  run_hf_cache="$OUT/run${run}-native-mcg-hf-cache"
  run_cidfile="$OUT/run${run}-native-mcg-container.cid"
  mkdir -m 0700 "$run_output" "$run_cache" "$run_hf_cache"
  cache_seed_receipt="$OUT/run${run}-runtime-cache-seed-receipt.json"
  if [[ "$SNAPSHOT_SEED_RUNTIME_CACHE" == 1 ]]; then
    if [[ "$run" -eq 1 ]]; then
      cache_seed_source="$NATIVE_BOOT1_CACHE_SEED"
    else
      cache_seed_source="$OUT/run1-native-mcg-runtime-cache"
    fi
    snapshot_runtime_cache \
      "$cache_seed_source" "$run_cache" "$cache_seed_receipt"
  fi
  cp -a --reflink=auto \
    "$WIKITEXT_CACHE_SOURCE" \
    "$run_hf_cache/Salesforce___wikitext"
  (
    cd "$run_hf_cache/Salesforce___wikitext"
    sha256sum -c "$WIKITEXT_CACHE_EVIDENCE" >/dev/null
  )
  ACTIVE_CIDFILE="$run_cidfile"
  position_name="run${run}-position-kld.safetensors"
  position_container="/results/$position_name"
  position_host="$run_output/$position_name"
  position_validation="$OUT/run${run}-position-kld.validation.json"
  raw_record="$OUT/run${run}-record.raw.json"
  accepted_record="$OUT/run${run}-record.accepted.json"
  set +e
  docker run --rm --name "$CONTAINER_NAME" --cidfile "$run_cidfile" \
    --gpus all --runtime nvidia --ipc host --network host --shm-size 64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$PROTECTED_MODEL:/model:ro" \
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
  [[ "$tee_status" -eq 0 ]] || die "tee failed in native-MCG run $run"

  record="$(sed -n 's/^fallback_prefill_kld_done //p' \
    "$OUT/run${run}.log" | tail -n 1)"
  [[ -n "$record" ]] || die "native-MCG run $run emitted no final KLD record"
  printf '%s\n' "$record" > "$raw_record"
  jq -e --arg path "$position_container" \
    --arg token_sha256 "$REFERENCE_TOKEN_IDS_U32LE_SHA256" '
    .total_positions == 2047 and .mean_kld >= 0 and .elapsed_sec > 0 and
    .token_ids_u32le_sha256 == $token_sha256 and
    .per_position.path == $path and .per_position.positions == 2047 and
    .per_position.tensor == "kld_ref_to_model" and
    (.per_position.sha256 | test("^[0-9a-f]{64}$"))
  ' "$raw_record" >/dev/null

  run_log="$OUT/run${run}.log"
  [[ "$(grep -Fc 'EXL3 SQG layer model.layers.' "$run_log" || true)" -eq 0 ]] || \
    die "native-MCG run $run entered SQG dispatch"
  dispatch_proof="$OUT/run${run}-native-mcg-dispatch-proof.json"
  native_counts=()
  for layer in 6 28 52 77; do
    line="EXL3 native-MCG control layer model.layers.${layer}.mlp.experts: retaining 768 exclusive MCG tensors on native nonfused dispatch"
    count="$(grep -Fc "$line" "$run_log" || true)"
    [[ "$count" -ge 1 ]] || \
      die "native-MCG run $run did not prove native dispatch for layer $layer"
    native_counts+=( "$count" )
  done
  reservation_counts=()
  for layer in 6 28 52; do
    line="EXL3 native-MCG control layer model.layers.${layer}.mlp.experts: reserving preserved R7 fused slot"
    count="$(grep -Fc "$line" "$run_log" || true)"
    [[ "$count" -ge 1 ]] || \
      die "native-MCG run $run did not prove slot reservation for layer $layer"
    reservation_counts+=( "$count" )
  done
  accounting_line='R7 preserved fused accounting closed: actual=45 reserved=3 total=48 reserved_layers=[6, 28, 52]'
  accounting_count="$(grep -Fc "$accounting_line" "$run_log" || true)"
  [[ "$accounting_count" -ge 1 ]] || \
    die "native-MCG run $run did not prove exact 45+3 fused accounting"
  run_log_sha256="$(sha256sum "$run_log" | awk '{print $1}')"
  jq -n \
    --arg log "$run_log" --arg log_sha256 "$run_log_sha256" \
    --argjson n6 "${native_counts[0]}" --argjson n28 "${native_counts[1]}" \
    --argjson n52 "${native_counts[2]}" --argjson n77 "${native_counts[3]}" \
    --argjson r6 "${reservation_counts[0]}" \
    --argjson r28 "${reservation_counts[1]}" \
    --argjson r52 "${reservation_counts[2]}" \
    --argjson accounting "$accounting_count" '
    {
      schema: "glm52-native-mcg-dispatch-proof-v1",
      log: $log, log_sha256: $log_sha256,
      selected_layers: [6,28,52,77],
      native_dispatch_records: [
        {layer:6,count:$n6},{layer:28,count:$n28},
        {layer:52,count:$n52},{layer:77,count:$n77}],
      reservation_records: [
        {layer:6,count:$r6},{layer:28,count:$r28},{layer:52,count:$r52}],
      fused_accounting: {actual:45,reserved:3,total:48,
        reserved_layers:[6,28,52],count:$accounting},
      sqg_dispatch_records: 0,
      native_mcg_dispatch_proved: true
    }
  ' > "$dispatch_proof"
  dispatch_proof_sha256="$(sha256sum "$dispatch_proof" | awk '{print $1}')"
  [[ "$docker_status" -eq 0 ]] || \
    die "native-MCG run $run failed with Docker exit $docker_status"
  [[ -z "$(find "$run_cache" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \) \
    -print -quit)" ]] || \
    die "native-MCG run $run runtime cache contains a special file"
  cache_file_count="$(cache_file_count_cheap "$run_cache")"
  cache_receipt="$OUT/run${run}-runtime-cache-receipt.json"
  if [[ -f "$cache_seed_receipt" ]]; then
    jq --argjson post_run_file_count "$cache_file_count" \
      --argjson source_arm_candidate "$([[ "$run" -eq 1 ]] && printf true || printf false)" \
      '. + {
        started_empty:false,seeded_snapshot:true,isolated_per_run:true,
        fresh_container_python_engine_workers_model_load:true,
        destination_file_count_after_boot:$post_run_file_count,
        cross_arm_candidate_cache_seed:$source_arm_candidate
      }' "$cache_seed_receipt" > "$cache_receipt"
  else
    jq -n --arg path "$run_cache" --argjson file_count "$cache_file_count" '
      {
        schema:"glm52-runtime-cache-receipt-v1",path:$path,
        started_empty:true,seeded_snapshot:false,isolated_per_run:true,
        fresh_container_python_engine_workers_model_load:true,
        file_count_after_boot:$file_count,byte_hashing_skipped:true,
        cross_arm_candidate_cache_seed:false,
        cache_semantics:"compiled executable code only; no model, logits, KV, RNG, or process state"
      }
    ' > "$cache_receipt"
  fi
  cache_receipt_sha256="$(sha256sum "$cache_receipt" | awk '{print $1}')"

  position_sha256="$(jq -er '.per_position.sha256' "$raw_record")"
  position_mean="$(jq -er '.mean_kld' "$raw_record")"
  [[ -f "$position_host" && ! -L "$position_host" ]] || \
    die "native-MCG run $run did not publish a real per-position tensor"
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
    --arg run_log "$run_log" --arg run_log_sha256 "$run_log_sha256" \
    --arg dispatch_proof "$dispatch_proof" \
    --arg dispatch_proof_sha256 "$dispatch_proof_sha256" \
    --arg cache_receipt "$cache_receipt" \
    --arg cache_receipt_sha256 "$cache_receipt_sha256" \
    --slurpfile runtime_cache "$cache_receipt" '
    .raw_record_sha256 = $raw_record_sha256 |
    .docker_exit_status = 0 |
    .runtime_cache = ($runtime_cache[0] + {
      receipt:$cache_receipt,receipt_sha256:$cache_receipt_sha256
    }) |
    .runtime_dispatch = {
      log: $run_log, log_sha256: $run_log_sha256,
      proof: $dispatch_proof, proof_sha256: $dispatch_proof_sha256,
      selected_layers: [6,28,52,77], native_mcg_dispatch_proved: true,
      sqg_dispatch_records: 0,
      fused_accounting: {actual:45,reserved:3,total:48,
        reserved_layers:[6,28,52]}
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
    printf '%s  %s\n' "$position_validation_sha256" "$position_validation"
    printf '%s  %s\n' "$run_log_sha256" "$run_log"
    printf '%s  %s\n' "$dispatch_proof_sha256" "$dispatch_proof"
    printf '%s  %s\n' "$accepted_record_sha256" "$accepted_record"
  } >> "$OUT/per-position-evidence.sha256"
  sha256sum -c "$OUT/per-position-evidence.sha256" >/dev/null
  ACCEPTED_RECORDS+=( "$accepted_record" )
done

verify_sealed_eval_inputs
[[ "$(wc -l < "$OUT/per-position-evidence.sha256")" -eq $((RUNS * 5)) ]] || \
  die "native-MCG per-position evidence census differs"
sha256sum -c "$OUT/per-position-evidence.sha256" >/dev/null
POSITION_EVIDENCE_SHA256="$(sha256sum \
  "$OUT/per-position-evidence.sha256" | awk '{print $1}')"

: > "$OUT/runs.jsonl"
for accepted_record in "${ACCEPTED_RECORDS[@]}"; do
  jq -c . "$accepted_record" >> "$OUT/runs.jsonl"
done
RUNS_JSONL_SHA256="$(sha256sum "$OUT/runs.jsonl" | awk '{print $1}')"

BASELINE_SUMMARY=/home/brandonmusic/klc-linux/release-gg-sparkinfer-20260721/results/20260809T194958Z-kld-fp8-dcp4/summary.json
BASELINE_SUMMARY_SHA256=07096cd5f0a683bd00a1e169c547ef48eaaec8f6270dfce355fa7b0a06834ad4
printf '%s  %s\n' "$BASELINE_SUMMARY_SHA256" "$BASELINE_SUMMARY" | sha256sum -c -
BASELINE_MEAN="$(jq -er '.mean_kld' "$BASELINE_SUMMARY")"
BASELINE_SD="$(jq -er '.sample_sd_kld' "$BASELINE_SUMMARY")"

jq -s -e \
  --arg model "$PROTECTED_MODEL" \
  --arg image "$IMAGE" --arg image_id "$RUNTIME_IMAGE_ID" \
  --arg overlay "$RUNTIME_OVERLAY" --arg overlay_sha256 "$OVERLAY_MANIFEST_SHA256" \
  --arg native_args "$NATIVE_ARGS" --arg native_args_sha256 "$NATIVE_ARGS_SHA256" \
  --arg reference_sha256 "$REFERENCE_SHA256" \
  --arg token_sha256 "$REFERENCE_TOKEN_IDS_U32LE_SHA256" \
  --arg preflight "$OUT/native-mcg-preflight.json" \
  --arg preflight_sha256 "$CONTROL_PREFLIGHT_SHA256" \
  --arg model_evidence "$CONTROL_MODEL_EVIDENCE" \
  --arg model_evidence_sha256 "$CONTROL_MODEL_EVIDENCE_SHA256" \
  --arg runtime_evidence "$CONTROL_RUNTIME_EVIDENCE" \
  --arg runtime_evidence_sha256 "$CONTROL_RUNTIME_EVIDENCE_SHA256" \
  --argjson runtime_file_count "$CONTROL_RUNTIME_FILE_COUNT" \
  --arg position_evidence "$OUT/per-position-evidence.sha256" \
  --arg position_evidence_sha256 "$POSITION_EVIDENCE_SHA256" \
  --arg runs_jsonl "$OUT/runs.jsonl" --arg runs_jsonl_sha256 "$RUNS_JSONL_SHA256" \
  --arg baseline_summary "$BASELINE_SUMMARY" \
  --argjson baseline_mean "$BASELINE_MEAN" --argjson baseline_sd "$BASELINE_SD" \
  --argjson directional_test_fast "$DIRECTIONAL_TEST_FAST_JSON" \
  --argjson expected_runs "$RUNS" '
  if length != $expected_runs then error("native-MCG run count mismatch")
  elif (all(.[].per_position;
      .positions == 2047 and .tensor == "kld_ref_to_model" and
      .independently_validated == true) | not) then
    error("native-MCG per-position evidence differs")
  elif (all(.[].runtime_dispatch;
      .native_mcg_dispatch_proved == true and .sqg_dispatch_records == 0 and
      .selected_layers == [6,28,52,77] and
      .fused_accounting == {actual:45,reserved:3,total:48,
        reserved_layers:[6,28,52]}) | not) then
    error("native-MCG runtime dispatch evidence differs")
  else
    map(.mean_kld) as $values |
    ($values | add / length) as $mean |
    ([$values[] as $x | (($x - $mean) * ($x - $mean))]
      | add / (length - 1) | sqrt) as $sd |
    {
      schema: "glm52-native-mcg-control-kld-result-v1",
      directional_test_fast_mode: $directional_test_fast,
      protected_model: $model,
      protected_model_mount: "read_only",
      candidate_mounted: false,
      reference_sha256: $reference_sha256,
      reference_token_ids_u32le_sha256: $token_sha256,
      selected_layers: [6,28,52,77],
      selected_codebook: "mcg",
      runtime: {
        image: $image, image_id: $image_id,
        runtime_overlay: {path:$overlay,manifest_sha256:$overlay_sha256},
        native_args: {path:$native_args,sha256:$native_args_sha256}
      },
      regime: {
        kv_cache_dtype:"fp8",rope:"bfloat16",tensor_parallel_size:4,
        decode_context_parallel_size:4,dcp_comm_backend:"a2a",
        dcp_kv_cache_interleave_size:64,context_tokens:2048,
        scored_positions:2047
      },
      model_preflight: {
        path:$preflight,sha256:$preflight_sha256,
        selected_file_evidence:$model_evidence,
        selected_file_evidence_sha256:$model_evidence_sha256,
        complete_runtime_evidence:$runtime_evidence,
        complete_runtime_evidence_sha256:$runtime_evidence_sha256,
        complete_runtime_file_count:$runtime_file_count,
        rehashed_before_every_launch:($directional_test_fast | not)
      },
      runtime_dispatch: {
        native_mcg_dispatch_proved:true,sqg_dispatch_records:0,
        selected_layers:[6,28,52,77],
        fused_accounting:{actual:45,reserved:3,total:48,
          reserved_layers:[6,28,52]},
        per_run:map(.runtime_dispatch)
      },
      control_result: {
        runs:length,values:$values,mean_kld:$mean,sample_sd_kld:$sd,
        min_kld:($values|min),max_kld:($values|max),
        elapsed_seconds:map(.elapsed_sec),
        paired_per_position_outputs:map(.per_position),
        paired_per_position_evidence:{
          manifest:$position_evidence,manifest_sha256:$position_evidence_sha256},
        runs_jsonl:{path:$runs_jsonl,sha256:$runs_jsonl_sha256}
      },
      preserved_fused_baseline: {
        summary:$baseline_summary,runs:5,mean_kld:$baseline_mean,
        sample_sd_kld:$baseline_sd
      },
      dispatch_effect_vs_fused_baseline: {
        delta_native_mcg_minus_fused_baseline:($mean-$baseline_mean),
        relative_delta:(($mean-$baseline_mean)/$baseline_mean),
        mean_direction:(if $mean < $baseline_mean then "lower"
          elif $mean > $baseline_mean then "higher" else "equal" end),
        causal_limit:
          "Use the paired SQG-vs-native-MCG analyzer for the treatment comparison; scalar baseline subtraction alone is not causal"
      },
      inference_limit:
        "One fixed 2047-position prompt; repeats estimate runtime variation, not text/model generalization"
    }
  end
' "${ACCEPTED_RECORDS[@]}" > "$OUT/summary.json.partial"

sha256sum -c "$OUT/per-position-evidence.sha256" >/dev/null
mv "$OUT/summary.json.partial" "$OUT/summary.json"
jq . "$OUT/summary.json" | tee "$OUT/summary.txt"
printf '%s\n' "$OUT"
