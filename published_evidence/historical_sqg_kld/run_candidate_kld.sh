#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "usage: $0 /absolute/path/to/candidate-model"

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$EXPERIMENT_DIR/run_candidate_kld.sh"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results}"
PRODUCTION_MODEL=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED
PRODUCTION_CONTAINER=glm-r33-fixed
CANDIDATE="$(realpath -e -- "$1")"
EXPECTED_SQG_LAYERS="${EXPECTED_SQG_LAYERS:-6,28,52,77}"
PURE_SQG_VALIDATOR="$EXPERIMENT_DIR/validate_pure_sqg_candidate.py"

[[ -d "$CANDIDATE" ]] || die "candidate is not a directory: $CANDIDATE"
case "$CANDIDATE" in
  "$PRODUCTION_MODEL"|"$PRODUCTION_MODEL"/*)
    die "refusing to evaluate the protected production checkpoint in place"
    ;;
esac

for required in .manifest_verified config.json model.safetensors.index.json; do
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
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || \
  die "PREFLIGHT_ONLY must be 0 or 1"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)-sqg}"
[[ "$STAMP" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "STAMP may contain only letters, digits, dot, underscore, and dash"
OUT="$RESULTS_ROOT/$STAMP-candidate-kld-fp8-dcp4"
[[ ! -e "$OUT" ]] || die "result path already exists: $OUT"
mkdir -p "$OUT"

CONTAINER_NAME="${CONTAINER_NAME:-glm52-sqg-kld-${STAMP:0:36}}"
[[ "$CONTAINER_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] || \
  die "invalid Docker container name: $CONTAINER_NAME"

ROOT=/home/brandonmusic/klc-linux/release-gg-sparkinfer-20260721
KLD=/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current
REFERENCE="$KLD/reference_hf/reference-logits"
REFERENCE_LOGITS="$REFERENCE/logits_0.safetensors"
REFERENCE_MANIFEST="$REFERENCE/manifest.json"
REFERENCE_SHA256=87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063
REFERENCE_MANIFEST_SHA256=985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606
FALLBACK_SHA256=4fc1d276bb895e09fb1a8ab57c97e851e20abb1c3fe5a9bee598231e23fb0c27
PROMPT_LOGPROB_SHA256=47f867c3ff81cc1778bae3f1a3189dd2ec90d6e469cf60f7d1e43a1c76989d6c
LOGPROB_SHA256=21d98eea20b8c92e0a65a5badffc3782dfc5a77427470ecd5d9edcedc681a041
FALLBACK="$KLD/prefill_kld_fallback_hybrid.py"
PROMPT_LOGPROB="$ROOT/eval-overlay/vllm/v1/worker/gpu/sample/prompt_logprob.py"
LOGPROB="$ROOT/eval-overlay/vllm/v1/worker/gpu/sample/logprob.py"

printf '%s  %s\n' "$REFERENCE_SHA256" "$REFERENCE_LOGITS" | sha256sum -c -
printf '%s  %s\n' "$REFERENCE_MANIFEST_SHA256" "$REFERENCE_MANIFEST" | \
  sha256sum -c -
printf '%s  %s\n' "$FALLBACK_SHA256" "$FALLBACK" | sha256sum -c -
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
if [[ -f "$CANDIDATE/LOCAL_CORRECTED_BUILD.json" ]]; then
  aux_base="$(jq -r '.unchanged_base // empty' \
    "$CANDIDATE/LOCAL_CORRECTED_BUILD.json")"
  aux_routed="$(jq -r '.corrected_routed // empty' \
    "$CANDIDATE/LOCAL_CORRECTED_BUILD.json")"
  [[ -z "$aux_base" || -d "$aux_base" ]] || \
    die "candidate auxiliary base is missing: $aux_base"
  [[ -z "$aux_routed" || -d "$aux_routed" ]] || \
    die "candidate auxiliary routed directory is missing: $aux_routed"
  [[ -z "$aux_base" ]] || MODEL_MOUNTS+=( -v "$aux_base:$aux_base:ro" )
  [[ -z "$aux_routed" ]] || \
    MODEL_MOUNTS+=( -v "$aux_routed:$aux_routed:ro" )
fi

RUNTIME_ARGS=()
runtime_overlay=""
extra_args_file=""
if [[ -n "${RUNTIME_OVERLAY:-}" ]]; then
  runtime_overlay="$(realpath -e -- "$RUNTIME_OVERLAY")"
  [[ -d "$runtime_overlay" ]] || \
    die "RUNTIME_OVERLAY must be a directory: $runtime_overlay"
  RUNTIME_ARGS+=(
    -v "$runtime_overlay:/sqg-runtime-overlay:ro"
    -e "PYTHONPATH=/sqg-runtime-overlay:/deps"
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

[[ -n "$runtime_overlay" ]] || die "RUNTIME_OVERLAY is required for SQG"
[[ -z "${EXL3_EXT_SO:-}" ]] || \
  die "EXL3_EXT_SO must not bypass the sealed exact-r33 SQG overlay"
OVERLAY_MANIFEST="$runtime_overlay/SHA256SUMS.runtime-overlay"
OVERLAY_MANIFEST_SHA256=b1908926d7cba6ee92b9e373b0d6c9deae9a7493c83a3e0e9ee720a5abce9a47
[[ -f "$OVERLAY_MANIFEST" ]] || die "runtime overlay manifest is missing"
printf '%s  %s\n' "$OVERLAY_MANIFEST_SHA256" "$OVERLAY_MANIFEST" | \
  sha256sum -c -
(
  cd "$runtime_overlay"
  sha256sum -c SHA256SUMS.runtime-overlay >/dev/null
)
[[ ! -e "$runtime_overlay/b12x" ]] || \
  die "runtime overlay must not shadow the production b12x package"

[[ -n "$extra_args_file" ]] || \
  die "EXTRA_DOCKER_ARGS_FILE is required for exact-r33 boot parity"
R33_EXTRA_ARGS_SHA256=ec2208c1958971bf14cfbdd526ec62f02f72ebb2964de55891d0884213490bdc
printf '%s  %s\n' "$R33_EXTRA_ARGS_SHA256" "$extra_args_file" | \
  sha256sum -c -

python3 "$PURE_SQG_VALIDATOR" "$CANDIDATE" \
  --source "$PRODUCTION_MODEL" --layers "$EXPECTED_SQG_LAYERS" \
  > "$OUT/candidate-pure-sqg-preflight.json"
jq -e '
  .selected_layers == [6,28,52,77] and
  .selected_trellis_tensors == 3072 and
  .selected_sqg_markers == 3072 and
  .selected_mcg_markers == 0 and
  .sqg_markers_outside_selected_layers == 0 and
  .marker_payloads_verified == true and
  .tensor_overrides == {}
' "$OUT/candidate-pure-sqg-preflight.json" >/dev/null

candidate_provenance="$(realpath -e -- "${CANDIDATE_PROVENANCE:-/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_no_bf16_test/offline_codec/results/final-candidate-n4-audit.json}")"
CANDIDATE_PROVENANCE_SHA256=ee678a8471f8e9da99bdbcf6791b3b3f20c440d79c87f73daa54616610ce5552
printf '%s  %s\n' "$CANDIDATE_PROVENANCE_SHA256" "$candidate_provenance" | \
  sha256sum -c -
jq -e --arg candidate "$CANDIDATE" '
  .valid == true and .candidate == $candidate and
  .aggregate.selected_tensors == 3072 and
  .aggregate.sqg_converted_tensors == 3072 and
  .aggregate.mcg_k5_passthrough_tensors == 0 and
  .closures.all_selected_layers_exclusive_sqg == true and
  .closures.tensor_override_set_empty == true
' "$candidate_provenance" >/dev/null

docker image inspect "$IMAGE" > "$OUT/image-inspect.json"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "$OUT/gpu-preflight.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "$OUT/gpu-compute-preflight.csv"
printf '%s\n' "$CANDIDATE" > "$OUT/candidate-path.txt"
sha256sum \
  "$CANDIDATE/.manifest_verified" \
  "$CANDIDATE/config.json" \
  "$CANDIDATE/model.safetensors.index.json" \
  > "$OUT/candidate-metadata.sha256"
if [[ -f "$CANDIDATE/SQG_CANDIDATE.json" ]]; then
  sha256sum "$CANDIDATE/SQG_CANDIDATE.json" >> \
    "$OUT/candidate-metadata.sha256"
fi
sha256sum "$candidate_provenance" > "$OUT/candidate-provenance.sha256"
sha256sum "$RUNNER" "$PURE_SQG_VALIDATOR" \
  "$FALLBACK" "$PROMPT_LOGPROB" "$LOGPROB" > \
  "$OUT/eval-code.sha256"
sha256sum "$OUT/candidate-pure-sqg-preflight.json" > \
  "$OUT/candidate-pure-sqg-preflight.sha256"
sha256sum "$RUNTIME_BASELINE_EVIDENCE" > \
  "$OUT/runtime-baseline-evidence-manifest.sha256"
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
  --arg baseline_evidence_manifest "$RUNTIME_BASELINE_EVIDENCE" \
  --arg baseline_evidence_sha256 "$RUNTIME_BASELINE_EVIDENCE_SHA256" \
  --arg reference_sha256 "$REFERENCE_SHA256" \
  --argjson runs "$RUNS" \
  --argjson kld_util "$KLD_UTIL" \
  '{
    schema: "glm52-sqg-candidate-kld-run-v1",
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
    baseline_evidence: {
      manifest: $baseline_evidence_manifest,
      sha256: $baseline_evidence_sha256
    },
    runtime_code_matches_baseline: false,
    comparison_scope:
      "SQG codebook plus required SQG loader/non-fused dispatch; not codebook-only",
    selected_sqg_layers: [6,28,52,77],
    selected_layer_payloads: {
      sqg: 3072,
      mcg: 0,
      tensor_overrides: 0
    },
    reference_sha256: $reference_sha256,
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
    requested_candidate_runs: $runs,
    inference_limit:
      "One fixed 2047-position prompt; repeat runs estimate runtime variation, not text/model generalization"
  }' > "$OUT/run-manifest.json"
: > "$OUT/runs.jsonl"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  jq -e '.selected_layer_payloads.mcg == 0 and
    .runtime_code_matches_baseline == false' \
    "$OUT/run-manifest.json" >/dev/null
  printf '%s\n' "$OUT"
  exit 0
fi

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for run in $(seq 1 "$RUNS"); do
  cleanup
  require_idle_gpus
  set +e
  docker run --rm --name "$CONTAINER_NAME" \
    --gpus all --runtime nvidia --ipc host --network host --shm-size 64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${MODEL_MOUNTS[@]}" \
    -v "$REFERENCE:/ref:ro" \
    -v "$FALLBACK:/kld/prefill_kld.py:ro" \
    -v "$KLD/pydeps:/deps:ro" \
    -v "$PROMPT_LOGPROB:$SITE/vllm/v1/worker/gpu/sample/prompt_logprob.py:ro" \
    -v "$LOGPROB:$SITE/vllm/v1/worker/gpu/sample/logprob.py:ro" \
    -v /home/brandonmusic/.cache/huggingface:/root/.cache/huggingface:rw \
    -v /home/brandonmusic/.cache/glm52-tr3-release:/cache:rw \
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
    2>&1 | tee "$OUT/run${run}.log"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  docker_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"

  [[ "$tee_status" -eq 0 ]] || die "tee failed in candidate run $run"
  record="$(sed -n 's/^fallback_prefill_kld_done //p' \
    "$OUT/run${run}.log" | tail -n 1)"
  [[ -n "$record" ]] || die "candidate run $run emitted no final KLD record"
  printf '%s\n' "$record" | jq -e \
    '.total_positions == 2047 and .mean_kld >= 0 and .elapsed_sec > 0' \
    >/dev/null
  for sqg_layer in 6 28 52 77; do
    grep -Fq \
      "EXL3 SQG layer model.layers.${sqg_layer}.mlp.experts: retaining 768 per-projection native tensors" \
      "$OUT/run${run}.log" || \
      die "candidate run $run did not prove SQG dispatch for layer $sqg_layer"
  done
  if [[ "$docker_status" -ne 0 && "$docker_status" -ne 139 ]]; then
    die "candidate run $run failed with Docker exit $docker_status"
  fi
  printf '%s\n' "$record" >> "$OUT/runs.jsonl"
done

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
  --arg reference_sha256 "$REFERENCE_SHA256" \
  --argjson baseline_mean "$RUNTIME_BASELINE_MEAN" \
  --argjson baseline_sd "$RUNTIME_BASELINE_SD" \
  --argjson baseline_runs "$RUNTIME_BASELINE_RUNS" \
  --argjson legacy_mean "$LEGACY_BASELINE_MEAN" \
  --argjson legacy_sd "$LEGACY_BASELINE_SD" \
  --argjson expected_runs "$RUNS" '
  if length != $expected_runs then
    error("candidate run count mismatch")
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
      schema: "glm52-sqg-candidate-kld-result-v1",
      baseline_was_rerun: false,
      candidate: $candidate,
      reference_sha256: $reference_sha256,
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
        runtime_code_matches_baseline: false
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
        elapsed_seconds: map(.elapsed_sec)
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
' "$OUT/runs.jsonl" > "$OUT/summary.json"

jq . "$OUT/summary.json" | tee "$OUT/summary.txt"
printf '%s\n' "$OUT"
