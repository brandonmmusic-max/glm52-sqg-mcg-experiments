#!/usr/bin/env bash
set -euo pipefail

# One diagnostic trace of the unchanged protected r33 checkpoint.  This is not
# a replacement native-dispatch control and is deliberately limited to one
# boot; its purpose is to pair routing/residual tensors with an SQG trace.
# The selected late layers 74--77 are naturally outside r33's fused allowlist,
# so they execute native/nonfused MCG without a special dispatch override.

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

[[ $# -eq 1 && "$1" = /* ]] || \
  die "usage: $0 /absolute/new-output-directory"

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test
PROTECTED_MODEL=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED
PRODUCTION_CONTAINER=glm-r33-fixed
OUTPUT="$1"
TRACE_OVERLAY=/home/brandonmusic/KLC_SANDBOXES/sqg-tail-trace-overlay-r1
EXTRA_ARGS_FILE="$PROJECT_ROOT/evaluation/r33_overlay_historical.args"
IMAGE=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
SITE=/opt/venv/lib/python3.12/site-packages
PYTHON_ENTRYPOINT=/opt/venv/bin/python
REFERENCE=/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current/reference_hf/reference-logits
FALLBACK="$PROJECT_ROOT/evaluation/prefill_kld_paired.py"
PYDEPS=/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current/pydeps
RELEASE_ROOT=/home/brandonmusic/klc-linux/release-gg-sparkinfer-20260721
PROMPT_LOGPROB="$RELEASE_ROOT/eval-overlay/vllm/v1/worker/gpu/sample/prompt_logprob.py"
LOGPROB="$RELEASE_ROOT/eval-overlay/vllm/v1/worker/gpu/sample/logprob.py"
CACHE_SEED=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-r1/candidate/fresh-sqg4-r1-candidate-kld-fp8-dcp4/run1-candidate-runtime-cache
WIKITEXT_SOURCE=/home/brandonmusic/.cache/huggingface/datasets/Salesforce___wikitext
CONTAINER_NAME=glm52-r33-late-trace-once
LAYERS="${SQG_TAIL_TRACE_LAYERS:-74,75,76,77}"

[[ "$LAYERS" == 74,75,76,77 ]] || die "trace layers must be exactly 74,75,76,77"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || die "output already exists: $OUTPUT"
for path in \
  "$PROTECTED_MODEL" "$TRACE_OVERLAY" "$REFERENCE" "$PYDEPS" \
  "$CACHE_SEED" "$WIKITEXT_SOURCE"; do
  [[ -d "$path" && ! -L "$path" ]] || die "required directory is absent or unsafe: $path"
done
for path in \
  "$PROTECTED_MODEL/.manifest_verified" "$EXTRA_ARGS_FILE" "$FALLBACK" \
  "$PROMPT_LOGPROB" "$LOGPROB"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required file is absent or unsafe: $path"
done
[[ "$(sha256sum "$TRACE_OVERLAY/SHA256SUMS.runtime-overlay" | awk '{print $1}')" == \
  92a79104c06fd53046448b339a438fce199028e22c888e7a3fdff076337bbe0a ]] || \
  die "trace overlay manifest differs"
(cd "$TRACE_OVERLAY" && sha256sum -c SHA256SUMS.runtime-overlay >/dev/null)
[[ "$(sha256sum "$EXTRA_ARGS_FILE" | awk '{print $1}')" == \
  8c2c10d09c7f0696dc47919210dde0f6db40352b12119fd1815227cac9478f3c ]] || \
  die "historical r33 runtime args differ"
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE")" == "$IMAGE" ]] || \
  die "r33 runtime image differs"
if [[ "$(docker inspect --format '{{.State.Running}}' \
  "$PRODUCTION_CONTAINER" 2>/dev/null || true)" == true ]]; then
  die "production container is running"
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  die "diagnostic container name is already occupied"
fi

busy="$({ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; } | \
  awk '$1 > 2048 {n++} END {print n+0}')"
[[ "$busy" -eq 0 ]] || die "$busy GPU(s) are above the idle-memory limit"

mkdir -m 0700 "$OUTPUT" "$OUTPUT/runtime-cache" "$OUTPUT/hf-datasets"
cp -a --reflink=auto "$CACHE_SEED/." "$OUTPUT/runtime-cache/"
cp -a --reflink=auto "$WIKITEXT_SOURCE" "$OUTPUT/hf-datasets/"
docker image inspect "$IMAGE" > "$OUTPUT/image-inspect.json"
sha256sum "$FALLBACK" "$PROMPT_LOGPROB" "$LOGPROB" \
  "$TRACE_OVERLAY/SHA256SUMS.runtime-overlay" "$EXTRA_ARGS_FILE" \
  > "$OUTPUT/input-code.sha256"

RUNTIME_ARGS=(
  -v "$TRACE_OVERLAY:/sqg-runtime-overlay:ro"
  -e PYTHONPATH=/sqg-runtime-overlay
)
while IFS= read -r extra_arg || [[ -n "$extra_arg" ]]; do
  [[ -z "$extra_arg" || "$extra_arg" == \#* ]] && continue
  RUNTIME_ARGS+=("$extra_arg")
done < "$EXTRA_ARGS_FILE"
RUNTIME_ARGS+=(--env=VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS=none)

PATTERN=FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
HF_OVERRIDES="$(printf \
  '{\"use_index_cache\":true,\"index_topk_pattern\":\"%s\"}' "$PATTERN")"
LLM_EXTRA='{"decode_context_parallel_size":4,"dcp_comm_backend":"a2a","dcp_kv_cache_interleave_size":64,"moe_backend":"b12x","kv_cache_memory_bytes":268435456,"enforce_eager":true,"async_scheduling":false,"disable_custom_all_reduce":true}'
POSITION=/results/r33-position-kld.safetensors

set +e
docker run --rm --name "$CONTAINER_NAME" \
  --gpus all --runtime nvidia --ipc host --network host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PROTECTED_MODEL:/model:ro" \
  -v "$OUTPUT:/results:rw" \
  -v "$REFERENCE:/ref:ro" \
  -v "$FALLBACK:/kld/prefill_kld.py:ro" \
  -v "$PYDEPS:/deps:ro" \
  -v "$PROMPT_LOGPROB:$SITE/vllm/v1/worker/gpu/sample/prompt_logprob.py:ro" \
  -v "$LOGPROB:$SITE/vllm/v1/worker/gpu/sample/logprob.py:ro" \
  -v /home/brandonmusic/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "$OUTPUT/hf-datasets:/hf-datasets:rw" \
  -v "$OUTPUT/runtime-cache:/cache:rw" \
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
  -e SQG_TAIL_TRACE=1 \
  -e "SQG_TAIL_TRACE_LAYERS=$LAYERS" \
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
    --model /model --reference-logits /ref \
    --context-length 2048 --stride 512 --max-windows 1 \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.90 \
    --dtype bfloat16 --kv-cache-dtype fp8 --load-format safetensors \
    --max-model-len 2560 --max-num-batched-tokens 2048 --max-num-seqs 1 \
    --quantization exl3 --attention-backend B12X_MLA_SPARSE \
    --hf-overrides "$HF_OVERRIDES" --llm-extra-json "$LLM_EXTRA" \
    --kld-chunk-rows 32 --per-position-output "$POSITION" \
  2>&1 | tee "$OUTPUT/run.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
[[ "${pipeline_status[1]}" -eq 0 ]] || die "tee failed"
[[ "${pipeline_status[0]}" -eq 0 ]] || die "diagnostic container failed"

record="$(sed -n 's/^fallback_prefill_kld_done //p' "$OUTPUT/run.log" | tail -n 1)"
[[ -n "$record" ]] || die "diagnostic run emitted no KLD record"
printf '%s\n' "$record" > "$OUTPUT/record.json"
jq -e '
  .total_positions == 2047 and .mean_kld >= 0 and
  .per_position.positions == 2047 and
  .token_ids_u32le_sha256 == "ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56"
' "$OUTPUT/record.json" >/dev/null || die "diagnostic KLD record differs"
[[ -f "$OUTPUT/r33-position-kld.safetensors" ]] || die "per-position tensor is absent"
sudo -n chown -R "$(id -un):$(id -gn)" \
  "$OUTPUT/r33-position-kld.safetensors" "$OUTPUT/tail-trace" || \
  die "could not make container trace evidence host-readable"
for layer in 074 075 076 077; do
  for rank in 000 001 002 003; do
    compgen -G "$OUTPUT/tail-trace/layer-$layer-rank-$rank-call-*.safetensors" \
      >/dev/null || die "trace lacks layer $layer DCP rank $rank"
  done
done
position_sha="$(sha256sum "$OUTPUT/r33-position-kld.safetensors" | awk '{print $1}')"
python3 "$PROJECT_ROOT/evaluation/validate_per_position_kld.py" \
  "$OUTPUT/r33-position-kld.safetensors" \
  --expected-sha256 "$position_sha" \
  --expected-mean-kld "$(jq -r .mean_kld "$OUTPUT/record.json")" \
  > "$OUTPUT/per-position-validation.json"
sha256sum "$OUTPUT/record.json" "$OUTPUT/r33-position-kld.safetensors" \
  "$OUTPUT/per-position-validation.json" "$OUTPUT"/tail-trace/*.safetensors \
  > "$OUTPUT/evidence.sha256"
printf 'r33 late trace complete: %s\n' "$OUTPUT"
