#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

paths_overlap() {
  local left=${1%/}/ right=${2%/}/
  [[ "$left" == "$right" || "$left" == "$right"* || "$right" == "$left"* ]]
}

canonical_directory() {
  local label=$1 source=$2 resolved
  if ! resolved=$(realpath -e -- "$source"); then
    die "$label cannot be resolved to an existing canonical path: $source"
  fi
  [[ -d "$resolved" ]] || die "$label is not a directory: $resolved"
  printf '%s\n' "$resolved"
}

reject_filesystem_root() {
  local label=$1 path=$2
  [[ "$path" != / ]] || die "$label must not be the filesystem root"
}

assert_disjoint_from_protected_roots() {
  local writable_label=$1 writable_path=$2 index
  for index in "${!protected_roots[@]}"; do
    if paths_overlap "$writable_path" "${protected_roots[$index]}"; then
      die "$writable_label must be disjoint from protected ${protected_labels[$index]} root: ${protected_roots[$index]}"
    fi
  done
}

[[ $# -eq 1 ]] || die "usage: FRESH_CAPTURE_PARENT=... FRESH_JIT_CACHE=... $0 smoke|full"
mode=$1
[[ "$mode" == smoke || "$mode" == full ]] || die "mode must be smoke or full"

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
model=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED
runtime_root=/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork
vllm_source=/home/brandonmusic/KLC_SANDBOXES/.r10_prompt_logits_port/vllm
image=voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82

# Resolve both writable destinations and every read-only host root before doing
# any ancestry comparison.  This makes aliases through symlinks compare by
# their actual target and protects every tree used as a bind-mount source.  The
# runtime root includes the mounted vLLM/B12X boot patches and EXL3 extension;
# vllm_source is the separately mounted installed-vLLM source tree.
project=$(canonical_directory "project/work" "$project")
model=$(canonical_directory "teacher model" "$model")
runtime_overlay=$(canonical_directory \
  "evaluation runtime overlay" "$project/evaluation/runtime_overlay")
runtime_root=$(canonical_directory "mounted B12X/runtime" "$runtime_root")
vllm_source=$(canonical_directory "mounted vLLM source" "$vllm_source")
plan=$project/evidence/document_plan.json

[[ -n "${FRESH_CAPTURE_PARENT:-}" ]] || die "FRESH_CAPTURE_PARENT is required"
[[ -n "${FRESH_JIT_CACHE:-}" ]] || die "FRESH_JIT_CACHE is required"
capture_parent=$(canonical_directory "capture parent" "$FRESH_CAPTURE_PARENT")
jit_cache=$(canonical_directory "JIT cache" "$FRESH_JIT_CACHE")
reject_filesystem_root "capture parent" "$capture_parent"
reject_filesystem_root "JIT cache" "$jit_cache"

protected_labels=(
  "teacher model"
  "project/work"
  "evaluation runtime overlay"
  "mounted vLLM source"
  "mounted B12X/runtime"
)
protected_roots=(
  "$model"
  "$project"
  "$runtime_overlay"
  "$vllm_source"
  "$runtime_root"
)
assert_disjoint_from_protected_roots "capture parent" "$capture_parent"
assert_disjoint_from_protected_roots "JIT cache" "$jit_cache"
paths_overlap "$capture_parent" "$jit_cache" && \
  die "capture and JIT paths must be disjoint (no ancestry overlap)"
[[ -z "$(find "$capture_parent" -mindepth 1 -print -quit)" ]] || \
  die "capture parent must be empty: $capture_parent"
[[ -f "$plan" ]] || die "sealed document plan is absent: $plan"
preflight_mount=()
preflight_args=()

if [[ "$mode" == full ]]; then
  [[ -n "${FRESH_DCP4_SMOKE_DIR:-}" ]] || \
    die "full mode requires FRESH_DCP4_SMOKE_DIR"
  smoke_dir=$(realpath -e -- "$FRESH_DCP4_SMOKE_DIR")
  paths_overlap "$capture_parent" "$smoke_dir" && \
    die "full capture parent must be disjoint from smoke evidence"
  paths_overlap "$jit_cache" "$smoke_dir" && \
    die "JIT cache must be disjoint from smoke evidence"
  smoke_token=$(/usr/bin/python3 - \
    "$smoke_dir" "$jit_cache" "$plan" "$project" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[4])
from src.calibration_capture import (
    validate_dcp4_smoke_evidence,
    validate_jit_cache_binding,
)

smoke = validate_dcp4_smoke_evidence(
    Path(sys.argv[1]),
    plan_path=Path(sys.argv[3]),
    project_root=Path(sys.argv[4]),
)
validate_jit_cache_binding(
    Path(sys.argv[2]),
    Path(sys.argv[1]),
    plan_path=Path(sys.argv[3]),
    project_root=Path(sys.argv[4]),
)
print(smoke["smoke_run_token"])
PY
  ) || die "strict DCP4 smoke/cache binding validation failed"
  container_name=glm52-fresh-sqg-capture-r1
  capture_mode=--capture
  capture_name=fresh-sqg-calibration-r1
  preflight_mount=(
    --mount "type=bind,src=$smoke_dir,dst=/preflight/smoke,readonly"
  )
  preflight_args=(
    --smoke-dir /preflight/smoke
    --jit-cache-dir /cache/jit
  )
else
  [[ -z "$(find "$jit_cache" -mindepth 1 -print -quit)" ]] || \
    die "DCP4 smoke requires a new empty dedicated JIT cache: $jit_cache"
  smoke_token=$(/usr/bin/python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
  )
  container_name=glm52-fresh-sqg-dcp4-smoke-r1
  capture_mode=--smoke-dcp4
  capture_name=fresh-sqg-dcp4-smoke-r1
fi

docker run --rm --name "$container_name" \
  --network none --ipc host --gpus all --runtime nvidia --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint /opt/venv/bin/python \
  -e CUDA_VISIBLE_DEVICES=3,1,2,0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e OMP_NUM_THREADS=16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONDONTWRITEBYTECODE=1 \
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
  -e KV_FP8_ROPE=0 \
  -e VLLM_EXL3_EXT_PATH=/opt/exllamav3-r7ext \
  -e VLLM_EXL3_R7_FUSED=1 \
  -e VLLM_EXL3_R7_FUSED_LAYERS=48 \
  -e VLLM_EXL3_R7_A1_MIN_ROWS=0 \
  -e VLLM_EXL3_TRELLIS_MIN_M=4 \
  -e VLLM_EXL3_TRELLIS_MAX_M=32 \
  -e VLLM_EXL3_TRELLIS_BLOCK_M=8 \
  -e VLLM_EXL3_PREFILL_CHUNK=128 \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 \
  -e VLLM_EXL3_PREFILL_BLOCK_M=64 \
  -e VLLM_EXL3_PREFILL_CAPACITY=2048 \
  -e VLLM_EXL3_PREFILL_SYMMETRIC_TILE=0 \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e TRITON_CACHE_DIR=/cache/jit/triton \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
  -e XDG_CACHE_HOME=/cache/jit \
  -e FRESH_SQG_RUNTIME_IMAGE_ID=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82 \
  -e FRESH_SQG_SMOKE_RUN_TOKEN="$smoke_token" \
  "${preflight_mount[@]}" \
  --mount type=bind,src="$project",dst=/work,readonly \
  --mount type=bind,src="$model",dst="$model",readonly \
  --mount type=bind,src="$capture_parent",dst=/capture \
  --mount type=bind,src="$jit_cache",dst=/cache/jit \
  --mount type=bind,src="$runtime_root/boot/envs.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/envs.py,readonly \
  --mount type=bind,src="$runtime_root/boot/deepseek_v2.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py,readonly \
  --mount type=bind,src="$runtime_root/boot/exl3.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py,readonly \
  --mount type=bind,src="$runtime_root/boot/utils.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/utils.py,readonly \
  --mount type=bind,src="$runtime_root/boot/mixed_trellis.py",dst=/opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/w4a16/mixed_trellis.py,readonly \
  --mount type=bind,src="$runtime_root/v39_ext/exllamav3",dst=/opt/exllamav3-r7ext,readonly \
  --mount type=bind,src="$vllm_source/model_executor/layers/fused_moe/layer.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/layer.py,readonly \
  --mount type=bind,src="$vllm_source/model_executor/layers/fused_moe/runner/moe_runner.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/runner/moe_runner.py,readonly \
  --mount type=bind,src="$vllm_source/model_executor/layers/fused_moe/config.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/config.py,readonly \
  --mount type=bind,src="$vllm_source/model_executor/layers/fused_moe/router/router_factory.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/router_factory.py,readonly \
  --mount type=bind,src="$vllm_source/model_executor/layers/fused_moe/router/grouped_topk_router.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py,readonly \
  --mount type=bind,src="$vllm_source/model_executor/layers/fused_moe/router/base_router.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/base_router.py,readonly \
  --mount type=bind,src="$vllm_source/model_executor/layers/fused_moe/router/fused_moe_router.py",dst=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/fused_moe_router.py,readonly \
  --workdir /work \
  "$image" \
  capture_calibration.py "$capture_mode" \
  --model "$model" \
  --owner-manifest "$model/calibration_manifest.json" \
  --corpus "$model/calibration/reap_recall_calib.jsonl" \
  --plan-file /work/evidence/document_plan.json \
  --capture-dir "/capture/$capture_name" \
  "${preflight_args[@]}"

if [[ "$mode" == smoke ]]; then
  smoke_dir="$capture_parent/$capture_name"
  # vLLM compiler subprocesses run as container root and may leave 0700/0600
  # cache subtrees.  Normalize only this already-canonical, dedicated bind in
  # a sealed networkless helper, rejecting links/special nodes, then compare
  # the helper's complete inventory with the host inventory before stamping.
  container_inventory_sha=$(docker run --rm \
    --name glm52-fresh-sqg-jit-normalize-r1 \
    --network none \
    --user 0:0 \
    --entrypoint /opt/venv/bin/python \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --mount "type=bind,src=$project,dst=/work,readonly" \
    --mount "type=bind,src=$jit_cache,dst=/cache/jit" \
    --workdir /work \
    -i "$image" - <<'PY'
from pathlib import Path

from src.calibration_capture import normalize_jit_cache_permissions

inventory = normalize_jit_cache_permissions(Path("/cache/jit"))
print(inventory["inventory_sha256"])
PY
  ) || die "post-smoke JIT cache normalization/inventory failed"
  [[ "$container_inventory_sha" =~ ^[0-9a-f]{64}$ ]] || \
    die "post-smoke container JIT inventory digest is malformed"

  /usr/bin/python3 - \
    "$jit_cache" "$smoke_dir" "$plan" "$project" \
    "$container_inventory_sha" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[4])
from src.calibration_capture import write_jit_cache_binding

write_jit_cache_binding(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    plan_path=Path(sys.argv[3]),
    project_root=Path(sys.argv[4]),
    expected_inventory_sha256=sys.argv[5],
)
PY
  printf 'DCP4 smoke and dedicated JIT cache binding sealed: %s\n' "$smoke_dir"
fi
