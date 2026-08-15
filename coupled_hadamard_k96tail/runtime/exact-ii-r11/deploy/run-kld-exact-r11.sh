#!/usr/bin/env bash
set -euo pipefail

if (($# != 2)); then
  printf 'usage: %s MODEL_DIR RESULTS_DIR\n' "$0" >&2
  exit 2
fi

model_dir=$(realpath "$1")
results_dir=$(realpath -m "$2")
image=${IMAGE:-verdictai/glm52-k96-ii-r11:20260815}
reference_dir=${REFERENCE_DIR:-/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current/reference_hf/reference-logits}
cache_dir=${CACHE_DIR:-/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/cache/kld}
kld_dir=${KLD_DIR:-/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/kld}
container_tmp=${CONTAINER_TMP:-/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/cache/tmp}
result_name=${RESULT_NAME:-kld_exact_ii_r11_tp4dcp1.json}
name=${NAME:-glm52-k96-exact-ii-r11-kld}

mkdir -p "$results_dir/kld" "$cache_dir" "$container_tmp"
image_id=$(docker image inspect --format '{{.Id}}' "$image")

exec docker run --rm --name "$name" --gpus all --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e IMAGE_REF="$image" \
  -e IMAGE_ID="$image_id" \
  -e VLLM_GLM_SQG_W4A8_TOPOLOGY_ATTESTATION=tp4dcp1 \
  -e VLLM_GLM_SQG_W4A8_EVIDENCE_DIR=/results/runtime-evidence \
  -e VLLM_USE_DIRECT_DCP_A2A=0 \
  -e VLLM_USE_DIRECT_DCP_Q_GATHER=0 \
  -e VLLM_USE_DIRECT_DCP_KV_GATHER=0 \
  -e VLLM_DCP_GLOBAL_TOPK=1 \
  -e VLLM_USE_B12X_DCP_A2A=1 \
  -e VLLM_DCP_A2A_MAX_TOKENS=16 \
  -e VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
  -e VLLM_B12X_MLA_CKV_GATHER=1 \
  -e VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=140000 \
  -e VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0 \
  -e VLLM_DCP_QUERY_SPLIT=0 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e NCCL_NVLS_ENABLE=0 \
  -e B12X_GLM_W4A8_ACCEPT_ARCH=sm_120 \
  -e B12X_GLM_W4A8_KERNEL=m128n64 \
  -e B12X_GLM_W4A8_V2_BLOCKS=8 \
  -e B12X_GLM_W4A8_V2_STAGES=2 \
  -e B12X_MOE_FORCE_A16=0 \
  -e B12X_MOE_FORCE_A8=0 \
  -e HF_HOME=/cache/hf \
  -e XDG_CACHE_HOME=/cache \
  -e TMPDIR=/container-tmp \
  -v "$model_dir:/model:ro" \
  -v "$reference_dir:/reference:ro" \
  -v "$results_dir:/results:rw" \
  -v "$cache_dir:/cache:rw" \
  -v "$container_tmp:/container-tmp:rw" \
  -v "$kld_dir:/qualification/kld:ro" \
  --entrypoint /opt/venv/bin/python \
  "$image" /qualification/kld/run_kld_sm120.py \
  --model /model \
  --reference-logits /reference \
  --result-json "/results/kld/$result_name" \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --num-gpu-blocks-override 64 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2304 \
  --max-num-batched-tokens 2304 \
  --max-num-seqs 1
