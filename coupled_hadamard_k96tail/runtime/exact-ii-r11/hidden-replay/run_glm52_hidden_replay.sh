#!/usr/bin/env bash
set -euo pipefail

acceptance=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812
project=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
model=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail
source_model=/home/brandonmusic/models/GLM-5.2-SQG-W4A8
reference=/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current/reference_hf/reference-logits
token_file=/home/brandonmusic/KLC_SANDBOXES/rtx6kpro-glm52-k96-publication/benchmarks/data/glm52-kld-tokens-2048.json
baseline_root=$acceptance/RESULTS/coupled-k96tail-full-exact-ii-r11-tpfix
baseline_kld=$baseline_root/kld/kld_exact_ii_r11_tp4dcp1.json
results=$acceptance/RESULTS/coupled-k96tail-hidden-replay-exact-ii-r11
cache=$acceptance/cache/coupled-k96tail-hidden-replay
container_tmp=$cache/container-tmp
base_image=verdictai/glm52-k96-ii-r11:20260815-tpfix
capture_image=verdictai/glm52-k96-ii-r11:20260815-tpfix-hidden-replay
project_name=glm52-k96tail-exact-r11-hidden-replay

log() {
  printf 'GLM hidden replay: %s %s\n' "$(date --iso-8601=seconds)" "$*"
}

jq_pass() {
  jq -e "$2" "$1" >/dev/null 2>&1 ||
    sudo -n jq -e "$2" "$1" >/dev/null 2>&1
}

wait_for_gpu_drain() {
  local blockers
  while true; do
    blockers=$(
      nvidia-smi --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, '
            {
              gsub(/[[:space:]]/, "", $1)
              gsub(/[[:space:]]/, "", $2)
              if (($2 + 0) > 2048) print $1 ":" $2 "MiB"
            }
          '
    )
    if [[ -z "$blockers" ]]; then
      return
    fi
    log "waiting for GPU allocations to clear: $(tr '\n' ' ' <<<"$blockers")"
    sleep 15
  done
}

if [[ -f "$results/hidden-replay-kld.json" ]] &&
   jq_pass "$results/hidden-replay-kld.json" \
     '.complete == true and .qualification_pass == true and .total_positions == 2047'; then
  log "sealed replay receipt already exists"
  exit 0
fi

while ! jq_pass "$baseline_kld" \
        '.complete == true and .total_positions == 2047 and .statistics.nonfinite_count == 0'; do
  log "waiting for the sealed baseline full-logit KLD receipt"
  sleep 30
done

mkdir -p "$results" "$cache" "$container_tmp"
docker image inspect "$base_image" >/dev/null
base_image_id=$(docker image inspect --format '{{.Id}}' "$base_image")
docker image inspect "$base_image" | jq -e '
  .[0].Config.Labels["ai.verdict.infernal-invocation.base"] == "r11" and
  .[0].Config.Labels["ai.verdict.target.topology"] == "tp4-dcp4-mtp3" and
  .[0].Config.Labels["ai.verdict.coupled.tp4-preactivation"] ==
    "all-gather-reassembly-v1"
' >/dev/null
log "building capture derivative from $base_image_id"
docker build \
  --build-arg "BASE_IMAGE=$base_image" \
  -f "$acceptance/hidden_replay/Dockerfile.capture" \
  -t "$capture_image" \
  "$acceptance"
capture_image_id=$(docker image inspect --format '{{.Id}}' "$capture_image")
docker image inspect "$capture_image" | jq -e '
  .[0].Config.Labels["ai.verdict.infernal-invocation.base"] == "r11" and
  .[0].Config.Labels["ai.verdict.capture.runtime-lineage"] ==
    "exact-infernal-invocation-r11-tpfix" and
  .[0].Config.Labels["ai.verdict.capture.semantic-point"] ==
    "after-final-rmsnorm-before-lm-head"
' >/dev/null

run_name=raw-$(date -u +%Y%m%dT%H%M%SZ)
raw_host=$results/$run_name
mkdir -p "$raw_host"
wait_for_gpu_drain
log "capturing all 2,048 GLM hidden rows and repeating full-logit KLD"
docker run --rm --name "$project_name-capture" --gpus all --ipc host \
  --shm-size 32g --ulimit memlock=-1 \
  --ulimit nofile=1048576:1048576 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e IMAGE_REF="$capture_image" \
  -e IMAGE_ID="$capture_image_id" \
  -e VLLM_GLM_SQG_W4A8_TOPOLOGY_ATTESTATION=tp4dcp1 \
  -e VLLM_GLM_SQG_W4A8_EVIDENCE_DIR=/results/runtime-evidence \
  -e VLLM_KLD_HIDDEN_CAPTURE_DIR=/results/hidden-raw \
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
  -v "$model:/model:ro" \
  -v "$reference:/reference:ro" \
  -v "$raw_host:/results:rw" \
  -v "$cache:/cache:rw" \
  -v "$container_tmp:/container-tmp:rw" \
  -v "$acceptance/kld:/qualification/kld:ro" \
  --entrypoint /opt/venv/bin/python \
  "$capture_image" /qualification/kld/run_kld_sm120.py \
  --model /model \
  --reference-logits /reference \
  --result-json /results/capture-full-kld.json \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --num-gpu-blocks-override 64 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2304 \
  --max-num-batched-tokens 2304 \
  --max-num-seqs 1

jq_pass "$raw_host/capture-full-kld.json" \
  '.complete == true and .total_positions == 2047 and .statistics.nonfinite_count == 0'

log "sealing the raw [2048,6144] capture"
docker run --rm --entrypoint /opt/venv/bin/python \
  -v "$raw_host:/capture:rw" \
  -v "$token_file:/tokens.json:ro" \
  "$capture_image" \
  /opt/glm52-hidden-replay/finalize_glm52_hidden_capture.py \
  --raw-dir /capture/hidden-raw \
  --token-file /tokens.json \
  --full-kld-receipt /capture/capture-full-kld.json \
  --output-dir /capture/hidden

log "exporting and comparing the unchanged source/candidate BF16 LM head"
docker run --rm --entrypoint /opt/venv/bin/python \
  -v "$source_model:/source:ro" \
  -v "$model:/model:ro" \
  -v "$results:/results:rw" \
  "$capture_image" \
  /opt/glm52-hidden-replay/export_glm52_lm_head.py \
  --source-model /source --candidate-model /model \
  --output-dir /results/lm-head

log "running deterministic two-pass full-vocabulary hidden replay"
wait_for_gpu_drain
docker run --rm --gpus '"device=0"' --ipc=host --entrypoint /opt/venv/bin/python \
  -v "$raw_host:/capture:ro" \
  -v "$results:/results:rw" \
  -v "$reference:/reference:ro" \
  -v "$baseline_kld:/baseline-full-kld.json:ro" \
  "$capture_image" \
  /opt/glm52-hidden-replay/compare_glm52_hidden_replay.py \
  --reference-dir /reference \
  --candidate-hidden-dir /capture/hidden \
  --lm-head-dir /results/lm-head \
  --baseline-full-kld /baseline-full-kld.json \
  --capture-full-kld /capture/capture-full-kld.json \
  --output /results/hidden-replay-kld.json

jq_pass "$results/hidden-replay-kld.json" \
  '.complete == true and .qualification_pass == true and .total_positions == 2047'
ln -sfn "$run_name" "$results/latest-capture"
touch "$results/hidden-replay.complete"
log "one-context hidden replay qualified with capture image $capture_image_id"
