#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
ACCEPTANCE_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812
SOURCE_MODEL=/home/brandonmusic/models/GLM-5.2-SQG-W4A8
LAYER_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers
FINAL_MODEL=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail
REPRO_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-reproduction
RESULTS_ROOT=$ACCEPTANCE_ROOT/RESULTS/coupled-k96tail-full-tp4dcp1
SOURCE_KLD=$ACCEPTANCE_ROOT/RESULTS/coupled-tail-source-control-tp4dcp1-routes-v2/kld/kld_sm120_tp4dcp1.json
REFERENCE_DIR=/home/brandonmusic/klc-linux/glm52_hybrid_opt/kld_eval_current/reference_hf/reference-logits
IMAGE=${IMAGE:-verdictai/glm52-sqg-coupled-k96tail-sm120:goal019ffabf-final}
CACHE_DIR=$ACCEPTANCE_ROOT/cache/coupled-k96tail-final
ALLOCATION_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-allocations-v1
PROFILE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1/final_profiles
SCORE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-tail-rate-scores-no-shortcut-v5
KLD_JSON=$RESULTS_ROOT/kld/kld_sm120_tp4dcp1.json
CODEC_JSON=$RESULTS_ROOT/model_codec_validation.json
TAIL_JSON=$RESULTS_ROOT/kld_position_tail.json
MTP3_RESULTS=$RESULTS_ROOT/mtp3
MTP3_SMOKE=$MTP3_RESULTS/smoke_16tok.json
COMPLETE_JSON=$RESULTS_ROOT/FULL_ACCEPTANCE.json

jq_pass() {
  local path=$1 expression=$2
  jq -e "$expression" "$path" >/dev/null 2>&1 ||
    sudo -n jq -e "$expression" "$path" >/dev/null 2>&1
}

layer_sealed() {
  local layer=$1 padded expected
  padded=$(printf '%03d' "$layer")
  [[ -f "$LAYER_ROOT/r7-experts-layer-${padded}.safetensors" &&
     -f "$LAYER_ROOT/r7-experts-layer-${padded}.json" &&
     -f "$LAYER_ROOT/r7-experts-layer-${padded}.quality.json" &&
     -f "$LAYER_ROOT/runtime-oracle-layer-${padded}.json" ]] || return 1
  if ((layer == 3)); then
    jq_pass "$LAYER_ROOT/r7-experts-layer-${padded}.json" \
      '.schema == "glm52-coupled-selected-layer-runtime-v1" and
       .complete == true and .bit_census == {"k3":720,"k4":48,"total":768} and
       .bits_per_weight == 3.0625' &&
    jq_pass "$LAYER_ROOT/r7-experts-layer-${padded}.quality.json" \
      '.schema == "glm52-coupled-layer-quality-tails-v1" and .complete == true' &&
    jq_pass "$LAYER_ROOT/runtime-oracle-layer-${padded}.json" \
      '.complete == true and .pass == true and
       .bit_census == {"k3":720,"k4":48,"total":768}'
  else
    jq_pass "$LAYER_ROOT/r7-experts-layer-${padded}.json" \
      '.schema == "glm52-coupled-selected-layer-runtime-v3" and
       .complete == true and .bit_census == {"k3":672,"k4":96,"total":768} and
       .bits_per_weight == 3.125 and
       .final_profile_binding.no_b300_owner_speed_rescue == true' &&
    jq_pass "$LAYER_ROOT/r7-experts-layer-${padded}.quality.json" \
      '.schema == "glm52-coupled-layer-quality-tails-v2" and .complete == true' &&
    jq_pass "$LAYER_ROOT/runtime-oracle-layer-${padded}.json" \
      '.complete == true and .pass == true and
       .bit_census == {"k3":672,"k4":96,"total":768}'
  fi
}

while :; do
  sealed=0
  for layer in $(seq 3 77); do
    layer_sealed "$layer" && sealed=$((sealed + 1))
  done
  printf 'finalizer waiting: %s/75 target layers have passing runtime oracles; MTP78 is preserved\n' "$sealed"
  [[ "$sealed" == 75 ]] && break
  sleep 60
done

docker image inspect "$IMAGE" >/dev/null
image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
mkdir -p "$RESULTS_ROOT/kld" "$CACHE_DIR" "$REPRO_ROOT"

# Oracle containers normally create world-readable receipts. Normalize only
# an exact oracle file if a restrictive container umask made it unreadable.
for layer in $(seq 3 77); do
  padded=$(printf '%03d' "$layer")
  oracle=$LAYER_ROOT/runtime-oracle-layer-${padded}.json
  if [[ ! -r "$oracle" ]]; then
    sudo -n chown "$(id -u):$(id -g)" "$oracle"
    chmod u+rw,go+r "$oracle"
  fi
done

if [[ ! -e "$FINAL_MODEL" ]]; then
  python3 "$PROJECT_ROOT/scripts/assemble_coupled_checkpoint.py" \
    --source "$SOURCE_MODEL" --layer-root "$LAYER_ROOT" \
    --output "$FINAL_MODEL" --layers $(seq 3 77) \
    >"$RESULTS_ROOT/assembly.log"
elif [[ ! -f "$FINAL_MODEL/COUPLED_REENCODE_MANIFEST.json" ]] ||
     ! jq_pass "$FINAL_MODEL/COUPLED_REENCODE_MANIFEST.json" \
       '.complete == true and .all_target_routed_layers_coupled == true and
        .mtp_layer_78_policy == "preserve_source_unchanged" and
        .routed_layer_average_rate_is_uniform == false and
        .per_layer_bit_census["3"] == {"k3":720,"k4":48,"total":768} and
        .per_layer_bit_census["4"] == {"k3":672,"k4":96,"total":768}'; then
  printf 'refusing partial or unsealed final model directory: %s\n' "$FINAL_MODEL" >&2
  exit 2
fi

if [[ ! -f "$CODEC_JSON" ]]; then
  python3 "$ACCEPTANCE_ROOT/scripts/validate_model_codec.py" \
    --model "$FINAL_MODEL" --revision local-k96tail-coupled-assembly \
    --offline --require-target-coupled-preserved-mtp78 --sample-shards 76 \
    --result-json "$CODEC_JSON" >"$RESULTS_ROOT/model-codec.log"
fi
jq_pass "$CODEC_JSON" \
  '.pass == true and .routed_layer_count == 76 and
   .mtp_layer78_routed == 768 and .mtp_layer78_preserved == true and
   (.coupled_layers | length) == 75'

KLD_COMMAND_STATUS=0
if [[ ! -f "$KLD_JSON" ]]; then
  set +e
  env IMAGE="$IMAGE" IMAGE_ID="$image_id" MODEL_DIR="$FINAL_MODEL" \
    MODEL_REVISION=local-k96tail-coupled-assembly CACHE_DIR="$CACHE_DIR" \
    RESULTS_DIR="$RESULTS_ROOT" REFERENCE_DIR="$REFERENCE_DIR" \
    TOPOLOGY_ATTESTATION=tp4dcp1 \
    docker compose -p glm52-coupled-k96tail-final-tp4dcp1 \
      -f "$ACCEPTANCE_ROOT/compose.yaml" --profile tools run --rm \
      -e TP_SIZE=4 -e PP_SIZE=1 -e DCP_SIZE=1 \
      -e VLLM_GLM_SQG_W4A8_TOPOLOGY_ATTESTATION=tp4dcp1 \
      kld --model /model --reference-logits /reference \
      --result-json /results/kld/kld_sm120_tp4dcp1.json \
      --tensor-parallel-size 4 --pipeline-parallel-size 1 \
      --decode-context-parallel-size 1 --num-gpu-blocks-override 128 \
      --gpu-memory-utilization 0.90 --max-model-len 2304 \
      --max-num-batched-tokens 2304 --max-num-seqs 1
  KLD_COMMAND_STATUS=$?
  set -e
fi
if ! jq_pass "$KLD_JSON" \
  '.complete == true and .total_positions == 2047 and
   .runtime.tensor_parallel_size == 4 and
   .runtime.pipeline_parallel_size == 1 and
   .runtime.decode_context_parallel_size == 1 and
   .runtime.topology_attestation == "tp4dcp1" and
   .statistics.nonfinite_count == 0 and
   .statistics.trim_fraction_per_side == 0.0'; then
  printf 'TP4/PP1/DCP1 KLD receipt is absent or incomplete (command status %s)\n' \
    "$KLD_COMMAND_STATUS" >&2
  exit 2
fi
if ((KLD_COMMAND_STATUS != 0)); then
  printf 'KLD command exited %s after writing a complete sealed receipt; continuing\n' \
    "$KLD_COMMAND_STATUS"
fi
if [[ ! -r "$KLD_JSON" ]]; then
  sudo -n chown "$(id -u):$(id -g)" "$KLD_JSON"
  chmod u+rw,go+r "$KLD_JSON"
fi

if [[ ! -f "$TAIL_JSON" ]]; then
  python3 "$PROJECT_ROOT/scripts/analyze_kld_position_tail.py" \
    --baseline "$SOURCE_KLD" --candidate "$KLD_JSON" \
    --remove 10 20 30 40 96 --output "$TAIL_JSON" \
    >"$RESULTS_ROOT/kld-tail.log"
fi

if [[ ! -f "$MTP3_SMOKE" ]]; then
  MTP3_PROJECT=glm52-coupled-k96tail-final-mtp3
  MTP3_CONTAINER=glm52-sqg-coupled-sm120-prod-mtp3
  MTP3_PORT=9433
  mkdir -p "$MTP3_RESULTS/evidence"
  export IMAGE IMAGE_ID="$image_id" MODEL_DIR="$FINAL_MODEL"
  export MODEL_REVISION=local-k96tail-coupled-assembly
  export CACHE_DIR RESULTS_DIR="$MTP3_RESULTS" REFERENCE_DIR
  export TOPOLOGY_ATTESTATION=tp1pp4dcp1
  export SERVED_MODEL_NAME=GLM-5.2-SQG-Coupled-H512-H128-K96Tail-NoShortcut
  export BIND_ADDRESS=127.0.0.1 HOST_PORT="$MTP3_PORT"
  export PROD_MAX_MODEL_LEN=8192 PROD_MAX_NUM_SEQS=1
  export PROD_MAX_BATCHED_TOKENS=2048 PROD_GPU_MEMORY_UTILIZATION=0.90
  cleanup_mtp3() {
    docker compose -p "$MTP3_PROJECT" -f "$ACCEPTANCE_ROOT/compose.yaml" \
      --profile prod down --remove-orphans=false >/dev/null 2>&1 || true
  }
  trap cleanup_mtp3 EXIT INT TERM
  cleanup_mtp3
  docker compose -p "$MTP3_PROJECT" -f "$ACCEPTANCE_ROOT/compose.yaml" \
    --profile prod up -d server-prod
  healthy=0
  for _ in $(seq 1 120); do
    state=$(docker inspect --format \
      '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
      "$MTP3_CONTAINER" 2>/dev/null || true)
    if [[ "$state" == healthy ]]; then
      healthy=1
      break
    fi
    if [[ "$state" == exited || "$state" == dead || "$state" == unhealthy ]]; then
      docker logs --tail 300 "$MTP3_CONTAINER" >"$MTP3_RESULTS/server-failure.log" 2>&1 || true
      printf 'MTP3 production server entered terminal state: %s\n' "$state" >&2
      exit 2
    fi
    sleep 30
  done
  if ((healthy != 1)); then
    docker logs --tail 300 "$MTP3_CONTAINER" >"$MTP3_RESULTS/server-timeout.log" 2>&1 || true
    printf 'MTP3 production server did not become healthy\n' >&2
    exit 2
  fi
  python3 "$ACCEPTANCE_ROOT/scripts/smoke_test.py" \
    --url "http://127.0.0.1:$MTP3_PORT" \
    --model "$SERVED_MODEL_NAME" --max-tokens 16 \
    --result-json "$MTP3_SMOKE" >"$MTP3_RESULTS/smoke.log"
  docker logs "$MTP3_CONTAINER" >"$MTP3_RESULTS/server.log" 2>&1 || true
  cleanup_mtp3
  trap - EXIT INT TERM
fi
jq_pass "$MTP3_SMOKE" \
  '.schema == "glm52-sqg-w4a8-sm120-smoke-v1" and .pass == true and
   .problems == [] and (.tokens | length) == 16'

if [[ ! -f "$COMPLETE_JSON" ||
      ! -f "$REPRO_ROOT/sealed-bundle/README.md" ||
      ! -f "$REPRO_ROOT/sealed-bundle/SHA256SUMS" ]]; then
  python3 "$PROJECT_ROOT/scripts/seal_coupled_k96tail_release.py" \
    --model "$FINAL_MODEL" --layer-root "$LAYER_ROOT" \
    --codec-receipt "$CODEC_JSON" --source-kld "$SOURCE_KLD" \
    --candidate-kld "$KLD_JSON" --tail-analysis "$TAIL_JSON" \
    --mtp3-smoke "$MTP3_SMOKE" --results-root "$RESULTS_ROOT" \
    --repro-root "$REPRO_ROOT" --project-root "$PROJECT_ROOT" \
    --acceptance-root "$ACCEPTANCE_ROOT" \
    --allocation-root "$ALLOCATION_ROOT" --profile-root "$PROFILE_ROOT" \
    --score-root "$SCORE_ROOT" --image-ref "$IMAGE" --image-id "$image_id" \
    >"$RESULTS_ROOT/seal-release.log"
fi
jq_pass "$COMPLETE_JSON" '.complete == true and .quality_pass == true'
printf 'full no-shortcut coupled K96-tail model, MTP3 smoke, TP4/DCP1 KLD, and reproduction bundle sealed\n'
