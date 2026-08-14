#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 2 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ ]] || \
  die "usage: $0 START_LAYER END_LAYER"
start_layer=$1
end_layer=$2
((end_layer >= start_layer && end_layer - start_layer <= 3 \
  && start_layer >= 3 && end_layer <= 78)) || \
  die "range must contain one to four consecutive routed layers"

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
wave=$(printf 'wave-%03d-%03d' "$start_layer" "$end_layer")
INPUT_ROOT=${WAVE_INPUT_ROOT:-/media/brandonmusic/nvme1n1p3/glm52-coupled-wave-inputs/$wave}
PREPARATION_ROOT=${PREPARATION_ROOT:-$INPUT_ROOT/derived/wave_preflights/$wave}
BINDING_ROOT=${BINDING_ROOT:-$INPUT_ROOT/derived/wave_preflights/$wave}
CAPTURE_ROOT=${CAPTURE_ROOT:-$INPUT_ROOT/capture_view}
CANDIDATE_ROOT=${CANDIDATE_ROOT:-/media/brandonmusic/nvme1n1p3/glm52-coupled-3p0625-no-shortcut-work/$wave}
DETACH=${DETACH:-0}
[[ "$DETACH" == 0 || "$DETACH" == 1 ]] || die "DETACH must be 0 or 1"
SOURCE_SQG_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-W4A8
ALLOCATION_ROOT=${ALLOCATION_ROOT:-/media/brandonmusic/nvme1n1p3/glm52-coupled-layer-native-allocations-no-shortcut-v7}
ALLOCATION_SUFFIX=${ALLOCATION_SUFFIX:-.allocation.json}
[[ -n "$ALLOCATION_SUFFIX" && "$ALLOCATION_SUFFIX" != */* ]] || \
  die "ALLOCATION_SUFFIX must be a nonempty filename suffix"
QSRT_ROOT=/home/brandonmusic/KLC_SANDBOXES/qsrt-glm52-port
EXLLAMA_ROOT=${FRESH_SQG_EXLLAMA_PYTHON_ROOT:-$PROJECT_ROOT/runtime-dependencies/exllamav3-python}
EXLLAMA_EXTENSION_ROOT=${FRESH_SQG_EXLLAMA_EXTENSION_ROOT:-$PROJECT_ROOT/runtime-dependencies/v39_ext/exllamav3}
EXLLAMA_EXTENSION_SHA256=e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd
IMAGE=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
selected_layers=$(seq -s, "$start_layer" "$end_layer")
run_layers_csv=${RUN_LAYERS:-$selected_layers}
IFS=, read -r -a run_layers <<< "$run_layers_csv"
(( ${#run_layers[@]} >= 1 && ${#run_layers[@]} <= 4 )) || \
  die "RUN_LAYERS must select one to four layers"
seen=,
for layer in "${run_layers[@]}"; do
  [[ "$layer" =~ ^[0-9]+$ ]] || die "RUN_LAYERS contains a nonnumeric layer"
  ((layer >= start_layer && layer <= end_layer)) || \
    die "RUN_LAYERS layer $layer lies outside $start_layer..$end_layer"
  [[ "$seen" != *",$layer,"* ]] || die "RUN_LAYERS repeats layer $layer"
  seen+="$layer,"
done

for path in "$PROJECT_ROOT" "$PREPARATION_ROOT" "$BINDING_ROOT" "$CAPTURE_ROOT" \
  "$CANDIDATE_ROOT" "$SOURCE_SQG_ROOT" "$ALLOCATION_ROOT" "$QSRT_ROOT" \
  "$EXLLAMA_ROOT" "$EXLLAMA_EXTENSION_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]] || die "required directory is absent: $path"
done
[[ -f "$EXLLAMA_ROOT/exllamav3/__init__.py" ]] || \
  die "pinned ExLlamaV3 Python package is incomplete: $EXLLAMA_ROOT"
exllama_extension=$EXLLAMA_EXTENSION_ROOT/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
[[ -f "$exllama_extension" && ! -L "$exllama_extension" ]] || \
  die "pinned ExLlamaV3 extension is absent: $exllama_extension"
[[ $(sha256sum "$exllama_extension" | awk '{print $1}') == "$EXLLAMA_EXTENSION_SHA256" ]] || \
  die "pinned ExLlamaV3 extension hash differs: $exllama_extension"
for layer in "${run_layers[@]}"; do
  allocation="$ALLOCATION_ROOT/layer_$(printf '%03d' "$layer")${ALLOCATION_SUFFIX}"
  [[ -f "$allocation" && ! -L "$allocation" ]] || \
    die "layer-native allocation is absent: $allocation"
done
[[ -f "$BINDING_ROOT/wave-bf16-shard-manifest.json" && \
   ! -L "$BINDING_ROOT/wave-bf16-shard-manifest.json" ]] || \
  die "derived wave BF16 identity manifest is absent"
BIT_CONTRACT_SHA256=$(sha256sum "$BINDING_ROOT/bit-contract.json" | awk '{print $1}')
mkdir -p "$CANDIDATE_ROOT/logs"

names=()
pids=()
phase_names=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$DETACH" == 0 ]]; then
    for name in "${names[@]}"; do
      docker rm -f "$name" >/dev/null 2>&1 || true
    done
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

launch_scorer() {
  local layer=$1
  local phase=$2
  local draw=${3:-}
  local gpu padded allocation_name name log_path threads token
  local -a lifecycle phase_args

  padded=$(printf '%03d' "$layer")
  allocation_name="layer_${padded}${ALLOCATION_SUFFIX}"
  if [[ "$phase" == draw ]]; then
    [[ "$draw" == 0 || "$draw" == 6 ]] || die "draw phase requires draw 0 or 6"
    # Eight draw arms are spread round-robin across four GPUs.  A full wave
    # therefore runs two independent exact scorers per GPU instead of
    # serializing draw 0 and draw 6 inside one process.
    if [[ "$draw" == 0 ]]; then
      gpu=$(((2 * (layer - start_layer)) % 4))
    else
      gpu=$(((2 * (layer - start_layer) + 1) % 4))
    fi
    threads=4
    token="d${draw}"
    phase_args=(--phase draw --draw "$draw")
  else
    [[ "$phase" == all || "$phase" == selected ]] || die "invalid scoring phase: $phase"
    gpu=$(((layer - start_layer) % 4))
    threads=8
    token=$phase
    phase_args=(--phase "$phase")
  fi
  name="glm52-goal019ffa7c-score-${wave}-l${layer}-${token}-r${BASHPID}"
  names+=("$name")
  phase_names+=("$name")
  lifecycle=(--rm)
  log_path="$CANDIDATE_ROOT/logs/score-l${padded}-${token}.log"
  if [[ "$DETACH" == 1 ]]; then
    lifecycle=(-d)
    log_path="${log_path}.launch-${BASHPID}"
  fi
  docker run "${lifecycle[@]}" --name "$name" --network none --gpus "device=$gpu" \
    --shm-size 32g --cpus "$threads" \
    --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
    --mount "type=bind,src=$PREPARATION_ROOT,dst=/output,readonly" \
    --mount "type=bind,src=$PREPARATION_ROOT,dst=/workspace/sqg-run/wave_preflights/$wave,readonly" \
    --mount "type=bind,src=$BINDING_ROOT,dst=/binding,readonly" \
    --mount "type=bind,src=$BINDING_ROOT/bit-contract.json,dst=/workspace/sqg-run/wave_preflight_inputs/$wave/bit-contract.json,readonly" \
    --mount "type=bind,src=$BINDING_ROOT/source-seal.json,dst=/workspace/sqg-run/wave_preflight_inputs/$wave/source-seal.json,readonly" \
    --mount "type=bind,src=$CANDIDATE_ROOT,dst=/candidate" \
    --mount "type=bind,src=$CAPTURE_ROOT,dst=/capture,readonly" \
    --mount "type=bind,src=$CAPTURE_ROOT,dst=/workspace/glm52-w4a8/capture/bf16-pp8-production-r1/sqg-view,readonly" \
    --mount "type=bind,src=$QSRT_ROOT,dst=/qsrt,readonly" \
    --mount "type=bind,src=$SOURCE_SQG_ROOT,dst=/source-sqg,readonly" \
    --mount "type=bind,src=$ALLOCATION_ROOT,dst=/allocation,readonly" \
    --mount "type=bind,src=$EXLLAMA_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
    --mount "type=bind,src=$EXLLAMA_ROOT,dst=/workspace/glm52-w4a8/code/exllamav3-v0.0.43,readonly" \
    --mount "type=bind,src=$EXLLAMA_EXTENSION_ROOT,dst=/opt/exllamav3-extension,readonly" \
    --mount "type=bind,src=$PROJECT_ROOT/kquant,dst=/workspace/glm52-w4a8/code/glm52_fresh_sqg_test/kquant,readonly" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/opt/exllamav3-extension:/opt/exllamav3-r7ext:/opt/exllamav3-python:/work:/work/kquant \
    -e "OMP_NUM_THREADS=$threads" -e "MKL_NUM_THREADS=$threads" \
    -e "OPENBLAS_NUM_THREADS=$threads" -e "NUMEXPR_NUM_THREADS=$threads" \
    -e GIT_CONFIG_COUNT=3 \
    -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/work/kquant \
    -e GIT_CONFIG_KEY_1=safe.directory -e GIT_CONFIG_VALUE_1=/qsrt \
    -e GIT_CONFIG_KEY_2=safe.directory -e GIT_CONFIG_VALUE_2=/workspace/glm52-w4a8/code/glm52_fresh_sqg_test/kquant \
    -e "FRESH_SQG_SELECTED_LAYERS=$selected_layers" \
    -e FRESH_SQG_PLAN_CONTRACT=/work/evidence/contiguous_document_plan_r1.json \
    -e FRESH_SQG_BF16_MANIFEST=/binding/wave-bf16-shard-manifest.json \
    -e "FRESH_SQG_BIT_CONTRACT_SHA256=$BIT_CONTRACT_SHA256" \
    -e FRESH_SQG_RANK_PRIVATE_TRITON=1 \
    --entrypoint /opt/venv/bin/python -w /work "$IMAGE" \
    scripts/score_select_coupled_mixed_rate.py \
      --preflight /output/preflight.json \
      --candidate-root /candidate --qsrt-root /qsrt \
      --source-sqg-root /source-sqg \
      --allocation "/allocation/$allocation_name" \
      --layer "$layer" --draws 0 6 "${phase_args[@]}" \
      --device cuda:0 --threads "$threads" --chunk-rows 256 \
    >"$log_path" 2>&1 &
  pids+=("$!")
}

wait_phase() {
  local failed=0 index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      printf 'scorer failed: %s\n' "${phase_names[$index]}" >&2
      failed=1
    fi
  done
  pids=()
  phase_names=()
  ((failed == 0))
}

if [[ "$DETACH" == 1 ]]; then
  # Detached callers retain the original one-container-per-layer behavior;
  # the synchronous campaign uses the explicitly staged parallel path below.
  for layer in "${run_layers[@]}"; do
    launch_scorer "$layer" all
  done
  wait_phase || exit 1
else
  for layer in "${run_layers[@]}"; do
    launch_scorer "$layer" draw 0
    launch_scorer "$layer" draw 6
  done
  wait_phase || exit 1
  for layer in "${run_layers[@]}"; do
    launch_scorer "$layer" selected
  done
  wait_phase || exit 1
fi

if [[ "$DETACH" == 1 ]]; then
  printf 'coupled wave scoring launched detached: wave=%s root=%s containers=%s\n' \
    "$wave" "$CANDIDATE_ROOT" "${names[*]}"
else
  printf 'coupled wave scoring complete: wave=%s root=%s\n' "$wave" "$CANDIDATE_ROOT"
fi
