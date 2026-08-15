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
preflight_wave=$wave
if ((start_layer == 75 && end_layer == 77)); then preflight_wave=wave-074-077; fi
INPUT_ROOT=${WAVE_INPUT_ROOT:-/media/brandonmusic/nvme1n1p3/glm52-coupled-wave-inputs/$wave}
PREPARATION_ROOT=$INPUT_ROOT/derived/wave_preflights/$wave
CAPTURE_ROOT=$INPUT_ROOT/capture_view
RECIPE_ROOT=${RECIPE_ROOT:-/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1}
DETACH=${DETACH:-0}
GPU_BASE=${GPU_BASE:-0}
SOURCE_SQG_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-W4A8
QSRT_ROOT=/home/brandonmusic/KLC_SANDBOXES/qsrt-glm52-port
SQG_EXTENSION_ROOT=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33-saturation.VUybIb/sealed
EXLLAMA_ROOT=${FRESH_SQG_EXLLAMA_PYTHON_ROOT:-$PROJECT_ROOT/runtime-dependencies/exllamav3-python}
EXLLAMA_EXTENSION_ROOT=${FRESH_SQG_EXLLAMA_EXTENSION_ROOT:-$PROJECT_ROOT/runtime-dependencies/v39_ext/exllamav3}
EXLLAMA_EXTENSION_SHA256=e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd
IMAGE=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
EXTENSION_SHA256=d29010f6ad51caf2e1a22f07365ab3548fcdb3e0ed3ee15d88330cee24de9614
[[ "$DETACH" == 0 || "$DETACH" == 1 ]] || die "DETACH must be 0 or 1"
[[ "$GPU_BASE" =~ ^[0-9]+$ ]] || die "GPU_BASE must be a nonnegative integer"

selected_layers=$(seq -s, "$start_layer" "$end_layer")
contract_layers=$selected_layers
if ((start_layer == 75 && end_layer == 77)); then contract_layers=74,75,76,77; fi
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

for path in "$PROJECT_ROOT" "$PREPARATION_ROOT" "$CAPTURE_ROOT" \
  "$SOURCE_SQG_ROOT" "$QSRT_ROOT" "$SQG_EXTENSION_ROOT" "$EXLLAMA_ROOT" \
  "$EXLLAMA_EXTENSION_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]] || die "required directory is absent: $path"
done
[[ -f "$EXLLAMA_ROOT/exllamav3/__init__.py" ]] || \
  die "pinned ExLlamaV3 Python package is incomplete: $EXLLAMA_ROOT"
exllama_extension=$EXLLAMA_EXTENSION_ROOT/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
[[ -f "$exllama_extension" && ! -L "$exllama_extension" ]] || \
  die "pinned ExLlamaV3 extension is absent: $exllama_extension"
[[ $(sha256sum "$exllama_extension" | awk '{print $1}') == "$EXLLAMA_EXTENSION_SHA256" ]] || \
  die "pinned ExLlamaV3 extension hash differs: $exllama_extension"
for path in "$PREPARATION_ROOT/preflight.json" \
  "$PREPARATION_ROOT/bit-contract.json" \
  "$PREPARATION_ROOT/source-seal.json" \
  "$PREPARATION_ROOT/wave-bf16-shard-manifest.json" \
  "$CAPTURE_ROOT/capture_manifest.json"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required wave input is absent: $path"
done
mkdir -p "$RECIPE_ROOT/logs" "$RECIPE_ROOT/final_profiles"
BIT_CONTRACT_SHA256=$(sha256sum "$PREPARATION_ROOT/bit-contract.json" | awk '{print $1}')

names=()
pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$DETACH" == 0 ]]; then
    for name in "${names[@]}"; do docker rm -f "$name" >/dev/null 2>&1 || true; done
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

for layer in "${run_layers[@]}"; do
  gpu=$((GPU_BASE + layer - start_layer))
  padded=$(printf '%03d' "$layer")
  name=glm52-goal019ffa7c-recipe-${wave}-l${layer}-r${BASHPID}
  names+=("$name")
  lifecycle=(--rm)
  log_path=$RECIPE_ROOT/logs/recipe-l${padded}.log
  if [[ "$DETACH" == 1 ]]; then
    lifecycle=(-d)
    log_path=${log_path}.launch-${BASHPID}
  fi
  extra=()
  if [[ -n "${SMOKE_CELL:-}" || -n "${SMOKE_EXPERT:-}" ]]; then
    [[ -n "${SMOKE_CELL:-}" && "${SMOKE_EXPERT:-}" =~ ^[0-9]+$ ]] || \
      die "SMOKE_CELL and numeric SMOKE_EXPERT must be supplied together"
    extra+=(--smoke-cell "$SMOKE_CELL" --smoke-expert "$SMOKE_EXPERT")
  fi
  docker run "${lifecycle[@]}" --name "$name" --network none --gpus "device=$gpu" \
    --shm-size 16g --cpus 6 \
    --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
    --mount "type=bind,src=$PREPARATION_ROOT,dst=/output,readonly" \
    --mount "type=bind,src=$PREPARATION_ROOT,dst=/workspace/sqg-run/wave_preflights/$preflight_wave,readonly" \
    --mount "type=bind,src=$PREPARATION_ROOT/bit-contract.json,dst=/workspace/sqg-run/wave_preflight_inputs/$preflight_wave/bit-contract.json,readonly" \
    --mount "type=bind,src=$PREPARATION_ROOT/source-seal.json,dst=/workspace/sqg-run/wave_preflight_inputs/$preflight_wave/source-seal.json,readonly" \
    --mount "type=bind,src=$CAPTURE_ROOT,dst=/capture,readonly" \
    --mount "type=bind,src=$CAPTURE_ROOT,dst=/workspace/glm52-w4a8/capture/bf16-pp8-production-r1/sqg-view,readonly" \
    --mount "type=bind,src=$RECIPE_ROOT,dst=/recipe" \
    --mount "type=bind,src=$QSRT_ROOT,dst=/qsrt,readonly" \
    --mount "type=bind,src=$SOURCE_SQG_ROOT,dst=/source-sqg,readonly" \
    --mount "type=bind,src=$SQG_EXTENSION_ROOT,dst=/sqg-extension,readonly" \
    --mount "type=bind,src=$SQG_EXTENSION_ROOT/sqg-extension-seal.json,dst=/workspace/glm52-w4a8/state/sm103-encoder-dependencies/sqg-extension-seal.json,readonly" \
    --mount "type=bind,src=$EXLLAMA_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
    --mount "type=bind,src=$EXLLAMA_ROOT,dst=/workspace/glm52-w4a8/code/exllamav3-v0.0.43,readonly" \
    --mount "type=bind,src=$EXLLAMA_EXTENSION_ROOT,dst=/opt/exllamav3-extension,readonly" \
    --mount "type=bind,src=$PROJECT_ROOT/kquant,dst=/workspace/glm52-w4a8/code/glm52_fresh_sqg_test/kquant,readonly" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/opt/exllamav3-extension:/opt/exllamav3-r7ext:/opt/exllamav3-python:/work:/work/kquant \
    -e OMP_NUM_THREADS=6 -e MKL_NUM_THREADS=6 \
    -e OPENBLAS_NUM_THREADS=6 -e NUMEXPR_NUM_THREADS=6 \
    -e GIT_CONFIG_COUNT=3 \
    -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/work/kquant \
    -e GIT_CONFIG_KEY_1=safe.directory -e GIT_CONFIG_VALUE_1=/qsrt \
    -e GIT_CONFIG_KEY_2=safe.directory -e GIT_CONFIG_VALUE_2=/workspace/glm52-w4a8/code/glm52_fresh_sqg_test/kquant \
    -e KQUANT_SQG_EXTENSION_PATH=/sqg-extension/kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so \
    -e "KQUANT_SQG_EXTENSION_SHA256=$EXTENSION_SHA256" \
    -e KQUANT_SQG_REQUIRE_PREBUILT=1 -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e FRESH_SQG_RANK_PRIVATE_TRITON=1 \
    -e "FRESH_SQG_SELECTED_LAYERS=$contract_layers" \
    -e FRESH_SQG_PLAN_CONTRACT=/work/evidence/contiguous_document_plan_r1.json \
    -e FRESH_SQG_BF16_MANIFEST=/output/wave-bf16-shard-manifest.json \
    -e "FRESH_SQG_BIT_CONTRACT_SHA256=$BIT_CONTRACT_SHA256" \
    --entrypoint /opt/venv/bin/python -w /work "$IMAGE" \
    scripts/run_coupled_no_shortcut_recipe.py \
      --preflight /output/preflight.json --source-sqg-root /source-sqg \
      --qsrt-root /qsrt --recipe-root /recipe \
      --final-profile-root /recipe/final_profiles --layer "$layer" \
      --device cuda:0 --threads 6 --chunk-rows 256 "${extra[@]}" \
    >"$log_path" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'coupled recipe worker failed: %s\n' "${names[$index]}" >&2
    failed=1
  fi
done
((failed == 0)) || exit 1
if [[ -z "${SMOKE_CELL:-}" ]]; then
  for layer in "${run_layers[@]}"; do
    padded=$(printf '%03d' "$layer")
    jq -e --argjson layer "$layer" \
      '.schema == "glm52-updated-qsrt-coupled-no-shortcut-layer-recipe-v1" and
       .complete == true and .layer == $layer and
       .no_b300_owner_speed_rescue == true and
       .no_fleet_beta_shortcut == true and
       .source_is_frozen_sqg_checkpoint == true and
       .official_bf16_weight_shards_read == false' \
      "$RECIPE_ROOT/layer_${padded}/NO_SHORTCUT_COUPLED_RECIPE.json" >/dev/null
  done
fi
printf 'coupled no-shortcut recipe wave complete: wave=%s root=%s\n' "$wave" "$RECIPE_ROOT"
