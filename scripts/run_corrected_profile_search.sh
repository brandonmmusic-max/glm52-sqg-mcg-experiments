#!/usr/bin/env bash
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: $0 /absolute/successor-root"

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
OUTPUT_ROOT="$1"
BF16_ROOT="$PROJECT_ROOT/bf16_layers"
CAPTURE_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1"
SQG_EXTENSION_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33.p7n1IJ/sealed"
EXLLAMA_R7EXT_ROOT="${FRESH_SQG_EXLLAMA_R7EXT_ROOT:-/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3}"
IMAGE="sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
EXTENSION_NAME="kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so"
EXTENSION_SHA256="c987778677653388f7766e66150850c4dda44bc6b055929ece34eaee87f3cda4"

[[ "$OUTPUT_ROOT" = /* && -f "$OUTPUT_ROOT/preflight.json" ]] || die "successor preflight missing"
[[ -f "$OUTPUT_ROOT/successor_preflight_receipt.json" ]] || die "successor receipt missing"
mkdir -p "$OUTPUT_ROOT/logs"

run_container() {
  local name="$1"
  local gpu="$2"
  shift 2
  docker run --rm --name "$name" \
    --network none --gpus all --shm-size 8g --cpus 3 \
    --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
    --mount "type=bind,src=$BF16_ROOT,dst=$BF16_ROOT,readonly" \
    --mount "type=bind,src=$CAPTURE_ROOT,dst=/capture,readonly" \
    --mount "type=bind,src=$SQG_EXTENSION_ROOT,dst=/sqg-extension,readonly" \
    --mount "type=bind,src=$EXLLAMA_R7EXT_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
    --mount "type=bind,src=$OUTPUT_ROOT,dst=/output" \
    -e "CUDA_VISIBLE_DEVICES=$gpu" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/opt/exllamav3-r7ext:/opt/exllamav3-python:/work \
    -e OMP_NUM_THREADS=3 -e MKL_NUM_THREADS=3 \
    -e OPENBLAS_NUM_THREADS=3 -e NUMEXPR_NUM_THREADS=3 \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
    -e GIT_CONFIG_VALUE_0=/work/kquant \
    -e "KQUANT_SQG_EXTENSION_PATH=/sqg-extension/$EXTENSION_NAME" \
    -e "KQUANT_SQG_EXTENSION_SHA256=$EXTENSION_SHA256" \
    -e KQUANT_SQG_REQUIRE_PREBUILT=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e "FRESH_SQG_RUNTIME_IMAGE_ID=$IMAGE" \
    -w /work "$IMAGE" "$@"
}

# One real expert: layer 28 expert 0 has gate K4 and up K3.  The helper also
# rebuilds its candidate-conditional H2 and encodes down before it can pass.
run_container glm52-sqg-absfix-smoke 1 \
  scripts/smoke_absolute_gate_scale.py \
    --preflight /output/preflight.json --device cuda:0 --threads 3 \
  >"$OUTPUT_ROOT/logs/absolute-scale-smoke.log" 2>&1

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  cat "$OUTPUT_ROOT/logs/absolute-scale-smoke.log"
  printf 'Corrected real-expert smoke complete. Profile workers were not launched.\n'
  exit 0
fi

layers=(6 28 52 77)
gpus=(0 1 2 3)

# Publish preregistration before concurrent workers touch cell directories.
for index in 0 1 2 3; do
  layer="${layers[$index]}"
  gpu="${gpus[$index]}"
  run_container "glm52-sqg-absfix-init-$layer" "$gpu" \
    scripts/profile_search_shard.py \
      --preflight /output/preflight.json --layer "$layer" \
      --device cuda:0 --threads 3 --initialize-only \
    >"$OUTPUT_ROOT/logs/profile-init-$layer.log" 2>&1
done

# Sixteen processes: one layer per physical GPU, four workers per GPU, and
# four unique cells per worker.  Each cell retains its own 16-expert H13
# session and rebuilds expert-local candidate H2 before down encoding.
pids=()
names=()
for index in 0 1 2 3; do
  layer="${layers[$index]}"
  gpu="${gpus[$index]}"
  for worker in 0 1 2 3; do
    name="glm52-sqg-absfix-l${layer}-p${worker}"
    run_container "$name" "$gpu" \
      scripts/profile_search_shard.py \
        --preflight /output/preflight.json --layer "$layer" \
        --device cuda:0 --threads 3 --worker-index "$worker" \
      >"$OUTPUT_ROOT/logs/profile-l${layer}-p${worker}.log" 2>&1 &
    pids+=("$!")
    names+=("$name")
  done
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'profile worker failed: %s\n' "${names[$index]}" >&2
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || die "one or more corrected profile workers failed"

# Freeze the selection only after all 16 exact cells per layer validate.
pids=()
names=()
for index in 0 1 2 3; do
  layer="${layers[$index]}"
  gpu="${gpus[$index]}"
  name="glm52-sqg-absfix-select-$layer"
  run_container "$name" "$gpu" \
    scripts/finalize_corrected_profile_selection.py \
      --preflight /output/preflight.json --layer "$layer" --device cuda:0 \
    >"$OUTPUT_ROOT/logs/profile-select-$layer.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done
failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'selection finalizer failed: %s\n' "${names[$index]}" >&2
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || die "one or more corrected selections failed"

printf 'Corrected profile search complete. Full final encoding was not launched.\n'
