#!/usr/bin/env bash
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 ]] || die \
  "usage: $0 /absolute/successor-root /absolute/new-candidate-path"

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
SUCCESSOR_ROOT="$(realpath -e -- "$1")"
CANDIDATE_REQUESTED="$2"
SOURCE_MODEL="/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED"
BF16_ROOT="$PROJECT_ROOT/bf16_layers"
CAPTURE_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1"
SQG_EXTENSION_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33.p7n1IJ/sealed"
EXLLAMA_R7EXT_ROOT="${FRESH_SQG_EXLLAMA_R7EXT_ROOT:-/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3}"
IMAGE="sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
EXTENSION_NAME="kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so"
EXTENSION_SHA256="c987778677653388f7766e66150850c4dda44bc6b055929ece34eaee87f3cda4"

[[ "$2" = /* ]] || die "candidate path must be absolute"
CANDIDATE_PARENT="$(realpath -e -- "$(dirname -- "$CANDIDATE_REQUESTED")")"
CANDIDATE_NAME="$(basename -- "$CANDIDATE_REQUESTED")"
[[ "$CANDIDATE_NAME" != "." && "$CANDIDATE_NAME" != ".." ]] || \
  die "candidate basename is unsafe"
CANDIDATE_OUTPUT="$CANDIDATE_PARENT/$CANDIDATE_NAME"
[[ ! -e "$CANDIDATE_OUTPUT" && ! -L "$CANDIDATE_OUTPUT" ]] || \
  die "candidate path must not exist: $CANDIDATE_OUTPUT"
[[ "$CANDIDATE_OUTPUT" != "$SOURCE_MODEL" && \
   "$CANDIDATE_OUTPUT" != "$SOURCE_MODEL/"* ]] || \
  die "candidate path must not be the protected source or lie inside it"

[[ -f "$SUCCESSOR_ROOT/preflight.json" ]] || die "successor preflight missing"
[[ -f "$SUCCESSOR_ROOT/successor_preflight_receipt.json" ]] || \
  die "successor receipt missing"
[[ -d "$PROJECT_ROOT" && -d "$BF16_ROOT" && -d "$CAPTURE_ROOT" ]] || \
  die "project, BF16 subset, or capture is missing"
[[ -d "$SOURCE_MODEL" && -d "$SQG_EXTENSION_ROOT" && \
   -d "$EXLLAMA_R7EXT_ROOT" ]] || die "sealed runtime input missing"

layers=(6 28 52 77)
gpus=(0 1 2 3)
starts=(0 64 128 192)
ends=(64 128 192 256)

# This handoff begins only after all four corrected selections are frozen.
# Merely checking these small receipts adds no model-payload hashing.
for layer in "${layers[@]}"; do
  printf -v padded_layer '%03d' "$layer"
  [[ -f "$SUCCESSOR_ROOT/layer_${padded_layer}/profile_search/selection.json" ]] || \
    die "corrected frozen selection missing for layer $layer"
done

mkdir -p "$SUCCESSOR_ROOT/logs"

workflow_names=()
for layer in "${layers[@]}"; do
  for shard in 0 1 2 3; do
    workflow_names+=("glm52-sqg-absfix-final-l${layer}-p${shard}")
  done
  workflow_names+=("glm52-sqg-absfix-assemble-l${layer}")
done
workflow_names+=("glm52-sqg-absfix-seal" "glm52-sqg-absfix-materialize")

for name in "${workflow_names[@]}"; do
  if docker container inspect "$name" >/dev/null 2>&1; then
    die "workflow container name already exists: $name"
  fi
done

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for name in "${workflow_names[@]}"; do
    docker rm -f "$name" >/dev/null 2>&1 || true
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

run_gpu_container() {
  local name="$1"
  local gpu="$2"
  local cpus="$3"
  shift 3
  docker run --rm --name "$name" \
    --network none --gpus all --shm-size 8g --cpus "$cpus" \
    --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
    --mount "type=bind,src=$BF16_ROOT,dst=$BF16_ROOT,readonly" \
    --mount "type=bind,src=$CAPTURE_ROOT,dst=/capture,readonly" \
    --mount "type=bind,src=$SQG_EXTENSION_ROOT,dst=/sqg-extension,readonly" \
    --mount "type=bind,src=$EXLLAMA_R7EXT_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
    --mount "type=bind,src=$SUCCESSOR_ROOT,dst=/output" \
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
    --entrypoint /opt/venv/bin/python \
    -w /work "$IMAGE" "$@"
}

wait_for_jobs() {
  local description="$1"
  shift
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || die "one or more $description failed"
}

# Timed encode path: 16 resumable workers, four per GPU, with no BF16 or
# capture payload rehash. Each worker owns a disjoint 64-expert directory.
pids=()
for index in 0 1 2 3; do
  layer="${layers[$index]}"
  gpu="${gpus[$index]}"
  for shard in 0 1 2 3; do
    start="${starts[$shard]}"
    end="${ends[$shard]}"
    name="glm52-sqg-absfix-final-l${layer}-p${shard}"
    run_gpu_container "$name" "$gpu" 3 \
      scripts/encode_final_shard.py \
        --preflight /output/preflight.json \
        --layer "$layer" --start "$start" --end "$end" \
        --device cuda:0 --threads 3 \
      >"$SUCCESSOR_ROOT/logs/final-l${layer}-${start}-${end}.log" 2>&1 &
    pids+=("$!")
  done
done
wait_for_jobs "final expert shard workers" "${pids[@]}"

# The four layer consolidators publish validated shard artifacts by hardlink
# where possible, then assemble the four final SQG layer payloads in parallel.
pids=()
for index in 0 1 2 3; do
  layer="${layers[$index]}"
  gpu="${gpus[$index]}"
  name="glm52-sqg-absfix-assemble-l${layer}"
  run_gpu_container "$name" "$gpu" 3 \
    scripts/consolidate_final_shards.py \
      --preflight /output/preflight.json --layer "$layer" --device cuda:0 \
    >"$SUCCESSOR_ROOT/logs/assemble-l${layer}.log" 2>&1 &
  pids+=("$!")
done
wait_for_jobs "layer assembly jobs" "${pids[@]}"

# The seal is the required four-layer boundary validation. It proves 3,072
# SQG tensors, zero MCG tensors, and the exact frozen per-tensor K3/K4 census.
run_gpu_container glm52-sqg-absfix-seal 0 4 \
  -m src.run_fresh_sqg seal-run --preflight /output/preflight.json \
  >"$SUCCESSOR_ROOT/logs/seal-run.log" 2>&1
[[ -f "$SUCCESSOR_ROOT/run_seal.json" ]] || die "run seal was not produced"

# Materialize into a brand-new sibling checkpoint. Source and destination must
# remain visible through the same parent bind mount so unchanged checkpoint
# shards can be hardlinked; a nested read-only source bind creates a distinct
# mount boundary and Linux rejects those hardlinks with EXDEV. The materializer
# itself enforces distinct source/output paths and only writes under output.
docker run --rm --name glm52-sqg-absfix-materialize \
  --network none --cpus 4 \
  --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
  --mount "type=bind,src=$SUCCESSOR_ROOT,dst=/output,readonly" \
  --mount "type=bind,src=$CANDIDATE_PARENT,dst=$CANDIDATE_PARENT" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work \
  --entrypoint /opt/venv/bin/python \
  -w /work "$IMAGE" \
  scripts/materialize_fast_directional.py \
    --source-model "$SOURCE_MODEL" \
    --teacher-receipt /work/evidence/teacher_model_identity.json \
    --run-seal /output/run_seal.json \
    --artifacts-root /output \
    --bit-contract /work/contracts/frozen_bit_allocations.json \
    --output "$CANDIDATE_OUTPUT" \
  >"$SUCCESSOR_ROOT/logs/materialize-candidate.log" 2>&1

[[ -f "$CANDIDATE_OUTPUT/.manifest_verified" && \
   -f "$CANDIDATE_OUTPUT/MANIFEST.json" ]] || \
  die "materialized candidate completion marker is absent"

printf 'Corrected four-layer SQG candidate complete: %s\n' "$CANDIDATE_OUTPUT"
