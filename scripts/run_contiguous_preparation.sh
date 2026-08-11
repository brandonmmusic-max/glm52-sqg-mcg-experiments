#!/usr/bin/env bash
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: env FRESH_SQG_... $0 /absolute/output-root"

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
OUTPUT_ROOT="$1"
BF16_ROOT="${FRESH_SQG_BF16_ROOT:?FRESH_SQG_BF16_ROOT is required}"
CAPTURE_ROOT="${FRESH_SQG_CAPTURE_ROOT:?FRESH_SQG_CAPTURE_ROOT is required}"
SOURCE_SEAL="${FRESH_SQG_SOURCE_SEAL:?FRESH_SQG_SOURCE_SEAL is required}"
BIT_CONTRACT="${FRESH_SQG_BIT_CONTRACT:?FRESH_SQG_BIT_CONTRACT is required}"
LAYERS_CSV="${FRESH_SQG_SELECTED_LAYERS:?FRESH_SQG_SELECTED_LAYERS is required}"
PLAN_CONTRACT="${FRESH_SQG_PLAN_CONTRACT:?FRESH_SQG_PLAN_CONTRACT is required}"
BF16_MANIFEST="${FRESH_SQG_BF16_MANIFEST:?FRESH_SQG_BF16_MANIFEST is required}"
BIT_SHA256="${FRESH_SQG_BIT_CONTRACT_SHA256:?FRESH_SQG_BIT_CONTRACT_SHA256 is required}"
SQG_EXTENSION_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33.p7n1IJ/sealed"
EXLLAMA_ROOT="/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3"
IMAGE="sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
EXTENSION="kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so"
EXTENSION_SHA256="c987778677653388f7766e66150850c4dda44bc6b055929ece34eaee87f3cda4"
IFS=, read -r -a layers <<<"$LAYERS_CSV"
[[ ${#layers[@]} -eq 4 ]] || die "exactly four layers are required"

[[ "$OUTPUT_ROOT" = /* ]] || die "output root must be absolute"
if [[ -e "$OUTPUT_ROOT" ]]; then
  [[ -d "$OUTPUT_ROOT" && -z "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]] || die "output root must be empty"
else
  mkdir -p "$OUTPUT_ROOT"
fi

common=(
  --rm --network none --gpus all --shm-size 16g
  --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly"
  --mount "type=bind,src=$BF16_ROOT,dst=$BF16_ROOT,readonly"
  --mount "type=bind,src=$CAPTURE_ROOT,dst=/capture,readonly"
  --mount "type=bind,src=$SQG_EXTENSION_ROOT,dst=/sqg-extension,readonly"
  --mount "type=bind,src=$EXLLAMA_ROOT,dst=/opt/exllamav3-r7ext,readonly"
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/output"
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH=/opt/exllamav3-r7ext:/opt/exllamav3-python:/work
  -e "FRESH_SQG_SELECTED_LAYERS=$LAYERS_CSV"
  -e "FRESH_SQG_PLAN_CONTRACT=$PLAN_CONTRACT"
  -e "FRESH_SQG_BF16_MANIFEST=$BF16_MANIFEST"
  -e "FRESH_SQG_BIT_CONTRACT_SHA256=$BIT_SHA256"
  -e FRESH_SQG_EXL3_RUNTIME_SHA256=4f17ea448dba5c79b3c5ca29c352251778039d293dd5ca53ba685633d1295e5b
  -e "KQUANT_SQG_EXTENSION_PATH=/sqg-extension/$EXTENSION"
  -e "KQUANT_SQG_EXTENSION_SHA256=$EXTENSION_SHA256"
  -e KQUANT_SQG_REQUIRE_PREBUILT=1
  -e TORCH_CUDA_ARCH_LIST=12.0
  -e "FRESH_SQG_RUNTIME_IMAGE_ID=$IMAGE"
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory
  -e GIT_CONFIG_VALUE_0=/work/kquant
)

docker run --name glm52-contig-preflight "${common[@]}" \
  --entrypoint /opt/venv/bin/python \
  -w /work "$IMAGE" \
  -m src.run_fresh_sqg preflight \
  --source-seal "$SOURCE_SEAL" \
  --capture-dir /capture \
  --bit-contract "$BIT_CONTRACT" \
  --kquant-root /work/kquant \
  --exllamav3-root /opt/exllamav3-python \
  --sqg-extension-seal /sqg-extension/sqg-extension-seal.json \
  --output-root /output \
  --run-id "glm52-contiguous-${layers[0]}-${layers[3]}-alpha025-r1" \
  --skip-capture-payload-hashes \
  --skip-source-payload-hashes

docker run --name glm52-contig-receipt "${common[@]}" \
  --entrypoint /opt/venv/bin/python \
  -w /work "$IMAGE" \
  scripts/write_pilot_preflight_receipt.py --preflight /output/preflight.json

pids=()
for index in 0 1 2 3; do
  layer="${layers[$index]}"
  docker run --name "glm52-contig-prepare-$layer" --cpus 6 \
    "${common[@]}" -e "CUDA_VISIBLE_DEVICES=$index" \
    --entrypoint /opt/venv/bin/python \
    -w /work "$IMAGE" \
    scripts/prepare_layer_fast.py \
    --preflight /output/preflight.json --layer "$layer" --device cuda:0 \
    >"$OUTPUT_ROOT/prepare-$layer.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
[[ $failed -eq 0 ]] || die "one or more contiguous preparation workers failed"

docker run --name glm52-contig-real-smoke --cpus 4 \
  "${common[@]}" -e CUDA_VISIBLE_DEVICES=0 \
  --entrypoint /opt/venv/bin/python \
  -w /work "$IMAGE" \
  scripts/smoke_absolute_gate_scale.py \
  --preflight /output/preflight.json --layer "${layers[0]}" \
  --device cuda:0 --threads 4 >"$OUTPUT_ROOT/absolute-scale-smoke.log" 2>&1

printf 'Contiguous preparation complete: %s\n' "$OUTPUT_ROOT"
