#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test
OUTPUT_ROOT=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-alpha025-profile-native-l77-r1
CAPTURE_ROOT=/home/brandonmusic/KLC_CAPTURE_RUNS/contig-late-capture-r1/contig-late-r1
BF16_ROOT=$PROJECT_ROOT/bf16_contiguous_late
EXTENSION_ROOT=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33-saturation.VUybIb/sealed
EXLLAMA_ROOT=/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3
IMAGE=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
EXTENSION=kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so
EXTENSION_SHA256=d29010f6ad51caf2e1a22f07365ab3548fcdb3e0ed3ee15d88330cee24de9614
NAME=glm52-alpha025-retained-corouting-l77

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

docker run --name "$NAME" --rm --network none --gpus all --ipc=host --cpus 24 \
  --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/output" \
  --mount "type=bind,src=$CAPTURE_ROOT,dst=/capture,readonly" \
  --mount "type=bind,src=$BF16_ROOT,dst=$BF16_ROOT,readonly" \
  --mount "type=bind,src=$EXTENSION_ROOT,dst=/sqg-extension,readonly" \
  --mount "type=bind,src=$EXLLAMA_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
  -e HOME=/tmp -e CUDA_VISIBLE_DEVICES=3 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/exllamav3-r7ext:/opt/exllamav3-python:/work \
  -e OMP_NUM_THREADS=24 -e MKL_NUM_THREADS=24 \
  -e OPENBLAS_NUM_THREADS=24 -e NUMEXPR_NUM_THREADS=24 \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=/work/kquant \
  -e KQUANT_SQG_REQUIRE_PREBUILT=1 \
  -e KQUANT_SQG_EXTENSION_PATH=/sqg-extension/$EXTENSION \
  -e KQUANT_SQG_EXTENSION_SHA256=$EXTENSION_SHA256 \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e FRESH_SQG_RUNTIME_IMAGE_ID=$IMAGE \
  -e FRESH_SQG_SELECTED_LAYERS=74,75,76,77 \
  -e FRESH_SQG_PLAN_CONTRACT=/work/evidence/contiguous_document_plan_r1.json \
  -e FRESH_SQG_BIT_CONTRACT_SHA256=b70773e4d11fb0495d74f4a6977d1e10a9417907d31efdc3ea5de0b243c32a09 \
  -e FRESH_SQG_BF16_MANIFEST=/work/evidence/contiguous_late_bf16_manifest_r1.json \
  -e FRESH_SQG_EXL3_RUNTIME_SHA256=4f17ea448dba5c79b3c5ca29c352251778039d293dd5ca53ba685633d1295e5b \
  --entrypoint /opt/venv/bin/python -w /work "$IMAGE" \
  scripts/run_corouting_retained_profiles.py \
    --preflight /output/preflight.json --layer 77 --device cuda:0 \
    --threads 24 --chunk-rows 256

docker run --rm --network none \
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/output" \
  alpine:3.20 chmod -R a+rX /output/layer_077/winner_native_profile_search/corouting_retained_r1
