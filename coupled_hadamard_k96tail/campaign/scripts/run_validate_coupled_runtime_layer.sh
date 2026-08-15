#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 1 && $1 =~ ^[0-9]+$ ]] || die "usage: $0 LAYER"
layer=$1
((layer >= 3 && layer <= 78)) || die "routed layer must lie in 3..78"
padded=$(printf '%03d' "$layer")

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
if [[ -x /workspace/k96-runtime/bundle/rootfs/lib64/ld-linux-x86-64.so.2 &&
      -x "$PROJECT_ROOT/scripts/run_validate_coupled_runtime_layer_vast_native.sh" ]]; then
  exec "$PROJECT_ROOT/scripts/run_validate_coupled_runtime_layer_vast_native.sh" "$layer"
fi
OVERLAY_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/build-context/overlay
QSRT_ROOT=/home/brandonmusic/KLC_SANDBOXES/qsrt-glm52-port
EXLLAMA_ROOT=${FRESH_SQG_EXLLAMA_EXTENSION_ROOT:-$PROJECT_ROOT/runtime-dependencies/v39_ext/exllamav3}
EXLLAMA_EXTENSION_SHA256=e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd
IMAGE=${IMAGE:-sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82}
LAYER_ROOT=${LAYER_ROOT:?Set LAYER_ROOT to the sealed selected-layer directory}
GPU=${GPU:-0}
TOKENS=${TOKENS:-2}
TOPK=${TOPK:-4}

for root in "$PROJECT_ROOT" "$OVERLAY_ROOT" "$QSRT_ROOT" "$EXLLAMA_ROOT" "$LAYER_ROOT"; do
  [[ -d "$root" && ! -L "$root" ]] || die "required directory is absent: $root"
done
exllama_extension=$EXLLAMA_ROOT/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
[[ -f "$exllama_extension" && ! -L "$exllama_extension" ]] || \
  die "pinned ExLlamaV3 extension is absent: $exllama_extension"
[[ $(sha256sum "$exllama_extension" | awk '{print $1}') == "$EXLLAMA_EXTENSION_SHA256" ]] || \
  die "pinned ExLlamaV3 extension hash differs: $exllama_extension"
shard="$LAYER_ROOT/r7-experts-layer-${padded}.safetensors"
manifest="$LAYER_ROOT/r7-experts-layer-${padded}.json"
result="$LAYER_ROOT/runtime-oracle-layer-${padded}.json"
[[ -f "$shard" && -f "$manifest" ]] || die "selected layer seal is absent: $padded"
[[ ! -e "$result" ]] || die "refusing existing runtime oracle: $result"

name="glm52-goal019ffa7c-runtime-oracle-l${layer}-r${BASHPID}"
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

docker run --rm --name "$name" --network none --gpus "device=$GPU" \
  --ipc=host --shm-size 32g --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/overlay:/qsrt:/work:/work/kquant:/opt/exllamav3-r7ext:/opt/exllamav3-python \
  -e B12X_GLM_W4A8_ACCEPT_ARCH=sm_120 \
  -e B12X_GLM_W4A8_KERNEL=m128n64 \
  -e B12X_GLM_W4A8_V2_BLOCKS=8 \
  -e B12X_GLM_W4A8_V2_STAGES=2 \
  -e CUTE_DSL_ARCH=sm_120a -e TORCH_CUDA_ARCH_LIST=12.0a \
  --mount "type=bind,src=$OVERLAY_ROOT,dst=/overlay,readonly" \
  --mount "type=bind,src=$QSRT_ROOT,dst=/qsrt,readonly" \
  --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
  --mount "type=bind,src=$LAYER_ROOT,dst=/layers" \
  --mount "type=bind,src=$EXLLAMA_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
  "$IMAGE" /work/scripts/validate_coupled_runtime_layer.py \
    --shard "/layers/$(basename "$shard")" \
    --manifest "/layers/$(basename "$manifest")" \
    --qsrt-root /qsrt --tokens "$TOKENS" --topk "$TOPK" \
    --result-json "/layers/$(basename "$result")"

printf 'coupled runtime oracle sealed: layer=%s result=%s\n' "$layer" "$result"
