#!/usr/bin/env bash
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: $0 /absolute/empty/successor-root"

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
PREDECESSOR_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode.LFlh2R"
SUCCESSOR_ROOT="$1"
BF16_ROOT="$PROJECT_ROOT/bf16_layers"
CAPTURE_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1"
SQG_EXTENSION_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33.p7n1IJ/sealed"
EXLLAMA_R7EXT_ROOT="${FRESH_SQG_EXLLAMA_R7EXT_ROOT:-/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3}"
IMAGE="sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
EXTENSION_NAME="kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so"
EXTENSION_SHA256="c987778677653388f7766e66150850c4dda44bc6b055929ece34eaee87f3cda4"

[[ "$SUCCESSOR_ROOT" = /* ]] || die "successor root must be absolute"
[[ "$SUCCESSOR_ROOT" != "$PREDECESSOR_ROOT" ]] || die "successor root must differ"
[[ -d "$PROJECT_ROOT" && -d "$PREDECESSOR_ROOT" && -d "$BF16_ROOT" ]] || die "required input missing"
[[ -d "$CAPTURE_ROOT" && -d "$SQG_EXTENSION_ROOT" && -d "$EXLLAMA_R7EXT_ROOT" ]] || die "sealed runtime input missing"
if [[ -e "$SUCCESSOR_ROOT" ]]; then
  [[ -d "$SUCCESSOR_ROOT" ]] || die "successor root exists and is not a directory"
  [[ -z "$(find "$SUCCESSOR_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "successor root must be empty"
else
  mkdir -p "$SUCCESSOR_ROOT"
fi

exec docker run --rm --name glm52-sqg-absfix-import \
  --network none --gpus all --shm-size 8g --cpus 4 \
  --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
  --mount "type=bind,src=$BF16_ROOT,dst=$BF16_ROOT,readonly" \
  --mount "type=bind,src=$CAPTURE_ROOT,dst=/capture,readonly" \
  --mount "type=bind,src=$SQG_EXTENSION_ROOT,dst=/sqg-extension,readonly" \
  --mount "type=bind,src=$EXLLAMA_R7EXT_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
  --mount "type=bind,src=$PREDECESSOR_ROOT,dst=/predecessor,readonly" \
  --mount "type=bind,src=$SUCCESSOR_ROOT,dst=/output" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/exllamav3-r7ext:/opt/exllamav3-python:/work \
  -e GIT_CONFIG_COUNT=1 \
  -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=/work/kquant \
  -e KQUANT_SQG_EXTENSION_PATH="/sqg-extension/$EXTENSION_NAME" \
  -e KQUANT_SQG_EXTENSION_SHA256="$EXTENSION_SHA256" \
  -e KQUANT_SQG_REQUIRE_PREBUILT=1 \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e FRESH_SQG_RUNTIME_IMAGE_ID="$IMAGE" \
  -w /work "$IMAGE" \
  scripts/rebind_absolute_gate_scale_fix.py \
    --predecessor-root /predecessor \
    --output-root /output \
    --apply

