#!/usr/bin/env bash
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 1; }

[[ $# -ge 4 ]] || \
  die "usage: $0 selection|holdout /absolute/output.json BASELINE_LABEL LABEL=/absolute/path [...]"
[[ "$1" == selection || "$1" == holdout ]] || die "role must be selection or holdout"
[[ "$2" = /* ]] || die "output must be absolute"

ROLE="$1"
OUTPUT="$2"
BASELINE_LABEL="$3"
shift 3

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
WORKSPACE_ROOT="/home/brandonmusic/KLC_SANDBOXES"
MODELS_ROOT="/home/brandonmusic/models"
EXLLAMA_ROOT="/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3"
SQG_EXTENSION_ROOT="$WORKSPACE_ROOT/fresh-sqg-extension-r33.p7n1IJ/sealed"
IMAGE="sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
EXTENSION="kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so"
EXTENSION_SHA256="c987778677653388f7766e66150850c4dda44bc6b055929ece34eaee87f3cda4"

[[ -d "$PROJECT_ROOT" && -d "$MODELS_ROOT" && -d "$EXLLAMA_ROOT" ]] || \
  die "project, model, or ExLlama runtime root is absent"
[[ -f "$SQG_EXTENSION_ROOT/$EXTENSION" ]] || die "sealed SQG extension is absent"
[[ ! -e "$OUTPUT" ]] || die "output already exists: $OUTPUT"
mkdir -p "$(dirname "$OUTPUT")"

candidate_args=()
for candidate in "$@"; do
  [[ "$candidate" == *=/* ]] || die "candidate must be LABEL=/absolute/path"
  path="${candidate#*=}"
  [[ -d "$path" ]] || die "candidate root is absent: $path"
  candidate_args+=(--candidate "$candidate")
done

docker run --rm --network none --gpus all --shm-size 64g \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=32g \
  --mount "type=bind,src=$WORKSPACE_ROOT,dst=$WORKSPACE_ROOT" \
  --mount "type=bind,src=$MODELS_ROOT,dst=$MODELS_ROOT,readonly" \
  --mount "type=bind,src=$EXLLAMA_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/exllamav3-r7ext:/opt/exllamav3-python:"$PROJECT_ROOT" \
  -e KQUANT_SQG_REQUIRE_PREBUILT=1 \
  -e "KQUANT_SQG_EXTENSION_PATH=$SQG_EXTENSION_ROOT/$EXTENSION" \
  -e "KQUANT_SQG_EXTENSION_SHA256=$EXTENSION_SHA256" \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  --entrypoint /opt/venv/bin/python -w "$PROJECT_ROOT" "$IMAGE" \
  scripts/score_signed_top8_blends.py \
    --role "$ROLE" --baseline-label "$BASELINE_LABEL" \
    --chunk-rows 1024 --output "$OUTPUT" "${candidate_args[@]}"

printf 'Signed top-8 %s score complete: %s\n' "$ROLE" "$OUTPUT"
