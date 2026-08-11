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
CAPTURE_ROOT="${FRESH_SQG_CAPTURE_ROOT:-$WORKSPACE_ROOT/fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1}"

[[ -d "$PROJECT_ROOT" && -d "$MODELS_ROOT" && -d "$EXLLAMA_ROOT" ]] || \
  die "project, model, or ExLlama runtime root is absent"
[[ -f "$SQG_EXTENSION_ROOT/$EXTENSION" ]] || die "sealed SQG extension is absent"
[[ -d "$CAPTURE_ROOT" && -f "$CAPTURE_ROOT/capture_manifest.json" ]] || \
  die "capture root is absent or incomplete: $CAPTURE_ROOT"
[[ ! -e "$OUTPUT" ]] || die "output already exists: $OUTPUT"
mkdir -p "$(dirname "$OUTPUT")"

candidate_args=()
for candidate in "$@"; do
  [[ "$candidate" == *=/* ]] || die "candidate must be LABEL=/absolute/path"
  path="${candidate#*=}"
  [[ -d "$path" ]] || die "candidate root is absent: $path"
  candidate_args+=(--candidate "$candidate")
done

dynamic_args=(
  --layers "${SQG_SCORE_LAYERS:-6,28,52,77}"
  --mcg-root "${FRESH_SQG_MCG_ROOT:-$MODELS_ROOT/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED}"
  --bf16-root "${FRESH_SQG_BF16_ROOT:-$PROJECT_ROOT/bf16_layers}"
  --capture-root "$CAPTURE_ROOT"
)
if [[ -n "${SQG_SCORE_MCG_BASELINE_LABEL:-}" ]]; then
  dynamic_args+=(--mcg-baseline-label "$SQG_SCORE_MCG_BASELINE_LABEL")
fi
if [[ -n "${FRESH_SQG_PERMUTATION_ROOT:-}" ]]; then
  [[ -d "$FRESH_SQG_PERMUTATION_ROOT" ]] || \
    die "permutation root is absent: $FRESH_SQG_PERMUTATION_ROOT"
  dynamic_args+=(--permutation-root "$FRESH_SQG_PERMUTATION_ROOT")
fi

docker run --rm --network none --gpus all --shm-size 64g \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=32g \
  --mount "type=bind,src=$WORKSPACE_ROOT,dst=$WORKSPACE_ROOT" \
  --mount "type=bind,src=$CAPTURE_ROOT,dst=$CAPTURE_ROOT,readonly" \
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
    --chunk-rows 1024 --output "$OUTPUT" \
    "${dynamic_args[@]}" "${candidate_args[@]}"

printf 'Signed top-8 %s score complete: %s\n' "$ROLE" "$OUTPUT"
