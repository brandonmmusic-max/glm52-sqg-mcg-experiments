#!/usr/bin/env bash
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 ]] || die "usage: $0 /absolute/artifacts-root /absolute/candidate-model"
[[ "$1" = /* && "$2" = /* ]] || die "both paths must be absolute"

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
MODELS_ROOT="/home/brandonmusic/models"
SOURCE_MODEL="$MODELS_ROOT/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED"
ARTIFACTS_ROOT="$(realpath -e -- "$1")"
CANDIDATE_MODEL="$2"
BIT_CONTRACT="${FRESH_SQG_BIT_CONTRACT:-/work/contracts/frozen_bit_allocations.json}"
EXLLAMA_ROOT="/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3"
SQG_EXTENSION_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33.p7n1IJ/sealed"
IMAGE="sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
EXTENSION="kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so"
EXTENSION_SHA256="c987778677653388f7766e66150850c4dda44bc6b055929ece34eaee87f3cda4"

[[ "$CANDIDATE_MODEL" == "$MODELS_ROOT"/* ]] || \
  die "candidate must be a direct model under $MODELS_ROOT"
[[ ! -e "$CANDIDATE_MODEL" ]] || die "candidate output already exists"
[[ -f "$ARTIFACTS_ROOT/run_seal.json" ]] || die "artifact run seal is absent"
[[ -d "$SOURCE_MODEL" && -d "$EXLLAMA_ROOT" ]] || die "source model or runtime is absent"

log="$ARTIFACTS_ROOT/logs/materialize-candidate.log"
receipt="$ARTIFACTS_ROOT/materialize-receipt.json"
[[ ! -e "$log" && ! -e "$receipt" ]] || \
  die "materialization log or receipt already exists"

docker run --rm --network none --gpus all --shm-size 16g \
  --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
  --mount "type=bind,src=$ARTIFACTS_ROOT,dst=/output,readonly" \
  --mount "type=bind,src=$MODELS_ROOT,dst=$MODELS_ROOT" \
  --mount "type=bind,src=$SQG_EXTENSION_ROOT,dst=/sqg-extension,readonly" \
  --mount "type=bind,src=$EXLLAMA_ROOT,dst=/opt/exllamav3-r7ext,readonly" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/exllamav3-r7ext:/opt/exllamav3-python:/work \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e KQUANT_SQG_REQUIRE_PREBUILT=1 \
  -e "KQUANT_SQG_EXTENSION_PATH=/sqg-extension/$EXTENSION" \
  -e "KQUANT_SQG_EXTENSION_SHA256=$EXTENSION_SHA256" \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e "FRESH_SQG_SELECTED_LAYERS=${FRESH_SQG_SELECTED_LAYERS:-6,28,52,77}" \
  -e "FRESH_SQG_BIT_CONTRACT_SHA256=${FRESH_SQG_BIT_CONTRACT_SHA256:-1fe5a065ef31c2e4c27589415b87bb77a91f095c55eaf4594807ca22645dab33}" \
  --entrypoint /opt/venv/bin/python -w /work "$IMAGE" \
  scripts/materialize_fast_directional.py \
    --source-model "$SOURCE_MODEL" \
    --teacher-receipt /work/evidence/teacher_model_identity.json \
    --run-seal /output/run_seal.json \
    --artifacts-root /output \
    --bit-contract "$BIT_CONTRACT" \
    --output "$CANDIDATE_MODEL" >"$receipt" 2>"$log"

docker run --rm --network none \
  --mount "type=bind,src=$CANDIDATE_MODEL,dst=/candidate" \
  alpine:3.20 chmod -R a+rX /candidate

jq -e --arg output "$CANDIDATE_MODEL" '
  .candidate.candidate == $output and
  .candidate.schema == "glm52-pure-sqg-four-layer-preflight-v2" and
  .candidate.selected_trellis_tensors == 3072 and
  .candidate.selected_sqg_markers == 3072 and
  .candidate.selected_mcg_markers == 0 and
  .candidate.marker_payloads_verified == true and
  .candidate.verified_marker_present == true and
  .candidate.source_bytes_mutated == false
' "$receipt" >/dev/null || die "materialized candidate receipt differs"

printf 'Materialized H13 blend candidate: %s\n' "$CANDIDATE_MODEL"
