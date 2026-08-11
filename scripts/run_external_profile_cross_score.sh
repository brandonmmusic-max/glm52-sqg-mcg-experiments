#!/usr/bin/env bash
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 1; }

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
PROFILE_ROOT="${PROFILE_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-contig-late-a025-r1}"
FULL_CAPTURE_ROOT="${FULL_CAPTURE_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1}"
BF16_ROOT="${FRESH_SQG_BF16_ROOT:-$PROJECT_ROOT/bf16_contiguous_late}"
OUTPUT_ROOT="${1:-$PROJECT_ROOT/results/external_profile_cross_score_l77_r1}"
IMAGE="sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
GPU="${CROSS_SCORE_GPU:-0}"

[[ "$PROFILE_ROOT" = /* && -f "$PROFILE_ROOT/preflight.json" ]] || die "profile preflight missing"
[[ "$FULL_CAPTURE_ROOT" = /* && -f "$FULL_CAPTURE_ROOT/capture_manifest.json" ]] || die "full capture missing"
[[ "$OUTPUT_ROOT" = /* ]] || die "output root must be absolute"
mkdir -p "$OUTPUT_ROOT"

docker run --rm --name glm52-sqg-l77-external-cross-score \
  --network none --gpus all --shm-size 16g --cpus 16 \
  --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly" \
  --mount "type=bind,src=$BF16_ROOT,dst=$BF16_ROOT,readonly" \
  --mount "type=bind,src=$PROFILE_ROOT,dst=/profile,readonly" \
  --mount "type=bind,src=$FULL_CAPTURE_ROOT,dst=/full-capture,readonly" \
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/cross-score" \
  -e "CUDA_VISIBLE_DEVICES=$GPU" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/work/kquant:/work \
  -e OMP_NUM_THREADS=12 -e MKL_NUM_THREADS=12 \
  -e OPENBLAS_NUM_THREADS=12 -e NUMEXPR_NUM_THREADS=12 \
  -w /work "$IMAGE" \
  scripts/cross_score_reduced_profiles.py \
    --preflight /profile/preflight.json \
    --full-capture /full-capture \
    --full-plan /work/evidence/document_plan.json \
    --reduced-plan /work/evidence/contiguous_document_plan_r1.json \
    --preregistration /profile/layer_077/profile_search/preregistration.json \
    --draw0-artifacts /profile/layer_077/profile_search/cells/draw-00__identity/experts \
    --draw3-artifacts /profile/layer_077/profile_search/cells/draw-03__identity/experts \
    --output-dir /cross-score \
    --layer 77 --role selection --device cuda:0 --threads 12
