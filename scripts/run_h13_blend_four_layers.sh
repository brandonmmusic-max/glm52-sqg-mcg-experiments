#!/usr/bin/env bash
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 ]] || die "usage: $0 /absolute/fresh-output-root LOCAL_ALPHA"
[[ "$1" = /* ]] || die "output root must be absolute"
[[ "$2" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || die "LOCAL_ALPHA must be in [0,1]"

PROJECT_ROOT="/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test"
BF16_ROOT="${FRESH_SQG_BF16_ROOT:-$PROJECT_ROOT/bf16_layers}"
CAPTURE_ROOT="${FRESH_SQG_CAPTURE_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1}"
BASELINE_ROOT="${FRESH_SQG_BASELINE_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2}"
SQG_EXTENSION_ROOT="/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33.p7n1IJ/sealed"
EXLLAMA_ROOT="/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork/v39_ext/exllamav3"
IMAGE="sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82"
EXTENSION="kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so"
EXTENSION_SHA256="c987778677653388f7766e66150850c4dda44bc6b055929ece34eaee87f3cda4"
OUTPUT_ROOT="$(realpath -e -- "$1")"
LOCAL_ALPHA="$2"

[[ -d "$PROJECT_ROOT" && -d "$BF16_ROOT" && -d "$CAPTURE_ROOT" ]] || \
  die "project, BF16 subset, or capture is missing"
[[ -d "$BASELINE_ROOT" && -d "$SQG_EXTENSION_ROOT" && -d "$EXLLAMA_ROOT" ]] || \
  die "sealed runtime input is missing"
[[ "$OUTPUT_ROOT" != "$BASELINE_ROOT" ]] || die "output must differ from baseline"
mkdir -p "$OUTPUT_ROOT/logs"

IFS=, read -r -a layers <<<"${FRESH_SQG_SELECTED_LAYERS:-6,28,52,77}"
[[ ${#layers[@]} -eq 4 ]] || die "exactly four selected layers are required"
gpus=(0 1 2 3)
starts=(0 64 128 192)
ends=(64 128 192 256)
names=()

pilot_env=()
if [[ -n "${FRESH_SQG_BF16_MANIFEST:-}" ]]; then
  pilot_env+=( -e "FRESH_SQG_BF16_MANIFEST=$FRESH_SQG_BF16_MANIFEST" )
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for name in "${names[@]}"; do
    docker rm -f "$name" >/dev/null 2>&1 || true
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

common_opts=(
  --rm --network none --gpus all --shm-size 16g
  --mount "type=bind,src=$PROJECT_ROOT,dst=/work,readonly"
  --mount "type=bind,src=$BF16_ROOT,dst=$BF16_ROOT,readonly"
  --mount "type=bind,src=$CAPTURE_ROOT,dst=/capture,readonly"
  --mount "type=bind,src=$SQG_EXTENSION_ROOT,dst=/sqg-extension,readonly"
  --mount "type=bind,src=$EXLLAMA_ROOT,dst=/opt/exllamav3-r7ext,readonly"
  --mount "type=bind,src=$BASELINE_ROOT,dst=/output,readonly"
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/candidate"
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH=/opt/exllamav3-r7ext:/opt/exllamav3-python:/work
  -e OMP_NUM_THREADS=3 -e MKL_NUM_THREADS=3
  -e OPENBLAS_NUM_THREADS=3 -e NUMEXPR_NUM_THREADS=3
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory
  -e GIT_CONFIG_VALUE_0=/work/kquant
  -e "KQUANT_SQG_EXTENSION_PATH=/sqg-extension/$EXTENSION"
  -e "KQUANT_SQG_EXTENSION_SHA256=$EXTENSION_SHA256"
  -e KQUANT_SQG_REQUIRE_PREBUILT=1
  -e TORCH_CUDA_ARCH_LIST=12.0
  -e "FRESH_SQG_RUNTIME_IMAGE_ID=$IMAGE"
  -e "FRESH_SQG_SELECTED_LAYERS=${FRESH_SQG_SELECTED_LAYERS:-6,28,52,77}"
  -e "FRESH_SQG_PLAN_CONTRACT=${FRESH_SQG_PLAN_CONTRACT:-/work/evidence/document_plan.json}"
  -e "FRESH_SQG_BIT_CONTRACT_SHA256=${FRESH_SQG_BIT_CONTRACT_SHA256:-1fe5a065ef31c2e4c27589415b87bb77a91f095c55eaf4594807ca22645dab33}"
  "${pilot_env[@]}"
)

run_worker() {
  local name="$1" gpu="$2" layer="$3" start="$4" end="$5"
  docker run --name "$name" --cpus 3 \
    "${common_opts[@]}" -e "CUDA_VISIBLE_DEVICES=$gpu" \
    --entrypoint /opt/venv/bin/python -w /work "$IMAGE" \
    scripts/encode_expert_local_h13_shard.py \
      --preflight /output/preflight.json \
      --output-root /candidate \
      --layer "$layer" --start "$start" --end "$end" \
      --local-alpha "$LOCAL_ALPHA" \
      --device cuda:0 --threads 3 --chunk-rows 1024
}

pids=()
for index in 0 1 2 3; do
  layer="${layers[$index]}"
  gpu="${gpus[$index]}"
  for shard in 0 1 2 3; do
    start="${starts[$shard]}"
    end="${ends[$shard]}"
    name="glm52-sqg-h13a${LOCAL_ALPHA//./p}-l${layer}-p${shard}"
    names+=("$name")
    run_worker "$name" "$gpu" "$layer" "$start" "$end" \
      >"$OUTPUT_ROOT/logs/l${layer}-${start}-${end}.log" 2>&1 &
    pids+=("$!")
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
[[ "$failed" -eq 0 ]] || die "one or more fixed-alpha H13 workers failed"

assembly_pids=()
for index in 0 1 2 3; do
  layer="${layers[$index]}"
  gpu="${gpus[$index]}"
  name="glm52-sqg-h13a${LOCAL_ALPHA//./p}-assemble-${layer}"
  names+=("$name")
  docker run --name "$name" "${common_opts[@]}" \
    -e "CUDA_VISIBLE_DEVICES=$gpu" \
    --entrypoint /opt/venv/bin/python -w /work "$IMAGE" \
    scripts/assemble_expert_local_h13_layer.py \
      --preflight /output/preflight.json --output-root /candidate \
      --layer "$layer" --local-alpha "$LOCAL_ALPHA" --device cuda:0 \
      >"$OUTPUT_ROOT/logs/assemble-l${layer}.log" 2>&1 &
  assembly_pids+=("$!")
done
failed=0
for pid in "${assembly_pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
[[ "$failed" -eq 0 ]] || die "one or more fixed-alpha assemblies failed"

docker run --name "glm52-sqg-h13a${LOCAL_ALPHA//./p}-seal" \
  "${common_opts[@]}" -e CUDA_VISIBLE_DEVICES=0 \
  --entrypoint /opt/venv/bin/python -w /work "$IMAGE" \
  scripts/seal_expert_local_h13_run.py \
    --root /candidate --preflight /output/preflight.json \
    --local-alpha "$LOCAL_ALPHA" >"$OUTPUT_ROOT/logs/seal-run.log" 2>&1

docker run --rm --network none \
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/candidate" \
  alpine:3.20 chmod -R a+rX /candidate

printf 'Fixed-alpha H13 four-layer encode complete: alpha=%s root=%s\n' \
  "$LOCAL_ALPHA" "$OUTPUT_ROOT"
