#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'prepare Vast K96 node: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 2 || $# -eq 4 ]] || \
  die "usage: $0 START1 END1 [START2 END2]"
for value in "$@"; do [[ "$value" =~ ^[0-9]+$ ]] || die "layer values must be integers"; done

project=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
source_root=/home/brandonmusic/models/GLM-5.2-SQG-W4A8
source_repo=brandonmusic/GLM-5.2-SQG-W4A8
source_revision=593dd0d2de6f79ce4e65303930c22c75e1359d44
control_repo=brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail
control_revision=75ad511457186492d75ed00f58634d4fbcd83f7d
control_stage=/workspace/k96-source-allocation-inputs
control_source=$control_stage/reproduction/source-allocation-inputs
control_root=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/coupled-tail-source-control-tp4dcp1-routes-v2/kld
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HF_HOME/token}
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$source_root" /workspace/k96-logs

ranges=("$1:$2")
if (($# == 4)); then ranges+=("$3:$4"); fi
includes=(
  --include model.safetensors.index.json
  --include quantization_config.json
  --include FULL_SQG_NATIVE_MANIFEST.json
)
for range in "${ranges[@]}"; do
  IFS=: read -r start end <<< "$range"
  ((start >= 3 && end <= 77 && start <= end && end - start <= 3)) || \
    die "invalid wave range: $range"
  for layer in $(seq "$start" "$end"); do
    includes+=(--include "r7-experts-layer-$(printf '%03d' "$layer").safetensors")
  done
done

hf download "$source_repo" --revision "$source_revision" \
  --local-dir "$source_root" --max-workers 8 "${includes[@]}" \
  > /workspace/k96-logs/source-download.log 2>&1 &
pids=("$!")
labels=(source)
hf download "$control_repo" --revision "$control_revision" \
  --local-dir "$control_stage" --max-workers 4 \
  --include 'reproduction/source-allocation-inputs/*' \
  > /workspace/k96-logs/source-allocation-inputs-download.log 2>&1 &
pids+=("$!")
labels+=(source-allocation-inputs)
for range in "${ranges[@]}"; do
  IFS=: read -r start end <<< "$range"
  wave=$(printf 'wave-%03d-%03d' "$start" "$end")
  "$project/scripts/download_coupled_wave_inputs.sh" "$start" "$end" \
    > "/workspace/k96-logs/${wave}-inputs.log" 2>&1 &
  pids+=("$!")
  labels+=("$wave")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'preparation download failed: %s\n' "${labels[$index]}" >&2
    failed=1
  fi
done
((failed == 0)) || exit 1

(
  cd "$control_source"
  sha256sum -c SHA256SUMS
) || die "source allocation inputs failed their checksum manifest"
mkdir -p "$control_root"
for name in kld_sm120_tp4dcp1.json routed_experts_tp4dcp1.npz; do
  source_path=$control_source/$name
  target_path=$control_root/$name
  [[ -f "$source_path" && ! -L "$source_path" ]] || \
    die "source allocation input is incomplete: $source_path"
  if [[ -e "$target_path" ]]; then
    [[ -f "$target_path" && ! -L "$target_path" ]] || \
      die "unsafe source allocation target: $target_path"
    [[ $(sha256sum "$source_path" | cut -d' ' -f1) == \
       $(sha256sum "$target_path" | cut -d' ' -f1) ]] || \
      die "existing source allocation input differs: $target_path"
  else
    install -m 0644 "$source_path" "$target_path"
  fi
done

for required in model.safetensors.index.json quantization_config.json \
  FULL_SQG_NATIVE_MANIFEST.json; do
  [[ -f "$source_root/$required" && ! -L "$source_root/$required" ]] || \
    die "source metadata is incomplete: $required"
done
for range in "${ranges[@]}"; do
  IFS=: read -r start end <<< "$range"
  for layer in $(seq "$start" "$end"); do
    shard=$source_root/r7-experts-layer-$(printf '%03d' "$layer").safetensors
    [[ -f "$shard" && ! -L "$shard" ]] || die "source shard is incomplete: $shard"
  done
done

printf 'Vast node inputs prepared for ranges: %s\n' "${ranges[*]}"
