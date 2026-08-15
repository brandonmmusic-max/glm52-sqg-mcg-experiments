#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'archive Vast K96 evidence: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 3 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ ]] || \
  die "usage: $0 START END HF_REPO"

start=$1
end=$2
repo=$3
((start >= 4 && end <= 77 && start <= end && end - start <= 3)) || \
  die "invalid layer range: $start..$end"

project=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
recipe=/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1
scores=/media/brandonmusic/nvme1n1p3/glm52-coupled-tail-rate-scores-no-shortcut-v5
allocations=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-allocations-v1
work=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-work-v1
layers=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers
evidence=$project/evidence/full-coupled-k96tail-no-shortcut
wave=$(printf 'wave-%03d-%03d' "$start" "$end")
campaign_log=/workspace/k96-logs/campaign-$(printf '%03d-%03d' "$start" "$end").log
stage=/workspace/k96-evidence-staging/$wave
destination=$stage/reproduction/$wave

export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HF_HOME/token}
export HF_XET_HIGH_PERFORMANCE=1

[[ ! -L "$stage" ]] || die "unsafe staging root: $stage"
mkdir -p "$destination"

recipe_members=()
score_members=()
allocation_members=()
parity_members=()
for layer in $(seq "$start" "$end"); do
  padded=$(printf '%03d' "$layer")
  oracle=$layers/runtime-oracle-layer-${padded}.json
  [[ -f "$oracle" && ! -L "$oracle" ]] || die "runtime oracle absent: $oracle"
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-selected-layer-b12x-oracle-v2" and
     .complete == true and .pass == true and .finite == true and
     .nonzero == true and .layer == $layer and
     .bit_census == {"k3": 672, "k4": 96, "total": 768} and
     .bits_per_weight == 3.125 and
     .production_endpoint == "route_packed_direct_e4m3_w4a8"' \
    "$oracle" >/dev/null || die "runtime oracle differs: $oracle"

  recipe_members+=("layer_${padded}" "final_profiles/layer_${padded}")
  score_members+=("layer_${padded}")
  for path in "$allocations"/layer_${padded}.*.allocation.json; do
    [[ -f "$path" ]] || die "allocation evidence absent for layer $layer"
    allocation_members+=("$(basename "$path")")
  done
  parity_path=layer-${padded}-k096-score-encode-parity.json
  [[ -f "$evidence/$parity_path" ]] || die "parity evidence absent: $parity_path"
  parity_members+=("$parity_path")
done

tar -czf "$destination/recipe-and-profiles.tgz" -C "$recipe" "${recipe_members[@]}"
tar -czf "$destination/tail-scores.tgz" -C "$scores" "${score_members[@]}"
tar -czf "$destination/allocations.tgz" -C "$allocations" "${allocation_members[@]}"
tar -czf "$destination/parity-proofs.tgz" -C "$evidence" "${parity_members[@]}"
if [[ -d "$work/$wave" && ! -L "$work/$wave" ]]; then
  tar --exclude='*.safetensors' -czf "$destination/candidate-metadata.tgz" \
    -C "$work" "$wave"
fi
cp "$campaign_log" "$destination/campaign.log"
(
  cd "$destination"
  sha256sum -- *.tgz campaign.log > SHA256SUMS
  sha256sum -c SHA256SUMS
)

hf upload "$repo" "$stage" . --repo-type model \
  --commit-message "Publish sealed reproduction evidence for layers $start-$end"
printf 'wave reproduction evidence persisted to Hub: %s repo=%s\n' "$wave" "$repo"
