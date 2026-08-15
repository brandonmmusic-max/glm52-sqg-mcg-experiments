#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 2 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ ]] || \
  die "usage: $0 START_LAYER END_LAYER"

start=$1
end=$2
((end >= start && end - start <= 3 && start >= 3 && end <= 77)) || \
  die "wave must contain one to four consecutive coupled routed layers"
wave=$(printf 'wave-%03d-%03d' "$start" "$end")
source_wave=$wave
manifest_layers=($(seq "$start" "$end"))
if ((start == 75 && end == 77)); then
  source_wave=wave-074-077
  manifest_layers=(74 75 76 77)
fi
repo=brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians
revision=a05b3b92d749f6a641af5cfd52de2b4720380dfd
stage=${WAVE_INPUT_ROOT:-/media/brandonmusic/nvme1n1p3/glm52-coupled-wave-inputs/$wave}

includes=(--include "capture_view/capture_manifest.json")
if [[ "$source_wave" == "$wave" ]]; then
  includes+=(--include "derived/wave_preflights/$wave/*")
else
  includes+=(
    --include "derived/wave_preflights/$source_wave/preflight.json"
    --include "derived/wave_preflights/$source_wave/bit-contract.json"
    --include "derived/wave_preflights/$source_wave/source-seal.json"
  )
fi
for layer in $(seq "$start" "$end"); do
  padded=$(printf '%03d' "$layer")
  includes+=(--include "capture_view/layer_${padded}/*")
  if [[ "$source_wave" != "$wave" ]]; then
    includes+=(--include "derived/wave_preflights/$source_wave/layer_${padded}/*")
  fi
done

mkdir -p "$stage"
hf download "$repo" --repo-type dataset --revision "$revision" \
  --local-dir "$stage" "${includes[@]}"

if [[ "$source_wave" != "$wave" ]]; then
  source_preflight=$stage/derived/wave_preflights/$source_wave
  target_preflight=$stage/derived/wave_preflights/$wave
  mkdir -p "$target_preflight"
  cp -aln "$source_preflight"/. "$target_preflight"/
fi

for required in \
  "$stage/derived/wave_preflights/$wave/preflight.json" \
  "$stage/derived/wave_preflights/$wave/bit-contract.json" \
  "$stage/derived/wave_preflights/$wave/source-seal.json" \
  "$stage/capture_view/capture_manifest.json"; do
  [[ -f "$required" && ! -L "$required" ]] || die "download is incomplete: $required"
done
for layer in $(seq "$start" "$end"); do
  padded=$(printf '%03d' "$layer")
  [[ -f "$stage/capture_view/layer_${padded}/hidden.bf16.bin" ]] || \
    die "capture is incomplete for layer $layer"
done

python3 /home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625/scripts/build_wave_bf16_manifest_from_source_seal.py \
  --source-seal "$stage/derived/wave_preflights/$wave/source-seal.json" \
  --layers "${manifest_layers[@]}" \
  --output "$stage/derived/wave_preflights/$wave/wave-bf16-shard-manifest.json"

printf 'coupled wave inputs complete: wave=%s root=%s\n' "$wave" "$stage"
