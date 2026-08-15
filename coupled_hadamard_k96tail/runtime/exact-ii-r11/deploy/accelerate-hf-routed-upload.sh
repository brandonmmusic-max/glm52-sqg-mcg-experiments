#!/usr/bin/env bash
set -euo pipefail

repo="${HF_MODEL_REPO:-brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail}"
model_root="${MODEL_ROOT:-/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail}"
# One default adaptive Xet client already adapts to multiple internal streams.
# Multiple outer clients drove the uplink into CAS timeouts and reduced the
# measured end-to-end transfer rate.
workers="${UPLOAD_WORKERS:-1}"
hf_cli="${HF_CLI:-/home/brandonmusic/.local/bin/hf}"

upload_layer() {
  local layer="$1"
  local name="r7-experts-layer-${layer}.safetensors"
  local source="${model_root}/${name}"

  [[ -f "${source}" ]] || {
    printf 'missing routed layer file: %s\n' "${source}" >&2
    return 1
  }

  env -u HF_XET_HIGH_PERFORMANCE -u HF_HUB_DISABLE_XET \
    "${hf_cli}" upload \
    "${repo}" "${source}" "${name}" \
    --repo-type model --revision main --quiet \
    --commit-message "Upload final coupled-K96 routed layer ${layer}"
}

export repo model_root hf_cli
export -f upload_layer

# Layers 51--77 were already uploaded byte-identically by the rental nodes.
# Upload only the locally finalized coupled layers plus preserved MTP78.
{
  seq -w 003 050
  printf '%s\n' 078
} | xargs -r -n 1 -P "${workers}" bash -c 'upload_layer "$1"' _
