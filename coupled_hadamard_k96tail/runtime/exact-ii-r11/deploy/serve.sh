#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose="${root}/deploy/compose.yaml"
image="${IMAGE:-verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix}"

require_label() {
  local key="$1"
  local expected="$2"
  local actual
  actual="$(docker image inspect "${image}" --format "{{ index .Config.Labels \"${key}\" }}")"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'FATAL: image %s label %s=%q, expected %q\n' \
      "${image}" "${key}" "${actual}" "${expected}" >&2
    exit 1
  fi
}

# This qualification is intentionally exact II r11/MTP3. r13 is used only as
# the native-SQG donor and v20 is not an acceptable runtime base.
require_label ai.verdict.infernal-invocation.base r11
require_label ai.verdict.target.topology tp4-dcp4-mtp3
require_label ai.verdict.coupled.tp4-preactivation all-gather-reassembly-v1
require_label ai.verdict.mtp.exl3-prefix canonical-mtp-block-v1

mkdir -p "${JIT_CACHE:-${root}/cache}" "${RESULTS_DIR:-${root}/results}"
IMAGE="${image}" docker compose -p glm52-k96-ii-r11 -f "${compose}" up -d
docker compose -p glm52-k96-ii-r11 -f "${compose}" ps
