#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hidden_receipt=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/coupled-k96tail-hidden-replay-exact-ii-r11/hidden-replay-kld.json
container="${NAME:-glm52-k96-ii-r11-tp4dcp4mtp3}"
port="${PORT:-8000}"
quality_results="${QUALITY_RESULTS_DIR:-${root}/results/characterization-5x}"

log() {
  printf 'GLM exact-r11 characterization: %s %s\n' \
    "$(date --iso-8601=seconds)" "$*"
}

wait_for_gpu_drain() {
  local blockers
  while true; do
    blockers=$(
      nvidia-smi --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, '
            {
              gsub(/[[:space:]]/, "", $1)
              gsub(/[[:space:]]/, "", $2)
              if (($2 + 0) > 2048) print $1 ":" $2 "MiB"
            }
          '
    )
    [[ -z "${blockers}" ]] && return
    log "waiting for GPU allocations to clear: $(tr '\n' ' ' <<<"${blockers}")"
    sleep 15
  done
}

if [[ ! -f "${hidden_receipt}" ]]; then
  log "FATAL: hidden replay receipt is missing: ${hidden_receipt}"
  exit 1
fi

# A failed predeclared compatibility tolerance does not erase the measurement.
# This launcher permits downstream behavioral characterization only when the
# replay is complete, the repeated serving run is bit-for-bit stable, and the
# mean replay delta is within its predeclared limit. It does not convert the
# failed per-position replay gate into a qualification pass.
if ! jq -e '
  .complete == true and
  .total_positions == 2047 and
  .crosscheck.repeated_full_kld_mean_abs_delta == 0 and
  .crosscheck.repeated_full_kld_position_max_abs_delta == 0 and
  (.crosscheck.hidden_vs_baseline_mean_abs_delta <=
    .crosscheck.mean_delta_limit)
' "${hidden_receipt}" >/dev/null; then
  log "FATAL: hidden replay does not satisfy characterization prerequisites"
  exit 1
fi

qualification_pass="$(jq -r '.qualification_pass' "${hidden_receipt}")"
position_delta="$(jq -r '.crosscheck.hidden_vs_baseline_position_max_abs_delta' \
  "${hidden_receipt}")"
position_limit="$(jq -r '.crosscheck.position_delta_limit' "${hidden_receipt}")"
log "hidden replay measured qualification_pass=${qualification_pass}; max position delta=${position_delta}, limit=${position_limit}"
log "proceeding with independent TP4/DCP4/MTP3 behavioral characterization"

wait_for_gpu_drain
"${root}/deploy/serve.sh"

for ((attempt=1; attempt<=480; attempt++)); do
  if curl --fail --silent --max-time 10 \
      "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    log "server ready; starting frozen Estonia/LAVD five-run measurements"
    QUALITY_RESULTS_DIR="${quality_results}" \
      exec "${root}/deploy/run-quality-5x.sh"
  fi
  if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null \
      | grep -qx true; then
    log "FATAL: serving container is not running"
    docker logs --tail 200 "${container}" || true
    exit 1
  fi
  if (( attempt % 10 == 0 )); then
    log "server still loading (attempt ${attempt}/480)"
  fi
  sleep 15
done

log "FATAL: server did not become ready within two hours"
docker logs --tail 300 "${container}" || true
exit 1
