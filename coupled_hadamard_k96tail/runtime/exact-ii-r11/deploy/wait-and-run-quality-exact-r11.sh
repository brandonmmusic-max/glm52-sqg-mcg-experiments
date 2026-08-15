#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hidden_receipt=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/coupled-k96tail-hidden-replay-exact-ii-r11/hidden-replay-kld.json
hidden_service=glm52-k96tail-exact-r11-hidden-replay.service
container="${NAME:-glm52-k96-ii-r11-tp4dcp4mtp3}"
port="${PORT:-8000}"

log() {
  printf 'GLM exact-r11 qualification: %s %s\n' "$(date --iso-8601=seconds)" "$*"
}

jq_pass() {
  [[ -f "$1" ]] && jq -e "$2" "$1" >/dev/null 2>&1
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

while ! jq_pass "${hidden_receipt}" \
  '.complete == true and .qualification_pass == true and .total_positions == 2047'; do
  if systemctl --user is-failed --quiet "${hidden_service}"; then
    log "FATAL: hidden replay service failed before producing a sealed receipt"
    systemctl --user --no-pager --full status "${hidden_service}" || true
    exit 1
  fi
  log "waiting for exact-r11 hidden-replay qualification"
  sleep 30
done

wait_for_gpu_drain
log "starting exact II r11 TP4/DCP4/MTP3 server"
"${root}/deploy/serve.sh"

for ((attempt=1; attempt<=480; attempt++)); do
  if curl --fail --silent --max-time 10 \
      "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    log "server ready; starting frozen Estonia/LAVD five-run measurements"
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

