#!/usr/bin/env bash
# GLM-5.2 SQG full-W4A8 SM120 local serving driver.
#
# Commands:
#   ./serve.sh preflight   host + model + reference checks (no GPU load)
#   ./serve.sh build       build the SM120 image from the sealed build context
#   ./serve.sh probe       one-layer native kernel GPU probe
#   ./serve.sh start       start the acceptance server (MTP0, eager, BF16 KV)
#   ./serve.sh start-prod  start the production server (MTP3, FP8 KV) -- separate regime
#   ./serve.sh status      compose + health + GPU status
#   ./serve.sh logs [-f]   server logs
#   ./serve.sh smoke       deterministic 16-token completion + finite gate
#   ./serve.sh verify      prove native W4A8 path from logs + evidence receipts
#   ./serve.sh kld         sealed BF16-reference KLD (stops/restores this project's server)
#   ./serve.sh stop        stop this compose project's containers only
#   ./serve.sh restart     stop + start
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}"

PROJECT=glm52-sqg-coupled-h512-h128-sm120-local
COMPOSE=(docker compose -p "${PROJECT}" -f "${SCRIPT_DIR}/compose.yaml")
BASE_IMAGE_DIGEST="voipmonitor/vllm@sha256:7c0899fb9b3d09fbebcdd45f7b34cf5533e8e22ded03edbe524c4dcb1e367420"

if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
  echo "No .env found; copy .env.example to .env first." >&2
  exit 2
fi
set -a
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/.env"
set +a
: "${IMAGE:?IMAGE must be set in .env}"
: "${MODEL_DIR:?MODEL_DIR must be set in .env}"
: "${RESULTS_DIR:?RESULTS_DIR must be set in .env}"
: "${CACHE_DIR:?CACHE_DIR must be set in .env}"
mkdir -p "${RESULTS_DIR}" "${CACHE_DIR}"

log() { printf '[serve] %s\n' "$*"; }
die() { printf '[serve] ERROR: %s\n' "$*" >&2; exit 1; }

require_gpus() {
  local rows
  rows="$(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader 2>/dev/null)" \
    || die "nvidia-smi unavailable"
  local count
  count="$(wc -l <<<"${rows}")"
  (( count >= 4 )) || die "need 4 visible SM120 GPUs, found ${count}"
  local bad
  bad="$(awk -F', ' '$1<4 && $2!="12.0"' <<<"${rows}")"
  [[ -z "${bad}" ]] || die "GPUs 0-3 must be compute capability 12.0: ${bad}"
}

port_in_use_by_other() {
  local port="${HOST_PORT:-9418}"
  if ss -ltn "sport = :${port}" | grep -q LISTEN; then
    # Allowed when it is this project's own published port.
    if ! docker ps --filter "label=com.docker.compose.project=${PROJECT}" \
        --format '{{.Ports}}' | grep -q ":${port}->"; then
      die "port ${port} is already in use by another process"
    fi
  fi
}

verify_model_revision() {
  log "validating the local coupled checkpoint and exact declared mixed-rate census"
  python3 "${SCRIPT_DIR}/scripts/validate_model_codec.py" \
    --model "${MODEL_DIR}" \
    --revision local-coupled-assembly \
    --offline --quick --require-all-coupled \
    --result-json "${RESULTS_DIR}/model_codec_quick.json"
}

image_id() { docker image inspect --format '{{.Id}}' "${IMAGE}" 2>/dev/null || true; }

server_running() {
  docker ps --filter "name=glm52-sqg-coupled-sm120-accept" --filter status=running -q | grep -q . || return 1
}

cmd_preflight() {
  require_gpus
  port_in_use_by_other
  docker image inspect "${IMAGE}" >/dev/null 2>&1 || log "image ${IMAGE} not built yet (run ./serve.sh build)"
  [[ -d "${MODEL_DIR}" ]] || die "MODEL_DIR ${MODEL_DIR} does not exist"
  [[ -f "${REFERENCE_DIR}/manifest.json" ]] || die "sealed reference manifest missing under REFERENCE_DIR"
  python3 "${SCRIPT_DIR}/scripts/preflight.py" \
    --model "${MODEL_DIR}" \
    --reference "${REFERENCE_DIR}" \
    --port "${HOST_PORT:-9418}" \
    --image "${IMAGE}"
  verify_model_revision
  log "preflight PASS"
}

cmd_build() {
  log "building ${IMAGE} from sealed build context (base ${BASE_IMAGE_DIGEST})"
  docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE_DIGEST}" \
    --build-arg RELEASE_IMAGE_REF="${IMAGE}" \
    -t "${IMAGE}" \
    "${SCRIPT_DIR}/build-context"
  log "built image id: $(image_id)"
}

cmd_probe() {
  require_gpus
  "${COMPOSE[@]}" --profile tools run --rm gpu-probe
  log "gpu probe receipt: ${RESULTS_DIR}/gpu_probe_layer3.json"
}

cmd_start() {
  require_gpus
  port_in_use_by_other
  verify_model_revision
  mkdir -p "${RESULTS_DIR}/evidence"
  rm -f "${RESULTS_DIR}/evidence"/pp-*.json 2>/dev/null || true
  log "starting acceptance server (MTP0, enforce-eager, FP8 KV, TP1/PP4/DCP1) on ${BIND_ADDRESS:-127.0.0.1}:${HOST_PORT:-9418}"
  log "image: ${IMAGE} ($(image_id))"
  "${COMPOSE[@]}" up -d server
}

cmd_start_prod() {
  require_gpus
  port_in_use_by_other
  verify_model_revision
  log "starting PRODUCTION server (MTP3, FP8 KV). Benchmarks from this service are MTP3 results."
  "${COMPOSE[@]}" --profile prod up -d server-prod
}

cmd_status() {
  "${COMPOSE[@]}" ps
  local cid
  cid="$(docker ps -q --filter name=glm52-sqg-coupled-sm120)"
  if [[ -n "${cid}" ]]; then
    docker inspect --format '{{.Name}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' ${cid}
  fi
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
}

cmd_logs() {
  "${COMPOSE[@]}" logs "${@}" server 2>/dev/null || docker logs "${@}" glm52-sqg-coupled-sm120-accept
}

cmd_smoke() {
  python3 "${SCRIPT_DIR}/scripts/smoke_test.py" \
    --url "http://${BIND_ADDRESS:-127.0.0.1}:${HOST_PORT:-9418}" \
    --model "${SERVED_MODEL_NAME:-GLM-5.2-SQG-W4A8}" \
    --result-json "${RESULTS_DIR}/smoke_16tok.json"
}

cmd_verify() {
  python3 "${SCRIPT_DIR}/scripts/verify_runtime_path.py" \
    --container glm52-sqg-coupled-sm120-accept \
    --evidence-dir "${RESULTS_DIR}/evidence" \
    --expect-tp 1 --expect-pp 4 \
    --result-json "${RESULTS_DIR}/runtime_path_verification.json"
}

cmd_kld() {
  require_gpus
  local was_running=0
  if server_running; then
    was_running=1
    log "stopping this project's acceptance server to free GPUs for KLD"
    "${COMPOSE[@]}" stop server
  fi
  IMAGE_ID="$(image_id)" "${COMPOSE[@]}" --profile tools run --rm kld || {
    if (( was_running )); then "${COMPOSE[@]}" up -d server; fi
    die "KLD run failed; server restored=${was_running}. See ${RESULTS_DIR}/kld/"
  }
  log "KLD receipt: ${RESULTS_DIR}/kld/kld_sm120_tp1pp4dcp1.json"
  if (( was_running )); then
    log "restoring acceptance server"
    "${COMPOSE[@]}" up -d server
  fi
}

cmd_stop() {
  log "stopping compose project ${PROJECT} only"
  "${COMPOSE[@]}" --profile prod --profile tools down --remove-orphans=false
}

case "${1:-}" in
  preflight) shift; cmd_preflight "$@" ;;
  build) shift; cmd_build "$@" ;;
  probe) shift; cmd_probe "$@" ;;
  start) shift; cmd_start "$@" ;;
  start-prod) shift; cmd_start_prod "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs) shift; cmd_logs "$@" ;;
  smoke) shift; cmd_smoke "$@" ;;
  verify) shift; cmd_verify "$@" ;;
  kld) shift; cmd_kld "$@" ;;
  stop) shift; cmd_stop "$@" ;;
  restart) shift; cmd_stop; cmd_start ;;
  *)
    sed -n '2,16p' "${BASH_SOURCE[0]}"
    exit 2
    ;;
esac
