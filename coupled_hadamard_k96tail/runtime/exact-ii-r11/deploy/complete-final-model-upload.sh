#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
routed_unit="${ROUTED_UPLOAD_UNIT:-glm52-k96tail-final-model-upload-routed.service}"

log() {
  printf 'GLM canonical final Hub upload: %s %s\n' \
    "$(date --iso-8601=seconds)" "$*"
}

# A targeted routed-layer uploader may already be transferring the only blobs
# that are not present in an existing public repository. Let it finish first so
# the canonical full-folder pass can deduplicate those paths and then commit all
# remaining source-identical K6/BF16/config files without competing for uplink.
while systemctl --user is-active --quiet "${routed_unit}"; do
  log "waiting for the targeted routed-layer uploader"
  sleep 60
done

log "starting the resumable canonical full-folder verification/upload pass"
exec "${root}/deploy/wait-and-upload-final-model.sh"
