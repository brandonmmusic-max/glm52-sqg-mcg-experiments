#!/usr/bin/env bash
set -euo pipefail

repo=brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail
model=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail
hf=/home/brandonmusic/.local/bin/hf
hf_python="${HF_PYTHON:-/home/brandonmusic/.hf-cli/venv/bin/python}"
verification_receipt="${model}/HUB_FILE_VERIFICATION.json"

log() {
  printf 'GLM final Hub upload: %s %s\n' "$(date --iso-8601=seconds)" "$*"
}

# Exact-r11 serving and quality evidence are already sealed before publication.
# The server is intentionally stopped to release all four GPUs, so publication
# has no live-serving dependency.

log "starting resumable public tensor/config upload with default adaptive Xet mode"
unset HF_XET_HIGH_PERFORMANCE
unset HF_HUB_DISABLE_XET
export HF_HOME=/home/brandonmusic/.cache/huggingface
export HF_TOKEN_PATH=/home/brandonmusic/.cache/huggingface/token
"${hf}" upload-large-folder "${repo}" "${model}" \
  --repo-type model --revision main --num-workers 2 --no-bars \
  --exclude README.md \
  --exclude HUB_FILE_VERIFICATION.json

log "verifying every uploaded model file against Hub metadata"
"${hf_python}" "$(dirname "${BASH_SOURCE[0]}")/verify-final-hf-model.py" \
  --repo "${repo}" \
  --model-root "${model}" \
  --exclude README.md \
  --exclude HUB_FILE_VERIFICATION.json \
  --output "${verification_receipt}"

log "publishing the Hub file-verification receipt"
"${hf}" upload "${repo}" "${verification_receipt}" \
  HUB_FILE_VERIFICATION.json \
  --repo-type model --revision main \
  --commit-message 'Verify K96Tail model files against Hub metadata'

log "model payload and support-file upload verified"
