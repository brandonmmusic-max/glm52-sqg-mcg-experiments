#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${HF_MODEL_REPO:-brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail}"
model_root="${MODEL_ROOT:-/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail}"
publication_root="${PUBLICATION_ROOT:-/home/brandonmusic/KLC_SANDBOXES/glm52-sqg-mcg-experiments-github/coupled_hadamard_k96tail}"
hf_cli="${HF_CLI:-/home/brandonmusic/.local/bin/hf}"
bulk_unit="${BULK_UPLOAD_UNIT:-glm52-k96tail-final-model-upload.service}"
ready_marker="${PUBLICATION_READY_MARKER:-${root}/results/hf-publication.ready}"
complete_marker="${root}/results/hf-publication.complete"
published_provenance="${root}/results/RELEASE_PROVENANCE.published.json"
hub_file_receipt="${model_root}/HUB_FILE_VERIFICATION.json"

log() {
  printf 'GLM final Hub publication: %s %s\n' "$(date --iso-8601=seconds)" "$*"
}

while [[ ! -f "${ready_marker}" ]]; do
  log "waiting for the final card, receipts, and reproduction closure"
  sleep 60
done

while systemctl --user is-active --quiet "${bulk_unit}"; do
  log "waiting for the resumable tensor/config upload"
  sleep 60
done

bulk_result="$(systemctl --user show "${bulk_unit}" --property=Result --value)"
bulk_status="$(systemctl --user show "${bulk_unit}" --property=ExecMainStatus --value)"
if [[ "${bulk_result}" != success || "${bulk_status}" != 0 ]]; then
  log "FATAL: bulk uploader result=${bulk_result} status=${bulk_status}"
  exit 1
fi
jq -e '
  .schema == "glm52-k96tail-hub-file-verification-v1" and
  .complete == true and
  (.missing_paths | length) == 0 and
  (.mismatches | length) == 0
' "${hub_file_receipt}" >/dev/null

export HF_XET_HIGH_PERFORMANCE=1
unset HF_HUB_DISABLE_XET

log "uploading the complete browsable reproduction, scripts, and results closure"
"${hf_cli}" upload "${repo}" "${publication_root}" \
  reproduction/coupled_hadamard_k96tail \
  --repo-type model --revision main \
  --exclude '**/__pycache__/**' \
  --exclude '**/.pytest_cache/**' \
  --exclude '**/*.pyc' \
  --commit-message 'Publish K96Tail reproduction, scripts, and sealed results'

# Publish the model card after every link target is public. Release provenance
# is promoted from local staging state only after the anonymous checks below.
log "publishing final model card"
"${hf_cli}" upload "${repo}" "${model_root}/README.md" README.md \
  --repo-type model --revision main \
  --commit-message 'Publish final K96Tail model card and scores'

api_json="$(curl --fail --silent --show-error --max-time 60 \
  "https://huggingface.co/api/models/${repo}")"
jq -e '.private == false and (.sha | type == "string" and length == 40)' \
  <<<"${api_json}" >/dev/null

expected_files="$(mktemp)"
remote_files="$(mktemp)"
missing_files="$(mktemp)"
card_download="$(mktemp)"
provenance_download="$(mktemp)"
trap 'rm -f "${expected_files}" "${remote_files}" "${missing_files}" \
  "${card_download}" "${provenance_download}"' EXIT
find "${model_root}" -maxdepth 1 -type f -printf '%f\n' | sort -u \
  >"${expected_files}"
jq -r '.siblings[].rfilename' <<<"${api_json}" | sort -u >"${remote_files}"
comm -23 "${expected_files}" "${remote_files}" >"${missing_files}"
if [[ -s "${missing_files}" ]]; then
  log "FATAL: anonymous Hub API is missing local model files"
  sed -n '1,40p' "${missing_files}" >&2
  exit 1
fi

verified_card_revision="$(jq -r '.sha' <<<"${api_json}")"
curl --fail --silent --show-error --max-time 120 \
  "https://huggingface.co/${repo}/resolve/${verified_card_revision}/README.md" \
  >"${card_download}"
cmp "${model_root}/README.md" "${card_download}"

jq \
  --arg verified_at "$(date --iso-8601=seconds)" \
  --arg verified_card_revision "${verified_card_revision}" \
  '.status = "published" |
   .complete = true |
   .publication.complete = true |
   .publication.ready_marker_present = true |
   .publication.tensor_config_upload_complete = true |
   .publication.anonymous_verification_complete = true |
   .publication.verified_at = $verified_at |
   .publication.verified_card_revision = $verified_card_revision |
   .publication.public_revision = null |
   .publication.public_revision_semantics =
     "The non-self-referential final repository head is recorded in the local hf-publication receipt."' \
  "${model_root}/RELEASE_PROVENANCE.json" >"${published_provenance}"

log "uploading anonymously verified final release provenance"
"${hf_cli}" upload "${repo}" \
  "${published_provenance}" RELEASE_PROVENANCE.json \
  --repo-type model --revision main \
  --commit-message 'Seal K96Tail release provenance'

final_api_json="$(curl --fail --silent --show-error --max-time 60 \
  "https://huggingface.co/api/models/${repo}")"
jq -e '.private == false and (.sha | type == "string" and length == 40)' \
  <<<"${final_api_json}" >/dev/null
final_revision="$(jq -r '.sha' <<<"${final_api_json}")"
curl --fail --silent --show-error --max-time 120 \
  "https://huggingface.co/${repo}/resolve/${final_revision}/RELEASE_PROVENANCE.json" \
  >"${provenance_download}"
cmp "${published_provenance}" "${provenance_download}"

model_card_sha256="$(sha256sum "${model_root}/README.md" | awk '{print $1}')"
provenance_sha256="$(sha256sum "${published_provenance}" | awk '{print $1}')"
hub_file_verification_sha256="$(sha256sum "${hub_file_receipt}" | awk '{print $1}')"
expected_file_count="$(wc -l <"${expected_files}")"
jq -n \
  --arg schema glm52-k96tail-hf-publication-v1 \
  --arg created_at "$(date --iso-8601=seconds)" \
  --arg repo "${repo}" \
  --arg revision "${final_revision}" \
  --arg verified_card_revision "${verified_card_revision}" \
  --arg model_card_sha256 "${model_card_sha256}" \
  --arg provenance_sha256 "${provenance_sha256}" \
  --arg hub_file_verification_sha256 "${hub_file_verification_sha256}" \
  --argjson expected_top_level_files "${expected_file_count}" \
  '{
    schema:$schema,
    created_at:$created_at,
    repo:$repo,
    public:true,
    revision:$revision,
    verified_card_revision:$verified_card_revision,
    anonymous_verification_complete:true,
    expected_top_level_files:$expected_top_level_files,
    missing_top_level_files:0,
    model_card_sha256:$model_card_sha256,
    release_provenance_sha256:$provenance_sha256,
    hub_file_verification_sha256:$hub_file_verification_sha256
  }' \
  >"${root}/results/hf-publication.json"
install -m 0644 "${published_provenance}" \
  "${model_root}/RELEASE_PROVENANCE.json"
touch "${complete_marker}"
log "sealed public revision ${final_revision}"
