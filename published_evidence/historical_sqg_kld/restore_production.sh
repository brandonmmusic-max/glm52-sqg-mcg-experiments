#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 0 ]] || \
  die "no positional arguments; put CHECK_ONLY=1 before this command"

CONTAINER=glm-r33-fixed
EXPECTED_IMAGE=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
EXPECTED_MODEL=/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED
EXPECTED_SERVED_MODEL=GLM-5.2-EXL3-TR3v4-3.5bpw
BASE_URL=http://127.0.0.1:8000
WAIT_SECONDS="${WAIT_SECONDS:-900}"

[[ "$(docker inspect --format '{{.Image}}' "$CONTAINER")" == \
  "$EXPECTED_IMAGE" ]] || die "production container image identity drifted"
[[ "$(docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' \
  "$CONTAINER")" == "no" ]] || die "production restart policy drifted"
[[ "$(docker inspect --format '{{.HostConfig.NetworkMode}}' \
  "$CONTAINER")" == "host" ]] || die "production network mode drifted"

model_mount_ok="$(docker inspect "$CONTAINER" | jq -r \
  --arg source "$EXPECTED_MODEL" '
    any(.[0].Mounts[]?;
      .Source == $source and .Destination == $source and .RW == false)
  ')"
[[ "$model_mount_ok" == "true" ]] || die "production read-only model mount drifted"

model_env_ok="$(docker inspect "$CONTAINER" | jq -r \
  --arg model "MODEL=$EXPECTED_MODEL" \
  --arg served "SERVED_MODEL_NAME=$EXPECTED_SERVED_MODEL" '
    (.[0].Config.Env | index($model)) != null and
    (.[0].Config.Env | index($served)) != null
  ')"
[[ "$model_env_ok" == "true" ]] || die "production model environment drifted"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  printf 'production_identity_check=passed state=%s\n' \
    "$(docker inspect --format '{{.State.Status}}' "$CONTAINER")"
  exit 0
fi

if [[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER")" != \
  "true" ]]; then
  if ss -ltn | awk 'NR > 1 { print $4 }' | rg -q '(^|:)8000$'; then
    die "TCP port 8000 is already occupied"
  fi
  docker start "$CONTAINER" >/dev/null
fi

deadline=$((SECONDS + WAIT_SECONDS))
while (( SECONDS < deadline )); do
  if [[ "$(docker inspect --format '{{.State.Running}}' \
    "$CONTAINER")" != "true" ]]; then
    docker logs --tail 120 "$CONTAINER" >&2 || true
    die "production container exited during startup"
  fi
  if curl --fail --silent --show-error --max-time 5 \
    "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

curl --fail --silent --show-error --max-time 5 \
  "$BASE_URL/health" >/dev/null || die "production health endpoint timed out"

models_json="$(curl --fail --silent --show-error --max-time 10 \
  "$BASE_URL/v1/models")"
printf '%s\n' "$models_json" | jq -e --arg expected "$EXPECTED_SERVED_MODEL" \
  'any(.data[]?; .id == $expected)' >/dev/null || \
  die "served model identity does not match $EXPECTED_SERVED_MODEL"

large_gpu_count="$({
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
} | awk '$1 >= 80000 { n++ } END { print n + 0 }')"
[[ "$large_gpu_count" -eq 4 ]] || \
  die "expected four GPUs with at least 80 GiB allocated, got $large_gpu_count"

if [[ "${SMOKE_GENERATION:-1}" == "1" ]]; then
  response="$(curl --fail --silent --show-error --max-time 120 \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$EXPECTED_SERVED_MODEL\",\"prompt\":\"1\",\"max_tokens\":1,\"temperature\":0}" \
    "$BASE_URL/v1/completions")"
  printf '%s\n' "$response" | jq -e \
    '(.choices | length) == 1 and (.choices[0].text | type == "string")' \
    >/dev/null || die "production one-token generation gate failed"
fi

printf 'production_restore=passed container=%s image=%s model=%s\n' \
  "$CONTAINER" "$EXPECTED_IMAGE" "$EXPECTED_SERVED_MODEL"
