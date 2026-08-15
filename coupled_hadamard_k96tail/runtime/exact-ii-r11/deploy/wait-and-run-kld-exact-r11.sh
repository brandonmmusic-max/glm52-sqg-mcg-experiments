#!/usr/bin/env bash
set -euo pipefail

if (($# != 2)); then
  printf 'usage: %s MODEL_DIR RESULTS_DIR\n' "$0" >&2
  exit 2
fi

model_dir=$(realpath "$1")
results_dir=$(realpath -m "$2")
poll_seconds=${GPU_POLL_SECONDS:-15}
max_foreign_mib=${MAX_FOREIGN_GPU_MIB:-2048}
last_report=0

while true; do
  now=$(date +%s)
  blockers=$(
    nvidia-smi --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, -v limit="$max_foreign_mib" '
          {
            gsub(/[[:space:]]/, "", $1)
            gsub(/[[:space:]]/, "", $2)
            if (($2 + 0) > limit) print $1 ":" $2 "MiB"
          }
        '
  )

  if [[ -z "$blockers" ]]; then
    break
  fi

  if ((now - last_report >= 60)); then
    printf '%s waiting for foreign GPU allocations to clear: %s\n' \
      "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<<"$blockers")"
    last_report=$now
  fi
  sleep "$poll_seconds"
done

printf '%s GPUs clear; starting exact Infernal Invocation r11 KLD\n' \
  "$(date --iso-8601=seconds)"

export IMAGE=${IMAGE:-verdictai/glm52-k96-ii-r11:20260815-tpfix}
export RESULT_NAME=${RESULT_NAME:-kld_exact_ii_r11_tp4dcp1.json}
export NAME=${NAME:-glm52-k96-exact-ii-r11-tpfix-kld}

exec "$(dirname "$0")/run-kld-exact-r11.sh" "$model_dir" "$results_dir"
