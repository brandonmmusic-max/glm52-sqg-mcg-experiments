#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${PORT:-8000}"
model="${SERVED_MODEL_NAME:-GLM-5.2-SQG-Coupled-K96Tail}"
python=/home/brandonmusic/klc-env/bin/python
bench_archive="${root}/benchmarks/llm_decode_bench.py.gz"
bench_commit=0b4185b5b435e948b199c9077a00b084864aa963
bench_sha256=59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7
image="${IMAGE:-verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix}"
container="${NAME:-glm52-k96-ii-r11-tp4dcp4mtp3}"
results_root="${QUALITY_RESULTS_DIR:-${root}/results/qualification-5x}"
reuse_results="${QUALITY_REUSE_RESULTS_DIR:-}"
if [[ -n "${reuse_results}" ]]; then
  results="$(readlink -f "${reuse_results}")"
  run_id="${results##*/}"
  [[ -d "${results}" ]] || {
    printf 'GLM quality 5x: reuse directory does not exist: %s\n' "${results}" >&2
    exit 1
  }
else
  run_id="exact-r11-tp4dcp4mtp3-$(date -u +%Y%m%dT%H%M%SZ)"
  results="${results_root}/${run_id}"
fi

log() {
  printf 'GLM quality 5x: %s %s\n' "$(date --iso-8601=seconds)" "$*"
}

bench="$(mktemp --suffix=.llm_decode_bench.measured.py)"
trap 'rm -f "${bench}"' EXIT
gzip -dc "${bench_archive}" >"${bench}"
printf '%s  %s\n' "${bench_sha256}" "${bench}" | sha256sum -c - >/dev/null

mkdir -p "${results}"
if [[ -z "${reuse_results}" ]]; then
  curl --fail --silent --show-error --max-time 30 \
    "http://127.0.0.1:${port}/v1/models" >"${results}/v1-models.json"
  docker image inspect "${image}" >"${results}/image-inspect.json"
  docker inspect "${container}" >"${results}/container-inspect.json"
else
  log "reusing completed task results from ${results}"
  jq -e 'length == 1' "${results}/image-inspect.json" >/dev/null
  jq -e 'length == 1' "${results}/container-inspect.json" >/dev/null
  recorded_container_id="$(jq -r '.[0].Id' "${results}/container-inspect.json")"
  current_container_id="$(docker inspect "${container}" --format '{{.Id}}')"
  [[ "${recorded_container_id}" == "${current_container_id}" ]] || {
    log "FATAL: live container differs from the recorded task-run container"
    exit 1
  }
fi
container_started_at="$(jq -r '.[0].State.StartedAt' "${results}/container-inspect.json")"
container_started_epoch="$(date -d "${container_started_at}" +%s)"

jq -e '
  .[0].Config.Labels["ai.verdict.infernal-invocation.base"] == "r11" and
  .[0].Config.Labels["ai.verdict.target.topology"] == "tp4-dcp4-mtp3" and
  .[0].Config.Labels["ai.verdict.coupled.tp4-preactivation"] ==
    "all-gather-reassembly-v1" and
  .[0].Config.Labels["ai.verdict.mtp.exl3-prefix"] ==
    "canonical-mtp-block-v1"
' "${results}/image-inspect.json" >/dev/null

if [[ -z "${reuse_results}" ]]; then
  log "Estonia: five sequential runs, reasoning_effort=high"
  "${python}" "${bench}" \
    --port "${port}" --model "${model}" \
    --test-profile estonia --profile-concurrency 1 --profile-runs 5 \
    --max-tokens 60000 --display-mode plain \
    --output "${results}/estonia-c1-r5.json" </dev/null \
    >"${results}/estonia-c1-r5.log" 2>&1
fi

jq -e '
  .metadata.mode == "completion_stats" and
  .metadata.version == "0.4.29" and
  .metadata.test_profile == "estonia" and
  .metadata.requested_runs == 5 and
  .metadata.fixed_concurrency == 1 and
  (.runs | length) == 5 and
  all(.runs[]; .ok == true)
' "${results}/estonia-c1-r5.json" >/dev/null

if [[ -z "${reuse_results}" ]]; then
  log "LAVD: five concurrent runs, reasoning_effort=high"
  "${python}" "${bench}" \
    --port "${port}" --model "${model}" \
    --test-profile lavd --profile-concurrency 5 --profile-runs 5 \
    --max-tokens 80000 --display-mode plain \
    --output "${results}/lavd-c5-r5.json" </dev/null \
    >"${results}/lavd-c5-r5.log" 2>&1
fi

jq -e '
  .metadata.mode == "completion_stats" and
  .metadata.version == "0.4.29" and
  .metadata.test_profile == "lavd-test" and
  .metadata.requested_runs == 5 and
  .metadata.fixed_concurrency == 5 and
  (.runs | length) == 5 and
  all(.runs[]; .ok == true)
' "${results}/lavd-c5-r5.json" >/dev/null

curl --fail --silent --show-error --max-time 30 \
  "http://127.0.0.1:${port}/metrics" >"${results}/runtime-metrics.final.prom"
draft_sequences="$(awk '/^vllm:spec_decode_num_drafts_total\{/ {print $2; exit}' \
  "${results}/runtime-metrics.final.prom")"
draft_tokens="$(awk '/^vllm:spec_decode_num_draft_tokens_total\{/ {print $2; exit}' \
  "${results}/runtime-metrics.final.prom")"
accepted_tokens="$(awk '/^vllm:spec_decode_num_accepted_tokens_total\{/ {print $2; exit}' \
  "${results}/runtime-metrics.final.prom")"
accepted_position_0="$(awk '/^vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="0"/ {print $2; exit}' \
  "${results}/runtime-metrics.final.prom")"
accepted_position_1="$(awk '/^vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="1"/ {print $2; exit}' \
  "${results}/runtime-metrics.final.prom")"
accepted_position_2="$(awk '/^vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="2"/ {print $2; exit}' \
  "${results}/runtime-metrics.final.prom")"
generation_tokens="$(awk '/^vllm:generation_tokens_total\{/ {print $2; exit}' \
  "${results}/runtime-metrics.final.prom")"
for metric in \
  "${draft_sequences}" "${draft_tokens}" "${accepted_tokens}" \
  "${accepted_position_0}" "${accepted_position_1}" \
  "${accepted_position_2}" "${generation_tokens}"; do
  [[ "${metric}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    log "FATAL: final MTP metric is absent or nonnumeric"
    exit 1
  }
done
jq -n \
  --arg schema glm52-k96-exact-r11-mtp3-metrics-v1 \
  --arg created_at "$(date --iso-8601=seconds)" \
  --arg image_id "$(docker image inspect "${image}" --format '{{.Id}}')" \
  --arg container_id "$(docker inspect "${container}" --format '{{.Id}}')" \
  --argjson draft_sequences "${draft_sequences}" \
  --argjson draft_tokens "${draft_tokens}" \
  --argjson accepted_tokens "${accepted_tokens}" \
  --argjson accepted_position_0 "${accepted_position_0}" \
  --argjson accepted_position_1 "${accepted_position_1}" \
  --argjson accepted_position_2 "${accepted_position_2}" \
  --argjson generation_tokens "${generation_tokens}" \
  '{
    schema: $schema,
    created_at: $created_at,
    image_id: $image_id,
    container_id: $container_id,
    topology: "TP4/DCP4/MTP3",
    speculative_depth: 3,
    draft_sequences: $draft_sequences,
    draft_tokens: $draft_tokens,
    accepted_tokens: $accepted_tokens,
    aggregate_acceptance_rate: ($accepted_tokens / $draft_tokens),
    accepted_tokens_per_position: [
      $accepted_position_0,
      $accepted_position_1,
      $accepted_position_2
    ],
    acceptance_rate_per_position: [
      ($accepted_position_0 / $draft_sequences),
      ($accepted_position_1 / $draft_sequences),
      ($accepted_position_2 / $draft_sequences)
    ],
    generation_tokens: $generation_tokens
  }' >"${results}/runtime-mtp-summary.json"

log "sealing exact-r11 server log and four native-SQG rank receipts"
docker logs "${container}" >"${results}/server.log" 2>&1
mkdir -p "${results}/runtime-evidence"
evidence_source="${root}/results/runtime-evidence"
while IFS= read -r evidence; do
  cp "${evidence}" "${results}/runtime-evidence/"
done < <(find "${evidence_source}" -maxdepth 1 -type f \
  -name 'pp-*-pid-*.json' -newermt "@${container_started_epoch}" -print | sort)

rank_receipts=$(find "${results}/runtime-evidence" -maxdepth 1 -type f \
  -name 'pp-*-pid-*.json' | wc -l)
if [[ "${rank_receipts}" -ne 4 ]]; then
  log "FATAL: expected four native-SQG rank receipts, found ${rank_receipts}"
  exit 1
fi
while IFS= read -r evidence; do
  jq -e '
    .schema == "glm52-native-sqg-w4a8-rank-evidence-v1" and
    .complete == true and
    .tensor_parallel_size == 4 and
    .activation_endpoint == "full-w4a8" and
    .activation_endpoint_scope == "routed_experts" and
    .allow_a16_fallback == false and
    # The final MTP3 loader executes preserved source-SQG MTP layer 78 in
    # addition to coupled target layers 3--77.
    .loaded_layers == [range(3; 79)] and
    .executed_layers == [range(3; 79)]
  ' "${evidence}" >/dev/null
done < <(find "${results}/runtime-evidence" -maxdepth 1 -type f \
  -name 'pp-*-pid-*.json' -print | sort)

if rg -i \
    'CUDA error|NCCL[^[:cntrl:]]*(error|failed)|Xid|out of memory|Traceback' \
    "${results}/server.log" >"${results}/server-error-audit.log"; then
  log "FATAL: server error audit matched fatal runtime text"
  exit 1
fi

jq -n \
  --arg schema glm52-k96-exact-r11-quality-v1 \
  --arg created_at "$(date --iso-8601=seconds)" \
  --arg image_ref "${image}" \
  --arg image_id "$(docker image inspect "${image}" --format '{{.Id}}')" \
  --arg model "${model}" \
  --arg benchmark_commit "${bench_commit}" \
  --arg benchmark_sha256 "${bench_sha256}" \
  --argjson native_rank_receipts "${rank_receipts}" \
  --slurpfile estonia "${results}/estonia-c1-r5.json" \
  --slurpfile lavd "${results}/lavd-c5-r5.json" \
  '{
    schema: $schema,
    created_at: $created_at,
    runtime: {
      image_ref: $image_ref,
      image_id: $image_id,
      infernal_invocation_base: "r11",
      topology: "TP4/DCP4/MTP3",
      kv_cache_dtype: "nvfp4_ds_mla"
    },
    model: $model,
    benchmark: {
      repository: "local-inference-lab/llm-inference-bench",
      version: "0.4.29",
      commit: $benchmark_commit,
      script_sha256: $benchmark_sha256
    },
    native_sqg: {
      complete_rank_receipts: $native_rank_receipts,
      expected_rank_receipts: 4,
      fatal_error_audit_matches: 0
    },
    estonia: {
      concurrency: 1,
      requested_runs: 5,
      summary: $estonia[0].selected_summary,
      runs: [$estonia[0].runs[] | {
        run_index, ok, correct, score_label, completion_tokens,
        hit_max_tokens, estimated_tokens
      }]
    },
    lavd: {
      concurrency: 5,
      requested_runs: 5,
      summary: $lavd[0].selected_summary,
      runs: [$lavd[0].runs[] | {
        run_index, ok, correct, score_label, score_detail, parsed_answer,
        completion_tokens, hit_max_tokens, estimated_tokens
      }]
    }
  }' >"${results}/quality-summary.json"

(cd "${results}" && sha256sum \
  v1-models.json image-inspect.json container-inspect.json \
  estonia-c1-r5.json estonia-c1-r5.log \
  lavd-c5-r5.json lavd-c5-r5.log quality-summary.json \
  runtime-metrics.final.prom runtime-mtp-summary.json \
  server.log server-error-audit.log runtime-evidence/*.json \
  >SHA256SUMS)
ln -sfn "${run_id}" "${results_root}/latest"
touch "${results}/qualification.complete"
log "sealed ${results}/quality-summary.json"
