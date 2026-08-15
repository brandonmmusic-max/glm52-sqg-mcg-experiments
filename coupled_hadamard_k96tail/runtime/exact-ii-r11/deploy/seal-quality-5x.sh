#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${SERVED_MODEL_NAME:-GLM-5.2-SQG-Coupled-K96Tail}"
image="${IMAGE:-verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix}"
bench_commit=0b4185b5b435e948b199c9077a00b084864aa963
bench_sha256=59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7
bench_archive="${root}/benchmarks/llm_decode_bench.py.gz"
results="${1:-${QUALITY_RUN_DIR:-}}"

log() {
  printf 'GLM quality seal: %s %s\n' "$(date --iso-8601=seconds)" "$*"
}

if [[ -z "${results}" || ! -d "${results}" ]]; then
  log "FATAL: pass an existing qualification run directory"
  exit 2
fi
[[ -f "${bench_archive}" ]] || {
  log "FATAL: exact measured benchmark archive is absent"
  exit 1
}
observed_bench_sha256="$(gzip -dc "${bench_archive}" | sha256sum | awk '{print $1}')"
[[ "${observed_bench_sha256}" == "${bench_sha256}" ]] || {
  log "FATAL: exact measured benchmark hash differs"
  exit 1
}
results="$(realpath "${results}")"
run_id="$(basename "${results}")"
results_root="$(dirname "${results}")"

for required in \
  v1-models.json image-inspect.json container-inspect.json \
  estonia-c1-r5.json estonia-c1-r5.log \
  lavd-c5-r5.json lavd-c5-r5.log server.log; do
  if [[ ! -f "${results}/${required}" ]]; then
    log "FATAL: missing ${results}/${required}"
    exit 1
  fi
done

jq -e '
  .[0].Config.Labels["ai.verdict.infernal-invocation.base"] == "r11" and
  .[0].Config.Labels["ai.verdict.target.topology"] == "tp4-dcp4-mtp3" and
  .[0].Config.Labels["ai.verdict.coupled.tp4-preactivation"] ==
    "all-gather-reassembly-v1" and
  .[0].Config.Labels["ai.verdict.mtp.exl3-prefix"] ==
    "canonical-mtp-block-v1"
' "${results}/image-inspect.json" >/dev/null

jq -e '
  .metadata.mode == "completion_stats" and
  .metadata.version == "0.4.29" and
  .metadata.test_profile == "estonia" and
  .metadata.requested_runs == 5 and
  .metadata.fixed_concurrency == 1 and
  (.runs | length) == 5 and
  all(.runs[]; .ok == true and .correct == true and
    .hit_max_tokens == false)
' "${results}/estonia-c1-r5.json" >/dev/null

jq -e '
  .metadata.mode == "completion_stats" and
  .metadata.version == "0.4.29" and
  .metadata.test_profile == "lavd-test" and
  .metadata.requested_runs == 5 and
  .metadata.fixed_concurrency == 5 and
  (.runs | length) == 5 and
  all(.runs[]; .ok == true and .correct == true and
    .hit_max_tokens == false)
' "${results}/lavd-c5-r5.json" >/dev/null

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

image_id="$(jq -er '.[0].Id' "${results}/image-inspect.json")"
jq -n \
  --arg schema glm52-k96-exact-r11-quality-v1 \
  --arg created_at "$(date --iso-8601=seconds)" \
  --arg image_ref "${image}" \
  --arg image_id "${image_id}" \
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
      fatal_error_audit_matches: 0,
      loaded_and_executed_layers: [range(3; 79)]
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
  server.log server-error-audit.log runtime-evidence/*.json \
  >SHA256SUMS)
ln -sfn "${run_id}" "${results_root}/latest"
touch "${results}/qualification.complete"
log "sealed ${results}/quality-summary.json"
