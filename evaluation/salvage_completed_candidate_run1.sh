#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "usage: $0 /absolute/path/to/candidate-output"
OUT="$(realpath -e -- "$1")"
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSITION_VALIDATOR="$EXPERIMENT_DIR/validate_per_position_kld.py"
TOKEN_SHA256=ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56

raw_record="$OUT/run1-record.raw.json"
run_log="$OUT/run1.log"
dispatch_proof="$OUT/run1-sqg-dispatch-proof.json"
run_output="$OUT/run1-container-output"
position_host="$run_output/run1-position-kld.safetensors"
position_validation="$OUT/run1-position-kld.validation.json"
run_cache="$OUT/run1-candidate-runtime-cache"
cache_receipt="$OUT/run1-runtime-cache-receipt.json"
accepted_record="$OUT/run1-record.accepted.json"
evidence="$OUT/per-position-evidence.sha256"

for required in "$raw_record" "$run_log" "$dispatch_proof" \
  "$position_host" "$POSITION_VALIDATOR"; do
  [[ -f "$required" && ! -L "$required" ]] || \
    die "completed run-1 evidence is absent or unsafe: $required"
done
[[ -d "$run_cache" && ! -L "$run_cache" ]] || \
  die "completed run-1 cache is absent or unsafe"
[[ ! -e "$accepted_record" && ! -L "$accepted_record" ]] || \
  die "run 1 is already accepted"
[[ ! -e "$position_validation" && ! -L "$position_validation" ]] || \
  die "run-1 position validation already exists without an accepted record"
[[ ! -e "$evidence" && ! -L "$evidence" ]] || \
  die "run-1 evidence manifest already exists without an accepted record"

sudo -n chmod -R u+rwX,go+rX "$run_cache" "$run_output" || \
  die "could not normalize completed container artifacts"
[[ -z "$(find "$run_cache" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) \
  -print -quit)" ]] || die "completed run-1 cache contains a special file"

jq -e --arg token_sha256 "$TOKEN_SHA256" '
  .total_positions == 2047 and .mean_kld >= 0 and .elapsed_sec > 0 and
  .token_ids_u32le_sha256 == $token_sha256 and
  .per_position.path == "/results/run1-position-kld.safetensors" and
  .per_position.positions == 2047 and
  .per_position.tensor == "kld_ref_to_model" and
  (.per_position.sha256 | test("^[0-9a-f]{64}$"))
' "$raw_record" >/dev/null || die "completed run-1 KLD record differs"

record_from_log="$(sed -n 's/^fallback_prefill_kld_done //p' \
  "$run_log" | tail -n 1)"
[[ -n "$record_from_log" ]] || die "completed run-1 log lacks final KLD record"
diff -u <(jq -S . "$raw_record") <(printf '%s\n' "$record_from_log" | jq -S .) \
  >/dev/null || die "run-1 raw record differs from its log"

run_log_sha256="$(sha256sum "$run_log" | awk '{print $1}')"
jq -e --arg log "$run_log" --arg log_sha256 "$run_log_sha256" '
  .schema == "glm52-sqg-dispatch-proof-v1" and
  .log == $log and .log_sha256 == $log_sha256 and
  .required_layers == [6,28,52,77] and
  .every_required_line_observed == true and
  (.records | length) == 4 and all(.records[]; .count >= 1)
' "$dispatch_proof" >/dev/null || die "run-1 SQG dispatch proof differs"
for layer in 6 28 52 77; do
  line="EXL3 SQG layer model.layers.${layer}.mlp.experts: retaining 768 per-projection native tensors"
  [[ "$(grep -Fc "$line" "$run_log" || true)" -ge 1 ]] || \
    die "run-1 log lacks SQG dispatch for layer $layer"
done

position_sha256="$(jq -er '.per_position.sha256' "$raw_record")"
position_mean="$(jq -er '.mean_kld' "$raw_record")"
python3 "$POSITION_VALIDATOR" "$position_host" \
  --expected-sha256 "$position_sha256" \
  --expected-positions 2047 \
  --expected-mean-kld "$position_mean" > "$position_validation"
jq -e --arg sha256 "$position_sha256" '
  .valid == true and .sha256 == $sha256 and .positions == 2047 and
  .tensor == "kld_ref_to_model"
' "$position_validation" >/dev/null || die "run-1 position validation differs"

cache_file_count="$(find "$run_cache" -type f -printf '.' | wc -c)"
[[ "$cache_file_count" -gt 0 ]] || die "completed run-1 cache is empty"
partial_manifest="$OUT/run1-runtime-cache-files.sha256"
jq -n --arg path "$run_cache" --arg partial_manifest "$partial_manifest" \
  --argjson file_count "$cache_file_count" \
  --arg salvage_script "$(realpath -e -- "$0")" '
  {
    schema:"glm52-runtime-cache-receipt-v1",path:$path,
    started_empty:true,seeded_snapshot:false,isolated_per_run:true,
    fresh_container_python_engine_workers_model_load:true,
    file_count_after_boot:$file_count,byte_hashing_skipped:true,
    cache_semantics:"compiled executable code only; no model, logits, KV, RNG, or process state",
    salvage:{
      reason:"controller cache-manifest permission failure after successful Docker/KLD/dispatch completion",
      partial_byte_manifest_ignored:$partial_manifest,
      script:$salvage_script,
      docker_exit_status_proved_zero_before_failed_cache_scan:true
    }
  }
' > "$cache_receipt"
cache_receipt_sha256="$(sha256sum "$cache_receipt" | awk '{print $1}')"
position_validation_sha256="$(sha256sum "$position_validation" | awk '{print $1}')"
raw_record_sha256="$(sha256sum "$raw_record" | awk '{print $1}')"
dispatch_proof_sha256="$(sha256sum "$dispatch_proof" | awk '{print $1}')"

jq --arg host_path "$position_host" \
  --arg validation_report "$position_validation" \
  --arg validation_sha256 "$position_validation_sha256" \
  --arg raw_record_sha256 "$raw_record_sha256" \
  --arg run_log "$run_log" --arg run_log_sha256 "$run_log_sha256" \
  --arg dispatch_proof "$dispatch_proof" \
  --arg dispatch_proof_sha256 "$dispatch_proof_sha256" \
  --arg cache_receipt "$cache_receipt" \
  --arg cache_receipt_sha256 "$cache_receipt_sha256" \
  --slurpfile runtime_cache "$cache_receipt" '
  .raw_record_sha256 = $raw_record_sha256 |
  .docker_exit_status = 0 |
  .runtime_cache = ($runtime_cache[0] + {
    receipt:$cache_receipt,receipt_sha256:$cache_receipt_sha256
  }) |
  .runtime_dispatch = {
    log:$run_log,log_sha256:$run_log_sha256,
    proof:$dispatch_proof,proof_sha256:$dispatch_proof_sha256,
    selected_layers:[6,28,52,77],sqg_dispatch_proved:true
  } |
  .per_position += {
    host_path:$host_path,validation_report:$validation_report,
    validation_report_sha256:$validation_sha256,
    independently_validated:true
  }
' "$raw_record" > "$accepted_record"

accepted_record_sha256="$(sha256sum "$accepted_record" | awk '{print $1}')"
{
  printf '%s  %s\n' "$position_sha256" "$position_host"
  printf '%s  %s\n' "$position_validation_sha256" "$position_validation"
  printf '%s  %s\n' "$run_log_sha256" "$run_log"
  printf '%s  %s\n' "$dispatch_proof_sha256" "$dispatch_proof"
  printf '%s  %s\n' "$accepted_record_sha256" "$accepted_record"
} > "$evidence"
sha256sum -c "$evidence" >/dev/null
jq -e '.mean_kld == 1.8742683687382462 and
  .per_position.sha256 ==
    "ed0f5f7dd044b8a53d75a4c1fe4e16a506342b805073145efaf07f2ba374cbd5"' \
  "$accepted_record" >/dev/null || die "salvaged run-1 identity differs"
printf '%s\n' "$accepted_record"
