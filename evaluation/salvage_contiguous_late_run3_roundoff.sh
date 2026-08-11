#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

[[ $# -eq 1 ]] || die "usage: $0 /absolute/path/to/candidate-output"
OUT="$(realpath -e -- "$1")"
EVALUATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="$EVALUATION_DIR/validate_per_position_kld.py"
ORIGINAL_VALIDATOR_SHA256=3a3388657a0d6fa72341db9a10c2d59bada1b7be78f8792737a8aaaba1ba7017
EXPECTED_POSITION_SHA256=4ec9a3457dc13ff90d0115b0cb8b0170e6f0167f5860fc91c50f3d68fb7f7236
EXPECTED_MEAN_KLD=0.06256102924107773
TOKEN_SHA256=ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56
EXPECTED_MINIMUM=-1.1123514553901259e-07
EXPECTED_NEGATIVE_COUNT=8

raw="$OUT/run3-record.raw.json"
log="$OUT/run3.log"
proof="$OUT/run3-sqg-dispatch-proof.json"
position="$OUT/run3-container-output/run3-position-kld.safetensors"
validation="$OUT/run3-position-kld.validation.json"
cache="$OUT/run3-candidate-runtime-cache"
cache_receipt="$OUT/run3-runtime-cache-receipt.json"
accepted="$OUT/run3-record.accepted.json"
evidence="$OUT/per-position-evidence.sha256"
recovery="$OUT/run3-roundoff-validator-recovery.json"
eval_code="$OUT/eval-code.sha256"

for path in "$raw" "$log" "$proof" "$position" "$cache_receipt" \
  "$eval_code" "$VALIDATOR" "$evidence"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required run-3 evidence missing: $path"
done
[[ -d "$cache" && ! -L "$cache" ]] || die "run-3 runtime cache missing"
[[ ! -e "$accepted" && ! -L "$accepted" ]] || die "run 3 is already accepted"
[[ ! -s "$validation" ]] || die "failed run-3 validation output is unexpectedly nonempty"
[[ ! -e "$recovery" && ! -L "$recovery" ]] || die "recovery receipt already exists"
[[ "$(wc -l < "$evidence")" -eq 10 ]] || \
  die "exactly two earlier accepted runs are required"
sha256sum -c "$evidence" >/dev/null
grep -F "$ORIGINAL_VALIDATOR_SHA256  $VALIDATOR" "$eval_code" >/dev/null || \
  die "original run did not bind the prior validator"
RECOVERY_VALIDATOR_SHA256="$(sha256sum "$VALIDATOR" | awk '{print $1}')"
[[ "$RECOVERY_VALIDATOR_SHA256" != "$ORIGINAL_VALIDATOR_SHA256" ]] || \
  die "roundoff validator was not revised"

jq -e --arg token "$TOKEN_SHA256" --arg sha "$EXPECTED_POSITION_SHA256" \
  --argjson mean "$EXPECTED_MEAN_KLD" '
  .total_positions == 2047 and .elapsed_sec > 0 and
  .token_ids_u32le_sha256 == $token and .mean_kld == $mean and
  .per_position == {
    path:"/results/run3-position-kld.safetensors",positions:2047,
    sha256:$sha,tensor:"kld_ref_to_model"
  }
' "$raw" >/dev/null || die "preserved raw KLD record differs"
record_from_log="$(sed -n 's/^fallback_prefill_kld_done //p' "$log" | tail -n 1)"
[[ -n "$record_from_log" ]] || die "run-3 log has no terminal KLD record"
diff -u <(jq -S . "$raw") <(printf '%s\n' "$record_from_log" | jq -S .) \
  >/dev/null || die "raw record differs from run log"

log_sha="$(sha256sum "$log" | awk '{print $1}')"
jq -e --arg log "$log" --arg sha "$log_sha" '
  .schema == "glm52-sqg-dispatch-proof-v1" and
  .log == $log and .log_sha256 == $sha and
  .required_layers == [74,75,76,77] and
  .every_required_line_observed == true and
  (.records | length) == 4 and all(.records[]; .count == 4)
' "$proof" >/dev/null || die "run-3 SQG dispatch proof differs"
for layer in 74 75 76 77; do
  line="EXL3 SQG layer model.layers.${layer}.mlp.experts: retaining 768 per-projection native tensors"
  [[ "$(grep -Fc "$line" "$log")" -eq 4 ]] || \
    die "run-3 dispatch count differs for layer $layer"
done

jq -e '
  .schema == "glm52-runtime-cache-seed-receipt-v1" and
  .started_empty == false and .seeded_snapshot == true and
  .isolated_per_run == true and
  .fresh_container_python_engine_workers_model_load == true and
  .byte_hashing_skipped == true and
  .source_arm == "rejected_sqg_candidate_compiled_code_cache" and
  .source_file_count > 0 and
  .destination_file_count_before_boot == .source_file_count and
  .destination_file_count_after_boot >= .destination_file_count_before_boot
' "$cache_receipt" >/dev/null || die "run-3 cache receipt differs"
cache_count="$(find "$cache" -type f -printf '.' | wc -c)"
[[ "$cache_count" -eq "$(jq -er '.destination_file_count_after_boot' "$cache_receipt")" ]] || \
  die "run-3 cache census differs"
[[ -z "$(find "$cache" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) \
  -print -quit)" ]] || die "run-3 cache contains a special file"

validation_tmp="$validation.partial"
recovery_tmp="$recovery.partial"
accepted_tmp="$accepted.partial"
evidence_tmp="$evidence.partial"
trap 'rm -f -- "$validation_tmp" "$recovery_tmp" "$accepted_tmp" "$evidence_tmp"' EXIT
python3 "$VALIDATOR" "$position" \
  --expected-sha256 "$EXPECTED_POSITION_SHA256" \
  --expected-positions 2047 --expected-mean-kld "$EXPECTED_MEAN_KLD" \
  > "$validation_tmp"
jq -e --arg sha "$EXPECTED_POSITION_SHA256" \
  --argjson count "$EXPECTED_NEGATIVE_COUNT" \
  --argjson minimum "$EXPECTED_MINIMUM" '
  .valid == true and .sha256 == $sha and .positions == 2047 and
  .tensor == "kld_ref_to_model" and
  .negative_roundoff_count == $count and .minimum_kld == $minimum and
  .negative_roundoff_tolerance == 2.384185791015625e-7
' "$validation_tmp" >/dev/null || die "roundoff validation report differs"

raw_sha="$(sha256sum "$raw" | awk '{print $1}')"
proof_sha="$(sha256sum "$proof" | awk '{print $1}')"
cache_sha="$(sha256sum "$cache_receipt" | awk '{print $1}')"
eval_code_sha="$(sha256sum "$eval_code" | awk '{print $1}')"
validation_sha="$(sha256sum "$validation_tmp" | awk '{print $1}')"
jq -n --arg old_sha "$ORIGINAL_VALIDATOR_SHA256" \
  --arg new_sha "$RECOVERY_VALIDATOR_SHA256" \
  --arg eval_code "$eval_code" --arg eval_code_sha "$eval_code_sha" \
  --arg position "$position" --arg position_sha "$EXPECTED_POSITION_SHA256" \
  --arg raw "$raw" --arg raw_sha "$raw_sha" \
  --arg log "$log" --arg log_sha "$log_sha" \
  --arg validation "$validation" --arg validation_sha "$validation_sha" '
  {
    schema:"glm52-kld-float32-roundoff-validator-recovery-v2",
    complete:true,boot_rerun:false,kld_tensor_modified:false,
    reason:"one persisted float32 per-position KL reduction was -1.1124e-7, below the old decimal -1e-7 floor but within two float32 epsilons",
    original_validator_sha256:$old_sha,
    recovery_validator_sha256:$new_sha,
    negative_roundoff_tolerance:2.384185791015625e-7,
    original_eval_code:{path:$eval_code,sha256:$eval_code_sha},
    preserved_position:{path:$position,sha256:$position_sha,positions:2047},
    preserved_raw_record:{path:$raw,sha256:$raw_sha,mean_kld:0.06256102924107773},
    preserved_log:{path:$log,sha256:$log_sha},
    recovery_validation:{path:$validation,sha256:$validation_sha}
  }
' > "$recovery_tmp"
recovery_sha="$(sha256sum "$recovery_tmp" | awk '{print $1}')"

jq --arg host "$position" --arg validation "$validation" \
  --arg validation_sha "$validation_sha" --arg raw_sha "$raw_sha" \
  --arg cache_receipt "$cache_receipt" --arg cache_sha "$cache_sha" \
  --arg log "$log" --arg log_sha "$log_sha" \
  --arg proof "$proof" --arg proof_sha "$proof_sha" \
  --arg recovery "$recovery" --arg recovery_sha "$recovery_sha" \
  --slurpfile runtime_cache "$cache_receipt" '
  .raw_record_sha256 = $raw_sha |
  .docker_exit_status = 0 |
  .runtime_cache = ($runtime_cache[0] + {
    receipt:$cache_receipt,receipt_sha256:$cache_sha
  }) |
  .runtime_dispatch = {
    log:$log,log_sha256:$log_sha,proof:$proof,proof_sha256:$proof_sha,
    selected_layers:[74,75,76,77],sqg_dispatch_proved:true
  } |
  .per_position += {
    host_path:$host,validation_report:$validation,
    validation_report_sha256:$validation_sha,independently_validated:true
  } |
  .validation_recovery = {
    reason:"bounded float32 per-position KL cancellation roundoff",
    receipt:$recovery,receipt_sha256:$recovery_sha,
    boot_rerun:false,tensor_modified:false
  }
' "$raw" > "$accepted_tmp"

mv -- "$validation_tmp" "$validation"
mv -- "$recovery_tmp" "$recovery"
mv -- "$accepted_tmp" "$accepted"
accepted_sha="$(sha256sum "$accepted" | awk '{print $1}')"
cp -- "$evidence" "$evidence_tmp"
{
  printf '%s  %s\n' "$EXPECTED_POSITION_SHA256" "$position"
  printf '%s  %s\n' "$validation_sha" "$validation"
  printf '%s  %s\n' "$log_sha" "$log"
  printf '%s  %s\n' "$proof_sha" "$proof"
  printf '%s  %s\n' "$accepted_sha" "$accepted"
} >> "$evidence_tmp"
[[ "$(wc -l < "$evidence_tmp")" -eq 15 ]] || die "recovered evidence census differs"
sha256sum -c "$evidence_tmp" >/dev/null
mv -f -- "$evidence_tmp" "$evidence"
printf '%s\n' "$accepted"
