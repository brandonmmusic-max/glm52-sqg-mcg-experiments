#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

[[ $# -eq 1 ]] || die "usage: $0 /absolute/path/to/candidate-output"
OUT="$(realpath -e -- "$1")"
EVALUATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="$EVALUATION_DIR/validate_per_position_kld.py"
OLD_VALIDATOR_SHA256=f50f34df3262e9d4ed3e6181c0225f7a2af4671482ad178e01b17c12492702ae
NEW_VALIDATOR_SHA256=3a3388657a0d6fa72341db9a10c2d59bada1b7be78f8792737a8aaaba1ba7017
EXPECTED_POSITION_SHA256=26f1073e3ac1a28506a41d0fd05f2f540512ff6bbc66b288f9e2ac1573729be1
EXPECTED_MEAN_KLD=0.06169908118680389
TOKEN_SHA256=ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56

raw="$OUT/run1-record.raw.json"
log="$OUT/run1.log"
proof="$OUT/run1-sqg-dispatch-proof.json"
position="$OUT/run1-container-output/run1-position-kld.safetensors"
validation="$OUT/run1-position-kld.validation.json"
cache="$OUT/run1-candidate-runtime-cache"
cache_receipt="$OUT/run1-runtime-cache-receipt.json"
accepted="$OUT/run1-record.accepted.json"
evidence="$OUT/per-position-evidence.sha256"
recovery="$OUT/run1-roundoff-validator-recovery.json"
eval_code="$OUT/eval-code.sha256"

for path in "$raw" "$log" "$proof" "$position" "$cache_receipt" \
  "$eval_code" "$VALIDATOR"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required run-1 evidence missing: $path"
done
[[ -d "$cache" && ! -L "$cache" ]] || die "run-1 runtime cache missing"
[[ ! -e "$accepted" && ! -L "$accepted" ]] || die "run 1 is already accepted"
[[ ! -e "$evidence" && ! -L "$evidence" ]] || die "run-1 evidence already exists"
[[ ! -e "$recovery" && ! -L "$recovery" ]] || die "recovery receipt already exists"
[[ "$(sha256sum "$VALIDATOR" | awk '{print $1}')" == \
  "$NEW_VALIDATOR_SHA256" ]] || die "roundoff-aware validator hash differs"
grep -F "$OLD_VALIDATOR_SHA256  $VALIDATOR" "$eval_code" >/dev/null || \
  die "original run did not bind the strict validator"

jq -e --arg token "$TOKEN_SHA256" \
  --arg sha "$EXPECTED_POSITION_SHA256" \
  --argjson mean "$EXPECTED_MEAN_KLD" '
  .total_positions == 2047 and .elapsed_sec > 0 and
  .token_ids_u32le_sha256 == $token and .mean_kld == $mean and
  .per_position == {
    path:"/results/run1-position-kld.safetensors",positions:2047,
    sha256:$sha,tensor:"kld_ref_to_model"
  }
' "$raw" >/dev/null || die "preserved raw KLD record differs"
record_from_log="$(sed -n 's/^fallback_prefill_kld_done //p' "$log" | tail -n 1)"
[[ -n "$record_from_log" ]] || die "run-1 log has no terminal KLD record"
diff -u <(jq -S . "$raw") <(printf '%s\n' "$record_from_log" | jq -S .) \
  >/dev/null || die "raw record differs from run log"

log_sha="$(sha256sum "$log" | awk '{print $1}')"
jq -e --arg log "$log" --arg sha "$log_sha" '
  .schema == "glm52-sqg-dispatch-proof-v1" and
  .log == $log and .log_sha256 == $sha and
  .required_layers == [6,28,52,77] and
  .every_required_line_observed == true and
  (.records | length) == 4 and all(.records[]; .count == 4)
' "$proof" >/dev/null || die "run-1 SQG dispatch proof differs"
for layer in 6 28 52 77; do
  line="EXL3 SQG layer model.layers.${layer}.mlp.experts: retaining 768 per-projection native tensors"
  [[ "$(grep -Fc "$line" "$log")" -eq 4 ]] || \
    die "run-1 dispatch count differs for layer $layer"
done

jq -e '
  .schema == "glm52-runtime-cache-seed-receipt-v1" and
  .started_empty == false and .seeded_snapshot == true and
  .isolated_per_run == true and
  .fresh_container_python_engine_workers_model_load == true and
  .byte_hashing_skipped == true and
  .source_arm == "rejected_sqg_candidate_compiled_code_cache" and
  .source_file_count == 13684 and
  .destination_file_count_before_boot == 13684 and
  .destination_file_count_after_boot == 13684
' "$cache_receipt" >/dev/null || die "run-1 cache receipt differs"
[[ -z "$(find "$cache" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) \
  -print -quit)" ]] || die "run-1 cache contains a special file"
[[ "$(find "$cache" -type f -printf '.' | wc -c)" -eq 13684 ]] || \
  die "run-1 cache census differs"

validation_tmp="$validation.partial"
recovery_tmp="$recovery.partial"
accepted_tmp="$accepted.partial"
trap 'rm -f -- "$validation_tmp" "$recovery_tmp" "$accepted_tmp"' EXIT
python3 "$VALIDATOR" "$position" \
  --expected-sha256 "$EXPECTED_POSITION_SHA256" \
  --expected-positions 2047 --expected-mean-kld "$EXPECTED_MEAN_KLD" \
  > "$validation_tmp"
jq -e --arg sha "$EXPECTED_POSITION_SHA256" '
  .valid == true and .sha256 == $sha and .positions == 2047 and
  .tensor == "kld_ref_to_model" and
  .negative_roundoff_count == 7 and
  .minimum_kld == -4.6551978272191263e-08 and
  .negative_roundoff_tolerance == 1e-7
' "$validation_tmp" >/dev/null || die "roundoff validation report differs"

raw_sha="$(sha256sum "$raw" | awk '{print $1}')"
proof_sha="$(sha256sum "$proof" | awk '{print $1}')"
cache_sha="$(sha256sum "$cache_receipt" | awk '{print $1}')"
eval_code_sha="$(sha256sum "$eval_code" | awk '{print $1}')"
validation_sha="$(sha256sum "$validation_tmp" | awk '{print $1}')"
jq -n --arg old_sha "$OLD_VALIDATOR_SHA256" \
  --arg new_sha "$NEW_VALIDATOR_SHA256" \
  --arg eval_code "$eval_code" --arg eval_code_sha "$eval_code_sha" \
  --arg position "$position" --arg position_sha "$EXPECTED_POSITION_SHA256" \
  --arg raw "$raw" --arg raw_sha "$raw_sha" \
  --arg log "$log" --arg log_sha "$log_sha" \
  --arg validation "$validation" --arg validation_sha "$validation_sha" '
  {
    schema:"glm52-kld-float32-roundoff-validator-recovery-v1",
    complete:true,boot_rerun:false,kld_tensor_modified:false,
    reason:"seven persisted float32 per-position KL reductions were between zero and -4.66e-8",
    original_validator_sha256:$old_sha,
    recovery_validator_sha256:$new_sha,
    negative_roundoff_tolerance:1e-7,
    original_eval_code:{path:$eval_code,sha256:$eval_code_sha},
    preserved_position:{path:$position,sha256:$position_sha,positions:2047},
    preserved_raw_record:{path:$raw,sha256:$raw_sha,mean_kld:0.06169908118680389},
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
    selected_layers:[6,28,52,77],sqg_dispatch_proved:true
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

mv -f -- "$validation_tmp" "$validation"
mv -- "$recovery_tmp" "$recovery"
mv -- "$accepted_tmp" "$accepted"
accepted_sha="$(sha256sum "$accepted" | awk '{print $1}')"
{
  printf '%s  %s\n' "$EXPECTED_POSITION_SHA256" "$position"
  printf '%s  %s\n' "$validation_sha" "$validation"
  printf '%s  %s\n' "$log_sha" "$log"
  printf '%s  %s\n' "$proof_sha" "$proof"
  printf '%s  %s\n' "$accepted_sha" "$accepted"
} > "$evidence"
sha256sum -c "$evidence" >/dev/null
printf '%s\n' "$accepted"
