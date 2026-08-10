#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -le 1 ]] || die "usage: $0 [/absolute/path/to/ABSRMS-r2-candidate]"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALUATION_DIR="$PROJECT_DIR/evaluation"
CANDIDATE="${1:-/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-FRESH-SQG4-ABSRMS-r2}"
CANDIDATE="$(realpath -e -- "$CANDIDATE")"
ARTIFACTS_ROOT="${FRESH_ARTIFACTS_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2}"
ARTIFACTS_ROOT="$(realpath -e -- "$ARTIFACTS_ROOT")"
EVAL_ROOT="${FRESH_EVAL_ROOT:-/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-absrms-r2}"
STAMP="${KLD_STAMP:-fresh-sqg4-absrms-r2}"
[[ "$STAMP" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe KLD_STAMP"

REJECTED_OUT=/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-r1/candidate/fresh-sqg4-r1-candidate-kld-fp8-dcp4
REJECTED_CACHE="$REJECTED_OUT/run1-candidate-runtime-cache"
REJECTED_RECEIPT="$REJECTED_OUT/run1-runtime-cache-receipt.json"
REJECTED_RECORD="$REJECTED_OUT/run1-record.accepted.json"
REJECTED_MANIFEST="$REJECTED_OUT/run-manifest.json"
REJECTED_IMAGE_INSPECT="$REJECTED_OUT/image-inspect.json"
REJECTED_OVERLAY_FILES="$REJECTED_OUT/runtime-overlay-files.sha256"
REJECTED_RECEIPT_SHA256=e71af3a67335996bcdc41aedfdfeb7c21f6e36b09dd95260fbe86a2b3f93612d
REJECTED_RECORD_SHA256=77bcdb94502abbc43b6258ebe586bff8e38ef75c5cff6d238cd055570871f600
REJECTED_MANIFEST_SHA256=d3a499c392a9a9efeb16b3cd47e87b5bfa5920c48202e023553fbde81b270cb2
REJECTED_IMAGE_INSPECT_SHA256=af4d0ff0082e1aff48869504e75d552113ad6a979372cc2ac2d31104b2b9b8b9
REJECTED_OVERLAY_FILES_SHA256=b9e32584b7b9724902e103acc37fbf57d52ad3dfed4a5d269cf85e5b49bcc210
RUNTIME_IMAGE_ID=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82

for rejected_file in "$REJECTED_RECEIPT" "$REJECTED_RECORD" \
  "$REJECTED_MANIFEST" "$REJECTED_IMAGE_INSPECT" "$REJECTED_OVERLAY_FILES"; do
  [[ -f "$rejected_file" && ! -L "$rejected_file" ]] || \
    die "rejected-candidate cache evidence is absent or unsafe: $rejected_file"
done
[[ -d "$REJECTED_CACHE" && ! -L "$REJECTED_CACHE" ]] || \
  die "rejected-candidate compiled-code cache is absent or unsafe"
printf '%s  %s\n' \
  "$REJECTED_RECEIPT_SHA256" "$REJECTED_RECEIPT" \
  "$REJECTED_RECORD_SHA256" "$REJECTED_RECORD" \
  "$REJECTED_MANIFEST_SHA256" "$REJECTED_MANIFEST" \
  "$REJECTED_IMAGE_INSPECT_SHA256" "$REJECTED_IMAGE_INSPECT" \
  "$REJECTED_OVERLAY_FILES_SHA256" "$REJECTED_OVERLAY_FILES" | \
  sha256sum -c - >/dev/null

jq -e --arg cache "$REJECTED_CACHE" '
  .schema == "glm52-runtime-cache-receipt-v1" and
  .path == $cache and .started_empty == true and
  .seeded_snapshot == false and .isolated_per_run == true and
  .fresh_container_python_engine_workers_model_load == true and
  .file_count_after_boot == 13684 and .byte_hashing_skipped == true and
  .cache_semantics ==
    "compiled executable code only; no model, logits, KV, RNG, or process state"
' "$REJECTED_RECEIPT" >/dev/null || die "rejected cache receipt differs"
jq -e \
  --arg cache "$REJECTED_CACHE" \
  --arg receipt "$REJECTED_RECEIPT" \
  --arg receipt_sha256 "$REJECTED_RECEIPT_SHA256" '
  .docker_exit_status == 0 and
  .runtime_cache.path == $cache and
  .runtime_cache.receipt == $receipt and
  .runtime_cache.receipt_sha256 == $receipt_sha256 and
  .runtime_cache.byte_hashing_skipped == true and
  .runtime_dispatch.sqg_dispatch_proved == true and
  .runtime_dispatch.selected_layers == [6,28,52,77] and
  .per_position.positions == 2047 and
  .per_position.independently_validated == true
' "$REJECTED_RECORD" >/dev/null || die "rejected cache accepted record differs"
jq -e --arg image "$RUNTIME_IMAGE_ID" '
  .schema == "glm52-fresh-sqg-candidate-kld-run-v2" and
  .runtime_image_id == $image and .requested_candidate_runs == 5 and
  .regime == {
    kv_cache_dtype:"fp8",rope:"bfloat16",tensor_parallel_size:4,
    decode_context_parallel_size:4,dcp_comm_backend:"a2a",
    dcp_kv_cache_interleave_size:64,context_tokens:2048,
    scored_positions:2047,gpu_memory_utilization:0.90
  }
' "$REJECTED_MANIFEST" >/dev/null || die "rejected cache runtime contract differs"
[[ "$(jq -er '.[0].Id' "$REJECTED_IMAGE_INSPECT")" == "$RUNTIME_IMAGE_ID" ]] || \
  die "rejected cache runtime image differs"
[[ -z "$(sudo -n find "$REJECTED_CACHE" \
  \( -type l -o -type b -o -type c -o -type p -o -type s \) \
  -print -quit)" ]] || die "rejected cache contains a symlink or special file"
[[ "$(sudo -n find "$REJECTED_CACHE" -type f -printf '.' | wc -c)" -eq 13684 ]] || \
  die "rejected cache file census differs"

# Prove cache-code ABI compatibility without hashing the 13,684 cache files:
# the rejected run recorded the same sealed overlay inventory used now.
old_overlay_normalized="$(mktemp)"
current_overlay_normalized="$(mktemp)"
cleanup_temps() {
  rm -f -- "$old_overlay_normalized" "$current_overlay_normalized"
}
trap cleanup_temps EXIT
awk '$2 !~ /SHA256SUMS.runtime-overlay$/ {sub(/^\.\//, "", $2); print $1, $2}' \
  "$REJECTED_OVERLAY_FILES" | sort > "$old_overlay_normalized"
awk '{sub(/^\.\//, "", $2); print $1, $2}' \
  "$EVALUATION_DIR/runtime_overlay/SHA256SUMS.runtime-overlay" | sort \
  > "$current_overlay_normalized"
cmp -s "$old_overlay_normalized" "$current_overlay_normalized" || \
  die "rejected compiled-code cache overlay ABI differs from the current overlay"

CANDIDATE_ROOT="$EVAL_ROOT/candidate"
NATIVE_ROOT="$EVAL_ROOT/native"
PAIRED_ROOT="$EVAL_ROOT/paired"
CANDIDATE_OUT="$CANDIDATE_ROOT/$STAMP-candidate-kld-fp8-dcp4"
NATIVE_OUT="$NATIVE_ROOT/$STAMP-native-mcg-control-kld-fp8-dcp4"
PAIR_JSON="$PAIRED_ROOT/$STAMP-sqg-vs-native-mcg.json"
PAIR_TENSOR="$PAIRED_ROOT/$STAMP-sqg-vs-native-mcg.safetensors"

quarantine_incomplete_boot() {
  local output="$1" run="$2" arm="$3"
  local entries=( "$output/run${run}"* )
  [[ "${#entries[@]}" -gt 0 ]] || return 0
  local quarantine="$output/quarantine-${arm}-run${run}-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$quarantine"
  mv -- "${entries[@]}" "$quarantine/"
  printf 'Quarantined incomplete %s boot %s at %s\n' "$arm" "$run" "$quarantine"
}

resume_from_for_arm() {
  local output="$1" arm="$2"
  if [[ ! -e "$output" ]]; then
    printf '1\n'
    return 0
  fi
  [[ -d "$output" && ! -L "$output" ]] || die "$arm output is unsafe: $output"
  local completed=0 run record
  for run in 1 2 3 4 5; do
    record="$output/run${run}-record.accepted.json"
    if [[ -f "$record" && ! -L "$record" ]]; then
      [[ "$run" -eq $((completed + 1)) ]] || \
        die "$arm accepted boots are not consecutive"
      completed="$run"
    else
      break
    fi
  done
  if [[ "$completed" -eq 5 ]]; then
    [[ -f "$output/summary.json" && ! -L "$output/summary.json" ]] || \
      die "$arm has five accepted boots but no summary; finalize manually"
    printf '6\n'
    return 0
  fi
  if [[ "$completed" -eq 0 ]]; then
    local quarantine="${output}.quarantine-incomplete-run1-$(date -u +%Y%m%dT%H%M%SZ)"
    mv -- "$output" "$quarantine"
    printf 'Quarantined incomplete %s boot 1 at %s\n' "$arm" "$quarantine" >&2
    printf '1\n'
    return 0
  fi
  [[ -f "$output/per-position-evidence.sha256" &&
     "$(wc -l < "$output/per-position-evidence.sha256")" -eq $((completed * 5)) ]] || \
    die "$arm accepted evidence census differs"
  sha256sum -c "$output/per-position-evidence.sha256" >/dev/null
  quarantine_incomplete_boot "$output" $((completed + 1)) "$arm" >&2
  printf '%s\n' $((completed + 1))
}

candidate_resume="$(resume_from_for_arm "$CANDIDATE_OUT" candidate)"
if [[ "$candidate_resume" -le 5 ]]; then
  printf 'Candidate ABSRMS-r2 KLD: starting at fresh boot %s of 5\n' \
    "$candidate_resume"
  RUNS=5 \
  RESUME_FROM_RUN="$candidate_resume" \
  SNAPSHOT_SEED_FROM_RUN1=0 \
  CANDIDATE_BOOT_CACHE_SEED="$REJECTED_CACHE" \
  CANDIDATE_BOOT_CACHE_SEED_RECEIPT="$REJECTED_RECEIPT" \
  CANDIDATE_BOOT_CACHE_SEED_RECORD="$REJECTED_RECORD" \
  DIRECTIONAL_TEST_FAST=1 \
  STAMP="$STAMP" \
  RESULTS_ROOT="$CANDIDATE_ROOT" \
  FRESH_ARTIFACTS_ROOT="$ARTIFACTS_ROOT" \
  RUNTIME_OVERLAY="$EVALUATION_DIR/runtime_overlay" \
  EXTRA_DOCKER_ARGS_FILE="$EVALUATION_DIR/r33_exact_runtime.args" \
    "$EVALUATION_DIR/run_fresh_kld.sh" "$CANDIDATE"
fi

jq -e --arg seed "$REJECTED_CACHE" '
  .schema == "glm52-fresh-sqg-candidate-kld-result-v2" and
  .candidate_result.runs == 5 and
  .runtime_cache_seed.enabled == true and
  .runtime_cache_seed.source == $seed and
  .runtime_cache_seed.independent_snapshot_per_boot == true and
  .runtime_cache_seed.cache_payload_byte_hashing_skipped == true and
  .runtime_cache_seed.fresh_container_python_engine_workers_model_load_every_boot == true and
  (.candidate_result.paired_per_position_outputs | length) == 5 and
  all(.candidate_result.paired_per_position_outputs[];
    .positions == 2047 and .independently_validated == true)
' "$CANDIDATE_OUT/summary.json" >/dev/null || \
  die "candidate arm did not publish five accepted dispatch-proved boots"

# The exact runtime regime already has a five-boot native MCG baseline. For
# the time-priority directional test, stop after the five corrected-candidate
# boots when explicitly requested. The default still produces a newly paired
# native arm.
if [[ "${CANDIDATE_ONLY:-0}" == "1" ]]; then
  jq -n --arg candidate_summary "$CANDIDATE_OUT/summary.json" \
    --arg existing_native_baseline \
      "/home/brandonmusic/klc-linux/release-gg-sparkinfer-20260721/results/20260809T194958Z-kld-fp8-dcp4/summary.json" '
    {
      candidate_summary:$candidate_summary,
      existing_native_baseline:$existing_native_baseline,
      fresh_native_boots_run:0
    }
  '
  exit 0
fi

native_resume="$(resume_from_for_arm "$NATIVE_OUT" native-mcg)"
if [[ "$native_resume" -le 5 ]]; then
  printf 'Native-MCG paired control: starting at fresh boot %s of 5\n' \
    "$native_resume"
  RUNS=5 \
  RESUME_FROM_RUN="$native_resume" \
  SNAPSHOT_SEED_RUNTIME_CACHE=1 \
  NATIVE_BOOT1_CACHE_SEED="$REJECTED_CACHE" \
  DIRECTIONAL_TEST_FAST=1 \
  STAMP="$STAMP" \
  NATIVE_MCG_RESULTS_ROOT="$NATIVE_ROOT" \
    "$EVALUATION_DIR/run_native_mcg_control.sh"
fi

jq -e '
  .schema == "glm52-native-mcg-control-kld-result-v1" and
  .control_result.runs == 5 and
  .runtime_dispatch.native_mcg_dispatch_proved == true and
  .runtime_dispatch.sqg_dispatch_records == 0 and
  (.control_result.paired_per_position_outputs | length) == 5 and
  all(.control_result.paired_per_position_outputs[];
    .positions == 2047 and .independently_validated == true)
' "$NATIVE_OUT/summary.json" >/dev/null || \
  die "native-MCG arm did not publish five accepted dispatch-proved boots"

mkdir -p "$PAIRED_ROOT"
if [[ -f "$PAIR_JSON" && ! -L "$PAIR_JSON" ]]; then
  jq -e --arg candidate "$CANDIDATE_OUT/summary.json" \
    --arg control "$NATIVE_OUT/summary.json" '
    .schema == "glm52-sqg-native-mcg-paired-kld-analysis-v1" and
    .candidate_summary == $candidate and .control_summary == $control and
    .runs_per_arm == 5 and .positions == 2047
  ' "$PAIR_JSON" >/dev/null || die "existing paired analysis differs"
  [[ -f "$PAIR_TENSOR" && ! -L "$PAIR_TENSOR" ]] || \
    die "existing paired analysis lacks its real tensor"
  [[ "$(sha256sum "$PAIR_TENSOR" | awk '{print $1}')" == \
    "$(jq -er '.paired_tensor.sha256' "$PAIR_JSON")" ]] || \
    die "existing paired tensor hash differs"
else
  for partial_pair in "$PAIR_JSON.partial" "$PAIR_TENSOR.partial"; do
    [[ ! -e "$partial_pair" && ! -L "$partial_pair" ]] || \
      mv -- "$partial_pair" "$partial_pair.quarantine-$(date -u +%Y%m%dT%H%M%SZ)"
  done
  if [[ -e "$PAIR_TENSOR" || -L "$PAIR_TENSOR" ]]; then
    mv -- "$PAIR_TENSOR" \
      "$PAIR_TENSOR.quarantine-unpaired-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  python3 "$EVALUATION_DIR/analyze_native_mcg_pair.py" \
    "$CANDIDATE_OUT/summary.json" "$NATIVE_OUT/summary.json" \
    --expected-runs 5 --json-output "$PAIR_JSON" --tensor-output "$PAIR_TENSOR"
fi

jq -n \
  --arg candidate_summary "$CANDIDATE_OUT/summary.json" \
  --arg native_summary "$NATIVE_OUT/summary.json" \
  --arg paired_json "$PAIR_JSON" --arg paired_tensor "$PAIR_TENSOR" '
  {
    candidate_summary:$candidate_summary,
    native_mcg_summary:$native_summary,
    paired_analysis:$paired_json,
    paired_tensor:$paired_tensor
  }
'
