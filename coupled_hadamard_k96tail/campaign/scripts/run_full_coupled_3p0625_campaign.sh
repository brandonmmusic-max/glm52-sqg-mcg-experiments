#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf '[campaign] ERROR: %s\n' "$*" >&2; exit 2; }
log() { printf '[campaign] %s %s\n' "$(date --iso-8601=seconds)" "$*"; }

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
ACCEPTANCE_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812
SOURCE_SQG_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-W4A8
SEALED_LAYER3_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-3.0625bpw-pilot-l003
INPUT_BASE=/media/brandonmusic/nvme1n1p3/glm52-coupled-wave-inputs
RECIPE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1
PROFILE_ROOT=$RECIPE_ROOT/final_profiles
SCORE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-tail-rate-scores-no-shortcut-v5
ALLOCATION_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-allocations-v1
WORK_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-work-v1
LAYER_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers
MODEL_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail
REPRO_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-reproduction
EVIDENCE_ROOT=$PROJECT_ROOT/evidence/full-coupled-k96tail-no-shortcut
WAVE_ARCHIVE_ROOT=$REPRO_ROOT/wave-archives
CAMPAIGN_LOG=$ACCEPTANCE_ROOT/RESULTS/full_coupled_k96tail_no_shortcut_campaign.log
STOP_FILE=$ACCEPTANCE_ROOT/STOP_FULL_COUPLED_K96TAIL_NO_SHORTCUT
KLD_JSON=$ACCEPTANCE_ROOT/RESULTS/coupled-tail-source-control-tp4dcp1-routes-v2/kld/kld_sm120_tp4dcp1.json
ROUTES_NPZ=$ACCEPTANCE_ROOT/RESULTS/coupled-tail-source-control-tp4dcp1-routes-v2/kld/routed_experts_tp4dcp1.npz
START_WAVE=${START_WAVE:-3}
STOP_WAVE=${STOP_WAVE:-75}
CLEANUP_VALIDATED_WAVES=${CLEANUP_VALIDATED_WAVES:-1}

[[ "$START_WAVE" =~ ^[0-9]+$ && "$STOP_WAVE" =~ ^[0-9]+$ ]] || \
  die "START_WAVE and STOP_WAVE must be integers"
((START_WAVE >= 3 && STOP_WAVE <= 75 && START_WAVE <= STOP_WAVE)) || \
  die "campaign wave starts must lie within 3..75"
(((START_WAVE - 3) % 4 == 0 && (STOP_WAVE - 3) % 4 == 0)) || \
  die "campaign wave starts must be congruent to 3 modulo 4"
[[ "$CLEANUP_VALIDATED_WAVES" == 0 || "$CLEANUP_VALIDATED_WAVES" == 1 ]] || \
  die "CLEANUP_VALIDATED_WAVES must be 0 or 1"

for root in "$PROJECT_ROOT" "$ACCEPTANCE_ROOT" "$SOURCE_SQG_ROOT" \
  "$SEALED_LAYER3_ROOT" "$INPUT_BASE" "$RECIPE_ROOT"; do
  [[ -d "$root" && ! -L "$root" ]] || die "required root is absent or unsafe: $root"
done
mkdir -p "$PROFILE_ROOT" "$SCORE_ROOT" "$ALLOCATION_ROOT" "$WORK_ROOT" \
  "$LAYER_ROOT" "$REPRO_ROOT" "$EVIDENCE_ROOT" "$WAVE_ARCHIVE_ROOT"
[[ -f "$KLD_JSON" && ! -L "$KLD_JSON" ]] || die "sealed source KLD is absent: $KLD_JSON"
[[ -f "$ROUTES_NPZ" && ! -L "$ROUTES_NPZ" ]] || die "sealed source routes are absent: $ROUTES_NPZ"
exec >>"$CAMPAIGN_LOG" 2>&1

seed_layer3() {
  local suffix source target
  for suffix in .safetensors .json .quality.json; do
    source=$SEALED_LAYER3_ROOT/r7-experts-layer-003${suffix}
    target=$LAYER_ROOT/r7-experts-layer-003${suffix}
    [[ -f "$source" && ! -L "$source" ]] || die "sealed layer-3 source is absent: $source"
    if [[ -e "$target" ]]; then
      [[ -f "$target" && ! -L "$target" ]] || die "unsafe layer-3 target: $target"
      [[ $(sha256sum "$source" | cut -d' ' -f1) == \
         $(sha256sum "$target" | cut -d' ' -f1) ]] || \
        die "existing layer-3 target differs from sealed K48 source: $target"
    else
      ln "$source" "$target"
    fi
  done
}

layer3_seal_passes() {
  local manifest=$LAYER_ROOT/r7-experts-layer-003.json
  local quality=$LAYER_ROOT/r7-experts-layer-003.quality.json
  local shard=$LAYER_ROOT/r7-experts-layer-003.safetensors
  [[ -f "$shard" && ! -L "$shard" ]] || return 1
  jq -e \
    '.schema == "glm52-coupled-selected-layer-runtime-v1" and
     .complete == true and .layer == 3 and
     .bit_census == {"k3": 720, "k4": 48, "total": 768} and
     .bits_per_weight == 3.0625 and
     (.manifest_id | type == "string") and (.shard_sha256 | type == "string")' \
    "$manifest" >/dev/null 2>&1 &&
  jq -e \
    '.schema == "glm52-coupled-layer-quality-tails-v1" and
     .complete == true and .layer == 3 and
     .rate_contract == "independent_per_tensor_k3_k4" and
     .holdout_used_for_selection == false and
     (.archive_id | type == "string")' "$quality" >/dev/null 2>&1 &&
  [[ $(sha256sum "$shard" | cut -d' ' -f1) == \
     $(jq -r .shard_sha256 "$manifest") ]]
}

layer3_oracle_passes() {
  local path=$LAYER_ROOT/runtime-oracle-layer-003.json
  [[ -f "$path" && ! -L "$path" ]] || return 1
  sudo -n jq -e \
    '.schema == "glm52-coupled-selected-layer-b12x-oracle-v2" and
     .complete == true and .pass == true and .finite == true and
     .nonzero == true and .layer == 3 and
     .bit_census == {"k3": 720, "k4": 48, "total": 768} and
     .bits_per_weight == 3.0625 and
     .production_endpoint == "route_packed_direct_e4m3_w4a8"' \
    "$path" >/dev/null 2>&1
}

allocation_passes() {
  local layer=$1
  local target=$2
  local padded tag path expected_k3 expected_bpw expected_units
  padded=$(printf '%03d' "$layer")
  tag=$(printf '%03d' "$target")
  path=$ALLOCATION_ROOT/layer_${padded}.tail-v7-k${tag}.allocation.json
  expected_k3=$((768 - target))
  expected_units=$((3 * expected_k3 + 4 * target))
  expected_bpw=$(python3 -c "print($expected_units / 768)")
  jq -e \
    --argjson layer "$layer" --argjson k3 "$expected_k3" \
    --argjson k4 "$target" --argjson units "$expected_units" \
    --argjson bpw "$expected_bpw" \
    '.schema == "glm52-coupled-tail-aware-layer-native-allocation-v7" and
     .complete == true and .layer == $layer and
     .histogram == {"3": $k3, "4": $k4} and
     .bit_units == $units and .bpw == $bpw and
     .final_profile_binding.no_b300_owner_speed_rescue == true and
     (.selected_beta == .final_profile_binding.selected_beta) and
     (.allocation_id | type == "string")' "$path" >/dev/null 2>&1
}

ensure_allocations() {
  local layer=$1 target=96 padded tag output log_path score guarded
  padded=$(printf '%03d' "$layer")
  tag=$(printf '%03d' "$target")
  output=$ALLOCATION_ROOT/layer_${padded}.tail-v7-k${tag}.allocation.json
  if [[ -e "$output" ]]; then
    allocation_passes "$layer" "$target" || \
      die "existing allocation differs: $output"
  else
    log_path=$SCORE_ROOT/logs/finalize-l${padded}-k${tag}.log
    log "finalizing layer $layer allocation K4=$target"
    python3 "$PROJECT_ROOT/scripts/score_coupled_tail_triplet_candidates.py" \
      --output-root "$SCORE_ROOT" --allocation-output "$output" \
      --layer "$layer" --target-k4 "$target" \
      --tail-fraction 0.02 --diagnostic-tail-document-count 40 \
      --tail-weight 1.0 --body-regression-limit 0.01 --finalize \
      >"$log_path"
    allocation_passes "$layer" "$target" || \
      die "new allocation failed validation: $output"
  fi
  score=$SCORE_ROOT/layer_${padded}/tail_triplet_scores.k4-096.json
  guarded=$ALLOCATION_ROOT/layer_${padded}.kld-route-v1-k096.allocation.json
  if [[ ! -e "$guarded" ]]; then
    log "guarding layer $layer K96 allocation with frozen source worst-40 routes"
    python3 "$PROJECT_ROOT/scripts/build_kld_route_guarded_allocation.py" \
      --base-allocation "$output" --score-manifest "$score" \
      --kld-json "$KLD_JSON" --routes-npz "$ROUTES_NPZ" \
      --layer "$layer" --worst-count 40 \
      --total-regression-limit 0.01 --body-regression-limit 0.01 \
      --output "$guarded"
  fi
  guarded_allocation_passes "$layer" || \
    die "KLD-route guarded allocation failed validation: $guarded"
}

guarded_allocation_passes() {
  local layer=$1 padded path
  padded=$(printf '%03d' "$layer")
  path=$ALLOCATION_ROOT/layer_${padded}.kld-route-v1-k096.allocation.json
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-tail-aware-layer-native-allocation-v7" and
     .complete == true and .production_eligible == true and .layer == $layer and
     .histogram == {"3": 672, "4": 96} and
     .bit_units == 2400 and .bpw == 3.125 and
     .final_profile_binding.no_b300_owner_speed_rescue == true and
     (.selected_beta == .final_profile_binding.selected_beta) and
     .kld_route_policy.end_to_end_quality_claim == false and
     (.allocation_id | type == "string")' "$path" >/dev/null 2>&1
}

recipe_passes() {
  local layer=$1 padded result binding selection
  padded=$(printf '%03d' "$layer")
  result=$RECIPE_ROOT/layer_${padded}/NO_SHORTCUT_COUPLED_RECIPE.json
  binding=$PROFILE_ROOT/layer_${padded}/w4a8_native_profile_search/final_profile_binding.json
  selection=$PROFILE_ROOT/layer_${padded}/w4a8_native_profile_search/selection.json
  [[ -f "$result" && ! -L "$result" && -f "$binding" && ! -L "$binding" \
     && -f "$selection" && ! -L "$selection" ]] || return 1
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-updated-qsrt-coupled-no-shortcut-layer-recipe-v1" and
     .complete == true and .layer == $layer and
     .no_b300_owner_speed_rescue == true and
     .no_fleet_beta_shortcut == true and
     .source_is_frozen_sqg_checkpoint == true and
     .official_bf16_weight_shards_read == false' "$result" >/dev/null 2>&1 &&
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-updated-qsrt-coupled-final-profile-binding-v1" and
     .complete == true and .layer == $layer and
     .no_b300_owner_speed_rescue == true and
     ((.bootstrap_reused_byte_for_byte == true) !=
      (.profile_reselected_exactly_once == true))' "$binding" >/dev/null 2>&1 &&
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-updated-qsrt-coupled-profile-selection-v1" and
     .complete == true and .layer == $layer and
     .arithmetic.updated_qsrt_coupled_hadamard == true and
     .no_b300_identity_only_rescue == true and
     .profile_selection_precedes_rate_allocation == true and
     .holdout_used_for_choice == false' "$selection" >/dev/null 2>&1
}

parity_path() {
  printf '%s/layer-%03d-k096-score-encode-parity.json' "$EVIDENCE_ROOT" "$1"
}

parity_passes() {
  local layer=$1 path
  path=$(parity_path "$layer")
  jq -e --argjson layer "$layer" \
    '.complete == true and .all_exact == true and .layer == $layer and
     .numerical_chunk_rows == 256 and
     .allocation.histogram == {"3": 672, "4": 96} and
     .allocation.bpw == 3.125 and (.experts | length == 256)' \
    "$path" >/dev/null 2>&1
}

layer_manifest_passes() {
  local layer=$1 padded manifest quality shard
  padded=$(printf '%03d' "$layer")
  manifest=$LAYER_ROOT/r7-experts-layer-${padded}.json
  quality=$LAYER_ROOT/r7-experts-layer-${padded}.quality.json
  shard=$LAYER_ROOT/r7-experts-layer-${padded}.safetensors
  [[ -f "$shard" && ! -L "$shard" && -f "$quality" && ! -L "$quality" ]] || return 1
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-selected-layer-runtime-v3" and
     .complete == true and .layer == $layer and
     .bit_census == {"k3": 672, "k4": 96, "total": 768} and
     .bits_per_weight == 3.125 and
     .final_profile_binding.no_b300_owner_speed_rescue == true and
     (.manifest_id | type == "string") and (.shard_sha256 | type == "string")' \
    "$manifest" >/dev/null 2>&1 || return 1
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-layer-quality-tails-v2" and
     .complete == true and .layer == $layer and
     .allocation_binding.histogram == {"3": 672, "4": 96} and
     .allocation_binding.bpw == 3.125 and
     (.archive_id | type == "string")' "$quality" >/dev/null 2>&1
}

runtime_oracle_passes() {
  local layer=$1 padded path
  padded=$(printf '%03d' "$layer")
  path=$LAYER_ROOT/runtime-oracle-layer-${padded}.json
  [[ -f "$path" && ! -L "$path" ]] || return 1
  sudo -n jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-selected-layer-b12x-oracle-v2" and
     .complete == true and .pass == true and .finite == true and
     .nonzero == true and .layer == $layer and
     .bit_census == {"k3": 672, "k4": 96, "total": 768} and
     .bits_per_weight == 3.125 and
     .production_endpoint == "route_packed_direct_e4m3_w4a8"' \
    "$path" >/dev/null 2>&1
}

selection_passes() {
  local layer=$1 candidate_root=$2 padded selection score summary
  padded=$(printf '%03d' "$layer")
  selection=$candidate_root/layer_${padded}/coupled_selection_manifest.json
  score=$candidate_root/layer_${padded}/selected_selection_holdout_score.json
  summary=$candidate_root/layer_${padded}/coupled_result_summary.json
  jq -e --argjson layer "$layer" \
    '.schema == "glm52-coupled-mixed-rate-selection-v4" and
     .complete == true and .layer == $layer and .holdout_used == false and
     .allocation_binding.histogram == {"3": 672, "4": 96} and
     .allocation_binding.bpw == 3.125 and
     .final_profile_binding.no_b300_owner_speed_rescue == true and
     (.selected_draws | length == 256)' "$selection" >/dev/null 2>&1 &&
  jq -e --argjson layer "$layer" \
    '.complete == true and .layer == $layer and
     (.roles | keys | sort) == ["holdout", "selection"]' \
    "$score" >/dev/null 2>&1 &&
  jq -e --argjson layer "$layer" \
    '.complete == true and .layer == $layer and
     .schema == "glm52-coupled-mixed-rate-layer-result-v2" and
     .allocation_binding.histogram == {"3": 672, "4": 96}' \
    "$summary" >/dev/null 2>&1
}

ensure_oracle() {
  local layer=$1 gpu=$2
  if runtime_oracle_passes "$layer"; then return; fi
  log "running native runtime oracle for layer $layer on GPU $gpu"
  LAYER_ROOT=$LAYER_ROOT GPU=$gpu \
    "$PROJECT_ROOT/scripts/run_validate_coupled_runtime_layer.sh" "$layer"
  runtime_oracle_passes "$layer" || die "runtime oracle failed for layer $layer"
}

archive_and_cleanup_wave() {
  local wave=$1 candidate_root=$2 input_root=$3 archive
  archive=$WAVE_ARCHIVE_ROOT/${wave}.candidate-metadata.tgz
  if [[ -d "$candidate_root" && ! -L "$candidate_root" && ! -e "$archive" ]]; then
    log "archiving non-payload candidate evidence for $wave"
    tar --exclude='*.safetensors' -czf "$archive" \
      -C "$WORK_ROOT" "$wave"
  fi
  case "$candidate_root" in
    "$WORK_ROOT"/wave-[0-9][0-9][0-9]-[0-9][0-9][0-9]) ;;
    *) die "unsafe candidate cleanup target: $candidate_root" ;;
  esac
  case "$input_root" in
    "$INPUT_BASE"/wave-[0-9][0-9][0-9]-[0-9][0-9][0-9]) ;;
    *) die "unsafe input cleanup target: $input_root" ;;
  esac
  log "removing validated rolling scratch for $wave"
  if [[ -d "$candidate_root" && ! -L "$candidate_root" ]]; then
    rm -rf -- "$candidate_root"
  elif [[ -e "$candidate_root" ]]; then
    die "candidate cleanup target is not a plain directory: $candidate_root"
  fi
  if [[ -d "$input_root" && ! -L "$input_root" ]]; then
    rm -rf -- "$input_root"
  elif [[ -e "$input_root" ]]; then
    die "input cleanup target is not a plain directory: $input_root"
  fi
}

if [[ -e "$STOP_FILE" ]]; then
  die "stop file exists before launch: $STOP_FILE"
fi

seed_layer3
layer3_seal_passes || die "preserved layer-3 K48 seal differs"
if ! layer3_oracle_passes; then
  log "regenerating the missing runtime oracle for preserved K48 layer 3"
  LAYER_ROOT=$LAYER_ROOT GPU=0 \
    "$PROJECT_ROOT/scripts/run_validate_coupled_runtime_layer.sh" 3
fi
layer3_oracle_passes || die "preserved layer-3 runtime oracle failed"

log "starting hybrid K48-layer3 / K96-layers4-78 campaign waves $START_WAVE..$STOP_WAVE"
for start in $(seq "$START_WAVE" 4 "$STOP_WAVE"); do
  [[ ! -e "$STOP_FILE" ]] || die "stop file requested at wave boundary: $STOP_FILE"
  end=$((start + 3))
  wave=$(printf 'wave-%03d-%03d' "$start" "$end")
  input_root=$INPUT_BASE/$wave
  candidate_root=$WORK_ROOT/$wave
  wave_layers_csv=$(seq -s, "$start" "$end")
  IFS=, read -r -a wave_layers <<< "$wave_layers_csv"

  pending_layers=()
  for layer in "${wave_layers[@]}"; do
    if ((layer == 3)); then
      layer3_seal_passes || die "preserved layer-3 seal changed"
      layer3_oracle_passes || die "preserved layer-3 oracle changed"
      continue
    fi
    if layer_manifest_passes "$layer"; then
      parity_passes "$layer" || \
        die "layer $layer is materialized without its earlier parity seal"
      ensure_oracle "$layer" $((layer - start))
    else
      padded=$(printf '%03d' "$layer")
      for suffix in .safetensors .json .quality.json; do
        [[ ! -e "$LAYER_ROOT/r7-experts-layer-${padded}${suffix}" ]] || \
          die "layer $layer has an incomplete or invalid materialization"
      done
      pending_layers+=("$layer")
    fi
  done
  if ((${#pending_layers[@]} == 0)); then
    if ((CLEANUP_VALIDATED_WAVES)); then
      archive_and_cleanup_wave "$wave" "$candidate_root" "$input_root"
    fi
    log "$wave already sealed; skipping"
    continue
  fi
  run_layers_csv=$(IFS=,; printf '%s' "${pending_layers[*]}")

  log "ensuring pinned saved inputs for $wave"
  WAVE_INPUT_ROOT=$input_root \
    "$PROJECT_ROOT/scripts/download_coupled_wave_inputs.sh" "$start" "$end"

  log "sealing per-layer no-shortcut profile and beta recipe for $wave layers=$run_layers_csv"
  RUN_LAYERS=$run_layers_csv WAVE_INPUT_ROOT=$input_root \
    RECIPE_ROOT=$RECIPE_ROOT DETACH=0 \
    "$PROJECT_ROOT/scripts/run_coupled_recipe_wave.sh" "$start" "$end"
  for layer in "${pending_layers[@]}"; do
    recipe_passes "$layer" || die "no-shortcut recipe failed validation: $layer"
  done

  log "scoring layer-native K3/K4 triplets for $wave layers=$run_layers_csv"
  RUN_LAYERS=$run_layers_csv WAVE_INPUT_ROOT=$input_root SCORE_ROOT=$SCORE_ROOT \
    PROFILE_ROOT=$PROFILE_ROOT \
    DETACH=0 "$PROJECT_ROOT/scripts/run_coupled_tail_score_wave.sh" "$start" "$end"
  for layer in "${pending_layers[@]}"; do ensure_allocations "$layer"; done

  log "encoding exact K96 allocations for $wave layers=$run_layers_csv"
  RUN_LAYERS=$run_layers_csv WAVE_INPUT_ROOT=$input_root OUTPUT_ROOT=$candidate_root \
    ALLOCATION_ROOT=$ALLOCATION_ROOT \
    PROFILE_ROOT=$PROFILE_ROOT \
    ALLOCATION_SUFFIX=.kld-route-v1-k096.allocation.json DETACH=0 \
    "$PROJECT_ROOT/scripts/run_coupled_transcode_wave.sh" "$start" "$end"

  for layer in "${pending_layers[@]}"; do
    if parity_passes "$layer"; then continue; fi
    padded=$(printf '%03d' "$layer")
    log "proving scorer/encoder payload parity for layer $layer"
    python3 "$PROJECT_ROOT/scripts/validate_coupled_score_encode_parity.py" \
      --score-root "$SCORE_ROOT" --candidate-root "$candidate_root" \
      --allocation "$ALLOCATION_ROOT/layer_${padded}.kld-route-v1-k096.allocation.json" \
      --layer "$layer" --chunk-rows 256 --output "$(parity_path "$layer")"
    parity_passes "$layer" || die "payload parity failed for layer $layer"
  done

  log "selecting coupled draws on disjoint selection/holdout for $wave"
  RUN_LAYERS=$run_layers_csv WAVE_INPUT_ROOT=$input_root \
    CANDIDATE_ROOT=$candidate_root ALLOCATION_ROOT=$ALLOCATION_ROOT \
    ALLOCATION_SUFFIX=.kld-route-v1-k096.allocation.json DETACH=0 \
    "$PROJECT_ROOT/scripts/run_score_select_coupled_wave.sh" "$start" "$end"
  for layer in "${pending_layers[@]}"; do
    selection_passes "$layer" "$candidate_root" || \
      die "selection/holdout seal failed for layer $layer"
  done

  log "materializing selected runtime shards for $wave"
  RUN_LAYERS=$run_layers_csv CANDIDATE_ROOT=$candidate_root LAYER_ROOT=$LAYER_ROOT \
    "$PROJECT_ROOT/scripts/run_materialize_coupled_wave.sh" "$start" "$end"
  for layer in "${pending_layers[@]}"; do
    layer_manifest_passes "$layer" || die "materialized layer failed seal: $layer"
    ensure_oracle "$layer" $((layer - start))
  done

  if ((CLEANUP_VALIDATED_WAVES)); then
    archive_and_cleanup_wave "$wave" "$candidate_root" "$input_root"
  fi
  log "$wave complete"
done

layer3_seal_passes || die "preserved layer-3 K48 seal is incomplete"
layer3_oracle_passes || die "preserved layer-3 runtime oracle is incomplete"
for layer in $(seq 4 78); do
  recipe_passes "$layer" || die "full no-shortcut recipe set is incomplete at $layer"
  layer_manifest_passes "$layer" || die "full layer set is incomplete at $layer"
  runtime_oracle_passes "$layer" || die "runtime oracle set is incomplete at $layer"
  parity_passes "$layer" || die "parity set is incomplete at $layer"
done

if [[ ! -e "$MODEL_ROOT" ]]; then
  log "assembling full coupled checkpoint at $MODEL_ROOT"
  python3 "$PROJECT_ROOT/scripts/assemble_coupled_checkpoint.py" \
    --source "$SOURCE_SQG_ROOT" --layer-root "$LAYER_ROOT" \
    --output "$MODEL_ROOT" --layers $(seq 3 78)
fi
[[ -f "$MODEL_ROOT/COUPLED_REENCODE_MANIFEST.json" ]] || \
  die "full checkpoint assembly manifest is absent"
log "running full-coupled K96 codec census"
python3 "$ACCEPTANCE_ROOT/scripts/validate_model_codec.py" \
  --model "$MODEL_ROOT" --revision local-no-shortcut-k96tail-coupled-assembly \
  --offline --quick --require-all-coupled \
  --result-json "$ACCEPTANCE_ROOT/RESULTS/full_coupled_k96tail_no_shortcut_model_codec_quick.json"

log "running full untrimmed TP4/PP1/DCP1 KLD and distribution gates"
"$PROJECT_ROOT/scripts/run_finalize_k96tail_model.sh"

log "campaign encode, per-layer validation, assembly, codec census, and KLD complete"
