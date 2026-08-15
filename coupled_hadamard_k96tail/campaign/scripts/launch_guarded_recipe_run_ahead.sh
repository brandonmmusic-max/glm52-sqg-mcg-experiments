#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 4 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ \
  && $3 =~ ^[0-9]+$ && $4 =~ ^[0-9]+$ ]] || \
  die "usage: $0 CURRENT_START CURRENT_END NEXT_START NEXT_END"

current_start=$1
current_end=$2
next_start=$3
next_end=$4
((current_end - current_start == 3 && next_end - next_start == 3 \
  && next_start == current_start + 4 && next_end <= 78)) || \
  die "the arguments must name consecutive four-layer waves"

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
CAMPAIGN_UNIT=${CAMPAIGN_UNIT:-glm52-full-coupled-k96tail-no-shortcut-goal019ffa7c.service}
INPUT_BASE=${INPUT_BASE:-/media/brandonmusic/nvme1n1p3/glm52-coupled-wave-inputs}
RECIPE_ROOT=${RECIPE_ROOT:-/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1}
RECIPE_LAUNCHER=$PROJECT_ROOT/scripts/run_coupled_recipe_wave.sh
REMOTE_FLEET_MARKER=${REMOTE_FLEET_MARKER:-/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/REMOTE_FLEET_OWNS_LAYERS_051_077}

# The distributed K96 campaign assigns layers 51-77 to rented nodes.  Leave
# this launcher harmless when an older local orchestrator retries its planned
# run-ahead, so it cannot duplicate or contaminate those remote-owned layers.
if [[ -e "$REMOTE_FLEET_MARKER" ]]; then
  printf 'local recipe run-ahead disabled by remote-fleet marker: %s\n' \
    "$REMOTE_FLEET_MARKER"
  exit 0
fi
current_wave=$(printf 'wave-%03d-%03d' "$current_start" "$current_end")
next_wave=$(printf 'wave-%03d-%03d' "$next_start" "$next_end")
next_input=$INPUT_BASE/$next_wave

[[ -x "$RECIPE_LAUNCHER" ]] || die "recipe launcher is absent: $RECIPE_LAUNCHER"
for required in \
  "$next_input/derived/wave_preflights/$next_wave/preflight.json" \
  "$next_input/derived/wave_preflights/$next_wave/bit-contract.json" \
  "$next_input/derived/wave_preflights/$next_wave/source-seal.json" \
  "$next_input/derived/wave_preflights/$next_wave/wave-bf16-shard-manifest.json" \
  "$next_input/capture_view/capture_manifest.json"; do
  [[ -f "$required" && ! -L "$required" ]] || \
    die "next-wave saved input is absent: $required"
done
for layer in $(seq "$next_start" "$next_end"); do
  padded=$(printf '%03d' "$layer")
  [[ -f "$next_input/capture_view/layer_${padded}/hidden.bf16.bin" ]] || \
    die "next-wave activation capture is absent for layer $layer"
done

main_pid=$(systemctl --user show "$CAMPAIGN_UNIT" -p MainPID --value)
[[ "$main_pid" =~ ^[1-9][0-9]*$ ]] || die "campaign unit has no live main process"
systemctl --user is-active --quiet "$CAMPAIGN_UNIT" || \
  die "campaign unit is not active: $CAMPAIGN_UNIT"
pgrep -P "$main_pid" -f "run_coupled_recipe_wave.sh $current_start $current_end" \
  >/dev/null || die "campaign is not in the expected recipe stage: $current_wave"

watcher='set -euo pipefail
current_layer=$1
next_layer=$2
main_pid=$3
current_start=$4
current_end=$5
current_wave=$6
next_start=$7
next_end=$8
next_input=$9
recipe_root=${10}
recipe_launcher=${11}
current_padded=$(printf "%03d" "$current_layer")
current_seal="$recipe_root/layer_${current_padded}/NO_SHORTCUT_COUPLED_RECIPE.json"
# A freshly launched controller can exist briefly before Docker publishes the
# per-layer container name. Wait for that exact container unless the durable
# recipe seal proves the controller is intentionally skipping this layer.
while :; do
  names=$(docker ps --format "{{.Names}}")
  if grep -Fq "recipe-${current_wave}-l${current_layer}-" <<<"$names"; then
    break
  fi
  if [[ -f "$current_seal" ]]; then
    break
  fi
  pgrep -P "$main_pid" -f "run_coupled_recipe_wave.sh $current_start $current_end" \
    >/dev/null || exit 0
  sleep 0.2
done
while :; do
  names=$(docker ps --format "{{.Names}}")
  if ! grep -Fq "recipe-${current_wave}-l${current_layer}-" <<<"$names"; then
    break
  fi
  sleep 0.2
done
pgrep -P "$main_pid" -f "run_coupled_recipe_wave.sh $current_start $current_end" \
  >/dev/null || exit 0
names=$(docker ps --format "{{.Names}}")
grep -Fq "recipe-${current_wave}-" <<<"$names" || exit 0
if grep -Fq "score-${current_wave}-" <<<"$names"; then exit 0; fi
exec /usr/bin/env RUN_LAYERS="$next_layer" DETACH=0 \
  WAVE_INPUT_ROOT="$next_input" "$recipe_launcher" "$next_start" "$next_end"'

units=()
for offset in 0 1 2 3; do
  current_layer=$((current_start + offset))
  next_layer=$((next_start + offset))
  unit=$(printf 'glm52-recipe-runahead-wave-%03d-%03d-l%d.service' \
    "$next_start" "$next_end" "$next_layer")
  units+=("$unit")
  systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
  systemd-run --user --unit="$unit" \
    --description="Guarded recipe run-ahead for layer $next_layer" \
    --expand-environment=no \
    --property=KillMode=control-group --property=TimeoutStopSec=10 \
    /usr/bin/bash -c "$watcher" _ "$current_layer" "$next_layer" \
      "$main_pid" "$current_start" "$current_end" "$current_wave" \
      "$next_start" "$next_end" "$next_input" "$RECIPE_ROOT" \
      "$RECIPE_LAUNCHER" \
    >/dev/null
done

guard='set -euo pipefail
main_pid=$1
current_start=$2
current_end=$3
current_wave=$4
campaign_unit=$5
shift 5
misses=0
while systemctl --user is-active --quiet "$campaign_unit"; do
  names=$(docker ps --format "{{.Names}}")
  if grep -Fq "score-${current_wave}-" <<<"$names"; then
    break
  fi
  if pgrep -P "$main_pid" \
    -f "run_coupled_recipe_wave.sh $current_start $current_end" >/dev/null; then
    misses=0
  else
    misses=$((misses + 1))
    # A Bash child can disappear from pgrep for one sampling interval while
    # the wrapper reaps another worker. Require five seconds of sustained
    # absence before treating the recipe stage as complete.
    if ((misses >= 25)); then
      break
    fi
  fi
  sleep 0.2
done
systemctl --user stop "$@" || true'
guard_unit=$(printf 'glm52-recipe-runahead-guard-wave-%03d-%03d.service' \
  "$next_start" "$next_end")
systemctl --user reset-failed "$guard_unit" >/dev/null 2>&1 || true
systemd-run --user --unit="$guard_unit" \
  --description="Stop recipe run-ahead before $current_wave scoring" \
  --expand-environment=no \
  --property=KillMode=control-group --property=TimeoutStopSec=15 \
  /usr/bin/bash -c "$guard" _ "$main_pid" "$current_start" "$current_end" \
    "$current_wave" "$CAMPAIGN_UNIT" "${units[@]}" >/dev/null

printf 'guarded recipe run-ahead armed: current=%s next=%s units=%s\n' \
  "$current_wave" "$next_wave" "${units[*]}"
