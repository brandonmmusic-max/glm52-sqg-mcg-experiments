#!/usr/bin/env bash
set -euo pipefail

[[ ${GPU_OVERRIDE:-} =~ ^[0-3]$ ]] || {
  printf 'GPU_OVERRIDE must be one physical GPU index in 0..3\n' >&2
  exit 2
}

# The canonical launcher derives a layer-relative physical GPU.  For bounded
# tail helpers, replace only docker's device=N argument so work can move to a
# board whose primary layer has completed.  Every encoder argument and mounted
# input remains owned by the canonical launcher.
docker() {
  local argument
  local -a forwarded=()
  for argument in "$@"; do
    if [[ $argument =~ ^device=[0-3]$ ]]; then
      argument="device=$GPU_OVERRIDE"
    fi
    forwarded+=("$argument")
  done
  command docker "${forwarded[@]}"
}
export -f docker

exec bash /home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625/scripts/run_coupled_transcode_wave.sh "$@"
