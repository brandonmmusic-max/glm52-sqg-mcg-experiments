#!/usr/bin/env bash
set -euo pipefail

[[ ${GPU_OVERRIDE:-} =~ ^[0-3]$ ]] || {
  printf 'GPU_OVERRIDE must be one physical GPU index in 0..3\n' >&2
  exit 2
}

# Redirect only docker's physical device argument.  The canonical recipe
# launcher continues to own every input, contract, and recipe argument.
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

exec bash /home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625/scripts/run_coupled_recipe_wave.sh "$@"
