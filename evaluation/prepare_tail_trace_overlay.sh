#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

[[ $# -eq 1 ]] || die "usage: $0 /absolute/new-overlay-directory"
[[ "$1" = /* ]] || die "overlay destination must be absolute"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE="$SCRIPT_DIR/runtime_overlay"
TRACE_MODEL="$SCRIPT_DIR/trace_overrides/deepseek_v2.py"
DESTINATION="$1"

[[ -d "$SOURCE" && -f "$TRACE_MODEL" ]] || die "trace overlay source is absent"
if [[ -e "$DESTINATION" ]]; then
  [[ -d "$DESTINATION" && -z "$(find "$DESTINATION" -mindepth 1 -print -quit)" ]] || \
    die "destination must be absent or empty"
else
  mkdir -p "$DESTINATION"
fi

rsync -a "$SOURCE/" "$DESTINATION/"
cp "$TRACE_MODEL" \
  "$DESTINATION/_overrides/vllm/model_executor/models/deepseek_v2.py"
(
  cd "$DESTINATION"
  find . -type f ! -name SHA256SUMS.runtime-overlay -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS.runtime-overlay
  sha256sum -c SHA256SUMS.runtime-overlay >/dev/null
)
printf 'Tail-trace overlay prepared: %s\n' "$DESTINATION"
