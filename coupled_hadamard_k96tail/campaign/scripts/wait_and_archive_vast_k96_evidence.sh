#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'wait/archive Vast K96 evidence: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 4 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ ]] || \
  die "usage: $0 START END SUPERVISOR_PROGRAM HF_REPO"

start=$1
end=$2
program=$3
repo=$4
wave=$(printf '%03d-%03d' "$start" "$end")
success="wave encoded, sealed, and persisted to Hub: $start..$end repo=$repo"
log=/workspace/k96-logs/wave-${wave}-supervisor.log

while ! grep -Fqx "$success" "$log" 2>/dev/null; do
  state=$(supervisorctl status "$program" 2>/dev/null | awk '{print $2}' || true)
  case "$state" in
    RUNNING|STARTING|BACKOFF|STOPPED|EXITED) ;;
    FATAL) die "$program entered FATAL before its success seal" ;;
    *) ;;
  esac
  sleep 30
done

exec /home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625/scripts/archive_vast_k96_evidence.sh \
  "$start" "$end" "$repo"
