#!/usr/bin/env bash
set -u

conf_root=/etc/supervisor/conf.d
quarantine_root=/workspace/k96-quarantine/runahead-configs
canonical_guard=$conf_root/k96-score-runahead.conf

mkdir -p "$quarantine_root"

while true; do
  changed=0

  while read -r program _; do
    [[ $program == k96-score-runahead* ]] || continue
    supervisorctl stop "$program" >/dev/null 2>&1 || true
    changed=1
  done < <(supervisorctl status 2>/dev/null || true)

  while IFS= read -r config; do
    stamp=$(date +%Y%m%dT%H%M%S.%N)
    mv "$config" "$quarantine_root/$(basename "$config").$stamp.disabled"
    changed=1
  done < <(find "$conf_root" -maxdepth 1 -type f \
    -name '*score-runahead*.conf' -size +0c -print 2>/dev/null)

  if [[ ! -L $canonical_guard || $(readlink "$canonical_guard" 2>/dev/null || true) != /dev/null ]]; then
    if [[ -e $canonical_guard || -L $canonical_guard ]]; then
      stamp=$(date +%Y%m%dT%H%M%S.%N)
      mv "$canonical_guard" "$quarantine_root/$(basename "$canonical_guard").$stamp.disabled"
    fi
    ln -s /dev/null "$canonical_guard"
    changed=1
  fi

  if (( changed )); then
    supervisorctl reread >/dev/null 2>&1 || true
    supervisorctl update >/dev/null 2>&1 || true
  fi

  sleep 1
done
