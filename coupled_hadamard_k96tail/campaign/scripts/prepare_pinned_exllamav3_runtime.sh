#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
RUNTIME_ROOT=${FRESH_SQG_RUNTIME_DEPENDENCY_ROOT:-$PROJECT_ROOT/runtime-dependencies}
PYTHON_ROOT=$RUNTIME_ROOT/exllamav3-python
EXTENSION_ROOT=$RUNTIME_ROOT/v39_ext/exllamav3
PYTHON_IMAGE=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
EXTENSION_IMAGE=verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a
EXTENSION_IMAGE_ID=sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac
PYTHON_TREE_SHA256=834c8de389c700126e5746e7bae9876b3e443014480651e59919cdd68fc20506
EXTENSION_SHA256=e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd

tree_sha256() {
  local root=$1
  (
    cd "$root"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
  )
}

extract_tree() {
  local image=$1 source=$2 target=$3 temporary container=
  [[ ! -e "$target" ]] || die "refusing to overwrite runtime dependency: $target"
  temporary=$(mktemp -d "$PROJECT_ROOT/.runtime-dependency.XXXXXXXX")
  cleanup() {
    [[ -z "$container" ]] || docker rm -f "$container" >/dev/null 2>&1 || true
    rm -rf -- "$temporary"
  }
  trap cleanup RETURN
  mkdir -p "$temporary/tree"
  container=$(docker create "$image")
  docker cp "$container:$source/." "$temporary/tree/"
  docker rm "$container" >/dev/null
  container=
  mkdir -p "$(dirname "$target")"
  mv "$temporary/tree" "$target"
  rmdir "$temporary"
  trap - RETURN
}

[[ -d "$PROJECT_ROOT" && ! -L "$PROJECT_ROOT" ]] || die "project root is absent or unsafe"
mkdir -p "$RUNTIME_ROOT"
[[ -d "$RUNTIME_ROOT" && ! -L "$RUNTIME_ROOT" ]] || die "runtime root is unsafe"

[[ $(docker image inspect --format '{{.Id}}' "$PYTHON_IMAGE") == "$PYTHON_IMAGE" ]] || \
  die "pinned Python source image ID differs"
[[ $(docker image inspect --format '{{.Id}}' "$EXTENSION_IMAGE") == "$EXTENSION_IMAGE_ID" ]] || \
  die "pinned extension source image ID differs"

if [[ ! -e "$PYTHON_ROOT" ]]; then
  extract_tree "$PYTHON_IMAGE" /opt/exllamav3-python "$PYTHON_ROOT"
fi
[[ -d "$PYTHON_ROOT" && ! -L "$PYTHON_ROOT" ]] || die "Python package root is unsafe"
[[ -f "$PYTHON_ROOT/exllamav3/__init__.py" ]] || die "Python package is incomplete"
[[ $(tree_sha256 "$PYTHON_ROOT") == "$PYTHON_TREE_SHA256" ]] || \
  die "pinned Python package tree hash differs"

if [[ ! -e "$EXTENSION_ROOT" ]]; then
  extract_tree "$EXTENSION_IMAGE" /opt/exllamav3 "$EXTENSION_ROOT"
fi
[[ -d "$EXTENSION_ROOT" && ! -L "$EXTENSION_ROOT" ]] || die "extension root is unsafe"
extension=$EXTENSION_ROOT/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
[[ -f "$extension" && ! -L "$extension" ]] || die "pinned extension is absent"
[[ $(sha256sum "$extension" | awk '{print $1}') == "$EXTENSION_SHA256" ]] || \
  die "pinned extension hash differs"

printf 'pinned ExLlamaV3 runtime ready: python_tree=%s extension=%s root=%s\n' \
  "$PYTHON_TREE_SHA256" "$EXTENSION_SHA256" "$RUNTIME_ROOT"
