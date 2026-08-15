#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'vast docker shim: %s\n' "$*" >&2; exit 2; }

case "${1:-}" in
  rm)
    # Synchronous Vast workers have no container object to remove.
    exit 0
    ;;
  run)
    shift
    ;;
  *)
    die "only docker run and docker rm are supported"
    ;;
esac

gpu=
entrypoint=
workdir=/
detached=0
runtime_root=/workspace/k96-runtime/bundle/rootfs
driver_root=/workspace/k96-runtime/driver-libs
declare -a bindings=()
declare -a environment=()

while (($#)); do
  case "$1" in
    --rm)
      shift
      ;;
    -d)
      detached=1
      shift
      ;;
    --name|--network|--shm-size|--cpus|--ipc)
      (($# >= 2)) || die "missing value for $1"
      shift 2
      ;;
    --network=*|--shm-size=*|--cpus=*|--ipc=*)
      shift
      ;;
    --gpus)
      (($# >= 2)) || die "missing value for --gpus"
      gpu=${2#device=}
      shift 2
      ;;
    --gpus=*)
      gpu=${1#--gpus=}
      gpu=${gpu#device=}
      shift
      ;;
    --mount)
      (($# >= 2)) || die "missing value for --mount"
      bindings+=("$2")
      shift 2
      ;;
    -e|--env)
      (($# >= 2)) || die "missing value for $1"
      environment+=("$2")
      shift 2
      ;;
    --entrypoint)
      (($# >= 2)) || die "missing value for --entrypoint"
      entrypoint=$2
      shift 2
      ;;
    -w|--workdir)
      (($# >= 2)) || die "missing value for $1"
      workdir=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "unsupported docker run option: $1"
      ;;
    *)
      # The pinned image reference is an integrity assertion.  Its userspace
      # was already unpacked and /opt/venv points into that exact digest.
      image=$1
      shift
      break
      ;;
  esac
done

[[ "${image:-}" == sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82 ]] || \
  die "unexpected runtime image: ${image:-missing}"
((detached == 0)) || die "detached mode is intentionally unsupported"
[[ "$gpu" =~ ^[0-7]$ ]] || die "invalid or missing physical GPU: $gpu"
[[ -n "$entrypoint" ]] || die "an explicit entrypoint is required"
[[ -x "$runtime_root$entrypoint" || -L "$runtime_root$entrypoint" ]] || \
  die "entrypoint is not present in the pinned rootfs: $entrypoint"
[[ -f "$driver_root/libcuda.so.1" && -f "$driver_root/libnvidia-ml.so.1" ]] || \
  die "host NVIDIA driver bridge is incomplete: $driver_root"
command -v proot >/dev/null || die "proot is not installed"

declare -a proot_bindings=()
for specification in "${bindings[@]}"; do
  source_path=
  target_path=
  IFS=, read -r -a fields <<< "$specification"
  for field in "${fields[@]}"; do
    case "$field" in
      src=*|source=*) source_path=${field#*=} ;;
      dst=*|destination=*) target_path=${field#*=} ;;
    esac
  done
  [[ -n "$source_path" && -n "$target_path" ]] || \
    die "invalid bind specification: $specification"
  [[ -e "$source_path" && ! -L "$source_path" ]] || \
    die "bind source is absent or unsafe: $source_path"
  mkdir -p "$runtime_root$(dirname "$target_path")"
  if [[ -d "$source_path" ]]; then
    mkdir -p "$runtime_root$target_path"
  else
    [[ ! -d "$runtime_root$target_path" ]] || die "file bind target is a directory: $target_path"
    touch "$runtime_root$target_path"
  fi
  proot_bindings+=(-b "$source_path:$target_path")
done

mkdir -p "$runtime_root$workdir" "$runtime_root/host-driver-libs" \
  /workspace/k96-proot-tmp
export CUDA_VISIBLE_DEVICES=$gpu
export PROOT_TMP_DIR=/workspace/k96-proot-tmp
export PROOT_NO_SECCOMP=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for assignment in "${environment[@]}"; do
  [[ "$assignment" == *=* ]] || die "environment entry is not an assignment: $assignment"
  export "$assignment"
done

exec proot -0 -r "$runtime_root" \
  -b /dev -b /proc -b /sys -b "$driver_root:/host-driver-libs" \
  "${proot_bindings[@]}" -w "$workdir" /usr/bin/env \
  PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/host-driver-libs \
  VIRTUAL_ENV=/opt/venv NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS \
  "$entrypoint" "$@"
