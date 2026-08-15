# syntax=docker/dockerfile:1.7
# GLM-5.2 native SQG full-W4A8 -- SM120 local acceptance image.
#
# BASE_IMAGE is deliberately required and must be supplied by digest. The
# accepted local base is the current infernal-invocation SM120 image
# (local-inference-lab/vllm dev/infernal-invocation @ ce5f50f6 + integration
# PRs 285-287, b12x master @ 184d7d52 + PRs 145-146, CUDA 13.3.1,
# PyTorch 2.13.0, cutlass-dsl 4.6.2).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG RELEASE_IMAGE_REF=verdictai/glm52-sqg-w4a8-sm120:local-r1
# 0 = sealed full-W4A8 (any A16 route is rejected). 1 = permit the env-gated
# experimental act-A16 arm; still defaults to a8 at runtime.
ARG ALLOW_EXPERIMENTAL_ACT_A16=0

LABEL org.opencontainers.image.title="GLM-5.2 coupled SQG H512/H128 K96-tail W4A8 SM120 runtime" \
      org.opencontainers.image.description="GLM-5.2 runtime with updated-QSRT coupled H512/H128 SiLU routed experts, sealed layer 3 at K48 and layers 4-77 at K96, source MTP78 preserved, direct-E4M3 W4A8, and native SQG K6 W6A16 non-routed tensors" \
      ai.verdict.release.image-ref="${RELEASE_IMAGE_REF}" \
      ai.verdict.model.source-repo="brandonmusic/GLM-5.2-SQG-W4A8" \
      ai.verdict.model.source-revision="593dd0d2de6f79ce4e65303930c22c75e1359d44" \
      ai.verdict.arch="sm_120a" \
      ai.verdict.topology="topology-neutral-tp4dcp1-accepted" \
      ai.verdict.routed.endpoint="sqg-coupled-h512-h128-w4a8-direct-e4m3" \
      ai.verdict.routed.bits-per-weight="3.1291118421052633" \
      ai.verdict.routed.coupled-target-bits-per-weight="3.1241666666666665" \
      ai.verdict.routed.census="layer3:k3=720,k4=48;layers4-77:k3=672,k4=96;mtp78:k3=384,k4=384" \
      ai.verdict.coupled.tp4-preactivation="all-gather-reassembly-v1" \
      ai.verdict.nonrouted.endpoint="native-sqg-k6-w6a16" \
      ai.verdict.routed.schedule="m128n64-blocks8-stages2" \
      ai.verdict.glm-w4a8.kernel-sha256="b95dde9347cf679836eedc1c76f0f569a4fb37dcc594a674e41666e4d0f162ad" \
      ai.verdict.glm-w4a8.activation="h-e4m3,act-e4m3" \
      ai.verdict.glm-w4a8.a16-fallback="forbidden" \
      ai.verdict.experimental.act-a16-arm="${ALLOW_EXPERIMENTAL_ACT_A16}"

RUN base_digest="${BASE_IMAGE##*@sha256:}" \
    && test "${BASE_IMAGE}" != "${base_digest}" \
    && test "${#base_digest}" -eq 64 \
    && printf '%s\n' "${base_digest}" | grep -Eq '^[0-9a-f]{64}$'

COPY overlay/ /opt/glm52-sqg-w4a8-sm120/source-overlay/overlay/
COPY OVERLAY_SHA256SUMS /opt/glm52-sqg-w4a8-sm120/source-overlay/OVERLAY_SHA256SUMS
COPY OVERLAY_PROVENANCE.json /opt/glm52-sqg-w4a8-sm120/OVERLAY_PROVENANCE.json
COPY verify-runtime-contract-sm120.py /opt/glm52-sqg-w4a8-sm120/verify-runtime-contract-sm120.py
COPY test_native_sqg_k6_w6a16_contract.py /opt/glm52-sqg-w4a8-sm120/test_native_sqg_k6_w6a16_contract.py

# This base resolves vllm/b12x from the /opt/infernal-invocation source
# checkouts (PYTHONPATH precedence) and also carries site-packages copies.
# Verify the sealed overlay, install it into BOTH roots so every entrypoint
# sees identical code, and run the static contract checker against the
# authoritative checkout.
RUN set -e; \
    SP="$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    CK=/opt/infernal-invocation; \
    echo "site-packages: ${SP}; checkout: ${CK}"; \
    test -d "${SP}/vllm" && test -d "${SP}/b12x"; \
    test -d "${CK}/vllm/vllm" && test -d "${CK}/b12x/b12x"; \
    printf '%s\n' "${SP}" > /opt/glm52-sqg-w4a8-sm120/SITE_PACKAGES; \
    cd /opt/glm52-sqg-w4a8-sm120/source-overlay; \
    sha256sum -c OVERLAY_SHA256SUMS; \
    # b12x is a COMPLETE package swap (self-consistent SQG worktree tree +
    # K6 patches); vllm is a file-level overlay onto the branch tree.
    rm -rf "${CK}/b12x/b12x" "${SP}/b12x"; \
    mkdir -p "${CK}/b12x/b12x" "${SP}/b12x"; \
    for root in "${CK}/vllm/vllm:vllm" "${CK}/b12x/b12x:b12x" "${SP}/vllm:vllm" "${SP}/b12x:b12x"; do \
      dst="${root%%:*}"; src="${root##*:}"; \
      cp -a "overlay/${src}/." "${dst}/"; \
    done; \
    find "${CK}" "${SP}/vllm/model_executor/layers/quantization" "${SP}/b12x" \
      -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true; \
    EXTRA=""; \
    if [ "${ALLOW_EXPERIMENTAL_ACT_A16}" = "1" ]; then \
      EXTRA="--allow-experimental-act-a16"; \
    fi; \
    /opt/venv/bin/python /opt/glm52-sqg-w4a8-sm120/verify-runtime-contract-sm120.py \
      --vllm-root "${CK}/vllm/vllm" \
      --b12x-root "${CK}/b12x/b12x" \
      --overlay-root /opt/glm52-sqg-w4a8-sm120/source-overlay ${EXTRA}; \
    /opt/venv/bin/python /opt/glm52-sqg-w4a8-sm120/verify-runtime-contract-sm120.py \
      --vllm-root "${SP}/vllm" \
      --b12x-root "${SP}/b12x" \
      --overlay-root /opt/glm52-sqg-w4a8-sm120/source-overlay ${EXTRA}

# CPU import gate: the overlay must resolve against this image's vLLM and
# B12X API surfaces without a GPU.
RUN /opt/venv/bin/python - <<'PY'
import importlib

mods = [
    "vllm.model_executor.layers.quantization.exl3",
    "b12x.gemm.trellis_linear.api",
    "b12x.gemm.trellis_linear.w4a8",
    "b12x.gemm.trellis_linear._small_m",
    "b12x.moe.glm_sqg_w4a8",
    "b12x.moe._shared.kernels.glm_trellis_w4a8",
    "b12x.moe._shared.kernels.glm_trellis_transform",
    "b12x.moe._shared.kernels.w4a16.kernel",
    "b12x.moe._shared.kernels.w4a16.prepare",
    "b12x.moe._shared.kernels.trellis_w4a8",
    "b12x.moe.fused_moe._impl",
    "b12x.moe.ep_moe",
]
for name in mods:
    module = importlib.import_module(name)
    assert module.__file__.startswith("/opt/infernal-invocation"), (
        name,
        module.__file__,
    )
exl3 = importlib.import_module("vllm.model_executor.layers.quantization.exl3")
assert hasattr(exl3.Exl3Config, "glm_sqg_layer_is_encoded")
assert hasattr(exl3.Exl3Config, "glm_sqg_layer_is_coupled")
assert hasattr(exl3, "_attested_glm_sqg_regime")
api = importlib.import_module("b12x.gemm.trellis_linear.api")
assert hasattr(api, "run_sqg_k6_w6a16")
lazy = importlib.import_module("b12x.gemm.trellis_linear")
assert hasattr(lazy, "run_sqg_k6_w6a16")
runtime = importlib.import_module("b12x.moe.glm_sqg_w4a8")
contract = runtime.glm_route_packed_w4a8_kernel_contract()
assert contract == {
    "kernel": "m128n64",
    "blocks_per_cta": 8,
    "stages": 2,
    "route_block_rows": 128,
    "tile_n": 64,
}, contract
print("SM120 overlay import gate: PASS", contract)
PY

# KLD tokenizer input (wikitext) needs the datasets library.
RUN /opt/venv/bin/pip install --no-cache-dir 'datasets>=3.0.0' \
    && /opt/venv/bin/python -c 'import datasets; print("datasets", datasets.__version__)'

COPY serve-glm52-sqg-sm120 /usr/local/bin/serve-glm52-sqg-sm120
COPY kld/run_kld_sm120.py /opt/glm52-sqg-w4a8-sm120/kld/run_kld_sm120.py
COPY scripts/gpu_probe_one_layer.py /opt/glm52-sqg-w4a8-sm120/scripts/gpu_probe_one_layer.py
RUN chmod 0755 /usr/local/bin/serve-glm52-sqg-sm120

ENV B12X_GLM_W4A8_ACCEPT_ARCH=sm_120 \
    B12X_GLM_W4A8_KERNEL=m128n64 \
    B12X_GLM_W4A8_V2_BLOCKS=8 \
    B12X_GLM_W4A8_V2_STAGES=2 \
    B12X_MOE_FORCE_A16=0 \
    VLLM_GLM_SQG_W4A8_WORKSPACE_LANES=2 \
    CUTE_DSL_ARCH=sm_120a \
    TORCH_CUDA_ARCH_LIST=12.0a

ENTRYPOINT ["/usr/local/bin/serve-glm52-sqg-sm120"]
CMD []
