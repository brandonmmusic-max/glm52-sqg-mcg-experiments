# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG RELEASE_IMAGE_REF=verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix

LABEL org.opencontainers.image.title="Infernal Invocation r11 GLM-5.2 K96 runtime" \
      org.opencontainers.image.description="Exact II r11 MTP3 lineage with native SQG donor commits, coupled H512/H128 K96 execution, and DCP4 corrections" \
      ai.verdict.release.image-ref="${RELEASE_IMAGE_REF}" \
      ai.verdict.infernal-invocation.base="r11" \
      ai.verdict.infernal-invocation.base-digest="sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971" \
      ai.verdict.vllm.r11-tree="908522a320ecc26582926228c9644af085f5a86c" \
      ai.verdict.b12x.r11-tree="5d648d944a047d4fac5c2035309c207b3faebd9c" \
      ai.verdict.vllm.sqg-donor="local-inference-lab/vllm-pr315-ca966847" \
      ai.verdict.b12x.sqg-donor="local-inference-lab/b12x-pr197-b234532" \
      ai.verdict.model.profile="glm52-sqg-coupled-h512-h128-k96tail" \
      ai.verdict.coupled.tp4-preactivation="all-gather-reassembly-v1" \
      ai.verdict.mtp.exl3-prefix="canonical-mtp-block-v1" \
      ai.verdict.target.topology="tp4-dcp4-mtp3"

RUN base_digest="${BASE_IMAGE##*@sha256:}" \
    && test "${BASE_IMAGE}" != "${base_digest}" \
    && test "${#base_digest}" -eq 64 \
    && printf '%s\n' "${base_digest}" | grep -Eq '^[0-9a-f]{64}$'

COPY vllm/vllm/ /opt/ii-r11-k96/source/vllm/
COPY b12x/b12x/ /opt/ii-r11-k96/source/b12x/
COPY verify_runtime.py /opt/ii-r11-k96/verify_runtime.py
COPY PROVENANCE.json /opt/ii-r11-k96/PROVENANCE.json

# r11 resolves the integration checkouts first but also carries installed
# package copies. Install the same exact r11+donor+K96 trees into both roots so
# every entrypoint and worker sees one source composition.
RUN set -eu; \
    SP="$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    CK=/opt/infernal-invocation; \
    test -d "${CK}/vllm/vllm"; \
    test -d "${CK}/b12x/b12x"; \
    test -d "${SP}/vllm"; \
    test -d "${SP}/b12x"; \
    # Overlay the r11+donor source without deleting compiled extensions and \
    # generated version metadata already present in the pinned r11 image. \
    cp -a /opt/ii-r11-k96/source/vllm/. "${CK}/vllm/vllm/"; \
    cp -a /opt/ii-r11-k96/source/b12x/. "${CK}/b12x/b12x/"; \
    cp -a /opt/ii-r11-k96/source/vllm/. "${SP}/vllm/"; \
    cp -a /opt/ii-r11-k96/source/b12x/. "${SP}/b12x/"; \
    find "${CK}/vllm/vllm" "${CK}/b12x/b12x" "${SP}/vllm" "${SP}/b12x" \
      -type d -name __pycache__ -prune -exec rm -rf {} +; \
    /opt/venv/bin/python /opt/ii-r11-k96/verify_runtime.py

ENV B12X_GLM_W4A8_ACCEPT_ARCH=sm_120 \
    B12X_GLM_W4A8_KERNEL=m128n64 \
    B12X_GLM_W4A8_V2_BLOCKS=8 \
    B12X_GLM_W4A8_V2_STAGES=2 \
    B12X_MOE_FORCE_A16=0 \
    VLLM_GLM_SQG_W4A8_WORKSPACE_LANES=2 \
    CUTE_DSL_ARCH=sm_120a \
    TORCH_CUDA_ARCH_LIST=12.0a
