# SQG candidate KLD operations

This directory contains only test orchestration and evidence. It does not
modify the protected production checkpoint or rerun its baseline KLD.

## Production service identity

- Service manager: standalone Docker container `glm-r33-fixed`; it is not a
  Compose service and has no Docker healthcheck.
- Host operator: `brandonmusic` (member of the `docker` group).
- Container process user: image default, UID 0 (`Config.User` is empty).
- Restart policy: `no`; network and IPC modes: `host`.
- Image ID:
  `sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82`.
- Entrypoint: `/usr/local/bin/serve-gilded-gnosis.sh`.
- Model: `/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED`,
  mounted at the identical container path read-only.
- Served name: `GLM-5.2-EXL3-TR3v4-3.5bpw`; host-network port: 8000.
- Runtime geometry: TP4, DCP4, MTP3, four GPUs, EXL3, B12X A16 MoE.

The inactive system unit `glm52-eval-resume.service` belongs to a separate
all-EXL3 resume workflow and does not own this container.

The authorized stop used for this experiment was:

```bash
docker stop glm-r33-fixed
```

Docker recorded a clean exit (`ExitCode=0`, `OOMKilled=false`) after the
initial authorized stop. At `2026-08-10T00:35:48Z`, an operator invoked the
restore identity check with `CHECK_ONLY=1` in the wrong shell position. That
briefly started the retained container. It was immediately stopped again; the
30-second stop deadline expired while vLLM was starting, so Docker recorded
`ExitCode=137` at `2026-08-10T00:36:41Z`. The model mount was read-only, the
layer-6 candidate encoder remained alive, and the correctly invoked identity
check below subsequently passed with state `exited`. The retained stopped
container is still the exact restart record; do not reconstruct it with a new
`docker run` command.

Restore and gate the original service with:

```bash
/home/brandonmusic/KLC_SANDBOXES/sqg_candidate_kld_20260809/restore_production.sh
```

That script verifies the immutable image ID, read-only model mount, model
environment, restart policy, and host network before `docker start`. It then
requires `/health`, the exact `/v1/models` identity, at least 80 GiB allocated
on each of four GPUs, and a one-token completion. A check that does not start
the container is available as:

```bash
CHECK_ONLY=1 \
  /home/brandonmusic/KLC_SANDBOXES/sqg_candidate_kld_20260809/restore_production.sh
```

## Candidate-only KLD gate

The runner scores the candidate against the saved 2,048-token BF16 logits. It
does not invoke, load, or measure the protected baseline model. It defaults to
five candidate runs because a same-base-image r33 five-run distribution is
already preserved:

- mean KLD: `0.0624498626218156`
- sample SD: `0.0015327574926078513`
- individual KLDs: `0.06252960619413393`, `0.06165074611207335`,
  `0.06490997858682186`, `0.06234928724525699`, `0.06080969497079187`
- immutable image ID:
  `sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82`
- summary:
  `/home/brandonmusic/klc-linux/release-gg-sparkinfer-20260721/results/20260809T194958Z-kld-fp8-dcp4/summary.json`

The earlier published r26 five-run distribution is retained as a secondary,
historical comparison:

- mean KLD: `0.061282244905043234`
- sample SD: `0.0013762397578544292`
- individual KLDs: `0.061643184582613184`, `0.06234428913227424`,
  `0.06262627287172272`, `0.06048326423363492`, `0.05931421370497114`
- regime: FP8 KV, BF16 RoPE, TP4/DCP4, A2A, DCP interleave 64, 2,047
  scored positions

The reference logits SHA-256 is
`87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063`.
The runner also pins the reference manifest, all three evaluation-code hashes,
and the preserved r33 image inspection, eval manifest, five raw logs, JSONL,
and summary before starting a GPU process.

This exact experiment requires the sealed SQG overlay and exact r33 boot
arguments. A no-launch preflight is available with `PREFLIGHT_ONLY=1`; the
five-run launch uses the same command without that variable:

```bash
IMAGE=sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82 \
SITE=/opt/venv/lib/python3.12/site-packages \
PYTHON_ENTRYPOINT=/opt/venv/bin/python \
RUNTIME_OVERLAY=/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_no_bf16_test/runtime_overlay \
EXTRA_DOCKER_ARGS_FILE=/home/brandonmusic/KLC_SANDBOXES/sqg_candidate_kld_20260809/r33_exact_runtime.args \
RUNS=5 \
STAMP=sqg-four-layer-r1 \
  /home/brandonmusic/KLC_SANDBOXES/sqg_candidate_kld_20260809/run_candidate_kld.sh \
  /home/brandonmusic/KLC_SANDBOXES/glm52_sqg_no_bf16_test/artifacts/GLM-5.2-EXL3-3.5bpw-SQG-L006-L028-L052-L077-r1
```

The preflight requires exactly layers 6, 28, 52, and 77, verifies 3,072 SQG
marker payloads and zero MCG markers in those layers, rejects tensor overrides,
and confirms each K3/K4 bit map matches the protected source. It rejects a
different image, overlay, extension mount, or r33 argument file.

Every output directory is fresh and contains the image inspection, candidate
metadata hashes, runtime-overlay and extra-argument hashes when used, run
manifest, raw logs, parsed JSONL, and `summary.json`.
The summary reports mean direction and a conservative Welch repeat-noise check
against the preserved five-run baseline. The base image, reference, evaluation
code, extension, and evaluation regime match. The SQG loader and required
non-fused dispatch for selected layers do not match the baseline runtime path,
so the result is explicitly labeled codebook-plus-runtime-dispatch, not a
codebook-only causal comparison. Repeat runs estimate runtime variation for one
fixed prompt; they do not establish generalization across text.
