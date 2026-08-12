# GLM-5.2 SQG full-W4A8 — SM120 local serving package

Reproducible local Docker serving package for the published
`brandonmusic/GLM-5.2-SQG-W4A8` checkpoint (immutable revision
`593dd0d2de6f79ce4e65303930c22c75e1359d44`, tag
`accepted-glm52-sqg-w4a8-b300-r1`) on four RTX PRO 6000 Blackwell GPUs
(SM120, CC 12.0).

**Base**: `voipmonitor/vllm@sha256:7c0899fb9b3d09fbebcdd45f7b34cf5533e8e22ded03edbe524c4dcb1e367420`
(`infernal-invocation-vllm6c50b0a-b12x1584743-fi1ac6942-cu133-torch213-20260812-r3` —
local-inference-lab/vllm `dev/infernal-invocation` @ `ce5f50f6` + PRs 285–287,
CUDA 13.3.1, PyTorch 2.13.0, cutlass-dsl 4.6.2). The b12x package inside the
built image is a complete swap to the SQG runtime worktree tree
(`glm52-sqg-runtime-20260809`, parent `7cecbb2c`) plus the native-SQG-K6
patch set, so the kernels are exactly the SQG-campaign-validated set.

## Validated topology

* **TP4 / DCP1** (`tp4dcp1` attestation) — the acceptance topology. TP4
  slices weights along the intermediate axis (2048 → 4×512, 128-aligned);
  codec semantics are identical to the planned TP4/DCP4 regime (DCP only
  shards decode-time KV ownership).
* **TP4 / DCP4** (`tp4dcp4`) is wired and boots, but is **numerically broken
  by upstream branch defects** (see "Known branch defects") — do not use it
  for serving or measurement until those are fixed upstream.

Codec preserved exactly as published: independent per-tensor K3/K4 routed
experts (direct-E4M3 SQG labels, MXFP8/E4M3 `h` and `act`, exact GLM
`SiLU(gate)*up`, FP32 accumulation/reduction, Hadamard/suh on the activation
side, svh in the output epilogue, sealed 384/384 per-layer census, per-layer
derived down-targets), native SQG **K6 W6A16** for the 380 non-routed
matrices, BF16 direct tensors as stored. No A16, MCG, BF16-MMA, or
dequantize-then-GEMM fallback exists for these tensors; guards fail closed.
KV cache: **fp8_ds_mla** (656 B/token records — the SM120
`B12X_MLA_SPARSE` backend's native format), explicit `--block-size 64`
matching the sparse-indexer page geometry.

## Quick start

```bash
cp .env.example .env            # DCP_SIZE=1 + tp4dcp1 are the defaults
./serve.sh build
./serve.sh preflight
./serve.sh probe                # one-layer TP4 reshard + native kernel GPU probe
./serve.sh start                # acceptance: MTP0, enforce-eager, fp8 KV, TP4/DCP1
./serve.sh smoke
./serve.sh verify
./serve.sh kld
./serve.sh stop
```

Local API URL: `http://127.0.0.1:9418/v1` (localhost only).

## Acceptance vs production

* `server` (default): **MTP0**, `--enforce-eager`, fp8 KV, max len 8192.
  All acceptance receipts (smoke, verify, KLD) describe THIS configuration.
* `server-prod` (`./serve.sh start-prod`): **MTP3** speculative decoding, FP8
  KV, CUDA graphs, max len 65536 (DCP1 KV budget). Its numbers are MTP3
  numbers — never mix them with MTP0 acceptance results. Optional
  `nvfp4_ds_mla` KV via `PROD_KV_CACHE_DTYPE` +
  `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` (self-describing records, no scales file).

## Topology attestation

The checkpoint's `quantization_config.json` seals the B300 acceptance regime
(`tp=1, pp=8, dcp=1`); weights are topology-neutral whole logical tensors.
`VLLM_GLM_SQG_W4A8_TOPOLOGY_ATTESTATION` re-points ONLY the engine-topology
expectation to another enumerated regime (`tp4dcp1` fallback, `tp4dcp4`
planned) and is logged loudly at load. Every codec gate — census, down
targets, codebook sentinels, K3/K4 pool partitions, no-A16 — is enforced
unchanged. Nothing is re-encoded.

## Known branch defects found during acceptance (dev/infernal-invocation)

All reproduced with receipts in `RESULTS/`; the first three are patched in
this image's overlay, the fourth is avoided by DCP1:

1. `w4a16/prepare.py` gate rejected `sqg_xor_cheb_t12` at K5/K6
   (obsolete on the worktree tree now shipped; kept for reference).
2. `mla_attention.py` DCP LSE-merge assumed a `.decode` metadata attribute
   `B12xMLASparseMetadata` doesn't have (patched with `getattr`).
3. `v1/attention/ops/common.py` `mask_dcp_empty_shards_` crashed on
   zero-sequence batches (pure-prefill profile pass under DCP>1; patched).
4. **DCP4 decode merges produce numerically wrong (finite) attention output**
   with `B12X_MLA_SPARSE` (their launcher defaults DCP=1, so unexercised
   upstream). Additionally, an engine envelope with
   `max_num_batched_tokens == index_topk (2048)` produced Xid-31 MMU faults
   during warmup. Symptoms, tracebacks, and the DCP1-vs-DCP4 discriminating
   smoke are receipted.

## Ground-truth codec verification

`scripts/ground_truth_oracle.py` decodes a real K6 matrix through the runtime
endpoint (identity probe) and compares against the ORIGINAL BF16 weight from
`zai-org/GLM-5.2`: **rel-RMSE 0.0407** (expected K6 quant error), while the
K6 endpoint and the independent generic W4A16 dense path agree bit-exactly.
Receipt: `RESULTS/ground_truth_oracle.json`.

## Gates and receipts (RESULTS/)

| Gate | Receipt |
|---|---|
| deletion manifest (user-directed cleanup) | `DELETION_MANIFEST_20260812.md` |
| model codec + census + HF revision proof | `model_codec_validation.json` |
| one-layer TP4 reshard + kernel probe | `gpu_probe_layer3.json` |
| ground-truth K6 decode vs original BF16 | `ground_truth_oracle.json` |
| native W4A8 load+execute closure (per worker) | `evidence/pp-00-pid-*.json` |
| runtime-path verification (logs + evidence) | `runtime_path_verification.json` |
| deterministic 16-token smoke | `smoke_16tok.json` |
| DCP1-vs-DCP4 discriminator smoke | `smoke_dcp1_diag.json` |
| sealed BF16-reference KLD | `kld/kld_sm120_tp4dcp1.json` |

The KLD receipt keeps all 2,047 per-position values plus mean, median,
symmetrically trimmed mean (trim fraction recorded), p95/p99, CVaR worst-1%,
max, finite counts, token-sequence hash, reference hashes, image id, kernel
hashes and schedule, topology, attention backend, GPU names/CC, and
CUDA/PyTorch/vLLM/B12X versions.

## Provenance

See `build-context/OVERLAY_PROVENANCE.json` and `SHA256SUMS`. The frozen SM120
PR11 route-packed kernel is byte-pinned
(`b95dde9347cf679836eedc1c76f0f569a4fb37dcc594a674e41666e4d0f162ad`); the
vLLM integration is the newest validated (B300 r7/K6) lineage with
predecessor hashes recorded. Base image, model revision, and overlay hashes
are stamped into the image labels.
