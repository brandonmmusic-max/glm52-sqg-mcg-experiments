# Fresh SQG calibration capture contract

## Status

The capture implementation and CPU tests are complete. No model workload has
been launched and no calibration payload has been produced. The existing
3.5-bpw model is a read-only activation generator; this workflow never edits,
repackages, or restarts it.

This capture exists to calibrate a genuinely fresh SQG treatment. It does not
carry forward any MCG transform, scale, permutation, seed, decoded weight, or
payload byte. It also does not treat a global/source-upstream down-projection
covariance as an expert's candidate-specific `H2`.

## Sealed population and split

The owner calibration selection is reused, but its documents are repartitioned
without changing the selected population. The frozen inputs are:

- owner manifest:
  `/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED/calibration_manifest.json`
- owner-manifest SHA256:
  `b14c763fbc8feca6539f1411139fbfae58a7a906df5cbd5b9d9e104ab0475699`
- owner capture fingerprint:
  `2efd10279b8c953e3e46a469d9ec9970593795859c7a2cebc98c9ea707115b51`
- corpus:
  `/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED/calibration/reap_recall_calib.jsonl`
- corpus SHA256:
  `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`

All 4,497 owner-selected documents and 1,050,468 tokens are retained. A
document identity is `sha256(text UTF-8)`. Its lowercase hex identity is
hashed with BLAKE2b, digest size 8; the digest is interpreted as little-endian
`uint64` modulo 5. Buckets 0-2 are fit, 3 is selection, and 4 is holdout. The
resulting hard closure is:

| Role | Documents | Tokens |
| --- | ---: | ---: |
| fit | 2,638 | 601,343 |
| selection | 892 | 219,650 |
| holdout | 967 | 229,475 |

Splitting only at complete-document boundaries prevents token leakage between
Hessian fitting, candidate selection, and the untouched confirmation gate.
The document plan additionally binds each retokenized ID stream as explicit
little-endian `uint32` bytes and refuses tokenizer drift.

## Capture regime

The four frozen pilot layers are 6, 28, 52, and 77. They span early, middle,
late, and final routed blocks; every selected layer receives the complete
1,050,468-token population. This is a four-layer pilot, not evidence that four
layers estimate the full-model KLD effect precisely.

The runtime gates are:

- TP4, PP1, DP1, and DCP4, with `dcp_comm_backend=a2a` and
  `dcp_kv_cache_interleave_size=64`;
- expert parallel, EPLB, sequence-parallel MoE, and DBO disabled;
- one request and one complete document at a time;
- eager execution, no prefix cache, chunked prefill enabled, and no speculative
  decode;
- `max_num_batched_tokens=2048`, `kv_cache_memory_bytes=268435456`, and GPU
  memory utilization `0.90`, matching the saved r33 KLD endpoint;
- `max_model_len=4352`, the sole scheduler deviation from the saved KLD
  endpoint's 2560, because the sealed owner corpus includes intact documents
  through 4096 tokens plus one generated-token slot;
- custom EXL3 weights and the B12X MoE backend from the exact KLD image;
- `B12X_MLA_SPARSE`, requested FP8 KV (effective `fp8_ds_mla`), and
  `KV_FP8_ROPE=0`, matching the saved-logit KLD endpoint; and
- BF16 hidden states at the routed-MoE input boundary.

The exact `Exl3MoEMethod` is externally routed: its `is_monolithic` property is
false and `apply_monolithic` refuses execution. `VLLM_EXL3_R7_FUSED=1` fuses
expert execution but does not bypass `router.select_experts`, so capture keeps
the serving EXL3 expert path instead of substituting a CUTLASS/NVFP4 path.

Apart from the documented corpus-length extension, scheduler/cache controls
match the DCP4 KLD runner. Before a full capture, a dedicated one-document
DCP4 smoke must prove that TP rank zero sees exactly every document row at all
four selected MoE boundaries. The first sealed document has 3,586 tokens, so
the required 3,586 rows per layer specifically prove accumulation across the
2,048-token scheduler chunk boundary. The worker's end-of-document gate
compares the observed row count with the declared token count independently
for each layer. The smoke then calls the abort RPC, retains only exact-size
`.partial` ABI files in a fresh smoke directory, and must not create any final
payload, layer manifest, or `capture_manifest.json`. A failed smoke blocks the
full capture; DCP1 is not an unrecorded fallback. The production NVFP4 MLA
outer-scale file is deliberately not used for this experiment.

On TP rank zero, each selected live router instance is wrapped. The original
`select_experts` executes first and its return is passed through unchanged.
The wrapper captures its exact IDs, raw float32 weights, actual router logits,
and owning hidden states. Every row is independently recomputed as:

1. `score = sigmoid(actual_router_logits)`;
2. select top 8 using `score + live_e_score_correction_bias`;
3. gather the unbiased `score` at those IDs;
4. normalize the eight gathered values;
5. apply the sole audited scale product of 2.5.

The exact live IDs are preserved. An independent Torch sigmoid reconstructs
the effective weights on those IDs and rejects any weight outside the fixed
`rtol=2e-5, atol=2e-6` gate. For selection, each row computes
`max(unselected biased score) - min(selected biased score)`. The live set is
admissible only when that excess is at most `2e-6`; a larger excess means the
router selected a materially sub-top-8 expert and aborts capture. This narrow
boundary accounts for the fused CUDA kernel's `0.5*tanhf(0.5*x)+0.5` sigmoid:
an exact-image 10,000,000-row probe found 77 fused-versus-Torch set differences,
all with excess exactly zero or `1.9073486328125e-6`, while reconstructed live-ID
weights had zero failures and maximum absolute error `2.98e-8`. The layer
manifest seals the count of boundary-ambiguous rows, the maximum selection
excess, all 1,050,468 checked rows, and zero selection or weight violations.
This subrecord is explicitly versioned as
`glm52-fused-live-route-admissibility-v2`, so the former exact
`torch.topk`-ID oracle cannot be mistaken for this bounded gate.

The inspected runtime places the scale as router/routed-expert `1.0` times
actual `MoERunner` output `2.5`; the owning GLM MoE also declares `2.5`. The
capture separately reads all four locations, requires router equals routed
experts, runner equals owner, and uses only the actual runner value. This
prevents an owner config value from being mistaken for an applied runner
scale. The validator also understands the algebraically equivalent `2.5 x
1.0` placement but rejects every inconsistent placement, input/output
transform, and double scaling. `topk_weights.f32le.bin` stores effective
applied gates that sum to 2.5. Evidence retains a streaming SHA256 and
statistics for the exact raw router-return weights and each audited scale.

## Exact runtime provenance

Model capture is permitted only under:

```text
voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
```

The driver refuses host Python, requires `/opt/venv/bin/python`, and binds the
OCI identity declared by the inspected launcher. It independently hashes the
mounted `envs.py`, `deepseek_v2.py`, `exl3.py`, model-loader `utils.py`, B12X
`mixed_trellis.py`, EXL3 extension shared object, and exact modular-FusedMoE
owner/runner/router source files before importing the model. The live classes
must resolve to `DeepseekV2MoE`, `MoERunner`, `GroupedTopKRouter`, and
`Exl3MoEMethod` from those exact files. The same image declaration and exact
r33 execution environment are checked again on all four workers. Those hashes
and byte counts are sealed in
`capture_manifest.json`; status validation compares them with hard-coded
expected hashes. A declaration alone is therefore insufficient to pass.

The command, driver, plan helper, runtime gate, and worker implementation are
also a runtime-computed exact allowlist. Smoke and full evidence record every
allowed relative path, byte count, SHA256, and one canonical manifest digest.
Full mode recomputes this manifest before Docker and refuses a smoke made by
different local code; completed-capture validation repeats the comparison.
The allowlist includes the independent teacher-identity implementation and its
CLI wrapper.

The legacy model `MANIFEST.json` is not used as a payload identity authority.
Before either smoke or full capture imports vLLM, tokenizes, or loads the model,
the direct Python driver independently hashes all 343,070,719,678 allowlisted
teacher bytes against `evidence/teacher_model_identity.json`. This closes 156
indexed safetensors payloads, 75 R7 loader sidecars, and eight loader/tokenizer
identity files. The frozen canonical seal is
`e256a8c5c6e3734e47da61ecb7649bee596091178981ded0bbc00b69c50b766a` and the
exact receipt-file SHA256 is
`eb88fcd2bf66b0cfef195a232d9b81efcb107c400381b22c913a2c738413479e`.
Metadata-only validation is not accepted. The complete full-validation record,
including both hashes and `all_file_bytes_sha256_validated=true`, is sealed into
smoke and full runtime evidence and is required by completed-capture validation.

The environment seal includes the DCP A2A implementation and cutover, global
top-k/query-split controls, B12X MoE/A16 path, sparse indexer and FP8 GEMM,
the historical 48-layer fused budget, A1 threshold, and all Trellis prefill
limits. The teacher is loaded as the original all-MCG model with no selected-
layer reservation or candidate allowlist. Every TP rank must observe the exact
post-load fused set `6,7,8,10..54` (48 layers), no reserved slots, and selected
layer modes `6/28/52=fused`, `77=nonfused`; this is sealed into both smoke and
full-capture evidence. In particular, the preserved endpoint uses prefill
capacity 2048, not 1024. Additional DCP/EXL3 aliases in the sealed namespaces
are a fatal provenance error.

## On-disk ABI

Each `layer_NNN` directory contains exactly:

| File | Dtype | Shape |
| --- | --- | --- |
| `hidden.bf16.bin` | little-endian BF16 bits | `[1,050,468, 6,144]` |
| `topk_ids.u8.bin` | uint8 | `[1,050,468, 8]` |
| `topk_weights.f32le.bin` | little-endian float32 | `[1,050,468, 8]` |
| `doc_epochs.u32le.bin` | little-endian uint32 | `[1,050,468]` |
| `token_positions.u16le.bin` | little-endian uint16 | `[1,050,468]` |
| `role_ids.u8.bin` | uint8 | `[1,050,468]` |

One layer is 12,957,522,780 bytes and all four are 51,830,091,120 bytes
(about 48.3 GiB), before manifests and later Hessians. Files are written as
`.partial`, flushed and fsynced, and promoted only after all document and token
counts close. An aborted run retains partials and never deletes or overwrites
them. `capture_manifest.json` is the only completion sentinel.

## Hessian semantics

Only fit documents enter encoder Hessians.

For layer-global gate/up input geometry, with hidden row `x_t` and exact
effective applied gates `g_tj`:

```text
w_t   = sum_j g_tj^2
H13   = sum_t w_t x_t x_t^T / sum_t w_t
```

For down projection of expert `e`, each exact decoded SQG gate/up candidate is
replayed first:

```text
y_te       = SiLU(x_t @ decoded_gate_e.T) * (x_t @ decoded_up_e.T)
raw_H2_e   = sum_(t routed to e) g_te^2 y_te y_te^T / sum g_te^2
```

`raw_H2_e` is then passed through the hash-sealed KQuant
`adaptive_identity_shrinkage` implementation with maximum local alpha 0.75.
That policy computes Kish effective sample size from every retained
`g_te^2`, evaluates weighted OAS, and shrinks only toward
`trace(raw_H2_e) / 2048 * I`. Evidence records routed rows, fit documents,
gate-square mass, ESS, OAS shrinkage, uncapped local alpha, capped local alpha,
cap, identity scale, raw covariance hash, decoded candidate hashes, and final
Hessian hash. There is no pooled expert prior, layer-global `H2`, source-upstream
shortcut, borrowing, or fallback. Fewer than six routed rows or six fit
documents fails rather than substituting another covariance.

Because `H2` depends on decoded upstream values, it must be rebuilt for each
gate/up candidate that can feed a down candidate. Reusing one `H2` across
different decoded upstream candidates is outside this contract.

## Commands, intentionally not run

Plan preparation tokenizes and seals only; it does not load the model:

```bash
/usr/bin/python3 capture_calibration.py --plan \
  --model /home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED \
  --owner-manifest /home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED/calibration_manifest.json \
  --corpus /home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED/calibration/reap_recall_calib.jsonl \
  --plan-file /a/new/path/document_plan.json
```

The host vLLM installation is not valid for model capture. The exact inert
launcher is [`run_capture_container.sh`](../run_capture_container.sh). It
seals the full r33 execution environment, exact source hashes, TP4/DCP4 A2A
settings, physical GPU order, and a dedicated compilation cache. It has no
network or published port, mounts the activation-generator model and runtime
sources read-only, never mounts an online-quantization cache, and does not
address the production container.

First create separate empty smoke and full parents plus a new, empty cache
directory dedicated to these two runs. Capture and cache paths must have no
ancestry overlap. Then, only when model execution is explicitly authorized,
run the smoke:

```bash
FRESH_CAPTURE_PARENT=/a/new/empty/smoke-parent \
FRESH_JIT_CACHE=/a/new/dedicated/smoke-full-jit-cache \
./run_capture_container.sh smoke
```

The smoke runs exactly the first sealed document, requires rank zero's layer
counts to equal that document's token count for all four layers across the
2,048-token chunk boundary, calls the abort RPC, and writes
`dcp4_smoke_evidence.json`. It must leave no `capture_manifest.json`. After a
successful container exit the host launcher hashes the complete generated JIT
tree and writes a binding stamp covering its path/inode, file inventory,
smoke UUID/evidence digest, current plan, image, and capture-code manifest.
Only after that closure passes, use another empty parent for full capture while
reusing that exact cache directory:

```bash
FRESH_CAPTURE_PARENT=/a/new/empty/full-capture-parent \
FRESH_JIT_CACHE=/a/new/dedicated/smoke-full-jit-cache \
FRESH_DCP4_SMOKE_DIR=/a/new/empty/smoke-parent/fresh-sqg-dcp4-smoke-r1 \
./run_capture_container.sh full
```

Full mode independently validates every smoke field, current plan and first
document, all-rank fused population, four router audits, exact-size partial
files, current runtime/code hashes, and the unchanged recursive JIT inventory
before Docker starts. The direct capture process repeats this validation against
the read-only mounted smoke directory and JIT binding before any model load; it
seals the smoke UUID, smoke-evidence SHA256, JIT binding ID, binding-stamp hash,
and cache-inventory digest into `capture_manifest.json`. A direct
`capture_calibration.py --capture` invocation without those mounted preflight
inputs fails. Added, removed, replaced, or edited cache content fails.
The cache may contain only regular JIT/runtime compilation files and
directories; symlinks and special files fail. It is not a model, capture,
quantization, or sidecar path. The NVFP4 outer-scale JSON is intentionally
absent. Adding it or changing the requested FP8 KV regime would target another
metric.

Read-only validation after a completed capture is:

```bash
/usr/bin/python3 capture_calibration.py --status \
  --capture-dir /the/sealed/capture-directory
```

The CPU suite validates the real owner split and digests, exact runtime
provenance gates, wrapper ordering, unordered route alignment, both legal
scale placements, rejection of owner/runner disagreement and unknown scales,
raw/effective gate closure, candidate replay, exact KQuant shrinkage parity,
ESS, and no-borrow support gates:

```bash
/usr/bin/python3 -m pytest -q
```
