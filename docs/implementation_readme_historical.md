# GLM 5.2 fresh SQG bridge

This directory is an isolated treatment encoder. It does not read or accept
MCG trellis bytes, `suh`/`svh`, signs, scales, seeds, decoded weights, or old
permutations. Production calls require an immutable official-BF16
shard/tensor binding and a newly calibrated 2048-wide `h2_reverse` physical
order. The existing model's `bit_map` may be used only by the run manifest to
freeze each tensor's K3/K4 assignment.

The encoder performs these operations for each treatment tensor:

1. verify the official BF16 shard/tensor binding;
2. clone the source, apply one calibration-derived permutation to gate rows,
   up rows, or down columns, and transpose to EXL `[K, N]` orientation;
3. require a finite dense covariance that cannot enter KQuant's fallback;
4. run fresh KQuant signs, Hadamards, channel scales, G-scale search, dense-H
   BlockLDLQ, the exact `sqg_xor_cheb_t12` K3/K4 LUT, and C128 tail-biting;
5. pack native EXL words with an independent PyTorch packer;
6. independently unpack and decode only the packed states and stored FP16
   scale vectors; and
7. emit `trellis`, `suh`, `svh`, and the exclusive `.sqg = 0x53514731` marker.

`SharedResidualProfile` is the explicit topology-neutral interface. An input
profile leaves the shared `suh` untouched and moves each tensor's searched
G-scale to private `svh`; an output profile does the inverse. The isolated
KQuant backend accepts these caller-owned vectors directly and never consults
its module-global first-tensor scale cache.

`prepare_dense_h_session` creates caller-owned reusable BlockLDL state. Gate
and up tensors sharing one layer H13 must pass the same compatible
`DenseHSession`; production gate/up calls fail closed without it. The session
is bound to the H hash, normalization, device, damping, and input profile and
uses a private lock. It never enters module-global state.

Interrupted gate/up sequences are resumed with
`resume_dense_h_session`. The worker validates the exact sealed gate-then-up
tensor-ID prefix, re-factorizes the bound H13 without encoding a sacrificial
weight, and requires the new BlockLDL fingerprint to equal the one in every
prior tensor manifest before restoring the ordinal.

Every tensor manifest records the official source binding, fresh physical
permutation/calibration evidence, transform seeds and hashes, fresh scale
hashes, dense-H hash and no-fallback proof, frozen LUT/K/C128 identity, packed
payload hash, SQG marker, and independent stored-FP16 closure metrics. The run
manifest additionally seals the forbidden-read list and exact K3/K4 census.

CPU-only synthetic tests exercise packing, tail-biting, independent decode,
source/Hessian clone isolation, explicit residual profiles, zero-MCG payloads,
fresh permutation closure, and manifest rejection. They do not launch a model
or a production quantization job:

```bash
python -m pytest
```

The real encoder is loaded explicitly with `load_kquant_runtime(kquant_root,
exllamav3_root)`. A treatment runner must use a fresh destination and should
materialize nothing until the per-tensor and run manifests validate.

The four-layer live-router calibration design, sealed owner-document split,
raw capture ABI, GLM scale-placement audit, and candidate-specific adaptive
`H2` contract are documented in
[`docs/fresh_sqg_calibration_capture.md`](docs/fresh_sqg_calibration_capture.md).
Its model-workload command has not been run.

## Four-layer experiment protocol

The worker in `src/run_fresh_sqg.py` implements the preregistered layers
6/28/52/77 experiment. It requires exactly eight fresh sign draws and four
fresh magnitude families: identity, aggregate RMS, BMMLaw quarter RMS, and
inverse quarter RMS. All 32 cells use the same 16-expert panel derived from
fit gate-square mass. No proxy pruning is allowed. Every cell performs exact
gate/up SQG encode and stored-FP16 decode, exact candidate-conditional H2,
exact down SQG encode, and routed selection replay that sums expert outputs
before squaring. Selection uses a 31-comparison Bonferroni-corrected paired
document bootstrap against identity/draw-00. Holdout is report-only after the
profile and bytes are frozen.

Preflight hashes the complete executable ExLlamaV3 package, the precompiled
`exllamav3_ext` directory and binary, all local `src` and
`bmmlaw_r7_encoder` Python code, the exact KQuant revision/diff, and the real
CUDA codec smoke evidence. It also requires the separately built and sealed
`kquant_sqg_quantize_ext_v22` shared object. Layer workers load that exact
hash through KQuant's prebuilt override; they receive no writable Torch
extension cache and cannot enter the JIT path. The ExLlama extension directory
must be the first entry in `PYTHONPATH`; JIT compilation is not accepted.

`plan-jobs` writes four inert argv/environment records with one layer per
visible GPU. It does not execute them. An operator can later run the recorded
commands explicitly. Every stage is resumable, but no serving container or
existing model is touched.

## Final layer ABI

Each `layer_LLL/final/fresh-sqg-layer-LLL.safetensors` contains exactly two
shared FP16 vectors plus 768 SQG treatment tensors. Gate/up store private
`svh`; down stores private `suh`; the opposite side is resolved through the
sidecar's exact `vector_refs`. Each tensor has one packed `trellis` and scalar
I32 `.sqg = 0x53514731`. The validator requires 2,306 keys per layer, exact
384 K3/384 K4 allocation, 768 SQG and zero MCG tensors, all fresh payload
hashes, 256 fit-derived permutations, candidate-H2 lineage, and the 256
sealed expert construction artifacts.

The four-layer `run_seal.json` proves 3,072 SQG, 0 MCG, 1,536 K3, and 1,536
K4 tensors. `materialize_fresh_sqg_candidate.py` consumes that sealed ABI and
builds a new flat runtime checkpoint from an explicit allowlist. It hardlinks
only unchanged, unselected runtime files. Each selected legacy shard and
sidecar is excluded and replaced by a new-inode SQG shard plus a generated
sidecar bound to the original fresh artifact payload hashes. The only selected-
layer quantization control inherited from the existing model is the frozen
per-tensor K3/K4 assignment. The contract is pinned byte-for-byte at SHA256
`1fe5a065ef31c2e4c27589415b87bb77a91f095c55eaf4594807ca22645dab33`;
matching only its aggregate 384/384 histogram is insufficient.

No recursive clone is used, and no source calibration file, encoding artifact,
MCG transform, scale, permutation, seed, payload, or decoded weight is copied
into a selected layer. `MANIFEST.json` and the last-written
`.manifest_verified` marker close the exact candidate file set. Materializing
or validating does not load the model:

```bash
python materialize_fresh_sqg_candidate.py --materialize \
  --source-model /home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED \
  --teacher-receipt evidence/teacher_model_identity.json \
  --artifacts-root /a/sealed/fresh-four-layer-run \
  --run-seal /a/sealed/fresh-four-layer-run/run_seal.json \
  --bit-contract contracts/frozen_bit_allocations.json \
  --output /a/new-existing-output-parent/fresh-sqg-candidate
```

The output parent must already be a real directory, and the final candidate
path must not exist.

The KLD runner validates that complete construction chain again before model
startup. Every accepted run must also publish a 2,047-element float32
`KL(ref||model)` safetensors file whose SHA256, exact metadata, finite
nonnegative values, and mean agreement are independently checked before the
scalar record enters `runs.jsonl`. The complete 2,048-token prompt is pinned
as little-endian uint32 bytes at SHA256
`ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56`;
the older first-16-token fingerprint is not the acceptance gate. The mounted
KLD dependency tree is exact-inventoried and byte-hashed, and Hugging Face,
datasets, and Transformers network fetches are disabled. A metadata-only,
scalar-only, or old hybrid candidate cannot pass this gate.
`PREFLIGHT_ONLY=1` performs the read-only candidate/runtime checks without
launching a container:

```bash
FRESH_ARTIFACTS_ROOT=/a/sealed/fresh-four-layer-run \
RUNTIME_OVERLAY="$PWD/evaluation/runtime_overlay" \
EXTRA_DOCKER_ARGS_FILE="$PWD/evaluation/r33_exact_runtime.args" \
PREFLIGHT_ONLY=1 \
evaluation/run_fresh_kld.sh /a/new-existing-output-parent/fresh-sqg-candidate
```

The mandatory dispatch control is a separate five-boot arm over the protected
all-MCG checkpoint. It accepts no model argument, mounts that checkpoint
read-only, uses the exact native-control args, rehashes the sealed runtime model
files before every launch, and requires native-dispatch plus 45+3 fused-slot
accounting proof in every log. It must use a result root disjoint from candidate
results:

```bash
NATIVE_MCG_RESULTS_ROOT=/a/new/native-mcg-control-results \
STAMP=fresh-sqg-native-mcg-r1 \
  evaluation/run_native_mcg_control.sh
```

Once both five-run summaries exist, compare the actual position vectors rather
than only their scalar means:

```bash
python3 evaluation/analyze_native_mcg_pair.py \
  /a/candidate-results/summary.json \
  /a/native-mcg-control-results/summary.json \
  --json-output /a/paired-results/sqg-vs-native-mcg.json \
  --tensor-output /a/paired-results/sqg-vs-native-mcg.safetensors
```

The complete preregistered runtime controls and interpretation limits are in
[`evaluation/CONTROL_ARMS.md`](evaluation/CONTROL_ARMS.md).
