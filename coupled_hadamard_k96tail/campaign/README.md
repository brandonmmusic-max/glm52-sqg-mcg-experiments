# GLM 5.2 fresh SQG bridge

The active mixed K48-layer3/K96-layers4--77 coupled-Hadamard distributed
re-encode, including exact pins, rental-node assignment, validation gates,
incident recovery, artifact layout, live ledger, and final-result placeholders,
is documented in
[`docs/K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md`](docs/K96_COUPLED_DISTRIBUTED_REPRODUCTION_20260814.md).
Its machine-readable live contract and read-only audit instructions are under
[`reproduction/`](reproduction/README.md).

The complete experiment history, methodology, failures, holdout qualifications,
and measured results are in
[`docs/sqg_mcg_experiment_record.md`](docs/sqg_mcg_experiment_record.md).
The separate read-only QSRT/Kimi K3 K1 feasibility audit is in
[`docs/qsrt_kimi_k3_k1_feasibility.md`](docs/qsrt_kimi_k3_k1_feasibility.md).
The current-QSRT snapshot and history comparison, including the new coupled
K3 Hadamard path, is in
[`docs/qsrt_current_update_review_2026-08-10.md`](docs/qsrt_current_update_review_2026-08-10.md).
The high-throughput four-GPU encoding procedure is documented in
[`docs/fast_parallel_sqg_encoding.md`](docs/fast_parallel_sqg_encoding.md).
The conditional full-model construction, fused-serving, validation, Docker,
and publication contract is in
[`docs/conditional_full_model_sqg_w4a8_plan.md`](docs/conditional_full_model_sqg_w4a8_plan.md).
The construction-only rolling full-W4A8 materializer and exact fixed-point
recapture commands are in
[`docs/PROGRESSIVE_FULL_W4A8_RECAPTURE.md`](docs/PROGRESSIVE_FULL_W4A8_RECAPTURE.md).

## Current measured status (2026-08-11)

The original conditional gate was **NO-GO** on its fixed-byte quality and
compact-core evidence.  The owner has since explicitly authorized a full
native-SQG W4A8 build to measure the corrected encoder.  Construction is in
progress, but deployment acceptance is still open: no corrected full-model
KLD, LAVD, Estonia, or integrated serving result is claimed.

- On the late 74--77 tensor inventory, CPU Test 8a RNE-rounded each regularized
  MCG lookup-table label to E4M3 before applying the existing Hadamard and
  scales; it did not round the final reconstructed tensor.  This increased
  existing BF16 weight-error energy by `4.515%` overall (`2.802%` K3,
  `11.558%` K4), while native SQG E4M3 labels had zero label-endpoint
  conversion error.
- The fully SQG late contiguous block completed five A16 KLD boots at
  `0.06242648`, versus the matching r33 mean `0.06244986`: `-0.037%`, far
  inside repeat noise. This rules out an obvious four-layer propagation
  catastrophe; it does not demonstrate a KLD win.
- The completed late `0/25/50/75/100%` expert-local H13 panel selected
  `alpha=0.25` (75% layer-shared/25% expert-local) among SQG arms on selection
  and secondary holdout.  It won all four support quartiles, but MCG remained
  the formal winner: alpha 0.25 was `2.7705%` worse on 52,263 holdout
  positions and improved only `39.03%`.  No layerwise or support-adaptive
  schedule passed the MCG gates.
- A second five-boot set using the same candidate path, selected-treatment
  manifest, and runtime identity is complete. Enumerating
  all balanced 5v5 partitions gives a 95% mean-delta null envelope of
  `±0.0013867` and a same-checkpoint position-win interval of `47.61–51.00%`.
  Boot noise can therefore produce four-layer p99/CVaR movements of this
  scale.  Future gates retain the raw metrics and compare them with this
  checkpoint/prompt-specific empirical p95 reference; the 252 balanced
  partitions reuse only ten boots and are not independent experiments.
- A paired late trace plus an unchanged-r33 trace null found that most
  `43–54%` route-set churn is runtime noise: r33-vs-r33 itself produces
  `45–50%`.  One cross-arm pair showed `1.49–1.72x` the MoE-output drift of
  one r33-r33 pair; post-residual ratios rose from `1.04x` to `1.29x`.  This
  is treatment-consistent but unreplicated mechanistic evidence, not a causal
  estimate or proof that compounding is absent.
- The change in the aggregate cross-expert interaction term accounts for only
  `2.26%` of the late proxy
  regression; `97.74%` is already present in individual routed-expert SSE.
  The late `25% local / 75% shared` H13 is therefore not frozen fleet-wide.
- Test 8b evaluated every layer-77 expert with native SQG E4M3 weights and
  exact-path K32 MXFP8 activations.  Full W4A8 increased signed top-8
  functional NMSE by `22.9802%` on selection and `22.7790%` on holdout versus
  matched SQG A16.  `h`-only A8 cost about 4%; heavy-tailed SwiGLU `act`-only
  A8 cost about 19%.  This is a preregistered red result.
- Test 8c-core completed 126 real GLM-shape cases with the exact 384/384 K3/K4
  layer-77 census.  W4A8 projected to `1.2972x` end-to-end at M=1 but fell to
  `1.1310x` at M=3,072 and `1.0684x` at M=4,096 (declared MoE fraction 0.31),
  below the long-prefill migration floor.  This is compact-core evidence, not
  a route-packed serving result.
- PR11 subsequently repaired the route-packed large-M schedule without
  changing model bytes.  Its full-W4A8 speed-only arm reached `1.75505x` at
  M=3,072 and `1.82359x` at M=4,096, projecting to `1.15389x` and `1.16280x`
  at the declared 31% MoE share.  This closes the isolated kernel-collapse
  question; the arm remains quality-unqualified.
- The first beta-1 full-W4A8 `(H,B)` down pilot was invalidated after scoring:
  KQuant transformed a caller Hessian that was already in label/effective
  coordinates, and its re-encoded private down `suh` drifted from the operand
  used to fit the target.  Its +5.7690%/+5.4980% selection/holdout result is a
  rejected pilot, not the corrected encoder's quality estimate.
- The correction now separates the canonical target operand from KQuant's
  `q_pre = Q R D` caller coordinate, anchors both private down `suh` and shared
  `svh` exactly, builds profile-specific W4A8-native H13, and defines an exact
  all-eight-triplet K3/K4 allocator.  The corrected beta-1 layer-77 score is
  now complete: signed top-8 NMSE improved by `9.5762%` on selection and
  `9.3043%` on secondary holdout versus the matched base full-W4A8 path, with
  both paired-document 95% intervals favorable.  This cuts the Test 8b
  full-W4A8-versus-A16 penalty from about 23% to `10.22%`/`10.86%`; it does
  not eliminate it.
- The beta-1 repair improves energy-dominant absolute tails but only
  `28.84%`/`29.91%` of positions.  Absolute squared-error worst-1% CVaR falls
  by roughly 30%, while relative-error p99 rises slightly.  Beta therefore
  remains a fit-only selection problem rather than a fleet-wide constant
  chosen from selection or holdout.
- The sealed fit/allocation-only beta panel is complete.  Across seven arms,
  `beta=0.0625` minimized the 16-expert complete-function objective at
  `0.0126999894` NMSE / `21,912.9237` SSE, `0.3344%` below beta 0.  Beta 1 was
  catastrophic (`0.0436718783` NMSE) and is excluded.  Selection and
  secondary-holdout rows did not enter H13, target construction, scoring, or
  hyperparameter choice.
- Full-wave layers 3--6 completed fit-only preparation with 256 fresh
  permutations per layer, zero fallback, and zero MCG inputs.  A real layer-5
  mixed-rate smoke passed, but these preparation files are not final encoded
  layers.
- The real-CUDA same-rate batching audit rejected the proposed encoder
  shortcut. K4 groups of two or four changed bytes, and K3x4 changed the
  all-K3 candidate on the first selected-beta high-route-mass panel expert.
  Final encoding therefore remains singleton per tensor; inference-kernel
  speed is unaffected.

The immediate production run uses the selected `beta=0.0625` to score every
layer-77 expert's eight realized W4A8 rate triplets, close the exact 384-K4 /
384-K3 allocation, and perform the selected-only re-encode.  Rolling full-wave
construction continues behind it: W4A8-native profile selection, native H13,
candidate-conditioned down encoding, progressive exact-W4A8 recapture, final
KLD and benchmark acceptance, and only then release publication.  See the
final handoff section in the experiment record for exact methodology and
exclusions.

This directory is an isolated treatment encoder. It does not read or accept
MCG trellis bytes, `suh`/`svh`, signs, scales, seeds, decoded weights, or old
permutations. Production calls require an immutable official-BF16
shard/tensor binding and a newly calibrated 2048-wide `h2_reverse` physical
order. The existing model's `bit_map` is lineage only for the final build.  A
fresh v3 allocator scores all eight realized W4A8 gate/up/down K3/K4 triplets
per expert and closes the topology-neutral layer budget at exactly 384 K3 and
384 K4 tensors without using the inherited map to choose rates.

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
The separated-layer and late contiguous-block workloads have now been run;
their accepted results and discovered failures are recorded in the experiment
history linked above.

## Four-layer experiment protocol

The intended preregistered layers 6/28/52/77 protocol requires exactly eight
fresh sign draws and four fresh magnitude families: identity, aggregate RMS,
BMMLaw quarter RMS, and inverse quarter RMS.  The accepted accelerated Test 3
execution used four draws and 16 cells; that deviation is recorded in the
experiment history and must not be described as the full 32-cell protocol.
All cells use the same 16-expert panel derived from fit gate-square mass. No
proxy pruning is allowed. Every cell performs exact
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
