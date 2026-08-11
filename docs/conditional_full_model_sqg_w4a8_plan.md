# Conditional full-model SQG/W4A8 execution and serving plan

Status: active full native-W4A8 measurement build. Tests 8b, 8c-core, the PR11
route-packed layer kernel, the coordinate-corrected `(H,B)` down repair, and
fit-only beta selection are complete. The owner explicitly authorized the
full encode under the corrected contract. Integrated serving and final
full-model quality remain release gates rather than construction blockers.

The current executable contract and first-wave boundary are summarized in the
[full native-W4A8 construction-status record](full_w4a8_construction_status_2026-08-11.md).
The first 3--6 wave has launched bootstrap native-W4A8 profile search under a
canonical whole-build binding; no final profile selection, per-layer beta
choice, encoded first-wave layer, or new full-model quality result existed at
that publication point.

The target is not merely a second 3.5-bpw checkpoint.  The deliverable is a
fully SQG, BMMLaw-calibrated GLM-5.2 model whose compact K3/K4 bytes can be
served through validated GLM kernels, followed by same-regime KLD,
`llm_decode_bench`, LAVD, and Estonia evidence.

## Decision state before Tests 8b and 8c

- MCG remains the formal winner of the late 74--77 signed-top-8 H13 grid.
- Among the SQG arms, `local-alpha=0.25` (25% expert-local and 75% layer-shared
  H13) is the stable diagnostic winner.  It won all four layers and all four
  effective-support quartiles.  Pure expert-local H13 was decisively worse.
- The 25% local arm improved the extreme absolute tails on selection, but its
  mean signed-top-8 NMSE was worse than MCG.  The document-disjoint,
  encoder-unseen but analysis-seen holdout also remained worse in mean NMSE.
- The exploratory all-SQG layerwise search did not find a combination that
  passed all MCG hard gates and did not materially advance over uniform 25%
  local H13.

Consequently, the alpha evidence supplies a reproducible SQG calibration
prior; it does not independently prove that a full SQG quant will beat MCG.
The full build is justified only if exact-E4M3 W4A8 quality and deployable
prefill speed supply the missing benefit.

## Original disposition and corrected build decision, 2026-08-11

The original fixed-byte conditional program was **NO-GO**:

- Test 8b was red.  Relative to the matched SQG-A16 arm, full W4A8 increased
  signed weighted top-8 NMSE by 22.9802% on selection and 22.7790% on
  holdout.  `h`-only A8 cost 3.9183%/4.0036%; `act`-only A8 cost
  19.1831%/18.8584%.  Full W4A8 improved only 0.0293%/0.0153% of positions.
  Its relative-error p99 worsened 5.8496%/5.8198%, and squared-error worst-1%
  CVaR worsened 42.7243%/44.7741%.  This is a replicated functional
  regression, not final-logit boot noise.
- Test 8c-core proved that the native compact kernel reaches the faster FP8
  path, but it did not establish the required long-prefill serving win.  The
  route-histogram serial-core projection declined from 1.2972x end-to-end at
  M=1 to 1.1310x at M=3,072 and 1.0684x at M=4,096, using the declared 0.31
  MoE fraction.  The latter two are below the 1.15x migration floor.  This
  compact result is not route-packed serving acceptance.
- The later route-packed hybrid layer repaired the original kernel defect
  without changing model bytes.  The M64xN256/K128 four-warp kernel reached
  1.1812x A16/hybrid at M=3,072 and 1.1686x at M=4,096 and bit-matched the
  original one-warp arithmetic.  At the declared 0.31 MoE fraction, however,
  those ratios project to only 1.0499x and 1.0468x whole-prefill speedups.
  The isolated layer kernel is green; the integrated serving gate remains
  open and below threshold by projection.

PR11 subsequently measured the full-W4A8 route-packed arm at `1.7550x` for
M=3,072 and `1.8236x` for M=4,096, projecting to `1.1539x`/`1.1628x` at the
declared 31% MoE fraction. This clears the isolated long-prefill speed floor;
the actual four-GPU workload fraction and integrated DCP4 result remain open.

The coordinate-corrected, dual-scale-anchored `(H,B)` down repair reduced
layer-77 signed top-8 NMSE by `9.5762%`/`9.3043%` versus base full W4A8, while
remaining `10.22%`/`10.86%` worse than SQG A16. A fit-only panel then selected
`beta=0.0625`; beta 1 was catastrophically worse and is excluded. This is the
active construction contract, not a claim that the final quality gate passed.

- The alpha panel selected 25% expert-local/75% layer-shared H13 among SQG
  candidates, but MCG remained the formal mean-NMSE winner.  The alpha-0.25
  holdout was 2.7705% worse in mean signed-top-8 NMSE and improved 39.0295% of
  positions.  No support-quartile or layerwise schedule displaced uniform
  alpha 0.25 or passed the MCG hard gates.

Those original results did not authorize a release. The later owner directive,
PR11 full-W4A8 speed result, corrected down objective, and fit-only beta panel
now authorize the 75-layer native-W4A8 construction as a measurement program.
Docker and Hugging Face publication remain conditional on the final quality
and integrated-serving battery; construction progress is not release
acceptance.

## Gate A: Test 8b, activation quality

Test 8b uses sealed layer 77, native SQG E4M3 labels, frozen per-tensor K3/K4
rates, and exact activation-side scale/Hadamard order.  It reports selection
and holdout separately for:

1. SQG A16;
2. A8 only at `h`;
3. A8 only at `act = SiLU(gate) * up`; and
4. full W4A8 at both activation sites.

The primary comparison is the incremental deterministic damage relative to
the matched SQG-A16 numeric oracle.  The sealed layer-77 MCG-A16 scalar is a
cross-harness reference only; it is not used to claim matched MCG tails.

The preregistered interpretation is:

- **Green:** full W4A8 is no more than 1% worse than SQG A16 in signed-top-8
  NMSE on both roles, and its per-position p99 and worst-1% CVaR are no more
  than 2% worse.  Neither isolated activation point shows a hidden larger
  regression.
- **Amber:** mean damage is 1--3%, or a p99/CVaR damage is 2--5%.  Before a
  full build, capture `H2_A8` from the exact upstream A8 gate/up path and run
  a layer-77 down-only re-encode with gate/up bytes, rates, transforms, and
  profiles frozen.  Activation clipping/scaling changes must be isolated in
  separate arms.
- **Red:** full W4A8 is more than 3% worse in mean on both roles, or more than
  5% worse in a replicated tail metric and the H2_A8/down-only repair does not
  recover it.  Do not start the full quant.

These thresholds are engineering gates for the deterministic layer proxy,
not claims about statistical significance or final-logit KLD.  Test 8b can
kill the W4A8 thesis; it cannot prove final model quality.

## Gate B: Test 8c, GLM W4A8 speed

Test 8c has two stages because the existing Kimi route kernel has the wrong
intermediate width, activation, and rate-container contract for GLM.

### 8c-core

The compact core must:

- consume the same checkpoint-native SQG tensor in both arms;
- preserve independent per-tensor K3/K4 assignments;
- decode SQG labels directly into E4M3 MMA registers;
- never materialize a dense FP8 or FP16 weight for the W4A8 timing arm;
- use actual layer-77 transformed captures and route-count distributions;
- cover gate/up `K=6144,N=2048` and down `K=2048,N=6144` at K3 and K4;
- cover global token counts 1, 128, 512, 1024, 2048, 3072, and 4096;
- compare compact SQG A16 and W4A8 under balanced ABBA replay with fixed
  workspaces, at least 200 samples per arm, confidence intervals, and sealed
  kernel identities; and
- pass an independent numeric reference on SM120 before timings are accepted.

The core is a falsification gate.  Its serial-call projection is explicitly
not a serving result.

### 8c-layer

If the core is promising, the migration decision requires a route-packed GLM
layer implementation with:

- hidden width 6144, intermediate width 2048, 256 experts, and top-8 routing;
- separate gate, up, and down K3/K4 maps (never one rate per expert);
- GLM SwiGLU rather than Kimi SiTU;
- route packing into useful prefill tiles;
- compact in-loop SQG decode, FP8 MMA, and FP32 accumulation;
- shared gate/up `suh`, expert-private gate/up `svh`, expert-private down
  `suh`, and shared down `svh` in their correct locations;
- activation-side Hadamards, with stored labels left on the E4M3 grid;
- caller-owned graph-safe workspace; and
- three timing arms: W4A8 with route packing, A16 under the identical routing
  schedule, and the current production A16 path.

The speed report must distinguish MoE-layer speed from whole-model prefill.
For measured MoE fraction `f` and MoE speedup `S_moe`, use:

```text
S_end_to_end = 1 / ((1 - f) + f / S_moe)
```

At the historical `f = 0.31`, 1.15x end-to-end requires about 1.73x MoE,
1.20x requires about 2.16x MoE, and 1.30x requires about 3.91x MoE.  The final
projection must replace 0.31 with a measurement from the actual brief and
long-prefill GLM regimes.

- **Green:** route-packed measured/projected end-to-end prefill is at least
  1.20x in the long-prefill workload, with the lower confidence bound at least
  1.15x.  C1 decode may retain A16 if W4A8 has no byte advantage there.
- **Amber:** the central projection is 1.15--1.20x or its interval crosses
  1.15x.  Continue only if the quality result is green and the remaining
  kernel overhead has a measured, bounded removal plan.
- **Red:** the route-packed projection is below 1.15x, or the compact core is
  below the MoE speed required to make 1.15x possible.  Do not incur the full
  serving-stack migration for W4A8.

Measured disposition: the isolated route-packed layer kernel is green against
its dispatch-matched A16 layer control, but its 0.31-MoE Amdahl projection is
red against the whole-prefill threshold. The original M64xN8 slowdown is no
longer a blocker. The remaining Test 8c evidence must come from the integrated
vLLM/DCP4 workload so the actual MoE share and non-MoE overlap replace the
historical 0.31 assumption.

## Full-model construction contract after a GO

### 1. Freeze inputs and identities

- Pin official GLM-5.2 BF16 revision
  `b4734de4facf877f85769a911abafc5283eab3d9` and every source shard hash.
- Pin KQuant/QSRT, the isolated SQG encoder, ExLlamaV3, the serving runtime,
  and CUDA extension hashes.
- Pin the calibration document plan, tokenizer, router semantics, applied
  gate definition, and fit/selection/holdout roles.
- Pin the exact target average bpw and metadata accounting before encoding.
- Preserve topology neutrality: rate allocation is per tensor.  It is never
  collapsed to one rate per expert.
- Resolve the high-tier contract explicitly.  The current production MCG map
  contains K5 tensors in some layers, while the first native SQG W4A8 kernel
  supports K3/K4.  The initial deployable SQG design should therefore optimize
  an exact per-layer 384-K3/384-K4 assignment at 3.5 expert bpw, unless a K5
  direct-E4M3 decoder and its compensating rate allocation are implemented and
  benchmarked first.  K5 tensors may not be silently dropped, rounded, or
  treated as K4.

### 2. Recover disk space without losing evidence

- First hash-close and publish the full-alpha and Test 8b/8c results.
- Then remove only superseded nonwinning alpha roots and disposable partial
  BF16 shard copies.  Retain the selected candidate, preparation data,
  manifests, source bindings, code, and compact result artifacts.
- Stream official BF16 shards in bounded batches and release each source shard
  only after its encoded payload and source-binding closure are sealed.
- Never delete or alter the current production MCG checkpoint during the
  build.

The expected final routed SQG payload is about 317.35 GB decimal (295.55 GiB),
with about 11.3 GB retained preparation state plus the active capture/profile/
assembly frontier.  The dedicated NVMe currently has about 335 GB free, which
is insufficient by itself.  Its completed, already-published historical
`r10-local-corrected-rebuild` occupies about 293 GB; after exact-path and
publication checks, removing that superseded rolling payload is the preferred
way to create a roughly 628-GB free construction volume.  This deletion must
not include the current production checkpoint or the published reproducibility
code/receipts.

### 3. Capture routed calibration

- Capture exact top-8 expert IDs and applied router gates in the same serving
  regime used by validation.
- Keep fit, selection, and holdout document-disjoint.  The existing secondary
  holdout is analysis-seen; final confirmation requires a new blind corpus.
- Build gate/up H13 per expert from routed fit rows with gate-square weighting,
  using `0.75 * H_layer + 0.25 * H_local,e` as the initial prior.
- Record effective support and shrink more weak experts toward the layer
  covariance only if a preregistered support rule beats the uniform prior.
  The current quartile result does not justify ad-hoc per-expert alpha tuning.

### 4. Search profiles and rotations natively

- Re-run profile, sign, and rotation search under the chosen H13 prior.  The
  current nonzero-alpha arms reused profiles selected under alpha zero, so
  those profiles are not a valid fleet-wide final search.
- Keep Hadamard transforms on the activation side for W4A8.
- Keep output-side scales in the FP32 epilogue where the ABI permits; apply
  input-side scales to activations before A8 quantization.
- Select using signed router-weighted summed expert outputs before squaring,
  not unsigned or independent-expert proxy error.

### 5. Encode gate/up and rebuild candidate-conditioned down calibration

- Encode gate and up with dense-H BlockLDLQ, the native E4M3 SQG codebook, and
  C128 tail-biting.
- Preserve an explicit per-tensor K3/K4 map.  A later SQG-native reallocation
  may change which tensors receive K4 while preserving the total bpw/census;
  it must be a separately scored arm.
- Reconstruct the exact candidate gate/up path and form expert-local H2 from
  its SwiGLU activations.
- For the W4A8 target, reconstruct the complete upstream candidate through
  input scaling, H128, MXFP8 activation, native E4M3 gate/up labels, FP32
  accumulation, output transforms, SwiGLU, down scaling/H128, and `act` A8.
  Calling the resulting operand `Q`, ordinary `H2_A8 = Q^T Q` with the old
  weight target is not sufficient because it ignores the upstream activation
  residual.  The repair objective is cross-term-aware distillation:

  ```text
  minimize ||Q W_hat - Y_BF16||^2 over compact SQG W_hat
  H = Q^T Q
  B = Q^T Y_BF16
  ```

  `Y_BF16` is the teacher expert output in the matching transformed output
  coordinates.  Accumulate `H` and `B` on fit documents with gate-square
  weighting and support shrinkage, then freeze them before selection/holdout.
  A down encoder that accepts only `(H, original W)` cannot express this
  repair and must be extended rather than mislabeled `H2_A8`.
- Encode down only after its matching H2 is sealed.  Never reuse stale BF16-
  upstream or A16-upstream H2 for the final W4A8 candidate.

### 6. Convert progressively

- Process front to back in bounded layer blocks.
- After each block or two, run fixed-point recapture through the partially
  converted candidate so downstream calibration observes actual upstream
  errors and routing changes.
- Seal each block's source hashes, Hessians, rates, profiles, permutations,
  transforms, packed payloads, independent decode closure, and zero-MCG
  census before advancing.
- Fail closed on a routing/tail catastrophe; do not silently fall back to MCG
  bytes, transforms, or scales inside the treatment.

Use four-layer waves (the last wave contains three layers), one layer per GPU,
four expert workers per GPU, and three CPU threads per worker.  This is the
proven 16-process/48-thread layout.  Prefetch only the next layer's official
source shards: audited nonmonotonic shard reuse makes a whole next-block
prefetch unnecessarily expensive.

### 7. Assemble the model

- Materialize a new checkpoint path.  Do not mutate the current MCG model.
- Require all 75 routed MoE layers to contain the intended SQG tensors and no
  selected-layer MCG state.
- Verify exact total K3/K4/K5-or-high-tier policy, payload bytes, metadata
  bytes, model bpw, source revision, and runtime ABI.
- For the K3/K4-only release contract, require exactly 57,600 SQG tensors,
  28,800 K3 tensors, 28,800 K4 tensors, and zero routed MCG tensors across
  layers 3--77.

The audited construction estimate is 12--16 hours after the missing
full-model supervisor, alpha-native search, H2_A8, and partial-candidate
capture support are implemented.  Kernel work and the final evaluation
battery are additional time.  Official routed BF16 source spans 277 shards
and about 1.485 TB, so downloads must be streamed behind four-layer encoding.

## Serving-kernel deliverables

The full quant is not complete until it has a deployable runtime.

The authoritative serving base is
`/home/brandonmusic/glm52-exl3-sparkinfer` at commit
`0f9b71d5049228589534614c6a15c139c8238959`, corresponding to the pinned
v31 `verdictai/glm52-exl3-sparkinfer` deployment.  The older
`klc-linux/glm52-mdispatch` tree is historical evidence, not the release
source.

The current mixed SQG A16 runtime chooses one tier per expert and therefore
cannot express this model's independent gate/up/down rates.  Before a fused
release it must use projection-specific descriptor arrays and six payload
slabs:

```text
gate_k3, gate_k4
up_k3,   up_k4
down_k3, down_k4
```

This representation preserves one shared route pack while choosing each
projection's K3/K4 decoder independently.

1. **Decode/C1 path:** port the applicable MCG fused-dispatch and
   shared-input/shared-BF16 scheduling ideas to SQG without importing MCG
   reconstruction state.  Compare SQG A16 under identical dispatch against
   production A16.  W4A8 is optional at C1 because weight bytes are unchanged.
2. **Prefill path:** route-pack tokens by expert and rate, apply activation
   transforms/QDQ once per packed tile, decode compact SQG directly into FP8
   MMA fragments, execute gate/up and down, and fuse/reduce top-8 outputs where
   profitable.
3. **Correctness:** test K3 and K4 independently and mixed per-tensor layers;
   mismatched gate/up rates; empty, tiny, median, skewed, and maximum expert
   row counts; graph replay; multi-rank dispatch; and exact scale/Hadamard
   placement.
4. **Controls:** keep an identical-routing A16 arm and the current production
   arm.  A speedup over the slow expert-loop fallback alone is not evidence
   that FP8 MMA helped.
5. **Profiling:** report weight traffic, trellis-decode instructions,
   occupancy, tensor-core utilization, transforms/QDQ, route packing,
   collectives, reduction, and complete-layer latency.

## Final acceptance battery

### Quality

- Run the established final-logit KLD harness in the exact production regime,
  with the authoritative matching MCG checkpoint as control.
- Use several document-paired prompts, ordinary and symmetrically trimmed
  paired means, and null-calibrated tail metrics.  Do not infer a win from a
  four-layer or one-prompt noise-floor result.
- Report KLD for A16 and W4A8 separately so activation loss is not confounded
  with the codebook.
- Run LAVD and Estonia through the local `llm_decode_bench.py` profile
  contracts.  Preserve their exact model, sampling, MTP, concurrency, context,
  run-count, and token-cap regimes; a smoke pass is not the release battery.

### Performance

- Run `llm_decode_bench.py` on the same image/runtime for the new model and
  frozen controls.
- Include C1/C2/C4 decode and representative 0/8K/16K/32K context cells.
- Include brief, medium, and long prefill, including the long-context legal
  workload for which W4A8 is intended.
- Report TTFT/prefill, inter-token/decode, end-to-end latency, throughput,
  memory/KV capacity, graph status, and failure counts.
- Separate A16 and W4A8, warm and cold cache, and fused versus matched
  non-fused controls.

### Release rule

The candidate is accepted only if:

- all construction/source/zero-MCG closure checks pass;
- final KLD is within the preregistered quality budget and tails remain inside
  the matched empirical null allowance;
- LAVD and Estonia meet the frozen release thresholds;
- no correctness or graph-replay failure appears in the kernel matrix; and
- measured long-prefill gain clears the serving-value bar without sacrificing
  the current decode improvement.

All failures, retries, excluded pre-inference attempts, kernel substitutions,
and deviations from this contract must be recorded before rerunning.

## Packaging and publication after acceptance

Publishing is conditional on the full acceptance battery; a locally encoded
checkpoint or a kernel smoke test is not sufficient.

### Hugging Face model repository

- Create a new repository in the user's Hugging Face namespace rather than
  overwriting the existing MCG release.
- Upload the complete checkpoint, tokenizer/configuration files, quantization
  metadata, exact per-tensor rate contract, source revision, model-byte and
  manifest hashes, and runtime compatibility metadata.
- Include a model card that distinguishes measured facts from projections and
  reports A16 and W4A8 KLD, LAVD, Estonia, prefill, decode, memory/capacity, and
  known limitations under their exact regimes.
- Include the construction receipt, zero-MCG census, kernel/image identity,
  and links to the public experiment and runtime source repositories.
- Upload with resumable Hugging Face tooling and verify the final remote commit
  and representative large-file hashes before calling publication complete.

### Runtime image

- Build a versioned image in the user's Docker Hub namespace (the exact
  existing namespace/repository spelling must be read from the current
  working MCG deployment rather than guessed).
- Pin the source commit, CUDA/PyTorch/vLLM/B12X/ExLlama components, compiled
  SM120 extensions, and model ABI in image labels and a build receipt.
- Exercise the image locally on all four RTX PRO 6000 Blackwell GPUs before
  pushing it, including startup, graph capture, short generation, KLD regime,
  prefill/decode cells, LAVD, and Estonia.
- Push an immutable version tag and record its registry digest.  A mutable tag
  may additionally point to it, but the release files must use the digest or
  immutable tag.

### Launch artifacts

Ship both:

- `compose.yaml`, with the tested image tag/digest, model mount, four-GPU
  topology, IPC/shared-memory, ports, health check, required environment, and
  safe defaults; and
- `serve.sh`, which resolves its own directory, validates the model/runtime
  contract, prints the exact launch identity, fails on missing inputs, and
  starts the same command used in the accepted benchmarks.

The artifacts must not contain private tokens or machine-specific temporary
paths.  Provide documented environment variables for the local model path,
served model name, port, and optional performance settings.  Validate both a
fresh Compose launch and direct `serve.sh` launch before publication.
