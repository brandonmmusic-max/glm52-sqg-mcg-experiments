# SQG versus MCG on GLM-5.2 3.5 bpw: experiment record

**Record status:** living audit, updated 2026-08-11.  Tests 0–7b, the CPU
weight-endpoint portion of Test 8, Test 9, and the late 74--77 construction,
holdout proxy, five-boot A16 endpoint, ten-boot same-checkpoint null, paired
runtime trace, and r33-r33 trace control of Test 10 have local artifacts.  The
late H13 grid is active; no middle/early block or GLM W4A8 quality/speed result
is claimed yet.

This document reconstructs the SQG/MCG investigation in chronological and
causal order.  It distinguishes accepted measurements from rejected pilots,
runtime-confounded observations, external claims, and untested hypotheses.  It
is intentionally stricter than a progress narrative: a passing codec closure,
unit test, or packed-byte hash proves only the property that it actually checks.

## Reading conventions

- **Measured** means a result exists in a local artifact and is quoted from it.
- **Validated** means an independent closure check or sealed provenance check
  exists in addition to the measurement.
- **Rejected** means a run exposed a known defect and is not admissible as a
  quality result.
- **Historical/superseded** means the run is useful for learning about the
  pipeline but is not the current estimate of SQG quality.
- **Runtime-confounded** means the candidate changed both encoded weights and
  the execution/dispatch path.  It cannot identify the codebook effect alone.
- **Hypothesis** and **planned test** are predictions or designs, not results.

Lower NMSE, Hessian-weighted NMSE, and `KL(reference || model)` are better.
Positive “SQG reduction” percentages mean SQG has lower distortion.  Positive
`candidate - baseline` KLD means the candidate is worse.

## Invariants that must not be blurred

### Calibration granularity is not rate-allocation granularity

The model remains **topology-neutral with a per-tensor K3/K4 assignment**.  A
gate, up, or down tensor retains its own frozen bit rate; no expert receives a
single shared “expert bpw,” and routing topology is not used to change the
number of bits assigned to an expert.

Expert-conditioned calibration is compatible with that rule.  It changes the
error metric used to choose a tensor's labels, scales, or transforms; it does
not change the frozen per-tensor K3/K4 map.  Thus “expert-local H13” means an
expert-conditioned activation covariance for encoding that expert's gate and
up tensors.  It does **not** mean expert-level bit allocation.

### A clean selected SQG layer contains no MCG encoding state

For the fresh BF16 treatment, selected-layer SQG tensors may inherit only the
frozen per-tensor K3/K4 treatment map.  MCG payload bytes, scale vectors,
transforms, permutations, seeds, decoded weights, and sidecar lineage are not
inputs to the fresh SQG encoding.  All 768 eligible tensors in each selected
layer must be SQG, not a mixture retained to improve the score.

### Three different questions require three different endpoints

1. **Weight distortion:** is the packed reconstruction closer to BF16?
2. **Routed expert-function distortion:** is the complete expert computation
   closer on the activation distribution that actually reaches that expert?
3. **Final model quality:** are final logits closer to the saved BF16 logits?

The first two are diagnostic proxies.  Only the third directly answers the KLD
question.  W4A8 performance and quality add a fourth question because A8
quantization is absent from the current A16 measurements.

## Experiment ledger

| ID | Experiment | Status | What it can establish |
|---|---|---|---|
| 0 | Re-encode decoded MCG reconstruction as SQG, four layers | Historical; completed | Directional feasibility without BF16 weights; not clean SQG-from-BF16 quality |
| 1 | Fresh routed calibration capture | Completed through an explicit recovered-capture ABI | Fit/selection/holdout activations and routes for four layers, with a documented recovery limitation |
| 2 | Fresh BF16 SQG with mean-one gate/up profile | Rejected | Exposed an absolute-versus-relative scale-semantics bug |
| 3 | Corrected fresh BF16 SQG encoding, four layers | Completed | Clean packed SQG treatment at the frozen K3/K4 map |
| 4 | Corrected four-layer final-logit KLD | Completed but runtime-confounded | Combined effect of four SQG layers plus the SQG dispatch path |
| 5 | Native-dispatch MCG control | Preregistered; no completed local result found | Required separation of dispatch effect from codebook effect |
| 6a | Raw packed-byte NMSE, SQG versus MCG | Completed | Ordinary BF16 weight-space distortion |
| 6b | Hessian-weighted distortion under several activation geometries | Completed | Diagnoses objective alignment and candidate-specific down quality |
| 7 | Re-encode gate/up with expert-local/global-prior H13 and rebuild H2 | Completed packed-byte comparison | Confirms that the current SQG gate/up encoding was misaligned with routed expert geometry; does not establish final KLD |
| 7a | Materialize the expert-local-H13 treatment as a runnable four-layer candidate | Completed | Binds the sealed Test 7 bytes into a separate runnable checkpoint without re-encoding or mutating the protected source |
| 7b | Expert-local-H13 four-layer final-logit KLD | Completed; same-dispatch mean direction worse but repeat-noise inconclusive | Did not demonstrate that Test 7's isolated-expert gain lowers final-logit KLD on the one fixed prompt |
| 8a | Direct E4M3 weight endpoint | Completed on CPU | Exact incremental E4M3 endpoint distortion for matched MCG and SQG bytes |
| 8b/8c | Full GLM W4A8 quality and speed | Not run | A8 quality and actual FP8-MMA speed |
| 9 | Signed top-8, tail-constrained H13 blend ablation | Completed for separated layers | Selected `local-alpha=0.25` on the separated panel, but its final-logit tail remained unresolved |
| 10 | Contiguous block propagation | Late A16 endpoint, null, paired trace, and trace control complete; late H13 grid active | Late-block scalar catastrophe screen and calibration diagnosis; middle/early/W4A8 remain |

---

## Test 0 — no-BF16 MCG-reconstruction to SQG conversion

### Question and hypothesis

The narrow question was whether SQG could move final-logit KLD in the right
direction when the only available weight source was the existing 3.5-bpw MCG
checkpoint.  The hypothesis was explicitly weak: even a requantization of the
MCG reconstruction might expose a useful codebook direction.

### Methodology

- Decode each complete topology-neutral MCG tensor to its stored regularized
  reconstruction.
- Feed that already-quantized reconstruction to KQuant's production
  `sqg_xor_cheb_t12` Viterbi encoder.
- Preserve each tensor's original K3 or K4 rate.
- Copy the existing MCG `suh` and `svh` tensors unchanged.
- Select layers 6, 28, 52, and 77, each of which had no K5 tensors and had an
  exact census of 384 K3 plus 384 K4 tensors.
- Require every selected tensor to carry the SQG marker and no selected tensor
  to retain an MCG marker.
- Materialize a candidate that changes only those four layers; run five fresh
  candidate boots against the previously measured r33 BF16-logit reference
  regime: TP4, DCP4 A2A, interleave 64, FP8 KV, BF16 RoPE, 2,048 context tokens,
  and 2,047 scored positions.

Layer 3 was encoded during staging but excluded because its source map included
28 K5 tensors, outside this SQG contract.

### Fit, selection, and holdout

There was no new BF16 calibration population, Hessian fit, profile selection,
or document holdout in this conversion.  That absence is central to the test:
it was deliberately a “can this work at all without BF16/Hessians?” probe.

### Assumptions

- A useful signal might survive requantization of an MCG reconstruction.
- Reusing MCG `suh`/`svh` was acceptable only for this directional probe.
- The four separated layers were a broad positional sample, not a statistical
  estimator of a full-model conversion.

### Validation

- The shard validator closed marker exclusivity, packed hashes, legacy
  shapes/dtypes, unchanged payloads where required, and aggregate `suh`/`svh`
  hashes.
- Reconstruction provenance bound each SQG packed tensor to the decoded MCG
  reconstruction it consumed.
- The candidate audit required 3,072 SQG markers and zero MCG markers in the
  selected layers, validated changed model-index/header payloads, and checked
  source immutability.

### Result

The five candidate KLD values were:

```text
0.06494029413596611
0.06845580140500897
0.06664981401007228
0.06408844960937171
0.06395135181589742
mean      = 0.0656171421952633
sample SD = 0.0019166558827614939
```

The existing r33 baseline was `0.0624498626218156` with sample SD
`0.0015327574926078513`.  The candidate-minus-baseline difference was
`+0.003167279573447694`, or `+5.071715838%` worse.  The candidate and baseline
runtime code did not match, so this is the combined SQG-codebook plus
loader/non-fused-dispatch result, not a codebook-only estimate.

### Interpretation and supersession

This result rejected the idea that a mechanical MCG-reconstruction-to-SQG
conversion was sufficient.  It did **not** reject fresh SQG from BF16 because
it inherited MCG quantization error and MCG scale/transform choices and had no
candidate-matched Hessian.  Test 3 superseded it with a clean BF16 encode.

### Next hypothesis generated

A fair SQG test needed official BF16 tensors, newly captured routed activation
geometry, SQG-native profiles/transforms, and a down Hessian rebuilt from each
decoded SQG gate/up candidate.

---

## Test 1 — fresh routed calibration capture

### Question and design

The capture was designed to support a genuinely fresh SQG encode without
carrying forward MCG encoding state.  It captured the hidden vector entering
each selected routed-MoE layer and the exact live top-8 expert IDs and applied
router weights.

### Population and deterministic split

All 4,497 owner-selected documents and 1,050,468 tokens were retained.  A
document identity was `sha256(text UTF-8)`.  Its lowercase hex identity was
hashed with BLAKE2b (digest size 8), interpreted as a little-endian `uint64`,
and reduced modulo 5:

| Role | Buckets | Documents | Tokens | Authorized use |
|---|---|---:|---:|---|
| Fit | 0–2 | 2,638 | 601,343 | Hessians and fit-derived profile bases |
| Selection | 3 | 892 | 219,650 | Candidate/profile choice only |
| Holdout | 4 | 967 | 229,475 | Untouched confirmation |

Documents were never split across roles.  The retokenized token streams were
also sealed as little-endian `uint32` bytes to reject tokenizer drift.

### Layers and runtime regime

Layers 6, 28, 52, and 77 were selected as separated early, middle, late, and
final routed blocks.  Every layer received the entire population.  The capture
used TP4/PP1/DP1/DCP4, A2A, interleave 64, one complete document/request,
eager execution, chunked prefill, no prefix cache, no speculative decoding,
and the exact r33 B12X/EXL3 activation-generator path.  Hidden states were
captured in BF16 at the routed-MoE input.

`max_model_len=4352` was the one documented scheduler deviation from the saved
KLD endpoint's 2560, required to preserve documents through 4,096 tokens.  The
remaining relevant scheduler/cache controls matched the KLD endpoint.

### Route and scale semantics

For each row, the live router ran first and its output was returned unchanged.
The capture retained live IDs, weights, logits, and hidden states.  An
independent reconstruction applied sigmoid, selected top 8 using the correction
bias, gathered the unbiased scores, normalized, and applied the audited `2.5`
scale.  The stored effective gates sum to 2.5.

The layer-global gate/up covariance used by the original fresh encoder was

```text
w_t = sum_j g_tj^2
H13_global = sum_t w_t x_t x_t^T / sum_t w_t
```

Only fit rows entered it.

For expert `e`, down-projection H2 was constructed after decoding the gate/up
candidate:

```text
y_te = SiLU(x_t @ decoded_gate_e.T) * (x_t @ decoded_up_e.T)
raw_H2_e = sum_(t routed to e) g_te^2 y_te y_te^T / sum g_te^2
```

KQuant's weighted-OAS policy then shrank `raw_H2_e` toward its scaled identity,
with local alpha capped at 0.75.  A different decoded gate/up candidate required
a new H2.

### Validation and recovery incident

The accepted root reports `complete=true`, four layer manifests, 1,050,468
rows/layer, 24 final payloads, no partial payloads, and the exact split above.
It seals BF16 hidden values, route IDs, route weights, document epochs, token
positions, and role IDs.  Each layer payload is 12,957,522,780 bytes; all four
are 51,830,091,120 bytes (about 48.3 GiB) before manifests/Hessians.

The capture is not represented as an ordinary v1 completion.  Worker
finalization promoted the payloads after document/token closure, but the host
v1 manifest validation failed before the in-memory finalization record was
persisted.  A CPU-only, incident-specific recovery emitted the distinct
`glm52-fresh-sqg-calibration-capture-recovered-v1` ABI and sealed the recoverable
facts.  Its promotion proof records:

- 4,497 documents and 1,050,468 rows/layer satisfied the promotion precondition;
- all 24 expected final payload names were present;
- zero `.partial` payloads remained;
- payload-manifest SHA-256
  `a26904de01c4cc1d1017548adbb3d3f1cd2af28f8d859f2b70f5cebc9b58012b`;
- bound smoke, JIT, runtime, teacher identity, router class/configuration, and
  archived capture-code provenance.

The recovery explicitly does **not** fabricate three pieces of ephemeral
worker evidence that existed only in process memory: the original run UUID,
raw-router-return SHA/statistics, and reference-route diagnostics.  Those
fields are recorded as unavailable.  This is a documentation limitation of
the recovered evidence, not permission to claim those values were persisted.

The preregistered 10-million-row route probe had found 77 fused-versus-Torch
top-8 set differences, all at a selection boundary excess no greater than
`1.9073486328125e-6`; live-ID reconstructed weights had maximum absolute error
`2.98e-8`.  Those probe results explain the bounded route-admissibility rule,
but they should not be confused with the unavailable per-capture ephemeral
diagnostic fields.

### Assumptions and holdouts

- The activation generator was the already-quantized MCG model.  Consequently,
  this captures the distribution induced by the deployed MCG upstream path,
  not a hypothetical full-SQG upstream model.
- Four separated layers deliberately avoid a contiguous error-propagation
  experiment.  They do not test whether errors compound, cancel, or alter
  routing across adjacent SQG layers.
- Fit/selection/holdout independence applies to documents, but later tests must
  respect each role for that separation to matter.

### Next hypothesis generated

The fresh encoder could now test whether SQG is superior when it is given the
same BF16 sources, frozen bit map, routed calibration population, and
candidate-specific down H2 rather than MCG reconstruction inputs.

---

## Test 2 — rejected mean-one gate/up profile pilot

### Intended methodology

The pilot searched SQG-native residual/channel profiles and encoded gate/up
under the captured dense H13, decoded the candidate, rebuilt candidate-specific
H2, and encoded down.  It was intended to prove the fresh BF16 path before the
full four-layer wave.

### Error discovered

The gate/up `input_channel_scale_profile` was incorrectly normalized to mean
one.  In KQuant this profile is an **absolute input-channel RMS**, unlike the
down `output_channel_scale_profile`, which is a relative channel modulation and
is properly mean-normalized.

The rejected signature was unmistakable:

- gate/up global-scale search landed at exactly the `2.05` search boundary for
  every tensor;
- gate/up source-relative reconstruction RMSE exceeded 1.0;
- down error remained in a normal quantization range.

Codec encode/decode closure still passed because the encoder and decoder could
faithfully reproduce the same badly scaled reconstruction.  This established
an important validation lesson: **packed closure does not prove the candidate
is sensibly scaled relative to BF16.**

### Recovery and validation design

The invalid pilot was retained and not edited in place.  The corrected
successor retained the run ID and transform seeds but imported only valid H13,
raw profile-scale, and permutation evidence.  It did not import the rejected
profile search, H2, down encoding, final-layer bytes, or treatment artifacts.

A mandatory absolute-scale smoke encoded layer 28 expert 0, whose gate was K4
and up was K3.  The corrected values were:

| Tensor | Global scale | Source-relative RMSE |
|---|---:|---:|
| L28 E0 gate K4 | 1.14528307 | 0.0727482 |
| L28 E0 up K3 | 1.17992495 | 0.1435269 |

The same expert then rebuilt H2 from those corrected decoded gate/up weights
before down was encoded.

### Result and next hypothesis

The mean-one gate/up pilot is rejected and has no admissible KLD or weight-error
conclusion.  The smoke established that absolute gate/up profiles removed the
catastrophic scaling signature, permitting a corrected full pilot.

---

## Test 3 — corrected fresh BF16 SQG four-layer encode

### Question

Does a clean SQG encoding from official BF16 weights, using fresh calibration,
the corrected profile semantics, and candidate-specific down H2, produce a
valid same-rate four-layer treatment suitable for KLD testing?

### Frozen controls

- Same official BF16 tensors.
- Exact topology-neutral per-tensor K3/K4 map inherited from the 3.5-bpw MCG
  model: 384 K3 and 384 K4 tensors per selected layer.
- Same four layers: 6, 28, 52, and 77.
- SQG `sqg_xor_cheb_t12`, C128 tail-biting, dense-Hessian/BlockLDL binding.
- No MCG selected-layer transforms, scales, permutations, seeds, payloads, or
  decoded weights.

### Profile search and selection

The actual completed search was a **16-cell prefix per layer**, not the
originally contemplated 32 cells.  It evaluated four sign draws crossed with
four magnitude families: identity, aggregate RMS, BMMLaw quarter RMS, and
inverse-quarter RMS.  This was an explicit operator-time-priority decision and
must not be misreported as an eight-draw search.

Each cell performed real SQG encoding on a 16-expert panel chosen by fit mass,
decoded gate/up, rebuilt candidate-specific H2, and encoded down.  There was no
proxy pruning.  The selection set contained 892 documents/219,650 tokens.
Aggregate routed relative error and a Bonferroni paired document bootstrap
against draw-00/identity were used, with fallback to baseline if no candidate
had a significantly positive lower bound.

Selected cells were:

| Layer | Selected cell |
|---|---|
| 6 | `draw-00__identity` |
| 28 | `draw-02__identity` |
| 52 | `draw-03__identity` |
| 77 | `draw-00__identity` |

### Holdout status

The 967-document/229,475-token holdout was not used for calibration or profile
choice, but it was also **not scored**.  Each run-seal entry records
`status=skipped`, `reason=operator_time_priority`, and no routed holdout report.
The profile-selection result therefore lacks an independent held-out
confirmation.  Any prose implying that this run passed a holdout gate is
incorrect.

### Parallel encoding methodology

One expert was the indivisible scheduling unit: gate and up were encoded under
the same H13, decoded, H2 was rebuilt from their resulting SwiGLU path, and down
was then encoded.  Sixteen processes ran across four RTX PRO 6000 Blackwell
GPUs: four processes/GPU, 64 experts/process, and three CPU threads/process.
The final treatment encoded 1,024 experts and 3,072 tensors in approximately
15.5 minutes, sustaining about 70–72 experts/minute after startup.

Repeated full source/capture hashes were removed from the timed worker path
after one sealed preflight.  The mathematical encode, BF16-relative error
guards, packed closure, candidate-specific H2, expert artifacts, layer
assembly, and final run seal remained active.

### Validation result

- 3,072 SQG tensors and zero MCG tensors in the four selected layers.
- 1,536 K3 and 1,536 K4 tensors overall; 384 of each per layer.
- Every expert artifact recorded packed closure and BF16-source binding.
- Every layer was assembled and the run was sealed.
- Run-seal SHA-256:
  `2f973b57c4c77c98de1f6bff5518df529a683d82f1760c6ac3c7225a99e1a292`.

### Encoded size result

Because the exact rate map and tensor shapes were frozen, SQG did not reduce
selected-layer payload size.  Small metadata/marker differences made it 664
bytes larger across the four shards:

| Layer | MCG bytes | SQG bytes | Delta |
|---|---:|---:|---:|
| 6 | 4,231,302,560 | 4,231,302,720 | +160 |
| 28 | 4,231,304,864 | 4,231,305,040 | +176 |
| 52 | 4,231,304,888 | 4,231,305,048 | +160 |
| 77 | 4,231,304,864 | 4,231,305,032 | +168 |
| **Total** | **16,925,217,176** | **16,925,217,840** | **+664** |

The relative increase was about `0.000003923%`.  A size win would require a
different rate allocation, lower rate, or metadata/layout change; a same-rate
codebook substitution cannot be credited with one.

### Next hypothesis generated

The treatment was now clean enough for a final-logit test, but its runtime
required the SQG loader/non-fused dispatch.  A causal interpretation therefore
required an MCG checkpoint forced through the same native dispatch path.

---

## Test 4 — corrected four-layer final-logit KLD

### Methodology

- Candidate: clean BF16 SQG treatment from Test 3 in layers 6, 28, 52, and 77;
  all other model layers remained MCG.
- Reference: sealed BF16 logits and token IDs, scored as
  `KL(BF16 reference || candidate)` at 2,047 positions from one fixed
  2,048-token prompt.
- Runtime: exact same r33 base image ID as the preserved baseline, TP4/DCP4,
  A2A, interleave 64, FP8 KV, BF16 RoPE.
- Candidate runtime overlay: enabled for SQG loading and native/non-fused
  selected-layer execution.
- Five fresh candidate boots.  The r33 MCG baseline was reused, not rerun.

The r33 result is the proper baseline.  The older r26 mean `0.061282244905043234`
is a secondary historical result from a different image and must not be used
as the primary comparator.

### Results

```text
Candidate runs:
0.06169908118680389
0.06205119761221136
0.062187488789569624
0.06401166746517507
0.06430004636288912

candidate mean      = 0.0628498962833298
candidate sample SD = 0.001209723746733114
r33 MCG mean        = 0.0624498626218156
r33 MCG sample SD   = 0.0015327574926078513
candidate - r33     = +0.00040003366151420555
relative delta      = +0.6405677206%
```

The repeat-noise standard error of the difference was
`0.0008732441897379576`; Welch `t=0.4581005705` with 7.590 degrees of freedom.
Using the recorded conservative two-sided critical t of 2.776, the direction
was inconclusive.

### Validation and error recovery

- Every run produced a separately hashed, independently validated 2,047-value
  per-position KLD tensor.
- Reference logit SHA-256 was
  `87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063`.
- Reference token-ID SHA-256 was
  `ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56`.
- Summary SHA-256 was
  `85b41a89870702632c62d5f7edfd9df42a68a65250fcd78cf0ff89256410cacc`.

Boot 1 exposed seven tiny negative float32 KLD entries, minimum
`-4.6551978272191263e-08`.  The validator had treated all negative values as
impossible.  It was corrected to accept bounded floating-point roundoff down
to `-1e-7` while leaving the tensor unchanged and still rejecting lower values.
The accepted recovery receipt binds the raw output, log, dispatch proof, old
and corrected validator identities, and tensor.  No value was clamped.

### Holdouts and limitations

- The five boots repeat the same prompt.  Their variance describes runtime
  variation, not generalization across independent text.
- The four layers are separated, so the test is structurally weak for
  contiguous compounding or changed routing across a sequence of SQG layers.
- The candidate runtime code did not match the baseline runtime code.  The
  measured `+0.6406%` is the effect of **SQG weights plus required SQG
  loader/native dispatch**, not the isolated SQG codebook effect.
- This was W4A16.  It did not exercise E4M3 activation quantization or FP8 MMA.

### Valid conclusion

The observed combined path was slightly worse on average but statistically
inconclusive under repeat-run noise.  This test provides neither evidence that
SQG weights alone are worse nor evidence that they improve final-logit KLD.
The direct full-model expansion gate was therefore not passed.

### Next hypothesis generated

Two confounds needed separation:

1. force the unchanged MCG checkpoint through the same native selected-layer
   dispatch and compare paired per-position outputs; and
2. test whether layer-global H13 optimized the SQG weights for the wrong
   activation geometry, especially in early experts.

---

## Test 5 — native-dispatch MCG control

### Status

This control is fully specified in the repository, but no completed local
five-run native-control summary and paired analyzer output were found during
this audit.  It is a **planned mandatory control**, not a measured result.

An externally supplied observation claimed a 2.6–3.3% KLD movement from fused
versus non-fused MoE dispatch with codebook held fixed.  That figure is useful
as a warning that dispatch could be larger than the observed 0.64% SQG delta,
but it is not promoted here to a locally verified result.

### Preregistered design

- Use the protected unchanged MCG checkpoint read-only.
- Force exactly layers 6, 28, 52, and 77 through the same native dispatch class
  required by the SQG candidate.
- Preserve the fused budget as 45 actually fused MCG layers plus three reserved
  selected-layer slots, total 48.  Layer 77 was already non-fused.
- Require MCG-only markers, zero SQG dispatch lines, four selected-layer native
  MCG dispatch proofs, and exact reservation accounting.
- Use the exact same image, overlay, DCP4/A2A/interleave-64 regime, FP8 KV,
  prompt, BF16 reference, and KLD code.
- Run five fresh boots in a result root disjoint from both candidate and project
  roots.
- Pair and independently rehash all ten per-position tensors.  Report
  `candidate SQG - native MCG`, per-position quantiles and fraction improved,
  plus a circular-block bootstrap.  Scalar mean subtraction alone is not the
  causal analysis.

### What this test would and would not answer

It would isolate the four-layer SQG weight treatment from the selected-layer
dispatch change on the one fixed prompt.  It still would not test new documents,
full-model contiguous compounding, W4A8, or speed.

---

## Test 6a — raw packed-byte NMSE

### Question

At the exact same BF16 tensors and exact same K3/K4 rates, are the stored SQG
reconstructions geometrically closer to BF16 than the production MCG
reconstructions?

### Methodology

- Match all 3,072 gate, up, and down tensors across layers 6, 28, 52, and 77.
- Decode MCG entirely on CPU from production packed trellis bytes and stored
  FP16 scale vectors.
- Require each MCG decode to match its sealed production reconstruction
  SHA-256 byte-for-byte.
- Read SQG error from the independent packed-byte closure recorded during the
  clean BF16 encoding.
- Require identical official BF16 source binding and identical K3/K4 rate for
  every pair.
- Aggregate as `sum squared error / sum BF16 weight energy`, rather than giving
  every tensor equal macro weight.
- Run with no GPU visible.

### Result

| Group | Tensors | MCG NMSE | SQG NMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| Overall | 3,072 | 0.01593874402 | 0.01353123658 | **15.10475%** | 3,033 (98.7%) |
| K3 | 1,536 | 0.02577473051 | 0.02180891375 | 15.38645% | 1,504 |
| K4 | 1,536 | 0.006142337435 | 0.005286868681 | 13.92741% | 1,529 |
| Down | 1,024 | 0.008240417637 | 0.006907981213 | 16.16953% | 1,013 |
| Gate | 1,024 | 0.01822427758 | 0.01550568960 | 14.91740% | 1,010 |
| Up | 1,024 | 0.02156730274 | 0.01836553851 | 14.84546% | 1,010 |

By layer:

| Layer | MCG NMSE | SQG NMSE | SQG reduction | SQG wins / 768 |
|---|---:|---:|---:|---:|
| 6 | 0.01771224243 | 0.01556055110 | 12.14805% | 739 |
| 28 | 0.01539679504 | 0.01297328979 | 15.74032% | 767 |
| 52 | 0.01503056828 | 0.01277436628 | 15.01076% | 759 |
| 77 | 0.01555572863 | 0.01284144010 | 17.44880% | 768 |

Thirty-nine tensors favored MCG: 32 K3 and seven K4; 29 were in layer 6, one
in layer 28, nine in layer 52, and none in layer 77.

### Validation

The JSON and rendered report hashes are respectively:

```text
ac9cb98a5009c9bbe1b4abb6d3ccb345ab2dced65a88c829c8730eb22aa55461
cdf9e1d58a1f2f0943cf30e7bf1cb800459060459f9ae203e789ac3684e7f60d
```

### Interpretation

SQG is decisively better in ordinary same-rate weight space for this four-layer
sample.  This rejects “SQG simply reconstructs these BF16 matrices worse” as an
explanation for the KLD result.  It does not establish functional superiority:
raw NMSE has no Hessian sensitivity, routing, SwiGLU, cross-layer propagation,
activation quantization, or runtime dispatch.

### Next hypothesis generated

The mismatch between raw NMSE and KLD could arise because SQG minimized error
in low-sensitivity directions while misaligning error with the activation
geometry that reaches individual experts.  Hessian-weighted scoring was the
next diagnostic.

---

## Test 6b — Hessian-weighted packed distortion

### Question

Do the same packed MCG and SQG reconstructions still favor SQG when error is
weighted by captured gate/up and down activation geometry?  Does the answer
change between the layer-global H13 used for encoding and the exact expert-local
geometry on which each expert runs?

### Methodology and controls

- Decode MCG and SQG from packed bytes and map both back to the same official
  BF16 coordinate order.
- Hold source tensor and K3/K4 assignment fixed.
- Compute exact FP32 GPU traces.  For weight error `E`, source weight `W`, and
  covariance `H`, tensor HNMSE is proportional to
  `trace(E H E^T) / trace(W H W^T)`.
- Add the encoder's 0.025 diagonal damping to gate/up H13.
- Score gate/up under both:
  - the sealed fit-only layer-global H13 actually used by the current SQG
    encoder; and
  - exact fit-route expert-local `H13_e`, weighted by the square of the applied
    expert gate, with no shrinkage for this diagnostic scorer.
- Rebuild down H2 three ways from BF16, decoded-MCG, and decoded-SQG upstream
  gate/up paths.  For each H2, score both MCG and SQG down reconstructions under
  the same matrix.

The scorer reports both energy aggregation and route-mass aggregation.  The
latter gives experts weight according to their observed squared-gate mass and
is especially important for interpreting an MoE.

### Results: gate/up

| Gate/up metric | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| Layer-global H13 | 2,048 | 0.007224539694 | 0.004192353373 | **41.97065%** | 2,048 |
| Exact expert-local H13_e | 2,048 | 0.003995579165 | 0.003833471345 | 4.05718% | 760 |

Under expert-local H13_e, the median tensor SQG/MCG ratio was `1.03535`, so the
slight energy aggregate advantage was concentrated rather than broad.  The
route-mass aggregate changed sign: MCG `0.0032345191795` versus SQG
`0.003258208915`, making SQG **0.732404% worse**.

Expert-local H13_e by layer:

| Layer | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins / 512 | Route-mass direction |
|---|---:|---:|---:|---:|---|
| 6 | 0.001552285499 | 0.001988494657 | **-28.10109%** | 19 | SQG 30.8477% worse |
| 28 | 0.005128204096 | 0.005395821598 | **-5.21854%** | 131 | SQG 8.20139% worse |
| 52 | 0.007041901968 | 0.006665499856 | +5.34518% | 322 | SQG 1.28924% better |
| 77 | 0.002934506604 | 0.002813184095 | +4.13434% | 288 | SQG 1.37889% worse |

By rate under expert-local scoring, SQG was 4.21091% better for K3 and 2.81238%
better for K4 in the energy aggregate, but 0.61403% and 1.313% worse,
respectively, under route-mass aggregation.

### Results: down projection

SQG down weights were better under every common H2, for every one of the 1,024
down tensors:

| H2 upstream path | MCG HNMSE | SQG HNMSE | SQG reduction | Route-mass reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| BF16 | 0.006960821425 | 0.005939443275 | 14.67324% | 9.06496% | 1,024 |
| MCG | 0.006960210586 | 0.005948506077 | 14.53554% | 9.01312% | 1,024 |
| SQG | 0.006963213676 | 0.005936689209 | 14.74211% | 9.080% | 1,024 |

Using SQG-upstream H2, the energy reductions were 26.0163% in layer 6,
25.4717% in layer 28, 28.1115% in layer 52, and 14.4799% in layer 77, with SQG
winning all 256 down tensors in every layer.

### Validation and implementation corrections

- Maximum H2 route-mass relative error: `0`.
- Maximum local-alpha absolute error: `0`.
- Maximum identity-scale relative error: `7.17190921409665e-6`.
- JSON SHA-256:
  `d653c88c73661042ff2da68f1b44864546248701dfe5a00b7d3d5174a7f9c34a`.
- Report SHA-256:
  `70605074f267c32e01ddec52e9b8bc0b9aa7c732d0f6f435d4594c03e6eb0f66`.

Development defects corrected before the accepted aggregate included loading
BF16 projection triplets by their declared source shards rather than assuming
all three projections shared one shard, and unpacking compact trellis states on
CPU before GPU transfer so temporary int64 unpacking did not exhaust GPU
memory.  Per-layer checkpoints were produced independently and then merged;
pre-correction partial attempts are not reported as results.

### Correct interpretation

The current shared-H13 SQG encoding is extremely good under the exact objective
it optimized, and its down tensors remain better under all three tested
candidate-specific H2 constructions.  Its early gate/up advantage largely
vanishes or reverses when evaluated on the exact activation subset routed to
each expert.

This does **not** prove that layer-global H13 is intrinsically better than
expert-local calibration.  It shows an objective-alignment problem: the current
SQG candidate was optimized against global H13 and is therefore expected to
look strongest under global H13.  MCG was not re-encoded for this comparison,
and an existing shared-H13 SQG candidate scored under H13_e is not the same
experiment as an SQG candidate optimized under H13_e.

The layer-6 reversal is large enough that it could plausibly consume the down
projection's advantage and much of the ordinary-NMSE benefit in a complete
expert computation.  Whether it actually does so is the question for Test 7.

### Next hypothesis generated

Re-encoding gate/up against an expert-conditioned covariance, with a conservative
layer-global prior, should reduce routed early-layer gate/up error.  Down H2
must then be rebuilt from those newly decoded gate/up candidates.  The test
must compare complete expert functions on untouched holdout documents, not just
rescore the old weights under a new matrix.

---

## Test 7 — expert-local H13 with layer-global prior

### Hypothesis tested

The early-layer SQG gate/up labels are misaligned with the activation geometry
of the experts that actually use them because they were optimized under one
layer-global H13.  Re-encoding each expert against a support-shrunk local
covariance should improve routed gate/up and complete-expert distortion,
especially in layers 6 and 28, without changing rate allocation.

### Independent variable and frozen controls

The experiment freezes:

- official BF16 source tensors;
- each tensor's K3/K4 rate;
- selected profile cell and absolute/relative profile vectors;
- physical permutation and transform seed;
- SQG codebook, encoder, and C128 tail-biting contract.

The independent calibration change was gate/up H13.  For expert `e`, fit rows
routed to `e` were used to compute

```text
H13_local,e = sum_(t routed to e) g_te^2 x_t x_t^T / sum g_te^2
```

KQuant's weighted-OAS calculation is used as a reliability estimator.  The
scaled-identity blend returned by that helper is discarded; the relevant
coefficient blends the local covariance toward the valid shared-coordinate
prior:

```text
alpha_e = min(0.75, 1 - OAS_shrinkage_e)
H13_blend,e = (1 - alpha_e) H13_global + alpha_e H13_local,e
```

The final census was min/median/mean/max `alpha_e=0.75` across all 1,024
experts.  Although weighted OAS supplied the reliability calculation, every
expert hit the local-alpha cap.  The realized treatment was therefore a fixed
**75% local / 25% layer-global** covariance blend, not a support-varying
adaptive blend.  The experiment establishes the effect of this fixed blend; it
does not establish that 0.75 is optimal.

Gate and up share the same expert-conditioned H13.  After their new packed
weights are decoded, the encoder rebuilds that expert's candidate-specific H2
from the new SwiGLU path and re-encodes down.  Reusing the previous H2/down
would invalidate the experiment.

### Selection and holdout discipline

Only fit rows build the new H13.  Selection and holdout rows are excluded from
that matrix and from the corrected encoder.  However, the selected profile
cells were chosen in Test 3 using selection rows and are frozen here.  Thus:

- the holdout remains unseen and is a valid confirmation set for this H13
  change;
- the selection set is not used to fit the new H13, but it influenced the
  frozen profile choice in the predecessor experiment;
- this isolates the H13 change, but it does not discover the profile that would
  be optimal after expert-local recalibration.

All 1,024 expert manifests record `fit_only=true`,
`selection_used_for_encoding_calibration=false`, `holdout_used=false`, and
`holdout_unseen=true`.  This is consistent with the intended split discipline.

### Packed-byte treatment census

The corrected root contains 1,024 complete expert artifact triples: one JSON
manifest, one safetensors payload, and one manifest seal per expert.  The
independent census found:

| Property | Result |
|---|---:|
| Layers | 6, 28, 52, 77 |
| Experts | 1,024; 256 per layer |
| SQG tensors | 3,072; gate/up/down per expert |
| K3 tensors | 1,536 |
| K4 tensors | 1,536 |
| MCG source tensors used by corrected encoder | 0 |
| Codebook | 3,072 `sqg_xor_cheb_t12` |
| C128 tail-biting | 3,072 |
| Packed-state/repack/decoded closure passed | 3,072 |
| Unique tensor IDs | 3,072 |
| Unique corrected trellis-payload hashes | 3,072 |
| Corrected payload hashes identical to current SQG | 0 |

Thus every one of the 3,072 trellis payloads changed relative to the current
layer-global-H13 SQG treatment.  This was a real re-encode, not a rescore of the
old bytes.  The rate map, profiles, permutations, transform seeds, codebook,
BF16 sources, and tail-biting contract nevertheless remained frozen.  Down H2
was rebuilt from each newly decoded corrected gate/up path before down was
encoded.

### Evaluation methodology

The scorer decoded three packed-byte arms for all 1,024 experts:

1. production MCG;
2. current clean SQG with layer-global H13;
3. corrected SQG with expert-local/global-prior H13.

On fit, selection, and holdout routed rows, it computed squared-gate-weighted
NMSE for:

- gate output;
- up output; and
- complete expert output
  `SiLU(hidden @ gate) * (hidden @ up) @ down`.

For each expert/role, both source energy and candidate squared error were
weighted by the square of the exact applied router gate, then summed before
division.  The complete-expert path was evaluated as
`(SiLU(hidden @ gate) * (hidden @ up)) @ down`, so it includes all three
projections and the nonlinearity.  Raw-weight NMSE used the same BF16 source
tensors but no routing weights.

The primary preregistered diagnostic was **holdout complete-expert-function
NMSE**.  Fit is an optimization diagnostic; selection reveals behavior on the
profile-choice population; holdout is the clean confirmation for the new H13
change.

### Overall results

| Metric | MCG | Current SQG | Corrected SQG | Corrected vs current | Corrected vs MCG |
|---|---:|---:|---:|---:|---:|
| Raw weights, all projections | 0.01593874364 | 0.01353123630 | 0.01445579835 | **6.8328% worse** | **9.3040% better** |
| Fit routed gate/up | 0.003058657929 | 0.003115421806 | 0.001676869676 | 46.1752% better | 45.1763% better |
| Selection routed gate/up | 0.002996738617 | 0.003327406100 | 0.002970626946 | **10.7224% better** | 0.8713% better |
| Holdout routed gate/up | 0.003046226476 | 0.003330348199 | 0.002980227734 | **10.5130% better** | 2.1666% better |
| Fit complete expert | 0.003025458345 | 0.002707252775 | 0.002169658062 | 19.8576% better | 28.2866% better |
| Selection complete expert | 0.003036441909 | 0.002967203829 | 0.002786487874 | **6.0904% better** | **8.2318% better** |
| Holdout complete expert | 0.003043410360 | 0.002979759966 | 0.002801428997 | **5.9847% better** | **7.9510% better** |

The overall numbers were independently recomputed from the 1,024 records'
stored denominators and arm-specific numerators, rather than accepted only from
the rendered summary.  The recomputation agreed to displayed precision.

Fit improvements are much larger than selection/holdout improvements, which is
expected because fit routes constructed the covariance.  The important result
is that the direction remains favorable on the untouched holdout: 10.5130%
better gate/up error and 5.9847% better complete-expert error than current SQG.

The old layer-global-H13 SQG gate/up arm was 11.0342% worse than MCG on
selection and 9.3270% worse on holdout.  The corrected arm moved to 0.8713% and
2.1666% better than MCG, respectively.  This sign reversal on unseen holdout
rows is the clearest evidence that the prior SQG gate/up encoding was aligned
to the layer-global objective rather than to the activation geometry of routed
experts.

### Holdout complete-expert result by layer

| Layer | MCG | Current SQG | Corrected SQG | Corrected vs current | Corrected vs MCG |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.000808166647 | 0.000967821148 | 0.000832549221 | **13.9770% better** | **3.0170% worse** |
| 28 | 0.007827259835 | 0.008630394828 | 0.007552345493 | **12.4913% better** | 3.5123% better |
| 52 | 0.009509290160 | 0.01002041474 | 0.009332718129 | **6.8630% better** | 1.8568% better |
| 77 | 0.002909124303 | 0.002833240676 | 0.002665836380 | **5.9086% better** | 8.3629% better |

Every layer improved over the current SQG treatment on the holdout complete
expert function.  The largest corrections were in layers 6 and 28, exactly
where the expert-local Hessian diagnostic had found the strongest mismatch.
The correction did not fully beat MCG in layer 6: it recovered most of the
gap, but remained 3.0170% worse.  The other three layers beat MCG.

Holdout gate/up showed the same broad pattern:

| Layer | Corrected vs current SQG | Corrected vs MCG |
|---:|---:|---:|
| 6 | 30.7044% better | 1.0459% worse |
| 28 | 14.7051% better | 2.4194% better |
| 52 | 7.2260% better | 1.6286% better |
| 77 | 12.4188% better | 2.5625% better |

### Raw-weight tradeoff by projection

The functional improvement did not come from lowering ordinary weight NMSE
relative to the current SQG treatment:

| Projection | Corrected NMSE | Corrected vs current SQG | Corrected vs MCG |
|---|---:|---:|---:|
| Gate | 0.01675775188 | 8.0749% worse | 8.0471% better |
| Up | 0.01991117581 | 8.4160% worse | 7.6789% better |
| Down | 0.006908966251 | 0.0143% worse | 16.1576% better |

This is the expected diagnostic signature of objective realignment: corrected
gate/up weights moved slightly farther from BF16 in unweighted Euclidean space
while becoming substantially closer on the routed activation distribution.
The corrected aggregate raw NMSE still remained 9.3040% below MCG.  Down's
ordinary NMSE was effectively unchanged from current SQG, while the packed
payloads and candidate-derived H2 were rebuilt.

### Validation and final artifact status

- JSON schema: `glm52-expert-local-h13-sqg-comparison-v1`.
- Comparison JSON records: 1,024 unique `(layer, expert)` pairs; 256 per layer.
- Every record has positive fit, selection, and holdout routed-row counts.
- Alpha min/median/mean/max: exactly 0.75.
- Comparison JSON SHA-256:
  `95996daed6bc04900ddb2994f0207c7d2129942fcf3c86678da21178ccf53398`.
- Rendered report SHA-256:
  `eed3056bec1bc4d2b427c3fc85007b55332fdbd0037d1220c2d560d602a7c3d1`.

Every expert artifact was validated at creation, and four assembled final-layer
artifacts were produced.  The comparison scorer independently verifies packed
decode hashes for expert 0 in each layer and reads all other arms from their
validated packed artifacts; it should not be described as rehashing every
decoded tensor during the comparison pass.

The assembler initially emitted the final-layer JSON/safetensors/seal files as
`root:root` mode `0600`, which prevented an independent normal-user read.  Their
read permissions were normalized without changing content, after which all
four JSON sidecar hashes and run-seal bindings were checked:

| Layer | Layer-manifest SHA-256 | Layer-shard SHA-256 |
|---:|---|---|
| 6 | `cd5c59bb9e9e7819d1436cd1d67b19b7e160c44450bccbfe7ffec379fa1a858f` | `74037be2d76677ada4500d2b6691be24d08e2a2312ea1b597f6677e398ce329d` |
| 28 | `cf63b4e1905e1d276487931601b346cfbbd527075494c740cc2bf33edd467981` | `39094ed8d52dd3f22943ebd87a8fc6812a8e8e933a3f20c7363d015c0e0f55e7` |
| 52 | `bdc79a151b208c32c6a42bf6b0d92c3653f8fba7ead8183a2983a3154273eb49` | `a2965963bc9bc32e1b2c33b8a84a67c49a42515873ea55305cb06c1252f61b28` |
| 77 | `9cb1dc23a51ead0984f1dc3b1331cfa52b67043a7aa06ca3d842b526abb88085` | `73b99eeb6a6c9b2517a4a6547a34509bd14cf5a4d61acc3919e2bbec7eaf59ff` |

The complete four-layer seal records 1,024 experts, 3,072 SQG tensors, zero
MCG tensors, 1,536 K3, 1,536 K4, zero other rates, candidate-conditioned H2
rebuilt, expert-local/global-prior H13, zero fallbacks, official BF16 source,
and no legacy-MCG artifact reads.  Its SHA-256 is
`4f491e023e05b276fe1f1ff28be59c475e802c320389e6b3336c5240b18fa85a`;
its internal run-seal ID is
`66e4c66ac173a1a62502f43c7a800c7684b261afc3c0b7710eb3f938810684a3`.
At sealing time, the seal correctly recorded that no runnable model had yet
been materialized.  Test 7a subsequently materialized a separate candidate
from those exact sealed artifacts.

### Interpretation

This test **confirms routed-activation-geometry misalignment in the current
SQG gate/up treatment**.  That conclusion is supported by:

1. a single intended calibration change with rate/codebook/profile/transform
   controls frozen;
2. all 3,072 tensors being genuinely re-encoded;
3. large fit improvements that persist with the same sign on untouched
   holdout documents;
4. improvements over current SQG in every layer's holdout complete-expert
   function; and
5. raw Euclidean NMSE becoming worse while routed functional NMSE becomes
   better, which directly distinguishes geometry alignment from a generic
   easier reconstruction.

The result does not prove that fully local H13 is optimal, nor that layer-global
H13 has no useful regularizing role.  Every expert hit the 0.75 cap, so only the
fixed 75/25 blend was tested.  Layer 6 also remains slightly worse than MCG on
the complete holdout expert function.

### Interpretation limits

The scorer evaluates each routed expert separately and weights it by `g^2`.
It does not sum all eight weighted expert outputs into one layer output.  It
therefore excludes cross-expert error cancellation/reinforcement.  It also
excludes changed downstream routing, adjacent-layer compounding, final logits,
and runtime dispatch.  The favorable result supports a new model candidate and
controlled KLD test; it does not itself prove lower KLD.

### Next hypothesis and test design

The immediate next step was to materialize the now-sealed corrected four-layer
treatment, then run its final-logit KLD through the same SQG-native
selected-layer dispatch used by the current layer-global-H13 SQG candidate.
That candidate test is Test 7b.  The unchanged MCG checkpoint must still be run
through that dispatch class before the SQG-versus-MCG comparison is causal.
The current one-prompt endpoint tests whether the 5.9847% held-out
isolated-expert improvement survives top-8 summation, downstream routing,
four-layer interaction, and final-logit projection on that text.  A later
document-paired panel is still needed for text generalization.

In parallel, an alpha ablation such as 0, 0.25, 0.5, 0.75, and 1.0 on a
preregistered expert panel is warranted because the intended support-adaptive
rule collapsed to the cap for every expert.  That ablation should use holdout
complete-expert error, not fit error, to choose how much global prior to retain.

---

## Test 7a — runnable candidate materialization

### Question and construction contract

The materialization question was deliberately narrower than either encoding or
inference: can the four sealed Test 7 layer artifacts be installed into a
separate runnable checkpoint while preserving the exact treatment and leaving
the protected production checkpoint unchanged?

The output candidate is
`GLM-5.2-EXL3-TR3v4-3.5bpw-SQG-H13E-OAS-r1`.  Its construction contract was:

- protected source checkpoint:
  `GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED`;
- selected layers: 6, 28, 52, and 77;
- selected codebook: `sqg_xor_cheb_t12` for all 3,072 gate/up/down tensors;
- unselected layers: unchanged MCG;
- the only inherited selected-layer quantization control: the frozen
  per-tensor K3/K4 assignment, contract SHA-256
  `1fe5a065ef31c2e4c27589415b87bb77a91f095c55eaf4594807ca22645dab33`;
- no tensor override exceptions; and
- no MCG decoded weights, payload bytes, scale/transform vectors,
  permutations, seeds, or stale shared-side scales as construction inputs to
  the selected SQG treatment.

The materializer used an explicit 240-file runtime allowlist: 228 unchanged
source files were hard-linked, four selected SQG payload shards and the run
seal were copied, and seven loader/sidecar files were generated.  The
top-level `MANIFEST.json` and `.manifest_verified` completion marker were then
written in addition to that 240-file allowlist.  No recursive model-tree clone
was used.  The protected source was not mutated, and no production container
or model workload was launched during materialization.

### Census and provenance validation

Each selected layer closed at exactly 768 SQG trellis tensors: 384 K3, 384 K4,
zero K5, 768 SQG markers, and zero MCG markers.  Aggregate selected-layer
counts were therefore 3,072 trellis tensors, 1,536 K3, 1,536 K4, 3,072 SQG
markers, and zero MCG markers.  No SQG marker was present outside the four
selected layers, and none of the selected legacy MCG layer inodes was reused.
The materialization pass reported both selected-payload hash verification and
marker-payload verification as successful.

The materializer used its declared fast directional-test identity path.  The
current pass:

- validated the run-seal canonical ID and its bindings to each layer manifest,
  selection record, and declared shard;
- performed a metadata-mode teacher check and rehashed all non-payload teacher
  files;
- relied on the already sealed full teacher-identity receipt for the 343-GB
  source payload provenance;
- skipped a new full teacher-payload rehash and skipped a redundant pre-copy
  selected-layer payload rehash; and
- validated the copied SQG payloads after materialization.

This is sufficient for the time-priority directional test because the source
and Test 7 artifacts were already sealed, but it must not be described as a
new full 343-GB production verification.  The candidate manifest's embedded
`all_file_bytes_sha256_validated=true` teacher record refers to that prior full
sealed teacher validation; the contemporaneous materializer log explicitly
records `verification_mode=metadata`,
`full_teacher_rehash_performed=false`, and
`pre_copy_layer_payload_rehash_performed=false`.

### Materialized-result validation

- Candidate schema: `glm52-fresh-sqg-four-layer-candidate-v1`.
- Candidate manifest ID:
  `f2b0428603a0569eef9287d2a55c544ae892c86df797ab4d12d3cab94e3ac01f`.
- Candidate `MANIFEST.json` SHA-256:
  `50afcdbfd9c23f17a78fa8a268107b95f809a959ca05a5e87dbc6830e97714e7`.
- `.manifest_verified` SHA-256:
  `6781c6e1b2639f00129b81a3d00cc6c4d693c4504e1ba204074021b36c620373`;
  its content is the exact candidate-manifest digest above.
- Copied Test 7 run-seal SHA-256:
  `4f491e023e05b276fe1f1ff28be59c475e802c320389e6b3336c5240b18fa85a`.
- Internal run-seal ID:
  `66e4c66ac173a1a62502f43c7a800c7684b261afc3c0b7710eb3f938810684a3`.
- Candidate reports `complete=true`, `source_bytes_mutated=false`,
  `model_workload_launched=false`, and `production_container_touched=false`.

### Permission defect and recovery

The materializer-created files inherited root-only mode `0600`.  The exact 14
affected files were the four selected `.safetensors` shards, their four JSON
sidecars, rewritten `config.json`, rewritten `quantization_config.json`,
rewritten `model.safetensors.index.json`, `FRESH_SQG_RUN_SEAL.json`,
`MANIFEST.json`, and `.manifest_verified`.  The 228 inherited hardlinks were
not part of this repair.

Only those 14 files were changed to world-readable mode with `chmod a+r`; no
file contents, model bytes, or ownership were changed.  Afterward all 12
allowlisted copied/generated files still matched their manifest-declared
hashes, the manifest and run-seal retained the exact hashes above, and the
completion marker still contained the manifest digest.  No re-encode and no
rematerialization occurred.

### Result and limitation

The result is a complete, runnable four-layer H13e/OAS candidate with the exact
same selected SQG bytes evaluated in Test 7.  Materialization itself says
nothing about KLD, top-8 expert summation, changed downstream routing, runtime
speed, or W4A8; those are inference endpoints.

### Next hypothesis generated

If Test 7's routed-function improvement survives the full network, this
candidate should lower final-logit KLD relative to the current
layer-global-H13 SQG candidate under the same SQG runtime/dispatch path.  A
comparison with the preserved fused/native r33 MCG result remains
dispatch-confounded until Test 5 is completed.  Test 7b evaluated this
prediction and did not demonstrate the predicted KLD win on its one prompt.

---

## Test 7b — expert-local-H13 four-layer final-logit KLD

### Status and hypothesis

This test is complete: five fresh boots were accepted and a same-dispatch
per-position analysis was sealed.  The sample-mean direction was worse for
H13e, but it was small relative to repeat variation and the test did not
establish a directional KLD difference.

The preregistered hypothesis was that the 5.9847% reduction in the
document-disjoint, encoder-unseen but analysis-seen secondary-holdout isolated
complete-expert NMSE from Test 7 would survive the effects absent from that
proxy: router-weighted summation of the top-8 experts, cross-expert error
cancellation or reinforcement, downstream routing, propagation through the
rest of the model, and final-logit projection.

### Exact endpoint methodology

- Candidate: the materialized Test 7a checkpoint, with SQG only in layers 6,
  28, 52, and 77 and unchanged MCG in all unselected layers.
- Reference: saved BF16 logits of shape 2,047 by 154,880 from one fixed
  2,048-token WikiText-2-raw-v1 sequence.  Reference SHA-256 is
  `87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063`.
- Token stream: exactly 2,048 tokens; token-ID U32LE SHA-256 is
  `ad0d213eb533e956bc8e40a585f4fcff61a89a0da4a09cc5d750b46ba6d05c56`.
  The first 16 IDs are `284, 8396, 425, 10960, 465, 284, 14721, 8396, 425,
  10960, 465, 374, 458, 6364, 4531, 1154`.
- Metric: `KL(BF16 reference || candidate)` at each of 2,047 next-token
  positions, with the per-position tensor retained and independently
  validated before a boot can be accepted.
- Runtime: r33 base image ID
  `sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82`,
  SQG runtime overlay, TP4, DCP4 A2A, interleave 64, B12X sparse MLA, FP8 KV
  (`fp8_ds_mla` effective path), BF16 RoPE, eager execution, no prefix cache,
  maximum model length 2,560, one 2,048-token window, and one sequence.
- Weight/activation path: `B12X_MOE_FORCE_A16=1`; this is W4A16 and does not
  exercise the proposed W4A8/E4M3 activation or FP8-MMA path.
- Repetition: five fresh boots.  Each boot uses a new container, Python
  process, engine, workers, and model load.  A sealed compiled-code cache is
  copied to an independent per-boot cache snapshot; it contains executable
  cache artifacts, not model weights, logits, KV state, RNG state, or process
  state.

The WikiText alias first fails under intentional offline mode, after which the
harness resolves the sealed cached `Salesforce/wikitext`, configuration
`wikitext-2-raw-v1`, commit
`b08601e04326c79dfdd32d625aee71d232d685c3`.  Network fetching remains
disabled.

### Holdout and inference limits

This final-logit prompt is a fixed 2,047-position evaluation endpoint; all five
boots score the same text against the same BF16 reference.  Fresh boots
estimate runtime/numerical variation.  They do **not** create five text
holdouts and do not estimate generalization to other documents.  Test 7's
calibration holdout remains the proper unseen-document evidence for the H13
fit, while this endpoint asks whether that proxy improvement reaches final
logits on one fixed sequence.  A several-document paired evaluation is still
required before generalizing the KLD result across text.

The four SQG layers are separated rather than contiguous.  This includes their
interaction through the intervening model, but it remains a weak design for
estimating error compounding in a full SQG conversion or a contiguous block.

### Comparator hierarchy

The primary treatment-effect comparator is the current layer-global-H13 SQG
four-layer candidate from Test 4: mean KLD `0.0628498962833298`, sample SD
`0.001209723746733114`.  It used the same selected layers, saved reference,
token stream, r33 image, SQG runtime overlay, and native/non-fused selected-layer
dispatch.  Subject to the limitation that it is a preserved five-boot result
rather than a newly paired rerun, `H13e - current SQG` is the cleanest available
estimate of the H13 recalibration effect.

The r33 MCG mean `0.0624498626218156` with sample SD
`0.0015327574926078513` is important but uses the baseline's mixed
fused/native dispatch rather than the candidate's selected-layer SQG-native
dispatch.  Therefore `H13e SQG - r33 MCG` remains a combined
weight-plus-dispatch comparison.  Even if the new candidate beats that scalar
mean, the native-dispatch MCG control in Test 5 is still required to attribute
the difference to SQG weights.  The older r26 mean must not be used in either
comparison.

### Pre-inference error 1: sealed cache census

The first launch stopped before model inference with
`ERROR: sealed WikiText cache file census differs`.  The evidence manifest
expected five files but the source directory contained four: the zero-byte
Hugging Face `...b08601e..._builder.lock` was absent.  The exact empty file was
restored.  Its SHA-256 is the standard empty-file digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

After repair, the cache contains exactly five regular files and all five
entries pass `sha256sum -c`; the sealing manifest itself has SHA-256
`03677042f4c8c8acfe8025345eded6cee9ab8b497542a441075f6418d6f1d249`.
The failed output directory was moved to
`.quarantine-incomplete-run1-20260810T194812Z`.  It is an empty quarantine
sentinel and contributes no accepted result.  No candidate bytes were changed.

### Pre-inference error 2: candidate file permissions

The second launch passed the reference, evaluation-code, WikiText-cache,
baseline-evidence, runtime-overlay, exact-runtime-argument, and mount checks,
then stopped before inference when normal-user `sha256sum` could not read the
candidate `MANIFEST.json`.  This exposed Test 7a's root-owned `0600` files.

The permission-only 14-file repair described in Test 7a followed.  Candidate
manifest SHA-256 remained
`50afcdbfd9c23f17a78fa8a268107b95f809a959ca05a5e87dbc6830e97714e7`,
and run-seal SHA-256 remained
`4f491e023e05b276fe1f1ff28be59c475e802c320389e6b3336c5240b18fa85a`.
The failed directory was moved to
`.quarantine-incomplete-run1-20260810T194830Z`; it too is an empty quarantine
sentinel with no accepted result.  The candidate was neither re-encoded nor
rematerialized.

### Completion and validation

After the two pre-inference repairs, all five fresh boots completed.  Every
boot passed the sealed reference-logit and reference-manifest checks,
evaluation-script and validator hashes, five-file WikiText cache census and
hash checks, r33 baseline evidence, runtime overlay, exact runtime arguments
and mounts, GPU preflight, candidate manifest/run-seal bindings, selected-layer
census, and token identity.  The run manifest records 3,072 selected SQG
payloads, zero selected MCG payloads, and zero tensor overrides.

Each accepted boot has a dispatch proof containing four observations for each
required layer 6, 28, 52, and 77, corresponding to the four TP workers.  Each
boot also has a raw record, accepted record, independently generated
per-position validation report, log hash, and compiled-cache receipt.  The
five-run per-position evidence manifest has SHA-256
`f35697f9678d63708d469afc5682f3757a285bff72f7cbeb619893aa253b3048`.

The KLD runner used its declared fast identity path: its preflight binds the
candidate manifest, run seal, selected-layer counts, frozen rate contract, and
construction exclusions, but does not rehash every selected payload before
every boot.  The copied payloads were already verified in Test 7a.  This
distinction prevents the per-boot preflight from being misreported as a full
payload audit.

An independent audit rehashed all ten per-position tensors: the five H13e
runs and the five current-SQG comparator runs.  Every digest matched its sealed
summary.  Every file contained exactly one `float32` tensor named
`kld_ref_to_model`, shape `(2047,)`, with the expected metadata and only finite
values.  Tensor means closed to their summary values within `1e-8`.

Tiny negative entries were retained as floating-point roundoff, never clamped:
30 entries across the five current-SQG tensors and 31 across the five H13e
tensors.  The minima were respectively
`-6.225205595455918e-8` and `-6.04663199510469e-8`, both within the sealed
validator bound of `-1e-7`.

The paired analyzer was rerun deterministically from both sealed summaries and
all ten source tensors with 10,000 circular-block bootstrap iterations, block
size 32, and seed 20260810.  It reproduced every reported scalar, position
fraction, quantile, and interval exactly.  Its canonical analysis ID is
`23d4430d6d848ef856e1e3689f5c239e5c488eaa94259ff12a3dd7be8398542f`.
The source summaries and paired output have these SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Current layer-global-H13 SQG summary | `85b41a89870702632c62d5f7edfd9df42a68a65250fcd78cf0ff89256410cacc` |
| H13e SQG summary | `eaffe0747086351a4841fcd5ae4fab7c297879edc843463317701612710949f0` |
| Paired-analysis JSON | `59f3d26af46dfd2b4f52858d42579921611f7f74a769a61bd042d136310b0c5e` |
| Paired tensor | `c18e1163213ec69b8102da57af95aa0e52c18e865c1dc4a074faf5a5a8178f5a` |

### Five-boot scalar result

The accepted H13e KLD values were:

```text
0.06284865529813749
0.06407613381306661
0.06323600870761410
0.06275519930212074
0.06264933124007170

H13e mean       = 0.06311306567220212
H13e sample SD  = 0.0005821610692488878
current SQG mean      = 0.0628498962833298
current SQG sample SD = 0.001209723746733114
H13e - current SQG    = +0.0002631693888723169
relative delta        = +0.4187268467%
```

For the same-dispatch H13 comparison, the repeat-noise standard error of the
difference was `0.0006003903819947504`, Welch `t=0.43833045425870587`, and
Welch degrees of freedom `5.7583881473073175`.  Under the recorded conservative
two-sided critical value of 2.776, the direction is inconclusive.  The sample
mean is 0.4187% worse, but this experiment does not establish that H13e has a
nonzero adverse KLD effect.

Against the preserved r33 MCG mean, the H13e sample mean was
`+0.0006632030503865224`, or `+1.061976796%` worse.  Its repeat-noise standard
error was `0.000733247167290064`, with Welch `t=0.9044740709160722` and
`df=5.130535897349214`; that direction is also inconclusive.  More
importantly, this remains a dispatch-confounded comparison because the r33 MCG
baseline did not run the selected layers through the same SQG-native dispatch.

### Paired per-position result

The paired analysis first averages the five runs within each arm at each fixed
token position, then forms `H13e - current SQG`; lower is better.  It is paired
by position, not a claim that boot numbers are statistically paired.

| Statistic | H13e minus current SQG |
|---|---:|
| Fraction lower | 51.83194919% |
| Fraction higher | 48.16805081% |
| p05 | -0.05092686783 |
| p25 | -0.0002175499514 |
| median | -0.00000008947618042 |
| p75 | +0.0001298491770 |
| p95 | +0.04817620730 |
| 95% circular-block interval for within-prompt mean | [-0.003223015635, +0.003732587051] |

Thus H13e was lower at a small majority of positions and its median delta was
essentially zero, while its arithmetic mean was positive.  The sign of the
mean is therefore tail-driven rather than a broad position-wise shift.  As a
post-hoc descriptive audit, the most negative and positive mean-by-position
deltas were `-0.928411603` and `+1.524134159`; symmetrically trimming the 20
most negative and 20 most positive positions, approximately 1% from each tail,
changed the remaining mean to `-0.001274674`.  This tail diagnostic is not a
preregistered significance test, but it identifies the next failure mode to
localize.

The circular-block interval spans zero and describes correlated position
structure within this one prompt only.  It is not a confidence interval over
documents, tasks, or a model population.

### Interpretation

Test 7b rejects only the claim that this fixed 75% expert-local/25%
layer-global H13 treatment has **demonstrated** a final-logit KLD win over the
current SQG treatment on the tested prompt.  The observed same-dispatch mean
was 0.4187% worse and inconclusive under repeat noise; 51.83% of positions
nonetheless improved, with rare large position deltas controlling the mean.

It does not overturn Test 7's direct evidence that the old gate/up encodings
were misaligned with routed expert-local activation geometry.  Instead, it
shows that improving isolated `g^2`-weighted expert-function NMSE by 5.9847%
is not sufficient to guarantee better final logits.  The isolated metric omits
the signed top-8 expert sum, correlated errors between experts, routing changes,
downstream amplification, and token-specific tail sensitivity.

This test also does not establish SQG-versus-MCG causal quality.  Both SQG arms
share the relevant native/non-fused selected-layer dispatch, so their
comparison isolates the H13 treatment as cleanly as the preserved-run design
allows.  The r33 MCG comparison still mixes weight treatment and dispatch;
Test 5 remains mandatory.

### Next hypothesis and test design

The next mechanistic hypothesis is that a small set of tokens, routed expert
combinations, or downstream routing transitions amplifies the H13e treatment,
while modest improvements across most positions cancel in the final scalar.
The next diagnostic should replay the sealed 2,048-token sequence through both
SQG candidates and capture, at layers 6, 28, 52, and 77:

1. each expert's signed weighted output and the fully summed top-8 MoE output;
2. router IDs, probabilities, and route changes after each treated layer;
3. current-versus-H13e residual norms at the affected token positions; and
4. the connection between those residuals and the large positive final-KLD
   tail identified above.

That result should inform a preregistered alpha ablation over 0, 0.25, 0.5,
0.75, and 1.0.  Selection should use a signed, summed top-8 layer-output or
downstream-sensitive objective on selection documents, then be confirmed on
untouched documents.  The current fixed 0.75 value was imposed by a cap, not
selected as a KLD optimum.

After that mechanistic repair, use several document-paired prompts and then
contiguous layer blocks before extrapolating to a full-model conversion.  In
parallel, complete Test 5 so any SQG-versus-MCG statement is dispatch
controlled.

### Post-experiment operational incident

After all five KLD boots and the summary were complete, the protected
`glm-r33-sharedbf16` container was started in an attempt to restore the
pre-test state.  The user immediately instructed that it not be restarted, so
it was stopped.  Docker records show `StartedAt=2026-08-10T20:16:59.706Z`,
`FinishedAt=2026-08-10T20:17:46.697Z`, final status `Exited (137)`, and
`OOMKilled=false`.  A subsequent audit found the container stopped, no H13e
KLD process active, and low GPU memory use of 562/1156/562/562 MiB attributable
to the standing background processes.

This occurred after result publication, did not overlap an experiment run,
and did not affect any candidate, reference, per-position tensor, or summary.
It is recorded here as an operational correction, not as part of the Test 7b
methodology or result.  The protected container remains stopped.

---

## Test 8 — direct E4M3 endpoint and W4A8

### Status and what is actually known

The CPU-only direct weight-endpoint falsification, Test 8a, is complete on all
3,072 expert tensors in the late 74--77 block.  No GLM full-path W4A8 KLD or
speed result has been completed.  Every model KLD result in this record was
explicitly forced through A16, so Test 8a establishes a weight-operand fact,
not an activation-quantization or tensor-core performance result.

KQuant's local technical brief supports the architectural hypothesis that SQG
can use a finite E4M3 label menu and that W4A8 execution machinery exists.  Its
own expert panels are useful prior evidence, but they are not a GLM MCG-versus-
SQG endpoint test.  In particular, the quoted 3.859% figure must not be stated
as “MCG loses 3.859% when converted to E4M3”: it compared FP16-oriented SQG
search followed by E4M3 rounding with E4M3-aware SQG search.

### Precise hypothesis

If SQG's stored labels are already E4M3 values, those labels can reach the FP8
MMA weight operand without an additional lossy label projection.  In the
tested MCG counterfactual, each regularized FP16 lookup-table label is
RNE-rounded to E4M3 before applying the existing Hadamard and scales; the
final reconstructed tensor is deliberately not rounded.  SQG's unique
possible advantage is on this **label endpoint conversion**.  Both approaches
still quantize activations.

The activation points are not equivalent:

```text
h   -> gate/up FP8 MMA
act = SiLU(gate) * up -> down FP8 MMA
```

`h` is post-normalization and comparatively well behaved.  `act` is a product
and is much more heavy-tailed.  The Hadamard can help flatten both, but the
activation quantizers/clipping policies should be calibrated separately.

For exact SQG weight labels, input-side `suh` belongs on the transformed
activation before A8 quantization/MMA; output-side `svh` can be applied in the
FP32 epilogue.  The Hadamard must remain on the activation side.  Folding it
into the stored weights would move labels off the E4M3 grid and destroy the
claimed exact endpoint.

### Test 8a: completed CPU endpoint distortion

For identical sealed BF16 tensors and fixed rates, measure:

1. MCG decoded from its regularized FP16 lookup-table labels;
2. MCG decoded after RNE-rounding those labels to E4M3, before Hadamard/scales;
3. SQG at its A16 reconstruction endpoint;
4. SQG native E4M3 labels used directly.

The implementation decoded the packed MCG and SQG bytes for exactly the same
sealed BF16 tensor inventory.  It accumulated BF16-reference SSE, source
energy, MCG conversion-only SSE, the MCG error/conversion cross term, and SQG
endpoint closure in float64 chunks on CPU.  No weight was re-encoded and no GPU
was used.  Rates, layers, projections, and tensors were retained separately in
the report; the aggregate covers 38,654,705,664 scalar elements.

The measured result is:

| Endpoint | Aggregate NMSE |
|---|---:|
| MCG A16 reconstruction | `0.01532616894` |
| MCG with LUT labels RNE-rounded to E4M3 | `0.01601821470` |
| SQG A16 reconstruction | `0.01331171565` |
| SQG native E4M3 labels | `0.01331171565` |

MCG label-to-E4M3 conversion increased MCG's existing BF16 reconstruction error by
`4.515452%`.  The conversion-only energy was `4.610213%` of the original MCG
BF16 error energy; the small negative cross term accounts for the difference.
All 3,072 MCG tensors worsened.  Split by rate, the BF16 error increase was
`2.802317%` for K3 and `11.558140%` for K4.  SQG A16 and native-E4M3
reconstructions closed exactly for every tensor with zero conversion SSE.
After the endpoint conversion, SQG's raw NMSE was `16.8964%` lower than MCG's
E4M3 NMSE.  That full difference is not solely the endpoint benefit: SQG was
already `13.1439%` better than MCG at the A16 endpoint, so the `16.8964%`
includes encoder/codebook/calibration differences as well as MCG label
rounding.

This confirms the exact-E4M3 weight-label premise and falsifies the hypothesis
that MCG's endpoint rounding is negligible.  It does **not** show that the
same percentages transfer to Hessian-weighted error, final-logit KLD, or W4A8
throughput, and it does not remove the activation-side error shared by both
codebooks.  The result JSON/Markdown SHA256 values are respectively
`9514bec6d77aec9c14ee5bc3bb5da141be46272a426dcb7b0e3bac5d752a83b5`
and `ac820079f91d340ff1e470a3b7beb040c7bc0c43a2bb32351dcc70d573e09e68`.

### Planned test 8b: exact-path W4A8 calibration

For the real W4A8 candidate, down H2 must be captured from the complete
upstream W4A8 path: activation-side transforms and `suh`, E4M3 quantization of
`h`, native E4M3 SQG weights, FP32 accumulation/output transforms, and SwiGLU.
An H2 made from decoded SQG weights with BF16 activations is appropriate for a
W4A16 isolation but is not fully matched to W4A8.

### Planned test 8c: performance

Benchmark MCG and SQG decode separately and decode-plus-MMA at K3/K4 for C1
decode and several prefill tile sizes.  FP8 peak throughput alone does not prove
speed: at C1, the trellis weight bytes are unchanged and decoder ALU/occupancy
may dominate.  Prefill is the stronger W4A8 opportunity because weights are
reused across many tokens.

Only after endpoint distortion, activation-path KLD, and decode cost are
measured should W4A8 be described as a demonstrated GLM improvement.

---

## Test 9 — signed top-8, tail-constrained H13 blend ablation

### Status and hypothesis

The calibration, selection, encoder-unseen holdout, five-boot final-logit KLD,
and paired runtime-trace phases are complete. The candidate failed the strict
final-logit tail gate, so it is not yet a full-quant recipe. This test addresses
the specific Test 7b failure mode: the
fixed 0.75 treatment improved 51.83195% of final-logit positions, yet a small
positive tail made its mean KLD worse. The hypothesis is not merely that some
expert-local H13 component improves average isolated expert error. It is that
an intermediate global/local blend can retain that broad benefit without
increasing the signed top-8 error tail that downstream layers can amplify.

The earlier label “OAS blend” was too broad. The weighted-OAS helper produced
a reliability coefficient for each expert, but every expert hit the imposed
0.75 cap. Therefore Test 7 tested one fixed 75% local/25% global blend. Test 9
treats alpha as the independent ablation variable and records the uncapped OAS
recommendation only as diagnostic evidence.

### Frozen construction and independent variable

For expert `e`, the gate/up Hessian used for a candidate alpha is

```text
H13(alpha,e) = (1 - alpha) H13_global + alpha H13_local,e
```

where `H13_local,e` is built only from fit rows routed to expert `e`, weighted
by the square of the applied route gate. The coarse panel is alpha 0, 0.25,
0.50, 0.75, and 1.0. Alpha 0 is the existing layer-shared-H13 SQG arm. Alpha
0.75 is the already completed Test 7 arm. Alpha 0.25, 0.50, and 1.0 are new
packed-byte encodes, not rescoring of old bytes.

All other construction choices remain frozen:

- official BF16 gate/up/down tensors;
- the per-tensor 1,536 K3/1,536 K4 assignment, with no per-expert topology
  change;
- profile cells, physical permutations, transform seeds, SQG codebook, and
  C128 tail-biting;
- the same H13 for each expert's gate and up projections;
- candidate-specific H2 rebuilt from the decoded gate/up candidate and its
  resulting SwiGLU path before down encoding; and
- zero imported MCG packed bytes, transforms, scales, permutations, or seeds
  in every selected-layer SQG payload, except the frozen bit assignment that
  defines the rate-matched experimental topology.

The implementation accepts `--local-alpha` only in the closed interval
`[0,1]`, writes the exact coefficient and a coefficient-specific construction
identifier into every expert manifest, propagates it through layer assembly,
and binds it into the four-layer run seal. Omitting the option preserves the
original capped Test 7 behavior; it is not silently reinterpreted as one of
the fixed-alpha arms.

### Selection, holdout, and tail rule

The calibration corpus is unchanged: each layer contains 1,050,468 captured
rows split before this ablation into 601,343 fit, 219,650 selection, and
229,475 holdout rows. Fit rows alone construct every candidate H13. The coarse
alpha is chosen on selection rows. Only the shared-H13 baseline and the
selected alpha are then rescored on the existing holdout rows. Thus the
holdout does not directly choose alpha, a profile, a rate, a permutation, or a
stopping rule in Test 9.

However, this holdout must not be called globally untouched: Test 7 already
reported aggregate alpha-0/alpha-0.75 results on all of it, and that result
helped motivate the alpha ablation. It remains encoder-unseen and is useful as
a secondary confirmation, but it is analysis-seen. A truly blind confirmation
requires newly captured documents whose statistics have not been inspected.
The fixed WikiText KLD prompt is also heavily reused and is a directional
endpoint, not a blind model-selection set.

The selection scorer corrects the principal omission in Test 7. For every
captured token and layer it:

1. computes the BF16 complete-expert function for each of the exact routed
   top-8 experts;
2. decodes each candidate's actual packed SQG bytes and computes the complete
   gate/up/SwiGLU/down function;
3. applies the exact captured signed route gate to each candidate-minus-BF16
   expert-output vector;
4. sums all eight signed error vectors; and
5. squares only after the top-8 sum.

It also retains the sum of individually squared routed errors. Their ratio and
difference expose cross-expert cancellation or reinforcement instead of
assuming independent positive errors.

The winner is deliberately tail constrained. Relative to alpha 0, a
nonbaseline candidate is eligible only if all five conditions hold on the
selection aggregate across layers 6, 28, 52, and 77:

1. signed top-8 NMSE is no worse;
2. the upper-CVaR of the worst 1% per-position squared errors is no worse;
3. p99 per-position squared error is no worse;
4. the upper-CVaR of the worst 1% per-position relative errors is no worse;
5. p99 per-position relative error is no worse.

Among eligible candidates, selection first maximizes the fraction of improved
positions, then minimizes worst-1% CVaR, then signed top-8 NMSE. Therefore a
candidate cannot win by moving 51% or more positions slightly in the right
direction while making a few positions catastrophically worse. If no
nonbaseline arm passes all hard constraints, the report says so and produces
only a tail-first diagnostic fallback; it does not mislabel that fallback as
a validated improvement.

Before the coarse-panel results were inspected, a second, explicitly
exploratory refinement was registered: enumerate the 625 ways to select one
of the five already encoded coefficients independently for each of the four
layers. This introduces no new encoding and combines the exact saved
per-position layer error vectors. It uses the same five hard gates and the
same tail-first ranking. A layerwise winner can advance only through the
encoder-unseen holdout and final-logit KLD checks; it cannot be reported as a
preregistered single-alpha result. This refinement tests the plausible case
that conditional geometry changes with depth while retaining a
topology-neutral bit allocation.

The first implementation of that exploratory ranking exposed a materiality
failure: it selected alpha 0.50 only for layer 6 and reported 96.702% improved
positions, but aggregate NMSE changed by just `0.000032%`. That is a
numerical-scale pseudo-win and is not advanced. After observing this failure,
the exploratory layerwise rule was corrected to require at least a 0.1%
aggregate NMSE reduction in addition to the five original tail gates. This
post-hoc correction is recorded explicitly; it does not change the
preregistered single-alpha winner.

The corrected layerwise winner must also justify four coefficients instead of
one. Relative to the preregistered uniform winner, it advances only if all
four tail statistics remain no worse and it adds either at least 0.1% relative
NMSE improvement or at least one percentage point of position wins. This
parsimony gate was likewise added after detecting the pseudo-win and is
therefore exploratory, not retrospectively described as preregistered.

### Runtime tail localization

The final-logit confirmation retains every per-position KLD value and adds an
environment-gated eager-mode trace for the tested layers. Each TP/DCP rank
writes absolute token positions, layer input, router logits, exact top-8 IDs
and applied weights, routed MoE output, residual before MoE, and residual after
MoE. Files are joined by absolute position rather than rank or write order.

For a paired baseline/candidate run, KLD entry `i` is aligned with trace token
position `i`, the hidden state that produces the logits for token `i+1`. The
analysis reports route-set changes, union-aligned route-weight L1 drift,
hidden/MoE/residual relative deltas, their correlation with positive KLD
deltas, and the same metrics restricted to the worst 1% positive KLD tail.
This determines whether any remaining mean regression comes from routing
transitions, expert-output reinforcement, or residual amplification.

Tracing is off by default and changes no model bytes. The trace overlay is a
freshly sealed copy of the evaluated runtime overlay with only the model-level
instrumentation file replaced. Trace runs diagnose the mechanism; the normal
untraced multi-boot KLD result remains the quality endpoint.

### Known limits before results

- The signed top-8 selection scorer uses frozen captured hidden states and
  routes. It isolates each treated layer but cannot represent hidden-state or
  route changes caused by an earlier treated layer. The runtime trace and
  final-logit KLD confirmation cover that missing path.
- Layers 6, 28, 52, and 77 remain separated. This test selects the H13 blend;
  it does not answer contiguous error propagation.
- The fixed KLD prompt is one 2,048-token document. Per-position coverage is
  not document-level generalization.
- This remains W4A16. W4A8 activation error and FP8-MMA speed are separate
  experiments.
- The selection proxy is computed with high-precision matrix products to
  isolate encoded-weight/function error. It is not represented as a runtime
  kernel benchmark.

### Errors discovered and validation so far

The initial fixed-alpha implementation correctly changed expert encoding but
left the layer assembler and run sealer bound to the old layer-global codec
construction constant. That would have caused validation to reject valid new
manifests or encouraged a manual validation bypass. Both stages now bind the
exact coefficient-specific construction before validation. No completed
candidate was accepted under the stale binding.

Targeted tests cover fixed-alpha bounds and construction naming, the existing
calibration Hessian contract, fresh-pipeline behavior, signed-sum-before-square
tail statistics, and the evaluation overlay/runner contract. The blend runs
also require exactly 256 expert manifests per layer, candidate-conditioned H2,
zero MCG inputs, four assembled layer artifacts, and a coefficient-bound run
seal before they can be scored. The documentation was also corrected before
selection to distinguish the existing analysis-seen holdout from a genuinely
new document holdout.

### Selection result

All five packed-byte arms completed with 3,072 SQG tensors, zero MCG tensors,
candidate-conditioned H2, and zero fallbacks. On the 219,650 selection
positions, alpha 0.25 was the sole nonbaseline candidate to pass all five hard
constraints:

| Alpha | Signed top-8 NMSE | Positions improved | Absolute CVaR 1% | Relative CVaR 1% | Relative p99 | Eligible |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `2.539505205e-3` | baseline | `639.961555` | `2.240171991e-2` | `1.936763292e-2` | baseline |
| 0.25 | `2.373400202e-3` | `89.701343%` | `637.414567` | `2.191427300e-2` | `1.887691540e-2` | yes |
| 0.50 | `2.356708812e-3` | `89.288413%` | `640.274580` | `2.214374643e-2` | `1.901942424e-2` | no |
| 0.75 | `2.378155986e-3` | `82.537218%` | `648.009155` | `2.285086742e-2` | `1.961015985e-2` | no |
| 1.00 | `2.449385903e-3` | `57.228318%` | `634.870348` | `2.562192395e-2` | `2.189990710e-2` | no |

Relative to alpha 0, alpha 0.25 reduced aggregate signed-top-8 NMSE by
`6.540841%`, absolute p99 by `6.194459%`, relative worst-1% CVaR by
`2.175935%`, and relative p99 by `2.533699%`. Alpha 0.50 achieved a slightly
lower mean NMSE but failed the absolute-CVaR gate; this is exactly why the
tail rule precedes position-count or mean ranking.

The exploratory 625-arm layerwise search first exposed and then corrected a
position-count materiality bug as described above. After the 0.1% floor, its
best mapping was layer 6 alpha 1.0, layer 28 alpha 0.50, and layers 52/77 alpha
0.25. It improved NMSE by only `0.003498%` and position wins by only `0.011382`
percentage points over uniform alpha 0.25, while failing the no-worse-tail
parsimony gate. It is not advanced; uniform alpha 0.25 remains the selected
treatment.

The primary selection JSON SHA256 is
`4ca919ae24761079a4cff7739f15edcc78302b203ad610832c5c427f33752c97`.
The corrected exploratory layerwise JSON SHA256 is
`66942f115647eeb245db9d101d4165b981374fadbd15ea83027d31c4bf8aad64`.
The alpha-0.25/0.50/1.0 run-seal IDs are respectively
`e2a392e0f76b044f496b3c9bb2bd00be4623fe34ad1e365206366b01a2080883`,
`eb1d6fcc2b709965c5b51c6cf35ad5c774ea138854c25f001fd9de64f66ad68f`,
and `5a922311dc4532b898e14511c7311aec578ad2a8477aa6678feb44da58419438`.

### Encoder-unseen holdout confirmation

The frozen alpha-0.25 decision was then compared only with alpha 0 on all
229,475 holdout positions. Alpha 0.25 passed the same five hard constraints:

| Arm | Signed top-8 NMSE | Positions improved | Absolute CVaR 1% | Absolute p99 | Relative CVaR 1% | Relative p99 |
|---|---:|---:|---:|---:|---:|---:|
| alpha 0 | `2.548697629e-3` | baseline | `673.343734` | `84.688052` | `2.219782997e-2` | `1.967832881e-2` |
| alpha 0.25 | `2.385867558e-3` | `89.261140%` | `671.095963` | `77.659488` | `2.191765363e-2` | `1.934645514e-2` |

This is a `6.388756%` NMSE reduction, `0.333822%` absolute-CVaR reduction,
`8.299357%` absolute-p99 reduction, `1.262179%` relative-CVaR reduction, and
`1.686493%` relative-p99 reduction. The selection-to-holdout position win rate
changed by only `-0.440204` percentage points. The directional result is
therefore not confined to selection rows.

The holdout JSON SHA256 is
`36b0adec0a107f264defbc24612864df6a038e4231fbe4b7996d191b3d1052f3`.
As stated above, this split is encoder-unseen but analysis-seen from Test 7;
it is secondary confirmation rather than a newly blind corpus.

### Five-boot final-logit confirmation

Alpha 0 and alpha 0.25 were each evaluated in five fresh container/engine/
worker/model boots under the same r33 A16, FP8-KV, TP4/DCP4 regime. This did
not rerun the native-MCG control. The accepted values were:

| Arm | Accepted boot KLD values | Mean | Sample SD |
|---|---|---:|---:|
| alpha 0 | `0.0616990812`, `0.0620511976`, `0.0621874888`, `0.0640116675`, `0.0643000464` | `0.0628498963` | `0.0012097237` |
| alpha 0.25 | `0.0638940859`, `0.0607529618`, `0.0635429494`, `0.0617011705`, `0.0624172908` | `0.0624616917` | `0.0012962439` |

The selected blend reduced mean KLD by `0.0003882046`, or `0.617669%`, and
improved 1,065/2,047 (`52.027357%`) positions in the five-boot mean vector.
That direction is promising but not resolved against fresh-boot variation:
the independent-boot standard error is `0.0007929287`, Welch `t=-0.489583`
with approximately `7.96` degrees of freedom. A 10,000-iteration circular
block bootstrap over the fixed prompt, block size 32, gives
`[-0.003572644, 0.002770346]`; it also crosses zero and is not a document-
generalization interval.

Most importantly, alpha 0.25 failed the preregistered tail constraints:

| Arm | Mean KLD | KLD p99 | KLD p99.5 | Worst-1% KLD CVaR |
|---|---:|---:|---:|---:|
| alpha 0 | `0.062849897` | `1.079668074` | `1.569784798` | `1.838378917` |
| alpha 0.25 | `0.062461691` | `1.206657501` | `1.653692652` | `1.904047408` |

Thus the blend moved mean KLD in the desired direction while increasing p99
by `11.762%` and worst-1% CVaR by `3.572%`. Its positive per-position delta
mass (`22.457388`) nearly cancels its negative mass (`-23.252044`). The
correct conclusion is not “alpha 0.25 wins”; it is that expert-local H13 has a
credible favorable center effect but the harmful tail remains uncontrolled.

The final-logit JSON SHA256 is
`8fdbccd25f9b51110e3a3b7d9536114b5593d43d350df517fcb90d9cb09095df`.

### Runtime trace and negative control

One alpha-0 and one alpha-0.25 boot were then traced at layers 6, 28, 52, and
77. The DCP4 runtime emitted warmup calls and four exact rank replicas. The
analysis rejects repeated-position warmups, selects only the complete ordered
0--2047 invocation, and requires every selected tensor to be byte-numerically
identical across ranks before retaining rank 0. A test now covers BF16 loading,
complete-invocation selection, and exact DCP replica proof.

The direct alpha-0.25-minus-alpha-0 trace showed 51.539% position wins and a
mean KLD delta of `-0.003215336` in that one boot pair. Routing and residual
differences grew with depth:

| Layer | Route sets changed | Mean route L1 | Hidden relative delta | MoE-output relative delta | Residual-after relative delta |
|---:|---:|---:|---:|---:|---:|
| 6 | `19.541%` | `0.047123` | `0.000322` | `0.010851` | `0.000417` |
| 28 | `35.564%` | `0.088852` | `0.011558` | `0.039460` | `0.008547` |
| 52 | `41.133%` | `0.114272` | `0.024936` | `0.085438` | `0.021358` |
| 77 | `51.343%` | `0.133400` | `0.027029` | `0.022037` | `0.030197` |

However, layer-6 input and routing occur before the first changed expert
weights execute. Their nonzero difference is therefore a runtime negative-
control failure, not an H13 effect. A third trace boot reran the identical
alpha-0 checkpoint and found 20.225% layer-6 route-set changes, mean route L1
`0.047956`, and hidden relative delta `0.000346`—essentially the same as the
nominal treatment pair. Its final KLD changed by `+0.002725892`, comparable in
magnitude to the treatment pair's `-0.003215336`.

The post-MoE differences are larger than the same-checkpoint control, which is
consistent with a real treatment perturbation: at layer 6, MoE-output and
residual-after relative deltas were `0.010851`/`0.000417` for treatment versus
`0.007167`/`0.000287` for the control; at layer 77 they were
`0.022037`/`0.030197` versus `0.012750`/`0.020635`. But a single traced pair
cannot cleanly assign individual tail positions to the blend. The worst-1%
positive KLD-delta CVaR was `0.546969` for treatment and `0.519125` for the
same-checkpoint control, and only three of their worst 20 positions overlapped.
Correlations between positive KLD delta and traced route/output/residual
metrics were weak.

The trace therefore establishes two facts: errors and routing differences do
propagate and become substantial by layer 77, but the current fresh-boot
runtime variation is itself large enough to invalidate naive per-position
causal attribution. Contiguous-block tests must retain multi-boot final KLD
as the endpoint and use within-path candidate-aware calibration; trace metrics
remain mechanistic diagnostics with an explicit same-checkpoint control.

The treatment-trace and same-checkpoint-control JSON SHA256 values are
`6cb994a21eff72f9413553dd258b0dee1ac787abf661586c50804797f56ac528`
and `3e112aef36fcd30e59ac81c50422da53682968ffc40eca67b0c11d3d29c9edfb`.

### Test 9 decision

Uniform `local-alpha=0.25` is the best calibration/holdout blend tested and has a
favorable but noise-limited mean final-KLD direction. It does **not** pass the
final-logit tail gate and is not sufficient evidence for a full SQG quant.
“Best tested” is conditional on profiles, permutations, and transform choices
frozen from the layer-global-H search; nonzero-alpha arms did not receive
winner-native profile/rotation searches.  Any fleet recipe requires that
search to be repeated natively for the selected blend.
The later alpha-0.25 late-checkpoint null does not clear this separated-layer
failure: it is not a matched checkpoint null, and Test 9's harmful-tail
statistics exceed even that late p95 reference.  The comparison is retained
only as context, not as a formal recalibration of Test 9.
The next quality experiment is the planned early/middle/late contiguous-block
test, preserving `local-alpha=0.25` as the simple preregistered arm while measuring
candidate-conditioned downstream activations, routing, residual propagation,
and the positive KLD tail. W4A8 remains a separate endpoint and speed test.

## Test 10: preregistered contiguous-block propagation experiment

### Question and frozen blocks

Test 10 asks whether four adjacent SQG expert layers preserve the proxy gain
from Test 9 without allowing routing, residual, or positive final-logit KLD
errors to compound.  It is deliberately a propagation test, not another
search over isolated layers.  The three frozen blocks are:

- early: layers 10--13;
- middle: layers 38--41; and
- late: layers 74--77.

Layers 6--9 were rejected as the early block because layer 9 contains three
K5 tensors in the production allocation.  Substituting them or silently
retaining MCG would violate the topology-neutral all-K3/K4 test.  Every chosen
layer instead contains exactly 384 K3 and 384 K4 tensors, or 2,688 bit units
over its 768 expert tensors.

The reduced document plan was frozen before encoding.  It contains 217 whole
documents and 253,863 tokens: 131 documents/150,368 tokens for fit, 41/51,232
for selection, and 45/52,263 for holdout.  No document crosses a split.  The
plan SHA256 is
`2282bf5acbe6d094463113e9ed7a4ee05a5a9a484a011a01f6d8ce4b8da84afb`
and its canonical fingerprint is
`4caecdb1547937bbf999284b14d5778568df9052784ea8122c3da9b1bde07bfa`.

### Treatment and calibration contract

For each block, the unchanged r33 MCG checkpoint supplies the routed capture;
official pinned BF16 weights supply all 3,072 treatment tensors.  Preparation,
profile permutations, and expert-private downstream construction are rebuilt
for the block.  The frozen treatment is the Test 9 winner:

```text
H13_e = 0.75 * H13_layer + 0.25 * H13_local,e
```

Here `alpha=0.25` is explicitly the **expert-local** coefficient; it is not a
75% expert-local blend.  Local statistics remain route-probability weighted.
The down-projection H2 is rebuilt from the decoded gate/up candidate within
each expert, so the nonlinearity sees the actual upstream SQG candidate rather
than BF16 gate/up weights.  Expert-private rotation/permutation search is
retained, and the layer's production K3/K4 allocation is frozen rather than
reoptimized to favor SQG.

Every selected tensor must be newly encoded as SQG.  A block with any retained
MCG transform, code, or scale is invalid.  The existing model is never
modified in place; each block is materialized as a separate candidate.

### Endpoints and tail gate

The primary endpoint remains final-logit KLD against the sealed BF16 logits,
with repeated fresh boots and paired per-position vectors.  Report:

- mean KLD and repeat variation;
- percentage and count of positions improved;
- median, p95, p99, p99.5, and p99.9 per-position KLD;
- worst-1% positive-delta CVaR, total positive and negative delta mass, and
  maximum regression;
- route-set changes and route-probability L1 distance at every layer in the
  block; and
- hidden-state, signed router-weighted top-8 MoE output, and post-residual
  relative drift through the block.

Advancement requires more than a lower arithmetic mean: the positive tail may
not materially worsen, and the position win fraction must move in the desired
direction rather than relying on a small number of large improvements.  The
saved native-dispatch MCG control is reused; it is not rerun merely to produce
another baseline number.

### Holdouts and known limitation

Profile selection and H13 construction use only fit/selection rows.  The
document-disjoint holdout is report-only.  Final prompt logits are a separate
fixed endpoint but not a corpus-generalization guarantee.

The initial block capture is produced by the MCG teacher.  Consequently, H13
for layers after the first block layer is not a fixed-point recapture from the
partially converted SQG candidate.  The runtime KLD and trace still directly
measure actual contiguous error propagation, while candidate-specific H2 is
exact within each expert.  If a block improves the center but fails its tail
gate, the next hypothesis is candidate-aware recapture/re-encode for the
downstream block layers, not an immediate full-model quant.

The late block is run first because it reuses four already-sealed official
shards and directly tests the neighborhood of the prior layer-77 sample.
Middle and early follow under the same frozen rules.  W4A8 quality and speed
remain a separate test after A16 block quality is understood.

### Late-block execution record

The layers 74--77 capture completed from the unchanged r33 MCG teacher over
all 217 frozen documents.  DCP4 rank ownership was checked first on a one-
document smoke capture and then on the full run.  The full capture contains
253,863 aligned rows for each layer and is sealed under run UUID
`fa5b5504-e695-4fdf-bc17-601352b8a295`.  It is a 12 GiB capture.  The official
BF16 subset comprises 15 pinned shards and 80,429,109,824 apparent bytes; its
manifest SHA256 is
`f0e4659daf871e10148262fa5113d9c668817a3345ae7ccc4f92c7196827220e`
and its source-seal SHA256 is
`151c2bc9615691a9a0246e32085afe2e917246ff0bf037382f9d1245a6d8e7d1`.
Four shards were inode-identical hard links to the earlier official subset;
no model payload was reconstructed from MCG.

The smoke capture recorded the full teacher-identity mode, while the intended
full launcher requested the metadata-fast mode.  Rather than alter already-
sealed smoke evidence, the full capture used the original recorded launcher
bytes with a process-local environment override that retained full teacher
identity validation.  This paid one extra teacher hash pass but preserved the
smoke-to-full code binding.  Three preparation attempts then failed before
encoding work: the first omitted the exact EXL3 runtime digest, the second
mounted the wrong ExLlama package root, and the third supplied the SQG seal
directory where the seal JSON path was required.  Each was corrected in place;
the same preparation root was reused, and no completed layer work was
discarded.  Four CPU workers subsequently completed H13, permutation, and
profile-scale preparation for all layers, followed by the real absolute gate-
scale smoke.

Profile search encoded and scored all 16 preregistered cells per layer: four
draws crossed with identity, quarter-RMS, inverse-quarter-RMS, and aggregate-
RMS families.  Four workers shared each layer GPU, for 16 concurrent workers
over the four GPUs.  The frozen selections were:

| Layer | Selected profile cell | Selection SHA256 |
|---:|---|---|
| 74 | `draw-00__identity` | `50b74b115cd657a1884a6b17841709826058489a175ac0a654046dfd9ea9ba08` |
| 75 | `draw-00__identity` | `85825d7592cd78116d45c4f0b632b0e245a8921c6d64c8a59b4b9e58b82ac0f2` |
| 76 | `draw-00__identity` | `d57a874c0bd858e824ec450da38bafc2a3523e78f663d9eb46a7741f9579ab06` |
| 77 | `draw-03__identity` | `1f2104d2dc6803eabc0fd6ef1d5d28e33bdce2bb15dcd5eda9d76b465c426547` |

Layers 74--76 retained the identity baseline because no nonbaseline profile
passed the multiplicity-controlled paired-document improvement rule, even
where another cell had a lower point estimate.  Layer 77 selected draw-03
identity under that rule.  Holdout rows were not used in these choices.

The fixed-alpha encode then produced 256 expert artifacts and 768 SQG tensors
per layer, with the frozen 384 K3/384 K4 allocation.  Every expert manifest
records `local_alpha=0.25`, `global_alpha=0.75`, and fixed-alpha override true.
Down H2 was reconstructed from that expert's decoded gate/up SQG candidate.
The first launcher invocation contained a mistyped bit-contract digest and
failed at environment validation before loading a tensor or writing an expert
artifact.  The corrected invocation reused the same empty output root and did
not rerun profile search.  The completed run-seal SHA256 is
`567066231b882732d5bc82a093123002573a9cc8cf507ec763e2c2e7184606cf`.
Its census is 3,072 SQG tensors, 1,536 K3 and 1,536 K4, and zero MCG tensors.

The separately materialized runnable checkpoint reports 3,072 selected SQG
markers, zero selected MCG markers, exact packed-marker closure, and no source
mutation.  Its manifest SHA256 is
`5720c1aba18af0917d142f1f755cd03a8311591a638c80cf53d4f410791f30b6`;
the materialization-receipt SHA256 is
`b9fd238a90370d4ea0fae4dfbb150390ae23f71e5ed11c0a3db1f54d9b35f3ca`.
Unchanged files are hard links to the protected MCG checkpoint, while the four
selected layer payloads are independent copies of the sealed SQG assemblies.

### Reduced-capture representativeness check

Before interpreting the reduced 217-document search fleet-wide, layer 77 was
compared with the earlier 4,497-document capture.  The full/reduced fit row
counts were 601,343/150,368.  Their H13 diagonals correlated at `0.900610` and
the trace differed by only `-0.422682%`, but the relative Frobenius difference
was `0.503860`, cosine was `0.891940`, and effective rank fell `20.1385%`.
The profile selection also reversed: the earlier search selected draw-00
identity while the reduced search selected draw-03 identity.  This comparison
alone was not causal because the two searches used different expert panels,
permutations, profiles, and candidate bytes.

The required fixed-byte external cross-score was therefore run.  The exact
reduced-search draw-00 and draw-03 bytes, identical 16-expert panel, and exact
captured routes/gates were scored on full-capture documents excluded from all
roles of the 217-document plan.  On 851 selection-role documents/168,418 rows,
draw-03 reduced aggregate signed-routed error by `3.739074%`; the paired-
document 95% improvement interval was
`[6.181615e-5, 6.448072e-5]`.  On 922 holdout-role documents/177,212 rows, the
reduction was `3.787702%`, interval
`[6.264174e-5, 6.525134e-5]`.  Both lower bounds exceed zero.  Thus the matrix
difference and selection reversal remain real diagnostics, but the selected
reduced draw-03 candidate generalizes across 1,773 external documents.  This
clears the layer-77 reduced-selector gate; it does not prove every layer or
future profile family is equally representative.

The representativeness JSON/Markdown hashes are
`8c2c586e744f1aaef63c9892791e413f2ee10a9bbeee0cae034235ed62991009`
and `5a2b7e4973685bf0abc8febf7ac095294fe69a6397d33afdb33b3cf40f3ff811`.

### KLD launch failures, roundoff recovery, and validation

The first late-block KLD attempt failed at engine initialization because the
runtime argument file still asserted reserved layers `6,28,52` from the
separated-layer design.  It emitted no inference record and is preserved under
`excluded-pre-inference-run1-attempt1/`; it contributes zero accepted boots.
The runner now derives the reserved set from the selected layer list, which is
`none` for 74--77.

Run 3 of the corrected attempt completed inference but the validator rejected
one persisted per-position value of `-1.112351455e-7`.  The old decimal floor,
`-1e-7`, was smaller in magnitude than one float32 machine epsilon.  The
validator was corrected to accept at most two float32 epsilons of reduction
cancellation (`2.384185791015625e-7`) while retaining the negative value,
forbidding clamping, and rejecting anything below that floor.  Unit tests and
pinned source hashes were updated.  The already emitted run-3 tensor was
revalidated and sealed in place; no model, logits, or tensor bytes were
changed, and runs 1--2 were not restarted.

### Late 74--77 five-boot A16 final-logit result

The five accepted values were:

```text
0.06191653468813083
0.06259026905884817
0.06256102924107773
0.06104441014214946
0.06402017712969793
```

Their mean is `0.06242648405198083`, sample SD
`0.001090293791630798`.  The matching preserved r33 mean is
`0.0624498626218156`, so the scalar difference is `-0.00002337857` or
`-0.037436%`.  Welch SE is `0.0008411999`, `t=-0.02779`, approximately
`7.22` degrees of freedom.  This is parity/no detectable change, not a KLD
win.  Unlike the separated 6/28/52 treatment, layers 74--77 are outside the
r33 fused allowlist in both arms, so no fused-slot reservation change is
introduced at the selected layers.  More importantly for this phase, there is
no large adverse scalar-KLD signal from four adjacent late layers.  The
summary SHA256 is
`50648c13e4b52bfb873e45aee0bba87e8956d53e57bf725dc23b438fccfb279b`;
the five per-position evidence manifest SHA256 is
`0bc54087a0583accbe26477aff04e0f850fd021dfc025b921678745b71a0772e`.

### Late 74--77 document-holdout signed top-8 result

The block was then scored against MCG on 52,263 holdout positions using the
exact top-8 expert IDs and applied gates.  Expert outputs were signed and
router-weighted, summed across all eight experts, and only then squared.  This
prevents a per-expert shortcut from hiding cancellation or reinforcement.

MCG signed-top-8 NMSE was `0.00683407168`; SQG `local-alpha=0.25` was
`0.00702340797`, `2.7705%` worse.  SQG improved 20,398 positions
(`39.0295%`) and worsened 31,865 (`60.9705%`).  Every individual layer was
worse in aggregate NMSE.  The relative-error p99 worsened from `0.02429805`
to `0.02480940`, while relative worst-1% CVaR improved slightly from
`0.02735032` to `0.02725573`.  Squared-error p99 worsened from `196.7451` to
`200.6144`, whereas squared-error CVaR improved from `646.3884` to `587.4295`.
No nonbaseline arm passed all hard constraints; MCG was retained.

The already-recorded cross terms rule out the aggregate cross-expert
interaction change as the main explanation.  SQG's total signed-top-8 SSE
exceeded MCG by
`59,657.7573`.  Of that regression, `58,309.3349` (`97.7397%`) was already
present in the sum of the eight individual routed-expert SSE values; only
`1,348.4224` (`2.2603%`) came from the change in cross-expert error terms.  The
summed-over-individual SSE ratio moved only from `1.0013385` for MCG to
`1.0019137` for SQG.  Thus the late `local-alpha=0.25` deficit is primarily an
individual expert calibration/encoding deficit, not individually superior
experts becoming worse only when their signed outputs are summed.

This creates a useful tension rather than a contradiction: the late block is
at final-logit scalar-KLD parity with no obvious scalar catastrophe, but the
five-boot endpoint did not produce a sealed per-position late-SQG-versus-r33
tail comparison against a matched null.  Tail advancement therefore remains
open.  Its frozen-hidden-state routed function proxy says
`local-alpha=0.25` is not the best late-block calibration.  The next
hypothesis is that downstream
residual/route cancellation hides a broad small proxy regression, or that the
separated-layer alpha did not transfer to late expert-local activation
geometry.  The paired routing/residual trace and a late-specific H13 blend
ablation are therefore required before middle/early construction or a full
quant.  The signed-top-8 JSON/Markdown hashes are
`a56837b4cd619b8eed8aa3f69e34b5e6b7c5caf2bbef4aee2c6a3486f76d3839`
and `9c45a4930d0b8ca05887a42e7baa04dafe7754fe69026538bf28d7d6eac6b2a8`.

### Statistical power and the role of four-layer blocks

Four of 75 routed MoE layers are 5.33% of the treatment surface.  Using the
observed late/r33 boot SDs, a two-sided 5% test with 80% power would require
approximately 309 boots per arm for an absolute effect of `0.0003`, or 391
per arm for a `0.4267%` relative effect.  Five boots cannot adjudicate a
realistic four-layer codebook improvement; they can screen catastrophes and
propagation failures.  Conversely, a half-model effect near 3% would require
roughly eight boots per arm under the same variance estimate.  These are
planning calculations, not evidence that effects scale linearly.

Accordingly, “late block passed” means no observed scalar compounding blowup
and no detectable scalar harm.  Its tail gate remains unclosed.  It does not
mean SQG lowers KLD.  A controlled KLD quality claim needs much greater
treatment scale, many more boots, or both.

### Completed same-checkpoint null and trace follow-up

A second independent five-boot set using the same candidate path,
selected-treatment manifest, and runtime identity completed at mean KLD
`0.0621148804` with sample SD `0.0016110603`.  Its five
accepted values were `0.0633416218`, `0.0606491331`, `0.0642350373`,
`0.0616561360`, and `0.0606924740`.  All five 2,047-position float32 vectors
passed independent metadata, hash, shape, finiteness, roundoff, and scalar
closure validation.  No candidate bytes, runtime inputs, or evidence vectors
were changed.

The analysis pooled those five boots with the original five boots of the same
assigned candidate/runtime arm and enumerated all 126 unique balanced 5-vs-5 partitions in
both directions, producing 252 directional null comparisons.  The arbitrary
original-minus-new group mean was `+0.0003116039`, despite there being no
treatment difference.  Across the null partitions, the 95% two-sided absolute
mean-delta envelope was `0.0013866981`; the 95% interval for the fraction of
positions improved by the arbitrarily named left arm was `[0.476087,
0.510015]`.  The directional 95th-percentile gates were `0.14460577` for
positive-delta p99, `0.26096641` for positive-delta worst-1% CVaR,
`12.0395564` for total positive-delta mass, and `0.81391150` for maximum
single-position regression.

The null result establishes that the former zero-tolerance p99/CVaR rule was a
false-negative risk under this fixed prompt and FP8-KV runtime: boots assigned
to the same candidate path and selected-treatment manifest produce tail
movements far larger than the expected four-layer mean effect.  Future arms
retain the raw preregistered metrics and compare directional harm with this
checkpoint/prompt-specific empirical p95 reference.  This is not a formal
false-positive-rate estimate: the 252 directional partitions reuse only ten
boots and are not independent experiments.  It does not generalize
automatically to another runtime and does not increase four-layer statistical
power.  The identity record binds the same candidate path,
selected-treatment manifest, and runtime identity, but its
`candidate_manifest_sha256` and `run_seal_sha256` fields are null; it does not
independently prove full-checkpoint byte identity.  The JSON and
Markdown SHA256 values are
`b3c72b2e4b4e9d4cfe9c0af1951f30599a34543091ebfd60124bac71a4daf582`
and `619bf45c052b4ad937af4e404cb838f7191ca06b991763ade26259462027e57f`.

The paired trace runner recorded hidden state, exact top-8 IDs/weights, signed
weighted MoE output, residual-before, and residual-after for all 2,047
positions at layers 74--77.  The r33 and SQG traces each required exact DCP4
replica closure.  The trace instrumentation emitted two full-size buffers and
one four-row buffer per layer/rank; only invocation 3 carried exact ordered
positions `0..2047`.  Invocation 0 was a warmup buffer whose positions were
all zero and was correctly rejected as a within-boot repeat.  The post-capture
validator was repaired to require complete layer/rank coverage rather than an
incorrect one-file-per-rank count; the completed 4.6 GiB r33 trace was retained
and never rerun for that repair.

The raw r33-versus-SQG diagnostic showed route-set changes of `43.234%`,
`53.981%`, `53.737%`, and `48.461%` at layers 74--77.  Crucially, the input to
the first treated layer was already different: layer-74 hidden relative error
was `0.0211652` and route-set churn was `43.234%` before SQG weights could
cause upstream drift.  This made causal attribution from the single pair
invalid.  One additional unchanged-r33 trace boot was therefore collected as
a trace null, not as a replacement five-boot dispatch control.  The r33-r33
pair changed route sets at `45.774%`, `49.829%`, `47.533%`, and `44.895%`, with
layer-74 hidden relative error `0.0235228`.  Most apparent route churn is
therefore runtime variation.

Against that one-pair trace null, the cross-arm pair shows an excess
MoE-output signal consistent with a treatment effect, but it is not a causal
estimate and has no sampling distribution.  Cross-arm versus null mean
relative MoE-output drift was
`0.063609/0.036958`, `0.064927/0.039499`, `0.031120/0.018356`, and
`0.019232/0.012887` across layers 74--77, or approximately `1.49--1.72x` the
same-checkpoint pair.  Post-residual drift ratios were much smaller and rose
gradually: `1.04x`, `1.15x`, `1.22x`, and `1.29x`.  There is an observed
treatment-consistent MoE perturbation and no blowup in this pair, but one
cross-arm pair and one r33-r33 pair cannot prove that adverse compounding is
absent.
Pearson and Spearman correlations between positive final-KLD delta and route
L1/MoE/residual drift remained close to zero; the worst final-KLD positions
cannot be causally localized from this one prompt/pair.

The cross-arm trace JSON/NPZ/Markdown SHA256 values are
`b3244ad9b8a0691bda2a75063575b9520b29cbef6a68db85c8ee2d9a66e79f9f`,
`7059cbd049544229e847269414a997f4db171834e9e4098c473ae35ac5fd9f19`,
and `0e4ce04598f6eef441941123ed040f56a16b16330a857a44235b3ead945624e3`.
The r33-r33 trace-null hashes are
`2bb3ee8860b6209e8ac1342e09d1860c67a36930b8b5f7f2846776d498faed69`,
`9216ee1b38e56f35040639f2fdfadb56322e4e657de0112d2620e0e3acb7d64c`,
and `e61e115e41aef17d73a7eb68cfe1ba2ea8d3403b976623b7fe8b81d8eec04d75`.

The raw trace roots were deleted after these derived JSON/NPZ/Markdown
artifacts were hash-closed to recover local disk space.  The derived evidence
remains available, but raw DCP-replica and source-trace validation cannot be
rerun from the current workspace.

For future endpoint work, the ordinary mean remains reported, while a frozen
symmetrically trimmed paired mean should become co-primary with the
null-calibrated tails.  A median alone is too insensitive because most
per-position KLD values lie near zero.  The larger power improvement is a
document-paired multi-prompt endpoint: more independent text averages the
heavy tail instead of repeatedly booting one 2,047-token passage.

Layers 74--77 are outside r33's fused allowlist and were naturally dispatch
matched.  Middle layers 38--41 are expected to fall inside the fused baseline
region; their preregistration must include a block-specific native-MCG dispatch
control before interpreting an SQG comparison.

## Cumulative findings

### Accepted findings

1. A mechanical MCG-reconstruction-to-SQG requantization was materially worse
   in the runtime-confounded four-layer KLD test; it is superseded as an SQG
   quality estimate.
2. A clean same-rate SQG encoding exists for all 3,072 selected tensors, with
   zero selected-layer MCG encodings and effectively identical size.
3. SQG is 15.10475% better in aggregate raw weight NMSE and wins 98.7% of the
   matched tensors.
4. SQG is 41.97065% better for gate/up under the layer-global H13 it optimized.
5. That gate/up advantage nearly disappears under exact expert-local geometry
   and reverses under routed-mass aggregation, driven especially by layers 6
   and 28.
6. SQG down weights are robustly better under BF16-, MCG-, and SQG-upstream H2,
   winning all 1,024 down tensors in each comparison.
7. Re-encoding under a fixed 75% expert-local/25% layer-global H13 blend
   improved document-disjoint, encoder-unseen but analysis-seen secondary-
   holdout routed gate/up NMSE by 10.5130% and complete
   expert-function NMSE by 5.9847% relative to the current SQG treatment.  All
   four layers improved on the complete-expert metric.
8. The expert-local correction made raw weight NMSE 6.8328% worse than current
   SQG while improving routed function error, directly confirming that the old
   encoding was misaligned with routed activation geometry rather than merely
   suffering higher Euclidean reconstruction error.
9. Corrected SQG beat MCG by 7.9510% on aggregate holdout complete-expert NMSE,
   although layer 6 remained 3.0170% worse than MCG.
10. The earlier corrected four-layer final-logit candidate was 0.64057% worse
    in mean KLD, but the direction was inconclusive and runtime dispatch was
    confounded.
11. The expert-local-H13 treatment has been materialized as a separate runnable
    candidate with all 3,072 selected tensors SQG, the frozen 1,536/1,536 K3/K4
    split, zero selected MCG state, and sealed source/run provenance.
12. Two attempted KLD launches failed before inference because of a missing
    zero-byte sealed cache lock and root-only materialized-file permissions.
    Both were repaired without re-encoding or rematerializing, and neither
    attempt contributes a result.
13. The expert-local-H13 candidate completed five accepted KLD boots with mean
    `0.06311306567220212` and sample SD `0.0005821610692488878`.  Against the
    same-dispatch current SQG result, its sample mean was 0.41873% worse, but
    Welch repeat-noise analysis was inconclusive.
14. H13e lowered KLD at 51.83195% of fixed-prompt positions and had an
    essentially zero median delta, yet rare positive position deltas drove the
    arithmetic mean higher.  The within-prompt circular-block interval also
    crossed zero.
15. Against the preserved r33 MCG scalar mean, H13e was 1.06198% worse and
    repeat-noise inconclusive; that comparison remains dispatch-confounded and
    is not a causal SQG-versus-MCG result.
16. The CPU Test 8a endpoint falsification covered all 3,072 late-block
    tensors.  RNE-rounding MCG regularized LUT labels to E4M3 before the
    existing Hadamard/scales increased BF16 error energy by 4.51545%
    overall, 2.80232% at K3 and 11.55814% at K4; all 3,072 tensors worsened.
17. SQG native E4M3 labels were exactly identical to the SQG A16 endpoint for
    all 3,072 tensors, with zero conversion SSE.  SQG E4M3 raw NMSE was
    16.8964% lower than MCG E4M3 raw NMSE on this inventory, but SQG was
    already 13.1439% better at A16; the full gap is not solely an endpoint
    conversion benefit.
18. The reduced layer-77 profile candidate selected on 217 documents beat the
    fixed draw-00 bytes by 3.7391%/3.7877% on 851 external selection-role and
    922 external holdout-role documents; both paired-document 95% intervals
    excluded zero.
19. The late 74--77 block completed five accepted A16 KLD boots at mean
    `0.0624264841`, only `-0.03744%` versus r33 and repeat-noise inconclusive.
    This found no obvious scalar catastrophe; its matched per-position tail
    gate remains unclosed, and it is not proof of lower KLD.
20. The same late `local-alpha=0.25` block was 2.7705% worse than MCG on the exact
    signed weighted top-8 holdout proxy and improved only 39.0295% of
    positions.  It is therefore not yet the final calibration recipe even
    though final-logit KLD remained at parity.
21. The late proxy regression is 97.7397% individual routed-expert SSE and
    only 2.2603% change in the aggregate cross-expert interaction term, so
    that interaction change is not its primary cause.
22. A same-checkpoint trace null reproduces 44.9--49.8% route-set churn and
    about 2% hidden/residual relative drift.  One cross-arm pair shows a
    treatment-consistent 1.49--1.72x MoE-output drift relative to one r33-r33
    pair and no residual blowup in that pair; it is not a causal estimate or
    proof that adverse compounding is absent.

### Findings that are not established

- Pure layer-global versus pure expert-local H13 has not been tested.  Test 7
  establishes that a fixed 75% local/25% global blend beats the current
  layer-global-H13 SQG treatment on isolated routed expert functions.
- Fully local H13 is not proven optimal.  The completed re-encode tested only a
  fixed 75% local/25% global blend because every expert hit the alpha cap.
- Test 7b did not demonstrate a final-logit KLD win for the fixed 75/25 H13e
  treatment on its one prompt.  Its inconclusive 0.4187% worse sample mean does
  not prove that expert-local geometry is harmful or that another shrinkage
  level cannot improve KLD.
- SQG is not proven better or worse than MCG in controlled final-logit KLD.
- The external 2.6–3.3% dispatch observation is not a completed local control.
- The late contiguous block shows no obvious scalar-KLD catastrophe, but its
  matched per-position tail gate is unclosed, and one four-layer block cannot
  predict a full-model conversion or prove a small favorable effect.
- SQG has not reduced model size at the frozen map.
- Test 8a does not show that the exact 4.51545% MCG endpoint energy increase
  transfers to Hessian-weighted error or model KLD.
- W4A8 has not yet shown a GLM activation-quality, KLD, or speed win.
- The alpha-0.25 blend selected on separated layers is not established as the
  correct late-block blend; its late signed-top-8 holdout result is adverse.
- The completed same-checkpoint null is an empirical reference from 252
  dependent balanced partitions of ten boots, not a formal false-positive
  calibration, and its identity record does not independently bind full
  checkpoint bytes.
- The completed paired trace and trace control are one pair each and do not
  supply a sampling distribution.

## Recommended experiment order from here

1. Finish the active late-specific H13 blend ablation under the signed summed
   top-8 objective, with exact MCG comparison.  Rebuild candidate-conditioned
   down H2 for every blend and keep every treatment layer fully SQG.
2. Freeze the selected mapping on the document-disjoint, encoder-unseen but
   analysis-seen secondary holdout, then rerun profile/rotation search natively
   for the selected blend and require a genuinely new corpus for blind
   confirmation.
3. If a late blend reverses the broad proxy regression without crossing the
   null-calibrated tail envelope, repeat the contiguous screen on middle and
   early blocks.  Otherwise try downstream fixed-point recapture before
   spending on more blocks.
4. Scale the treatment to roughly half the routed layers before asking whether
   SQG lowers KLD; four-layer five-boot endpoints are catastrophe screens.
5. Benchmark K3/K4 MCG and SQG decoder cost, then run exact-path W4A8 with
   separate `h` and heavy-tailed `act` activation calibration and H2 captured
   from the true upstream A8 path.
6. Greenlight a full BF16 SQG quant only if scaled A16 quality holds and the
   W4A8 prefill quality/speed result justifies a deployable runtime.

## Artifact and source index

### Primary reports and sealed results

- [Raw encoded NMSE report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/raw_encoded_nmse_sqg_vs_mcg.md)
- [Raw encoded NMSE JSON](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/raw_encoded_nmse_sqg_vs_mcg.json)
- [Hessian-weighted NMSE report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/hessian_weighted_nmse_sqg_vs_mcg.md)
- [Hessian-weighted NMSE JSON](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/hessian_weighted_nmse_sqg_vs_mcg.json)
- [Expert-local H13 routed-function report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/recalibrated_sqg_vs_mcg.md)
- [Expert-local H13 routed-function JSON](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/recalibrated_sqg_vs_mcg.json)
- [Expert-local H13 four-layer run seal](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-expert-h13-oas-r1/run_seal.json)
- [Expert-local H13 runnable-candidate manifest](/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-SQG-H13E-OAS-r1/MANIFEST.json)
- [Runnable-candidate verified marker](/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-SQG-H13E-OAS-r1/.manifest_verified)
- [Candidate materialization log](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-expert-h13-oas-r1/logs/materialize-candidate.log)
- [Completed expert-local-H13 five-boot KLD log](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-expert-h13-oas-r1/logs/kld-five-boot.log)
- [Expert-local-H13 KLD summary](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-h13e-oas-r1/candidate/fresh-sqg-h13e-oas-r1-candidate-kld-fp8-dcp4/summary.json)
- [Expert-local-H13 KLD output and per-boot evidence](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-h13e-oas-r1/candidate/fresh-sqg-h13e-oas-r1-candidate-kld-fp8-dcp4)
- [H13e versus current-SQG paired analysis](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/h13e_vs_current_sqg_kld.json)
- [H13e versus current-SQG paired tensors](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/h13e_vs_current_sqg_kld.safetensors)
- [First pre-inference quarantine: cache census](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-h13e-oas-r1/candidate/fresh-sqg-h13e-oas-r1-candidate-kld-fp8-dcp4.quarantine-incomplete-run1-20260810T194812Z)
- [Second pre-inference quarantine: permissions](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-h13e-oas-r1/candidate/fresh-sqg-h13e-oas-r1-candidate-kld-fp8-dcp4.quarantine-incomplete-run1-20260810T194830Z)
- [Corrected fresh-SQG run seal](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2/run_seal.json)
- [Corrected four-layer KLD summary](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-absrms-r2/candidate/fresh-sqg4-absrms-r2-candidate-kld-fp8-dcp4/summary.json)
- [Historical no-BF16 KLD summary](/home/brandonmusic/KLC_SANDBOXES/sqg_candidate_kld_20260809/results/sqg-four-layer-r1-candidate-kld-fp8-dcp4/summary.json)
- [Recovered capture manifest](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1/capture_manifest.json)
- [Direct E4M3 endpoint report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/e4m3_endpoint_distortion_late_r1.md)
- [Direct E4M3 endpoint JSON](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/e4m3_endpoint_distortion_late_r1.json)
- [Reduced/full capture representativeness report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/capture_representativeness_layer077_r1.md)
- [External selection-document fixed-byte cross-score](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/external_profile_cross_score_l77_r1/cross_score.json)
- [External holdout-document fixed-byte cross-score](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/external_profile_cross_score_l77_holdout_r1/cross_score.json)
- [Late contiguous-block five-boot KLD summary](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-contig-late-a025-r1/candidate/fresh-sqg-contig-late-a025-r1-candidate-kld-fp8-dcp4/summary.json)
- [Late contiguous-block signed-top-8 report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/signed_top8_contig_late_holdout_r1.md)
- [Late contiguous-block signed-top-8 JSON](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/signed_top8_contig_late_holdout_r1.json)
- [Late same-checkpoint empirical-null report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/contiguous_late_same_checkpoint_tail_null_r1.md)
- [Late same-checkpoint empirical-null JSON](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/contiguous_late_same_checkpoint_tail_null_r1.json)
- [Late cross-arm trace report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/contiguous_late_tail_trace_a025_r1.md)
- [Late cross-arm trace JSON](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/contiguous_late_tail_trace_a025_r1.json)
- [Late r33-r33 trace-null report](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/contiguous_late_same_checkpoint_tail_trace_null_r1.md)
- [Late r33-r33 trace-null JSON](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/results/contiguous_late_same_checkpoint_tail_trace_null_r1.json)

### Protocol and implementation

- [Fresh calibration contract](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/docs/fresh_sqg_calibration_capture.md)
- [Fast parallel SQG encoding and corrected-pilot record](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/docs/fast_parallel_sqg_encoding.md)
- [No-BF16 conversion record](/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_no_bf16_test/offline_codec/README.md)
- [Preregistered native-dispatch controls](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/evaluation/CONTROL_ARMS.md)
- [Hessian comparison implementation](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/scripts/compare_hessian_weighted_nmse.py)
- [Expert-local H13 re-encoder](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/scripts/encode_expert_local_h13_shard.py)
- [Expert-local packed comparison](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/scripts/compare_recalibrated_sqg.py)
- [H13e/current-SQG KLD analyzer](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/scripts/analyze_h13e_kld_pair.py)
- [Same-checkpoint tail-null analyzer](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/scripts/analyze_same_checkpoint_tail_null.py)
- [Late paired trace launcher](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/scripts/run_contiguous_late_trace_pair.sh)
- [Protected r33 one-boot trace diagnostic](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/evaluation/run_r33_trace_once.sh)
- [KQuant technical brief](/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/kquant/docs/qsrt-technical-brief.md)

### Expert-local H13 treatment

- [Expert-local H13 run root](/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-expert-h13-oas-r1)
