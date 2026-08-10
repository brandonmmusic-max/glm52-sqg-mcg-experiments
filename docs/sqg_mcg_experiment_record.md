# SQG versus MCG on GLM-5.2 3.5 bpw: experiment record

**Record status:** living audit, updated 2026-08-10.  Tests 0–7b now have local
artifacts.  Test 7 is a packed-byte routed-function result; Test 7b is the
completed five-boot final-logit KLD endpoint for its materialized candidate.

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
| 8 | Direct E4M3 endpoint and full GLM W4A8 tests | Not run | Weight-endpoint advantage, A8 quality, and actual FP8-MMA speed |

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

The preregistered hypothesis was that the 5.9847% reduction in
untouched-holdout isolated complete-expert NMSE from Test 7 would survive the
effects absent from that
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

No local GLM-specific result currently compares MCG-to-E4M3 against native SQG
E4M3 labels on identical tensors, and no GLM full-path W4A8 KLD or speed result
has been completed.  The four-layer candidate KLD runs above were explicitly
forced through the A16 path.

KQuant's local technical brief supports the architectural hypothesis that SQG
can use a finite E4M3 label menu and that W4A8 execution machinery exists.  Its
own expert panels are useful prior evidence, but they are not a GLM MCG-versus-
SQG endpoint test.  In particular, the quoted 3.859% figure must not be stated
as “MCG loses 3.859% when converted to E4M3”: it compared FP16-oriented SQG
search followed by E4M3 rounding with E4M3-aware SQG search.

### Precise hypothesis

If SQG's stored labels are already E4M3 values, those labels can reach the FP8
MMA weight operand without an additional lossy projection.  MCG decodes to an
FP16-valued reconstruction and must be rounded to E4M3 before FP8 MMA.  SQG's
unique possible advantage is on this **weight conversion**.  Both approaches
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

### Planned falsification test 8a: CPU endpoint distortion

For identical sealed BF16 tensors and fixed rates, measure:

1. MCG decoded at FP16;
2. MCG decoded and rounded to E4M3;
3. SQG at its A16 reconstruction endpoint;
4. SQG native E4M3 labels used directly.

Report raw SSE/NMSE, Hessian-weighted distortion, and the **incremental error
caused only by the E4M3 endpoint**.  This can be performed without a new model
encode and, subject to decoder availability, mostly on CPU.  It directly tests
the proposed unique SQG advantage.

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
   improved untouched-holdout routed gate/up NMSE by 10.5130% and complete
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
- Four separated layers do not predict full-model contiguous error propagation.
- SQG has not reduced model size at the frozen map.
- MCG-to-E4M3 has not been shown locally to incur 3.859% error.
- W4A8 has not yet shown a GLM KLD or speed win.

## Recommended experiment order from here

1. Run the unchanged MCG checkpoint through the exact SQG-native selected-layer
   dispatch and complete the paired per-position causal comparison.
2. Localize Test 7b's positive KLD tail with signed top-8 summed expert outputs,
   router changes, and downstream residuals at the affected positions.
3. Run the preregistered H13 alpha ablation on a panel using a signed summed
   top-8 or downstream-sensitive selection objective, then untouched-document
   confirmation.
4. Extend the surviving corrected candidate to several document-paired KLD
   prompts, retaining paired per-position outputs.
5. Test three four-layer **contiguous** blocks (early/middle/late) before a full
   conversion to measure propagation that the separated design cannot see.
6. Run the direct E4M3 endpoint falsification and decode-cost microbenchmark.
7. Only then attempt exact-path W4A8 and, if the evidence remains favorable, a
   full BF16 SQG quantization.

## Artifact and source index

### Primary reports and sealed results

- [Raw encoded NMSE report](../results/raw_encoded_nmse_sqg_vs_mcg.md)
- [Raw encoded NMSE JSON](../results/raw_encoded_nmse_sqg_vs_mcg.json)
- [Hessian-weighted NMSE report](../results/hessian_weighted_nmse_sqg_vs_mcg.md)
- [Hessian-weighted NMSE JSON](../results/hessian_weighted_nmse_sqg_vs_mcg.json)
- [Expert-local H13 routed-function report](../results/recalibrated_sqg_vs_mcg.md)
- [Expert-local H13 routed-function JSON](../results/recalibrated_sqg_vs_mcg.json)
- [Expert-local H13 four-layer run seal](../published_evidence/h13e_encode/run_seal.json)
- [Expert-local H13 runnable-candidate manifest](../published_evidence/model_manifest/MANIFEST.json)
- [Runnable-candidate verified marker](../published_evidence/model_manifest/manifest_verified.txt)
- [Candidate materialization log](../published_evidence/h13e_encode/logs/materialize-candidate.log)
- [Completed expert-local-H13 five-boot KLD log](../published_evidence/h13e_encode/logs/kld-five-boot.log)
- [Expert-local-H13 KLD summary](../published_evidence/h13e_kld/candidate/fresh-sqg-h13e-oas-r1-candidate-kld-fp8-dcp4/summary.json)
- [Expert-local-H13 KLD output and per-boot evidence](../published_evidence/h13e_kld/candidate/fresh-sqg-h13e-oas-r1-candidate-kld-fp8-dcp4)
- [H13e versus current-SQG paired analysis](../results/h13e_vs_current_sqg_kld.json)
- [H13e versus current-SQG paired tensors](../results/h13e_vs_current_sqg_kld.safetensors)
- [First pre-inference quarantine: cache census](../published_evidence/h13e_kld/candidate/fresh-sqg-h13e-oas-r1-candidate-kld-fp8-dcp4.quarantine-incomplete-run1-20260810T194812Z)
- [Second pre-inference quarantine: permissions](../published_evidence/h13e_kld/candidate/fresh-sqg-h13e-oas-r1-candidate-kld-fp8-dcp4.quarantine-incomplete-run1-20260810T194830Z)
- [Corrected fresh-SQG run seal](../published_evidence/shared_h13_encode/run_seal.json)
- [Corrected four-layer KLD summary](../published_evidence/shared_h13_kld/candidate/fresh-sqg4-absrms-r2-candidate-kld-fp8-dcp4/summary.json)
- [Historical no-BF16 KLD summary](../published_evidence/historical_sqg_kld/results/sqg-four-layer-r1-candidate-kld-fp8-dcp4/summary.json)
- [Recovered capture manifest](../published_evidence/calibration/capture_manifest.json)

### Protocol and implementation

- [Fresh calibration contract](../docs/fresh_sqg_calibration_capture.md)
- [Fast parallel SQG encoding and corrected-pilot record](../docs/fast_parallel_sqg_encoding.md)
- [No-BF16 conversion record](../published_evidence/offline_codec/README.md)
- [Preregistered native-dispatch controls](../evaluation/CONTROL_ARMS.md)
- [Hessian comparison implementation](../scripts/compare_hessian_weighted_nmse.py)
- [Expert-local H13 re-encoder](../scripts/encode_expert_local_h13_shard.py)
- [Expert-local packed comparison](../scripts/compare_recalibrated_sqg.py)
- [H13e/current-SQG KLD analyzer](../scripts/analyze_h13e_kld_pair.py)
- [KQuant technical brief](../kquant/docs/qsrt-technical-brief.md)

### Expert-local H13 treatment

- [Expert-local H13 run root](../published_evidence/h13e_encode)
