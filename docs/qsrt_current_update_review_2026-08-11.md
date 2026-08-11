# QSRT head review after the GLM falsification gates

**Review date:** 2026-08-11

**QSRT master:** `83f61fc0d150be988463ea0d69115441c486345c`

**Open GLM/Fruit PR:** [local-inference-lab/qsrt#4](https://github.com/local-inference-lab/qsrt/pull/4), head `2113af303f37cedf4b538dcf68eb699d5e31f7df`

**Execution boundary:** implementation and repository review; QSRT's Kimi/Fruit results are not treated as GLM quality or serving results

This addendum supersedes the action order in the 2026-08-10 review where later
GLM measurements now exist. It asks which changes in current
[`local-inference-lab/qsrt`](https://github.com/local-inference-lab/qsrt) have a
direct mechanism against the remaining GLM-5.2 failures. The protected GLM
format remains topology-neutral, with independent per-tensor K3/K4 assignments;
uniform K3, H308, and fixed P24/P33 allocation are not adopted.

## Decision summary

| QSRT item | GLM decision | Reason |
|---|---|---|
| Saturating-cost Viterbi closure (`27d4894`) | Adopted and tested | Prevents reachable FP16 path costs from becoming indistinguishable from unreachable states at K2/K3/K4 |
| Worker fail-fast propagation (`d161606`) | Adopt pattern | Avoids silently continuing after a candidate worker fails |
| Final atoms-v2 rebuild comparison (`83f61fc`) | Adopt concept | Final package should be reconstructed from sealed candidates and compared byte-for-byte before publication |
| Sample-space down refit, `ridge_refit_down` | Tested; reject the full-strength encoded target | The floating target improved every expert, but realized K3/K4 bytes regressed `0.9885%` on holdout because individual-expert damage exceeded the cancellation gain |
| Unary-bounded co-routing selector | Tested; retain as a secondary expert-private selector | The frozen 16-expert assignment improved holdout mean NMSE `0.4684%`, but the bootstrap crossed zero and the maximum position worsened |
| Uniform-K3 coupled conditioning | Reject as the GLM production format | It breaks the protected per-tensor rate assignment, and the matched mixed-rate GLM coupled pilot regressed strongly |
| H308 / fixed P24/P33 atoms-v2 profiles | Reject as a direct format port | Their Kimi geometry and rate records do not represent GLM's independent tensor-level K3/K4 choices |
| Fruit global-H13 policy | Do not infer into GLM | It is a recipe choice, not evidence that fully shared H13 beats the measured GLM blend |
| Fruit TP1 runtime | Packaging reference only | It does not close GLM DCP4, long context, MTP, production concurrency, or route-packed prefill |

## 1. Viterbi saturation repair is immediately useful

QSRT commit [`27d4894`](https://github.com/local-inference-lab/qsrt/commit/27d4894)
clamps every reachable dynamic-programming cost to the maximum finite FP16
value while retaining infinity only for an unreachable predecessor. This is a
correctness repair rather than a new codebook or allocation result.

The same rule has been ported to the sealed GLM KQuant extension. Explicit
stressed K2, K3, and K4 closure tests pass. The port does not change the
protected bit map, codebook, transforms, or calibration objective.

## 2. The useful down-refit idea must end in encoded bytes

QSRT's `ridge_refit_down` solves the candidate-conditioned down correction in
sample space. In normal-equation form, the GLM version uses the exact upstream
candidate operand `Q` and BF16 teacher expert output `Y`:

```text
H = sum(gate^2 * Q^T Q)
B = sum(gate^2 * Q^T Y)
W* = solve(H, B)
```

For the admissible hybrid path, `Q` must be produced by the complete upstream
h-A8 execution: stored `suh`, H128 on the activation side, MXFP8 h,
native-E4M3 SQG gate/up labels, FP32 accumulation, stored output transforms,
and exact `SiLU(gate) * up`. Selection and holdout rows remain excluded from
the fit.

The critical adaptation beyond QSRT's oracle is to encode `W*` with dense
BlockLDLQ at the original down tensor's K3 or K4 assignment, then score the
realized trellis bytes. A floating refit that cannot survive re-encoding is not
a production quality gain.

That decisive encode has now completed for all 256 layer-77 experts. The
floating target reduced fit SSE for every expert, with a `53.26%` median
improvement. The realized bytes nevertheless regressed signed-top-8 NMSE by
`1.0985%` on selection and `0.9885%` on untouched holdout. Individual-expert
SSE rose `1.1136%` on holdout while the cross-expert term improved `25.0816%`.
The individual damage won. The full-strength target is rejected.

## 3. Coupled H128 did not transfer to this GLM format

The exact activation-parametric GLM coupled implementation was extended to
mixed per-tensor K3/K4 and exact `SiLU(gate) * up`. On the preregistered
layer-77 panel, its selected arm regressed against uncoupled alpha-0.25 by
`33.26%` on selection and `35.10%` on holdout; approximately `99.9%` of the
regression was individual-expert SSE rather than cross-expert cancellation.

That result rejects extending the current coupled H128 construction to block
74-77. It does not reject all activation-parametric joint objectives. The
uncoupled winner-native profile and realizable `(H,B)` down encode were tested
next under the protected mixed-rate format; the former improved, while the
latter failed after trellis re-encoding as recorded above.

## 4. Co-routing is a bounded secondary optimizer

QSRT's deterministic `select_corouted_candidate_modes` can choose among
several retained payloads against the signed summed routed output. Its
`unary_relative_slack` is the important guard: a candidate is eligible only
when its individual-expert error stays within a declared bound of that
expert's unary winner.

For GLM, this is applied only after candidate generation and ordinary unary
selection. It does not replace expert calibration and cannot rescue a
candidate family whose individual-expert SSE is already substantially worse.
The winner-native search therefore retains several encoded profile cells for a
later co-routing analysis, but retention is not promotion.

That analysis has now completed with exact 6,144-dimensional signed output
errors and no new encode. Four unary slacks from `0%` through `1%` were screened
on selection documents. The frozen `1%` assignment improved untouched-holdout
mean NMSE by `0.4684%`, from `0.002423681864` to `0.002412329518`, but its
paired-document interval crossed zero. Of the selection SSE gain, `93.32%`
came from choosing better individual expert profiles and only `6.68%` from
cross-expert cancellation. Holdout p99 and CVaR moved only slightly adversely,
while the maximum position worsened `13.59%`.

The useful mechanism is therefore expert-private profile choice first, with
co-routing only as a bounded tie-breaker. This panel does not justify a
runtime policy or a full-model promotion.

## 5. Atoms-v2 should preserve GLM's actual rate topology

Current QSRT atoms-v2 is valuable as a storage and reproducibility pattern:
tensor-parallel-independent logical records, explicit profile identity, and a
final reconstruction from the sealed candidate pool followed by byte
comparison. Its published profiles are nevertheless Kimi-specific.

A GLM atoms-v2 profile must instead carry, for every tensor:

- its independent K3 or K4 assignment;
- its private trellis payload and private transform side;
- the topology-neutral shared gate/up input and down output vectors;
- the expert permutation and exact source/calibration bindings; and
- no implicit conversion to uniform K3, H308, or expert-level rate modes.

The format is packaging work. It cannot improve NMSE by itself, but it is the
right eventual contract between the encoder, B12X, vLLM, and the Hugging Face
artifact.

## 6. PR #4 is groundwork, not closure for GLM serving

PR #4 adds an activation abstraction, logical geometry adapter, calibrated
Fruit construction, and an atoms-v2 package path. Its reported Fruit result is
encouraging for QSRT packaging, but its focused behavior tasks did not close
and its runtime scope is TP1. It does not supply the missing route-packed GLM
prefill kernel or validate GLM activation quantization.

The useful collaboration boundary is therefore:

1. reuse the source/activation and sealed-artifact interfaces;
2. define the mixed per-tensor GLM atoms-v2 profile;
3. adapt the Kimi route-map and fused activation rotation/quantization to GLM
   dimensions and exact SiLU;
4. validate DCP4, long context, MTP, and production concurrency in vLLM; and
5. keep the hybrid h-A8 quality arm separate from the rejected act-A8 arm.

## 7. Inference hygiene remains part of the codec decision

Open [issue #5](https://github.com/local-inference-lab/qsrt/issues/5) correctly
identifies winner-only bootstrap after a searched candidate grid as optimistic.
The GLM winner-native profile search uses one shared document-level resampling
design and a familywise Bonferroni bound against the frozen control. The chosen
candidate is then scored once on untouched holdout. Retained alternatives are
reserved for a separately declared co-routing analysis.

The repository still has no top-level license file. `THIRD_PARTY_NOTICES.md`
does not grant a license to copy, modify, or redistribute the repository's own
code. Upstream code reuse and public binary/source publication therefore need
an explicit license from the author or a clean-room implementation of the
relevant interface and algorithms.

## Revised execution order

1. Preserve the completed winner-native profile, full-strength `(H,B)` reject,
   and favorable-direction but inconclusive co-routing evidence.
2. Define the GLM atoms-v2 profile without changing the independent per-tensor
   K3/K4 map or topology-neutral transforms.
3. Port and benchmark the route-packed GLM hybrid/W4A8 prefill kernel; keep
   act-A8 disabled under the current quality evidence.
4. Retain multiple expert-private profile candidates in the next full-layer
   encode, but use unary quality as the primary selector and cancellation only
   within a strict bound.
5. Integrate the selected A16 or hybrid kernel through GLM DCP4, long context,
   MTP, and production concurrency without changing the model contract.
6. Measure workload-weighted prefill plus DCP4/long-context/MTP serving before
   deciding on a full 75-layer quant.

## Winner-native profile result

The mixed-rate layer-77 search completed all 16 preregistered cells over the
same 16-expert panel. It selected `draw-00__identity` over the frozen
`draw-03__identity` control:

| Split | draw-03 identity control | draw-00 identity winner | Relative change |
|---|---:|---:|---:|
| Selection | `0.001910524055` | `0.001881479481` | `-1.5202%` |
| Untouched holdout | `0.002443549285` | `0.002423681864` | `-0.8131%` |

The selection comparison passed the preregistered one-sided Bonferroni
document-bootstrap gate. The separately evaluated 45-document holdout also
had a positive paired-bootstrap improvement lower bound. Profile selection was
frozen before either holdout arm was compared.

This supports materializing `draw-00__identity` across all 256 layer-77
experts. It does not establish a KLD improvement and does not change the
protected per-tensor K3/K4 allocation.

## Realized `(H,B)` and co-routing follow-ups

The all-expert exact h-A8 `(H,B)` down re-encode failed after quantization:
holdout signed-top-8 NMSE increased `0.9885%`, despite a `25.08%` reduction in
the cross-expert error term. This rejects the direct full-strength refit target.

The separate retained-profile co-routing pilot made no new bytes. It froze an
expert-private assignment on selection documents and improved untouched
holdout mean NMSE `0.4684%`; the 95% document interval still crossed zero.
The gain was `93.32%` unary profile quality and `6.68%` cancellation. This is
useful directional evidence for expert-private profile selection, not a new
full-quant gate pass.
