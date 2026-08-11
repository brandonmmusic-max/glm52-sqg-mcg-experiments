# Current QSRT review for the GLM-5.2 SQG program

**Review date:** 2026-08-10  
**QSRT snapshot:** `e9f03bb85c925ab50bfd1340490b306940837dbb`  
**Execution:** read-only; no GPU exposed; QSRT CPU tests `498 passed, 1 skipped`

This appendix records techniques and evidence found in the current
[`local-inference-lab/qsrt`](https://github.com/local-inference-lab/qsrt)
repository and its history that may change the GLM-5.2 experiment order.  It
does not treat repository claims, synthetic Kimi fixtures, or newly added code
paths as GLM quality results.

## 1. H13 evidence supports measuring the blend

QSRT's own leak-free historical panel favors shrinkage, not either extreme:

| Expert-local coefficient | Complete-expert validation result |
|---:|---:|
| `0.25` | `2.88%` lower error; all seven experts improved |
| `0.50` | `3.19%` lower aggregate error; one expert regressed |
| `1.00` | `9.47%` higher aggregate error; all seven experts regressed |

Source: [historical measured panel](https://github.com/local-inference-lab/qsrt/blob/84122c166ce277c11f791b6b275e9946277805e7/docs/mode-adaptive-trellis-codec-plan.md#L1600-L1607).

Commit `21fe69793b2c4b4fb2fe90ba616cb9aa35404857` later changed the
production coefficient from `0.25` to `0.0` while also replacing source-H2
with decoded-candidate-conditioned H2.  No later isolated H13 ablation was
found that attributes a gain to full sharing alone.  The [current alpha-zero
constant](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/qsrt/qsrt.py#L221-L233)
therefore establishes the current recipe, not statistical optimality.

The coordinate basis can be shared while routed activation statistics remain
expert-conditioned.  The correct GLM action is to finish the uniform
`local-alpha={0,.25,.5,.75,1}` panel and the frozen all-SQG layerwise mapping.
Prefer one coefficient per layer or block with expert-local covariance inputs;
do not independently fit 256 alphas per layer without a much larger sample.

All GLM alpha references use the canonical equation

```text
H13(alpha,e) = (1-alpha) H13_layer + alpha H13_local,e
```

so `alpha=0.25` means 25% expert-local and 75% layer-shared.

## 2. Coupled activation-boundary Hadamard is the strongest new ablation

QSRT commit [`2fb1367`](https://github.com/local-inference-lab/qsrt/commit/2fb1367c42bf768aad84236cc33273be7ea3aa9a)
generalizes its exact coupled gate/up/down Hadamard path from uniform K2 to
uniform K3.  It is not an ordinary matrix-local EXL transform:

1. gate and up are interleaved and jointly rotated;
2. the preactivation transform is inverted before the coordinatewise
   nonlinearity;
3. a postactivation transform is matched to down; and
4. the residual-side transform closes across the gate/up input and down
   output.

Sources: [construction](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/qsrt/qsrt_coupled.py#L145-L196)
and [execution closure](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/qsrt/qsrt_coupled.py#L212-L286).

The measured evidence is currently K2 only: a fixed coupled transform reduced
routed SSE by `3.052%`, and expert-private draw selection added `1.308%`
pooled improvement on an external corpus.  Source: [QSRT technical
brief](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/docs/qsrt-technical-brief.md#L848-L865).

Limitations for GLM:

- the new K3 path has no published K3 quality result;
- current validation permits uniform K2 or uniform K3, not GLM's mixed K3/K4
  inventory;
- direct E4M3 labels survive only if inverse transforms remain on the
  activation/FP32-epilogue sides; and
- it must not be blindly stacked with existing matrix-local Hadamards.

After selecting the late H13 blend, run one matched four-layer GLM ablation
with the same byte census, alpha, H2 policy, and signed-top-8 scorer.  Extend
closure to both K3 and K4 and start with the small draw portfolio `{0,6}`.

## 3. Reassign K4 tensors under an SQG-native objective

QSRT's strongest allocation result used a fixed `22 x K3 + 2 x K4` expert
schedule at `3.0833` trellis bpw.  It reduced pooled routed SSE by `11.907%`
relative to uniform K3, and all 24 tested experts improved.  More elaborate
tile selection was slightly worse.  Source: [measured allocation
panel](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/docs/qsrt-technical-brief.md#L370-L396).

For the first GLM test, preserve topology neutrality and the exact total
`1,536 K3 / 1,536 K4` census.  Change only which tensors receive K4, using
SQG-native signed routed-function damage rather than the inherited MCG map.
This is distinct from whole-expert X4T promotion, which is outside the user's
per-tensor allocation contract.

A later record-level K2/K4 exchange inside a nominal K3 tensor could preserve
the tensor's average rate, but it needs a new format and runtime kernel and is
not part of the current controlled comparison.

## 4. Borrow the gate/up pair metric before a joint vector trellis

QSRT evaluates the local `2 x 2` activation metric of paired gate/up errors
and reports a median `4.90%` functional-metric improvement for a codebook
oracle.  Sources: [metric implementation](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/qsrt/coupled_expert_study.py#L248-L286)
and [reported status](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/docs/qsrt-technical-brief.md#L988-L997).

The cheap GLM borrowing is to use this metric to select among already
generated gate/up profile, rotation, or alternate-path pairs.  Then rebuild
candidate-specific H2 and score the complete expert.  A genuinely joint
vector trellis is a separate, higher-cost codec project.

## 5. Co-routing optimization remains secondary

QSRT contains a constrained solver that selects among retained expert
candidates after signed router-weighted summation while enforcing a unary-loss
bound.  Source: [solver](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/qsrt/coupled_expert_study.py#L477-L612).

QSRT measured the cross-expert term at only `0.929%` of diagonal mapped SSE.
The GLM late block similarly found that only `2.2603%` of its regression came
from the change in the aggregate cross-expert interaction term.  Retaining
multiple near-equal payloads for co-routing selection is therefore lower
priority than fixing individual-expert calibration and encoding.

## 6. W4A8 remains plausible at C1, but GLM must measure it

A historical QSRT synthetic TP12 routed fixture reported:

- W4A8 latency `167–320 us`;
- matched W4A16 latency `600–759 us`;
- speedup `2.24–3.58x`;
- activation-induced output NMSE `0.199–0.222%`; and
- cosine `0.998888–0.999007`.

Source: [historical measurements](https://github.com/local-inference-lab/qsrt/blob/84122c166ce277c11f791b6b275e9946277805e7/docs/sqrt-c-technical-brief.md#L277-L307).

These are not GLM TP4 end-to-end results, but they mean that unchanged weight
bytes alone do not prove A8 is useless at C1.  Kernel structure, occupancy,
A16 decoding overhead, and MMA work can dominate.

The GLM program should separately measure A16 codec quality, E4M3 `h`, the
heavy-tailed `act = SiLU(gate) * up`, `H2_A8` captured from the complete
upstream W4A8 path, C1 decode, short/medium batch, and long prefill.  Current
QSRT candidate-conditioned H2 is still based on A16 upstream replay; it does
not supersede the proposed GLM `H2_A8`.

## 7. Kimi K3 K1 feasibility is unchanged

QSRT's uniform-K2 expert payload is `682,207,608,832` bytes before
nonexperts, caches, and runtime workspace.  Source: [payload
census](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/docs/qsrt-technical-brief.md#L631-L637).

Current SQG accepts only K2 through K6; K1 is rejected.  Source: [rate
validation](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/qsrt/sqg_e4m3.py#L31-L35).

One-bit routed experts would still be about 340 GB decimal before scales.
Adding approximately 57B nonexpert parameters near eight bits puts static
storage near 397 GB before KV cache and workspace, already above four 96 GB
GPUs.  K1 Kimi therefore needs a distinct sub-one-bpw codec/capacity program,
lower-bit nonexperts, offload, or a combination; it is not a parameter change
to current QSRT.

## 8. Attribution record

Current HEAD contains no BMM, BMMLaw, Black Matrix Modulation, Brandon, or
`brandonmusic` attribution.  It credits QTIP, EXL3, and QuIP# in the [current
lineage](https://github.com/local-inference-lab/qsrt/blob/e9f03bb85c925ab50bfd1340490b306940837dbb/docs/qsrt-2bpw-codec.md#L79-L105).

Historical commit `bd0e95c21440b23c234c495ab3a2a2d674f8e78a` explicitly named
`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` and its public calibration encoder as
the source for the GLM shared-H recipe.  Source: [historical
attribution](https://github.com/local-inference-lab/qsrt/blob/bd0e95c21440b23c234c495ab3a2a2d674f8e78a/recipes/glm52_exl3_shared_h/README.md#L15-L31).
That recipe was removed in commit [`79461d3`](https://github.com/local-inference-lab/qsrt/commit/79461d37a3a863fd2859e5ae14438e184eaf9ca3)
during removal of legacy experiments.  Similar mechanisms exist in current
QSRT and BMM-law, but the repository alone does not establish that every
current mechanism was adopted from BMM-law.

## Revised action order

1. Finish the late H13 alpha grid and frozen secondary holdout.
2. Repeat profile/rotation search natively for the selected alpha so the
   layer-global arm does not retain a home-field advantage.
3. Run the matched coupled activation-boundary Hadamard ablation on the same
   late block, extending closure to both K3 and K4.
4. Re-optimize the per-tensor K3/K4 assignment under the exact existing byte
   census and an SQG-native signed routed-function objective.
5. Add the SiLU gate/up pair metric to candidate selection.
6. Only then freeze a full-quant recipe and run scaled A16 quality.
7. Validate W4A8 independently, including C1; do not assume the benefit is
   prefill-only.
8. Treat K1 Kimi as a separate codec and capacity project.
