# QSRT Kimi K3 K1 feasibility audit

Date: 2026-08-10

Upstream audited: `local-inference-lab/qsrt` commit
`8be1d78548554057e07ae3d9b510709f328ddbea`

Execution boundary: read-only research; no local GPUs used

## Conclusion

A Kimi K3 K1 research experiment is feasible. A full homogeneous K1 quant is
not yet justified, is not supported by the current QSRT implementation, and
does not fit a practical four-RTX-PRO-6000 runtime under QSRT's present
nonexpert precision plan.

The likely viable design is:

- a topology-neutral K1 trellis base for nearly all routed-expert
  coefficients;
- BMM expert-conditional H13 blended toward layer-global statistics;
- candidate-exact H2;
- expert-private coupled rotations;
- signed top-16 and positive-tail-aware allocation;
- a direct-E4M3 W4A8 endpoint;
- independently compressed nonexpert tensors near 4.25 bpw; and
- a very small per-tensor higher-rate correction budget.

This would be approximately `1.00447 bpw` for the expert payload and roughly
`1.0856 bpw` model-wide under the estimated compressed-nonexpert plan. It
would not be a literal one-bit entire model.

## Audit scope and evidence limits

The audit read the current [2-bpw codec
specification](https://github.com/local-inference-lab/qsrt/blob/8be1d78548554057e07ae3d9b510709f328ddbea/docs/qsrt-2bpw-codec.md),
the relevant encoder, storage, graph, calibration, and runtime paths, and the
[technical
brief](https://github.com/local-inference-lab/qsrt/blob/8be1d78548554057e07ae3d9b510709f328ddbea/docs/qsrt-technical-brief.md).
The CUDA-hidden CPU/unit suite completed with `491 passed, 1 skipped` in
26.28 seconds.

Passing tests establish internal behavior for implemented profiles. They do
not establish K1 quality, K1 runtime support, prompt-logit KLD, or TP4
performance. K1 is absent from current graph validation, offline CUDA Viterbi,
mixed-rate backend, storage, and the external B12X runtime; those paths begin
at K2.

QSRT reports a K2 expert container of `682,207,608,832` bytes and
`2.004464 bpw` after FP16 scales. Its technical brief reports mean prompt-logit
KLD `0.0851995464` over 32 windows and 65,504 positions. The repository does
not provide the raw logits, complete candidate artifact, per-position tail
analysis, runtime traces, or task-evaluation evidence required to reproduce
that claim independently. It is therefore recorded as an upstream report,
not a locally reproduced result.

## What the K2 codec does

The current codec represents Kimi K3 routed experts using:

- an L16 trellis with 14 history bits and a two-bit K2 branch;
- a Gaussian-rank law projected to a finite E4M3 alphabet;
- tail-biting Viterbi encoding;
- 256-coefficient optimization tiles with 128 coefficients of context on each
  side;
- a layer-shared exact residual Hadamard;
- expert-private signed H128 transforms around coupled gate/up and post-SiTU
  intermediate coordinates;
- a layer-global gate/up Hessian H13;
- an expert-specific down Hessian H2 built from the decoded upstream
  candidate;
- route-probability-squared complete-expert functional scoring;
- document-disjoint fit and confirmation sets; and
- uniform K2 encoding of every routed expert.

The especially useful ideas for BMM are the coupled transform across the
nonlinear boundary, expert-private intermediate rotations, candidate-specific
down calibration, and complete-expert scoring. QSRT's public results report a
3.052% pooled routed-SSE improvement from the paired transform on 22 of 24
sampled experts. That result motivates a GLM SwiGLU-specific analogue; it does
not justify copying the Kimi SiTU transform unchanged.

## The H13 issue

Current QSRT forces gate/up H13 to be layer-global; its candidate construction
effectively assigns zero expert-local blend. A shared coordinate basis makes
one Hessian dimensionally compatible with every expert, but does not imply
identical conditional activation geometry:

```text
E[h h^T | expert e routed] generally differs from E[h h^T].
```

The BMM test should use

```text
H13_e(alpha) = (1 - alpha) H13_local,e + alpha H13_layer
```

with gate-probability-squared local statistics and support-dependent shrinkage
toward the layer estimate. The notation here assigns alpha to the layer prior;
the GLM Test 9 implementation assigns alpha to the local component, so the two
reported coefficients must be converted rather than compared by name.

At K1 this metric alignment is more important: the encoder has so little rate
that a label chosen under the wrong conditional geometry is difficult to
repair downstream.

## K1 graph feasibility and quality warning

A natural L16 K1 extension has 15 history bits, two outgoing branches per
state, one candidate from each sign half, 65,536 total rank slots, tail-biting
closure, and the same topology-neutral 32-channel atoms.

An ephemeral CPU structural probe found:

- a bijective K1 mapping across all 65,536 ranks;
- one candidate from each sign half at every state;
- 151 realized E4M3 values over approximately `[-5.5, 5.5]`;
- fixed-scale one-step Gaussian candidate MSE `0.56243` for K1 versus
  `0.14733` for K2, a 3.82x ratio; and
- independently scale-optimized MSE `0.47659` for K1 versus `0.14459` for K2,
  a 3.30x ratio.

This is a structural proxy, not a real encode. It omits tail-biting Viterbi,
LDLQ feedback, fitted row scales, Hessians, rotations, signed routed sums, and
model logits. It is nonetheless a serious negative signal consistent with
the expected large rate-distortion step from two bits to one.

Published one-bit work likewise does not support naive homogeneous PTQ:
[BiLLM](https://arxiv.org/abs/2402.04291) uses salient-weight and binary
residual mechanisms, [PB-LLM](https://arxiv.org/abs/2310.00034) preserves
salient weights at higher precision, and
[OneBit](https://arxiv.org/abs/2402.11295) uses quantization-aware training and
distillation.

## Four-GPU storage and residency

Assuming existing 32-channel atoms and FP16 row scales:

| Expert quantity | Size |
|---|---:|
| K1 trellis per expert | 4,128,768 bytes |
| FP16 scales per expert | 18,432 bytes |
| Total per expert | 4,147,200 bytes |
| All routed experts plus container overhead | 341,865,005,056 bytes |
| Expert payload rate | 1.004473 bpw |
| Expert bytes per GPU under TP4 | 79.60 GiB |

With QSRT's documented MXFP8/BF16 nonexpert plan:

| Quantity | Estimate |
|---|---:|
| Total checkpoint | 376.37 GiB |
| Effective model-wide rate | 1.16299 bpw |
| Resident weights including replicated routers | 379.68 GiB |
| Resident weights per GPU | 94.92 GiB |
| Remaining nominal capacity per GPU | 0.67 GiB |

That cannot support normal KV cache, activations, CUDA modules, workspaces,
collectives, and allocator fragmentation.

If eligible nonexpert tensors are independently reduced to approximately 4.25
bpw:

| Quantity | Estimate |
|---|---:|
| Total checkpoint | 351.32 GiB |
| Effective model-wide rate | 1.08558 bpw |
| Resident weights including replicated routers | 354.62 GiB |
| Resident weights per GPU | 88.66 GiB |
| Remaining nominal capacity per GPU | 6.93 GiB |

That is plausibly loadable for short-context batch-one research, but is not a
comfortable production or one-million-context configuration. Reserving about
6 GiB for runtime leaves only roughly `0.0118` expert bpw for exceptions:
about 1.18% of coefficients promoted K1 to K2, or 0.4% promoted K1 to four
bits, before exception metadata.

These are accounting estimates. A load-only TP4 probe must measure actual
allocator state, replicated tensors, module code, graph buffers, and minimum
KV/workspace before encoding a full model.

## Runtime implications

One stored bit is not one-bit MMA. RTX PRO 6000 Blackwell provides FP4, FP8,
and BF16 tensor operations for these paths, not a qualified binary MMA. K1
labels still need decoding to E4M3, FP4, or FP16 registers before matrix
multiplication.

Direct E4M3 labels retain the same potential weight-endpoint advantage as SQG:
no additional weight-side E4M3 rounding. That does not prove speed. K1 needs
new graph/rank-law support, tail-biting encoding, format/profile schema,
CPU reference decode, CUDA offline encode, B12X runtime decode, TP4 loading,
and qualified A16/A8 kernels. The published K2 artifact was evaluated under
TP8, so TP4 collectives are also unqualified.

## Tail-aware go/no-go plan

1. **Seal the memory contract.** Measure exact TP4 resident memory with
   replicated routers, loaded kernels, allocator overhead, a small KV cache,
   activation buffers, and scratch. Require at least 4-6 GiB genuinely free
   per GPU. Stop if the current nonexpert plan is retained.
2. **Implement K1 format and CPU references.** Prove rank bijection, sign
   strata, deterministic pack/unpack, tail-biting closure, T12 agreement,
   realized label entropy, and reconstruction closure.
3. **Encode a support-stratified sample.** Use about 28 experts from seven
   early/middle/late layers, covering common, rare, and tail-heavy routing.
   Compare K1/K2 under no rotation, QSRT coupled rotation, expert-private
   rotation, the BMM local/global H13 grid, and candidate-exact H2.
4. **Use signed top-16 tail metrics.** Sum signed router-weighted expert-output
   errors before squaring. Gate on mean, position win fraction, p95/p99/p99.5/
   p99.9, worst-tail CVaR, maximum regression, routing changes, residual drift,
   and next-layer router-logit drift.
5. **Test small quality oracles.** Compare a joint 2x2 gate/up objective,
   structured or low-rank down correction, co-routing-aware phase selection,
   and a minute per-tensor higher-rate budget before spending more bits widely.
6. **Run contiguous blocks.** Insert matched K1 blocks in early, middle, and
   late regions. Advance beyond four layers only if the positive KLD tail,
   routing, and residual errors remain controlled.
7. **Qualify runtime last.** Benchmark K1A16 and K1A8 against K2A16 for C1,
   several prefill lengths, isolated decode, decode-plus-MMA, TP4 collectives,
   and exact peak VRAM.

The first full-model action should be a memory/load falsification, not an
encode. The first quality action should be the 28-expert K1/K2 panel, not a
full homogeneous K1 conversion.
