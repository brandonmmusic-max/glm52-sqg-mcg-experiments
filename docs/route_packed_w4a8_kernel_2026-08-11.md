# Route-packed GLM SQG W4A8 kernel result

Status: isolated layer-kernel correctness and timing complete; integrated
serving acceptance remains open.

## Question

Could the existing sealed mixed-K3/K4 SQG bytes run efficiently through FP8
tensor-core gate/up GEMMs without re-encoding, changing per-tensor rates, or
folding the topology-neutral transforms into the stored labels?

The initial M64xN8 one-warp implementation was approximately 2.0--2.1x slower
than the dispatch-matched compact W4A16 layer. Nsight localized the defect to
195,584 tiny CTAs per projection at M=4,096, 7.385x ideal L2-sector traffic,
4.21% tensor-pipe utilization, and only 3.56% DRAM utilization. It was a
scheduling defect rather than an FP8 or encoding limit.

## Replacement

The replacement keeps the native SQG storage contract unchanged:

- independent per-tensor K3/K4 assignments, with no uniform-K3 arm;
- direct E4M3 SQG labels and no dense weight materialization;
- activation-side `suh`/Hadamard and output-side `svh`;
- exact GLM `silu(gate) * up`;
- signed, gate-weighted top-8 accumulation.

Its launch covers M64xN256 with four warps, stages A in K128 tiles, and decodes
the compact K3 or K4 trellis pool directly into the MMA operands. Four M16 row
blocks per CTA were selected; a two-block CTA was retained as a negative
control.

## Correctness

The focused test ran both CTA configurations and required:

- bit equality with the original one-warp route-packed kernel;
- agreement with a dense decoded SQG reference;
- CUDA-graph replay equality.

```text
2 passed in 14.96s
```

Both complete layer benchmarks also passed finite-output checks, eager/graph
equality, exact compact-down oracles for sampled K3 and K4 experts, and signed
weighted top-8 closure. Candidate-versus-A16 layer distortion reproduced the
initial kernel values, so the speed repair did not alter arithmetic.

## Timing

The sealed layer-77 artifact was measured with 20 warmups and 200 balanced
ABBA samples per arm. Confidence intervals use 10,000 bootstrap replicates.
The table below uses the final same-environment r2 controls; this replaces the
earlier independent timing summary but reaches the same decision.

```text
                    M=3072                 M=4096
A16 median          32.043039 ms           39.417887 ms
hybrid median       27.201119 ms           33.733183 ms
A16 / hybrid         1.178004435            1.168519640
95% CI               1.176995--1.178972     1.167691--1.169322
layer NMSE           0.0002545059            0.0002096047
layer cosine         0.9998730585            0.9998958985
```

The isolated gate/up projection moved from `55.613` to `16.306` ms at
M=3,072 (`3.411x`) and from `73.065` to `19.552` ms at M=4,096 (`3.737x`).
The corresponding one-warp full hybrid layers were `68.277` and `86.468` ms.

The two-block CTA was slower than A16 at M=4,096: A16/hybrid `0.870585`.

The compact machine-readable result is
[`results/glm52_sqg_route_packed_w4a8_v2_l077_r1.json`](../results/glm52_sqg_route_packed_w4a8_v2_l077_r1.json).

Input identities:

```text
layer shard SHA256
04b07b35282673a684315d23db11c48f954ea802a83135e44ae75604ec7c8341

layer manifest SHA256
9301807125afa2b21aa073ffc86e5815f6645748482094c8b799eebaf25b931d

capture UUID
fa5b5504-e695-4fdf-bc17-601352b8a295
```

The measured runtime base was commit
`7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b` with the GLM files still dirty.
Exact measured file identities were:

```text
glm_trellis_w4a8.py
56952fc75c92d41a1e3696d1b12b5de7a6d9bea84cea525e51647312cfc028d8

benchmark_glm_sqg_route_packed_hybrid.py
7ba5926d8387935265f5f4ac2ec4dcc3cf0aed49ea544513408ba707435e1dce

test_trellis_linear.py
9339d10fb886ff82ad07e76254753a3828adbec50cec6008ef8cc4bca2077889
```

## Decision

The isolated route-packed kernel defect is repaired without re-encoding. The
same model bytes now run 1.17--1.18x faster than the dispatch-matched A16
layer instead of approximately 2x slower.

This does not yet pass deployment. With the preregistered 31% MoE share, the
layer ratios project to only 1.04915x at M=3,072 and 1.04680x at M=4,096 for
whole prefill, below the 1.15x migration threshold. The next admissible speed
evidence is an integrated, workload-weighted DCP4 serving benchmark. Encoding
remains frozen until that runtime result and the hybrid-versus-full-W4A8
down-path quality contract are fixed.

## Residual kernel diagnosis

An independent Nsight Compute pass on the M=4,096 M64xN256 projection found
that the original global-A reread collapse is gone, but the replacement is
not yet close to its attainable FP8 ceiling:

- 255 registers per thread and 16.02% achieved occupancy;
- 14,957,856 local spill requests, all reported as spill traffic;
- 48.87% compute and 40.95% memory utilization, but only 14.03% DRAM;
- 0.57 eligible warps per scheduler and no eligible warp in 56.16% of cycles;
- substantial uncoalesced global sectors and 3.7-way shared-load conflicts.

N128 and N64 variants passed the same focused correctness test but did not
improve the result: the M=4,096 A16/hybrid ratios were `1.17139x` and
`1.123997x`. Tile-width selection is therefore not the remaining lever. The
next kernel step is per-expert chunked-M reuse so a trellis fragment is decoded
once across multiple route blocks, together with T12-LUT shared staging and
lower register pressure; gate/up fusion is admissible only if it preserves the
same byte and arithmetic contracts.

The profiler report is retained locally at
`/tmp/glm52_sqg_fc1_m4096_m64n256_ncu.ncu-rep`; it is intentionally not
published because it embeds local absolute paths. The compact r2 result is
[`results/glm52_sqg_route_packed_w4a8_v2_l077_r2.json`](../results/glm52_sqg_route_packed_w4a8_v2_l077_r2.json).
