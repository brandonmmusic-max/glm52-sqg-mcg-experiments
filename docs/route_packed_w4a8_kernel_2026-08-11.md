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

```text
                    M=3072                 M=4096
A16 median          31.897568 ms           39.493376 ms
hybrid median       27.003600 ms           33.796177 ms
A16 / hybrid         1.181233895            1.168575247
95% CI               1.180526--1.181949     1.167744--1.169290
layer NMSE           0.0002545059            0.0002096047
layer cosine         0.9998730585            0.9998958985
```

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
layer ratios project to only 1.04994x at M=3,072 and 1.04681x at M=4,096 for
whole prefill, below the 1.15x migration threshold. The next admissible speed
evidence is an integrated, workload-weighted DCP4 serving benchmark. Encoding
remains frozen until that runtime result and the hybrid-versus-full-W4A8
down-path quality contract are fixed.
