# W4A8 quality and compact-core evidence

This directory preserves the exact benchmark harness, its CPU-side contract
test, and the modified compact W4A8 kernel used for Test 8c-core.  The result
itself is summarized in
[`results/glm52_sqg_w4a8_core_benchmark_l077_r1.md`](../../results/glm52_sqg_w4a8_core_benchmark_l077_r1.md).

The measured decision is conditional **NO-GO**:

- Test 8b increased full-layer signed top-8 NMSE by `22.9802%` on selection
  and `22.7790%` on the secondary holdout relative to matched SQG A16.
- Test 8c-core reached the FP8 path, but its route-histogram serial-core
  projection was only `1.1310x` end-to-end at global M=3,072 and `1.0684x`
  at M=4,096 under the declared MoE fraction `0.31`, below the preregistered
  `1.15x` migration floor.
- The benchmark is a dense compact-core falsification, not route-packed GLM
  serving acceptance.

## Published code identity

The runtime worktree was based on commit
`7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b`.  The benchmark and test were
untracked experiment files; `w4a8.py` was a modified runtime file.  Their
published byte identities are:

| File | SHA-256 |
|---|---|
| `code/benchmark_glm_sqg_w4a8_core.py` | `a0003c51da21ee5aacb53dd2dd7779398869a75fd2f52ded1725b74e03b42c2c` |
| `code/test_benchmark_glm_sqg_w4a8_core.py` | `34138b04c98e5d50a941970afa2f6fef2279318a22475bd76c88b0b2ac17d5d4` |
| `code/w4a8.py` | `c1f8818f0501791a9c1bfaf8e3d38531f33c3681bd88e8a10c8ee712fa8a3ac6` |

The unchanged compact A16 comparator was
`b12x/moe/_shared/kernels/w4a16/kernel.py`, SHA-256
`f8b81dbd3f735cec08c2ca9903ebb2ed24189f7ab9838e5c76abc9d0a2a9246b`.

## Omitted raw evidence

The 3,516,542-byte timing JSON is intentionally not duplicated in Git.  Its
original local path is
`glm52_fresh_sqg_test/results/glm52_sqg_w4a8_core_benchmark_l077_r1.json`
and its SHA-256 is
`6740f8be4480a9272866b03055ea0f2dee3e5ed8d4453a8e1d4f5d757f64bb58`.
It contains 126 cases, 200 balanced ABBA samples per arm, 20 warmups, numeric
checks, graph/eager identity records, route histograms, and bootstrap timing
intervals.  The compact Markdown report publishes its methodology, excluded
pre-result failures, numerical ranges, all seven aggregate projections, and
the decision boundary.

The large alpha-panel and Test 8b per-position NPZ arrays are also omitted.
Their exact source-relative paths, byte counts, and hashes are recorded in
[`OMITTED_EVIDENCE.sha256`](OMITTED_EVIDENCE.sha256); the corresponding
machine-readable aggregate and per-layer JSON files are published under
[`results/`](../../results/).
