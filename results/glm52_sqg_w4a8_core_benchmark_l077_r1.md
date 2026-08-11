# Test 8c-core: GLM-5.2 SQG W4A8 prefill benchmark

Status: complete compact-core result; **not** route-packed serving acceptance.

The benchmark used the sealed layer-77 alpha-0.25 SQG checkpoint, the exact
384-K3/384-K4 per-tensor census, BF16 capture rows cast at the deployed GLM
trellis FP16 boundary, and actual top-8 route-count histograms.  Both arms read
the same compact SQG bytes.  W4A16 used the current compact FP16 path; W4A8
applied the stored activation-side transform, per-K32 UE8M0/E4M3 activation
quantization, direct native-SQG-to-E4M3 decode, FP8 MMA, and the stored output
transform.  No dense weight was materialized.

All 126 cases used caller-owned workspaces, CUDA graph replay, 20 warmups, and
200 balanced ABBA measurements per arm.  All W4A16 and W4A8 eager/graph pairs
were bit-exact.

## Route-histogram serial-core projection

| Global tokens | SQG W4A8 MoE-core speedup | Projected end-to-end speedup at declared MoE fraction 0.31 |
|---:|---:|---:|
| 1 | 3.8321x | 1.2972x |
| 128 | 3.7662x | 1.2948x |
| 512 | 3.4135x | 1.2807x |
| 1,024 | 2.9157x | 1.2558x |
| 2,048 | 2.0146x | 1.1850x |
| 3,072 | 1.5966x | 1.1310x |
| 4,096 | 1.2601x | 1.0684x |

The short-tile result confirms that the native E4M3/SQG kernel reaches the FP8
hardware and can be much faster than the compact A16 comparator.  It does not
hold that advantage at the largest observed per-expert route counts: 10 of 18
cases at global M=3,072 and 12 of 18 at M=4,096 were individually slower in
W4A8.  The most-routed expert reached 1,182 rows at M=3,072 and 1,608 rows at
M=4,096.  The current dense W4A8 tiling is therefore not a sufficient long-
prefill kernel.

The central projection is below the preregistered 1.15x migration floor at
M=3,072 and M=4,096.  Because this is a serial sum of dense per-tensor calls,
not a route-packed GLM layer, it cannot by itself reject a future fused
route-packed implementation.  It does reject the claim that the current
compact kernel already establishes the required long-prefill serving win.

## Numerical delta inside the speed harness

The W4A8-versus-W4A16 output NMSE ranges were:

| Projection | Minimum | Median | Maximum |
|---|---:|---:|---:|
| gate | 2.0737e-4 | 3.3942e-4 | 7.6904e-4 |
| up | 1.5068e-4 | 4.0350e-4 | 1.1177e-3 |
| down | 4.8420e-4 | 7.8649e-4 | 1.4841e-3 |

These per-GEMM values are arithmetic checks, not the functional quality gate.
Test 8b evaluates the complete routed expert function and is authoritative for
the activation-quality decision.

## Excluded pre-result failures

The following attempts produced no accepted result file and are excluded:

1. A launch without the runtime worktree on `PYTHONPATH` failed at import.
2. The first benchmark implementation inherited BF16 buffers from the capture
   dtype.  That did not match GLM's deployed FP16 trellis boundary and failed
   during A16 kernel compilation.  The comparator and tests were corrected to
   FP16; no timing from that attempt was retained.
3. The first corrected launch did not expose the existing
   `exllamav3_ext.had_r_128` module and stopped before graph capture.  The
   sealed existing extension was placed on `PYTHONPATH`; no benchmark logic or
   model bytes changed.

The completed JSON is
`results/glm52_sqg_w4a8_core_benchmark_l077_r1.json`, SHA256
`6740f8be4480a9272866b03055ea0f2dee3e5ed8d4453a8e1d4f5d757f64bb58`.
