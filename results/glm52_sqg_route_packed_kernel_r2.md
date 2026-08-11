# Test 8c-layer: route-packed GLM SQG W4A8 kernel, r1 collapse and r2 correction

This is a runtime speed result on the sealed winner-native layer-77 alpha-0.25
candidate. It changes no model bytes, rates, transforms, or quality
conclusions. A16 remains the quality-isolation reference; the timed hybrid is
the quality-qualified configuration (FP8 gate/up with h-A8, A16 down with
`act` retained at A16).

## r1: the first route-packed implementation collapsed

| M | A16 full MoE layer | Hybrid | A16/hybrid |
|---:|---:|---:|---:|
| 3,072 | 40.089 ms | 81.490 ms | 0.49195 |
| 4,096 | 50.380 ms | 105.411 ms | 0.47794 |

Component profiling put ~86 ms of the 105.4 ms in the mixed-rate gate/up
route-packed FP8 projection; transforms and quantization were under 1 ms.
Diagnosis of the one-warp M64xN8 kernel: each CTA covered 8 of 2,048 output
columns but read the full M64xK6144 quantized-A slice from global memory with
no staging (~256x read amplification, ~60 GB per projection at M=4,096), one
warp had no latency cover, and trellis fragments were re-decoded per CTA. The
FP8 arithmetic itself was never the problem.

## r2: corrected M64xN256 tile kernel

One CTA = M64xN256 with 128 threads. A and its UE8M0 row scales are cp.async
double-buffered through xor-swizzled shared memory (staging discipline from
the fused W4A8 reference pipeline); each warp owns eight n8 strips and decodes
B fragments register-resident with the unchanged warp trellis primitive, so a
decoded fragment feeds four M16 MMAs; the K3/K4 pool is resolved once per CTA,
preserving fully independent per-tensor rates and all four gate/up rate pairs.
Compact payloads remain the only weight representation (no FP16
reconstruction, no dense FP8 materialization). The accumulation order is
identical to the one-warp kernel, and the unit test asserts bit-identical
outputs (`torch.equal`), plus dense-reference closure and CUDA-graph
capture/replay equality, for 4- and 2-block CTA variants.

## Same-environment measurements (20 warmups, 200 balanced ABBA samples)

The r2 environment reproduces the r1 collapse with the unchanged kernel
before fixing it:

| Median | M=3,072 | M=4,096 |
|---|---:|---:|
| gate/up projection, one-warp | 55.613 ms | 73.065 ms |
| gate/up projection, tile | 16.306 ms | 19.552 ms |
| projection speedup | 3.411x | 3.737x |
| hybrid layer, one-warp | 68.277 ms | 86.468 ms |
| hybrid layer, tile | 27.201 ms | 33.733 ms |
| A16/hybrid, one-warp | 0.4820 [0.4816, 0.4822] | 0.4642 [0.4639, 0.4645] |
| A16/hybrid, tile | 1.1780 [1.1770, 1.1790] | 1.1685 [1.1677, 1.1693] |
| Amdahl end-to-end projection at MoE fraction 0.31 | 1.0491 | 1.0468 |

Validation in the accepted runs: independent K3/K4 census intact; dense
per-rate oracle passed; signed top-8 routed sum passed; eager and CUDA-graph
outputs bit-exact in both arms; all outputs finite. Hybrid-vs-A16 layer-output
distortion at M=4,096: NMSE `2.0960e-4`, cosine `0.9998959`, RMSE `0.0087088`,
max abs `1.0` — the h-A8 quantization difference, bit-identical to the r1
kernel's outputs by construction.

## What this establishes and does not establish

- The route-packed W4A8 collapse was an implementation defect, now fixed: the
  hybrid layer is faster than A16 at both long-prefill sizes (1.17-1.18x)
  instead of ~2.1x slower.
- The preregistered speed gate is still not met: 1.73x full-MoE is required
  for a 1.15x end-to-end projection at MoE fraction 0.31, and the measured
  full-MoE speedup is 1.17-1.18x (projection ~1.047-1.049x).
- The residual gap is structural to the hybrid quality configuration: the
  required A16 down block costs ~13.4 ms of the 33.7 ms hybrid at M=4,096,
  capping the reachable full-MoE ratio near 2.8x; clearing 1.73x needs the
  gate/up projection near 8.6 ms, a further ~2.3x. Identified but unproven
  levers: removing the remaining ~2.4x per-expert decode redundancy with
  per-expert chunked M loops, moving the 4 KB T12 LUT to shared memory, and
  widening the staged K tile.
- This is a single-layer, single-GPU component result at two synthetic M
  values with an assumed MoE fraction; it is not serving acceptance and not a
  workload-weighted end-to-end measurement.

Result JSONs: `glm52_sqg_route_packed_hybrid_l077_m{3072,4096}_r1.json`
(accepted collapse baselines),
`glm52_sqg_route_packed_{hybrid,fc1}_l077_m{3072,4096}_r2_m64n256.json` and
`..._r2_m64n8_control.json` (r2 arms). Kernel sources and identity hashes:
[`evaluation/w4a8_route_packed_kernel_r2/`](../evaluation/w4a8_route_packed_kernel_r2/README.md).
