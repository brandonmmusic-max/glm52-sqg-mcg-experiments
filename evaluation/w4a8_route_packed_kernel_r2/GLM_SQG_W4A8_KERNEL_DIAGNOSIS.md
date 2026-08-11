# GLM-5.2 SQG route-packed W4A8 projection: diagnosis and corrected kernel

Status: implementation and static validation complete; GPU correctness and
benchmarks pending GPU availability (full-model capture holds all four GPUs).
This document will be finalized with measured numbers; nothing below claims a
measured speedup yet.

## Measured symptom (accepted r1 results)

| M | A16 full MoE layer | Hybrid FP8 gate/up + A16 down | A16/hybrid |
|---:|---:|---:|---:|
| 3,072 | 40.089 ms | 81.490 ms | 0.49195 |
| 4,096 | 50.380 ms | 105.411 ms | 0.47794 |

Component profiling at M=4,096 localized ~86 ms of the 105.4 ms in the
mixed-rate gate/up route-packed FP8 projection. Route packing + input
transform + MXFP8 quantization (~0.125 ms) and output transform + SwiGLU
(~0.814 ms) are immaterial.

## Root cause in the one-warp M64xN8 kernel

The projection kernel (`_GLMRoutePackedW4A8MixedProjectionLaunch`) launches a
grid of `(route_blocks, size_n/8)` one-warp CTAs. Three compounding defects:

1. **A-operand global-memory amplification, the dominant cost.** Each CTA
   covers only 8 of 2,048 output columns but reads the full M64xK6144
   quantized A slice (393 KB) directly from global memory, per lane, per k32
   step, per M16 group, with no shared-memory staging. Every route block's A
   bytes are therefore fetched by 256 independent CTAs: ~60 GB of A reads per
   projection call at M=4,096 against a ~200 MB unique footprint (~256x
   amplification). Two projection calls (gate, up) double it. At realistic
   L2/DRAM service rates this alone accounts for the bulk of the 86 ms.
2. **One warp per CTA.** A single warp cannot cover the latency chain of
   scalar trellis loads -> bit windowing -> LUT gathers -> MMA; there is no
   cp.async prefetch, no double buffering, and nothing to overlap decode with
   MMA. Achieved tensor-core utilization is a rounding error: the entire
   gate+up GEMM at M=4,096 is ~825 GFLOP, ~1-2 ms of FP8 MMA on this part.
3. **Fine N grid multiplies trellis/LUT overhead.** Each 16-column trellis
   tile is decoded by two N8 CTAs (n_high halves) and re-decoded by every M64
   block of the same expert, and the same lane geometry/base computations are
   re-executed per CTA.

The dense-core benchmark already proved the native SQG->E4M3 FP8 path beats
the A16 comparator at small/medium M (3.83x at M=1), so the arithmetic is not
the problem; the route-packed schedule is.

## Corrected kernel (`_GLMRoutePackedW4A8TileLaunch`, `m64n256`)

One CTA now covers **M64 x N256 with 128 threads (4 warps)**:

- **A staged once per CTA through shared memory.** The 4-block x 16-row x
  K128 A tile plus its UE8M0 row scales are cp.async double-buffered with the
  xor-swizzled 16B-unit layout taken from the fused W4A8 reference pipeline
  (`b12x/moe/fused/w4a8/gemm.py`), so fragment reads are bank-conflict-free
  and A global traffic per route block drops from `size_n/8` reads to
  `size_n/256` (256x -> 8x; ~60 GB -> ~1.9 GB per projection at M=4,096).
- **B stays compact and register-resident.** Each warp owns eight n8 strips;
  the existing `_decode_k32_bits_at_base` warp primitive already produces the
  exact m16n8k32 B fragment (2 u32/lane), so each decoded fragment feeds four
  M16 MMAs directly from registers. No FP16 reconstruction, no dense FP8
  materialization, no new weight bytes: the compact K3/K4 trellis pools and
  the 4,096-byte T12 LUT are the only weight-side inputs, unchanged.
- **Pipeline discipline copied from the reference:** issue k-tile t+1, commit,
  `cp_async_wait_group(1)`, sync, consume k-tile t, tail sync (the tail
  barrier fences the 2-stage overwrite). K-tile = K128 = 4 k32 MMA steps.
- **Rate independence preserved.** The expert's K3 or K4 pool and slot are
  resolved once per CTA and dispatched to a constexpr-specialized tile body,
  exactly like the one-warp mixed launch. All four gate/up rate pairs remain
  independent; nothing couples them.
- **Same numerics by construction.** The k32 accumulation order, MMA
  intrinsic (`mxfp8_mma_m16n8k32_f32_e4m3`), operand mapping (rows q/q+8,
  words `k32*8 + 2c`, SFA parity rule), and FP16 epilogue conversion are
  identical to the one-warp kernel, so outputs must match **bit-exactly**;
  the new unit test asserts `torch.equal` against the one-warp kernel, plus
  the dense reference tolerance check and CUDA-graph capture/replay equality.

Workspace, route packing (M64 expert-homogeneous blocks), entry-point
signature, caller-owned buffers, and CUDA-graph safety are unchanged. The
kernel is selected with `B12X_GLM_W4A8_KERNEL` (`m64n256` default,
`m64n8` = previous kernel for before/after) and `B12X_GLM_W4A8_V2_BLOCKS`
(4 default; 2 halves accumulator pressure by splitting each M64 block across
two CTAs). Shapes not divisible by N256/K128 fall back to the one-warp
kernel so legacy callers stay valid.

## Measured results (2026-08-11, GPU 0, 20 warmups / 200 balanced ABBA samples)

Correctness, from the unit tests and the sealed-layer benchmark validation:
independent K3/K4 census intact; dense per-rate oracle passed (K3 and K4,
zero distortion); signed top-8 routed sum passed; eager and CUDA-graph
outputs bit-exact in both arms; all outputs finite; the tile kernel's output
is **bit-identical** (`torch.equal`) to the one-warp kernel for mixed-rate
routed inputs, so the hybrid quality contract is numerically unchanged.
Hybrid vs dispatch-matched A16 layer output at M=4,096: NMSE `2.0960e-4`,
cosine `0.9998959`, RMSE `0.0087088`, max|diff| `1.0` (identical to the
one-warp arm by construction; this is the h-A8 quantization difference, not a
kernel property).

Same-environment before/after (klc-env interpreter; the accepted r1 numbers
in the table at the top were produced in the now-stripped venv — the one-warp
control reproduces the r1 collapse in this environment: 0.4642/0.4820):

| Metric (median) | M=3,072 | M=4,096 |
|---|---:|---:|
| gate/up projection, one-warp `m64n8` | 55.613 ms | 73.065 ms |
| gate/up projection, tile `m64n256` | **16.306 ms** | **19.552 ms** |
| projection speedup | **3.411x** | **3.737x** |
| full hybrid layer, one-warp | 68.277 ms | 86.468 ms |
| full hybrid layer, tile | **27.201 ms** | **33.733 ms** |
| A16 full layer (control, same run) | 32.909 / 32.043 ms | 40.142 / 39.418 ms |
| A16/hybrid, one-warp | 0.4820 [0.4816, 0.4822] | 0.4642 [0.4639, 0.4645] |
| A16/hybrid, tile | **1.1780 [1.1770, 1.1790]** | **1.1685 [1.1677, 1.1693]** |
| Amdahl end-to-end at MoE fraction 0.31 | 1.0491 | 1.0468 |

Transforms are unchanged (route-pack + input transform + MXFP8 ~0.10 ms;
output transform + exact SwiGLU ~0.64 ms at M=4,096).

## Gate statement

- "M=3,072 and M=4,096 must no longer collapse": **met**. The hybrid is now
  faster than A16 at both sizes instead of ~2.1x slower.
- "The current kernel must be beaten decisively": **met**. 3.4-3.7x on the
  isolated projection, 2.5-2.6x on the full hybrid layer, with
  non-overlapping bootstrap CIs.
- "~1.73x full-MoE speedup for 1.15x end-to-end at MoE fraction 0.31":
  **not met**. Measured full-MoE speedup is 1.17-1.18x, projecting to
  ~1.047-1.049x end-to-end.

The residual gap is now structural to the *hybrid quality configuration*,
not to the FP8 path: the A16 down block (down transform + W4A16 down
projection + weighted output sum) costs ~13.4 ms of the 33.7 ms hybrid at
M=4,096 and is required by the Test 8b quality contract (`act` stays A16).
With that block fixed, the hybrid's reachable ceiling is ~2.8x full-MoE;
clearing 1.73x requires the gate/up projection at ~8.6 ms, a further ~2.3x
past this kernel. The identified, unproven levers: eliminate the remaining
~2.4x per-expert decode redundancy (per-expert chunked M-loops so each
trellis fragment is decoded once per launch), move the 4 KB T12 LUT into
shared memory, and widen the staged K tile. Those are scheduling changes on
top of this kernel's structure, not a redesign.

Profiler note: `ncu` is not installed on this host (`nsys` only), so
achieved-occupancy/register counts are not reported as measured values;
configuration-level accounting (128 threads/CTA, 16.9 KB smem/CTA double
buffer, ~170 f32 accumulator+operand registers/thread) and the ABBA component
timings above are the supported evidence.

## Environment note

The accepted r1 results' interpreter (`glm52-sqg-runtime-20260809/.venv`) has
been stripped of torch/cutlass (only pytest/zmq/psutil remain), most likely by
the disk-space cleanup during the alpha-panel storage incident. The benchmark
rerun will use `/home/brandonmusic/klc-env/bin/python` (torch 2.11.0+cu130,
cutlass, cuda-python present) with `PYTHONPATH` pointing at this worktree, and
the new result JSONs record the interpreter and kernel selection in
`provenance`. No accepted result file is overwritten.
