# Route-packed W4A8 kernel r3: combined-kernel source snapshot

Code identity for the r3/r4/r5 result files in
[`results/glm52_sqg_route_packed_kernel_r3.md`](../../results/glm52_sqg_route_packed_kernel_r3.md).
The canonical kernel merges two parallel lines of work:

- the M128xN64 compact-B shared-staging launch (independent parallel agent
  work: A and the compact trellis words cp.async double-buffered together,
  T12 LUT resident in shared memory via `t12_in_shared=True`, eight M16
  blocks per CTA, no register spills, 128-row expert-homogeneous route
  packing); and
- the n16 pair-decode graft (this branch): both n8 halves of each staged
  16-column tile decoded in one pass instead of decode-and-discard twice,
  bit-identical by test, worth ~9% on the spill-free base.

Also included: parameterized pipeline depth (`B12X_GLM_W4A8_V2_STAGES`,
default 2 — stages 3/4 measured slower from occupancy loss), and the
env-gated **speed-only** full-W4A8 down arm in the hybrid benchmark
(`GLM_HYBRID_DOWN_A8=1`), which quantizes `act` to E4M3 per K32 and runs the
down projection through the same tile kernel with `shared_input=False`. That
arm reports its own layer distortion next to its timing and carries an
explicit not-quality-qualified marker in the payload; Test 8b's act-A8 red
finding is unchanged by any of this.

| File | Role |
|---|---|
| `glm_trellis_w4a8.py` | One-warp control kernel + canonical M128xN64 tile kernel with pair-decode and stage plumbing |
| `benchmark_glm_sqg_route_packed_hybrid.py` | Hybrid layer benchmark + env-gated full-W4A8 speed-only arm |
| `benchmark_glm_sqg_route_packed_fc1.py` | Isolated gate/up projection component benchmark |
| `test_trellis_linear.py` | GLM kernel tests incl. bit-equality vs the one-warp kernel (16/16 passing at r3) |

Kernel selection: `B12X_GLM_W4A8_KERNEL=m128n64` (default) or `m64n8`
(before/after control); `B12X_GLM_W4A8_V2_BLOCKS` (default 8);
`B12X_GLM_W4A8_V2_STAGES` (default 2). Live tree:
`/home/brandonmusic/KLC_SANDBOXES/glm52-sqg-runtime-20260809/b12x`
(intentionally dirty; this snapshot plus `SOURCES.sha256` is the durable
identity). Interpreter and env are recorded in each result JSON's
`provenance`.
