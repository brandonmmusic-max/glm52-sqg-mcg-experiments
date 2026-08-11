# Route-packed W4A8 kernel r2: source snapshot and raw benchmark artifacts

Companion to the published narrative
[`docs/route_packed_w4a8_kernel_2026-08-11.md`](../../docs/route_packed_w4a8_kernel_2026-08-11.md).
This directory preserves the code identity for the r2 result files; the live
tree is the intentionally dirty runtime worktree
`/home/brandonmusic/KLC_SANDBOXES/glm52-sqg-runtime-20260809/b12x`, where
these files are untracked or locally modified, so this snapshot plus
`SOURCES.sha256` is their durable identity.

| File | Role |
|---|---|
| `glm_trellis_w4a8.py` | Both projection kernels: the original one-warp `m64n8` launch (unchanged, selectable as the before/after control) and the corrected four-warp `m64n256` tile launch |
| `benchmark_glm_sqg_route_packed_hybrid.py` | Full hybrid MoE layer benchmark (A16 control vs FP8 gate/up + A16 down), balanced ABBA, CUDA-graph, distortion validation; provenance records the kernel arm |
| `benchmark_glm_sqg_route_packed_fc1.py` | Isolated gate/up projection component benchmark |
| `test_trellis_linear.py` | Worktree test module including `test_glm_route_packed_w4a8_tile_kernel_bit_matches_one_warp` (bit-equality vs the one-warp kernel, dense-reference closure, CUDA-graph equality; 4- and 2-block CTA parametrizations) |
| `GLM_SQG_W4A8_KERNEL_DIAGNOSIS.md` | Working diagnosis and measured tables as written in the worktree |

Kernel selection is runtime-only and model-format-neutral:
`B12X_GLM_W4A8_KERNEL=m64n256` (default) or `m64n8`;
`B12X_GLM_W4A8_V2_BLOCKS=4|2`. Shapes not divisible by N256/K128 fall back to
the one-warp kernel. No model bytes, rates, transforms, or packing workspaces
changed; the tile kernel's FP16 outputs are bit-identical to the one-warp
kernel by test.

Raw result JSONs in [`results/`](../../results/):

- `glm52_sqg_route_packed_hybrid_l077_m{3072,4096}_r1.json` — the accepted
  one-warp collapse baselines (A16/hybrid `0.49195` / `0.47794`).
- `glm52_sqg_route_packed_hybrid_l077_m{3072,4096}_r2_m64n256.json` — the
  corrected tile kernel (A16/hybrid `1.1780` / `1.1685`).
- `glm52_sqg_route_packed_hybrid_l077_m{3072,4096}_r2_m64n8_control.json` —
  the unchanged one-warp kernel rerun in the r2 environment (reproduces the
  collapse: `0.4820` / `0.4642`).
- `glm52_sqg_route_packed_fc1_l077_m{3072,4096}_r2_{m64n256,m64n8_control}.json`
  — isolated gate/up projection component timings: one-warp `55.613` /
  `73.065` ms versus tile `16.306` / `19.552` ms median (3.411x / 3.737x).

Environment: the accepted-r1 venv (`glm52-sqg-runtime-20260809/.venv`) was
found stripped of torch/cutlass (consistent with the disk-space cleanup
incident) and was not modified; r2 runs used
`/home/brandonmusic/klc-env/bin/python` (torch 2.11.0+cu130) with
`PYTHONPATH` set to the worktree plus the sealed `exllamav3_ext` directory.
Interpreter, kernel arm, and block setting are recorded in each r2 JSON's
`provenance`. The sealed model inputs are the winner-native layer-77
candidate (`fresh-sqg-alpha025-winner-native-l77-full-r1`) and the
contiguous-late layer-77 capture.
