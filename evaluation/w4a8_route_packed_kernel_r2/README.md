# Test 8c-layer r2: route-packed GLM SQG W4A8 kernel sources

Snapshot of the runtime-worktree files that produced the r2 route-packed
results. The live worktree is
`/home/brandonmusic/KLC_SANDBOXES/glm52-sqg-runtime-20260809/b12x` (an
intentionally dirty worktree; these files are untracked or locally modified
there, so this snapshot plus `SOURCES.sha256` is the code identity for the r2
result files).

| File | Role |
|---|---|
| `glm_trellis_w4a8.py` | Both projection kernels: the original one-warp `m64n8` launch (unchanged, selectable) and the corrected four-warp `m64n256` tile launch |
| `benchmark_glm_sqg_route_packed_hybrid.py` | Full hybrid MoE layer benchmark (A16 control vs FP8 gate/up + A16 down), ABBA, CUDA-graph, distortion validation |
| `benchmark_glm_sqg_route_packed_fc1.py` | Isolated gate/up projection component benchmark |
| `test_trellis_linear.py` | Worktree test module including `test_glm_route_packed_w4a8_tile_kernel_bit_matches_one_warp` (bit-equality vs the one-warp kernel, dense-reference check, CUDA-graph equality; parametrized over 4- and 2-block CTAs) |
| `GLM_SQG_W4A8_KERNEL_DIAGNOSIS.md` | Diagnosis, kernel design, measured tables, and gate statement |

Kernel selection is runtime-only and model-format-neutral:
`B12X_GLM_W4A8_KERNEL=m64n256` (default) or `m64n8` (previous kernel);
`B12X_GLM_W4A8_V2_BLOCKS=4|2`. Shapes not divisible by N256/K128 fall back to
the one-warp kernel. No model bytes, rates, transforms, or packing workspaces
changed; the tile kernel's FP16 outputs are bit-identical to the one-warp
kernel by test.

The r1 collapse results and all r2 reruns were produced on the sealed
winner-native layer-77 candidate
(`fresh-sqg-alpha025-winner-native-l77-full-r1`) with the contiguous-late
layer-77 capture. The r2 interpreter is `/home/brandonmusic/klc-env/bin/python`
(torch 2.11.0+cu130) with `PYTHONPATH` set to the worktree plus the sealed
`exllamav3_ext` directory; the accepted-r1 venv
(`glm52-sqg-runtime-20260809/.venv`) was found stripped of torch/cutlass
(consistent with the disk-space cleanup incident) and was not modified.
Interpreter, kernel arm, and block setting are recorded in each r2 JSON's
`provenance`.
