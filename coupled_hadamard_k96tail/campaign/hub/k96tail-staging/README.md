---
base_model: zai-org/GLM-5.2
library_name: vllm
pipeline_tag: text-generation
tags:
  - glm
  - sqg
  - w4a8
  - mixture-of-experts
  - blackwell
  - quantization
  - coupled-hadamard
  - staging
license: other
---

# GLM-5.2 SQG Coupled H512/H128 K96Tail

> **Incomplete merge staging repository. This is not yet a runnable model.**

This repository is the durable merge bus for a distributed re-encode of
`brandonmusic/GLM-5.2-SQG-W4A8`. Routed expert layers are uploaded only after
materialization, scorer/encoder byte-parity validation, disjoint
selection/holdout scoring, and the native B12X runtime oracle pass.

The target keeps routed layer 3 at its sealed K48 pilot assignment and uses the
updated QSRT coupled H512/H128 transform with K96 tail allocation for routed
layers 4 through 77. The K96 allocation contains 672 K3 cells and 96 K4 cells
per layer, or 3.125 routed payload bits per weight. Preserved BF16 components
and MTP layer 78 remain inherited byte-for-byte from the frozen source
checkpoint during final local assembly. MTP78 retains its source 384 K3 plus
384 K4 census (3.5 bpw), so the actual all-routed-layer average is
3.1291118421052633 bpw rather than a uniform K96 rate.

The coupled coordinate is residual H512 plus H128 before and after exact GLM
`silu(gate) * up`, with H13 local alpha 0.25 and candidate-conditioned H2.
Layers 4--77 independently run the full 16-cell bootstrap, seven-beta choice,
and a fresh 16-cell final profile when beta changes. Fleet-beta, B300
owner-fixed-beta, and identity-only shortcuts are excluded.

The staging payload consists of:

- `r7-experts-layer-NNN.safetensors`
- `r7-experts-layer-NNN.json`
- `r7-experts-layer-NNN.quality.json`
- `runtime-oracle-layer-NNN.json`

Remote waves also publish `reproduction/wave-AAA-BBB/` with a checked
`SHA256SUMS`, campaign log, recipe/profile archive, tail-score archive,
allocation archive, parity proofs, and candidate metadata when retained.
Final local sealing adds scripts, supervisor configs, wave logs, manifests,
oracles, model-codec evidence, KLD distribution evidence, MTP3 smoke, and a
top-level reproduction-bundle `SHA256SUMS`.

End-to-end KLD has not been measured for the assembled checkpoint yet. The
model card will be replaced with measured release evidence after every routed
layer is merged locally and the final TP4/DCP1 KLD run completes.

Calibration and saved Hessian/capture inputs are published at
`brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians` revision
`a05b3b92d749f6a641af5cfd52de2b4720380dfd`. The frozen SQG weight source is
`brandonmusic/GLM-5.2-SQG-W4A8` revision
`593dd0d2de6f79ce4e65303930c22c75e1359d44`; no original BF16 model download
is part of this re-encode.

Final model commit: **PENDING FINAL**
Full TP4/PP1/DCP1 mean KLD: **PENDING FINAL**
Full acceptance receipt SHA-256: **PENDING FINAL**
