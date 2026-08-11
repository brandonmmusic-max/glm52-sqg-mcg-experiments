# Fit-only full-W4A8 cross-term beta selection, layer 77

Status: complete hyperparameter selection. Selection and holdout rows were not
used. This is not final-logit KLD or full-model acceptance.

## Frozen design

- Sixteen layer-77 experts selected by equal-count strata of fit/calibration
  gate-square mass.
- `fit/calibration` built H13, native upstream candidates, the private down
  anchor, canonical `(H,B)`, and KQuant caller-coordinate covariance.
- `fit/allocation` scored complete-expert output SSE. No selection or holdout
  row entered construction or choice.
- Every gate/up/down K3/K4 triplet appeared exactly twice: 24 K3 and 24 K4
  tensors, exactly 3.5 bpw. Uniform K3 was not an arm.
- Betas: `0`, `0.03125`, `0.0625`, `0.125`, `0.25`, `0.5`, and `1`.
- Exact native SQG E4M3 weights, both activation QDQ points, exact GLM SiLU,
  corrected caller-coordinate DenseH, and anchored `suh/svh` were used.
- MCG inputs: zero.

## Result

| Beta | Fit/allocation NMSE | Fit/allocation SSE | Expert wins |
|---:|---:|---:|---:|
| 0 | 0.012742601916 | 21,986.448510 | 5 |
| 0.03125 | 0.012716441926 | 21,941.311318 | 3 |
| **0.0625** | **0.012699989438** | **21,912.923727** | 4 |
| 0.125 | 0.012714802383 | 21,938.482404 | 2 |
| 0.25 | 0.012747242951 | 21,994.456285 | 2 |
| 0.5 | 0.012885581781 | 22,233.150045 | 0 |
| 1 | 0.043671878309 | 75,352.703485 | 0 |

`beta=0.0625` lowers SSE by `0.3344%` versus beta 0. Beta 1 is catastrophically
worse and excluded from production. The obsolete beta-1 triplet run was
stopped after 73 expert records; those records remain preserved but cannot
enter final allocation or encoding.

Two attempts failed closed before accepting expert records: preliminary down
H2 contract initialization, then a pretty-JSON versus compact-canonical
execution-contract digest mismatch. Neither rewrote the prepared H13 or panel.
Both have explicit regression tests. The successful panel validated all 16
expert receipts; 23 focused tests passed, with Ruff and shell syntax clean.

Provenance:

- selection ID:
  `5b0d1f37868a3b04a0d6b632e0f025ee250bf66a5d924802f10da26c965fe3ee`;
- selection JSON SHA256:
  `0584b1dc888d18171a84423f4b3bafabf0757dc35fae1891c89a17f753ac00ed`;
- panel SHA256:
  `39c3d3bd0a0a9308b6f22b993f7f70ec5a4ef2a36f7b5d564148b94a838b4392`;
- immutable input ID:
  `f8c07df203ab0b853c9116109a21b0dc75dde2d0f7c5be9a6c79378aa048d337`.

Decision: freeze `beta=0.0625` for the layer-77 exact eight-triplet scorer and
384-K4 dynamic program. This selects a calibration parameter; it does not
claim W4A8 beats A16, MCG, or full-model KLD.
