# Publication scope

This repository is a compact, reviewable publication of the GLM-5.2 SQG/MCG
experiments. It includes the full source tree, tests, documentation, contracts,
machine-readable results, the vendored KQuant working tree, compact KLD outputs,
per-expert JSON records, run seals, manifests, and execution logs.

## Intentionally excluded payload bytes

The following local directories were not copied into Git because they contain
model-scale tensors or regenerated runtime caches:

| Local source | Local bytes | Exclusion | Preserved binding/evidence |
|---|---:|---|---|
| `glm52_fresh_sqg_test/bf16_layers` | 96,542,212,505 | BF16 model shards | source crosswalks, tensor hashes, model index references, and calibration manifests |
| `fresh-sqg-expert-h13-oas-r1` | 33,934,511,644 | expert and assembled `.safetensors` payloads | every non-tensor file, including expert JSON records, seals, layer JSON manifests, and logs |
| `fresh-sqg-encode-absfix-r2` | 51,868,187,151 | shared-`H13` encoded `.safetensors` payloads | every non-tensor file, including run seal, preflight receipts, manifests, and logs |
| `fresh-sqg-full2.GPlzPL/fresh-sqg-calibration-r1` | 51,832,356,623 | raw calibration tensors | `capture_manifest.json` and the project calibration contracts |
| `fresh-sqg-evaluation-h13e-oas-r1` | 4,155,166,200 | per-boot runtime and Hugging Face caches | all non-cache KLD records, per-position tensors, summaries, receipts, and logs |
| `fresh-sqg-evaluation-absrms-r2` | 4,150,156,922 | per-boot runtime and Hugging Face caches | all non-cache KLD records, per-position tensors, summaries, receipts, and logs |
| `KLC_CAPTURE_RUNS/contig-late-capture-r1` | about 12.5 GB | raw Test 10 hidden states and routing arrays | capture/layer manifests, frozen document plan, hashes, and compact status record |
| `glm52_fresh_sqg_test/bf16_contiguous_late` | model shards for layers 74--77 | official BF16 payload shards | 15-shard manifest and BF16 source seal |
| `fresh-sqg-contig-late-a025-r1` | prepared Hessians, per-expert permutations, profile tensor payloads, and smoke tensor payload | Test 10 preparation/profile payload | preparation logs, profile preregistrations/selections, smoke record, and preflight receipt |
| `fresh-sqg-contig-late-final-a025-r1` | 33,933,482,221 | 1,024 expert and four assembled SQG tensor payloads | run seal, materialization receipt, four layer assembly results, and compact logs |
| `GLM-5.2-EXL3-TR3v4-3.5bpw-SQG-CONTIG-L74-77-A025-r1` | 343,053,788,830 | materialized candidate tensors and unchanged hard-linked model payload | candidate manifest, copied run seal, and verification marker |
| `fresh-sqg-evaluation-contig-late-a025-r1/...candidate-kld-fp8-dcp4` | about 830 MB | runtime/cache payload and raw inference logs | rejected-attempt metadata plus compact accepted summaries, records, validations, and five 2,047-position vectors |
| `fresh-sqg-evaluation-contig-late-null-r1/...candidate-kld-fp8-dcp4` | about 830 MB | second same-candidate-path runtime/cache payload and raw logs | compact summaries, identity limitations, records, validations, and five 2,047-position vectors |
| late routed trace roots | about 4.6 GiB per traced arm | raw DCP4 hidden/router/MoE/residual buffers | compact JSON/Markdown analyses and SHA-256 bindings; analysis NPZ files remain local |
| `fresh-sqg-contig-late-alpha{000,050,075,100}-r1` | active multi-arm encode payloads | incomplete alpha-panel expert/model tensors and working state | resumable launcher only; no selection or holdout result is published before completion |

The omitted tensors cannot be reconstructed from this Git repository alone.
Their identities and the measurements derived from them remain auditable through
the published manifests and SHA-256 records. Absolute paths inside raw JSON and
logs are preserved as historical provenance and refer to the original machine.

## Included evidence layout

- `published_evidence/h13e_encode/`: all non-tensor expert-local encoding records.
- `published_evidence/shared_h13_encode/`: all non-tensor shared-`H13` encoding records.
- `published_evidence/h13e_kld/`: expert-local candidate KLD outputs without caches.
- `published_evidence/shared_h13_kld/`: same-dispatch shared-`H13` KLD outputs without caches.
- `published_evidence/historical_sqg_kld/`: the earlier four-layer candidate record.
- `published_evidence/calibration/`: recovered capture manifest.
- `published_evidence/model_manifest/`: final candidate model manifest and verification receipt.
- `published_evidence/contiguous_late/`: Test 10 late-block capture,
  preparation, profile selection, encoding seal, candidate manifest, rejected
  first KLD-attempt record, accepted five-boot KLD evidence, and an independent
  five-boot same-checkpoint null.

Test 9 retains its already-published compact summary JSON/Markdown and small
final-KLD/trace NPZ arrays. Newly added Test 10 analyses publish JSON/Markdown
and compact KLD position vectors, while raw routed-trace buffers, analysis NPZ
payloads, and per-layer NPZ arrays remain omitted. Per-layer JSON measurements
and the analysis code are published.

`SHA256SUMS` binds every published file except itself and Git metadata.
