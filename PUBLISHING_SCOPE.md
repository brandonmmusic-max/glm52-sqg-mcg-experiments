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

`SHA256SUMS` binds every published file except itself and Git metadata.
