# Compact execution evidence

This directory records the measured state at `2026-08-14T00:16:45-04:00`.
It is not a live status feed.

- `execution_snapshot_2026-08-14T001645-0400.json` states the layer and score
  counts at that timestamp.
- `score_receipts_2026-08-14T001645-0400.sha256` binds 581 atomic expert JSON
  receipts and their 581 row-SSE arrays without copying the arrays.
- `layers_003_018/` contains the compact layer manifests, quality receipts,
  and native B12X runtime oracle receipts for all 16 qualified layers.
- `parity_layers_004_018/` contains the exact scorer/encoder payload parity
  receipts for the 15 K96 layers. Layer 3 uses the separate preserved K48
  contract and is not represented as a K96 parity result.
- `layer_019_runtime_binding.json` is a durable example of the frozen SQG,
  calibration dataset, coupled transform, QSRT, and source-domain bindings.
- The two recovery records preserve the NVIDIA GSP failure diagnosis and the
  post-reboot single-writer recovery decision.

Large `.safetensors`, `.npz`, capture, model, log, and cache payloads are not
copied. Their compact receipts and hashes are the publication boundary.
