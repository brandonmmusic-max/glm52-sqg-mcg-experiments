# K96 coupled-Hadamard frozen evidence snapshot

This directory freezes the measured local state at
`2026-08-15T00:26:39-04:00`. It is not a live status feed and does not claim a
finished model.

- `status.json` is the compact campaign frontier and final-result disposition.
- `layers_003_046/` holds the 44 sealed non-tensor manifests, quality receipts,
  and passing native B12X runtime oracles.
- `parity_layers_004_046/` holds 43 exact scorer/encoder payload-parity proofs,
  each covering 256 experts and 768 projections.
- `allocations_layers_004_046/` holds the base and KLD/route-guarded K96
  allocations for every sealed K96 layer. All 43 guarded allocations remain
  production eligible.
- `recipe_layers_004_046/` holds compact no-shortcut recipe, beta,
  preregistration, final-profile binding, selection, and holdout lineage.
- `score_ledger.json` aggregates every complete score layer 4--50. Its
  `score_receipts_layers_004_050.sha256` companion binds all 12,032 expert JSON
  receipts and 12,032 paired row-SSE arrays without copying those payloads.
  The 24,064 bound files total 2,154,247,428 bytes; all embedded row-SSE hashes
  were checked against their arrays.
- `wave_archives.sha256` binds 11 omitted candidate-metadata archives totaling
  78,249,961 bytes. `campaign.log` is the 99,034-byte frozen controller log.

No `.safetensors`, row-SSE `.npz`, activation capture, Hessian payload, cache,
container layer, credential, SSH endpoint, token, or provider secret is copied.
The final Hub model commit and end-to-end candidate KLD are **PENDING FINAL**.
