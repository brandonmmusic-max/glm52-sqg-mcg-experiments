# Coupled-Hadamard K96-tail execution record for 2026-08-14

This dated record reports the measured state of the GLM-5.2 routed-expert
re-encode at `2026-08-14T00:16:45-04:00`. It is a status and incident record,
not the format specification. See
[`coupled_hadamard_k96tail_reencode.md`](coupled_hadamard_k96tail_reencode.md)
for the system contract.

## Artifact status

| Artifact or gate | Status | Measured result |
|---|---|---|
| Layer 3 coupled K48 shard | qualified | 720 K3, 48 K4, 3.0625 bpw, passing native runtime oracle |
| Layers 4 through 18 coupled K96 shards | qualified | 15 sealed, hash-bound layer shards, each with 672 K3, 96 K4, 3.125 bpw |
| Fully qualified routed layers | qualified | 16 of 76, layers 3 through 18 |
| Layer 19 through 22 no-shortcut recipes | qualified | Four complete recipe receipts and final profile bindings |
| Layer 19 through 22 exact triplet scoring | implemented | Running at the record timestamp with 581 of 1,024 expert receipts: layer 19 has 153, layer 20 has 146, layer 21 has 156, layer 22 has 126 |
| Layers 19 through 78 selected shards | unsupported | No additional complete layer shard exists |
| Full checkpoint assembly | unsupported | Requires 76 passing layer seals and runtime oracles |
| Candidate final-logit KLD | unsupported | No candidate KLD receipt exists |
| Release acceptance | unsupported | Codec census, KLD gates, MTP3 smoke, and reproduction seal have not run |

## Measurements for layers 4 through 18

All 15 K96 layers passed exact scorer/encoder parity. The receipts cover 3,840
experts and 11,520 projection payloads. Every payload comparison reports
`all_exact = true`.

All 16 selected layer shards, including layer 3, passed the native B12X
route-packed direct-E4M3 W4A8 oracle. B12X is the native runtime implementation
used by this oracle. Direct-E4M3 means that the SQG payload reaches the FP8 E4M3
activation endpoint without an A16 fallback. The measured relative RMSE range is
`0.0399174` to `0.0469858`, and the mean-cosine range is `0.9988958` to
`0.9992380`. Every oracle reports finite output, nonzero output, and pass.

The selected K96 layer beta distribution is:

| Selected beta | Layer count |
|---:|---:|
| 0.0625 | 1 |
| 0.125 | 4 |
| 0.25 | 8 |
| 0.5 | 2 |

Fourteen K96 layers ran a fresh 16-cell profile search after beta selection.
Layer 13 reused its bootstrap profile byte for byte. No layer used the B300
owner-speed shortcut.

The routed-function aggregate relative error ranges are:

| Role | Minimum | Maximum |
|---|---:|---:|
| Selection | `0.000141594` | `0.013846092` |
| Holdout | `0.000142455` | `0.014471883` |

These are layer-local routed-function measurements. They are not final-logit
KLD and must not be presented as a model-quality result.

## Saved inputs

The completed input snapshot for wave 31 through 34 contains a 15-shard source
manifest and four `hidden.bf16.bin` capture tensors. Each capture tensor is
12,897,349,632 bytes. This wave is staged and validated but has not entered the
recipe or encode stages.

The wave 35 through 38 directory contains about 6.3 GB and only a partial
capture checkout. It is not a complete staged wave because its required-file
and manifest validation has not passed.

## GPU driver incident

Exact scoring for layers 19 through 22 started with 32 workers. The NVIDIA
610.43.02 open kernel driver reported a GSP task-watchdog timeout for GPU 3 at
`2026-08-13T23:51:22-04:00`. CUDA, NVML, NVIDIA modeset, and the compositor then
blocked in the shared NVIDIA RM path. Expert receipt counts stopped at 579.

GPU 3 had been between 78 C and 81 C and did not report an active hardware
thermal slowdown. The incident is classified as a driver/GSP failure, not a
scorer exception or a thermal shutdown.

Terminating scorer containers, issuing a PCI function reset, attempting driver
unbind, and issuing a second reset did not release the shared RM state. A host
reboot restored all four GPUs. The saved NVMe input volume remounted by UUID.

All 579 expert JSON files and all 579 row-SSE files are complete atomic
receipts. No temporary, partial, or incomplete receipt exists in the score
roots. The normal scorer validates and skips them on resume.

## Resume preparation

The canonical resume unit is the user service
`glm52-full-coupled-k96tail-no-shortcut-goal019ffa7c.service`. It is enabled and
active with 32 exact-score containers at the timestamp of this record. The
duplicate system service
`glm52-full-coupled-k96tail-resume.service` is disabled and inactive so only one
campaign process can write the evidence roots.

The first post-reboot resume exposed temporary ExLlamaV3 paths that had been
cleared by the reboot. No encode work was lost. The dependency package was
restored under `glm52_fresh_sqg_3p0625/runtime-dependencies`. The ExLlamaV3 Python
package was extracted byte for byte from encode image
`sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82`
and has tree SHA-256
`834c8de389c700126e5746e7bae9876b3e443014480651e59919cdd68fc20506`.
The v39 extension was extracted from image ID
`sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac`
and has file SHA-256
`e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd`.

The five reachable launchers are `run_coupled_recipe_wave.sh`,
`run_coupled_tail_score_wave.sh`, `run_coupled_transcode_wave.sh`,
`run_score_select_coupled_wave.sh`, and
`run_validate_coupled_runtime_layer.sh`. They use persistent,
environment-overridable dependency defaults and verify the extension hash. The
reconstruction script
`scripts/prepare_pinned_exllamav3_runtime.sh` passed a fresh temporary
extraction test. The campaign resumed at `2026-08-14T00:13:59-04:00` and
advanced the atomic score receipts beyond the pre-reboot count.

The live campaign log is
`/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/full_coupled_k96tail_no_shortcut_campaign.log`.
The repository contains the timestamped
[receipt hash snapshot](../coupled_hadamard_k96tail/evidence/score_receipts_2026-08-14T001645-0400.sha256),
[execution snapshot](../coupled_hadamard_k96tail/evidence/execution_snapshot_2026-08-14T001645-0400.json),
and compact [layer and parity receipts](../coupled_hadamard_k96tail/evidence/README.md).
The layer shards and runtime-oracle receipts are under
`/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers`.

## Completed execution steps

The measured run completed these steps:

1. Updated QSRT to revision `453b483` and bound its GLM coupled changes by
   tracked-diff SHA-256.
2. Verified the approved KQuant revision, tracked diff, backend tree, and status
   hashes against the saved preflight contract.
3. Verified the frozen SQG source checkpoint and established that the recipe
   does not read official BF16 routed weight shards.
4. Bound saved capture and preflight records to Hessian dataset revision
   `a05b3b92d749f6a641af5cfd52de2b4720380dfd`.
5. Qualified the preserved layer-3 K48 coupled artifact.
6. Completed no-shortcut profile and beta recipes for layers 4 through 22.
7. Completed exact K3/K4 triplet scoring and K96 allocations for layers 4
   through 18.
8. Encoded layers 4 through 18 with the selected layer-native K96 allocations.
9. Proved exact scorer/encoder payload parity for layers 4 through 18.
10. Selected coupled draws on disjoint selection data and wrote report-only
    holdout measurements for layers 4 through 18.
11. Materialized sealed, hash-bound selected shards for layers 4 through 18.
12. Passed native B12X runtime oracles for layers 3 through 18.
13. Archived compact candidate evidence for waves 3 through 18 and removed
    only validated rolling scratch.
14. Wrote 581 atomic exact-score receipts for layers 19 through 22 by the
    timestamp of this record.
15. Restored GPU service and the saved input volume after the driver incident.
16. Reconstructed and verified persistent ExLlamaV3 dependencies, patched all
    five reachable launchers, and resumed 32 exact-score workers.

## Remaining gates

The run must complete these conditions before a release or quality claim:

1. Complete the remaining 443 expert scores for layers 19 through
   22.
2. Complete allocation, encode, parity, selection, materialization, and native
   runtime oracles for layers 19 through 78.
3. Assemble all 76 routed layers into the hybrid K48/K96 checkpoint.
4. Pass the 76-layer codec census, including MTP layer 78.
5. Run the complete untrimmed TP4/PP1/DCP1 KLD comparison against the same
   sealed BF16 logits as the source checkpoint.
6. Pass lower mean, lower p99, lower worst-one-percent CVaR, zero nonfinite, and
   zero trimmed-position gates.
7. Pass the 16-token MTP3 TP1/PP4/DCP1 smoke.
8. Seal the reproduction bundle and SHA-256 manifest.

No candidate KLD estimate, lower-KLD claim, release claim, or completion claim
is supported by this status record.
