# GLM-5.2 K96 coupled-Hadamard distributed re-encode

Status: live encode and merge campaign on 2026-08-14. This document is the
method, operations, incident, and reproduction ledger. A line marked
**PENDING FINAL** is not a measured result and must not be converted into a
claim before the corresponding sealed receipt exists.

The public staging repository is
`brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail`. It remains an incomplete
merge bus, not a runnable model, until the final local assembly, codec census,
MTP3 smoke, and full end-to-end KLD gates have passed.

## 1. Authoritative model contract

The target is deliberately non-uniform across routed layers:

| Layer range | Treatment | K3 | K4 | Routed payload bpw |
|---|---:|---:|---:|---:|
| 3 | preserve the sealed coupled K48 exception | 720 | 48 | 3.0625 |
| 4--77 | new layer-native K96 coupled re-encode | 672 | 96 | 3.125 |
| 78 | preserve the source MTP layer unchanged | 384 | 384 | 3.5 |

There are 76 routed layers in the assembled model. Seventy-five target layers
3--77 are coupled, but layer 3 is reused rather than re-encoded. The 74 new
K96 layers are 4--77. Including preserved MTP78, the exact routed-layer average
is `3.1291118421052633` bpw. It is incorrect to describe this checkpoint as a
uniform 3.0625-bpw or uniform K96 model.

Only the routed-expert shard and sidecar for each target layer are replaced.
Assembly hard-links every other plain source-model file and regenerates only
the model index/config/provenance records. Consequently the source checkpoint's
BF16/nonexpert tensors, embeddings, attention tensors, normalization tensors,
routers, LM head, and other non-routed-expert components remain byte-identical.
MTP layer 78's routed-expert shard and sidecar are also retained and sealed by
hash in `COUPLED_REENCODE_MANIFEST.json`.

The weight source is the frozen SQG checkpoint, not a newly downloaded BF16
model. Saved calibration captures and Hessians are loaded from the pinned
dataset. Every no-shortcut recipe receipt records
`source_is_frozen_sqg_checkpoint=true` and
`official_bf16_weight_shards_read=false`.

## 2. Coupled coordinate transform

This is the updated-QSRT coupled transformation, not just a different bit
allocation. For each routed `(gate, up, down)` expert triplet, the encoder uses:

- a coupled residual Hadamard in H512 blocks;
- an H128 preactivation transform on the interleaved gate/up boundary;
- an H128 postactivation transform on the intermediate/down boundary;
- exact GLM `silu(gate) * up` arithmetic with FP32 accumulation;
- H13 local alpha `0.25`; and
- candidate-conditioned downstream H2/down construction.

The pair of H128 transforms is the requested coupled or "double Hadamard"
treatment around the nonlinear boundary; H512 closes the residual coordinate.
The weight map is an exact self-inverse reparameterization and the H13/H2
Hessians are transformed into the same coordinates. GLM dimensions close the
blocks exactly: hidden 6144 is divisible by 512, interleaved gate/up width 4096
is divisible by 128, and intermediate width 2048 is divisible by 128.

The final runtime still uses standard route-packed direct-E4M3 SQG W4A8
payloads. Coupling changes the coordinates presented to quantization and the
associated Hessians; it does not introduce an unserved payload format.

## 3. No-shortcut per-layer recipe

Layers 4--77 must independently complete this sequence:

1. Load the pinned saved activation, profile scales, H13, H2/capture evidence,
   frozen SQG source layer, and execution binding. No official BF16 weight
   shard is loaded.
2. Run the complete 16-cell beta-0.0625 bootstrap profile.
3. Run the seven-beta selection panel. Profile selection occurs before rate
   allocation, and holdout is excluded from the choice.
4. If the selected beta differs from 0.0625, run exactly one fresh 16-cell
   final profile plus its holdout. If it does not differ, reuse the bootstrap
   profile byte-for-byte. The final binding requires exactly one of these two
   states.
5. Score all eight realized gate/up/down K3/K4 triplets for each of 256
   experts using the selected layer-native profile.
6. Solve the exact triplet dynamic program at a fixed budget of 96 K4 tensors.
   Then apply the frozen source checkpoint's TP4/PP1/DCP1 worst-40 KLD/route
   signal as an in-sample allocation heuristic. The chosen allocation must
   remain within both the 1% total and 1% body calibration regression guards.
   This signal is not end-to-end acceptance evidence.
7. Encode both draw candidates 0 and 6 for all experts and all three
   projections, writing one atomic JSON/safetensors pair per expert/draw.
8. Prove score/encoder payload parity for all 256 experts and all 768
   projection payloads before selection.
9. Run the frozen draw fit/selection phases, then the independent selected-mix
   holdout. Fit may propose draw 6; disjoint selection must confirm it.
   Holdout never selects bytes.
10. Materialize the selected 2,306-key runtime layer, seal its K3/K4 census,
    hashes, profile/allocation/selection lineage, and quality receipt.
11. Run the native B12X `route_packed_direct_e4m3_w4a8` oracle and require
    complete, finite, nonzero output and `pass=true`.

The B300 owner-fixed-beta shortcut, fleet-beta reuse, identity-only rescue, and
same-rate batched encoder shortcut are all forbidden. The earlier real-CUDA
batch audit found that same-rate grouping could change bytes, so the final
encoder remains singleton per tensor even when experts are scheduled in
parallel.

## 4. Immutable inputs and executable pins

| Component | Pinned identity |
|---|---|
| Frozen source checkpoint | `brandonmusic/GLM-5.2-SQG-W4A8@593dd0d2de6f79ce4e65303930c22c75e1359d44` |
| Hessian/capture dataset | `brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians@a05b3b92d749f6a641af5cfd52de2b4720380dfd` |
| QSRT source | `453b4834332d2735c5a326ca57fb6a8b36e776bf` |
| QSRT tracked diff SHA-256 | `33982c45a93c9291a5e0e63a40dd638863dc018bf1dcb9987e235d93e2eb278d` |
| Encoder image ID | `sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82` |
| SQG extension SHA-256 | `d29010f6ad51caf2e1a22f07365ab3548fcdb3e0ed3ee15d88330cee24de9614` |
| ExLlamaV3 extension source image | `sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac` |
| ExLlamaV3 extension SHA-256 | `e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd` |
| Approved KQuant backend SHA-256 | `fb63082016d2adc331be58f538622e1382c86390b0d1715c9a755433fb14c624` |
| Approved KQuant tracked diff SHA-256 | `82c994a6fa1e1c996f18c85f723c562fbe08bb8a7edf705ef9ece1ae1606958f` |
| Approved KQuant status SHA-256 | `8ad2cf65ff6656fd4cbf01df725a8d6055229f90142997ae90176cf3b37909ed` |
| Final serving/KLD image ID | `sha256:e5bdd0598179b7b695501a3d7a1ee33b04233eec7848e4f4588c6afebc402ee9` |

The two frozen route-allocation inputs are:

- source KLD SHA-256
  `5d8aedb462658c693f1ce790f48ce5ed3cd6876897b1367a5cd36e42c0e2d434`;
- exact routed-expert planes SHA-256
  `db43b66f74745d25dfcbe7bc4baea5960a0a6a36bd1fec21533a42a380fc4d02`.

The source receipt is a complete 2,047-position TP4/PP1/DCP1 run with no
trimming and mean KLD `0.07583317451217256` (median
`0.0006116087315604091`). That source result is the comparator and allocation
signal source, not a prediction of the new model's final KLD.

## 5. Distributed assignment

The rental fleet used four independent 8-GPU RTX PRO 6000 Blackwell nodes.
The public repo is the durable artifact bus; the final model is assembled only
on the local machine.

| Worker | Layers | GPUs used |
|---|---:|---:|
| Local four-GPU host | 47--50 | 0--3 |
| Node 1 wave A | 51--54 | 0--3 |
| Node 1 wave B | 55--58 | 4--7 |
| Node 2 wave A | 59--62 | 0--3 |
| Node 2 wave B | 63--66 | 4--7 |
| Node 3 wave A | 67--70 | 0--3 |
| Node 3 wave B | 71--74 | 4--7 |
| Node 4 terminal wave | 75--77 | 0--7 |

The terminal preparation contract is the immutable wave-074--077 input view,
but `RUN_LAYERS=75,76,77`; no new layer-74 worker is launched on node 4. This
allows the three-layer terminal wave to consume its pinned parent capture
without duplicating layer 74.

Supervisor configuration lives under `vast_supervisor/`. Preparation downloads
the exact source routed-expert shards and pinned Hessian/capture waves. Each
`run_vast_wave_and_upload.sh` invocation runs the ordinary campaign in
`PARTIAL_ONLY=1` mode, so a rental produces sealed layers but never attempts a
global assembly or KLD run.

## 6. Artifact and reproduction layout

Each sealed layer publishes four top-level files:

```text
r7-experts-layer-NNN.safetensors
r7-experts-layer-NNN.json
r7-experts-layer-NNN.quality.json
runtime-oracle-layer-NNN.json
```

Each remote wave also publishes:

```text
reproduction/wave-AAA-BBB/SHA256SUMS
reproduction/wave-AAA-BBB/campaign.log
reproduction/wave-AAA-BBB/recipe-and-profiles.tgz
reproduction/wave-AAA-BBB/tail-scores.tgz
reproduction/wave-AAA-BBB/allocations.tgz
reproduction/wave-AAA-BBB/parity-proofs.tgz
reproduction/wave-AAA-BBB/candidate-metadata.tgz  # present when retained
```

`SHA256SUMS` covers every mandatory archive and campaign log. Candidate
safetensor scratch is deliberately excluded from `candidate-metadata.tgz`
because the selected runtime shard and its hashes are the durable payload;
recipe, scores, allocation, parity, selection metadata, and logs remain.

The machine-readable campaign contract is
`reproduction/k96tail-distributed-campaign.json`. It intentionally has
`complete=false` and null final-result fields while the run is live. The
read-only audit entrypoint is:

```bash
python3 scripts/audit_k96tail_campaign.py
```

After final assembly and KLD, update the manifest's final fields and run:

```bash
python3 scripts/audit_k96tail_campaign.py --strict-complete
```

The strict audit requires passing runtime oracles for all target layers 3--77,
`complete=true`, and no null final-result field.

## 7. Merge, assembly, and acceptance sequence

The persistent local finalizer performs the following fail-closed sequence:

1. Wait for passing local runtime oracles for layers 47--50.
2. Stage and asynchronously upload those four layer quartets.
3. Poll the public Hub for every mandatory remote layer file and reproduction
   file for waves 051--054, 055--058, 059--062, 063--066, 067--070, 071--074,
   and 075--077.
4. Download the complete remote set; validate schema, K96 census, bpw,
   no-shortcut final-profile binding, shard SHA-256, quality receipt, and native
   runtime oracle for every layer.
5. Quarantine any incomplete local run-ahead recipe/profile directory for a
   remote-owned layer. Verify every wave `SHA256SUMS`, reject unsafe tar paths,
   and extract recipe, score, allocation, parity, and candidate metadata.
6. Merge validated layer files into the local sealed layer root and re-run all
   target-layer oracles.
7. Remove the intentional local wave-boundary stop marker and let the canonical
   campaign perform its full cross-layer validation.
8. Assemble a new, non-overwriting model directory from the frozen source plus
   coupled target layers 3--77. Preserve source MTP78 and every unchanged file.
9. Run the full 76-routed-layer codec census. It must report 75 coupled target
   layers and an unchanged MTP78.
10. Run the complete untrimmed 2,047-position KLD against the sealed BF16 logits
    at TP4/PP1/DCP1. Report mean, median, p95, p99, worst-1% CVaR, maximum, and
    the diagnostic source-worst-40 alignment; never remove positions from the
    acceptance mean.
11. Start the exact production MTP3 regime at TP1/PP4/DCP1 and require a clean
    16-token smoke receipt.
12. Seal `FULL_ACCEPTANCE.json` and a SHA-256-indexed reproduction bundle only
    if codec, KLD distribution, no-trimming, quality, MTP3, and all layer gates
    pass.

The final sealed bundle copies all Python/shell entrypoints, local and remote
campaign logs, wave metadata archives, layer manifests/quality/oracles,
allocations, profiles, scores, supervisor configs, runtime overlay, Docker
context, model codec receipt, KLD tail analysis, MTP3 receipt, this document,
and a top-level `SHA256SUMS`.

## 8. Live measured ledger

This section is a historical snapshot, not an automatically updating status
page.

At `2026-08-14T23:55:09-04:00`:

- Layers 3--46 had 44/44 passing native runtime oracles. Layer 3 had the K48
  census and layers 4--46 had the K96 census.
- Across those 44 oracles, mean cosine averaged `0.9989899491721933` and ranged
  from `0.9988675117492676` to `0.9992380142211914`. Relative RMSE averaged
  `0.0449653732675043` and ranged from `0.03991738334298134` to
  `0.04780655726790428`.
- Every sealed K96 layer 4--46 had a complete production-eligible guarded K96
  allocation: 43/43.
- Every sealed K96 layer 4--46 had exact scorer/encoder parity for all 256
  experts and all 768 projection payloads: 43/43.
- The local layer-47--50 scorer had completed 926/1,024 paired expert receipts:
  L47=208, L48=243, L49=256, L50=219. These are resumable atomic receipts, not
  completed layers.
- The local encode/finalizer services remained active. Remote waves were still
  in recipe/scoring execution and had not all published their sealed Hub
  quartets, so no end-to-end candidate KLD existed.

Earlier sealed wave checkpoints and per-layer runtime metrics through layer 46
are retained in
`/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/COUPLED_REENCODE_RUNTIME_RECOVERY_20260814.md`
and the per-layer oracle JSON files. The final result must come from
`coupled-k96tail-full-tp4dcp1/FULL_ACCEPTANCE.json`, not from extrapolating
these layer-local metrics.

## 9. Operational incidents and recovery

### Driver/GSP recovery

The original local campaign encountered an NVIDIA GSP watchdog failure during
wave 019--022. Atomic receipts stopped at 579/1,024. A host reboot reloaded the
driver; the campaign resumed those receipts rather than restarting. Persistent
ExLlamaV3 Python and extension dependencies were recovered from pinned images
and verified file-for-file/hash-for-hash. No source or candidate bytes were
accepted without the existing seals.

### Executing-shell edit

An earlier controller read a launcher while that same shell file was edited in
place, later reached a changed file offset, and exited with status 127. Atomic
receipts were preserved. The operational rule is now strict: never edit an
executing launcher inode. Future fixes are made before launch or in a new file,
then the owning wrapper is intentionally restarted by the operator.

### Unsafe remote score run-ahead overlap

An experimental remote run-ahead configuration scored the first 32 experts of
selected individually sealed layers. On node 1, a layer-57 helper overlapped
the canonical 55--58 scorer for about 23 seconds. All run-ahead programs were
stopped, their configs were disabled and retained under
`vast_supervisor/disabled-runahead-configs/`, and a one-second supervisor guard
was installed on nodes 1--3. The guard stops any program named
`k96-score-runahead*`, quarantines matching configs, and pins the canonical
config name to `/dev/null`.

Completed JSON/NPZ pairs were valid deterministic receipts and remain
resumable by the canonical scorer. Pair audits found no one-sided run-ahead
artifacts on nodes 2 or 3. A temporary one-sided layer-55 expert-129 file on
node 1 belonged to the canonical active worker, not the layer-57 helper; it
was left for its owner to finish rather than being deleted. No run-ahead helper
is part of the reproduction schedule.

### Frozen KLD/routes absent on rentals

Some rental workspaces did not initially contain the frozen source KLD and
exact route planes required by the guarded allocator. They were copied from
the sealed local acceptance tree, not regenerated. Both source and destination
were checked against the fixed hashes in section 4 before use. Those files are
fit/allocation evidence only; remote workers still do not run final KLD.

### Hub Xet timeout/stall

The large local layers-003--046 upload initially stopped making write progress
and retained only a closing/stale connection pattern. It was restarted as a
persistent user unit with `HF_HUB_DISABLE_XET=1`, four legacy large-folder
workers, and restart-on-failure. Staging content was not rewritten. Remote
layer and evidence uploads remain independently hash-gated, so transfer backend
changes do not change accepted bytes.

## 10. Runnable entrypoints

The canonical entrypoints are intentionally small wrappers over the same
pipeline used locally:

```text
scripts/download_coupled_wave_inputs.sh       fetch pinned saved inputs
scripts/run_coupled_recipe_wave.sh            full no-shortcut profile recipe
scripts/run_coupled_tail_score_wave.sh        eight-triplet scoring
scripts/run_coupled_transcode_wave.sh         exact K96 candidate encode
scripts/validate_coupled_score_encode_parity.py
scripts/run_score_select_coupled_wave.sh      draw selection and holdout
scripts/run_materialize_coupled_wave.sh       selected runtime layer
scripts/run_validate_coupled_runtime_layer.sh native B12X oracle
scripts/run_full_coupled_3p0625_campaign.sh    resumable wave controller
scripts/prepare_vast_k96_node.sh               rental input preparation
scripts/run_vast_wave_and_upload.sh            rental partial campaign/upload
scripts/archive_vast_k96_evidence.sh           wave evidence archive/upload
scripts/check_k96tail_hub_ready.py             public merge-bus readiness
scripts/wait_merge_finalize_k96tail.sh         verified merge/finalize wait
scripts/run_finalize_k96tail_model.sh          assembly/KLD/MTP3/release
scripts/seal_coupled_k96tail_release.py        final receipt/reproduction seal
scripts/audit_k96tail_campaign.py              read-only contract audit
```

The supervisor configs encode the exact assignment in section 5. Reproduction
must supply valid Hub credentials, the pinned images/runtime dependencies, and
the absolute storage layout expected by the scripts. It must not copy provider
portal/tunnel processes into the campaign or start any competing run-ahead
scorer.

## 11. Final result placeholders

Fill these only from sealed final artifacts, then mirror the values into
`reproduction/k96tail-distributed-campaign.json` and set its `complete` field
to true.

| Final field | Value |
|---|---|
| Public model commit | **PENDING FINAL** |
| Assembly manifest ID / SHA-256 | **PENDING FINAL** |
| Model codec receipt SHA-256 | **PENDING FINAL** |
| Candidate KLD receipt SHA-256 | **PENDING FINAL** |
| Full mean KLD | **PENDING FINAL** |
| Median / p95 / p99 | **PENDING FINAL** |
| Worst-1% CVaR / max | **PENDING FINAL** |
| Candidate-minus-source mean KLD | **PENDING FINAL** |
| MTP3 smoke SHA-256 | **PENDING FINAL** |
| `FULL_ACCEPTANCE.json` SHA-256 | **PENDING FINAL** |

No model-quality or KLD conclusion is made until these fields are sealed.
