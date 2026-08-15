# GLM-5.2 K96 coupled-Hadamard distributed re-encode

Status: the encode, merge, local assembly, codec census, exact-r11
TP4/DCP4/MTP3 runtime, Estonia, and LAVD artifacts are **qualified**. Overall
model quality is **research-only** because candidate mean KLD
`0.1401771516114036` is 84.849 percent worse than source mean
`0.07583317451217256`. The hidden replay is **research-only** because its
original preregistered maximum-position gate fails. The public file set is
**unsupported** as a complete model until routed layers 3 through 50 finish
uploading and a hash-bound revision passes anonymous verification. This
dated document is the method, operations, incident, and reproduction ledger.

The public repository is
`brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail`. The complete checkpoint is
assembled locally and its codec census passes, but the public tensor and model-
card revisions remain unsealed. Do not treat that repository as a complete
runnable model until the unsupported publication conditions in section 11 are
resolved.

## 1. Authoritative model contract

The target is deliberately non-uniform across routed layers:

| Layer range | Treatment | K3 | K4 | Routed payload bpw |
|---|---:|---:|---:|---:|
| 3 | preserve the sealed coupled K48 exception | 720 | 48 | 3.0625 |
| 4--77 | layer-native K96 coupled re-encode | 672 | 96 | 3.125 |
| 78 | preserve the source MTP layer unchanged | 384 | 384 | 3.5 |

There are 76 routed layers in the assembled model. Seventy-five target layers
3--77 are coupled, but layer 3 is reused rather than re-encoded. The 74 K96
layers are 4--77. Target layers 3--77 average `3.1241666666666665` bpw.
Including preserved MTP78, the exact routed-layer average is
`3.1291118421052633` bpw. It is incorrect to describe this checkpoint as a
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
The paired weight reparameterization and runtime boundary transforms are exact
inverses and preserve the unquantized expert function. A plain block Hadamard
is self-inverse. For a signed transform `D H`, the inverse is `H D`, with the
matching gate/up interleave and split. H13 and H2 are transformed into those
same coordinates. GLM dimensions close the blocks exactly: hidden 6144 is
divisible by 512, interleaved gate/up width 4096 is divisible by 128, and
intermediate width 2048 is divisible by 128.

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
| Final serving base | exact Infernal Invocation r11/MTP3, digest `sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971` |
| Sealed KLD measurement image | `verdictai/glm52-k96-ii-r11:20260815-tpfix`, ID `sha256:79dfbf6e697a1081016a3256ff5c96b1c2d6dcddd6f04662c9c693881310fa87` |
| Final serving candidate | `verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix` |
| Final serving candidate image ID | `sha256:ab6bd60716b0a8e453b6345cb10e43e79726d92729b1f29058e31d7cc1c67def` |
| r11 vLLM / B12X trees | `908522a320ecc26582926228c9644af085f5a86c` / `5d648d944a047d4fac5c2035309c207b3faebd9c` |
| Native-SQG donor heads | vLLM PR #315 `ca966847`; B12X PR #197 `b234532` |
| Supplemental MTP3 loader patch SHA-256 | `658cbdd678774b0a1167c6244ff78d627b77613709f610e26e8d0e1933cfa03e` |
| Pre-LM-head capture patch SHA-256 | `d2521df3bed23f311ccd16c5ad46a20d46ae0d5674e1c34dd976bb88ebf002fd` |
| Capture Dockerfile SHA-256 | `885b05cdcb4dd929549e86a6e438f5e0b25d0c470eb97893df29b6c29d196660` |
| Final TP4/DCP4/MTP3 Compose SHA-256 | `c31f89f090e7f046bbd347a3b28d0c4f4486ad9b8e862645c4fcae1caa8c9c58` |
| Five-run quality entrypoint SHA-256 | `807afc1518207faece5a378c391561e6a23fd5fbfada8c3ff1361a18574415eb` |
| Idempotent post-task sealer SHA-256 | `68b59ddfe3fb667b60ac774135789bbc7d75ce462812659557607b50541c0b49` |
| Combined quality summary SHA-256 | `4f19ba5e4a8676c80bc49e89d346b0985faa209f14bdd6d9713e9ee6c4397f57` |
| Runtime MTP summary SHA-256 | `d8a7f22f6da05423a00d972e186deb17ba9f36133aae692c2477d53cd4f0f4ff` |
| Measured combined run `SHA256SUMS` SHA-256 | `bf57161cec69b9e8e8ebb5e705c1c5c6ed88c80046364c2a07a8882e60d0e3a9`, preserved as the decompressed hash of `SHA256SUMS.measured.gz` |
| Quality benchmark | `local-inference-lab/llm-inference-bench` v0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963`, script SHA-256 `59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7` |
| Final serving topology | TP4/DCP4/MTP3 |
| Final serving context ceiling | `MAX_MODEL_LEN=262144` |

The quality runner records the active container's `.State.StartedAt` and copies
only native-SQG rank receipts whose modification times are newer than that
boundary. This prevents receipts left by a previous server from satisfying the
four-rank acceptance gate. The sealed receipt selected exactly the four rank
PIDs `767`, `825`, `926`, and `1030`. All four have schema
`glm52-native-sqg-w4a8-rank-evidence-v1`, `complete=true`, topology TP4,
activation endpoint `full-w4a8`, routed experts enabled, and
`allow_a16_fallback=false`; every rank loaded and executed layers 3--78. The
sealed fatal-audit file contains zero matches.

The two frozen route-allocation inputs are:

- source KLD SHA-256
  `5d8aedb462658c693f1ce790f48ce5ed3cd6876897b1367a5cd36e42c0e2d434`;
- exact routed-expert planes SHA-256
  `db43b66f74745d25dfcbe7bc4baea5960a0a6a36bd1fec21533a42a380fc4d02`.

The source receipt is a complete 2,047-position TP4/PP1/DCP1 run with no
trimming and mean KLD `0.07583317451217256` (median
`0.0006116087315604091`). That source result is the comparator and allocation
signal source, not a prediction of the new model's final KLD.

The final runtime lineage is not r13 or v20. Infernal Invocation r13 contributes
only the native GLM SQG loader/kernel donor heads named above. Its image,
TP4/DCP1/MTP0 qualification, and quality/throughput results are not results for
this checkpoint. v20 is excluded from final code, serving, and result identity.

The complete vLLM patch and supplemental MTP3 loader patch passed an ordered
`git apply --check` and apply replay on clean base
`ce5f50f6d01b02336c4207f11277fd7bedacb4d6`. `py_compile`, Ruff, and
diff-check passed. A static real-image check resolves both the canonical
layer-78 prefix and the speculative `mtp_block` prefix. Full pytest did not run
because a test dependency was unavailable, so no passing full-pytest result is
claimed.
The final candidate reached a four-rank loader observation: all four ranks
reported `288G` main plus `4.28G` MTP loaded at `80.28 GiB` model memory per
rank. The later sealed four-rank execution receipt establishes the
target-topology native-SQG path through layers 3--78. The sealed KLD receipts
retain their exact earlier `tpfix` measurement-image identity and are not
retroactively relabeled with the final MTP3 image.

The first Estonia request used the earlier `MAX_MODEL_LEN=131072` ceiling and
was rejected with HTTP 400 because the full request envelope exceeded it. No
valid generation or Estonia score existed, so this is a non-model
request-envelope diagnostic rather than a task-quality result. The final
Compose default is `262144`. With the exact checkpoint tokenizer and chat
template, Estonia is 133,186 input tokens and has a 193,186-token request
ceiling with 60,000 allowed output tokens; LAVD is 20,310 input tokens and has a
100,310-token ceiling with 80,000 allowed output tokens. The existing
140,000-token DCP full-CKV gather capacity covers Estonia prefill, so no
CKV-capacity change or reload is needed.

The nominal runs ending `084104Z` and `084128Z` had overlapping benchmark
clients, invalidating the intended concurrency-1 Estonia condition. They are
preserved only as rejected overlap diagnostics and contribute zero qualifying
runs. Only the two client units were stopped; the exact-r11 server stayed
loaded and healthy and returned to zero running/zero waiting requests. The sole
clean supervised replacement is run `084427Z`. Its v0.4.29 Estonia stage
completed 5/5 valid and correct, with pass/fail `5/0` and zero truncations. The
result JSON SHA-256 is
`62551513dabdfdb0c389edc15220e03d3d95d12e5f85e26125ca0ba8962b6e92`.
Completion tokens averaged `3371.6` (p50 `2980`, p90 `5068.6`, p99 `5859.16`),
aggregate generation was `42.67293371871177 tok/s`, mean per-request generation
was `43.5170233617816 tok/s`, mean elapsed time was `79.91317092299869 s`, and
mean TTFT was `0.9025994309980888 s`. Its reported `148,287 tok/s` prefill scout
reused the prefix cache warmed by the rejected overlap diagnostics, so it is
not a cold or uncached prefill measurement.

LAVD completed 5/5 valid and correct with four exact results and one near
result. Runs 1--4 produced exactly `72,46`; run 5 produced `71,45.75`, a
count delta of `-1` and hours delta of `-0.25`. There were no failures or
truncations. The raw result SHA-256 is
`2a9e2772049be73e59d14a31f1f47b17dbcd871f536af6a7f09febc8ab29b4e1`.
Completion tokens averaged `16682.2` (p50 `17738`, p90 `18309.8`, p99
`18447.68`), aggregate generation was `22.13984480814618 tok/s`, mean
per-request generation was `22.087665860284208 tok/s`, mean elapsed time was
`768.5865516199963 s`, and mean TTFT was `15.050551386599546 s`. The frozen
dataset SHA-256 is
`612f8041bbca048c044dd77ebd58964afded85b0d76513c013715a401b09dc34`
and prompt SHA-256 is
`5c83674d5f0fd2a727bf11c521a765f8be4a13087714a2151fc96078018c4aa0`.

The combined run is sealed at
`evidence/final-exact-ii-r11/qualification-5x/exact-r11-tp4dcp4mtp3-20260815T084427Z`.
`qualification.complete` is present and every entry in its
`SHA256SUMS` verifies. The original runner's post-task assertion expected
layers 3--77, but the final MTP3 composition correctly loads and executes the
preserved source-SQG MTP layer 78. The bound runner checks layers 3--78,
supports `QUALITY_REUSE_RESULTS_DIR`, captures final MTP metrics, and can seal
the already-valid task outputs without repeating them.

The byte-bound publication helper
`runtime/exact-ii-r11/deploy/accelerate-hf-routed-upload.sh` (SHA-256
`a4469e0f1fdcdba70b95e41fe442edd149bbabfc192b78c095115966c1a733bc`)
uploads only local routed layers 003--050 and preserved MTP layer 078, skipping
byte-identical layers 051--077. The active targeted upload restarted at
`2026-08-15T05:39:16-04:00` with one outer client and default adaptive Xet.
Adaptive concurrency began at 2. Early sustained evidence showed 22.09 Mbit/s,
success ratio 1.0, and zero errors. This first stage is not authoritative
evidence of full checkpoint completeness or a reproducible completion ETA.

The authoritative resumable full-folder verification/upload stage is
`runtime/exact-ii-r11/deploy/complete-final-model-upload.sh`, SHA-256
`f6b3352db4a288793254bd0c2656e31c805e3718b23fe2239a6649fcd7646d46`,
run by enabled persistent user unit
`glm52-k96tail-final-model-upload.service`, unit SHA-256
`f73b3a35718ff43955c37f4a9f9425b9f117576b260fba684130ce6b8ed7e1e8`.
It waits for the routed first stage and then verifies/uploads routed, K6, BF16,
configuration, and sidecar files as one canonical model folder.

The persistent final-release gate is
`runtime/exact-ii-r11/deploy/wait-and-publish-hf-release.sh`, SHA-256
`6c57cad0226dbd77e7f9eb660e42c344e58a4ed4868e07b4d0a7e01c754c35ae`.
It is run by enabled persistent user unit
`glm52-k96tail-final-hf-publication.service`, unit SHA-256
`ff033d7d3573bedf282b1535065aaf9f1a9d719cd53f0cd1fc89fb74be4eda62`.
It requires both `results/hf-publication.ready` and successful canonical upload,
then uploads the cache-excluded browsable reproduction, scripts, and sealed
results; publishes the model card; verifies anonymous file presence and exact
card bytes; promotes and uploads verified `RELEASE_PROVENANCE.json`; and writes
`hf-publication.json`/`hf-publication.complete`. The readiness marker remains
intentionally absent until the final local card, provenance, and manifests are
closed and reviewed.

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

The terminal preparation contract is the sealed wave-074--077 input view,
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
`reproduction/k96tail-distributed-campaign.json`. Its phase ledger records
encoding, distributed merge, local assembly, codec census, both KLD methods,
exact-r11 serving qualification, and both task suites as complete. It retains
`complete=false` while public tensor/card revisions and anonymous verification
remain null. The read-only audit entrypoint is:

```bash
python3 scripts/audit_k96tail_campaign.py
```

The completed local closure is checked with:

```bash
python3 scripts/audit_k96tail_campaign.py --strict-local
```

After every required result and publication field is sealed, update the manifest
and run:

```bash
python3 scripts/audit_k96tail_campaign.py --strict-complete
```

The strict local audit requires passing runtime oracles for all target layers
3--77 and every non-Hub final field. `--strict-complete` additionally requires
`complete=true` and no null Hub publication field.

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
10. Run and seal the complete untrimmed 2,047-position full-vocabulary KLD
    method against the BF16 reference at TP4/PP1/DCP1. Never remove positions
    from the acceptance mean.
11. Run and seal the pre-LM-head hidden-replay method on `[2048,6144]` BF16
    hidden states, scoring 2,047 positions through the unchanged
    `[154880,6144]` BF16 LM head. Compare it position-for-position with method
    10; neither method may silently substitute top-k probabilities.
12. Start only the derived exact Infernal Invocation r11/MTP3 image at
    TP4/DCP4/MTP3. The quality entrypoint pinned in section 4 must seal
    `server.log` and four receipts with schema
    `glm52-native-sqg-w4a8-rank-evidence-v1`, plus target-topology logs,
    graph/smoke evidence, MTP draft-shard evidence, and a DCP correctness
    discriminator. r13 and v20 images are forbidden substitutes.
13. After serving acceptance, run Estonia five times at concurrency 1 and LAVD
    five times at concurrency 5 with the pinned v0.4.29 benchmark; preserve raw
    JSONs/logs, benchmark provenance, and hashes. Reject any receipt that does
    not attest benchmark version `0.4.29`.
14. Seal the SHA-256-indexed reproduction bundle with every measured pass and
    failure. Mark the checkpoint research-only when the KLD quality gate fails;
    do not suppress the receipt or convert mechanical qualification into a
    quality pass.

The final sealed bundle copies all Python/shell entrypoints, local and remote
campaign logs, wave metadata archives, layer manifests/quality/oracles,
allocations, profiles, scores, supervisor configs, runtime overlay, Docker
context, model codec receipt, both KLD receipts/comparison, exact-r11 target
serving receipt, Estonia/LAVD raw results, this document, and a top-level
`SHA256SUMS`.

## 8. Completion ledger and historical snapshot

Evidence sealed on 2026-08-15:

- All target layer encodes are complete and sealed.
- All remote layers and reproduction evidence are merged locally.
- The final mixed-rate checkpoint assembly is complete.
- The 76-routed-layer codec census passes with 75 coupled target layers and
  preserved MTP78.
- All 75 target layers have complete mechanical manifests, quality receipts,
  and passing runtime oracles. Exact K96 scorer/encoder parity covers the 74
  K96 layers 4--77. Layer 3 is the sealed K48 exception and has no K96 parity
  receipt.
- `evidence/final-mechanical/MECHANICAL_EVIDENCE.json`, SHA-256
  `e74fb8907a0002045d9ed66cdb16c69635c04d2bcef0fef7eb45beec72a48a94`,
  binds those receipts, the assembly manifest, and the authoritative codec
  receipt.
- Exact-r11 TP4/DCP1 full-vocabulary KLD is sealed locally: mean
  `0.1401771516114036`, receipt SHA-256
  `7979c9c8b0c81714cd38e225646e42a88b2cb8eb03232be271373255c506a408`.
- The frozen source mean is `0.07583317451217256`. Candidate minus source is
  `0.06434397709923104`; the candidate is `1.84849x`, or 84.849 percent worse.
  The lower mean, lower p99, and lower worst-1% CVaR gates fail. Overall model
  quality is **research-only**.
- Pre-LM-head hidden replay is sealed locally: mean KLD
  `0.1401762649458023`, top-1 agreement `0.9174401563263312`, receipt SHA-256
  `34f14cada0424ddb1387fec78a96a16ebe2109ddf2c063862d52108d0450e6b2`.
- The original preregistered hidden-replay receipt has SHA-256
  `7433ad312740dabaa1f3dc0c6e2a8317e741a441bccf06eb4a6f2ebe0569b3ad`
  and `qualification_pass=false`. Mean absolute delta
  `8.866656012740393e-7` passes its `5e-5` limit, but maximum position delta
  about `0.0035558` fails its `5e-4` limit. Hidden replay is
  **research-only**.
- Exact-r11 TP4/DCP4/MTP3 qualification is sealed locally with four bound
  native-SQG rank receipts, full-W4A8 execution, no A16 fallback, layers 3--78,
  and zero fatal-audit matches.
- Estonia is 5/5 valid and correct. LAVD is 5/5 valid and correct, with four
  exact results and one disclosed near result.
- The machine contract remains `complete=false` because the public tensor and
  card revisions are **unsupported** until the upload and anonymous
  verification gates pass.

The hidden receipt uses schema `glm52-hidden-replay-one-context-kld-v2` over
2,047 next-token positions. Its `[2048,6144]` BF16 capture manifest is
`raw-20260815T075846Z/hidden/manifest.json`, SHA-256
`12412b014f730a89890f7b9416c6498976139d960498d4cdf60058df2f639cc1`.
The unchanged `[154880,6144]` BF16 LM-head tensor is byte-identical between
source and candidate.

The v2 native-versus-replay bounds are explicitly **post-observation**
operational compatibility bounds for deterministic BF16 offline-LM-head
reduction-order differences, not a preregistered scientific model-quality
gate:

| Comparison | Limit | Observed | Pass |
|---|---:|---:|---:|
| Mean absolute KLD delta | `5e-5` | `8.866656012740393e-7` | yes |
| Position absolute KLD delta, p99 | `1e-4` | `6.126456900251877e-5` | yes |
| Position absolute KLD delta, maximum | `5e-3` | `0.0035557552471569` | yes |

The independently repeated native full-logit receipt has SHA-256
`a67e896ffc1f879cdb3320042c50981532ce7837e0375fe34bcb0729fe685fc7`.
Its mean delta and per-position maximum delta versus the baseline native run
are both exactly `0.0`, so the repeated native result is exact at every scored
position. These observations pass the post-observation operational envelope.
They do not qualify the replay under its original preregistered gate and do not
establish model quality.

The remainder of this section is the earlier in-flight historical snapshot, not
an automatically updating status page.

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

Earlier transfer configurations are historical diagnostics, not the active
setting. The active targeted upload restarted at
`2026-08-15T05:39:16-04:00` with one outer client. The uploader unsets
`HF_XET_HIGH_PERFORMANCE` and `HF_HUB_DISABLE_XET`, so Xet uses its default
adaptive mode. Adaptive concurrency began at 2. Early sustained evidence
showed 22.09 Mbit/s, success ratio 1.0, and zero errors. Remote layer and
evidence bytes remain hash-gated. No completion ETA is sealed as a canonical
property.

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

The exact-r11 runtime closure adds these entrypoints:

```text
runtime/exact-ii-r11/deploy/run-kld-exact-r11.sh
runtime/exact-ii-r11/hidden-replay/run_glm52_hidden_replay.sh
runtime/exact-ii-r11/deploy/run-quality-5x.sh
runtime/exact-ii-r11/deploy/seal-quality-5x.sh
runtime/exact-ii-r11/deploy/wait-and-upload-final-model.sh
runtime/exact-ii-r11/deploy/verify-final-hf-model.py
runtime/exact-ii-r11/deploy/wait-and-publish-hf-release.sh
```

The measured benchmark source is stored exactly as
`runtime/exact-ii-r11/benchmarks/llm_decode_bench.py.gz`. The quality runner
extracts it to a temporary file and requires decompressed SHA-256
`59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7`
before execution. The adjacent `llm_decode_bench.ascii.py` is a readable
ASCII-normalized derivative and is explicitly not byte-identical to the
measured source.

`verify_runtime.py` is an image gate, not an unqualified host static check. The
host vLLM installation lacks the required exact-r11 symbol set. Re-run it only
inside the hash-bound candidate image:

```bash
docker run --rm \
  --entrypoint /opt/venv/bin/python \
  verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix \
  /opt/ii-r11-k96/verify_runtime.py
```

The supervisor configs encode the exact assignment in section 5. Reproduction
must supply valid Hub credentials, the pinned images/runtime dependencies, and
the absolute storage layout expected by the scripts. It must not copy provider
portal/tunnel processes into the campaign or start any competing run-ahead
scorer.

## 11. Result and publication boundary

The machine-contract copies retain `complete=false`. Public upload completion
would not change the model's research-only quality status.

| Object | Status | Value |
|---|---|---|
| Encoding and distributed merge | qualified | All target layers are sealed. |
| Local assembly and codec census | qualified | All 76 routed layers pass, including MTP78. |
| Full-vocabulary KLD measurement | research-only | Candidate mean `0.1401771516114036`, 84.849 percent worse than the source. |
| Hidden-replay comparison | research-only | Original preregistered maximum-position gate fails. |
| Exact-r11 TP4/DCP4/MTP3 runtime | qualified | Quality summary SHA-256 `4f19ba5e4a8676c80bc49e89d346b0985faa209f14bdd6d9713e9ee6c4397f57`. |
| MTP3 metrics | qualified | Summary SHA-256 `d8a7f22f6da05423a00d972e186deb17ba9f36133aae692c2477d53cd4f0f4ff`. |
| Estonia 5x | qualified | 5/5 correct, zero truncations. |
| LAVD 5x | qualified | 4 exact plus 1 near, zero truncations. |
| Public tensor revision | unsupported | Routed layers 3 through 50 are still uploading. |
| Public model-card revision | unsupported | The partial public file set has not passed anonymous hash verification. |

Server-side copy commit
`0c38e683eda27ca84982e3d513c89dd780dcdb22` verified 390 byte-identical
files totaling 25,685,857,224 bytes from source revision
`593dd0d2de6f79ce4e65303930c22c75e1359d44`. Layers 51--77 were already
present and MTP78 was copied. Two outer clients briefly measured 21.75 Mbit/s,
but that superseded configuration is not the active setting. The targeted
upload restarted at `2026-08-15T05:39:16-04:00` with one outer client and
default adaptive Xet. Adaptive concurrency began at 2. Early sustained evidence
showed 22.09 Mbit/s, success ratio 1.0, and zero errors. No completion ETA is
part of this canonical record.

`wait-and-upload-final-model.sh` does not wait for the intentionally stopped
qualification server. It excludes `README.md` and
`HUB_FILE_VERIFICATION.json`, uploads the remaining file set with two workers,
and runs `verify-final-hf-model.py`. The verifier compares every other local
file to Hugging Face file metadata. It checks LFS SHA-256 plus size or Git blob
SHA-1 plus size, writes the receipt atomically, and permits receipt upload only
when `complete=true`. Wrapper SHA-256:
`ce467eb1218cc8a5d9334916ea39d2d3d816b46df73cd5496b1e97b9576ac63a`.
Verifier SHA-256:
`aebc8b97fe332eb5b086e8f62b698e631fa10814f5fdf9504d201abb366efa11`.

No serving or task conclusion is inferred from the one-context KLD receipts;
those claims come from the separate sealed exact-r11 qualification bundle. No
public-release conclusion is made until anonymous Hub verification passes.
