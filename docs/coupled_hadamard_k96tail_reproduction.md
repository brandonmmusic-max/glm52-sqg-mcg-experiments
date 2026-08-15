# Reproducing the GLM-5.2 coupled-Hadamard K96-tail re-encode

This procedure builds the coupled H512/H128 routed-expert checkpoint from the
frozen SQG W4A8 model and the saved calibration dataset. It does not download
or read the official BF16 routed weight shards.

The procedure is `implemented` in the local campaign workspace. Mechanical
checkpoint construction and the TP4/DCP4/MTP3 runtime are `qualified`. Overall
model quality is `research-only` because exact full-vocabulary KLD regresses
against the frozen source. The
[campaign source snapshot](../coupled_hadamard_k96tail/README.md) mirrors the
active executable sources, exact QSRT and KQuant patches, changed-file
snapshots, and final runtime build context. The verifier identifies byte-equal
active files and validates the one finalized machine-contract normalization
against the hash-bound historical contract. It does not claim that finalized
documentation is byte-identical to pre-finalization active documentation. Its
automated integrity check passes against `SOURCE_SHA256SUMS`. The optional
active-workspace comparison is fail-closed; inspect any difference before
using an active tree. The repository
is not a standalone model payload: the
pinned checkpoint, saved captures, container images, and compiled extensions
remain external artifacts. Full cross-host execution is therefore
`unsupported` until a second host proves access to those pinned artifacts. Do
not substitute source trees that lack the listed revision and patch hashes.

## Required artifacts

| Role | Durable identity |
|---|---|
| Frozen SQG source checkpoint | `brandonmusic/GLM-5.2-SQG-W4A8` at revision `593dd0d2de6f79ce4e65303930c22c75e1359d44` |
| Saved calibration dataset | `brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians` at revision `a05b3b92d749f6a641af5cfd52de2b4720380dfd` |
| QSRT source | revision `453b4834332d2735c5a326ca57fb6a8b36e776bf` plus tracked diff SHA-256 `33982c45a93c9291a5e0e63a40dd638863dc018bf1dcb9987e235d93e2eb278d` |
| KQuant source | revision `104dd9233f850a3955f4991bea68b07dd34deeb8` plus tracked diff SHA-256 `82c994a6fa1e1c996f18c85f723c562fbe08bb8a7edf705ef9ece1ae1606958f` |
| Encode container | image ID `sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82` |
| ExLlamaV3 Python package | extracted from the encode container with tree SHA-256 `834c8de389c700126e5746e7bae9876b3e443014480651e59919cdd68fc20506` |
| ExLlamaV3 extension source | image `verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a`, image ID `sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac` |
| SQG encode extension | SHA-256 `d29010f6ad51caf2e1a22f07365ab3548fcdb3e0ed3ee15d88330cee24de9614` |
| ExLlamaV3 extension | SHA-256 `e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd` |
| Campaign and runtime source | [`coupled_hadamard_k96tail`](../coupled_hadamard_k96tail/README.md), verified by `SOURCE_SHA256SUMS` |

The source checkpoint must also match the manifest, model-index, and
quantization-config hashes listed in
[the format specification](coupled_hadamard_k96tail_reencode.md).

## Host requirements

The measured campaign uses Linux, Docker with the NVIDIA runtime, four visible
NVIDIA GPUs, Python 3.12, `hf`, `jq`, and enough local storage for one or more
four-layer capture waves. A four-layer saved capture wave is 51,589,398,528
bytes, which is 48.04 GiB or 51.59 GB. The full frozen SQG checkpoint is
already present locally for the measured run.

The launchers assign one layer to each GPU during recipe construction and eight
score workers to each GPU during exact triplet scoring. Every score worker is
limited to six CPU cores and uses one private CUDA context. Reducing worker
count changes wall time but not the sealed arithmetic.

Set explicit roots instead of relying on ephemeral session paths:

```bash
export COUPLED_PROJECT_ROOT=/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625
export COUPLED_SOURCE_SQG_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-W4A8
export COUPLED_QSRT_ROOT=/home/brandonmusic/KLC_SANDBOXES/qsrt-glm52-port
export RECIPE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-no-shortcut-recipe-v1
export FRESH_SQG_EXLLAMA_PYTHON_ROOT="$COUPLED_PROJECT_ROOT/runtime-dependencies/exllamav3-python"
export FRESH_SQG_EXLLAMA_EXTENSION_ROOT="$COUPLED_PROJECT_ROOT/runtime-dependencies/v39_ext/exllamav3"
```

The `COUPLED_*` and `RECIPE_ROOT` variables above are shell shorthand for the
manual commands in this document. The measured controller hardcodes its model,
input, recipe, score, allocation, work, and acceptance roots. It directly
consumes only the two `FRESH_SQG_EXLLAMA_*` overrides shown above. The campaign
launchers resolve the ExLlamaV3 package and extension through those overrides
or through persistent defaults inside the project. A path under `/tmp` is not
a reproducible dependency.

Create or verify the pinned runtime dependency tree before launching workers:

```bash
"$COUPLED_PROJECT_ROOT/scripts/prepare_pinned_exllamav3_runtime.sh"
```

The script refuses to overwrite an existing dependency, verifies both Docker
image IDs, extracts into a temporary directory under the project, and validates
the Python tree and extension hashes. A fresh temporary extraction test passed
for both artifacts.

## Verify executable bindings

Run these checks before any GPU worker starts:

```bash
git -C "$COUPLED_QSRT_ROOT" rev-parse HEAD
git -C "$COUPLED_QSRT_ROOT" diff --binary HEAD | sha256sum

git -C "$COUPLED_PROJECT_ROOT/kquant" rev-parse HEAD
git -C "$COUPLED_PROJECT_ROOT/kquant" diff --binary HEAD | sha256sum

sha256sum \
  "$COUPLED_PROJECT_ROOT/runtime-dependencies/v39_ext/exllamav3/exllamav3_ext.cpython-312-x86_64-linux-gnu.so"

(
  cd "$COUPLED_PROJECT_ROOT/runtime-dependencies/exllamav3-python"
  find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
)

sha256sum \
  /home/brandonmusic/KLC_SANDBOXES/fresh-sqg-extension-r33-saturation.VUybIb/sealed/kquant_sqg_quantize_ext_v22.cpython-312-x86_64-linux-gnu.so

docker image inspect \
  sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82

cd /path/to/glm52-sqg-mcg-experiments
python3 coupled_hadamard_k96tail/verify_source_sync.py
```

Expected Git revisions and hashes are in the required-artifacts table. The
KQuant preflight must also report backend SHA-256
`fb63082016d2adc331be58f538622e1382c86390b0d1715c9a755433fb14c624`
and status SHA-256
`8ad2cf65ff6656fd4cbf01df725a8d6055229f90142997ae90176cf3b37909ed`.
The ExLlamaV3 Python tree command must report
`834c8de389c700126e5746e7bae9876b3e443014480651e59919cdd68fc20506`.

## Verify the frozen SQG source

The measured local source is already downloaded. A clean host can obtain the
same checkpoint without obtaining the official BF16 model:

```bash
hf download brandonmusic/GLM-5.2-SQG-W4A8 \
  --revision 593dd0d2de6f79ce4e65303930c22c75e1359d44 \
  --local-dir "$COUPLED_SOURCE_SQG_ROOT"
```

Before encoding, verify the three source-domain hashes in the durable sample
[`layer_019_runtime_binding.json`](../coupled_hadamard_k96tail/evidence/layer_019_runtime_binding.json).
The live per-layer path is
`$RECIPE_ROOT/layer_019/runtime_binding.json`, with the layer number changed for
each layer. The recipe must report
`source_is_frozen_sqg_checkpoint = true` and
`official_bf16_weight_shards_read = false`.

## Run the campaign

The complete controller is `scripts/run_full_coupled_3p0625_campaign.sh` in the
coupled re-encode source tree. The filename reflects the project directory,
not the final mixed-rate average. The controller enforces K48 layer 3, K96
layers 4 through 77, and byte-preserved source-SQG MTP layer 78.

Run all 19 four-layer waves in the foreground:

```bash
cd "$COUPLED_PROJECT_ROOT"
START_WAVE=3 STOP_WAVE=75 CLEANUP_VALIDATED_WAVES=1 \
  scripts/run_full_coupled_3p0625_campaign.sh
```

`START_WAVE` and `STOP_WAVE` are wave starts and must be congruent to 3 modulo
4. A resume normally uses the same command. The controller validates and skips
sealed layers and atomic expert score receipts before starting more work.

The optional stop file is:

```text
/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/STOP_FULL_COUPLED_K96TAIL_NO_SHORTCUT
```

The controller checks it before launch and at every wave boundary.

## Per-wave execution order

For a wave whose first and last layers are `START` and `END`, the controller
runs these commands and gates in order. The following setup is runnable Bash
for wave 19 through 22. Change `START` and `END` together for another four-layer
wave.

```bash
START=19
END=22
RUN_LAYERS=$(seq -s, "$START" "$END")
WAVE=$(printf 'wave-%03d-%03d' "$START" "$END")
WAVE_INPUT_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-wave-inputs/$WAVE
PROFILE_ROOT=$RECIPE_ROOT/final_profiles
SCORE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-tail-rate-scores-no-shortcut-v5
ALLOCATION_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-allocations-v1
CANDIDATE_ROOT=/media/brandonmusic/nvme1n1p3/glm52-coupled-k96tail-no-shortcut-work-v1/$WAVE
LAYER_ROOT=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers
EVIDENCE_ROOT=$COUPLED_PROJECT_ROOT/evidence/full-coupled-k96tail-no-shortcut
cd "$COUPLED_PROJECT_ROOT"
```

### 1. Stage saved capture and preflight inputs

```bash
WAVE_INPUT_ROOT="$WAVE_INPUT_ROOT" \
  scripts/download_coupled_wave_inputs.sh "$START" "$END"
```

The script downloads only `capture_view` and `derived/wave_preflights` paths
from the pinned Hessian dataset revision. It requires a capture manifest,
per-layer `hidden.bf16.bin` files, preflight, bit contract, and source seal. It
then derives a local wave shard manifest from the source seal.

### 2. Select the layer-native profile and beta

```bash
RUN_LAYERS="$RUN_LAYERS" WAVE_INPUT_ROOT="$WAVE_INPUT_ROOT" \
  RECIPE_ROOT="$RECIPE_ROOT" DETACH=0 \
  scripts/run_coupled_recipe_wave.sh "$START" "$END"
```

This command runs one GPU worker per layer. The worker entry point is
`scripts/run_coupled_no_shortcut_recipe.py`, with shared implementation in
`scripts/coupled_recipe_core.py`. It seals the 16-cell bootstrap search,
seven-beta panel, optional fresh 16-cell search, report-only holdout, and final
profile binding.

### 3. Score exact K3/K4 triplet candidates

```bash
RUN_LAYERS="$RUN_LAYERS" WAVE_INPUT_ROOT="$WAVE_INPUT_ROOT" \
  SCORE_ROOT="$SCORE_ROOT" PROFILE_ROOT="$PROFILE_ROOT" \
  SCORE_SHARDS_PER_LAYER=8 DETACH=0 \
  scripts/run_coupled_tail_score_wave.sh "$START" "$END"
```

The worker entry point is `scripts/score_coupled_tail_triplet_candidates.py`.
It writes atomic expert JSON and row-SSE receipts. The controller then finalizes
exactly 96 K4 tensors per K96 layer and runs
`scripts/build_kld_route_guarded_allocation.py` against the sealed source KLD
and route records.

### 4. Encode the selected K96 allocation

```bash
RUN_LAYERS="$RUN_LAYERS" WAVE_INPUT_ROOT="$WAVE_INPUT_ROOT" \
  OUTPUT_ROOT="$CANDIDATE_ROOT" ALLOCATION_ROOT="$ALLOCATION_ROOT" \
  PROFILE_ROOT="$PROFILE_ROOT" \
  ALLOCATION_SUFFIX=.kld-route-v1-k096.allocation.json DETACH=0 \
  scripts/run_coupled_transcode_wave.sh "$START" "$END"
```

The worker entry point is `scripts/encode_coupled_mixed_rate_shard.py`. It uses
the final profile binding and guarded allocation. It reconstructs from the
frozen SQG checkpoint, applies the coupled transform, and writes candidate
expert payloads.

### 5. Prove scorer and encoder payload parity

For each layer, the controller runs:

```bash
for LAYER in $(seq "$START" "$END"); do
  PADDED=$(printf '%03d' "$LAYER")
  python3 scripts/validate_coupled_score_encode_parity.py \
    --score-root "$SCORE_ROOT" \
    --candidate-root "$CANDIDATE_ROOT" \
    --allocation "$ALLOCATION_ROOT/layer_${PADDED}.kld-route-v1-k096.allocation.json" \
    --layer "$LAYER" \
    --chunk-rows 256 \
    --output "$EVIDENCE_ROOT/layer-${PADDED}-k096-score-encode-parity.json"
done
```

The receipt must report exact agreement for 256 experts and 768 projection
payloads.

### 6. Select expert draws on disjoint data

```bash
RUN_LAYERS="$RUN_LAYERS" WAVE_INPUT_ROOT="$WAVE_INPUT_ROOT" \
  CANDIDATE_ROOT="$CANDIDATE_ROOT" ALLOCATION_ROOT="$ALLOCATION_ROOT" \
  ALLOCATION_SUFFIX=.kld-route-v1-k096.allocation.json DETACH=0 \
  scripts/run_score_select_coupled_wave.sh "$START" "$END"
```

The worker entry point is `scripts/score_select_coupled_mixed_rate.py`. Fit may
propose draw 6, but the disjoint selection role must confirm it. Otherwise draw
0 is used. Holdout is scored only after the decision is sealed.

### 7. Materialize and run the native oracle

```bash
RUN_LAYERS="$RUN_LAYERS" CANDIDATE_ROOT="$CANDIDATE_ROOT" \
  LAYER_ROOT="$LAYER_ROOT" \
  scripts/run_materialize_coupled_wave.sh "$START" "$END"
```

Materialization uses `scripts/materialize_selected_coupled_layer.py` and
archives compact quality evidence with
`scripts/archive_coupled_layer_quality.py`. The materialization launcher already
runs the native oracle for each layer. It must pass the route-packed direct-E4M3
W4A8 endpoint with no A16 fallback.

Run the direct command only to recover a missing oracle after materialization.
This example recovers layer 19 on GPU 0:

```bash
LAYER_ROOT="$LAYER_ROOT" GPU=0 \
  scripts/run_validate_coupled_runtime_layer.sh 19
```

### 8. Archive compact evidence and remove rolling scratch

After all layer manifests and runtime oracles pass, the controller stores a
wave archive that excludes `.safetensors`, then removes only the validated
wave input and candidate scratch directories. Sealed, hash-bound selected
layer shards, profiles, allocations, score receipts, parity receipts, oracles,
and the wave archive remain.

## Assemble and validate the model

After target layers 3 through 77 pass their layer gates, the controller runs:

```bash
python3 scripts/assemble_coupled_checkpoint.py \
  --source "$COUPLED_SOURCE_SQG_ROOT" \
  --layer-root /home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail-layers \
  --output /home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail \
  --layers $(seq 3 77)
```

Assembly hard-links unchanged source files, replaces only the selected routed
layer shards, preserves source-SQG MTP layer 78 byte for byte, and writes
`COUPLED_REENCODE_MANIFEST.json`. It refuses an existing output directory and
refuses a partial or mismatched layer seal.

The controller then runs the full codec census and
`scripts/run_finalize_k96tail_model.sh`. The finalizer evaluates:

1. All 75 target layer manifests and TP1 runtime oracles for layers 3 through
   77, plus exact scorer/encoder parity for the 74 K96 layers 4 through 77.
   Layer 3 is the sealed K48 exception and has no K96 parity receipt.
2. A codec census that includes preserved source-SQG MTP layer 78.
3. A complete untrimmed 2,047-position KLD receipt under TP4/PP1/DCP1.
4. Mean, p99, and worst-one-percent CVaR below the sealed source checkpoint.
5. The sealed exact-r11 TP4/DCP4/MTP3 quality summary and runtime MTP summary.
6. A sealed compact reproduction bundle with `SHA256SUMS`.

`scripts/analyze_kld_position_tail.py` writes diagnostic tail concentrations.
Its removal ladder does not remove positions from acceptance. The final seal is
written by `scripts/seal_coupled_k96tail_release.py`.

The assembled checkpoint and codec census pass. Exact TP4/DCP1 KLD covers all
2,047 positions with no nonfinite values. Candidate mean KLD is
`0.1401771516114036`, compared with source mean `0.07583317451217256`.
The delta is `0.06434397709923104`, the ratio is `1.84849x`, and the candidate
is 84.849 percent worse. Candidate p99 is `2.480538845062256` and
worst-one-percent CVaR is `4.160942645300002`. The source-relative quality gate
therefore fails and the model remains `research-only`.

The separate exact Infernal Invocation r11 runtime bundle is `qualified` at
TP4/DCP4/MTP3. Its four rank receipts cover loaded and executed layers 3
through 78 with no fatal audit matches. Estonia is 5/5 correct. LAVD is 5/5
correct under its published tolerance, with four exact answers and one near
answer of `71,45.75` versus `72,46`.

## Resume guarantees

The campaign can resume after a process, container, driver, or host failure
because it uses these boundaries:

- Expert score JSON and row-SSE files are atomic.
- Existing recipe, profile, beta, allocation, parity, selection, layer, and
  oracle receipts are validated before they are reused.
- A materialized K96 layer without its parity receipt is rejected. Layer 3 is
  checked against its separate K48 seal and oracle.
- Partial layer output is rejected instead of overwritten.
- A completed wave is skipped only after its layer and runtime gates pass.
- Model assembly refuses an existing partial destination.

These guarantees preserve completed work. They do not make a mismatched code
tree or dependency hash acceptable.

## Publication boundary

The Git publication contains the campaign script closure, complete QSRT and
KQuant working source snapshots, their exact tracked patches and changed files,
the final runtime build context, runtime dependency identities, and an
automated hash check under
[`coupled_hadamard_k96tail`](../coupled_hadamard_k96tail/README.md). This proves
the enumerated active-source equalities and the bounded finalized-contract
normalization. It does not embed the checkpoint, saved captures, compiled
extensions, or container images. The hash-bound
runtime closure and measured receipts under
`coupled_hadamard_k96tail/evidence/final-exact-ii-r11` are the compact execution
record. The completed 75-layer mechanical closure is under
`coupled_hadamard_k96tail/evidence/final-mechanical`. Public model publication
is `unsupported` until every local model file
is present on the Hub and `HUB_FILE_VERIFICATION.json` reports a complete
file-by-file verification. A transient upload rate is operational evidence,
not a completion claim or a reproducible ETA.
