# Fast parallel SQG encoding for GLM-5.2

This runbook records the accelerated encoding path proven during the four-layer
GLM-5.2 SQG pilot. The code and artifact format use the term **SQG**
(`sqg_xor_cheb_t12`); do not rename the on-disk ABI to SCQ even if SCQ is used
as conversational shorthand.

## Observed result

On four RTX PRO 6000 Blackwell GPUs with 48 logical CPU threads, the final
encoding of four layers completed as follows:

- 4 layers × 256 experts = 1,024 experts.
- 3 SQG tensors per expert = 3,072 freshly encoded tensors.
- 16 encoder processes total: 4 processes per GPU.
- 64 experts per process.
- 3 CPU threads per process, for 48 scheduled CPU threads total.
- Sustained aggregate rate after startup: approximately 70–72 experts/minute.
- Final encode wall time: approximately 15.5 minutes.
- Every encoder process exited successfully.

The large speedup came from keeping several independent expert pipelines ready
for each GPU and from removing repeated byte-neutral source/capture hashing
from every worker. It did **not** come from weakening the SQG math or replacing
real encoding with a proxy.

## Mathematical work that must stay together

One expert is the smallest scheduling unit. Its three projections must remain
in one worker and run in this order:

1. Load the expert's official BF16 `gate_proj`, `up_proj`, and `down_proj`.
2. Apply the frozen physical permutation and selected shared residual profile.
3. Encode gate and up with the same layer H13 dense-Hessian/BlockLDL binding.
4. Reconstruct the newly encoded gate and up tensors.
5. Build that expert's conditional H2 from the reconstructed candidate
   gate/up path and its routed fit rows.
6. Encode down with that candidate-derived, expert-local H2.
7. Persist the three SQG payloads, exact K3/K4 assignments, closure evidence,
   and zero MCG inputs.

Do not split gate, up, and down for one expert across workers. Experts can run
independently because H13 is immutable and its finalized BlockLDL factor is
reused read-only, while H2 is local to that expert. Each parallel shard owns a
separate `DenseHSession`; its evidence ordinals restart inside the shard, so the
artifact must record the shard range truthfully.

## Mandatory absolute-scale smoke gate

KQuant does not give its two explicit profile sides the same scale semantics:

- A gate/up `input_channel_scale_profile` is an **absolute input-channel RMS**.
  Do not divide it by its mean. The uniform control must use the mean of the
  measured absolute gate-input base, not a vector of literal ones.
- A down `output_channel_scale_profile` is a **relative channel modulation** in
  this construction and should remain mean-normalized to one.

This distinction must be tested before launching a layer wave. Encode and
independently decode at least one real K3 and one real K4 gate/up matrix using
the selected profile path. Reject the run if every searched global scale lands
on the search boundary or if source-relative reconstruction RMSE is
catastrophic. The invalid pilot signature was a gate/up global scale of exactly
`2.05` for every tensor and relative RMSE above `1.0`, while down stayed near
normal quantization error. Encoder/decode closure alone is insufficient because
both sides can faithfully reproduce the same badly scaled weight.

After gate/up changes, rebuild that expert's conditional H2 and re-encode down;
never reuse an H2 or down payload derived from the rejected gate/up candidate.

### Corrected-pilot recovery path

The rejected mean-one gate/up pilot is retained rather than edited in place.
Create a clean successor root with:

```bash
scripts/prepare_corrected_successor.sh \
  /home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2
```

The importer retains the exact `glm52-fresh-sqg-r1` run ID so signs and
transform seeds do not change. It validates the predecessor preflight and
recovery chain, small seals, current code, and BF16 file identities without
rehashing BF16 or capture payloads. It hardlinks the valid H13 and raw profile-
scale tensors when the runtime exposes both roots on one mount; across separate
Docker bind mounts it falls back only on `EXDEV` to a SHA-verified copy. It
rebinds H13/profile-scale/permutation JSON to the successor
preflight, and imports no profile-search, H2, down, final-layer, run-seal, or
candidate bytes.

After reviewing the successor receipt, run the corrected search:

```bash
scripts/run_corrected_profile_search.sh \
  /home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2
```

To stop after the first real expert and inspect its reconstruction receipt
before admitting the 16-worker wave, prefix the same command with
`SMOKE_ONLY=1`. Re-running the normal command resumes that exact smoke artifact
and proceeds into profile search.

This first encodes layer 28 expert 0, whose gate is K4 and up is K3. Both must
pass the real global-scale/source-RMSE gate; the same expert must then rebuild
candidate-conditional H2 and encode down. Only after that receipt exists does
the launcher start 16 profile workers: four processes per GPU, four disjoint
cells per process, and 16 real panel experts per cell. Four finalizers freeze
selection after every cell validates. The launcher stops there and does not
start the 256-expert final treatment.

## One-time work before parallel encoding

Perform expensive validation and calibration once for the run or wave, not in
every expert worker:

1. Seal the BF16 source, calibration capture, bit contract, KQuant source,
   ExLlama runtime, and SQG extension once.
2. Prepare every target layer's H13 and 256 physical permutations.
3. Finish profile scoring and freeze `profile_search/selection.json` for each
   layer before final encoding begins.
4. Confirm the frozen topology-neutral per-tensor K3/K4 map.
5. Confirm every selected cell is a real SQG encode with candidate-conditional
   down H2 and no MCG transform, scale, permutation, seed, payload, or decoded
   weight input.

The pilot selected from the completed four-draw/16-cell factorial prefix under
an explicit time-priority receipt. That is a **pilot selection decision**, not
an intrinsic requirement of the parallel encoder. A full-model production run
may use a different predeclared profile-search depth without changing the
sharding method.

## Proven worker layout

Assign one layer to each physical GPU in a four-layer wave. On each GPU, launch
four disjoint expert ranges:

```text
GPU 0 -> layer A -> experts 000:064, 064:128, 128:192, 192:256
GPU 1 -> layer B -> experts 000:064, 064:128, 128:192, 192:256
GPU 2 -> layer C -> experts 000:064, 064:128, 128:192, 192:256
GPU 3 -> layer D -> experts 000:064, 064:128, 128:192, 192:256
```

Use `--gpus all` plus one `CUDA_VISIBLE_DEVICES=N` value per container. Inside
each container the assigned GPU is logical `cuda:0`. Avoid combining Docker's
`--gpus device=N` filtering with a second contradictory visibility mapping.

Cap every worker explicitly:

```text
--cpus 3
OMP_NUM_THREADS=3
MKL_NUM_THREADS=3
OPENBLAS_NUM_THREADS=3
NUMEXPR_NUM_THREADS=3
torch.set_num_threads(3)
torch.set_num_interop_threads(1)
```

The current helper is
`scripts/encode_final_shard.py`. A representative invocation inside the sealed
runtime is:

```bash
python scripts/encode_final_shard.py \
  --preflight /output/preflight.json \
  --layer 6 \
  --start 0 \
  --end 64 \
  --device cuda:0 \
  --threads 3
```

The default helper path is sealed fast-resume: it trusts the already completed
preflight, reconstructs BF16 source bindings from the sealed index and current
file identities, opens the capture without rehashing it, and retains the small
code/extension identity checks. `--full-revalidate` remains available when a
new full validation is actually required.

## Asynchronous wave scheduler for a full model

For a full GLM-5.2 quantization, derive the eligible MoE layer list from the
model/index and bit contract. Do not hard-code a guessed layer count. Process
that list in waves of four:

1. Freeze preparation and profile selection for the next four layers.
2. Assign one layer to each GPU.
3. Launch four 64-expert shard workers per assigned layer.
4. Monitor completed expert manifests, worker exits, GPU utilization, RAM, and
   free disk space. Do not put full hashing on this timed path.
5. As soon as a layer's four shards finish, consolidate it; the next layer may
   be assigned to that GPU once CPU and I/O pressure permit.
6. Continue until every eligible layer is assembled.

The simple four-layer barrier used by the pilot is easy to audit. A more
aggressive full-model scheduler may refill a GPU as soon as its layer finishes,
but must preserve one layer's frozen selection and unique output directories.
Do not overlap heavy profile construction with 48 fully occupied encoder
threads unless measurement shows that the extra contention improves wall time.

## Resume and artifact layout

Each shard writes to its own directory:

```text
layer_LLL/expert_shards/experts_000_064
layer_LLL/expert_shards/experts_064_128
layer_LLL/expert_shards/experts_128_192
layer_LLL/expert_shards/experts_192_256
```

The encoder validates a contiguous prefix inside that directory and resumes
only the missing suffix. A failed shard can therefore restart without
re-encoding completed experts or changing the other three shards.

After all four shards succeed, run
`scripts/consolidate_final_shards.py`. It publishes the expert manifest,
manifest seal, and safetensors payload into the canonical expert directory by
hardlink where possible. `encode_layer` then sees a complete 256-expert prefix,
does not re-encode BF16 weights, and assembles the final layer artifact.

## Validation placement

Keep validation proportional and place it at boundaries:

- Full BF16/capture hashing: once before the run or after actual source drift.
- Worker startup: sealed metadata, code/extension identity, and current file
  identity checks; no 145 GB source/capture rehash per shard.
- Expert completion: SQG closure and artifact evidence produced by the real
  encoder.
- Layer completion: one canonical assembly validation.
- Run completion: one four-layer/full-run seal.
- KLD: real runtime dispatch proof and per-position logits/KLD, without
  repeatedly hashing a 343 GB checkpoint before every boot.

Repeated hashing can dominate the wall clock while changing no encoded byte.
Removing it from the timed path is safe only after a real seal exists and the
fast path records that it trusted that seal.

## Scaling expectations and decision gate

The four-layer result proves the encoding scheduler, not full-model quality.
Use the current candidate-versus-native paired KLD result as the decision gate
before expanding to every layer. If the direction is favorable, the observed
15.5-minute four-layer wave is the starting throughput estimate; allow extra
time for layer-to-layer variation, preparation/profile search, assembly,
storage pressure, and final model validation.

## Corrected four-layer final handoff

After the absolute gate/up RMS profile search has completed and frozen all four
selections, the prepared final handoff is:

```bash
bash scripts/run_corrected_final_encode.sh \
  /home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2 \
  /home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-FRESH-SQG4-ABSRMS-r2
```

The handoff does not rerun calibration or profile search. It starts 16 final
expert workers using the proven four-workers-per-GPU schedule, resumes only
missing expert suffixes, consolidates and assembles the four layers in
parallel, writes the required four-layer seal, then materializes into the
explicit new candidate path. The candidate path must not already exist.

The timed workers use sealed fast resume and do not hash the BF16 subset or
capture payloads again. Each final expert still performs the real SQG encode,
production closure checks, absolute gate/up saturation/RMSE guard, and
candidate-derived expert-local H2/down construction. Consolidation and the run
seal retain the required encoded-artifact integrity checks and prove 3,072 SQG
tensors, zero MCG tensors, and the frozen 1,536 K3/1,536 K4 per-tensor census.
The materializer uses the fast sealed context, validates copied selected
payloads, and never overwrites the existing model. Keep source and candidate
under the same writable parent mount so unchanged files can be hardlinked.
A nested read-only source bind crosses a Docker mount boundary and makes those
hardlinks fail with `EXDEV`; protect the source logically by requiring a new,
nonexistent candidate path and by checking that source file metadata is
unchanged. Normalize candidate ownership once after materialization if the
container wrote root-owned links or directories.

## Corrected candidate KLD handoff

After materialization, run the prepared five-boot candidate against the already
measured native-MCG baseline, without rerunning that baseline, with:

```bash
CANDIDATE_ONLY=1 bash scripts/run_absrms_r2_kld.sh \
  /home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-FRESH-SQG4-ABSRMS-r2
```

The workflow defaults `FRESH_ARTIFACTS_ROOT` to the actual corrected artifact
root `/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-encode-absfix-r2`. Results are
written below `/home/brandonmusic/KLC_SANDBOXES/fresh-sqg-evaluation-absrms-r2`
in the `candidate`, `native`, and `paired` subdirectories. Accepted boots are
resumed in place; only an incomplete next boot is moved to a recoverable
quarantine. Each corrected-candidate boot gets an independent snapshot of the
rejected candidate's compiled-code cache, while the process, engine workers,
model load, runtime dispatch proof, and 2,047-position KLD remain fresh.

BF16 reference logits are sufficient for the KLD comparison but are not a
replacement for BF16 weights or layer calibration/Hessian data when encoding
new layers. A full-model SQG conversion still requires the original BF16
weights and the chosen calibration evidence for every target layer.

## Measured corrected four-layer pilot result

The completed candidate changes eligible K3/K4 tensors in layers 6, 28, 52,
and 77 only. Every selected layer contains 384 K3 and 384 K4 SQG tensors (3,072
SQG tensors total) and zero selected-layer MCG tensors, transforms, scales,
permutations, seeds, payloads, or decoded-weight reads. The other model layers
remain the fixed native-MCG control. This is therefore final-output logit KLD
for the **combined downstream effect of four SQG layers**, not layer-local KLD.
Saved BF16 output logits cannot isolate the contribution of one internal layer.

The five fresh candidate boots measured:

```text
0.06169908118680389
0.06205119761221136
0.062187488789569624
0.06401166746517507
0.06430004636288912
mean = 0.0628498962833298
sample SD = 0.001209723746733114
```

The existing five-boot r33 native-MCG baseline was deliberately not rerun:

```text
mean = 0.0624498626218156
sample SD = 0.0015327574926078513
```

Candidate minus baseline is `+0.00040003366151420555`, or `+0.6405677%`.
Lower is better, so the observed mean moved slightly worse. The repeat-noise
standard error of the difference is `0.0008732441897379576`; the conservative
two-sided 95% direction is inconclusive. This one fixed 2,047-position prompt
does not provide evidence that the four-layer SQG treatment lowers KLD, so it
does not pass the KLD-improvement gate for scaling directly to a full-model
SQG conversion. The comparison includes the required SQG loader/non-fused
dispatch and is not a codebook-only benchmark.

Evidence:

- Candidate summary: `fresh-sqg-evaluation-absrms-r2/candidate/fresh-sqg4-absrms-r2-candidate-kld-fp8-dcp4/summary.json`
- Per-position evidence seal: `fresh-sqg-evaluation-absrms-r2/candidate/fresh-sqg4-absrms-r2-candidate-kld-fp8-dcp4/per-position-evidence.sha256`
- Encoding seal: `fresh-sqg-encode-absfix-r2/run_seal.json`
- Existing baseline: `/home/brandonmusic/klc-linux/release-gg-sparkinfer-20260721/results/20260809T194958Z-kld-fp8-dcp4/summary.json`

Boot 1 exposed seven tiny negative float32 per-position KLD values, with minimum
`-4.6551978272191263e-08`. The validator now accepts bounded roundoff down to
`-1e-7` without clamping or modifying the tensor and still rejects values below
that tolerance. Its recovery receipt binds the original raw output, log,
dispatch proof, old and corrected validator identities, and accepted tensor.
