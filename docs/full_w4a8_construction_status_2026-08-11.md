# Full native-SQG W4A8 construction status, 2026-08-11

Status: the first rolling wave is in construction. This is not full-model
quality, serving acceptance, or a release announcement.

## Frozen model-format decision

The authorized measurement build is full W4A8, not A16 or the earlier hybrid
arm. Every converted routed layer must preserve all of the following:

- native direct-E4M3 SQG labels;
- MXFP8 E4M3/UE8M0 K32 quantization at both `h` and
  `act = SiLU(gate) * up`;
- exact GLM `SiLU(gate) * up`, followed by FP32 accumulation;
- independent per-tensor gate, up, and down K3/K4 assignments, with exactly
  384 K3 and 384 K4 tensors in each routed layer;
- topology-neutral transforms and scales;
- Hadamard transforms on the activation side;
- candidate-specific, caller-coordinate down `(H,B)` construction with the
  private down input scale and shared down output scale anchored; and
- zero reuse of MCG payloads, scales, or transforms in converted layers.

An A16 arm remains useful as a numerical reference. It is not an allowed
fallback for an encoded production layer. Uniform K3 is also not the target
format.

## Completed evidence that changed the recipe

The coordinate-corrected layer-77 down repair lowered signed weighted top-8
NMSE by `9.5762%` on selection and `9.3043%` on the secondary holdout versus
the base full-W4A8 candidate. It remained `10.22%` and `10.86%` worse than the
matched SQG-A16 reference. This promotes the corrected `(H,B)` construction,
but it does not establish full-model quality.

The leakage-free layer-77 beta panel selected `beta=0.0625` using only
`fit/calibration` to construct candidates and `fit/allocation` to make the
choice. It improved panel SSE by `0.3344%` relative to beta 0. Beta 1 was
catastrophically worse and is rejected. Layer 77 is evidence for a bootstrap
prior, not a fleet-wide beta.

Same-rate encoding batches are also rejected. K4 batches of two and four
changed accepted bytes, and K3x4 failed on the first selected-beta,
high-route-mass panel expert. Production therefore retains fourteen singleton
physical encodes per expert. This changes construction time, not route-packed
inference speed.

The detailed measurements remain in:

- [coordinate-corrected layer-77 result](../results/glm52_full_w4a8_xterm_coordinate_fixed_l077_r2.md);
- [fit-only layer-77 beta result](../results/glm52_full_w4a8_beta_selection_l077_r1.md); and
- [same-rate batching audit](../results/glm52_w4a8_encoder_batch_equivalence_r1.md).

## Canonical per-layer beta and source binding

The full build closes profile/beta circularity once per layer:

1. Search a W4A8-native bootstrap profile under the declared prior
   `beta=0.0625`.
2. Freeze that bootstrap profile.
3. Construct `(H,B)` on `fit/calibration` and choose beta once on the
   document-disjoint `fit/allocation` 16-expert mixed-K3/K4 panel.
4. Seal a portable `beta_choice.json`. Selection and holdout are excluded from
   beta choice.
5. If the selected beta is exactly `0.0625`, reuse the bootstrap profile only
   with byte, SHA-256, selection-ID, and prior-identity proof. Otherwise run
   exactly one final native profile search at the selected beta. Do not reopen
   beta.
6. Bind triplet scoring, exact 384/384 allocation, selected encoding, and
   materialization to the final profile, beta choice, and complete source
   binding.

A bare numeric beta is not admissible provenance, and one layer's beta cannot
be copied to another layer. The canonical build binding seals the complete
choice-affecting `scripts`, `src`, BMM-Law encoder, and KQuant package trees,
plus the recapture entrypoints. The active first-wave binding ID is:

```text
32b9a4b109c4ef226709804db0a2fa6d34f305a7c22efb0051c95a99ffb82e85
```

Any source drift after that seal is a hard failure and requires a new output
root; the binding is not refreshed around existing choice output.

## Rolling fixed-point construction

The sealed plan has 19 non-overlapping waves: eighteen four-layer waves from
3--74, followed by the terminal 75--77 wave. It covers all 75 routed layers
exactly once. The machine-readable contract is
[full_model_rolling_wave_plan_v2.json](../contracts/full_model_rolling_wave_plan_v2.json):

```text
003-006  007-010  011-014  015-018  019-022
023-026  027-030  031-034  035-038  039-042
043-046  047-050  051-054  055-058  059-062
063-066  067-070  071-074  075-077

plan_id  b95c351e5d9af20e43a45e5d2318f2bd7ce934486480f8836bd0e2368e293c31
sha256  a919f5619c65275c59e4bcad4c584ae7d5c19eff3c2eca657ea008d21b30dc57
```

Each wave must close BF16 source and fixed-point capture, preparation,
bootstrap profile, per-layer beta, final profile, all eight K3/K4 triplet
scores, exact allocation, selected encoding, prefix materialization, and a
native-W4A8 DCP4 smoke before the next wave is recaptured. That next capture
must observe the exact already-converted prefix. An eight-layer capture cannot
be split retroactively into two valid waves because the second four layers
would not have seen the first four converted.

## Production-v2 lineage and storage closure

Production-v2 wraps the unchanged arithmetic scorer, exact-DP allocator, and
selected encoder in a fail-closed evidence chain:

```text
full-build binding + beta choice + final profile
  -> common score-root binding
  -> 256 expert-score envelopes
  -> production aggregate score
  -> exact 384/384 allocation
  -> selected encode
  -> production layer receipt
  -> progressive materialization
```

Parallel layers receive isolated legacy arithmetic roots, preventing recovery-
receipt races. Old unwrapped scores, bare numeric betas, mismatched profiles,
wrong rate censuses, old layer manifests, and any MCG payload are rejected.
Temporary candidate-payload caching was deliberately deferred: it would add
roughly 15 GiB per active layer and lacks a scorer-native payload/derived-target
closure. Selected candidates are re-encoded and equality-checked instead.

The storage path now uses hardlinks for unchanged files on the same filesystem
and atomic SHA-256-verified copies only after `EXDEV`. A two-phase retention
tool first validates and seals the consolidated layer plus all 256 expert
manifests, then permits pruning only with the exact receipt hash. Its compact
validator replays rate assignments, payload hashes, vector references,
permutations, native-SQG markers, zero-MCG lineage, and down-target bindings
against the consolidated shard. No real model artifact was deleted while this
policy was implemented.

Measured planning sizes are about `4.231 GiB` per consolidated layer and up to
`4.84 GiB` of removable expert intermediates per layer. The rolling policy
keeps only the active BF16/capture wave and immediate successor online and
retains explicit free-space watermarks.

## Runtime static integration boundary

The acceptance runtime is pinned to the measured PR11 route-packed schedule:
`m128n64`, eight M16 route groups per CTA, and two asynchronous stages. The
kernel source remains byte-identical to the measured version at SHA-256:

```text
b95dde9347cf679836eedc1c76f0f569a4fb37dcc594a674e41666e4d0f162ad
```

The high-level path preserves the exact h-A8 -> gate/up W4A8 -> GLM SwiGLU ->
act-A8 -> down W4A8 flow, independent K3/K4 pools, and signed top-8
accumulation. The vLLM integration adds live TP4/DCP4/MTP3, context,
scheduler, graph-workspace, and target/MTP isolation checks. Redundant loader
trellis/transformation dictionaries are released after B12X preparation so a
full checkpoint does not retain a second compact payload.

Static validation completed with 16 B12X tests and 23 vLLM GLM-SQG tests,
plus Ruff, compilation, and diff checks. This is not live acceptance. No full
artifact existed for the required TP4/DCP4 load, numerical oracle, eager versus
CUDA-graph equality, long-context, production-concurrency, KLD, LAVD, Estonia,
prefill, or decode gates.

## Current construction status

The first wave, layers 3--6, has sealed BF16 and routed-capture inputs and
completed preparation. Its bootstrap native-W4A8 profile search has launched
under canonical binding
`32b9a4b109c4ef226709804db0a2fa6d34f305a7c22efb0051c95a99ffb82e85`.

At this publication point there are no final first-wave profile selections,
per-layer beta choices, final encoded layers, progressive checkpoint, or
full-model quality results. In particular, there is no new full-model KLD,
LAVD, Estonia, integrated prefill/decode, or deployability claim. Construction
progress must not be read as evidence that SQG beats MCG or A16.

