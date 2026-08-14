# GLM-5.2 coupled-Hadamard K96-tail SQG re-encode

This document specifies the routed-expert re-encode that uses the updated QSRT
coupled Hadamard transform and the saved GLM-5.2 calibration captures. The
source weights are reconstructed from the frozen SQG W4A8 checkpoint. The
official BF16 routed weight shards are not read during this re-encode.

The artifact is called the coupled H512/H128 K96-tail checkpoint in this
repository. `K96` means that 96 of the 768 routed projection tensors in each
layer use the K4 trellis rate. It does not mean 96 bits or 96 layers.

## Qualification status

| System part | Status | Evidence required for the status |
|---|---|---|
| Coupled transform and GLM activation closure | implemented | Updated QSRT transform, exact `silu(gate) * up`, and transform closure tests |
| Per-layer profile, beta, allocation, encode, selection, and materialization pipeline | implemented | Fail-closed campaign launchers and receipt validators |
| Routed layers 3 through 18 | qualified | Hash-bound layer shards, layer manifests, quality receipts, exact scorer/encoder parity for layers 4 through 18, and passing native B12X runtime oracles |
| Routed layers 19 through 78 | unsupported | No complete, qualified shard set exists for these layers |
| Full 76-layer assembled checkpoint | unsupported | Assembly is forbidden until every routed layer passes the layer gates |
| End-to-end KLD improvement | research-only | The candidate TP4/PP1/DCP1 KLD run has not completed |
| Release checkpoint | unsupported | Codec census, full KLD distribution gates, MTP3 smoke, and sealed reproduction bundle remain open |

`implemented` means the code path and its fail-closed checks exist.
`qualified` means the named artifact passed the stated measurements.
`research-only` means the mechanism or evidence can guide an experiment but
does not establish model quality. `unsupported` means the repository must not
make the corresponding artifact or quality claim.

## Source and calibration identity

The source checkpoint is the SQG W4A8 model published as
`brandonmusic/GLM-5.2-SQG-W4A8`. The local source snapshot is bound to Hugging
Face revision `593dd0d2de6f79ce4e65303930c22c75e1359d44` and to these file-domain
hashes in each recipe runtime binding:

| Bound object | SHA-256 |
|---|---|
| Source SQG manifest | `57658b1f756db00b9b86e60fe1c5c0fb435537533712bc836fdd05d42347865e` |
| Source model index | `62182715e5ef3eb0069b38ff710a38e7a118154278768a8c7ab8efed60c501f5` |
| Source quantization configuration | `4172b107ffcc0fe4b3971455ba0a1430b9ee5546a7b640e43e84fde62f702d37` |

The saved capture and Hessian inputs come from the Hugging Face dataset
`brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians` at revision
`a05b3b92d749f6a641af5cfd52de2b4720380dfd`. Each four-layer wave downloads
only its saved capture view and preflight records. Each layer capture includes
`hidden.bf16.bin`, but that file is a saved activation capture, not a BF16 model
weight shard.

The recipe records both of these invariants:

```text
source_is_frozen_sqg_checkpoint = true
official_bf16_weight_shards_read = false
```

The source SQG reconstruction uses the native B12X dense identity probe in the
expert-private function-preserving basis. B12X is the native runtime
implementation used by the layer oracle. Its direct-E4M3 path decodes the SQG
payload into the FP8 E4M3 activation endpoint without falling back to the A16
route. This is not a lower-bit transcode of trellis bytes without calibration.
It reconstructs the frozen SQG function, applies the coupled coordinate change,
recalibrates in those coordinates, and encodes replacement trellis payloads.

## Coupled transform contract

Every routed expert contains `gate_proj`, `up_proj`, and `down_proj`. The
coupled transform closes the three projections as one function-preserving
block before quantization:

1. Apply a signed block Hadamard transform with block size 512 at the residual
   input boundary.
2. Transform the concatenated gate and up preactivation coordinates with block
   size 128.
3. Invert that preactivation transform before evaluating the activation.
4. Evaluate GLM's exact activation as `silu(gate) * up`.
5. Apply a signed block Hadamard transform with block size 128 after the
   activation.
6. Close the down projection and invert the residual output transform.

The corresponding H13 and H2 calibration terms are transformed into the same
coordinates. H13 uses 75 percent layer-global and 25 percent expert-local
evidence. The down objective uses a candidate-conditioned `(H, B)` target built
from the newly encoded gate and up path. Fit data is the only data allowed to
construct these calibration terms.

The paired weight reparameterization and runtime boundary transforms are exact
inverses and preserve the unquantized expert function. A plain block Hadamard
is self-inverse. For the signed form, the inverse of `D H` is `H D`, and the
gate/up path also performs the matching interleave and split. Quantization error
can change because the K3 and K4 trellis encoders see a different coordinate
system. A claim that the transform lowers final-logit KLD still requires the
full-model KLD gate.

## Rate contract

The checkpoint has a hybrid routed rate. The project directory name retains
the original 3.0625-bpw target, but the implemented K96-tail contract is:

| Routed layers | K3 tensors per layer | K4 tensors per layer | Per-layer bpw |
|---|---:|---:|---:|
| Layer 3 | 720 | 48 | 3.0625 |
| Layers 4 through 78 | 672 | 96 | 3.125 |

Each layer has 256 experts and three independently rated projection tensors per
expert, for 768 rated tensors. The 76-layer arithmetic mean is
`3.124177631579` bpw. The artifact is not a uniform 3.0625-bpw routed model.

Layer 3 is the sealed K48 coupled layer artifact. Layers 4 through 78 use K96.
The final assembly manifest must report `routed_layer_average_rate_is_uniform`
as false and must preserve the per-layer census.

## No-shortcut layer recipe

The layer recipe prevents a fleet-wide profile or beta choice from replacing
layer-native calibration:

1. A 16-cell bootstrap profile search evaluates four Hadamard draws crossed
   with four scale families.
2. A mass-stratified 16-expert beta panel evaluates
   `0`, `0.03125`, `0.0625`, `0.125`, `0.25`, `0.5`, and `1.0`.
3. If the selected beta differs from the bootstrap beta `0.0625`, a fresh
   16-cell profile search runs exactly once under the selected beta. If the
   selected beta is `0.0625`, the bootstrap selection is reused byte for byte.
4. Profile selection completes before K3/K4 allocation.
5. Holdout records are written only after selection is sealed and are never
   used for profile, beta, rate, or draw selection.

The B300 owner-speed shortcut was a paid-node recovery path that could replace
the full layer-native search with a fixed identity-oriented rescue profile.
This campaign does not use it. The final profile binding records which branch
was used and requires `no_b300_owner_speed_rescue = true`. The campaign rejects
an identity-only fallback and rejects a single fleet beta.

## Layer-native K96 allocation

The scorer encodes every expert projection under the K3 and K4 choices needed
for exact triplet allocation. It writes one atomic JSON receipt and one atomic
row-SSE array per expert. Existing receipts are validated and skipped on a
resume.

The base allocation minimizes the layer-native fit objective subject to exactly
96 K4 tensors. A second allocation guard uses the sealed source model's 40
largest TP4/PP1/DCP1 KLD positions and the exact routed experts at those
positions. The guard may exchange K4 assignments while enforcing one percent
total and body regression limits on the calibration objective.

The source-worst-40 signal is `research-only`. It is an in-sample allocation
signal, not end-to-end acceptance evidence. Layer 78 has no source route signal
and therefore uses its unguarded layer-native K96 allocation. No position is
removed from the final KLD measurement.

## Layer qualification gates

A routed layer is `qualified` only when all of these conditions pass:

1. The no-shortcut recipe, selected beta, and final profile binding are sealed.
2. The K96 allocation contains 672 K3 and 96 K4 tensors, 2,400 bit units, and
   3.125 bpw. Layer 3 uses its separate K48 contract.
3. The candidate encoder writes 256 complete experts with no fallback.
4. The scorer and encoder payloads match exactly for all 256 experts and all
   768 projection payloads.
5. The draw decision uses fit as a proposal and disjoint selection as the
   confirmation. Holdout is report-only.
6. The selected layer shard and quality receipt are materialized without
   mutating the source checkpoint.
7. The native B12X route-packed direct-E4M3 W4A8 oracle returns finite,
   nonzero output and passes the exact layer census. A16 fallback is forbidden.

Passing these gates establishes layer construction and runtime closure. It does
not establish final-logit KLD quality.

## Full-model acceptance gates

The full checkpoint remains `unsupported` until all 76 routed layers pass the
layer gates and these model gates pass:

1. Assemble routed layers 3 through 78 over the unchanged frozen source
   checkpoint and write `COUPLED_REENCODE_MANIFEST.json`.
2. Run a full codec census over all 76 routed layers, including all 768 routed
   tensors in MTP layer 78.
3. Run the candidate against the same sealed BF16 logits as the source under
   TP4, PP1, and DCP1 for all 2,047 positions.
4. Require zero nonfinite positions and zero trimmed positions.
5. Require candidate mean KLD, p99 KLD, and worst-one-percent CVaR to be lower
   than the source checkpoint.
6. Run a 16-token MTP3 production smoke under TP1, PP4, and DCP1.
7. Seal the scripts, compact receipts, runtime identity, model manifest, and
   SHA-256 manifest into a reproduction bundle.

The source checkpoint's measured untrimmed TP4/PP1/DCP1 distribution is mean
`0.075833174512`, median `0.000611608732`, p95 `0.380242288113`, p99
`1.397577404976`, worst-one-percent CVaR `2.207112874304`, and maximum
`5.978030681610`. The 40 largest positions contain 43.433 percent of the total
KLD. Those values explain the tail gates. They do not predict the candidate's
result.

## Runtime and code bindings

Qualified recipe receipts bind these executable inputs:

| Component | Binding |
|---|---|
| QSRT | revision `453b4834332d2735c5a326ca57fb6a8b36e776bf`, tracked diff SHA-256 `33982c45a93c9291a5e0e63a40dd638863dc018bf1dcb9987e235d93e2eb278d` |
| KQuant | revision `104dd9233f850a3955f4991bea68b07dd34deeb8`, tracked diff SHA-256 `82c994a6fa1e1c996f18c85f723c562fbe08bb8a7edf705ef9ece1ae1606958f` |
| KQuant backend tree | SHA-256 `fb63082016d2adc331be58f538622e1382c86390b0d1715c9a755433fb14c624` |
| KQuant status | SHA-256 `8ad2cf65ff6656fd4cbf01df725a8d6055229f90142997ae90176cf3b37909ed` |
| Encode image | `sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82` |
| SQG encoder extension | SHA-256 `d29010f6ad51caf2e1a22f07365ab3548fcdb3e0ed3ee15d88330cee24de9614` |
| ExLlamaV3 extension | SHA-256 `e88bc24d2c292a0b69a7ee27bb701557c16e535c9980ee870c931a2b495033bd` |

Revision alone is insufficient for QSRT and KQuant because both approved trees
contain tracked changes. A publication is reproducible only when it includes
the exact changed sources or patches and verifies the listed hashes.

The [campaign source snapshot](../coupled_hadamard_k96tail/README.md) contains
the active script closure, runtime build context, changed-file snapshots, exact
QSRT and KQuant patches, compact evidence, and an automated equivalence check.

See [the reproduction procedure](coupled_hadamard_k96tail_reproduction.md) for
the command sequence and [the dated execution record](coupled_hadamard_k96tail_status_2026-08-14.md)
for measured progress and incidents.
