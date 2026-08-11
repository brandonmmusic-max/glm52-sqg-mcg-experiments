# GLM-5.2 SQG versus MCG experiments

This repository preserves the code, methodology, compact evidence, and audited
results from the GLM-5.2 3.5-bpw experiments comparing the existing MCG
trellis representation with KQuant-style SQG. It now includes the signed
top-8, tail-constrained `H13` blend ablation, the preregistered contiguous-block
follow-up, the completed Test 8b/8c W4A8 falsifications, and a separate
QSRT/Kimi K3 K1 feasibility audit.

The [current-QSRT snapshot and history
review](docs/qsrt_current_update_review_2026-08-10.md) covers the new coupled
K3 Hadamard path, allocation evidence, W4A8 implications, and attribution
history without treating QSRT fixtures as GLM quality results.

The [post-falsification QSRT-head
addendum](docs/qsrt_current_update_review_2026-08-11.md) records which current
mechanisms transfer to the protected mixed per-tensor K3/K4 design and which
uniform-K3/H308 assumptions are rejected.

The experiment kept the production topology and the per-tensor K3/K4 bit
assignment fixed. Bits were **not** reallocated per expert. The initial
diagnostic layers were 6, 28, 52, and 77; the first contiguous block is layers
74--77. Every selected layer has 256 experts and three expert projections.

## Bottom line

The original conditional gate was **NO-GO**, but the owner has now explicitly
authorized a full native-SQG W4A8 measurement build using the corrected
encoder. Construction is active; release acceptance is not. No corrected
full-model KLD, LAVD, Estonia, integrated serving result, or claim of lower KLD
exists yet.

The [current construction-status record](docs/full_w4a8_construction_status_2026-08-11.md)
freezes the full-W4A8 format, per-layer beta/build-binding policy, 19-wave
fixed-point DAG, production-v2 lineage, compact retention contract, runtime
static-integration boundary, and exact limits of the first-wave launch.

| Measurement | Result | What it establishes |
|---|---:|---|
| Raw encoded-weight NMSE, 3,072 matched tensors | SQG `15.105%` lower than MCG | SQG is geometrically closer to BF16 under ordinary weight NMSE |
| Fixed 75%-local `H13`, held-out complete-expert NMSE | `5.985%` lower than shared-`H13` SQG; `7.951%` lower than MCG | Expert-local calibration better matches the isolated routed expert function |
| Direct E4M3 label endpoint, 3,072 late-block tensors | RNE-rounding regularized MCG LUT labels before Hadamard/scales increased error `4.515%`; SQG label-conversion SSE was zero | The final reconstructed MCG tensor was not rounded; this is not a W4A8 model result |
| Late 74--77 five-boot A16 KLD | `0.0624264841` vs r33 `0.0624498626` | `-0.03744%`, far inside repeat noise; no obvious scalar catastrophe and no demonstrated KLD win |
| Late alpha-0.25 signed top-8 holdout | `2.7705%` worse than MCG; `39.0295%` of positions improved | The separated-layer blend does not transfer cleanly to the late block |
| Late 74--77 all-arm H13 panel | MCG retained every formal hard-gate decision; alpha 0.25 was the stable SQG diagnostic | 25% expert-local/75% layer-shared is the SQG prior, not an MCG-beating result |
| Test 8b full W4A8 versus matched SQG A16 | `+22.9802%` selection NMSE; `+22.7790%` secondary-holdout NMSE | The current A8 policy is a replicated functional regression and fails the quality gate |
| Test 8c compact-core speed | projected `1.1310x` at M=3,072 and `1.0684x` at M=4,096, assuming MoE fraction `0.31` | The native FP8 path is real, but the serial-core projection misses the `1.15x` long-prefill migration floor and is not serving acceptance |
| Test 8c route-packed hybrid layer | final same-environment r2: `1.1780x` at M=3,072 and `1.1685x` at M=4,096 versus dispatch-matched A16 | The M64xN256/K128 kernel repairs the original slowdown without changing encoded bytes; 31%-MoE Amdahl projections are only `1.0491x`/`1.0468x`, so integrated serving remains unaccepted |
| PR11 route-packed full W4A8 | `1.7550x` at M=3,072 and `1.8236x` at M=4,096; projected `1.1539x`/`1.1628x` end to end at declared MoE fraction `0.31` | The isolated long-prefill speed arm clears the preregistered floor; actual four-GPU workload fraction and integrated serving remain unmeasured |
| Same-candidate-path 5v5 null | mean-delta p95 envelope `±0.0013867` | Ten reused boots provide a checkpoint/prompt-specific empirical reference, not independent experiments or a formal false-positive calibration |
| Late paired trace versus one r33-r33 pair | MoE drift ratio `1.49–1.72x`; post-residual `1.04–1.29x` | Treatment-consistent but unreplicated mechanistic evidence; not a causal estimate or proof that compounding is absent |
| Winner-native layer-77 profile | `-1.5202%` selection and `-0.8131%` untouched holdout versus the frozen profile | Repeating profile search under alpha 0.25 fixes a measurable home-field mismatch on the 16-expert panel |
| Exact h-A8 `(H,B)` down re-encode | `+0.9885%` holdout NMSE despite `-25.08%` cross-expert error | The floating refit does not survive K3/K4 trellis re-encoding; reject the full-strength target |
| Unary-bounded retained-profile selection | `-0.4684%` holdout NMSE; bootstrap crosses zero | Favorable expert-private profile signal; `93.32%` of selection gain is unary, not cancellation |
| Coordinate-corrected full-W4A8 `(H,B)` down repair | `-9.5762%` selection and `-9.3043%` secondary-holdout signed top-8 NMSE versus base full W4A8 | Correct caller coordinates plus exact dual-scale anchoring recover about half of Test 8b's activation damage; repaired W4A8 remains `+10.22%`/`+10.86%` versus SQG A16 |
| Fit-only full-W4A8 beta panel | selected `beta=0.0625`; `-0.3344%` SSE versus beta 0; beta 1 NMSE `0.04367` versus winner `0.01270` | Full-strength candidate conditioning is rejected; the selected beta now governs exact triplet allocation and final encoding |
| Same-rate encoder batching | K4x2/K4x4 changed bytes; K3x4 failed on the first selected-beta high-route-mass panel expert | Production encoding remains singleton per tensor; build time changes, not route-packed inference speed |

The direct E4M3 result strengthens SQG's W4A8 architecture thesis, but its full
`16.8964%` SQG-versus-MCG E4M3 NMSE gap is not solely an endpoint benefit:
SQG was already `13.1439%` better at A16. The completed GLM tests exposed heavy
damage from `act = SiLU(gate) * up` and a poor initial kernel schedule. PR11
repairs the isolated large-M speed path, and corrected candidate-path `(H,B)`
down calibration recovers roughly half of the activation penalty. Neither
result establishes model-level quality: the late block remains a scalar
no-catastrophe screen, and the old `25%` local / `75%` shared `H13` result is
not frozen as the fleet-wide answer.

## Current late-block and W4A8 status

The late contiguous block, layers 74-77, is the first of the preregistered
early/middle/late blocks.

| Stage | Status |
|---|---|
| 217-document plan | Sealed: 253,863 whole-document rows per layer; 150,368 fit, 51,232 selection, 52,263 holdout |
| Routed capture | Complete and sealed for layers 74-77 |
| Four-GPU layer preparation | Complete: 256 permutations/layer, fit-only, zero MCG inputs, zero fallback |
| Profile search | Complete: all 16 preregistered cells/layer; selected identity draw 0 for layers 74-76 and identity draw 3 for layer 77 |
| Alpha-0.25 encoding | Complete: alpha is 25% expert-local/75% layer-global; 1,024 experts and 3,072 SQG tensors |
| Rate/codebook census | Complete: 1,536 K3 + 1,536 K4; zero MCG tensors in treatment layers |
| Candidate materialization | Complete and separately sealed without mutating the protected source model |
| First KLD attempt | Excluded: failed before inference on a stale historical reserved-layer assertion; zero accepted records and no KLD result |
| Corrected five-boot A16 KLD | Complete: mean `0.0624264841`, `-0.03744%` versus r33, repeat-noise inconclusive |
| Same-candidate-path null | Complete: a second five-boot set and all balanced 5v5 partitions provide a prompt/runtime empirical reference; the identity record does not independently bind full checkpoint bytes |
| Routing/residual trace | Complete with one unchanged-r33 pair; treatment-consistent but unreplicated, with no causal compounding conclusion |
| Late signed top-8 holdout | Complete: alpha 0.25 was `2.7705%` worse than MCG; `97.7397%` of the regression was individual-expert SSE |
| Late-specific alpha panel | Complete: MCG retained the formal hard-gate win; alpha 0.25 was the best SQG diagnostic at `+4.3937%` selection and `+2.7705%` secondary-holdout mean NMSE versus MCG |
| Support-conditioned alpha analysis | Complete: alpha 0.25 won all four effective-support quartiles; the association is diagnostic and does not justify per-expert alpha selection |
| Exploratory layerwise alpha search | Complete: 0 of 625 all-SQG mappings passed MCG hard gates; the frozen uniform alpha-0.25 mapping also failed holdout |
| Test 8b activation quality | **Red**: full W4A8 was `22.9802%`/`22.7790%` worse than matched SQG A16 on selection/secondary holdout |
| Test 8c route-packed speed | **Isolated speed gate green, integrated serving open**: PR11 full W4A8 is `1.7550x`/`1.8236x` faster than A16 at M=3,072/4,096 and projects to `1.1539x`/`1.1628x` at the declared 31% MoE fraction |
| Corrected full-W4A8 down calibration | Complete on all layer-77 experts: `9.5762%`/`9.3043%` better than base full W4A8 on selection/secondary holdout, but still `10.22%`/`10.86%` worse than SQG A16 |
| Fit-only beta selection | Complete: `beta=0.0625`; all 16 receipts validated; selection and holdout unused; beta 1 rejected |
| Full native-W4A8 build | Active: layers 3--6 have sealed BF16/capture inputs and completed preparation; bootstrap native-W4A8 profile search launched under canonical source binding `32b9a4b1...`; no final profile selections, beta choices, encoded layers, progressive checkpoint, or full-model quality result yet |

The initial failure remains excluded. A later run-3 validator rejection exposed
a float32-roundoff floor that was smaller than one machine epsilon; the saved
position vector was revalidated under a pinned two-epsilon floor without
changing model, logits, or tensor bytes. Five accepted boots then completed.

Test 10 establishes scalar KLD parity/no obvious scalar catastrophe for one
late four-layer A16 block; it is not a general SQG quality win. The completed
alpha panel makes alpha 0.25 a reproducible historical SQG prior while
retaining MCG as the formal winner. The corrected down objective substantially
repairs but does not erase Test 8b's W4A8 quality gap. The full quant is now
being built under the selected fit-only beta and a W4A8-native profile/rate
process; integrated serving and final quality remain acceptance gates.

The late null does not numerically clear Test 9's separated-layer tail failure:
it is not a matched-checkpoint null for Test 9, and those harmful-tail metrics
exceed even the late p95 reference.

## Start here

- [Complete experiment ledger](docs/sqg_mcg_experiment_record.md) — methodology,
  holdouts, assumptions, discovered errors, validation, results, and the next
  hypothesis for every test.
- [Raw encoded NMSE report](results/raw_encoded_nmse_sqg_vs_mcg.md)
- [Hessian-weighted NMSE report](results/hessian_weighted_nmse_sqg_vs_mcg.md)
- [Expert-local calibration report](results/recalibrated_sqg_vs_mcg.md)
- [Paired KLD analysis](results/h13e_vs_current_sqg_kld.json)
- [Test 9 alpha-blend selection](results/h13_blend_selection_r1.md)
- [Test 9 encoder-unseen holdout](results/h13_blend_holdout_alpha025_r1.md)
- [Test 9 final-logit KLD](results/h13_blend_final_kld_r1.md)
- [Test 9 routing/residual trace](results/h13_blend_tail_trace_alpha025_r1.md)
- [Test 10 compact late-block evidence](published_evidence/contiguous_late/README.md)
- [Direct E4M3 endpoint result](results/e4m3_endpoint_distortion_late_r1.md)
- [Late signed top-8 holdout result](results/signed_top8_contig_late_holdout_r1.md)
- [Same-checkpoint KLD null](results/contiguous_late_same_checkpoint_tail_null_r1.md)
- [Late trace analysis](results/contiguous_late_tail_trace_a025_r1.md)
- [Same-checkpoint trace null](results/contiguous_late_same_checkpoint_tail_trace_null_r1.md)
- [Late all-arm H13 selection](results/contiguous_late_h13_blend_selection_r1.md)
- [Late all-arm H13 holdout](results/contiguous_late_h13_blend_holdout_full_r1.md)
- [Test 8b activation-quality result](results/glm52_w4a8_activation_quality_l077_r1.md)
- [Test 8c compact-core speed result](results/glm52_sqg_w4a8_core_benchmark_l077_r1.md)
- [Test 8c route-packed hybrid kernel result](docs/route_packed_w4a8_kernel_2026-08-11.md)
- [Test 8c route-packed final r2 machine-readable summary](results/glm52_sqg_route_packed_w4a8_v2_l077_r2.json)
- [Winner-native profile result](results/glm52_alpha025_winner_native_profile_l077_r1.md)
- [Exact h-A8 `(H,B)` down re-encode](results/glm52_uncoupled_h_a8_xterm_down_l077_r1.md)
- [Unary-bounded retained-profile co-routing](results/glm52_retained_profile_corouting_l077_r1.md)
- [Coordinate-corrected full-W4A8 down repair](results/glm52_full_w4a8_xterm_coordinate_fixed_l077_r2.md)
- [Fit-only full-W4A8 beta selection](results/glm52_full_w4a8_beta_selection_l077_r1.md)
- [Same-rate encoder batching equivalence audit](results/glm52_w4a8_encoder_batch_equivalence_r1.md)
- [Full native-W4A8 construction status](docs/full_w4a8_construction_status_2026-08-11.md)
- [Sealed 19-wave fixed-point plan](contracts/full_model_rolling_wave_plan_v2.json)
- [Conditional full-model plan and NO-GO decision](docs/conditional_full_model_sqg_w4a8_plan.md)
- [W4A8 code identity and omitted-evidence receipt](published_evidence/w4a8_core/README.md)
- [QSRT/Kimi K3 K1 feasibility audit](docs/qsrt_kimi_k3_k1_feasibility.md)
- [Fast parallel encoding notes](docs/fast_parallel_sqg_encoding.md)
- [Publication scope and excluded payloads](PUBLISHING_SCOPE.md)
- [Publication validation](PUBLICATION_VALIDATION.md)
- [Original implementation README](docs/implementation_readme_historical.md)

Machine-readable results live in [`results/`](results/), experiment contracts
in [`contracts/`](contracts/), compact execution evidence in
[`published_evidence/`](published_evidence/), and runnable analysis/encoding
code in [`scripts/`](scripts/), [`src/`](src/), and
[`bmmlaw_r7_encoder/`](bmmlaw_r7_encoder/).

## KQuant provenance

[`kquant/`](kquant/) is a vendored working tree based on
[`local-inference-lab/kquant`](https://github.com/local-inference-lab/kquant) at
commit `104dd9233f850a3955f4991bea68b07dd34deeb8`. It includes the local SQG
integration changes used here. See [KQuant local
provenance](kquant/LOCAL_PROVENANCE.md) and
[`kquant/LOCAL_CHANGES.patch`](kquant/LOCAL_CHANGES.patch).

## Important limits

- Raw and Hessian-weighted NMSE are proxies, not model-level KLD.
- The final KLD comparison uses one fixed 2,047-position prompt and five boots
  per arm. It does not estimate document-level or task-level generalization.
- Test 9's holdout was encoder-unseen but had already been inspected during
  Test 7, so it is not a globally blind confirmation set.
- Test 8b is a deterministic layer-77 routed-function oracle, not final-logit
  KLD. Its holdout is encoder-unseen but analysis-seen, so it is secondary
  replication rather than a newly blind confirmation corpus.
- Test 8c-core measures dense compact per-tensor calls and a serial route-
  histogram projection. The later route-packed test is a complete isolated
  layer benchmark, but it is still not an integrated vLLM/DCP4 end-to-end
  serving benchmark.
- The earlier four selected layers were separated; the new 74--77 block is
  contiguous but covers only 5.33% of the 75 routed layers.
- The MCG baseline and SQG candidate did not share the same selected-layer
  dispatch path. The preserved `+0.64057%` four-layer SQG-versus-r33 result is
  consequently a weight-plus-dispatch result, not a codebook-only result.
- This Git repository intentionally excludes model-scale tensor payloads and
  regenerated caches. Their manifests, hashes, receipts, and compact logs are
  preserved in `published_evidence/`.
- Direct MCG-versus-SQG E4M3 weight-endpoint distortion, GLM W4A8 activation
  quality, compact-core and route-packed speed, and a corrected exact-path
  `(H,B)` down repair have been measured. Full-model DCP4 serving, final-logit
  W4A8 KLD, LAVD, Estonia, and release quality remain unmeasured.

No inference service or GPU process is started by this repository.
