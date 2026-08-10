# GLM-5.2 SQG versus MCG experiments

This repository preserves the code, methodology, compact evidence, and audited
results from the four-layer GLM-5.2 3.5-bpw experiment comparing the existing
MCG trellis representation with KQuant-style SQG. It also contains the
expert-local `H13` follow-up and the final-logit KLD measurements.

The experiment kept the production topology and the per-tensor K3/K4 bit
assignment fixed. Bits were **not** reallocated per expert. The tested layers
were 6, 28, 52, and 77, with 256 experts and three expert projections per layer.

## Bottom line

SQG reconstructed the BF16 weights more closely than MCG in the raw encoded
comparison, and expert-local activation calibration substantially improved the
isolated routed-expert function. Those proxy improvements did **not** produce a
demonstrated final-logit KLD improvement in this four-layer test.

| Measurement | Result | What it establishes |
|---|---:|---|
| Raw encoded-weight NMSE, 3,072 matched tensors | SQG `15.105%` lower than MCG | SQG is geometrically closer to BF16 under ordinary weight NMSE |
| Expert-local `H13`, held-out complete-expert NMSE | `5.985%` lower than shared-`H13` SQG; `7.951%` lower than MCG | Expert-local calibration better matches the isolated routed expert function |
| Expert-local `H13` final KLD | `0.0631130657` mean, five fresh boots | Complete candidate result on one fixed prompt |
| Shared-`H13` SQG final KLD, same dispatch | `0.0628498963` mean, five boots | Valid comparator for the `H13` treatment |
| Expert-local minus shared-`H13` SQG | `+0.41873%` | Mean direction was worse; Welch repeat-noise result was inconclusive |
| Expert-local SQG minus r33 MCG | `+1.06198%` | Still dispatch-confounded; not an isolated codebook comparison |

The strongest supported conclusion is therefore narrower than “SQG wins” or
“MCG wins”: expert-local calibration fixed a real activation-geometry mismatch
in the isolated expert objective, but that improvement did not translate into a
lower measured KLD on this prompt. A native-dispatch MCG control is still
required before attributing the SQG-versus-MCG KLD difference to the codebook.

## Start here

- [Complete experiment ledger](docs/sqg_mcg_experiment_record.md) — methodology,
  holdouts, assumptions, discovered errors, validation, results, and the next
  hypothesis for every test.
- [Raw encoded NMSE report](results/raw_encoded_nmse_sqg_vs_mcg.md)
- [Hessian-weighted NMSE report](results/hessian_weighted_nmse_sqg_vs_mcg.md)
- [Expert-local calibration report](results/recalibrated_sqg_vs_mcg.md)
- [Paired KLD analysis](results/h13e_vs_current_sqg_kld.json)
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
- The four selected layers are separated rather than contiguous, so the test is
  weak for consecutive-layer error compounding and routing drift.
- The MCG baseline and SQG candidate did not share the same selected-layer
  dispatch path. The preserved `+0.64057%` four-layer SQG-versus-r33 result is
  consequently a weight-plus-dispatch result, not a codebook-only result.
- This Git repository intentionally excludes model-scale tensor payloads and
  regenerated caches. Their manifests, hashes, receipts, and compact logs are
  preserved in `published_evidence/`.

No inference service or GPU process is started by this repository.
