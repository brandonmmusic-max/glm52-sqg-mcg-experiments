# Same-checkpoint KLD tail null calibration

## Result

This report pools two independently collected five-boot groups of the exact same sealed late-block SQG checkpoint. It enumerates all 126 unique balanced 5-vs-5 partitions and both directions (252 directional comparisons).

The observed group-A minus group-B mean is `0.000311603876195`. Its direction is arbitrary: there is no treatment difference between the groups.

## Empirical 95% null gates

| Metric | Fail only above |
|---|---:|
| `delta_p99` | `0.144605771245` |
| `delta_p99_5` | `0.224204237381` |
| `positive_delta_p99` | `0.144605771245` |
| `positive_delta_cvar_1pct` | `0.260966408274` |
| `positive_delta_mass` | `12.0395563995` |
| `maximum_regression` | `0.813911502361` |

The two-sided 95% absolute mean-delta envelope is `0.00138669808587`. The same-checkpoint 95% interval for the fraction of positions where the arbitrarily named left group is lower is `[0.476086956522, 0.510014655594]`.

The retained decision rule is:

> A future treatment fails a tail metric only when its directional harm exceeds the corresponding same-checkpoint p95 null envelope; retain raw preregistered values alongside the null-calibrated verdict.

## Assumptions and limitations

- The envelope describes fresh-boot variation for one checkpoint, prompt, and FP8-KV runtime regime.
- Balanced partitions reuse ten observed boots and are not independent experiments.
- This calibration does not increase the statistical power of a four-layer treatment comparison.

This calibration changes the interpretation of the tail gate; it does not create power to resolve a small four-layer codebook effect.
