# Layer-77 unary-bounded retained-profile co-routing

This preregistered pilot made no new tensor bytes. It selected one of four
already encoded mixed-K3/K4 identity-profile draws for each of the frozen 16
panel experts. Selection minimized exact 6,144-dimensional signed,
gate-weighted top-8 error under per-expert unary slack arms of `0%`, `0.25%`,
`0.5%`, and `1%`. All four arms passed the Bonferroni-adjusted selection gate;
only the frozen `1%` winner was opened on holdout.

| Endpoint | Layer-shared draw 0 | Expert-private assignment | Change |
|---|---:|---:|---:|
| Selection NMSE | `0.001881479481` | `0.001879547687` | `-0.102674%` |
| Holdout NMSE | `0.002423681864` | `0.002412329518` | `-0.468393%` |

The holdout document bootstrap favored the candidate on average but crossed
zero: `[-8.96502e-07, 1.77375e-05]`. Of the selection SSE gain, `93.32%` came
from better individual-expert profiles and `6.68%` from the cross-expert term.

Across all holdout positions, `23.3664%` improved, `22.4824%` worsened, and
`54.1511%` were unchanged because no panel expert was routed. Among affected
positions, `50.9640%` improved. Mean error fell, but p99 rose `0.1926%`,
CVaR-1% rose `0.1073%`, and the maximum rose `13.5931%`.

Disposition: favorable directional evidence for expert-private profile
selection, not a promotion or full-quant gate pass. Use unary quality as the
primary selector and co-routing only inside a strict bound. A tail-aware rule
requires fresh documents.

- Preregistration ID:
  `7e741eab97be9f1056fcd50f73f7019e9b86ded27a00f40a32ce127026a70346`
- Result ID:
  `556d0d43919535821c0b0469369b69090487df7a95f3f7f09304076de5c8ede3`
- Result file SHA256:
  `11e483476866862b47f9fc74cd1c715fbcabdbe228bb5fd9a4b48a2b4361304b`
