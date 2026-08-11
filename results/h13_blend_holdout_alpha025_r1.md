# Signed top-8 H13 blend score — holdout

Errors are summed as signed router-weighted expert-output vectors before squaring.

| Blend | Signed top-8 NMSE | Sum/individual SSE | Positions improved | Squared-error CVaR 1% | Relative-error CVaR 1% | p99 relative error |
|---|---:|---:|---:|---:|---:|---:|
| alpha0 | 2.548697629e-03 | 1.003272 | 0.000% | 6.733437345e+02 | 2.219782997e-02 | 1.967832881e-02 |
| alpha025 | 2.385867558e-03 | 1.002277 | 89.261% | 6.710959635e+02 | 2.191765363e-02 | 1.934645514e-02 |

## Selection

- Status: `tail_and_mean_constraints_passed`
- Eligible candidates: `['alpha025']`
- Selected candidate: **`alpha025`**
- Diagnostic nonbaseline fallback: `None`

The improved-position fraction is optimized only after aggregate and positive-tail constraints pass.
