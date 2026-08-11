# Signed top-8 H13 blend score — selection

Errors are summed as signed router-weighted expert-output vectors before squaring.

| Blend | Signed top-8 NMSE | Sum/individual SSE | Positions improved | Squared-error CVaR 1% | Relative-error CVaR 1% | p99 relative error |
|---|---:|---:|---:|---:|---:|---:|
| alpha0 | 2.539505205e-03 | 1.002886 | 0.000% | 6.399615548e+02 | 2.240171991e-02 | 1.936763292e-02 |
| alpha025 | 2.373400202e-03 | 1.002068 | 89.701% | 6.374145670e+02 | 2.191427300e-02 | 1.887691540e-02 |
| alpha050 | 2.356708812e-03 | 1.000681 | 89.288% | 6.402745804e+02 | 2.214374643e-02 | 1.901942424e-02 |
| alpha075 | 2.378155986e-03 | 1.000076 | 82.537% | 6.480091546e+02 | 2.285086742e-02 | 1.961015985e-02 |
| alpha100 | 2.449385903e-03 | 1.000746 | 57.228% | 6.348703476e+02 | 2.562192395e-02 | 2.189990710e-02 |

## Selection

- Status: `tail_and_mean_constraints_passed`
- Eligible candidates: `['alpha025']`
- Selected candidate: **`alpha025`**
- Diagnostic nonbaseline fallback: `None`

The improved-position fraction is optimized only after aggregate and positive-tail constraints pass.
