# Layer-77 alpha-0.25 winner-native profile search

The original profile grid was selected under a different H13 geometry. This
test re-encoded all 16 draw/scale cells natively under
`0.75 H_layer + 0.25 H_local,e`, rebuilding candidate-conditioned H2 and down
bytes at the unchanged per-tensor K3/K4 assignments for the preregistered 16
experts. Selection used signed routed-output error and a paired document
bootstrap at `1 - 0.05/15`; holdout remained closed until the winner froze.

| Split | draw-03 identity control | draw-00 identity winner | Change |
|---|---:|---:|---:|
| Selection | `0.001910524055` | `0.001881479481` | `-1.5202%` |
| Untouched holdout | `0.002443549285` | `0.002423681864` | `-0.8131%` |

Selection's familywise-adjusted improvement lower bound was
`2.75108e-05`; the separately scored 45-document holdout lower bound was
`1.43841e-05`. Both were above zero.

Disposition: promote draw-0 identity for the complete layer-77 follow-up.
This is a 16-expert profile result, not a full-layer or final-logit KLD win.

- Selection ID:
  `6b88882d109c6e0795c997301dbfa38561e978acce9af0b96ce704fd68ac6158`
