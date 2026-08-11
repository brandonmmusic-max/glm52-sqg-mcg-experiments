# Layer-77 exact h-A8 `(H,B)` down re-encode

The all-256-expert test reused the completed alpha-0.25 winner-native gate/up
bytes, constructed the exact h-A8 upstream candidate, fitted both normal-
equation terms `(H,B)` on gate-square-weighted fit rows, and then encoded the
fitted down target at each tensor's unchanged K3 or K4 assignment. Selection
and holdout were excluded from fitting. Signed top-8 outputs were summed before
squaring.

The floating target improved fit SSE for every expert, with a `53.2637%`
median improvement. The realized bytes failed:

| Split | Base h-A8 NMSE | Re-encoded NMSE | Relative change | Position wins |
|---|---:|---:|---:|---:|
| Selection | `0.003787687545` | `0.003829293437` | `+1.098451%` | `32.2064%` |
| Holdout | `0.003891270008` | `0.003929733740` | `+0.988462%` | `33.2511%` |

On holdout, individual-expert SSE rose `1.1136%` while the cross-expert term
improved `25.0816%`. The individual damage overwhelmed the cancellation gain.
The paired-document interval was wholly adverse:
`[-4.41676e-05, -2.38926e-05]` under the baseline-minus-candidate convention.

Disposition: reject the full-strength fitted target. A floating refit oracle
is not evidence of quantized quality unless its realized trellis bytes pass.

- Preregistration SHA256:
  `14ec09cb567101e98bd106e70e9384d95bcbb6d117be79af5a4a132ca8e5d273`
- Result ID:
  `fb110d50d540ea6384d6b4c85fc88f2931b2d3b0249296e451a21ea5db3e9002`
