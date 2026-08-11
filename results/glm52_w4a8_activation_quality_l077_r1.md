# Test 8b: GLM-5.2 SQG A8 activation quality

This is a layer-77 functional oracle. It is not final-logit KLD and not a speed benchmark.

## Selection

| Arm | Signed top-8 NMSE | vs SQG A16 | vs sealed MCG A16 scalar | Improved positions vs A16 | Output-stage NMSE |
|---|---:|---:|---:|---:|---:|
| sqg_a16 | 3.665861676e-03 | +0.0000% | +3.4112% | 0.000% | 4.308530715e-03 |
| sqg_h_a8 | 3.809500307e-03 | +3.9183% | +7.4632% | 0.812% | 4.470152206e-03 |
| sqg_act_a8 | 4.369086531e-03 | +19.1831% | +23.2487% | 0.012% | 5.138887635e-03 |
| sqg_w4a8 | 4.508285622e-03 | +22.9802% | +27.1754% | 0.029% | 5.296980418e-03 |

### Activation operands

- `h`: NMSE `6.994922211e-04`, max |x| `0.035057`, pre-clamp overflows `0`.
- `act_from_a16`: NMSE `8.438497104e-04`, max |x| `2.95242`, pre-clamp overflows `0`.
- `act_from_h_a8`: NMSE `8.509878330e-04`, max |x| `2.95408`, pre-clamp overflows `0`.

The sealed MCG-A16 value `3.544936093e-03` is cross-harness scalar context only. It has no Test 8b per-position arm, so no MCG tail comparison is made.

### Magnitude path around H128

- `h`
  - `raw`: exact RMS `0.37254`, exact max `5.40625`, sampled p99 `0.945312`.
  - `after_suh_pre_h`: exact RMS `0.00571859`, exact max `0.0830078`, sampled p99 `0.0145111`.
  - `post_h_qdq_input`: exact RMS `0.00571859`, exact max `0.035057`, sampled p99 `0.0145323`.
- `act_from_a16`
  - `raw`: exact RMS `0.559609`, exact max `1660`, sampled p99 `1.13878`.
  - `after_suh_pre_h`: exact RMS `0.00739356`, exact max `20.7577`, sampled p99 `0.0162306`.
  - `post_h_qdq_input`: exact RMS `0.00739356`, exact max `2.95242`, sampled p99 `0.0161068`.
- `act_from_h_a8`
  - `raw`: exact RMS `0.559737`, exact max `1660.74`, sampled p99 `1.13785`.
  - `after_suh_pre_h`: exact RMS `0.00739405`, exact max `20.7669`, sampled p99 `0.0162011`.
  - `post_h_qdq_input`: exact RMS `0.00739405`, exact max `2.95408`, sampled p99 `0.0160897`.

## Holdout

| Arm | Signed top-8 NMSE | vs SQG A16 | vs sealed MCG A16 scalar | Improved positions vs A16 | Output-stage NMSE |
|---|---:|---:|---:|---:|---:|
| sqg_a16 | 3.742719281e-03 | +0.0000% | +1.5855% | 0.000% | 4.410270298e-03 |
| sqg_h_a8 | 3.892562028e-03 | +4.0036% | +5.6526% | 0.767% | 4.579136304e-03 |
| sqg_act_a8 | 4.448537722e-03 | +18.8584% | +20.7430% | 0.000% | 5.244343858e-03 |
| sqg_w4a8 | 4.595271612e-03 | +22.7790% | +24.7256% | 0.015% | 5.411799739e-03 |

### Activation operands

- `h`: NMSE `6.991316983e-04`, max |x| `0.0342082`, pre-clamp overflows `0`.
- `act_from_a16`: NMSE `8.403531857e-04`, max |x| `2.94037`, pre-clamp overflows `0`.
- `act_from_h_a8`: NMSE `8.480067118e-04`, max |x| `2.9432`, pre-clamp overflows `0`.

The sealed MCG-A16 value `3.684304064e-03` is cross-harness scalar context only. It has no Test 8b per-position arm, so no MCG tail comparison is made.

### Magnitude path around H128

- `h`
  - `raw`: exact RMS `0.376391`, exact max `5.40625`, sampled p99 `0.9375`.
  - `after_suh_pre_h`: exact RMS `0.0057777`, exact max `0.0830078`, sampled p99 `0.014389`.
  - `post_h_qdq_input`: exact RMS `0.0057777`, exact max `0.0342082`, sampled p99 `0.0147419`.
- `act_from_a16`
  - `raw`: exact RMS `0.582674`, exact max `1650.62`, sampled p99 `1.22087`.
  - `after_suh_pre_h`: exact RMS `0.0076905`, exact max `20.6403`, sampled p99 `0.017405`.
  - `post_h_qdq_input`: exact RMS `0.0076905`, exact max `2.94037`, sampled p99 `0.0169715`.
- `act_from_h_a8`
  - `raw`: exact RMS `0.58281`, exact max `1653.25`, sampled p99 `1.21886`.
  - `after_suh_pre_h`: exact RMS `0.00769113`, exact max `20.6732`, sampled p99 `0.0173663`.
  - `post_h_qdq_input`: exact RMS `0.00769113`, exact max `2.9432`, sampled p99 `0.016954`.

## Interpretation

The A16 arm is the matched SQG control. `h-A8` and `act-A8` isolate the two activation sites; `W4A8` composes them. Selection and holdout are reported separately. The holdout was not used by the encoder, but was already inspected in earlier analysis and is therefore secondary confirmation rather than a newly blind split.

Test 8b can reject W4A8 for activation-quality damage. Passing Test 8b cannot by itself greenlight a full quant: Test 8c must still demonstrate the required GLM-shape prefill speed, and this oracle does not measure final-logit KLD.
