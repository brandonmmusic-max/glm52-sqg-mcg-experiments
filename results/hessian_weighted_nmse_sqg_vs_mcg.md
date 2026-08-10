# Hessian-weighted encoded distortion: SQG versus MCG

Lower is better. All reported matrix traces and routed-output reductions are exact FP32 GPU computations; this is not KLD.

## Overall

| Metric | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| Gate/up: layer-global H13 | 2,048 | 7.224539694e-03 | 4.192353373e-03 | +41.971% | 2,048 (100.0%) |
| Gate/up: expert-local H13_e | 2,048 | 3.995579165e-03 | 3.833471345e-03 | +4.057% | 760 (37.1%) |
| Down: H2 from BF16 upstream | 1,024 | 6.960821425e-03 | 5.939443275e-03 | +14.673% | 1,024 (100.0%) |
| Down: H2 from MCG upstream | 1,024 | 6.960210586e-03 | 5.948506077e-03 | +14.536% | 1,024 (100.0%) |
| Down: H2 from SQG upstream | 1,024 | 6.963213676e-03 | 5.936689209e-03 | +14.742% | 1,024 (100.0%) |

## Expert-local H13_e by layer

| Group | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| 6 | 512 | 1.552285499e-03 | 1.988494657e-03 | -28.101% | 19 (3.7%) |
| 28 | 512 | 5.128204096e-03 | 5.395821598e-03 | -5.219% | 131 (25.6%) |
| 52 | 512 | 7.041901968e-03 | 6.665499856e-03 | +5.345% | 322 (62.9%) |
| 77 | 512 | 2.934506604e-03 | 2.813184095e-03 | +4.134% | 288 (56.2%) |

## Expert-local H13_e by rate

| Group | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| 3 | 1,413 | 5.133793959e-03 | 4.917614531e-03 | +4.211% | 494 (35.0%) |
| 4 | 635 | 1.429412201e-03 | 1.389211757e-03 | +2.812% | 266 (41.9%) |

## SQG-candidate H2 by layer

| Group | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| 6 | 256 | 3.331942723e-03 | 2.465095468e-03 | +26.016% | 256 (100.0%) |
| 28 | 256 | 3.757070546e-03 | 2.800080736e-03 | +25.472% | 256 (100.0%) |
| 52 | 256 | 5.473721546e-03 | 3.934976263e-03 | +28.112% | 256 (100.0%) |
| 77 | 256 | 7.003111988e-03 | 5.989065134e-03 | +14.480% | 256 (100.0%) |

## Validation

- Maximum SQG-H2 route-mass relative error: `0.000e+00`.
- Maximum SQG-H2 local-alpha absolute error: `0.000e+00`.
- Maximum SQG-H2 identity-scale relative error: `7.172e-06`.


## Contract

- Gate/up layer-global scores use each sealed fit-only H13 plus the encoder's 0.025 diagonal damping.
- Expert-local gate/up scores replay the exact fit routes and applied gate-square weights, with the same diagonal damping.
- Down scores rebuild the frozen OAS-to-scaled-identity H2 under BF16, decoded-MCG, and decoded-SQG upstream paths; each common H2 scores both down reconstructions.
- MCG and SQG are decoded from their packed bytes and mapped back to the same official BF16 coordinate order.
- The K3/K4 assignment and BF16 source payload are held fixed.

## Interpretation limit

Hessian-weighted local distortion is closer to functional sensitivity than raw NMSE, but it still does not include cross-layer compounding, changed routing, runtime dispatch, or final-logit KLD.
