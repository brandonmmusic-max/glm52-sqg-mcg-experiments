# Hessian-weighted encoded distortion: SQG versus MCG

Lower is better. All reported matrix traces and routed-output reductions are exact FP32 GPU computations; this is not KLD.

## Overall

| Metric | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| Gate/up: layer-global H13 | 512 | 1.068949903e-02 | 6.310977626e-03 | +40.961% | 512 (100.0%) |
| Gate/up: expert-local H13_e | 512 | 7.041901968e-03 | 6.665499856e-03 | +5.345% | 322 (62.9%) |
| Down: H2 from BF16 upstream | 256 | 5.488944531e-03 | 3.968464347e-03 | +27.701% | 256 (100.0%) |
| Down: H2 from MCG upstream | 256 | 5.466265578e-03 | 3.984358028e-03 | +27.110% | 256 (100.0%) |
| Down: H2 from SQG upstream | 256 | 5.473721546e-03 | 3.934976263e-03 | +28.112% | 256 (100.0%) |

## Contract

- Gate/up layer-global scores use each sealed fit-only H13 plus the encoder's 0.025 diagonal damping.
- Expert-local gate/up scores replay the exact fit routes and applied gate-square weights, with the same diagonal damping.
- Down scores rebuild the frozen OAS-to-scaled-identity H2 under BF16, decoded-MCG, and decoded-SQG upstream paths; each common H2 scores both down reconstructions.
- MCG and SQG are decoded from their packed bytes and mapped back to the same official BF16 coordinate order.
- The K3/K4 assignment and BF16 source payload are held fixed.

## Interpretation limit

Hessian-weighted local distortion is closer to functional sensitivity than raw NMSE, but it still does not include cross-layer compounding, changed routing, runtime dispatch, or final-logit KLD.
