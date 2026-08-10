# Hessian-weighted encoded distortion: SQG versus MCG

Lower is better. All reported matrix traces and routed-output reductions are exact FP32 GPU computations; this is not KLD.

## Overall

| Metric | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| Gate/up: layer-global H13 | 512 | 1.129762493e-02 | 6.380428551e-03 | +43.524% | 512 (100.0%) |
| Gate/up: expert-local H13_e | 512 | 5.128204096e-03 | 5.395821598e-03 | -5.219% | 131 (25.6%) |
| Down: H2 from BF16 upstream | 256 | 3.755475576e-03 | 2.810145087e-03 | +25.172% | 256 (100.0%) |
| Down: H2 from MCG upstream | 256 | 3.748343139e-03 | 2.817932561e-03 | +24.822% | 256 (100.0%) |
| Down: H2 from SQG upstream | 256 | 3.757070546e-03 | 2.800080736e-03 | +25.472% | 256 (100.0%) |

## Contract

- Gate/up layer-global scores use each sealed fit-only H13 plus the encoder's 0.025 diagonal damping.
- Expert-local gate/up scores replay the exact fit routes and applied gate-square weights, with the same diagonal damping.
- Down scores rebuild the frozen OAS-to-scaled-identity H2 under BF16, decoded-MCG, and decoded-SQG upstream paths; each common H2 scores both down reconstructions.
- MCG and SQG are decoded from their packed bytes and mapped back to the same official BF16 coordinate order.
- The K3/K4 assignment and BF16 source payload are held fixed.

## Interpretation limit

Hessian-weighted local distortion is closer to functional sensitivity than raw NMSE, but it still does not include cross-layer compounding, changed routing, runtime dispatch, or final-logit KLD.
