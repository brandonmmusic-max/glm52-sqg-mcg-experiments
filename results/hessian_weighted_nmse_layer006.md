# Hessian-weighted encoded distortion: SQG versus MCG

Lower is better. All reported matrix traces and routed-output reductions are exact FP32 GPU computations; this is not KLD.

## Overall

| Metric | Tensors | MCG HNMSE | SQG HNMSE | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|
| Gate/up: layer-global H13 | 512 | 4.906789746e-03 | 2.283047786e-03 | +53.472% | 512 (100.0%) |
| Gate/up: expert-local H13_e | 512 | 1.552285499e-03 | 1.988494657e-03 | -28.101% | 19 (3.7%) |
| Down: H2 from BF16 upstream | 256 | 3.335070369e-03 | 2.464644686e-03 | +26.099% | 256 (100.0%) |
| Down: H2 from MCG upstream | 256 | 3.329277077e-03 | 2.465691007e-03 | +25.939% | 256 (100.0%) |
| Down: H2 from SQG upstream | 256 | 3.331942723e-03 | 2.465095468e-03 | +26.016% | 256 (100.0%) |

## Contract

- Gate/up layer-global scores use each sealed fit-only H13 plus the encoder's 0.025 diagonal damping.
- Expert-local gate/up scores replay the exact fit routes and applied gate-square weights, with the same diagonal damping.
- Down scores rebuild the frozen OAS-to-scaled-identity H2 under BF16, decoded-MCG, and decoded-SQG upstream paths; each common H2 scores both down reconstructions.
- MCG and SQG are decoded from their packed bytes and mapped back to the same official BF16 coordinate order.
- The K3/K4 assignment and BF16 source payload are held fixed.

## Interpretation limit

Hessian-weighted local distortion is closer to functional sensitivity than raw NMSE, but it still does not include cross-layer compounding, changed routing, runtime dispatch, or final-logit KLD.
