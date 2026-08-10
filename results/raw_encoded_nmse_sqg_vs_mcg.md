# Raw encoded-weight NMSE: SQG versus MCG

Lower is better. This is an encoded-weight distortion comparison, not KLD.

## Overall result

- Matched tensors: **3,072**
- MCG energy-weighted NMSE: **`1.593874402e-02`**
- SQG energy-weighted NMSE: **`1.353123658e-02`**
- SQG/MCG: **`0.848952`**
- SQG raw-NMSE reduction: **`+15.105%`**
- SQG tensor wins: **3,033 / 3,072 (98.7%)**
- MCG exceptions: **39**; 32 were K3 and 7 were K4
- MCG exceptions by layer: L6=29, L28=1, L52=9, L77=0

### By rate

| Group | Tensors | MCG NMSE | SQG NMSE | SQG/MCG | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 1,536 | 2.577473051e-02 | 2.180891375e-02 | 0.846135 | +15.386% | 1,504 (97.9%) |
| 4 | 1,536 | 6.142337435e-03 | 5.286868681e-03 | 0.860726 | +13.927% | 1,529 (99.5%) |

### By projection

| Group | Tensors | MCG NMSE | SQG NMSE | SQG/MCG | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1,024 | 8.240417637e-03 | 6.907981213e-03 | 0.838305 | +16.170% | 1,013 (98.9%) |
| gate_proj | 1,024 | 1.822427758e-02 | 1.550568960e-02 | 0.850826 | +14.917% | 1,010 (98.6%) |
| up_proj | 1,024 | 2.156730274e-02 | 1.836553851e-02 | 0.851545 | +14.845% | 1,010 (98.6%) |

### By layer

| Group | Tensors | MCG NMSE | SQG NMSE | SQG/MCG | SQG reduction | SQG wins |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 768 | 1.771224243e-02 | 1.556055110e-02 | 0.878520 | +12.148% | 739 (96.2%) |
| 28 | 768 | 1.539679504e-02 | 1.297328979e-02 | 0.842597 | +15.740% | 767 (99.9%) |
| 52 | 768 | 1.503056828e-02 | 1.277436628e-02 | 0.849892 | +15.011% | 759 (98.8%) |
| 77 | 768 | 1.555572863e-02 | 1.284144010e-02 | 0.825512 | +17.449% | 768 (100.0%) |

## Method

- MCG was unpacked and decoded entirely on CPU from its production trellis bytes and stored FP16 scale vectors.
- Every MCG decoded tensor was required to match its sealed production reconstruction SHA-256 byte-for-byte.
- SQG NMSE is the square of `source_relative_rmse` from the independent packed-byte SQG closure recorded during encoding.
- Each pair was required to bind to the same official BF16 tensor payload and the same K3/K4 assignment.
- Aggregate NMSE is `sum(SSE) / sum(BF16 weight energy)`; per-tensor macro and paired statistics are retained in the JSON.
- No GPU was visible to this process.

## Interpretation limit

Raw weight NMSE does not include Hessian sensitivity, routing, error compounding, activation quantization, or runtime dispatch. It answers whether the stored SQG weights are geometrically closer to BF16 than the stored MCG weights, not whether full-model KLD is lower.
