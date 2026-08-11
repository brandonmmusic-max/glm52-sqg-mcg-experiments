# Test 8a: matched E4M3 endpoint distortion

This is a CPU-only weight-operand endpoint test, not KLD or W4A8 activation quality.

## Overall

- Matched tensors: **3,072**
- MCG A16 NMSE: **`1.532616894e-02`**
- MCG label-to-E4M3 NMSE: **`1.601821470e-02`**
- MCG BF16-error change: **`+4.5155%`**
- MCG conversion energy relative to its A16 BF16 error: **`+4.6102%`**
- MCG cross-term relative to its A16 BF16 error: **`-0.0948%`**
- SQG A16 NMSE: **`1.331171565e-02`**
- SQG native-E4M3 NMSE: **`1.331171565e-02`**
- SQG exact native endpoints: **True**

## By rate

| Rate | Tensors | MCG A16 | MCG E4M3 | MCG error change | SQG native E4M3 |
|---:|---:|---:|---:|---:|---:|
| K3 | 1,536 | 2.469302900e-02 | 2.538500605e-02 | +2.8023% | 2.129184017e-02 |
| K4 | 1,536 | 5.988110978e-03 | 6.680225249e-03 | +11.5581% | 5.356129160e-03 |

## Interpretation

The MCG conversion-energy term is the cost of moving its realized labels to E4M3. The BF16-error change also includes a signed cross-term with the existing quantization error, so it need not equal the conversion energy and can even improve individual tensors by chance.

SQG's zero conversion term establishes only its exact weight-label endpoint. It does not remove E4M3 activation error, prove W4A8 KLD parity, or demonstrate a speedup.
