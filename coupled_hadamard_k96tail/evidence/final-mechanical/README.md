# Final mechanical evidence

This directory binds the completed non-tensor mechanical campaign closure.
It contains exact source bytes inside deterministic gzip-compressed tar
archives, compact inner SHA-256 manifests, the assembly manifest, and the
authoritative model codec validation receipt.

The mechanical domain is 75 target layers, 3 through 77. All 75 have a
complete layer manifest, quality receipt, and passing B12X runtime oracle.
Exact scorer/encoder byte parity covers the 74 K96 layers 4 through 77. Layer
3 is the sealed K48 exception and has no K96 parity receipt.

The model codec receipt covers all 76 routed layers, including preserved
source-SQG MTP layer 78. It reports 177,613 indexed tensors, the same number on
disk, a passing per-layer census, and no problems.

Verify the closure from `coupled_hadamard_k96tail/` with:

```bash
python3 verify_source_sync.py
```

The verifier checks archive hashes, rejects unsafe archive members, validates
every inner file hash, and enforces the 75-oracle, 74-parity, layer-3-exception
boundary.
