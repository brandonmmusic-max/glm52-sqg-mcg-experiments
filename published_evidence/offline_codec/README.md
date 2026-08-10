# GLM-5.2 no-BF16 MCG to SQG experiment

This directory is a fresh, source-preserving experiment for answering one
narrow question: can KQuant's `sqg_xor_cheb_t12` codebook move the existing
GLM-5.2 3.5-bpw checkpoint's BF16-logit KLD in the right direction without
access to the original BF16 weights or captured Hessians?

`convert_legacy_mcg_to_sqg.py` decodes each complete topology-neutral EXL3 MCG
tensor into its stored regularized reconstruction and re-encodes that tensor
with KQuant's production SQG Viterbi kernel. It preserves the original tensor's
K3/K4 rate and copies its existing `suh` and `svh` tensors without alteration.
K5 cannot be converted by this SQG contract. The final four-layer candidate
therefore selects only source layers with no K5 tensors. Converted tensors use
the exclusive `<base>.sqg` int32 marker `0x53514731` (`SQG1`) and never retain
`<base>.mcg` in a selected layer.

This is quantization of an already quantized reconstruction. It is a valid
directional end-to-end KLD experiment, but it is not equivalent to encoding
from BF16 with the original expert-local Hessian. A negative result cannot
disprove SQG-from-BF16, and a positive result should be confirmed by a clean
BF16/Hessian re-quantization.

The final directional gate uses layers `[6, 28, 52, 77]`: the earliest
all-SQG-eligible routed layer, approximately one-third, two-thirds, and the
final routed layer. Every selected layer has exactly 384 K3 and 384 K4 tensors,
so the original 3.5-bpw per-tensor allocation is preserved exactly. A completed
layer-3 staging encode is retained only as unused evidence because its source
bit map contains 28 K5 tensors; it must not enter the final candidate.

`validate_legacy_sqg_shard.py` closes the completed shard against the source,
including the exclusive marker set, every converted packed hash, unchanged
legacy shapes/dtypes and payload bytes, and an aggregate byte hash over every
stored `suh`/`svh` tensor.

`augment_reconstruction_provenance.py` binds every converted packed tensor to
its exact contiguous regularized FP16 reconstruction hash. It also records a
runtime-folded post-Hadamard FP16 hash for deterministic runtime provenance.
That latter domain includes FP16 rounding at `had_r_128` intermediates and is
not byte-comparable to the source quantizer's numeric-core
FP32-Hadamard/final-FP16 reconstruction hash; only close numerical agreement is
expected across those domains.

`materialize_multilayer_candidate.py` hardlink-clones the protected source,
replaces only the four selected shards and their metadata, and writes explicit
per-layer SQG overrides with an empty tensor-override map. The final
`audit_multilayer_candidate.py` requires a selected-layer census of 3,072 SQG
markers and zero MCG markers, closes every model index/header and every changed
payload hash, binds all reconstruction evidence, and rechecks source
immutability.
