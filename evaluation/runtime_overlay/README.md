# GLM-5.2 SQG runtime overlay

Mount this directory at `/sqg-runtime-overlay` and prepend it to
`PYTHONPATH`. Python imports `sitecustomize.py` automatically; its narrow
import hook installs the exact stopped-production r33 Python boot files, with
only `vllm.model_executor.layers.quantization.exl3` carrying the SQG delta.
The production image's installed `b12x` remains authoritative for every MCG,
MoE, and attention path. Public B12X 1.2.1 commit
`7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b` is vendored under the isolated
Python/compiler/custom-op namespace `b12x_sqg` and is called only for SQG.

Converted matrices use an exclusive scalar `torch.int32` marker named
`<base>.sqg` with value `0x53514731`. Their `.mcg` marker must be absent.
Unconverted MCG matrices remain on the ExLlamaV3 path; SQG matrices dispatch
independently through `b12x_sqg.gemm.trellis_linear` with explicit codebook
`sqg_xor_cheb_t12`. `r7_routed_experts.codebook_overrides` selects SQG by
layer, while the stricter `codebook_tensor_overrides` map takes precedence for
optional mixed-codebook compatibility. The final pure-SQG experiment does not
use tensor overrides.

The stratified KLD arm records its selected routed layers in checkpoint
metadata. Any layer that contains SQG is forced off the MCG-only R7
fused/graph paths and retains native per-projection topology; each projection
then dispatches by its exclusive marker. Run vLLM with `--enforce-eager`, as
required by the existing non-rank-sliced EXL3 backend.

For the fresh four-layer experiment,
`VLLM_EXL3_R7_FUSED_ALLOWLIST` is mandatory and contains the exact 48-layer
set observed independently in all five preserved r33 baseline logs. SQG layers
6, 28, and 52 reserve their former count-budget slots even though SQG cannot
enter the MCG-only fused kernel; SQG layer 77 was already nonfused. The
allowlist prevents later MCG layers 55-57 from silently substituting for the
three treatment layers and changing the rest of the inference path.

The optional, experiment-specific
`VLLM_EXL3_NATIVE_MCG_CONTROL_LAYERS=6,28,52,77` arm accepts only an
all-MCG protected checkpoint. It forces exactly those four layers onto native
nonfused MCG dispatch and reserves the same three historical slots. Both the
SQG and native-MCG arms set
`VLLM_EXL3_R7_EXPECT_RESERVED_LAYERS=6,28,52`; the overlay-historical check
sets it to `none`. At the first layer above the historical fused range, the
loader fails unless actual-plus-reserved accounting is exactly 48.
