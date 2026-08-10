"""Fresh, zero-MCG SQG treatment bridge for GLM 5.2."""

from .codec import (
    DenseHessian,
    DenseHSession,
    EncodedSQGMatrix,
    KQuantRuntime,
    SharedResidualProfile,
    UniformSQGConfig,
    encode_uniform_sqg,
    load_kquant_runtime,
    prepare_dense_h_session,
    resume_dense_h_session,
    validate_gate_up_encode_smoke,
)
from .manifest import (
    BF16TensorBinding,
    FORBIDDEN_MCG_READS,
    FrozenBitBudget,
    SQG_MARKER,
    SyntheticTensorBinding,
    build_run_manifest,
    validate_tensor_manifest,
)
from .permutation import (
    FreshExpertPermutation,
    apply_glm_expert_permutation,
    derive_glm_h2_reverse_permutation,
    draw_fresh_expert_permutation,
    glm_expert_forward,
)

__all__ = [
    "BF16TensorBinding",
    "DenseHessian",
    "DenseHSession",
    "EncodedSQGMatrix",
    "FORBIDDEN_MCG_READS",
    "FreshExpertPermutation",
    "FrozenBitBudget",
    "KQuantRuntime",
    "SQG_MARKER",
    "SharedResidualProfile",
    "SyntheticTensorBinding",
    "UniformSQGConfig",
    "apply_glm_expert_permutation",
    "build_run_manifest",
    "derive_glm_h2_reverse_permutation",
    "draw_fresh_expert_permutation",
    "encode_uniform_sqg",
    "glm_expert_forward",
    "load_kquant_runtime",
    "prepare_dense_h_session",
    "resume_dense_h_session",
    "validate_gate_up_encode_smoke",
    "validate_tensor_manifest",
]
