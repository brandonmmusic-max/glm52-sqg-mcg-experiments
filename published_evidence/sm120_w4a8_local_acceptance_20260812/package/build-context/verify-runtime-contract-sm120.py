#!/usr/bin/env python3
"""Static, GPU-free release checks for the GLM SM120 full-W4A8 overlay.

Adapted from the sealed release bundle's verify-runtime-contract.py for the
SM120 local acceptance image: the vLLM integration is the newer r7/K6-aware
exl3.py (B300 lineage) with the explicit SM120 topology attestation, and the
non-routed K6 W6A16 endpoint must be present in b12x.gemm.trellis_linear.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

# The frozen SM120 PR11 route-packed kernel. This file must remain
# byte-identical to the accepted measurement.
KERNEL_SHA256 = (
    "b95dde9347cf679836eedc1c76f0f569"
    "a4fb37dcc594a674e41666e4d0f162ad"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_tokens(path: Path, tokens: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [token for token in tokens if token not in text]
    if missing:
        raise SystemExit(f"{path}: missing release contract tokens: {missing}")


def forbid_tokens(path: Path, tokens: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    present = [token for token in tokens if token in text]
    if present:
        raise SystemExit(f"{path}: forbidden tokens present: {present}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--b12x-root", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    args = parser.parse_args()

    kernel = args.b12x_root / "moe/_shared/kernels/glm_trellis_w4a8.py"
    runtime = args.b12x_root / "moe/glm_sqg_w4a8.py"
    exl3 = args.vllm_root / "model_executor/layers/quantization/exl3.py"
    api = args.b12x_root / "gemm/trellis_linear/api.py"
    small_m = args.b12x_root / "gemm/trellis_linear/_small_m.py"
    if digest(kernel) != KERNEL_SHA256:
        raise SystemExit(
            f"PR11 kernel mismatch: expected {KERNEL_SHA256}, got {digest(kernel)}"
        )

    require_tokens(
        runtime,
        (
            '"kernel": "m128n64"',
            '"blocks_per_cta": 8,',
            '"stages": 2,',
            "validate_glm_route_packed_w4a8_acceptance_kernel()",
            "quantize_mxfp8_rows_cute(",
            "run_glm_gate_up_output_transform_silu(",
            "run_glm_down_input_transform(",
            "direct_e4m3_weights: bool = True",
            "allow_a16_fallback: bool = False",
            "(12, 0)",
        ),
    )
    require_tokens(
        exl3,
        (
            '"full_w4a8"',
            '"rates": "independent_per_tensor_k3_k4"',
            '"direct_e4m3_weights": True',
            '"allow_a16_fallback": False',
            '"activation": "silu_gate_times_up"',
            '"topology": "topology_neutral"',
            "_apply_glm_sqg_w4a8(",
            "run_sqg_k6_w6a16",
            "_validate_sqg_k6_nonrouted_contract",
            "_validate_r7_routed_experts_contract",
            "_attested_glm_sqg_regime",
            '"tp4dcp4": (4, 1, 4)',
            "_record_glm_sqg_w4a8_evidence",
        ),
    )
    require_tokens(
        api,
        (
            "def run_sqg_k6_w6a16(",
            "native SQG K6 W6A16 forbids MCG marker lineage",
        ),
    )
    require_tokens(
        small_m,
        ("_reject_non_mcg_codebook", "refusing to route SQG"),
    )
    runtime_text = runtime.read_text(encoding="utf-8")
    if "glm_trellis_w4a16" in runtime_text or "B12X_MOE_FORCE_A16" in runtime_text:
        raise SystemExit("dedicated GLM full-W4A8 runtime contains an A16 route")
    if runtime_text.count("quantize_mxfp8_rows_cute(") < 2:
        raise SystemExit("full-W4A8 runtime must quantize both h and act")

    manifest = args.overlay_root / "OVERLAY_SHA256SUMS"
    if not manifest.is_file():
        raise SystemExit(f"missing overlay manifest: {manifest}")
    sealed_paths: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        sealed_paths.add(relative)
        path = args.overlay_root / relative
        if digest(path) != expected:
            raise SystemExit(f"overlay manifest mismatch: {relative}")
    actual_paths = {
        str(path.relative_to(args.overlay_root))
        for path in (args.overlay_root / "overlay").rglob("*")
        if path.is_file()
    }
    if actual_paths != sealed_paths:
        raise SystemExit(
            "overlay file census mismatch: "
            f"missing={sorted(sealed_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - sealed_paths)}"
        )

    for relative in (
        "gemm/trellis_linear/__init__.py",
        "gemm/trellis_linear/api.py",
        "gemm/trellis_linear/_small_m.py",
        "moe/__init__.py",
        "moe/_shared/kernels/w4a16/kernel.py",
        "moe/_shared/kernels/glm_trellis_transform.py",
        "moe/_shared/kernels/glm_trellis_w4a8.py",
        "moe/glm_sqg_w4a8.py",
        "_lib/intrinsics.py",
        "_lib/quant/sqg_e4m3.py",
    ):
        path = args.b12x_root / relative
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    compile(exl3.read_text(encoding="utf-8"), str(exl3), "exec")
    print("GLM native full-W4A8 SM120 static release contract: PASS")


if __name__ == "__main__":
    main()
