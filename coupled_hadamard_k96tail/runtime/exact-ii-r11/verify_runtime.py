"""Fail-closed static gate for the II r11-derived K96 image."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path


def main() -> None:
    r11_package = Path("/opt/infernal-invocation/vllm/vllm")
    if r11_package.is_dir():
        required_r11_artifacts = (
            r11_package / "_version.py",
            r11_package / "_C_stable_libtorch.abi3.so",
            r11_package / "fs_io_C.abi3.so",
        )
        missing = [str(path) for path in required_r11_artifacts if not path.is_file()]
        if missing:
            raise RuntimeError(
                "r11 source overlay removed compiled/generated artifacts: "
                + ", ".join(missing)
            )

    exl3 = importlib.import_module(
        "vllm.model_executor.layers.quantization.exl3"
    )
    deepseek_mtp = importlib.import_module(
        "vllm.model_executor.models.deepseek_mtp"
    )
    runtime = importlib.import_module("b12x.moe.glm_sqg_w4a8")
    transforms = importlib.import_module(
        "b12x.moe._shared.kernels.glm_trellis_transform"
    )
    common = importlib.import_module("vllm.v1.attention.ops.common")
    capturer = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer"
    )

    required = (
        (exl3.Exl3Config, "glm_sqg_layer_is_coupled"),
        (runtime, "prepare_weights"),
        (transforms, "run_glm_coupled_residual_h512"),
        (transforms, "run_glm_coupled_gate_up_output_transform_silu"),
        (capturer, "get_routed_experts_attn_gid"),
    )
    for owner, name in required:
        if not hasattr(owner, name):
            raise RuntimeError(f"II r11 K96 runtime is missing {owner}.{name}")

    prepare_signature = inspect.signature(runtime.prepare_weights)
    if "coupled_rotation_draws" not in prepare_signature.parameters:
        raise RuntimeError("B12X K96 draw-map ABI is absent")
    coupled_signature = inspect.signature(
        transforms.run_glm_coupled_gate_up_output_transform_silu
    )
    if not {"tp_rank", "tp_size"}.issubset(coupled_signature.parameters):
        raise RuntimeError("B12X coupled TP preactivation ABI is absent")
    coupled_source = inspect.getsource(
        transforms.run_glm_coupled_gate_up_output_transform_silu
    )
    required_coupled_fragments = (
        "get_tp_group().all_gather(pre_scaled, dim=1)",
        ".permute(0, 2, 1, 3)",
        "start = 2 * tp_rank * width",
    )
    if any(fragment not in coupled_source for fragment in required_coupled_fragments):
        raise RuntimeError("B12X coupled TP4 preactivation reassembly is absent")
    runtime_source = inspect.getsource(runtime.prepare_weights)
    required_rank_sign_fragments = (
        "2 * global_intermediate_size",
        "pre_start = 2 * tp_rank * intermediate_size",
        "post_start = tp_rank * intermediate_size",
    )
    if any(fragment not in runtime_source for fragment in required_rank_sign_fragments):
        raise RuntimeError("B12X coupled TP rank-offset signs are absent")
    common_source = inspect.getsource(common.mask_dcp_empty_shards_)
    if "if seq_lens.shape[0] > 0" not in common_source:
        raise RuntimeError("DCP zero-sequence merge guard is absent")
    mtp_storage = {
        "model.layers.78.mlp.shared_experts.down_proj": {"quant_format": "exl3"}
    }
    mtp_config = exl3.Exl3Config(tensor_storage=mtp_storage)
    mtp_runtime_prefix = "model.layers.78.mtp_block.mlp.shared_experts.down_proj"
    if mtp_config._storage_entry(mtp_runtime_prefix) is None:
        raise RuntimeError("MTP runtime-prefix EXL3 storage resolution is absent")
    mtp_packed_storage = {
        "model.layers.78.mlp.shared_experts.gate_proj": {
            "quant_format": "exl3"
        },
        "model.layers.78.mlp.shared_experts.up_proj": {
            "quant_format": "exl3"
        },
    }
    mtp_packed_config = deepseek_mtp._configure_deepseek_mtp_quant_config(
        exl3.Exl3Config(tensor_storage=mtp_packed_storage)
    )
    mtp_packed_prefix = (
        "model.layers.78.mtp_block.mlp.shared_experts.gate_up_proj"
    )
    if not mtp_packed_config._linear_prefix_is_exl3(mtp_packed_prefix):
        raise RuntimeError("MTP packed gate/up EXL3 quantization is absent")
    schedule = runtime.glm_route_packed_w4a8_kernel_contract()
    expected = {
        "kernel": "m128n64",
        "blocks_per_cta": 8,
        "stages": 2,
        "route_block_rows": 128,
        "tile_n": 64,
    }
    if schedule != expected:
        raise RuntimeError(f"unexpected GLM SQG schedule: {schedule!r}")
    print("II r11 K96 static runtime gate: PASS", schedule)


if __name__ == "__main__":
    main()
