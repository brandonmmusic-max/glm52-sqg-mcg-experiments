from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATCHED = ROOT / "patched_sources"
BASE = ROOT / "base_sources"
API = PATCHED / "b12x/gemm/trellis_linear/api.py"
INIT = PATCHED / "b12x/gemm/trellis_linear/__init__.py"
SMALL_M = PATCHED / "b12x/gemm/trellis_linear/_small_m.py"
INTRINSICS = PATCHED / "b12x/_lib/intrinsics.py"
W4A16 = PATCHED / "b12x/moe/_shared/kernels/w4a16/kernel.py"
VLLM = PATCHED / "vllm/model_executor/layers/quantization/exl3.py"
AUTHORITATIVE_VLLM_SHA256 = (
    "d6bdda804e3491d9d775029f71d1f1fa75da97e4e4c4f20eeafc9f8743f6561c"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text())
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), node],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_patch_is_rebased_on_authoritative_pp8_vllm() -> None:
    assert _sha256(BASE / "vllm/model_executor/layers/quantization/exl3.py") == (
        AUTHORITATIVE_VLLM_SHA256
    )


def test_dedicated_api_admits_only_native_sqg_k6_and_is_topology_neutral() -> None:
    calls = []

    def run_trellis256_dense(x, weight, **kwargs):
        calls.append((x, weight, kwargs))
        return "native-w6a16"

    run = _function(
        API,
        "run_sqg_k6_w6a16",
        {
            "torch": torch,
            "Optional": Optional,
            "PreparedWeight": object,
            "run_trellis256_dense": run_trellis256_dense,
        },
    )
    x = object()
    weight = SimpleNamespace(
        trellis_codebook="sqg_xor_cheb_t12",
        trellis_bits=6,
        trellis_pair_kind=None,
        mcg=None,
        mul1_e4m3=None,
    )
    assert run(x, weight) == "native-w6a16"
    assert calls[0][0] is x
    assert calls[0][1] is weight

    invalid = (
        {"trellis_codebook": "mcg"},
        {"trellis_bits": 5},
        {"trellis_pair_kind": "P33"},
        {"mcg": object()},
        {"mul1_e4m3": object()},
    )
    for change in invalid:
        bad = SimpleNamespace(**(vars(weight) | change))
        with pytest.raises(ValueError):
            run(x, bad)
    assert len(calls) == 1


def test_vllm_k6_sqg_marker_dispatches_only_to_dedicated_endpoint() -> None:
    source = VLLM.read_text()
    tree = ast.parse(source)
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_exl3_sqg_gemm"
    )
    body = ast.get_source_segment(source, node)
    assert body is not None
    assert "bits == 6" in body
    assert "api.run_sqg_k6_w6a16(x, weight)" in body
    assert "codebook=\"sqg_xor_cheb_t12\"" in body
    assert "_exl3_gemm" not in body
    assert "run_k6_mcg" not in body
    assert "api.run_w4a8" not in body
    assert "_SQG_SENTINEL = 0x53514731" in source
    assert '"sqg" if expected == "sqg_xor_cheb_t12"' in source
    assert "present != [expected_marker]" in source


def test_model_endpoint_contract_is_fail_closed() -> None:
    source = VLLM.read_text()
    tree = ast.parse(source)
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Exl3Config")
    method = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_validate_sqg_k6_nonrouted_contract"
    )
    holder = ast.ClassDef(
        name="Holder",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(body=[holder], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(VLLM), "exec"), namespace)
    valid = {
        "bits": 6,
        "codebook": "sqg_xor_cheb_t12",
        "execution": "native_sqg_k6_w6a16",
        "activation_endpoint": "a16",
    }
    instance = namespace["Holder"]()
    instance.sqg_k6_nonrouted = valid
    instance._validate_sqg_k6_nonrouted_contract()
    for field, bad in (
        ("bits", 4),
        ("codebook", "mcg"),
        ("execution", "full_w4a8"),
        ("activation_endpoint", "a8"),
        ("allow_mcg_fallback", True),
        ("allow_bf16_weight_fallback", True),
    ):
        instance.sqg_k6_nonrouted = valid | {field: bad}
        with pytest.raises(ValueError):
            instance._validate_sqg_k6_nonrouted_contract()


def test_k6_contiguous_stream_direct_e4m3_contract() -> None:
    intrinsics = INTRINSICS.read_text()
    kernel = W4A16.read_text()
    assert "bits not in (2, 3, 4, 5, 6)" in intrinsics
    assert "K6 SQG-XOR-Cheb-T12 requires the contiguous 64-bit stream layout" in intrinsics
    assert "shf.r.wrap.b32 w{slot}, $2, $3" in intrinsics
    assert "if width > 11:" in intrinsics
    assert "stream_layout=int(bits) == 6" in kernel
    sqg_arm = kernel.split('self.trellis_codebook == "mcg"', 1)[1]
    assert "packed_decode_sqg_xor_cheb_t12_to_e4m3x8" in sqg_arm
    assert "fp8x4_e4m3_to_half2x2" in sqg_arm


def test_cpu_k6_pack_state_and_label_closure() -> None:
    reference_path = Path(
        "/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_test/"
        "src/glm52_fresh_sqg/reference.py"
    )
    sqg_path = Path(
        "/home/brandonmusic/KLC_SANDBOXES/glm52-sqg-runtime-20260809/"
        "b12x/b12x/_lib/quant/sqg_e4m3.py"
    )
    ref_spec = importlib.util.spec_from_file_location("sqg_reference", reference_path)
    sqg_spec = importlib.util.spec_from_file_location("sqg_e4m3", sqg_path)
    assert ref_spec and ref_spec.loader and sqg_spec and sqg_spec.loader
    ref = importlib.util.module_from_spec(ref_spec)
    sqg = importlib.util.module_from_spec(sqg_spec)
    ref_spec.loader.exec_module(ref)
    sqg_spec.loader.exec_module(sqg)

    generator = torch.Generator().manual_seed(0x53514731)
    edges = torch.randint(0, 64, (2, 3, 256), dtype=torch.int16, generator=generator)
    packed = ref.pack_trellis_states(edges, 6)
    assert packed.dtype == torch.int16
    assert tuple(packed.shape) == (2, 3, 96)
    assert torch.equal(ref.unpack_trellis_edges(packed, 6), edges)

    states = ref.reconstruct_trellis_states(edges, 6).to(torch.int64) & 0xFFFF
    direct = sqg.sqg_xor_cheb_t12_direct_lut_cpu().reshape(5, 1 << 16)[4]
    t12 = sqg.sqg_xor_cheb_t12_lut_cpu()
    ranks = sqg._sqg_xor_cheb_t12_rank_for_codewords(states, 6)
    assert torch.equal(direct[states], t12[ranks >> 4])


def test_mcg_small_m_entrypoint_rejects_sqg_before_cuda() -> None:
    namespace = {
        "torch": torch,
        "_SQG_MARKER_SQG1": 0x53514731,
        "_MCG_MARKER": 0xCBAC1FED,
        "_MCG_CODEBOOK_NAMES": frozenset(
            {"mcg", "exl3_trellis_mcg", "w4a16/exl3_trellis_mcg"}
        ),
        "_SQG_CODEBOOK_NAMES": frozenset(
            {
                "sqg",
                "sqg_xor_cheb_t12",
                "exl3_trellis_sqg_cheb_e4m3",
                "w4a16/exl3_trellis_sqg_cheb_e4m3",
                "w4a8/exl3_trellis_sqg_cheb_e4m3",
            }
        ),
    }
    reject = _function(SMALL_M, "_reject_non_mcg_codebook", namespace)
    payload = torch.zeros(1, dtype=torch.int16)
    for marker in ("sqg", "sqg_xor_cheb_t12", 0x53514731, None):
        with pytest.raises(ValueError):
            reject(marker, payload)
    reject("mcg", payload)
    reject(0xCBAC1FED, payload)


def test_public_api_and_sources_name_the_w6a16_endpoint() -> None:
    assert '"run_sqg_k6_w6a16"' in INIT.read_text()
    assert "def run_sqg_k6_w6a16(" in API.read_text()
    assert "run_trellis256_dense(" in API.read_text()
    assert "run_trellis256_dense_w4a8" not in ast.get_source_segment(
        API.read_text(),
        next(
            node
            for node in ast.parse(API.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_sqg_k6_w6a16"
        ),
    )
