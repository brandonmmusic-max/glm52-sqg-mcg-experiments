"""Stream routed expert sources from an existing GLM SQG checkpoint.

The shipped routed checkpoint is already an approximate, function-preserving
source for a second SQG encode.  Its individual gate/up/down matrices live in
an expert-private intermediate basis, so comparing those matrices separately
with the originally named BF16 tensors is not a valid reconstruction test.
The invariant is the complete expert function.

This provider reconstructs the three dense effective matrices by identity
probing the native B12X trellis entry point.  It keeps that shared internal
basis intact, returns one BF16 expert triplet at a time, and records exact
decoded tensor identities for the coupled re-encode lineage.  No original
BF16 routed shard is read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from src.glm52_bf16_source import ExpertWeights
from src.glm52_fresh_sqg.reference import tensor_sha256


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
HIDDEN = 6144
INTERMEDIATE = 2048
NUM_EXPERTS = 256
ROUTED_LAYERS = tuple(range(3, 79))


def _sha256_file(path: Path, chunk_bytes: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SQGCheckpointValidation:
    model_root: Path
    index_sha256: str
    quantization_config_sha256: str
    manifest_sha256: str
    weight_map: Mapping[str, str]

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": "existing_sqg_function_preserving_source",
            "model_root": str(self.model_root),
            "index_sha256": self.index_sha256,
            "quantization_config_sha256": self.quantization_config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "original_bf16_routed_shards_read": False,
            "reconstruction": "native_b12x_dense_identity_probe",
            "coordinate_contract": "expert_private_function_preserving_basis",
        }


def decode_effective_matrix(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    device: str | torch.device,
    probe_rows: int = 512,
) -> torch.Tensor:
    """Decode one stored projection to effective EXL ``[input, output]``."""

    from b12x.gemm.trellis_linear import api

    if probe_rows <= 0:
        raise ValueError("probe_rows must be positive")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("native SQG identity probing requires CUDA")
    prepared = api.prepare_weight(
        trellis.to(target),
        suh.to(target),
        svh.to(target),
        codebook="sqg_xor_cheb_t12",
        params_dtype=torch.float16,
    )
    in_features = int(trellis.shape[0]) * 16
    out_features = int(svh.numel())
    output = torch.empty(
        (in_features, out_features), dtype=torch.bfloat16, device="cpu"
    )
    for begin in range(0, in_features, probe_rows):
        end = min(in_features, begin + probe_rows)
        rows = end - begin
        identity = torch.zeros(
            (rows, in_features), dtype=torch.float16, device=target
        )
        local = torch.arange(rows, device=target)
        identity[local, local + begin] = 1.0
        decoded = api.run(identity, prepared)[:, :out_features]
        output[begin:end].copy_(decoded.to(torch.bfloat16), non_blocking=False)
        del identity, decoded, local
    del prepared
    return output.contiguous()


class SQGCheckpointExpertSource:
    """Validated, bounded-memory source over routed SQG expert payloads."""

    def __init__(
        self,
        model_root: str | Path,
        *,
        probe_rows: int = 512,
    ) -> None:
        root = Path(model_root).resolve()
        index_path = root / "model.safetensors.index.json"
        quant_path = root / "quantization_config.json"
        manifest_path = root / "FULL_SQG_NATIVE_MANIFEST.json"
        for path in (index_path, quant_path, manifest_path):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("SQG checkpoint weight map is absent")
        if probe_rows <= 0:
            raise ValueError("probe_rows must be positive")
        self.probe_rows = int(probe_rows)
        self.validation = SQGCheckpointValidation(
            model_root=root,
            index_sha256=_sha256_file(index_path),
            quantization_config_sha256=_sha256_file(quant_path),
            manifest_sha256=_sha256_file(manifest_path),
            weight_map=dict(weight_map),
        )

    @property
    def model_root(self) -> Path:
        return self.validation.model_root

    def _tensor_names(self, layer: int, expert: int) -> dict[str, str]:
        base = f"model.layers.{layer}.mlp.experts"
        names = {
            "gate_trellis": f"{base}.{expert}.gate_proj.trellis",
            "up_trellis": f"{base}.{expert}.up_proj.trellis",
            "down_trellis": f"{base}.{expert}.down_proj.trellis",
            "gate_up_suh": f"{base}.r7_shared.gate_up_suh",
            "gate_svh": f"{base}.{expert}.gate_proj.svh",
            "up_svh": f"{base}.{expert}.up_proj.svh",
            "down_suh": f"{base}.{expert}.down_proj.suh",
            "down_svh": f"{base}.r7_shared.down_svh",
        }
        missing = [name for name in names.values() if name not in self.validation.weight_map]
        if missing:
            raise KeyError(f"SQG checkpoint lacks routed tensors: {missing}")
        return names

    def load_expert_bf16(
        self,
        layer: int,
        expert: int,
        *,
        device: str | torch.device,
    ) -> ExpertWeights:
        """Return one decoded BF16 triplet in its shared private basis."""

        if layer not in ROUTED_LAYERS:
            raise ValueError("layer is outside the GLM routed range 3..78")
        if not 0 <= expert < NUM_EXPERTS:
            raise ValueError("expert must lie in [0,255]")
        names = self._tensor_names(layer, expert)
        shards = {self.validation.weight_map[name] for name in names.values()}
        if len(shards) != 1:
            raise ValueError(
                f"layer {layer} routed tensors unexpectedly span shards: {sorted(shards)}"
            )
        shard_name = next(iter(shards))
        shard_path = self.model_root / shard_name
        if not shard_path.is_file() or shard_path.is_symlink():
            raise FileNotFoundError(shard_path)
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            values = {key: handle.get_tensor(name) for key, name in names.items()}
        gate_exl = decode_effective_matrix(
            values["gate_trellis"],
            values["gate_up_suh"],
            values["gate_svh"],
            device=device,
            probe_rows=self.probe_rows,
        )
        up_exl = decode_effective_matrix(
            values["up_trellis"],
            values["gate_up_suh"],
            values["up_svh"],
            device=device,
            probe_rows=self.probe_rows,
        )
        down_exl = decode_effective_matrix(
            values["down_trellis"],
            values["down_suh"],
            values["down_svh"],
            device=device,
            probe_rows=self.probe_rows,
        )
        gate_hf = gate_exl.T.contiguous()
        up_hf = up_exl.T.contiguous()
        down_hf = down_exl.T.contiguous()
        expected = {
            "gate_proj": (INTERMEDIATE, HIDDEN),
            "up_proj": (INTERMEDIATE, HIDDEN),
            "down_proj": (HIDDEN, INTERMEDIATE),
        }
        tensors = {
            "gate_proj": gate_hf,
            "up_proj": up_hf,
            "down_proj": down_hf,
        }
        for projection, value in tensors.items():
            if value.dtype != torch.bfloat16 or tuple(value.shape) != expected[projection]:
                raise RuntimeError(
                    f"decoded {projection} differs: {value.dtype} {tuple(value.shape)}"
                )
        hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
        del values, gate_exl, up_exl, down_exl
        return ExpertWeights(
            layer=layer,
            expert=expert,
            gate_hf=gate_hf,
            up_hf=up_hf,
            down_hf=down_hf,
            tensor_sha256=hashes,
            shard_names={projection: shard_name for projection in PROJECTIONS},
        )


def open_sqg_transcode_runtime(
    preflight_path: str | Path,
    *,
    model_root: str | Path,
    layer: int,
    device: str,
    bit_map: Mapping[str, int],
    probe_rows: int = 512,
):
    """Open saved calibration state with the frozen SQG model as its source.

    The predecessor preflight remains the authority for capture, Hessian,
    permutation, profile, and encoder inputs.  Its BF16 source binding is not
    reused or rewritten: this is a new, explicitly lossy transcode lineage,
    and :class:`SQGCheckpointExpertSource` supplies the replacement source.
    """

    from src.fresh_pipeline_calibration import LayerCaptureView
    from src.fresh_pipeline_common import load_json_object
    from src.fresh_pipeline_runner import LayerRuntime, PipelinePaths, PipelineSettings

    if not 3 <= int(layer) <= 78:
        raise ValueError("GLM routed layer must lie in 3..78")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("SQG transcode runtime requires CUDA")
    preflight_file = Path(preflight_path).resolve()
    preflight = load_json_object(preflight_file)
    paths = PipelinePaths.resolve(**preflight["paths"])
    settings = PipelineSettings(**preflight["settings"])
    source = SQGCheckpointExpertSource(model_root, probe_rows=probe_rows)
    expected = {
        f"model.layers.{int(layer)}.mlp.experts.{expert}.{projection}"
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    supplied = {str(key): int(value) for key, value in bit_map.items()}
    if set(supplied) != expected:
        missing = sorted(expected - set(supplied))
        extra = sorted(set(supplied) - expected)
        raise ValueError(
            "SQG transcode bit map differs from the exact routed layer domain: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    if any(bits not in (3, 4) for bits in supplied.values()):
        raise ValueError("SQG transcode bit map permits only K3/K4")
    return LayerRuntime(
        paths=paths,
        settings=settings,
        layer=int(layer),
        device=str(target),
        source=source,  # type: ignore[arg-type]
        capture=LayerCaptureView(
            paths.capture_dir,
            int(layer),
            validate=False,
            verify_hashes=False,
        ),
        bit_map=supplied,
        source_seal=source.validation.manifest(),
        preflight=preflight,
    )


__all__ = [
    "SQGCheckpointExpertSource",
    "SQGCheckpointValidation",
    "decode_effective_matrix",
    "open_sqg_transcode_runtime",
]
