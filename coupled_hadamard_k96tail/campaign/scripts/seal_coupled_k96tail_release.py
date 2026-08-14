#!/usr/bin/env python3
"""Seal full-model KLD statistics and the coupled K96-tail reproduction bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


SCHEMA = "glm52-coupled-k96tail-full-acceptance-v1"
ROUTED_LAYERS = tuple(range(3, 79))


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not a map: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, Any]:
    if len(values) != 2047 or any(
        not math.isfinite(value) or value < -1.0e-6 for value in values
    ):
        raise ValueError("KLD vector is not the sealed 2,047-position finite regime")
    ordered = sorted(values)
    total = math.fsum(values)
    result: dict[str, Any] = {
        "positions": len(values),
        "mean": total / len(values),
        "median": quantile(ordered, 0.5),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "max": ordered[-1],
        "cvar_worst_1pct": math.fsum(ordered[-math.ceil(len(values) * 0.01):])
        / math.ceil(len(values) * 0.01),
        "diagnostic_removal_ladder": [],
    }
    ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
    for count in (10, 20, 30, 40, 96):
        removed = math.fsum(values[index] for index in ranked[:count])
        result["diagnostic_removal_ladder"].append(
            {
                "removed_positions": count,
                "remaining_mean": (total - removed) / (len(values) - count),
                "removed_share_of_total": removed / total,
                "position_indices": ranked[:count],
            }
        )
    return result


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree_filtered(source: Path, destination: Path, suffixes: tuple[str, ...]) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name.endswith(suffixes):
            copy_file(path, destination / path.relative_to(source))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer-root", type=Path, required=True)
    parser.add_argument("--codec-receipt", type=Path, required=True)
    parser.add_argument("--source-kld", type=Path, required=True)
    parser.add_argument("--candidate-kld", type=Path, required=True)
    parser.add_argument("--tail-analysis", type=Path, required=True)
    parser.add_argument("--mtp3-smoke", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--repro-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--allocation-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    assembly = load(model / "COUPLED_REENCODE_MANIFEST.json")
    codec = load(args.codec_receipt)
    source = load(args.source_kld)
    candidate = load(args.candidate_kld)
    tail = load(args.tail_analysis)
    mtp3_smoke = load(args.mtp3_smoke)
    if assembly.get("complete") is not True or assembly.get(
        "all_routed_layers_coupled"
    ) is not True:
        raise ValueError("assembled model is not a complete 3..78 coupled checkpoint")
    expected_average_bpw = (3.0625 + 75 * 3.125) / 76
    if (
        assembly.get("per_layer_bit_census", {}).get("3")
        != {"k3": 720, "k4": 48, "total": 768}
        or any(
            assembly.get("per_layer_bit_census", {}).get(str(layer))
            != {"k3": 672, "k4": 96, "total": 768}
            for layer in range(4, 79)
        )
        or not math.isclose(
            float(assembly.get("routed_bits_per_weight", -1.0)),
            expected_average_bpw,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or assembly.get("routed_layer_average_rate_is_uniform") is not False
    ):
        raise ValueError("assembled model does not have the hybrid K48/K96 rate contract")
    if codec.get("pass") is not True or codec.get("routed_layer_count") != 76:
        raise ValueError("codec/closure receipt did not pass all 76 routed layers")
    if candidate.get("complete") is not True:
        raise ValueError("candidate KLD receipt is incomplete")
    if (
        candidate.get("statistics", {}).get("nonfinite_count") != 0
        or candidate.get("statistics", {}).get("trim_fraction_per_side") != 0.0
        or tail.get("schema") != "glm52-kld-position-tail-concentration-v1"
        or tail.get("complete") is not True
        or tail.get("interpretation")
        != "diagnostic_only_no_positions_are_removed_from_the_acceptance_kld"
        or mtp3_smoke.get("schema") != "glm52-sqg-w4a8-sm120-smoke-v1"
        or mtp3_smoke.get("pass") is not True
        or mtp3_smoke.get("problems") != []
        or not isinstance(mtp3_smoke.get("tokens"), list)
        or len(mtp3_smoke["tokens"]) != 16
    ):
        raise ValueError("untrimmed KLD, tail, or MTP3 receipt differs")
    runtime = candidate.get("runtime", {})
    if (
        runtime.get("tensor_parallel_size") != 4
        or runtime.get("pipeline_parallel_size") != 1
        or runtime.get("decode_context_parallel_size") != 1
        or runtime.get("topology_attestation") != "tp4dcp1"
    ):
        raise ValueError("candidate KLD is not the required TP4/PP1/DCP1 regime")
    if candidate.get("reference", {}).get("manifest_sha256") != source.get(
        "reference", {}
    ).get("manifest_sha256"):
        raise ValueError("candidate and source KLD do not share the sealed BF16 reference")

    source_values = [float(value) for value in source["position_kld"]]
    candidate_values = [float(value) for value in candidate["position_kld"]]
    source_summary = summarize(source_values)
    candidate_summary = summarize(candidate_values)
    source_worst40 = sorted(
        range(len(source_values)), key=lambda index: (-source_values[index], index)
    )[:40]
    source_worst40_set = set(source_worst40)
    candidate_without_source_worst40 = [
        value for index, value in enumerate(candidate_values)
        if index not in source_worst40_set
    ]
    aligned = {
        "source_worst40_positions": source_worst40,
        "source_worst40_kld_sum": math.fsum(source_values[i] for i in source_worst40),
        "candidate_same40_kld_sum": math.fsum(candidate_values[i] for i in source_worst40),
        "candidate_mean_without_source_worst40": math.fsum(
            candidate_without_source_worst40
        ) / len(candidate_without_source_worst40),
    }
    metric_deltas = {
        key: candidate_summary[key] - source_summary[key]
        for key in ("mean", "median", "p95", "p99", "cvar_worst_1pct", "max")
    }
    quality_gates = {
        "full_untrimmed_mean_below_source": metric_deltas["mean"] < 0.0,
        "p99_below_source": metric_deltas["p99"] < 0.0,
        "cvar_worst_1pct_below_source": metric_deltas["cvar_worst_1pct"] < 0.0,
        "no_nonfinite_positions": True,
        "no_positions_trimmed": True,
    }

    oracles: dict[str, Any] = {}
    for layer in ROUTED_LAYERS:
        padded = f"{layer:03d}"
        manifest_path = args.layer_root / f"r7-experts-layer-{padded}.json"
        quality_path = args.layer_root / f"r7-experts-layer-{padded}.quality.json"
        oracle_path = args.layer_root / f"runtime-oracle-layer-{padded}.json"
        manifest = load(manifest_path)
        quality = load(quality_path)
        oracle = load(oracle_path)
        expected_census = (
            {"k3": 720, "k4": 48, "total": 768}
            if layer == 3
            else {"k3": 672, "k4": 96, "total": 768}
        )
        expected_bpw = 3.0625 if layer == 3 else 3.125
        if layer == 3:
            layer_contract_pass = (
                manifest.get("schema")
                == "glm52-coupled-selected-layer-runtime-v1"
                and quality.get("schema")
                == "glm52-coupled-layer-quality-tails-v1"
                and manifest.get("final_profile_binding") is None
            )
        else:
            profile = manifest.get("final_profile_binding")
            layer_contract_pass = (
                manifest.get("schema")
                == "glm52-coupled-selected-layer-runtime-v3"
                and quality.get("schema")
                == "glm52-coupled-layer-quality-tails-v2"
                and isinstance(profile, dict)
                and profile.get("no_b300_owner_speed_rescue") is True
                and quality.get("final_profile_binding") == profile
                and oracle.get("final_profile_binding") == profile
            )
        if (
            not layer_contract_pass
            or manifest.get("complete") is not True
            or manifest.get("bit_census") != expected_census
            or float(manifest.get("bits_per_weight", -1.0)) != expected_bpw
            or quality.get("complete") is not True
            or oracle.get("complete") is not True
            or oracle.get("pass") is not True
            or oracle.get("bit_census") != expected_census
            or float(oracle.get("bits_per_weight", -1.0)) != expected_bpw
        ):
            raise ValueError(f"hybrid no-shortcut layer seal failed: layer {layer}")
        oracles[str(layer)] = {
            "sha256": sha256(oracle_path),
            "relative_rmse": oracle.get("relative_rmse"),
            "mean_cosine": oracle.get("mean_cosine"),
            "manifest_id": manifest.get("manifest_id"),
            "quality_archive_id": quality.get("archive_id"),
            "final_profile_binding_id": (
                manifest.get("final_profile_binding") or {}
            ).get("binding_id"),
        }

    acceptance = {
        "schema": SCHEMA,
        "complete": True,
        "quality_pass": all(quality_gates.values()),
        "model": str(model),
        "assembly_manifest_id": assembly["manifest_id"],
        "source_checkpoint": "/home/brandonmusic/models/GLM-5.2-SQG-W4A8",
        "source_checkpoint_downloaded_as_bf16": False,
        "hessian_dataset": "brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians",
        "hessian_revision": "a05b3b92d749f6a641af5cfd52de2b4720380dfd",
        "image_ref": args.image_ref,
        "image_id": args.image_id,
        "actual_routed_bits_per_weight": assembly["routed_bits_per_weight"],
        "uniform_rate": False,
        "layer_3_census": assembly["per_layer_bit_census"]["3"],
        "layers_4_through_78_census": assembly["per_layer_bit_census"]["4"],
        "one_layer_kld_role": "directional_allocation_experiment_only",
        "acceptance_kld_role": "full_end_to_end_model_acceptance",
        "source_kld": {
            "path": str(args.source_kld.resolve()),
            "sha256": sha256(args.source_kld),
            **source_summary,
        },
        "candidate_kld": {
            "path": str(args.candidate_kld.resolve()),
            "sha256": sha256(args.candidate_kld),
            **candidate_summary,
        },
        "candidate_minus_source": metric_deltas,
        "quality_gates": quality_gates,
        "position_aligned_source_worst40": aligned,
        "codec_receipt_sha256": sha256(args.codec_receipt),
        "tail_analysis_sha256": sha256(args.tail_analysis),
        "mtp3_smoke_sha256": sha256(args.mtp3_smoke),
        "mtp3_smoke": {
            "regime": "mtp3_tp1_pp4_dcp1",
            "completion_text": mtp3_smoke.get("completion_text"),
            "finish_reason": mtp3_smoke.get("finish_reason"),
            "tokens": len(mtp3_smoke["tokens"]),
            "pass": True,
        },
        "no_b300_owner_speed_rescue": True,
        "runtime_oracles": oracles,
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    receipt_path = args.results_root / "FULL_ACCEPTANCE.json"
    if receipt_path.exists():
        if load(receipt_path) != acceptance:
            raise ValueError("existing full-acceptance receipt differs")
    else:
        atomic_json(receipt_path, acceptance)

    bundle = args.repro_root / "sealed-bundle"
    if bundle.exists():
        if not (bundle / "README.md").is_file() or not (
            bundle / "SHA256SUMS"
        ).is_file():
            raise ValueError("existing reproduction bundle is partial")
        print(json.dumps(acceptance, indent=2, sort_keys=True))
        return
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".k96tail-bundle-", dir=bundle.parent
    ) as temporary_name:
        stage = Path(temporary_name)
        copy_tree_filtered(args.project_root / "scripts", stage / "scripts", (".py", ".sh"))
        for name in ("compose.yaml", "serve.sh"):
            copy_file(args.acceptance_root / name, stage / "runtime" / name)
        copy_file(
            args.acceptance_root / "build-context" / "Dockerfile",
            stage / "runtime" / "build-context" / "Dockerfile",
        )
        copy_file(
            args.acceptance_root / "build-context" / "OVERLAY_SHA256SUMS",
            stage / "runtime" / "build-context" / "OVERLAY_SHA256SUMS",
        )
        copy_tree_filtered(
            args.acceptance_root / "build-context" / "overlay",
            stage / "runtime" / "build-context" / "overlay",
            (".py", ".md"),
        )
        copy_file(
            args.acceptance_root / "kld" / "run_kld_sm120.py",
            stage / "runtime" / "kld" / "run_kld_sm120.py",
        )
        copy_file(
            args.acceptance_root / "scripts" / "validate_model_codec.py",
            stage / "runtime" / "scripts" / "validate_model_codec.py",
        )
        for layer in ROUTED_LAYERS:
            padded = f"{layer:03d}"
            for suffix in (".json", ".quality.json"):
                source_path = args.layer_root / f"r7-experts-layer-{padded}{suffix}"
                copy_file(source_path, stage / "layers" / source_path.name)
            oracle_path = args.layer_root / f"runtime-oracle-layer-{padded}.json"
            copy_file(oracle_path, stage / "layers" / oracle_path.name)
        copy_tree_filtered(args.allocation_root, stage / "allocations", (".json",))
        copy_tree_filtered(args.profile_root, stage / "profiles", (".json",))
        copy_tree_filtered(args.score_root, stage / "scores", (".json",))
        copy_file(model / "COUPLED_REENCODE_MANIFEST.json", stage / "COUPLED_REENCODE_MANIFEST.json")
        copy_file(args.codec_receipt, stage / "model_codec_validation.json")
        copy_file(args.tail_analysis, stage / "kld_position_tail.json")
        copy_file(args.mtp3_smoke, stage / "mtp3_smoke_16tok.json")
        copy_file(receipt_path, stage / "FULL_ACCEPTANCE.json")

        readme = f"""# GLM-5.2 coupled H512/H128 K96-tail SQG reproduction\n\nThis bundle reproduces and validates the distinct local model at `{model}`.\nThe shipped source checkpoint was decoded in place; the original BF16 model was not downloaded.\n\n- Hessian dataset: `brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians`\n- Pinned revision: `a05b3b92d749f6a641af5cfd52de2b4720380dfd`\n- Routed layers: 3 through 78, including MTP layer 78\n- Layer 3: 720 K3 + 48 K4 tensors (3.0625 bpw)\n- Layers 4-78: 672 K3 + 96 K4 tensors (3.125 bpw)\n- Actual routed-layer average: {assembly['routed_bits_per_weight']:.12f} bpw\n- Coordinates: residual H512, preactivation H128, postactivation H128\n- Activation: exact `silu(gate)*up`; local H13 alpha 0.25\n- Execution: route-packed direct-E4M3 full W4A8, no A16 fallback\n- Runtime image: `{args.image_ref}` (`{args.image_id}`)\n\nThe KLD allocation signal came from the sealed source model's worst 40 TP4/DCP1 positions and exact routes. It is an in-sample allocation heuristic, not acceptance evidence. The acceptance result is the full assembled model's end-to-end TP4/PP1/DCP1 run against the sealed BF16 logits. No positions are removed from the reported mean.\n\nFull mean KLD: {candidate_summary['mean']:.12g}\nMedian: {candidate_summary['median']:.12g}; p95: {candidate_summary['p95']:.12g}; p99: {candidate_summary['p99']:.12g}; max: {candidate_summary['max']:.12g}.\n\nSee `FULL_ACCEPTANCE.json` for the complete distribution, diagnostic removal ladder, aligned source-worst-40 comparison, image identity, layer oracles, and hashes.\n"""
        (stage / "README.md").write_text(readme, encoding="utf-8")
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        with (stage / "SHA256SUMS").open("w", encoding="utf-8") as handle:
            for path in files:
                if path.name != "SHA256SUMS":
                    handle.write(f"{sha256(path)}  {path.relative_to(stage)}\n")
        os.rename(stage, bundle)
    print(json.dumps(acceptance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
