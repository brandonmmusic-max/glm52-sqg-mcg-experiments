from __future__ import annotations

import hashlib
import json

from scripts.build_layerwise_h13_artifact import build


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(root, label):
    layers = {}
    for layer in (6, 28, 52, 77):
        layer_root = root / f"layer_{layer:03d}"
        final = layer_root / "final"
        profile = layer_root / "profile_search"
        final.mkdir(parents=True)
        profile.mkdir(parents=True)
        shard = final / f"layer-{layer}.safetensors"
        shard.write_bytes(f"{label}-{layer}".encode())
        shard_hash = _sha(shard)
        manifest = final / f"fresh-sqg-layer-{layer:03d}.json"
        manifest.write_text(
            json.dumps({"shard": shard.name, "shard_sha256": shard_hash}) + "\n"
        )
        manifest.with_suffix(".json.sha256").write_text(_sha(manifest) + "\n")
        selection = profile / "selection.json"
        selection.write_text(json.dumps({"selection_id": "selection"}) + "\n")
        layers[str(layer)] = {
            "K3": 384,
            "K4": 384,
            "layer_artifact": str(manifest),
            "layer_artifact_sha256": _sha(manifest),
            "layer_shard_sha256": shard_hash,
            "mcg_tensors": 0,
            "selected_cell_id": "cell",
            "selection_id": "selection",
            "selection_sha256": _sha(selection),
            "sqg_tensors": 768,
        }
    seal = {
        "run_id": "run",
        "run_seal_id": f"seal-{label}",
        "preflight_sha256": "preflight",
        "layers": layers,
        "lineage": {"fallback_count": 0},
    }
    (root / "run_seal.json").write_text(json.dumps(seal) + "\n")


def test_build_layerwise_artifact_hardlinks_selected_shards(tmp_path):
    roots = {label: tmp_path / label for label in ("alpha0", "alpha1")}
    for label, root in roots.items():
        root.mkdir()
        _candidate(root, label)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "analysis_type": "exploratory_layerwise_refinement",
                "source_role": "selection",
                "winner": {
                    "layer_to_label": {
                        "6": "alpha0",
                        "28": "alpha1",
                        "52": "alpha0",
                        "77": "alpha1",
                    }
                },
            }
        )
        + "\n"
    )
    output = tmp_path / "output"

    seal = build(selection_path=selection, candidates=roots, output_root=output)

    assert seal["lineage"]["requested_local_alpha_by_layer"]["28"] == "alpha1"
    assert seal["census"]["sqg_tensors"] == 3072
    source = roots["alpha1"] / "layer_028/final/layer-28.safetensors"
    linked = output / "layer_028/final/layer-28.safetensors"
    assert source.stat().st_ino == linked.stat().st_ino
    assert (output / "logs").is_dir()
