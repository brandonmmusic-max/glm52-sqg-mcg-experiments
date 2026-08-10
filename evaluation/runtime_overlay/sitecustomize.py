"""Install the exact r33 boot files plus the isolated SQG EXL3 override."""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys


_ROOT = os.path.join(os.path.dirname(__file__), "_overrides")
_SOURCES = {
    "vllm.model_executor.layers.quantization.exl3": os.path.join(
        _ROOT,
        "vllm",
        "model_executor",
        "layers",
        "quantization",
        "exl3.py",
    ),
    "vllm.model_executor.model_loader.utils": os.path.join(
        _ROOT, "vllm", "model_executor", "model_loader", "utils.py"
    ),
    "vllm.envs": os.path.join(_ROOT, "vllm", "envs.py"),
    "vllm.model_executor.models.deepseek_v2": os.path.join(
        _ROOT, "vllm", "model_executor", "models", "deepseek_v2.py"
    ),
    "b12x.moe._shared.kernels.w4a16.mixed_trellis": os.path.join(
        _ROOT,
        "b12x",
        "moe",
        "_shared",
        "kernels",
        "w4a16",
        "mixed_trellis.py",
    ),
}


class _SqgR33Override(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        del path, target
        source = _SOURCES.get(fullname)
        if source is None:
            return None
        return importlib.util.spec_from_file_location(fullname, source)


_missing = [name for name, source in _SOURCES.items() if not os.path.isfile(source)]
if _missing:
    raise RuntimeError(f"incomplete SQG r33 runtime overlay: {_missing}")
sys.meta_path.insert(0, _SqgR33Override())
