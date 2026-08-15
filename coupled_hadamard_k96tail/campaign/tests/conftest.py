from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "kquant"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
