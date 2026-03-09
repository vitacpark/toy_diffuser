from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src_path():
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    src_root_s = str(src_root)
    if src_root_s not in sys.path:
        sys.path.insert(0, src_root_s)

