"""Reproducibility manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_manifest(path: Path, values: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
    return path
