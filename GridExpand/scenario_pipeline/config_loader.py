"""Load, validate, and fingerprint scenario/run YAML files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .run_config import RunConfig
from .scenario_config import ScenarioConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    return raw


def configuration_hash(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scenario_identity_key(scenario_id: str, scenario_hash: str) -> str:
    """Return the stable scenario identity used across pipeline stages."""
    return f"scenario_{scenario_id}_{str(scenario_hash)[:12]}"


def load_scenario_config(path: str | Path) -> tuple[ScenarioConfig, str]:
    raw = _read_yaml(Path(path))
    return ScenarioConfig.from_dict(raw), configuration_hash(raw)


def load_run_config(path: str | Path) -> tuple[RunConfig, str]:
    resolved = Path(path).resolve()
    raw = _read_yaml(resolved)
    return RunConfig.from_dict(raw, base_dir=resolved.parent), configuration_hash(raw)
