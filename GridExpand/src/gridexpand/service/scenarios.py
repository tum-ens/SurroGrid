"""Scenario YAML files offered by the service."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gridexpand.scenario.config_loader import (
    load_scenario_config,
    scenario_identity_key,
)

_NAME_RE = re.compile(r"^[\w.-]+\.ya?ml$")


def scenario_files(dirs: Iterable[Path]) -> dict[str, Path]:
    """``file name -> path`` of all scenario YAMLs; the first directory wins on equal names."""
    files: dict[str, Path] = {}
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and _NAME_RE.match(path.name) and path.name not in files:
                files[path.name] = path.resolve()
    return files


def describe(name: str, path: Path) -> dict[str, Any]:
    """Summary of one scenario file; invalid files are listed with ``valid: false``."""
    info: dict[str, Any] = {"name": name, "path": str(path), "directory": str(path.parent)}
    try:
        scenario, config_hash = load_scenario_config(path)
    except Exception as exc:  # noqa: BLE001 - templates and broken files are shown, not hidden
        return info | {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    electrification = scenario.electrification
    adoption = {
        tech: {"mode": cfg.adoption_mode, "building_share": cfg.building_share}
        for tech, cfg in (("heat", electrification.heat), ("mobility", electrification.mobility),
                          ("pv_battery", electrification.pv_battery))
    }
    return info | {
        "valid": True,
        "template": name.startswith("00_") or "CHANGE_ME" in scenario.scenario_id,
        "id": scenario.scenario_id,
        "milestone_year": scenario.milestone_year,
        "heat_source": scenario.heat.space_heat_source,
        "adoption": adoption,
        "pv_maximum_fallback_share": scenario.pv.maximum_fallback_share,
        "time_aggregation": scenario.time_aggregation.enabled,
        "configuration_hash": config_hash,
        "scenario_key": scenario_identity_key(scenario.scenario_id, config_hash),
    }


def list_scenarios(dirs: Iterable[Path]) -> list[dict[str, Any]]:
    """Summaries of all scenario files (valid ones first)."""
    items = [describe(name, path) for name, path in scenario_files(dirs).items()]
    return sorted(items, key=lambda item: (not item["valid"], item["name"]))


def resolve(dirs: Iterable[Path], name: str) -> Path:
    """Path of a listed scenario file.

    Raises:
        KeyError: If ``name`` is not one of the listed file names (no paths accepted).
    """
    files = scenario_files(dirs)
    if name not in files:
        raise KeyError(name)
    return files[name]
