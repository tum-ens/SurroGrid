"""Typed execution/resource configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scenario_config import _mapping, _only, _positive


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    scenario_path: Path
    pipeline: str
    inputfile_id: str
    storage: str
    output_directory: Path | None
    n_cpu: int
    mobility_source: str
    demand_scope: str
    timeframe_mode: str
    target_network: str | None
    target_grid_id: int | None
    paired_directory: Path | None
    weather_source_hdf: Path | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base_dir: Path) -> "RunConfig":
        raw = _mapping(raw, "run configuration")
        _only(raw, {"run", "resources", "execution"}, "top-level run")
        run = _mapping(raw["run"], "run")
        resources = _mapping(raw["resources"], "resources")
        execution = _mapping(raw["execution"], "execution")
        _only(run, {"id", "scenario", "pipeline"}, "run")
        _only(
            resources,
            {"inputfile_id", "storage", "output_directory", "target_network", "target_grid_id", "paired_directory", "weather_source_hdf"},
            "resources",
        )
        _only(execution, {"n_cpu", "mobility_source", "demand_scope", "timeframe_mode"}, "execution")
        pipeline = str(run["pipeline"])
        if pipeline not in {"scenario", "paired_validation"}:
            raise ValueError("run.pipeline must be scenario or paired_validation.")
        storage = str(resources["storage"])
        if storage not in {"db", "h5"}:
            raise ValueError("resources.storage must be db or h5.")

        def path_or_none(value: Any) -> Path | None:
            if value in (None, ""):
                return None
            path = Path(str(value))
            return path if path.is_absolute() else (base_dir / path).resolve()

        scenario_path = path_or_none(run["scenario"])
        if scenario_path is None:
            raise ValueError("run.scenario is required.")
        return cls(
            run_id=str(run["id"]),
            scenario_path=scenario_path,
            pipeline=pipeline,
            inputfile_id=str(resources["inputfile_id"]),
            storage=storage,
            output_directory=path_or_none(resources.get("output_directory")),
            n_cpu=int(_positive(execution["n_cpu"], "execution.n_cpu")),
            mobility_source=str(execution["mobility_source"]),
            demand_scope=str(execution["demand_scope"]),
            timeframe_mode=str(execution["timeframe_mode"]),
            target_network=resources.get("target_network"),
            target_grid_id=(int(resources["target_grid_id"]) if resources.get("target_grid_id") is not None else None),
            paired_directory=path_or_none(resources.get("paired_directory")),
            weather_source_hdf=path_or_none(resources.get("weather_source_hdf")),
        )
