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
    grid_data_path: Path | None
    heat_profile_library: Path | None
    run_directory: Path | None
    model_case: str | None
    workers: int
    step3_cpus: int
    step3_cluster_concurrency: int
    step4_cpus: int
    seed: int
    cleanup_intermediates: bool
    resume: bool

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
            {
                "inputfile_id", "storage", "output_directory", "target_network",
                "target_grid_id", "paired_directory", "weather_source_hdf",
                "grid_data_path", "heat_profile_library", "run_directory",
            },
            "resources",
        )
        _only(
            execution,
            {
                "n_cpu", "mobility_source", "demand_scope", "timeframe_mode",
                "model_case", "workers", "step3_cpus",
                "step3_cluster_concurrency", "step4_cpus", "seed",
                "cleanup_intermediates", "resume",
            },
            "execution",
        )
        pipeline = str(run["pipeline"])
        if pipeline not in {"scenario", "paired_validation"}:
            raise ValueError("run.pipeline must be scenario or paired_validation.")
        storage = str(resources["storage"])
        if storage not in {"db", "h5"}:
            raise ValueError("resources.storage must be db or h5.")
        model_case = execution.get("model_case")
        if model_case is not None and str(model_case) not in {
            "pre", "post-inflex-heuristic", "post-hems-optimized",
            "post-hems-heuristic",
        }:
            raise ValueError(f"Unknown execution.model_case {model_case!r}.")
        for name in ("cleanup_intermediates", "resume"):
            if name in execution and not isinstance(execution[name], bool):
                raise ValueError(f"execution.{name} must be true or false.")

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
            grid_data_path=path_or_none(resources.get("grid_data_path")),
            heat_profile_library=path_or_none(resources.get("heat_profile_library")),
            run_directory=path_or_none(resources.get("run_directory")),
            model_case=(str(model_case) if model_case is not None else None),
            workers=int(_positive(execution.get("workers", 1), "execution.workers")),
            step3_cpus=int(_positive(execution.get("step3_cpus", 1), "execution.step3_cpus")),
            step3_cluster_concurrency=int(_positive(execution.get("step3_cluster_concurrency", 1), "execution.step3_cluster_concurrency")),
            step4_cpus=int(_positive(execution.get("step4_cpus", 1), "execution.step4_cpus")),
            seed=int(_positive(execution.get("seed", 91301), "execution.seed", allow_zero=True)),
            cleanup_intermediates=bool(execution.get("cleanup_intermediates", False)),
            resume=bool(execution.get("resume", False)),
        )
